import logging
import math
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union # Union untuk device

import bitsandbytes as bnb
from dac import DAC # Pastikan impor DAC sesuai dengan library yang Anda gunakan
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import get_scheduler

from sabda.config_schema import SabdaConfig, DataConfig
from sabda.layers import SabdaModel
from sabda.dataloader import create_sabda_collate_fn
from sabda.synthesizer import SabdaSynthesizer

from .training_utils import SabdaRunConfig, get_device 

# Logger khusus untuk kelas Trainer/Engine ini
logger_engine = logging.getLogger(__name__)

class Trainer:
    """
    The main training engine for SabdaTTS.
    Encapsulates the training loop, evaluation, checkpointing, and logging.
    """
    def __init__(
        self,
        model: SabdaModel,
        dac_model: DAC, 
        synthesizer: SabdaSynthesizer,
        sabda_config: SabdaConfig,
        run_config: SabdaRunConfig, # run_config.run_name diasumsikan sudah unik,
                                    # dan run_config.output_dir / run_config.run_name adalah path yang sudah ada.
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        collate_fn: Optional[callable] = None,
        device: Optional[Union[str, torch.device]] = None
        # Parameter 'run_name' (override) dihapus karena penentuan nama unik dilakukan oleh pemanggil.
    ):
        self.model = model
        self.sabda_config = sabda_config
        self.run_config = run_config 
        self.dac_model = dac_model
        self.synthesizer = synthesizer

        # Penentuan Device
        if isinstance(device, str):
            self.device = torch.device(device)
        elif isinstance(device, torch.device):
            self.device = device
        else:
            self.device = get_device(self.run_config.device_target)
        
        self.final_run_path = self.run_config.output_dir / self.run_config.run_name

        self.model.to(self.device)
        if self.dac_model:
            self.dac_model.to(self.device)
            self.dac_model.eval()

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        self.collate_fn = collate_fn if collate_fn else create_sabda_collate_fn(
            self.sabda_config.data 
        )
        
        self._train_loader_instance = self._get_train_dataloader()
        self._eval_loader_instance = self._get_eval_dataloader()

        if self._train_loader_instance and hasattr(self.train_dataset, '__len__'):
            logger_engine.info(f"Training DataLoader diinisialisasi dengan {len(self.train_dataset)} sampel.")
        # ... (log untuk eval_dataset seperti sebelumnya) ...
        if self._eval_loader_instance and hasattr(self.eval_dataset, '__len__') and self.eval_dataset is not None:
            logger_engine.info(f"Evaluation DataLoader diinisialisasi dengan {len(self.eval_dataset)} sampel.")
        elif self.eval_dataset:
             logger_engine.info("Evaluation dataset disediakan, DataLoader diinisialisasi (jumlah sampel tidak dapat ditentukan dari objek dataset).")
        else:
            logger_engine.info("Tidak ada evaluation dataset yang disediakan atau DataLoader tidak diinisialisasi.")

        self.optimizer = optimizer if optimizer else self._create_optimizer()
        self.scheduler = scheduler if scheduler else self._create_scheduler(self._train_loader_instance)

        self.grad_scaler = None
        self.amp_enabled = self.run_config.use_amp and self.device.type == 'cuda'
        if self.amp_enabled:
            self.grad_scaler = GradScaler(device='cuda')
            logger_engine.info("Automatic Mixed Precision (AMP) dengan GradScaler diaktifkan untuk CUDA.")
        elif self.run_config.use_amp and self.device.type != 'cuda':
            logger_engine.warning("AMP (use_amp=True) diminta, tetapi device bukan CUDA. GradScaler tidak akan digunakan.")

        # Subdirektori dibuat di dalam final_run_path yang sudah ada
        log_dir_path_tb = self.final_run_path / "tensorboard_logs"
        log_dir_path_tb.mkdir(parents=True, exist_ok=True) 
        self.writer = SummaryWriter(log_dir=str(log_dir_path_tb))
        logger_engine.info(f"Log TensorBoard akan disimpan di: {log_dir_path_tb}")

        self.checkpoint_dir = self.final_run_path / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger_engine.info(f"Checkpoints akan disimpan di: {self.checkpoint_dir}")
        
        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_loss = float('inf')

    def _create_optimizer(self) -> torch.optim.Optimizer: #
        logger_engine.debug("Membuat optimizer...")
        opt = bnb.optim.AdamW8bit(
            self.model.parameters(), 
            lr=self.run_config.learning_rate,
            betas=(0.9, 0.95) 
        )
        logger_engine.info(f"Optimizer: {type(opt).__name__} dibuat dengan LR: {self.run_config.learning_rate:.2e}.")
        return opt

    def _create_scheduler(self, train_loader: DataLoader) -> torch.optim.lr_scheduler._LRScheduler: #
        logger_engine.debug("Membuat learning rate scheduler...")
        if hasattr(train_loader, '__len__') and len(train_loader) > 0 :
            num_training_steps_per_epoch = len(train_loader)
        else:
            num_samples = len(self.train_dataset) if hasattr(self.train_dataset, '__len__') else (self.run_config.max_samples_dataset or 50000)
            num_training_steps_per_epoch = math.ceil(num_samples / self.run_config.batch_size)
            if num_training_steps_per_epoch == 0: num_training_steps_per_epoch = 1 
            logger_engine.warning(f"Panjang train_loader tidak tersedia atau nol. Mengaproksimasi steps_per_epoch ke {num_training_steps_per_epoch} untuk scheduler.")

        total_training_steps = num_training_steps_per_epoch * self.run_config.epochs
        if total_training_steps == 0: total_training_steps = 1 

        grad_accum = max(1, self.run_config.grad_accum_steps)
        effective_warmup_steps = self.run_config.warmup_steps // grad_accum
        effective_total_steps = max(1, total_training_steps // grad_accum)

        sched = get_scheduler(
            name="cosine", 
            optimizer=self.optimizer,
            num_warmup_steps=effective_warmup_steps,
            num_training_steps=effective_total_steps
        )
        logger_engine.info(f"Scheduler: {type(sched).__name__} dibuat. Total step efektif: {effective_total_steps}, Step warmup efektif: {effective_warmup_steps}.")
        return sched

    def _get_train_dataloader(self) -> DataLoader: #
        logger_engine.debug("Membuat/mendapatkan DataLoader training.")
        if hasattr(self, '_train_loader_instance') and self._train_loader_instance:
             return self._train_loader_instance
        self._train_loader_instance = DataLoader(
            self.train_dataset, batch_size=self.run_config.batch_size, shuffle=True,
            collate_fn=self.collate_fn, num_workers=self.run_config.num_workers_dataloader,
            pin_memory=self.run_config.pin_memory_dataloader, drop_last=True
        )
        return self._train_loader_instance

    def _get_eval_dataloader(self) -> Optional[DataLoader]: #
        if hasattr(self, '_eval_loader_instance') and self._eval_loader_instance is not None :
             return self._eval_loader_instance
        if not self.eval_dataset: 
            self._eval_loader_instance = None 
            return None
        logger_engine.debug("Membuat/mendapatkan DataLoader evaluasi.")
        self._eval_loader_instance = DataLoader(
            self.eval_dataset, batch_size=self.run_config.batch_size, shuffle=False,
            collate_fn=self.collate_fn, num_workers=self.run_config.num_workers_dataloader,
            pin_memory=self.run_config.pin_memory_dataloader
        )
        return self._eval_loader_instance
    
    def _calculate_loss(self, logits: torch.Tensor, batch_targets: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor: #
        max_len_in_batch_for_loss = target_lengths.max().item()
        if max_len_in_batch_for_loss <= 1: 
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        sliced_logits = logits[:, :max_len_in_batch_for_loss - 1, :, :]
        sliced_targets = batch_targets[:, 1:max_len_in_batch_for_loss, :]
        
        B, T_pred, C_channels, V_tgt = sliced_logits.shape
        if T_pred == 0: 
             return torch.tensor(0.0, device=logits.device, requires_grad=True)

        time_indices = torch.arange(T_pred, device=target_lengths.device).unsqueeze(0)
        valid_time_mask_per_sample = time_indices < (target_lengths.unsqueeze(1) - 1)
        final_mask_btc = valid_time_mask_per_sample.unsqueeze(-1).expand(-1, -1, C_channels)
        
        channel_weights_list = [4.0] + [1.0] * (C_channels - 1)
        if C_channels == 0: channel_weights_list = []
        elif len(channel_weights_list) != C_channels : 
            channel_weights_list = [1.0] * C_channels 
        
        accumulated_weighted_channel_loss = torch.tensor(0.0, device=logits.device)
        for c_idx in range(C_channels):
            logits_channel_c = sliced_logits[:, :, c_idx, :].reshape(-1, V_tgt)
            targets_channel_c = sliced_targets[:, :, c_idx].reshape(-1)
            mask_channel_c = final_mask_btc[:, :, c_idx].reshape(-1)
            valid_logits_for_channel = logits_channel_c[mask_channel_c]
            valid_targets_for_channel = targets_channel_c[mask_channel_c]
            if valid_targets_for_channel.numel() == 0: continue
            channel_loss = F.cross_entropy(
                valid_logits_for_channel, valid_targets_for_channel,
                ignore_index=self.sabda_config.data.audio_pad_value
            )
            accumulated_weighted_channel_loss += channel_weights_list[c_idx] * channel_loss
        
        sum_weights = sum(channel_weights_list)
        total_loss = accumulated_weighted_channel_loss / sum_weights if sum_weights > 0 else torch.tensor(0.0, device=logits.device)
        return total_loss

    def _prepare_batch_for_device(self, batch_cpu: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Pindahkan tensor ke device, non_blocking jika pin_memory aktif dan device CUDA
        return {k: v.to(self.device, non_blocking=self.run_config.pin_memory_dataloader and self.device.type == 'cuda') if isinstance(v, torch.Tensor) else v 
                for k, v in batch_cpu.items()}

    def _training_step(self, batch_cpu: Dict[str, torch.Tensor]) -> torch.Tensor: #
        self.model.train()
        batch = self._prepare_batch_for_device(batch_cpu)

        src_tokens_batch = batch['src_tokens']
        enc_self_attn_mask_batch = batch['enc_self_attn_mask']
        dec_cross_attn_mask_batch = batch['dec_cross_attn_mask']

        if self.run_config.unconditional_frac > 0 and random.random() < self.run_config.unconditional_frac:
            logger_engine.debug("Menggunakan training unconditional untuk batch ini.")
            src_tokens_batch = torch.zeros_like(src_tokens_batch)
            enc_self_attn_mask_batch = torch.zeros_like(enc_self_attn_mask_batch)
            dec_cross_attn_mask_batch = torch.zeros_like(dec_cross_attn_mask_batch)

        compute_dtype = torch.bfloat16 if self.sabda_config.train_args.dtype == "bfloat16" else torch.float16
        
        with autocast(
            device_type=self.device.type, 
            enabled=self.amp_enabled,
            dtype=compute_dtype if self.device.type == 'cuda' else None
        ):
            logits = self.model(
                src_tokens=src_tokens_batch, tgt_tokens=batch['tgt_tokens'],
                src_pos=batch['src_positions'], tgt_pos=batch['tgt_positions'],
                enc_self_attn_mask=enc_self_attn_mask_batch, dec_self_attn_mask=batch['dec_self_attn_mask'],
                dec_cross_attn_mask=dec_cross_attn_mask_batch, deterministic=False
            )
            loss = self._calculate_loss(logits, batch['tgt_tokens'], batch['tgt_lens'])
        return loss

    def train(self): #
        if self.run_config.resume_checkpoint_path:
            resume_path = Path(self.run_config.resume_checkpoint_path) if isinstance(self.run_config.resume_checkpoint_path, str) else self.run_config.resume_checkpoint_path
            if resume_path and resume_path.is_file():
                 completed_epoch, self.global_step, self.best_eval_loss = self._load_checkpoint(resume_path)
                 self.current_epoch = completed_epoch 
            else:
                logger_engine.warning(f"Resume checkpoint path '{self.run_config.resume_checkpoint_path}' tidak ditemukan atau bukan file. Memulai dari awal.")
        
        logger_engine.info(f"\n{'='*20} Memulai Proses Training {'='*20}\n"
                           f"  Target Epochs: {self.run_config.epochs}\n"
                           f"  Mulai dari Epoch: {self.current_epoch + 1}\n"
                           f"  Global Step Awal: {self.global_step}\n"
                           f"{'='*60}")

        train_loader = self._train_loader_instance
        if not train_loader or (hasattr(train_loader, '__len__') and len(train_loader) == 0):
            logger_engine.error("Training DataLoader kosong atau tidak tersedia. Tidak dapat memulai training.")
            return

        for epoch in range(self.current_epoch, self.run_config.epochs):
            self.current_epoch = epoch 
            self.model.train()
            if hasattr(train_loader.sampler, 'set_epoch') and isinstance(train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
                 train_loader.sampler.set_epoch(epoch)
            
            epoch_loss_sum = 0.0; num_batches_epoch = 0
            len_train_loader = len(train_loader) if hasattr(train_loader, '__len__') else 0
            initial_step_in_epoch = (self.global_step % len_train_loader) if self.global_step > 0 and len_train_loader > 0 else 0
            
            current_iterator = iter(train_loader)
            if epoch == self.current_epoch and initial_step_in_epoch > 0 and len_train_loader > 0:
                logger_engine.info(f"Melanjutkan epoch {epoch+1}, melewati {initial_step_in_epoch} batch pertama.")
                for _ in range(initial_step_in_epoch):
                    try: next(current_iterator)
                    except StopIteration: 
                        logger_engine.warning(f"Mencoba melewati {initial_step_in_epoch} batch, tetapi DataLoader habis lebih awal.")
                        break 
            
            progress_bar = tqdm(
                current_iterator, total=len_train_loader if len_train_loader > 0 else None, 
                desc=f"Epoch {epoch+1}/{self.run_config.epochs}", initial=initial_step_in_epoch,
                position=0, leave=True
            )

            for batch_idx_in_tqdm, batch_data in enumerate(progress_bar, start=initial_step_in_epoch):
                loss = self._training_step(batch_data)
                if torch.isnan(loss) or torch.isinf(loss):
                    logger_engine.error(f"Loss NaN atau Inf terdeteksi pada step {self.global_step}. Menghentikan training.")
                    self.writer.close(); return 

                epoch_loss_sum += loss.item(); num_batches_epoch +=1
                scaled_loss = loss / max(1, self.run_config.grad_accum_steps)

                if self.grad_scaler: self.grad_scaler.scale(scaled_loss).backward()
                else: scaled_loss.backward()

                if (self.global_step + 1) % max(1, self.run_config.grad_accum_steps) == 0 or \
                   (batch_idx_in_tqdm + 1) == len_train_loader: 
                    if self.grad_scaler:
                        self.grad_scaler.step(self.optimizer)
                        self.grad_scaler.update()
                    else: self.optimizer.step()
                    self.scheduler.step(); self.optimizer.zero_grad(set_to_none=True)
                
                self.global_step += 1
                if self.global_step % self.run_config.log_steps == 0:
                    current_lr = self.scheduler.get_last_lr()[0]
                    self.writer.add_scalar('Loss/train_step', loss.item(), self.global_step)
                    self.writer.add_scalar('LearningRate', current_lr, self.global_step)
                    progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.2e}", "step": f"{self.global_step}"})
                
                if self.run_config.eval_steps > 0 and self.global_step % self.run_config.eval_steps == 0:
                    self.evaluate()
                
                if self.run_config.save_steps > 0 and self.global_step % self.run_config.save_steps == 0:
                    self._save_checkpoint(current_loss=loss.item())
            
            avg_epoch_loss = epoch_loss_sum / num_batches_epoch if num_batches_epoch > 0 else float('nan')
            self.writer.add_scalar('Loss/train_epoch', avg_epoch_loss, epoch + 1)
            logger_engine.info(f"\n--- Epoch {epoch+1}/{self.run_config.epochs} Summary ---"
                               f"\n    Avg Train Loss: {avg_epoch_loss:.4f}"
                               f"\n    Global step: {self.global_step}"
                               f"\n--------------------------------------")
            self.evaluate()
            self._save_checkpoint(current_loss=avg_epoch_loss, filename_prefix="last_model")
        self.writer.close(); logger_engine.info(f"\n{'='*20} Proses Training Selesai {'='*20}")

    def evaluate(self): #
        eval_loader = self._eval_loader_instance
        if not eval_loader: 
            logger_engine.info("\nTidak ada evaluation dataset/loader. Melewati evaluasi.")
            self.model.train(); return

        num_eval_samples = 0
        if hasattr(self.eval_dataset, '__len__') and self.eval_dataset is not None: num_eval_samples = len(self.eval_dataset)
        elif eval_loader and hasattr(eval_loader.dataset, '__len__'): num_eval_samples = len(eval_loader.dataset)
        
        log_eval_header = f"\n--- Menjalankan Evaluasi pada Step {self.global_step} ---"
        if num_eval_samples > 0: log_eval_header += f"\n    Jumlah sampel validasi: {num_eval_samples}"
        else: log_eval_header += "\n    (Jumlah sampel validasi tidak dapat ditentukan)"
        logger_engine.info(log_eval_header)
            
        total_val_loss = 0.0; num_val_batches = 0

        # Ensure model is in eval mode for the entire evaluation
        original_model_mode_is_training = self.model.training
        self.model.eval()

        with torch.inference_mode(): # Disables gradient calculations
            len_eval_loader = len(eval_loader) if hasattr(eval_loader, '__len__') else None
            progress_bar_val = tqdm(enumerate(eval_loader), total=len_eval_loader, desc="Evaluating", position=0, leave=True)
            for batch_idx, batch_cpu in progress_bar_val:
                batch = self._prepare_batch_for_device(batch_cpu)
                # Calculate loss (existing logic)
                logits = self.model(
                    src_tokens=batch['src_tokens'], tgt_tokens=batch['tgt_tokens'],
                    src_pos=batch['src_positions'], tgt_pos=batch['tgt_positions'],
                    enc_self_attn_mask=batch['enc_self_attn_mask'], dec_self_attn_mask=batch['dec_self_attn_mask'],
                    dec_cross_attn_mask=batch['dec_cross_attn_mask'], deterministic=True
                )
                loss = self._calculate_loss(logits, batch['tgt_tokens'], batch['tgt_lens'])
                if torch.isnan(loss) or torch.isinf(loss):
                    logger_engine.warning(f"Loss validasi NaN atau Inf terdeteksi. Melewati batch ini.")
                    continue
                total_val_loss += loss.item()
                num_val_batches += 1
                progress_bar_val.set_postfix({"val_loss_batch": f"{loss.item():.4f}"})
        
        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else float('inf')
        if self.writer: # Check if SummaryWriter is initialized
            self.writer.add_scalar('Loss/validation', avg_val_loss, self.global_step)
        
        log_eval_summary = f"\n--- Evaluasi pada Step {self.global_step} ---"
        log_eval_summary += f"\n    Validation Loss: {avg_val_loss:.4f}"

        # --- Audio Generation during Evaluation ---
        if self.run_config.eval_prompts and self.synthesizer:
            logger_engine.info("--- Generating Audio Samples for Evaluation ---")
            # self.model.eval() is already active from above
            # self.synthesizer.model and self.synthesizer.dac_model are set to eval in SabdaSynthesizer.__init__

            for i, prompt_text in enumerate(self.run_config.eval_prompts):
                try:
                    logger_engine.info(f"  Generating audio for prompt ({i+1}/{len(self.run_config.eval_prompts)}): \"{prompt_text[:60]}...\"")
                    
                    waveform_tensor = self.synthesizer.generate(
                        text=prompt_text,
                        max_new_tokens=self.run_config.eval_gen_max_new_tokens,
                        temperature=self.run_config.eval_gen_temperature,
                        top_p=self.run_config.eval_gen_top_p if (self.run_config.eval_gen_top_p is not None and self.run_config.eval_gen_top_p < 1.0) else None, # Pass None if not used
                        cfg_scale=self.run_config.eval_gen_cfg_scale
                    ) 

                    if waveform_tensor is not None and waveform_tensor.numel() > 0:
                        waveform_to_log = waveform_tensor.squeeze().cpu() 
                        
                        sample_rate = self.synthesizer.dac_model.sample_rate # descriptive-audio-codec has this attribute
                                            
                        if self.writer: # Check if SummaryWriter is initialized
                            self.writer.add_audio(
                                f"Audio_Eval/Prompt_{i+1}", 
                                waveform_to_log,
                                self.global_step,
                                sample_rate=sample_rate 
                            )
                        logger_engine.info(f"    Logged audio for prompt {i+1} to TensorBoard (sr: {sample_rate}).")
                    else:
                        logger_engine.warning(f"    Audio generation returned None or empty for prompt {i+1}.")

                except Exception as e:
                    logger_engine.error(f"    Error generating/logging audio for prompt '{prompt_text}': {e}", exc_info=True)
            logger_engine.info("--- Finished Generating Audio Samples ---")
        # --- End of Audio Generation ---

        if avg_val_loss < self.best_eval_loss:
            self.best_eval_loss = avg_val_loss
            log_eval_summary += f"\n    New best validation loss: {self.best_eval_loss:.4f}. Menyimpan model terbaik."
            self._save_checkpoint(current_loss=avg_val_loss, is_best=True)
        
        logger_engine.info(log_eval_summary + "\n--- Selesai Evaluasi ---")
        
        if original_model_mode_is_training: # Restore model to training mode if it was training before eval
            self.model.train()

    def _save_checkpoint(self, current_loss: float, is_best: bool = False, filename_prefix: str = "ckpt"): #
        # checkpoint_dir sekarang adalah self.checkpoint_dir
        checkpoint_filename = f"{filename_prefix}.pth" if filename_prefix=="last_model" else f"{filename_prefix}_step_{self.global_step}.pth"
        checkpoint_path = (self.checkpoint_dir / "best_model.pth") if is_best else (self.checkpoint_dir / checkpoint_filename)
        save_state = {
            'epoch': self.current_epoch, 'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': current_loss, 'best_eval_loss': self.best_eval_loss
        }
        if self.scheduler: save_state['scheduler_state_dict'] = self.scheduler.state_dict()
        try:
            torch.save(save_state, checkpoint_path)
            logger_engine.info(f"\nCheckpoint disimpan ke: {checkpoint_path}")

        except Exception as e: logger_engine.error(f"\nGagal menyimpan checkpoint ke {checkpoint_path}: {e}", exc_info=True)

    def _load_checkpoint(self, checkpoint_path: Path) -> Tuple[int, int, float]: #
        if not checkpoint_path.is_file():
            logger_engine.warning(f"File checkpoint tidak ditemukan: {checkpoint_path}. Memulai training dari awal.")
            return 0, 0, float('inf')
        try:
            logger_engine.info(f"Memuat checkpoint dari: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device) 
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if self.optimizer and 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
                try: self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except Exception as e_sched: 
                     logger_engine.warning(f"Tidak dapat memuat state_dict scheduler: {e_sched}. Scheduler mungkin dimulai ulang.")
            
            completed_epoch = checkpoint.get('epoch', 0) 
            global_step_start = checkpoint.get('global_step', 0)
            best_eval_loss_loaded = checkpoint.get('best_eval_loss', float('inf'))
            last_loss = checkpoint.get('loss', float('inf'))
            logger_engine.info(
                f"\nCheckpoint berhasil dimuat.\n"
                f"  Akan melanjutkan dari Epoch: {completed_epoch + 1}\n"
                f"  Global step: {global_step_start}\n"
                f"  Loss tersimpan terakhir: {last_loss:.4f}\n"
                f"  Best eval loss tercatat: {best_eval_loss_loaded:.4f}"
            )
            return completed_epoch, global_step_start, best_eval_loss_loaded
        except Exception as e:
            logger_engine.error(f"Error saat memuat checkpoint dari {checkpoint_path}: {e}. Memulai dari awal.", exc_info=True)
            return 0, 0, float('inf')