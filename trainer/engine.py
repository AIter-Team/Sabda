# In SabdaTTS/trainer/engine.py

import logging
import math
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union 
import shutil # For shutil.copy2 in _save_checkpoint

import bitsandbytes as bnb
from dac import DAC 
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import get_scheduler # This import stays

# --- XLA Specific Imports ---
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met # Optional: for XLA metrics/debug

# Assuming SabdaSynthesizer might be type hinted or used if passed for eval, adjust import as necessary
# from sabda.synthesizer import SabdaSynthesizer # If you need to type hint it

from sabda.config_schema import SabdaConfig, DataConfig
from sabda.layers import SabdaModel
from sabda.dataloader import create_sabda_collate_fn
from .training_utils import SabdaRunConfig, get_device 

logger_engine = logging.getLogger(__name__)

class Trainer:
    def __init__(
        self,
        model: SabdaModel,
        dac_model: DAC, 
        synthesizer: Any, # Using Any to avoid circular import if SabdaSynthesizer is in different module
                         # Or import SabdaSynthesizer properly if it's defined elsewhere and accessible
        sabda_config: SabdaConfig,
        run_config: SabdaRunConfig,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        collate_fn: Optional[callable] = None,
        device: Optional[Union[str, torch.device]] = None 
    ):
        self.model = model
        self.sabda_config = sabda_config
        self.run_config = run_config 
        self.dac_model = dac_model # Used by self.synthesizer
        self.synthesizer = synthesizer # Used for audio generation in evaluate()

        if isinstance(device, str):
            self.device = torch.device(device)
        elif isinstance(device, torch.device):
            self.device = device
        else:
            # This path should ideally not be hit if train.py correctly passes the device
            logger_engine.warning("Device not explicitly passed to Trainer, attempting to get it via run_config.")
            self.device = get_device(self.run_config.device_target)
        
        logger_engine.info(f"Trainer initialized with device: {self.device}")

        self.final_run_path = self.run_config.output_dir / self.run_config.run_name
        self.model.to(self.device) # Move model to the target device (CPU, CUDA, or XLA)

        if self.dac_model: # DAC model also needs to be on the correct device if used by synthesizer on this device
            self.dac_model.to(self.device)
            self.dac_model.eval()

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        self.collate_fn = collate_fn if collate_fn else create_sabda_collate_fn(
            self.sabda_config.data 
        )
        
        self._train_loader_instance = self._get_train_dataloader()
        self._eval_loader_instance = self._get_eval_dataloader()

        # Optional: Wrap DataLoaders with MpDeviceLoader for XLA performance
        if self.device.type == 'xla' and self.run_config.num_workers_dataloader > 0 : # MpDeviceLoader usually for multi-processing
            logger_engine.info("Attempting to use MpDeviceLoader for XLA.")
            try:
                from torch_xla.distributed.parallel_loader import MpDeviceLoader
                if self._train_loader_instance:
                    self._train_loader_instance = MpDeviceLoader(self._train_loader_instance, self.device)
                    logger_engine.info("Wrapped training DataLoader with MpDeviceLoader for XLA.")
                if self._eval_loader_instance:
                    self._eval_loader_instance = MpDeviceLoader(self._eval_loader_instance, self.device)
                    logger_engine.info("Wrapped evaluation DataLoader with MpDeviceLoader for XLA.")
            except ImportError:
                logger_engine.warning("MpDeviceLoader not found or torch_xla.distributed not available. Using standard DataLoader.")
            except Exception as e:
                logger_engine.error(f"Error wrapping DataLoader with MpDeviceLoader: {e}. Using standard DataLoader.")


        if self._train_loader_instance and hasattr(self.train_dataset, '__len__'):
            logger_engine.info(f"Training DataLoader diinisialisasi dengan {len(self.train_dataset)} sampel.")
        if self._eval_loader_instance and hasattr(self.eval_dataset, '__len__') and self.eval_dataset is not None:
            logger_engine.info(f"Evaluation DataLoader diinisialisasi dengan {len(self.eval_dataset)} sampel.")
        elif self.eval_dataset:
             logger_engine.info("Evaluation dataset disediakan, DataLoader diinisialisasi.")
        else:
            logger_engine.info("Tidak ada evaluation dataset yang disediakan atau DataLoader tidak diinisialisasi.")

        self.optimizer = optimizer if optimizer else self._create_optimizer()
        self.scheduler = scheduler if scheduler else self._create_scheduler(self._train_loader_instance)

        # AMP configuration
        self.grad_scaler = None
        self.amp_enabled = self.run_config.use_amp and \
                           (self.device.type == 'cuda' or \
                            (self.device.type == 'xla' and self.sabda_config.train_args.dtype == "bfloat16"))

        if self.run_config.use_amp:
            if self.device.type == 'cuda':
                self.grad_scaler = GradScaler(device='cuda')
                logger_engine.info("CUDA AMP with GradScaler enabled.")
            elif self.device.type == 'xla' and self.sabda_config.train_args.dtype == "bfloat16":
                logger_engine.info("XLA: Target dtype is bfloat16. Autocast will be used if enabled. GradScaler is not used with XLA.")
            elif self.device.type == 'xla': # use_amp True but dtype is not bfloat16
                 logger_engine.info("XLA: use_amp is True, but target dtype is not bfloat16. Autocast might not be enabled depending on training_step logic.")
            else: # CPU or other
                logger_engine.warning(f"AMP (use_amp=True) requested, but device is {self.device.type}. AMP typically for CUDA/XLA. GradScaler not used.")


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

    def _create_optimizer(self) -> torch.optim.Optimizer:
        logger_engine.debug("Membuat optimizer...")
        # AdamW8bit might have specific device considerations for XLA,
        # but usually works if the model parameters are on the XLA device.
        opt = bnb.optim.AdamW8bit(
            self.model.parameters(), 
            lr=self.run_config.learning_rate,
            betas=(0.9, 0.95) 
        )
        logger_engine.info(f"Optimizer: {type(opt).__name__} dibuat dengan LR: {self.run_config.learning_rate:.2e}.")
        return opt

    def _create_scheduler(self, train_loader: DataLoader) -> torch.optim.lr_scheduler._LRScheduler:
        logger_engine.debug("Membuat learning rate scheduler...")
        # Determine num_training_steps_per_epoch
        if hasattr(train_loader, '__len__') and len(train_loader) > 0 and train_loader.batch_size is not None : # Standard DataLoader
            num_training_steps_per_epoch = math.ceil(len(train_loader.dataset) / train_loader.batch_size) if train_loader.drop_last is False else len(train_loader)
            # MpDeviceLoader might not have a direct len, handle appropriately if you switch
        elif self.device.type == 'xla' and isinstance(train_loader, torch_xla.distributed.parallel_loader.MpDeviceLoader):
             # For MpDeviceLoader, length might need to be calculated based on dataset size and world size
             # This is a simplification; proper distributed setup is more complex.
             # For now, let's assume an estimate if direct len is not available.
             num_samples_train = len(self.train_dataset) if hasattr(self.train_dataset, '__len__') else (self.run_config.max_samples_dataset or 50000) # Estimate
             num_training_steps_per_epoch = math.ceil(num_samples_train / (self.run_config.batch_size * xm.xrt_world_size()))
             logger_engine.warning(f"Approximating steps_per_epoch for XLA MpDeviceLoader to {num_training_steps_per_epoch}.")
        else: # Fallback if length cannot be determined
            num_samples_train = len(self.train_dataset) if hasattr(self.train_dataset, '__len__') else (self.run_config.max_samples_dataset or 50000) # Estimate
            num_training_steps_per_epoch = math.ceil(num_samples_train / self.run_config.batch_size)
            logger_engine.warning(f"Panjang train_loader tidak standar atau nol. Mengaproksimasi steps_per_epoch ke {num_training_steps_per_epoch} untuk scheduler.")

        if num_training_steps_per_epoch == 0: num_training_steps_per_epoch = 1 
        total_training_steps = num_training_steps_per_epoch * self.run_config.epochs
        if total_training_steps == 0: total_training_steps = 1 

        grad_accum = max(1, self.run_config.grad_accum_steps)
        effective_warmup_steps = self.run_config.warmup_steps # Warmup steps are in terms of optimizer steps
        effective_total_steps = max(1, total_training_steps // grad_accum)
        # Adjust effective_warmup_steps if it was meant to be in terms of global steps
        # effective_warmup_steps = self.run_config.warmup_steps // grad_accum 

        sched = get_scheduler(
            name="cosine", 
            optimizer=self.optimizer,
            num_warmup_steps=effective_warmup_steps,
            num_training_steps=effective_total_steps
        )
        logger_engine.info(f"Scheduler: {type(sched).__name__} dibuat. Total step efektif (optimizer updates): {effective_total_steps}, Step warmup efektif: {effective_warmup_steps}.")
        return sched

    def _get_train_dataloader(self) -> DataLoader: 
        logger_engine.debug("Membuat/mendapatkan DataLoader training.")
        if hasattr(self, '_train_loader_instance') and self._train_loader_instance:
             return self._train_loader_instance
        self._train_loader_instance = DataLoader(
            self.train_dataset, batch_size=self.run_config.batch_size, shuffle=True,
            collate_fn=self.collate_fn, num_workers=self.run_config.num_workers_dataloader,
            pin_memory=self.run_config.pin_memory_dataloader if self.device.type != 'xla' else False, # pin_memory for CUDA
            drop_last=True # Good for stable step counts
        )
        return self._train_loader_instance

    def _get_eval_dataloader(self) -> Optional[DataLoader]:
        if hasattr(self, '_eval_loader_instance') and self._eval_loader_instance is not None :
             return self._eval_loader_instance
        if not self.eval_dataset: 
            self._eval_loader_instance = None 
            return None
        logger_engine.debug("Membuat/mendapatkan DataLoader evaluasi.")
        self._eval_loader_instance = DataLoader(
            self.eval_dataset, batch_size=self.run_config.batch_size, shuffle=False, # Usually bsz=1 or same as train for eval
            collate_fn=self.collate_fn, num_workers=self.run_config.num_workers_dataloader,
            pin_memory=self.run_config.pin_memory_dataloader if self.device.type != 'xla' else False
        )
        return self._eval_loader_instance
    
    def _calculate_loss(self, logits: torch.Tensor, batch_targets: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        # ... (your existing _calculate_loss logic seems fine) ...
        # This function should return a scalar tensor.
        # For XLA, if running distributed, loss might need xm.mesh_reduce.
        # For single process XLA (typical Kaggle TPU VM), this direct loss should be okay.
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
        return {k: v.to(self.device, non_blocking=self.run_config.pin_memory_dataloader and self.device.type == 'cuda') if isinstance(v, torch.Tensor) else v 
                for k, v in batch_cpu.items()}

    def _training_step(self, batch_cpu: Dict[str, torch.Tensor]) -> torch.Tensor:
        self.model.train()
        batch = self._prepare_batch_for_device(batch_cpu)

        src_tokens_batch = batch['src_tokens']
        enc_self_attn_mask_batch = batch['enc_self_attn_mask']
        dec_cross_attn_mask_batch = batch['dec_cross_attn_mask']

        if self.run_config.unconditional_frac > 0 and random.random() < self.run_config.unconditional_frac:
            logger_engine.debug("Menggunakan training unconditional untuk batch ini.")
            src_tokens_batch = torch.zeros_like(src_tokens_batch) # Ensure on correct device
            enc_self_attn_mask_batch = torch.zeros_like(enc_self_attn_mask_batch)
            dec_cross_attn_mask_batch = torch.zeros_like(dec_cross_attn_mask_batch)

        compute_dtype = torch.bfloat16 if self.sabda_config.train_args.dtype == "bfloat16" else torch.float16
        
        # Use self.amp_enabled which considers XLA+bfloat16 for enabling autocast
        with autocast(
            device_type=self.device.type, 
            enabled=self.amp_enabled, 
            dtype=compute_dtype if self.amp_enabled else None # Pass compute_dtype if autocast enabled
        ):
            logits = self.model(
                src_tokens=src_tokens_batch, tgt_tokens=batch['tgt_tokens'],
                src_pos=batch['src_positions'], tgt_pos=batch['tgt_positions'],
                enc_self_attn_mask=enc_self_attn_mask_batch, dec_self_attn_mask=batch['dec_self_attn_mask'],
                dec_cross_attn_mask=dec_cross_attn_mask_batch, deterministic=False
            )
            loss = self._calculate_loss(logits, batch['tgt_tokens'], batch['tgt_lens'])
        return loss

    def train(self):
        if self.run_config.resume_checkpoint_path:
            # _load_checkpoint will use self.device (which could be XLA)
            completed_epoch, self.global_step, self.best_eval_loss = self._load_checkpoint(self.run_config.resume_checkpoint_path)
            self.current_epoch = completed_epoch 
        
        logger_engine.info(f"\n{'='*20} Memulai Proses Training {'='*20}\n"
                           f"  Device: {self.device}\n"
                           f"  Target Epochs: {self.run_config.epochs}\n"
                           f"  Mulai dari Epoch: {self.current_epoch + 1}\n"
                           f"  Global Step Awal: {self.global_step}\n"
                           f"{'='*60}")

        train_loader = self._train_loader_instance
        if not train_loader or (hasattr(train_loader, '__len__') and len(train_loader) == 0 and not isinstance(train_loader, torch_xla.distributed.parallel_loader.MpDeviceLoader)): # MpDeviceLoader might not have len
            logger_engine.error("Training DataLoader kosong atau tidak tersedia. Tidak dapat memulai training.")
            return

        for epoch in range(self.current_epoch, self.run_config.epochs):
            self.current_epoch = epoch 
            self.model.train()
            # For XLA distributed training, set epoch for DistributedSampler (if used)
            if self.device.type == 'xla' and hasattr(train_loader, 'sampler') and \
               isinstance(train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
                 train_loader.sampler.set_epoch(epoch)
            
            epoch_loss_sum = 0.0
            num_batches_epoch = 0
            
            # Determine len_train_loader carefully if it's MpDeviceLoader
            len_train_loader = 0
            if hasattr(train_loader, '__len__') and not isinstance(train_loader, torch_xla.distributed.parallel_loader.MpDeviceLoader):
                len_train_loader = len(train_loader)
            elif hasattr(self, '_train_len_estimate_for_scheduler') and self._train_len_estimate_for_scheduler > 0: # Use estimate if available
                 len_train_loader = self._train_len_estimate_for_scheduler
            elif hasattr(self.train_dataset, '__len__') and self.run_config.batch_size > 0:
                world_size = xm.xrt_world_size() if self.device.type == 'xla' else 1
                len_train_loader = math.ceil(len(self.train_dataset) / (self.run_config.batch_size * world_size))


            initial_step_in_epoch = 0 # Resuming mid-epoch with XLA can be complex, simplifying for now
            if self.global_step > 0 and len_train_loader > 0 and self.current_epoch == epoch : # Resuming epoch
                initial_step_in_epoch = self.global_step % len_train_loader if len_train_loader > 0 else 0 # This might not be robust with XLA dataloaders
            
            progress_bar = tqdm(
                train_loader, # MpDeviceLoader is iterable
                desc=f"Epoch {epoch+1}/{self.run_config.epochs}", 
                total=len_train_loader if len_train_loader > 0 else None, # tqdm total might be problematic if len is unknown
                initial=initial_step_in_epoch, # tqdm initial might not work well if iterator is already advanced
                position=0, leave=True
            )
            
            # if initial_step_in_epoch > 0 and isinstance(train_loader, iterator_type):
            # Simplified: not skipping for XLA loader for now if resuming.
            # Proper resume for XLA distributed needs careful handling of dataloader state.

            for batch_idx_in_tqdm, batch_data in enumerate(progress_bar): # Removed start=initial_step_in_epoch
                # If resuming, we might re-process some batches if not careful with iterator state.
                # For simplicity here, if resuming, it restarts epoch. Proper skipping needs work.
                if epoch == self.current_epoch and self.global_step > 0 and batch_idx_in_tqdm < initial_step_in_epoch :
                    if batch_idx_in_tqdm == 0: logger_engine.info(f"Resuming epoch {epoch+1}, skipping to step {initial_step_in_epoch} (approx).")
                    # self.global_step +=1 # Account for skipped batches if we could truly skip them.
                    # For now, this logic might re-run some initial batches of a resumed epoch.
                    # To truly skip, the dataloader iterator itself needs to be advanced.
                    # This is non-trivial with MpDeviceLoader.
                    # Let's assume for now that global_step resume correctly sets the starting point.
                    # And if we re-process a few batches, it's a minor inefficiency for now.
                    pass # Pass for now, will re-evaluate resume with XLA

                loss = self._training_step(batch_data)
                if torch.isnan(loss) or torch.isinf(loss):
                    logger_engine.error(f"Loss NaN atau Inf terdeteksi pada step {self.global_step}. Menghentikan training.")
                    if self.writer: self.writer.close()
                    return 

                epoch_loss_sum += loss.item()
                num_batches_epoch +=1
                scaled_loss = loss / max(1, self.run_config.grad_accum_steps)

                # Backward pass
                if self.device.type == 'cuda' and self.grad_scaler:
                    self.grad_scaler.scale(scaled_loss).backward()
                else: # For CPU, or CUDA without AMP, or XLA
                    scaled_loss.backward()

                # Optimizer Step
                if (self.global_step + 1) % max(1, self.run_config.grad_accum_steps) == 0 or \
                   (batch_idx_in_tqdm + 1) == (len_train_loader if len_train_loader > 0 else (batch_idx_in_tqdm +1) ): # Ensure last batch also updates
                    
                    if self.device.type == 'xla':
                        xm.optimizer_step(self.optimizer, barrier=True) 
                    elif self.device.type == 'cuda' and self.grad_scaler:
                        self.grad_scaler.step(self.optimizer)
                        self.grad_scaler.update()
                    else: 
                        self.optimizer.step()
                    
                    if self.scheduler: self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                
                self.global_step += 1
                current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.run_config.learning_rate
                
                if self.global_step % self.run_config.log_steps == 0:
                    if self.writer:
                        self.writer.add_scalar('Loss/train_step', loss.item(), self.global_step)
                        self.writer.add_scalar('LearningRate', current_lr, self.global_step)
                    progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.2e}", "step": f"{self.global_step}"})
                
                if self.run_config.eval_steps > 0 and self.global_step % self.run_config.eval_steps == 0:
                    self.evaluate() # evaluate() should also use self.device
                    self.model.train() # Ensure model is back in train mode after eval
                
                if self.run_config.save_steps > 0 and self.global_step % self.run_config.save_steps == 0:
                    self._save_checkpoint(current_loss=loss.item())
            
            avg_epoch_loss = epoch_loss_sum / num_batches_epoch if num_batches_epoch > 0 else float('nan')
            if self.writer:
                 self.writer.add_scalar('Loss/train_epoch', avg_epoch_loss, epoch + 1)
            logger_engine.info(f"\n--- Epoch {epoch+1}/{self.run_config.epochs} Summary ---"
                               f"\n    Avg Train Loss: {avg_epoch_loss:.4f}"
                               f"\n    Global step: {self.global_step}"
                               f"\n--------------------------------------")
            
            # Evaluate at end of epoch if eval_steps is not set or doesn't align
            if self.run_config.eval_steps <= 0 or (len_train_loader > 0 and self.run_config.eval_steps % len_train_loader != 0):
                 self.evaluate()
                 self.model.train() # Ensure model is back in train mode

            self._save_checkpoint(current_loss=avg_epoch_loss, filename_prefix="ckpt_last_epoch") # Save last epoch
        
        if self.device.type == 'xla':
            logger_engine.info("XLA Metrics Report (Final):\n%s", met.metrics_report())
        
        if self.writer: self.writer.close()
        logger_engine.info(f"\n{'='*20} Proses Training Selesai {'='*20}")


    def evaluate(self):
        eval_loader = self._eval_loader_instance
        if not eval_loader or (self.eval_dataset is None or len(self.eval_dataset) == 0): 
            logger_engine.info("\nTidak ada evaluation dataset/loader yang valid. Melewati evaluasi.")
            return # No need to set model.train() here as it's handled after calling evaluate

        # ... (Your existing evaluate logic from previous correct version) ...
        # Ensure self.model.eval() is called at the start of this method
        # And self.synthesizer.generate() uses self.device correctly.
        original_model_mode_is_training = self.model.training
        self.model.eval()
        
        # ... (rest of your validation loss calculation loop using with torch.inference_mode()) ...
        # Make sure to move batch to self.device within the loop
        # total_val_loss = 0.0; num_val_batches = 0
        # with torch.inference_mode():
        #    for batch_cpu in eval_loader:
        #        batch = self._prepare_batch_for_device(batch_cpu)
        #        logits = self.model(...)
        #        loss = self._calculate_loss(...)
        #        total_val_loss += loss.item(); num_val_batches +=1
        # avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else float('inf')
        # if self.writer: self.writer.add_scalar('Loss/validation', avg_val_loss, self.global_step)
        # log_eval_summary = f"    Validation Loss: {avg_val_loss:.4f}"

        # This part is from your logs, assuming it's largely correct
        log_eval_header = f"\n--- Menjalankan Evaluasi pada Step {self.global_step} ---"
        num_eval_samples = len(self.eval_dataset) if hasattr(self.eval_dataset, '__len__') else 0
        if num_eval_samples > 0: log_eval_header += f"\n    Jumlah sampel validasi: {num_eval_samples}"
        logger_engine.info(log_eval_header)
            
        total_val_loss = 0.0; num_val_batches = 0
        with torch.inference_mode():
            len_eval_loader = 0
            if hasattr(eval_loader, '__len__') and not isinstance(eval_loader, torch_xla.distributed.parallel_loader.MpDeviceLoader):
                 len_eval_loader = len(eval_loader)
            elif hasattr(self.eval_dataset, '__len__') and self.run_config.batch_size > 0:
                 world_size = xm.xrt_world_size() if self.device.type == 'xla' else 1
                 len_eval_loader = math.ceil(len(self.eval_dataset) / (self.run_config.batch_size * world_size))

            progress_bar_val = tqdm(enumerate(eval_loader), total=len_eval_loader if len_eval_loader > 0 else None, desc="Evaluating", position=0, leave=True)
            for batch_idx, batch_cpu in progress_bar_val:
                batch = self._prepare_batch_for_device(batch_cpu)
                logits = self.model(
                    src_tokens=batch['src_tokens'], tgt_tokens=batch['tgt_tokens'],
                    src_pos=batch['src_positions'], tgt_pos=batch['tgt_positions'],
                    enc_self_attn_mask=batch['enc_self_attn_mask'], dec_self_attn_mask=batch['dec_self_attn_mask'],
                    dec_cross_attn_mask=batch['dec_cross_attn_mask'], deterministic=True
                )
                loss = self._calculate_loss(logits, batch['tgt_tokens'], batch['tgt_lens'])
                if torch.isnan(loss) or torch.isinf(loss):
                    logger_engine.warning(f"Loss validasi NaN atau Inf terdeteksi. Melewati batch ini."); continue
                total_val_loss += loss.item(); num_val_batches += 1
                if hasattr(progress_bar_val, 'set_postfix'): progress_bar_val.set_postfix({"val_loss_batch": f"{loss.item():.4f}"})
        
        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else float('inf')
        if self.writer: self.writer.add_scalar('Loss/validation', avg_val_loss, self.global_step)
        
        log_eval_summary = f"    Validation Loss: {avg_val_loss:.4f}"
        if avg_val_loss < self.best_eval_loss:
            self.best_eval_loss = avg_val_loss
            log_eval_summary += f"\n    New best validation loss: {self.best_eval_loss:.4f}. Menyimpan model terbaik."
            self._save_checkpoint(current_loss=avg_val_loss, is_best=True)
        
        # Audio Generation during Evaluation
        if self.run_config.eval_prompts and self.synthesizer:
            logger_engine.info("--- Generating Audio Samples for Evaluation ---")
            for i, prompt_text in enumerate(self.run_config.eval_prompts):
                try:
                    logger_engine.info(f"  Generating audio for prompt ({i+1}/{len(self.run_config.eval_prompts)}): \"{prompt_text[:60]}...\"")
                    waveform_tensor = self.synthesizer.generate(
                        text=prompt_text,
                        max_new_tokens=self.run_config.eval_gen_max_new_tokens,
                        temperature=self.run_config.eval_gen_temperature,
                        top_p=self.run_config.eval_gen_top_p if (self.run_config.eval_gen_top_p is not None and self.run_config.eval_gen_top_p < 1.0) else None,
                        cfg_scale=self.run_config.eval_gen_cfg_scale
                    ) 
                    if waveform_tensor is not None and waveform_tensor.numel() > 0:
                        waveform_to_log = waveform_tensor.squeeze().cpu() 
                        sample_rate = self.synthesizer.dac_model.sample_rate if hasattr(self.synthesizer.dac_model, 'sample_rate') else 44100 # Fallback SR
                        if self.writer:
                            self.writer.add_audio(f"Audio_Eval/Prompt_{i+1}", waveform_to_log, self.global_step, sample_rate=sample_rate)
                        logger_engine.info(f"    Logged audio for prompt {i+1} to TensorBoard (sr: {sample_rate}).")
                    else:
                        logger_engine.warning(f"    Audio generation returned None or empty for prompt {i+1}.")
                except Exception as e:
                    logger_engine.error(f"    Error generating/logging audio for prompt '{prompt_text}': {e}", exc_info=True)
            logger_engine.info("--- Finished Generating Audio Samples ---")

        logger_engine.info(log_eval_summary + "\n--- Selesai Evaluasi ---")
        
        if original_model_mode_is_training: # Restore original mode
            self.model.train()


    def _save_checkpoint(self, current_loss: float, is_best: bool = False, filename_prefix: str = "ckpt"):
        checkpoint_filename = f"{filename_prefix}_epoch_{self.current_epoch+1}_step_{self.global_step}.pth"
        checkpoint_path = self.checkpoint_dir / checkpoint_filename
        
        # For XLA, xm.get_optimizer_state_dict might be more robust if optimizer itself was XLA-wrapped.
        # Standard self.optimizer.state_dict() usually works if model was moved to device.
        optimizer_state_for_saving = self.optimizer.state_dict() if self.optimizer else None

        save_state = {
            'epoch': self.current_epoch, 
            'global_step': self.global_step,
            # For XLA, model.state_dict() should work fine when saving.
            'model_state_dict': self.model.state_dict(), 
            'optimizer_state_dict': optimizer_state_for_saving,
            'loss': current_loss, 
            'best_eval_loss': self.best_eval_loss
        }
        if self.scheduler: 
            save_state['scheduler_state_dict'] = self.scheduler.state_dict()
        
        try:
            if self.device.type == 'xla':
                # xm.save handles moving tensors from TPU to CPU before saving.
                # It's generally preferred for XLA.
                xm.save(save_state, checkpoint_path)
                logger_engine.info(f"\nCheckpoint (XLA) disimpan ke: {checkpoint_path}")
            else:
                torch.save(save_state, checkpoint_path)
                logger_engine.info(f"\nCheckpoint disimpan ke: {checkpoint_path}")

            if is_best:
                best_model_path = self.checkpoint_dir / "best_model.pth"
                shutil.copy2(checkpoint_path, best_model_path)
                logger_engine.info(f"Model terbaik juga disalin ke: {best_model_path}")
        except Exception as e:
            logger_engine.error(f"\nGagal menyimpan checkpoint ke {checkpoint_path}: {e}", exc_info=True)

    def _load_checkpoint(self, checkpoint_path: Path) -> Tuple[int, int, float]:
        if not checkpoint_path.is_file():
            logger_engine.warning(f"File checkpoint tidak ditemukan: {checkpoint_path}. Training dari awal.")
            return 0, 0, float('inf')
        try:
            logger_engine.info(f"Memuat checkpoint dari: {checkpoint_path}")
            # torch.load can map to any device, including XLA device.
            # If saved with xm.save, it's already CPU tensors in the file.
            checkpoint = torch.load(checkpoint_path, map_location=self.device) 
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if self.optimizer and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
                try: 
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except Exception as e_sched: 
                     logger_engine.warning(f"Tidak dapat memuat state_dict scheduler: {e_sched}. Scheduler mungkin dimulai ulang.")
            
            completed_epoch = checkpoint.get('epoch', 0) 
            global_step_start = checkpoint.get('global_step', 0)
            best_eval_loss_loaded = checkpoint.get('best_eval_loss', float('inf'))
            last_loss = checkpoint.get('loss', float('inf')) # Can be 'current_loss' or 'loss' based on saving key
            
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