import logging
import math # Untuk math.ceil saat estimasi step
import random # Untuk unconditional training
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any # Any untuk dac_model

import torch
import torch.nn.functional as F # Jika _calculate_loss_hf_compatible membutuhkannya
import dac# Pastikan impor DAC benar

# Impor dari Hugging Face Transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Trainer as HfTrainer, # Beri alias agar tidak bentrok jika Anda punya kelas Trainer lain
    TrainerState,
    TrainerControl,
    TrainerCallback
)
from transformers.trainer_utils import EvalPrediction # Untuk compute_metrics jika dipakai nanti

# Impor dari paket sabda_tts Anda
# Asumsi skrip ini dijalankan dari root proyek, atau PYTHONPATH sudah diatur
# sehingga 'sabda_tts' bisa diimpor.
from sabda.config_schema import SabdaConfig, DataConfig
from sabda.layers import SabdaModel
from sabda.dataloader import SabdaDataset, create_sabda_collate_fn
# Impor utilitas logging dan seed dari trainer kustom (bisa dipakai ulang)
from trainer.training_utils import setup_logging, set_seed_for_reproducibility # Sesuaikan path jika perlu

# Logger untuk skrip ini
logger_hf_train = logging.getLogger(__name__)

# Langkah 1: Definisikan Data Class untuk Argumen Kustom
@dataclass
class ModelArguments:
    """
    Argumen yang berkaitan dengan model SabdaTTS dan path konfigurasi utama.
    """
    sabda_config_path: str = field(
        default="training_configs/sabda_v1_config.json",
        metadata={"help": "Path ke file SabdaConfig JSON (arsitektur model & data)."}
    )
    # Tambahkan argumen spesifik model lain di sini jika perlu di-override via CLI
    # contoh: pretrained_model_path: Optional[str] = field(default=None, metadata={"help": "Path ke model pre-trained jika ada."})

@dataclass
class DataTrainingArguments:
    """
    Argumen yang berkaitan dengan data, dan parameter training kustom lainnya
    yang tidak ada di TrainingArguments standar.
    """
    dataset_path: Optional[str] = field(
        default=None, 
        metadata={"help": "Path ke dataset. Jika None, akan diambil dari SabdaConfig."}
    )
    unconditional_frac: float = field(
        default=0.0,
        metadata={"help": "Fraksi batch untuk training unconditional (0.0 untuk menonaktifkan)."}
    )
    train_split_ratio: float = field(
        default=0.9,
        metadata={"help": "Rasio untuk pembagian set training (misalnya, 0.9 untuk 90% train)."}
    )
    max_samples_dataset: Optional[int] = field(
        default=None,
        metadata={"help": "Jumlah sampel maksimum dari dataset (untuk debugging)."}
    )
    # Anda bisa memindahkan num_workers dan pin_memory ke sini dari TrainingArguments
    # jika ingin kontrol yang lebih eksplisit dan terpisah, atau biarkan di TrainingArguments.
    # num_workers: int = field(default=0, metadata={"help": "Number of dataloader workers."})
    # pin_memory: bool = field(default=True, metadata={"help": "Pin memory for dataloader."})


# Langkah 2: Definisikan Fungsi `compute_loss`
# Fungsi ini akan dioper ke HfTrainer.
# Ia menerima output model dan input batch, lalu mengembalikan loss.
def compute_loss_for_sabda(model: SabdaModel, inputs: Dict[str, torch.Tensor], return_outputs: bool = False):
    """
    Menghitung loss untuk SabdaModel, kompatibel dengan HuggingFace Trainer.
    Mengadaptasi logika dari _calculate_loss di engine.py.
    """
    # 'inputs' adalah dictionary yang dikembalikan oleh data_collator.
    # Pastikan semua argumen yang dibutuhkan SabdaModel.forward() ada di 'inputs'.
    
    # Ambil output model (logits)
    # Model dipanggil dengan deterministic=False karena ini dalam konteks training
    model_outputs = model(
        src_tokens=inputs.get("src_tokens"),
        tgt_tokens=inputs.get("tgt_tokens"), 
        src_pos=inputs.get("src_positions"),
        tgt_pos=inputs.get("tgt_positions"),
        enc_self_attn_mask=inputs.get("enc_self_attn_mask"),
        dec_self_attn_mask=inputs.get("dec_self_attn_mask"),
        dec_cross_attn_mask=inputs.get("dec_cross_attn_mask"),
        deterministic=False 
    )
    logits = model_outputs # SabdaModel.forward() Anda mengembalikan logits

    # Akses DataConfig. Bisa dari model jika Anda menyimpannya di sana,
    # atau jika model adalah instance dari PreTrainedModel HF, bisa dari model.config.
    # Untuk sekarang, kita asumsikan model SabdaModel Anda memiliki atribut `config`
    # yang merupakan instance SabdaConfig, yang memiliki atribut `data`.
    # Ini perlu disesuaikan berdasarkan implementasi SabdaModel Anda.
    if not hasattr(model, 'config') or not hasattr(model.config, 'data'):
        raise ValueError("SabdaModel harus memiliki atribut 'config' (SabdaConfig) "
                         "dengan sub-atribut 'data' (DataConfig) untuk compute_loss ini.")
    data_cfg: DataConfig = model.config.data #

    # Logika inti perhitungan loss (diadaptasi dari _calculate_loss Anda)
    batch_targets = inputs.get("tgt_tokens")
    target_lengths = inputs.get("tgt_lens")

    max_len_in_batch_for_loss = target_lengths.max().item()
    if max_len_in_batch_for_loss <= 1: # BOS saja, tidak ada yang diprediksi
        loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        return (loss, model_outputs) if return_outputs else loss

    sliced_logits = logits[:, :max_len_in_batch_for_loss - 1, :, :]
    sliced_targets = batch_targets[:, 1:max_len_in_batch_for_loss, :]
    
    B, T_pred, C_channels, V_tgt = sliced_logits.shape
    if T_pred == 0: 
         loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
         return (loss, model_outputs) if return_outputs else loss

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
            ignore_index=data_cfg.audio_pad_value # Menggunakan data_cfg
        )
        accumulated_weighted_channel_loss += channel_weights_list[c_idx] * channel_loss
    
    sum_weights = sum(channel_weights_list)
    loss = accumulated_weighted_channel_loss / sum_weights if sum_weights > 0 else torch.tensor(0.0, device=logits.device)
    
    return (loss, model_outputs) if return_outputs else loss


def main():
    # Langkah 3: Parsing Argumen
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging dasar (Trainer HF memiliki sistem logging sendiri yang lebih canggih)
    # Anda bisa menggunakan setup_logging Anda untuk log dari skrip ini.
    # Trainer HF akan handle logging ke console, file, dan integrasi lain (TensorBoard, dll.)
    # berdasarkan TrainingArguments (misal, training_args.logging_dir)
    log_level = training_args.get_process_log_level()
    setup_logging(level=log_level) # Bisa diatur levelnya dari TrainingArguments

    logger_hf_train.warning(
        f"Proses rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16 or training_args.bf16}"
    )
    logger_hf_train.info(f"Training/evaluation parameters {training_args}")
    logger_hf_train.info(f"Model parameters {model_args}")
    logger_hf_train.info(f"Data/Custom Training parameters {data_args}")

    # Set seed sebelum inisialisasi apa pun (Trainer HF juga melakukannya)
    set_seed_for_reproducibility(training_args.seed)

    # Langkah 4: Memuat Komponen Inti SabdaTTS
    logger_hf_train.info(f"Memuat SabdaConfig dari: {model_args.sabda_config_path}")
    sabda_config_loaded = SabdaConfig.load(model_args.sabda_config_path)
    if sabda_config_loaded is None:
        logger_hf_train.error("Gagal memuat SabdaConfig. Keluar."); return
    logger_hf_train.info("SabdaConfig berhasil dimuat.")

    actual_dataset_path = Path(data_args.dataset_path if data_args.dataset_path else sabda_config_loaded.data.dataset_path)
    logger_hf_train.info(f"Path dataset yang akan digunakan: {actual_dataset_path}")
    if not actual_dataset_path.exists():
        logger_hf_train.error(f"Path dataset '{actual_dataset_path}' tidak ditemukan. Keluar."); return

    logger_hf_train.info("Memuat model DAC...")
    dac_model_loaded = dac.DAC.load(dac.utils.download())
    # Tidak perlu .to(device) di sini, Trainer HF akan menanganinya
    dac_model_loaded.eval()
    logger_hf_train.info("Model DAC berhasil dimuat.")

    logger_hf_train.info("Menginisialisasi SabdaModel...")
    # Simpan sabda_config ke dalam model agar bisa diakses oleh compute_loss
    sabda_model_instance = SabdaModel(sabda_config_loaded) 
    # Tidak perlu .to(device) di sini
    logger_hf_train.info(f"SabdaModel diinisialisasi dengan {sum(p.numel() for p in sabda_model_instance.parameters())/1e6:.2f}M parameter.")

    logger_hf_train.info(f"Menyiapkan dataset dari: {actual_dataset_path}")
    full_dataset = SabdaDataset(
        dataset_base_path=str(actual_dataset_path),
        data_config=sabda_config_loaded.data,
        dac_model=dac_model_loaded,
        target_sample_rate=44100, # Sebaiknya dari config
        max_samples=data_args.max_samples_dataset
    )
    if len(full_dataset) == 0: logger_hf_train.error("Dataset kosong. Keluar."); return
    
    train_dataset_instance, eval_dataset_instance = None, None
    # ... (logika split dataset Anda, menggunakan data_args.train_split_ratio dan training_args.seed) ...
    num_samples = len(full_dataset)
    if num_samples < 2: # Atau jika training_args.do_eval adalah False, eval_dataset_instance bisa None
        train_dataset_instance = full_dataset
        eval_dataset_instance = full_dataset if training_args.do_eval else None # Hanya buat jika do_eval
        logger_hf_train.warning(f"Dataset hanya {num_samples} sampel. Disesuaikan untuk train/eval.")
    else:
        train_size = int(data_args.train_split_ratio * num_samples)
        val_size = num_samples - train_size
        if val_size == 0 and train_size > 0 and training_args.do_eval: train_size -= 1; val_size = 1
        if train_size <= 0 :
             train_dataset_instance = full_dataset
             eval_dataset_instance = full_dataset if training_args.do_eval and val_size > 0 else None
             logger_hf_train.warning(f"Ukuran train setelah split {train_size}. Disesuaikan.")
        else:
            if val_size == 0 and training_args.do_eval: # Jika hanya ingin train set
                train_dataset_instance = full_dataset
                eval_dataset_instance = None
            elif not training_args.do_eval:
                train_dataset_instance = full_dataset
                eval_dataset_instance = None
            else:
                train_dataset_instance, eval_dataset_instance = torch.utils.data.random_split(
                    full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(training_args.seed))
    logger_hf_train.info(f"Ukuran dataset efektif: Training={len(train_dataset_instance)}, Evaluasi={len(eval_dataset_instance) if eval_dataset_instance else 0}.")


    # Data Collator (menghasilkan tensor CPU)
    data_collator_instance = create_sabda_collate_fn(sabda_config_loaded.data)

    # (Tempat untuk mendefinisikan compute_metrics dan callbacks jika perlu)
    # def compute_metrics_for_sabda(p: EvalPrediction):
    #     # ...
    #     return {"dummy_metric": 0.0}

    # Langkah 5: Inisialisasi HuggingFace Trainer
    trainer = HfTrainer(
        model=sabda_model_instance,
        args=training_args,
        train_dataset=train_dataset_instance,
        eval_dataset=eval_dataset_instance,
        data_collator=data_collator_instance,
        compute_loss=compute_loss_for_sabda, # Menggunakan fungsi compute_loss kustom kita
        # compute_metrics=compute_metrics_for_sabda, # Opsional
        # callbacks=[AudioGenerationCallback(...)], # Opsional, akan kita bahas nanti
    )
    logger_hf_train.info("Instance HuggingFace Trainer berhasil dibuat.")

    # Langkah 6: Mulai Training
    if training_args.do_train:
        logger_hf_train.info("*** Mulai Training ***")
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        
        # Simpan model dan state setelah training selesai
        trainer.save_model() # Menyimpan model ke output_dir
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state() # Menyimpan state Trainer (termasuk RNG state)
        logger_hf_train.info(f"Model dan state training disimpan ke {training_args.output_dir}")

    # Langkah 7: Evaluasi
    if training_args.do_eval:
        logger_hf_train.info("*** Mulai Evaluasi ***")
        metrics = trainer.evaluate()
        # Log dan simpan metrik evaluasi
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        logger_hf_train.info(f"Metrik evaluasi: {metrics}")

    logger_hf_train.info("Skrip Training SabdaTTS (dengan HuggingFace Trainer) Selesai.")


if __name__ == '__main__':
    main()