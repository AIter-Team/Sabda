import logging
from pathlib import Path
import torch
import dac

from sabda.config_schema import SabdaConfig
from sabda.layers import SabdaModel
from sabda.dataloader import SabdaDataset

from .training_utils import (
    SabdaRunConfig,
    parse_run_args,
    setup_logging,
    set_seed_for_reproducibility,
)
from .engine import Trainer


# Logger untuk skrip utama ini akan diinisialisasi setelah setup_logging
main_script_logger = logging.getLogger(__name__)

def main():
    parsed_cli_args = parse_run_args() #
    run_config = SabdaRunConfig(**vars(parsed_cli_args)) #

    cli_log_dir = run_config.output_dir / run_config.run_name / "cli_script_logs" 
    cli_log_dir.mkdir(parents=True, exist_ok=True)
    cli_log_file_path = cli_log_dir / "train_cli.log"
    setup_logging(log_file=str(cli_log_file_path)) #
    
    main_script_logger.info("SabdaTTS Training Script (CLI Entry Point) Started.")
    main_script_logger.info(f"Run Configuration from CLI/Defaults: {run_config}")

    set_seed_for_reproducibility(run_config.seed) #
    
    try:
        main_script_logger.info(f"Loading SabdaConfig from: {run_config.config_path}")
        sabda_config_loaded = SabdaConfig.load(run_config.config_path) #
        if sabda_config_loaded is None:
            main_script_logger.error(f"Gagal memuat SabdaConfig dari '{run_config.config_path}'. Keluar.")
            return
        main_script_logger.info("SabdaConfig berhasil dimuat.")

        # --- AMBIL dataset_path DARI SabdaConfig ---
        dataset_path_from_config = sabda_config_loaded.data.dataset_path
        if not dataset_path_from_config or not Path(dataset_path_from_config).exists():
            main_script_logger.error(
                f"Path dataset ('{dataset_path_from_config}') yang didefinisikan dalam "
                f"'{run_config.config_path}' tidak valid atau tidak ada. Harap periksa file config JSON Anda. Keluar."
            )
            return
        # Konversi ke Path object untuk penggunaan selanjutnya
        actual_dataset_base_path = Path(dataset_path_from_config)
        main_script_logger.info(f"Menggunakan path dataset dari SabdaConfig: {actual_dataset_base_path}")
        # --- SELESAI PENGAMBILAN dataset_path ---

        main_script_logger.info("Memuat model DAC...")
        dac_model_loaded = dac.DAC.load(dac.utils.download())
        main_script_logger.info("Model DAC berhasil dimuat.")

        main_script_logger.info("Menginisialisasi SabdaModel...")
        sabda_model_loaded = SabdaModel(sabda_config_loaded) #
        main_script_logger.info(f"SabdaModel diinisialisasi dengan {sum(p.numel() for p in sabda_model_loaded.parameters())/1e6:.2f}M parameter.")

        main_script_logger.info(f"Menyiapkan dataset dari: {actual_dataset_base_path}")
        full_dataset = SabdaDataset( #
            dataset_base_path=str(actual_dataset_base_path), # SabdaDataset mungkin expect str
            data_config=sabda_config_loaded.data,
            dac_model=dac_model_loaded,
            target_sample_rate=44100, 
            max_samples=run_config.max_samples_dataset
        )
        # ... (sisa logika split dataset tetap sama) ...
        if len(full_dataset) == 0:
            main_script_logger.error("Dataset kosong. Periksa path dataset dan metadata. Keluar."); return
        num_samples = len(full_dataset); train_dataset_instance, eval_dataset_instance = None, None
        if num_samples < 2:
            main_script_logger.warning(f"Dataset hanya {num_samples} sampel. Menggunakan semua untuk train/eval.")
            train_dataset_instance, eval_dataset_instance = full_dataset, full_dataset
        else:
            train_size = int(run_config.train_split_ratio * num_samples); val_size = num_samples - train_size
            if val_size == 0 and train_size > 0: train_size -= 1; val_size = 1
            if train_size <= 0 :
                 main_script_logger.warning(f"Ukuran train setelah split {train_size}. Menggunakan semua untuk train/eval.")
                 train_dataset_instance, eval_dataset_instance = full_dataset, full_dataset
            else:
                train_dataset_instance, eval_dataset_instance = torch.utils.data.random_split(
                    full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(run_config.seed))
        main_script_logger.info(f"Ukuran dataset efektif: Training={len(train_dataset_instance)}, Evaluasi={len(eval_dataset_instance if eval_dataset_instance else [])}.")

    except Exception as e: 
        main_script_logger.error(f"Error saat setup komponen awal: {e}", exc_info=True)
        return

    try:
        trainer_instance = Trainer( #
            model=sabda_model_loaded,
            dac_model=dac_model_loaded,
            sabda_config=sabda_config_loaded,
            run_config=run_config, 
            train_dataset=train_dataset_instance,
            eval_dataset=eval_dataset_instance
        )
        main_script_logger.info("Instance Trainer berhasil dibuat.")
        main_script_logger.info(f"Trainer akan beroperasi dengan nama run: '{trainer_instance.run_config.run_name}'.")
        main_script_logger.info(f"Trainer akan menggunakan dataset dari path: '{actual_dataset_base_path}'.") # Log path dataset yang digunakan

        main_script_logger.info("Memulai proses training via Trainer.train()...")
        trainer_instance.train()
        
        main_script_logger.info("Proses training utama berhasil diselesaikan.")
    except Exception as e: 
        main_script_logger.error(f"Error saat proses training utama: {e}", exc_info=True)
    finally:
        main_script_logger.info("Skrip Training SabdaTTS (CLI) Selesai.")

if __name__ == '__main__':
    main()