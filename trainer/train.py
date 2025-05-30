# In trainer/train.py

import logging
from pathlib import Path
import torch
# from sabda.synthesizer import SabdaSynthesizer # Ensure this import is correct
# from .training_utils import get_device # Ensure this import is correct

# Corrected import from dataclasses
from dataclasses import fields as dataclass_fields, field as dataclass_field_func, MISSING # <<< Import MISSING directly

from sabda.config_schema import SabdaConfig
from sabda.dataloader import SabdaDataset
from sabda.synthesizer import SabdaSynthesizer # Assuming this is the correct path

from .training_utils import (
    SabdaRunConfig,
    parse_run_args,
    setup_logging,
    set_seed_for_reproducibility,
    get_device,
    logger_utils # Assuming you might use this, or remove if not
)
from .engine import Trainer

main_script_logger = logging.getLogger(__name__)

def main():
    parsed_cli_args = parse_run_args() 
    run_config = SabdaRunConfig(**vars(parsed_cli_args)) 

    # Ensure eval_prompts gets its default if not properly set by CLI args
    # This fix was discussed for the TypeError: '_MISSING_TYPE' object is not iterable
    # Check if run_config.eval_prompts is the _MISSING_TYPE sentinel or None
    # The actual _MISSING_TYPE object might be specific to how the dataclass is initialized
    # if it bypasses standard __init__ through a library.
    # A more direct check is if it's None or not a list after argparse.
    if not isinstance(run_config.eval_prompts, list) or run_config.eval_prompts is None:
        main_script_logger.warning(
            f"eval_prompts not set or not a list (was: {type(run_config.eval_prompts)}). Attempting to use default_factory."
        )
        # Find the field information for eval_prompts
        eval_prompts_field_info = next((f for f in dataclass_fields(SabdaRunConfig) if f.name == 'eval_prompts'), None)
        
        # Check if a default_factory is defined and it's not the MISSING sentinel
        if eval_prompts_field_info and eval_prompts_field_info.default_factory is not MISSING: # <<< CORRECTED CHECK
            run_config.eval_prompts = eval_prompts_field_info.default_factory()
            main_script_logger.info(f"Manually set eval_prompts to default: {run_config.eval_prompts}")
        else:
            # Fallback if factory not found (should not happen if SabdaRunConfig is correct)
            run_config.eval_prompts = [
                "Halo Sabda TTS, ini adalah suara percobaan.",
                "Selamat datang di Indonesia, negeri yang indah."
            ]
            main_script_logger.warning(f"Could not find default_factory for eval_prompts (or it was MISSING), using hardcoded default.")

    # ... rest of your main function from the previous correct version ...
    # (Includes cli_log_dir setup, setup_logging, set_seed_for_reproducibility, etc.)

    cli_log_dir = run_config.output_dir / run_config.run_name / "cli_script_logs" 
    cli_log_dir.mkdir(parents=True, exist_ok=True)
    cli_log_file_path = cli_log_dir / "train_cli.log"
    setup_logging(log_file=str(cli_log_file_path)) 
    
    main_script_logger.info("SabdaTTS Training Script (CLI Entry Point) Started.")
    main_script_logger.info(f"Run Configuration (after potential defaults): {run_config}") # Log potentially modified run_config

    set_seed_for_reproducibility(run_config.seed) 
    
    try:
        main_script_logger.info(f"Loading SabdaConfig from: {run_config.config_path}")
        sabda_config_loaded = SabdaConfig.load(run_config.config_path) 
        if sabda_config_loaded is None:
            main_script_logger.error(f"Gagal memuat SabdaConfig dari '{run_config.config_path}'. Keluar.")
            return
        main_script_logger.info("SabdaConfig berhasil dimuat.")

        dataset_path_from_config = sabda_config_loaded.data.dataset_path
        if not dataset_path_from_config or not Path(dataset_path_from_config).exists():
            main_script_logger.error(
                f"Path dataset ('{dataset_path_from_config}') yang didefinisikan dalam "
                f"'{run_config.config_path}' tidak valid atau tidak ada. Harap periksa file config JSON Anda. Keluar."
            )
            return
        actual_dataset_base_path = Path(dataset_path_from_config)
        main_script_logger.info(f"Menggunakan path dataset dari SabdaConfig: {actual_dataset_base_path}")

        training_device = get_device(run_config.device_target) 
        main_script_logger.info(f"Initializing SabdaSynthesizer on device: {training_device}...")
        
        synthesizer = SabdaSynthesizer(
            config=sabda_config_loaded, 
            device=training_device
        )
        main_script_logger.info("SabdaSynthesizer initialized successfully.")
        main_script_logger.info(
            f"SabdaModel (within Synthesizer) initialized with "
            f"{sum(p.numel() for p in synthesizer.model.parameters())/1e6:.2f}M parameters."
        )

        main_script_logger.info(f"Menyiapkan dataset dari: {actual_dataset_base_path}")
        full_dataset = SabdaDataset( 
            dataset_base_path=str(actual_dataset_base_path), 
            data_config=sabda_config_loaded.data,
            dac_model=synthesizer.dac_model, 
            target_sample_rate= int(getattr(synthesizer.dac_model, 'sample_rate', 44100)),
            max_samples=run_config.max_samples_dataset
        )
        
        if len(full_dataset) == 0:
            main_script_logger.error("Dataset kosong. Periksa path dataset dan metadata. Keluar."); return
        
        num_samples = len(full_dataset)
        train_dataset_instance, eval_dataset_instance = None, None
        if num_samples < 2:
            main_script_logger.warning(f"Dataset hanya {num_samples} sampel. Menggunakan semua untuk train/eval.")
            train_dataset_instance = full_dataset
            if run_config.eval_steps > 0: # Only assign to eval_dataset if evaluation is active
                 eval_dataset_instance = full_dataset
            else:
                 eval_dataset_instance = None
        else:
            train_size = int(run_config.train_split_ratio * num_samples)
            val_size = num_samples - train_size
            
            if val_size == 0 and train_size > 0 and run_config.eval_steps > 0:
                train_size -= 1
                val_size = 1
            
            if train_size <= 0 :
                 main_script_logger.warning(f"Ukuran train setelah split {train_size}. Menggunakan semua untuk train.")
                 train_dataset_instance = full_dataset
                 if val_size > 0 and run_config.eval_steps > 0: # If somehow val_size is still there
                     eval_dataset_instance = torch.utils.data.Subset(full_dataset, range(train_size, num_samples)) # Placeholder if split was weird
                 else:
                     eval_dataset_instance = None
            else:
                if val_size > 0 and run_config.eval_steps > 0:
                    train_dataset_instance, eval_dataset_instance = torch.utils.data.random_split(
                        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(run_config.seed))
                else: 
                    train_dataset_instance = full_dataset # Or torch.utils.data.Subset(full_dataset, range(train_size))
                    eval_dataset_instance = None

        main_script_logger.info(f"Ukuran dataset efektif: Training={len(train_dataset_instance)}, Evaluasi={len(eval_dataset_instance) if eval_dataset_instance else 0}.")

    except Exception as e: 
        main_script_logger.error(f"Error saat setup komponen awal: {e}", exc_info=True)
        return

    try:
        trainer_instance = Trainer( 
            model=synthesizer.model,
            dac_model=synthesizer.dac_model,
            synthesizer=synthesizer,
            sabda_config=sabda_config_loaded,
            run_config=run_config, 
            train_dataset=train_dataset_instance,
            eval_dataset=eval_dataset_instance,
            device=training_device 
        )
        main_script_logger.info("Instance Trainer berhasil dibuat.")
        # ... (rest of your try block from the previous correct version) ...
        main_script_logger.info(f"Trainer akan beroperasi dengan nama run: '{trainer_instance.run_config.run_name}'.")
        main_script_logger.info(f"Trainer akan menggunakan dataset dari path: '{actual_dataset_base_path}'.")

        main_script_logger.info("Memulai proses training via Trainer.train()...")
        trainer_instance.train()
        
        main_script_logger.info("Proses training utama berhasil diselesaikan.")

    except Exception as e: 
        main_script_logger.error(f"Error saat proses training utama: {e}", exc_info=True)
    finally:
        main_script_logger.info("Skrip Training SabdaTTS (CLI) Selesai.")

if __name__ == '__main__':
    main()