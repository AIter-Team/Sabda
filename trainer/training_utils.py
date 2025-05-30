import argparse
import logging
import os
import random
from pathlib import Path
from dataclasses import dataclass, field, fields as dataclass_fields
from typing import Optional, List

import torch

# Logger untuk file utilitas ini
logger_utils = logging.getLogger(__name__)

@dataclass
class SabdaRunConfig:
    """
    Configuration for a specific training run.
    """
    # Paths and Naming
    config_path: str = field(
        default="training_configs/sabda_v1_config.json", #
        metadata={"help": "Path to the SabdaConfig JSON file (e.g., training_configs/sabda_v1_config.json)"}
    )
    output_dir: str = field(
        default="checkpoints/sabda_runs", #
        metadata={"help": "Directory to save checkpoints and logs."}
    )
    run_name: str = field(
        default="sabda_run",
        metadata={"help": "Name for the current training run (used for TensorBoard and checkpoint subdirectories)."}
    )
    resume_checkpoint_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a checkpoint file to resume training from."}
    )

    # Core Training Hyperparameters
    epochs: int = field(default=10, metadata={"help": "Total number of training epochs."})
    batch_size: int = field(default=2, metadata={"help": "Batch size for DataLoader."})
    learning_rate: float = field(default=1e-4, metadata={"help": "Peak learning rate for the optimizer."})
    warmup_steps: int = field(default=500, metadata={"help": "Number of warmup steps for the learning rate scheduler."})
    grad_accum_steps: int = field(default=1, metadata={"help": "Number of steps to accumulate gradients."})
    
    # Logging, Saving, and Evaluation Frequency
    eval_steps: int = field(default=200, metadata={"help": "Frequency (in steps) for evaluation."}) # Existing
    save_steps: int = field(default=1000, metadata={"help": "Frequency (in steps) to save a checkpoint."}) # Existing
    log_steps: int = field(default=50, metadata={"help": "Frequency (in steps) to log training metrics."}) # Existing

    # NEW: Evaluation Audio Generation Config
    eval_prompts: Optional[List[str]] = field(
        default_factory=lambda: [
            "Halo Sabda TTS, ini adalah suara percobaan.",
            "Selamat datang di Indonesia, negeri yang indah."
            # Add more Indonesian prompts as desired
        ],
        metadata={"help": "List of sentences to generate audio for during evaluation."}
    )
    eval_gen_max_new_tokens: Optional[int] = field(
        default=None, # If None, SabdaSynthesizer.generate will use SabdaConfig.data.audio_len
        metadata={"help": "Max new tokens for eval generation."}
    )
    eval_gen_temperature: float = field(
        default=0.7, 
        metadata={"help": "Temperature for evaluation generation."}
    )
    eval_gen_top_p: Optional[float] = field(
        default=0.9, 
        metadata={"help": "Top_p for evaluation generation (set to None or >=1.0 to disable)."}
    )
    eval_gen_cfg_scale: float = field(
        default=3.0, 
        metadata={"help": "CFG scale for evaluation generation."}
    )

    # Dataset and DataLoader
    train_split_ratio: float = field(default=0.9, metadata={"help": "Ratio for training set split (e.g., 0.9 for 90% train)."})
    max_samples_dataset: Optional[int] = field(default=None, metadata={"help": "Max samples from dataset (for debugging)."})
    num_workers_dataloader: int = field(default=0, metadata={"help": "Number of worker processes for DataLoader."})
    pin_memory_dataloader: bool = field(default=True, metadata={"help": "Use pinned memory for DataLoader."})
    
    # Reproducibility and Device
    seed: int = field(default=42, metadata={"help": "Random seed for reproducibility."})
    device_target: Optional[str] = field(default=None, metadata={"help": "Target device: 'cuda', 'cpu', 'tpu', or None for auto."})

    # Mixed Precision
    use_amp: bool = field(default=False, metadata={"help": "Use Automatic Mixed Precision (AMP)."})
    
    # Unconditional Training
    unconditional_frac: float = field(default=0.0, metadata={"help": "Fraction of batches for unconditional training (0.0 to disable)."})

    # Optional: Test sentences for audio generation during evaluation
    # test_sentences_eval: Optional[Dict[str, str]] = field(default=None, metadata={"help": "Dict of lang_tag: sentence for eval audio samples."})


    def __post_init__(self):
        # Ensure paths are Path objects for easier manipulation
        self.output_dir = Path(self.output_dir)
        if self.resume_checkpoint_path:
            self.resume_checkpoint_path = Path(self.resume_checkpoint_path)
        
        # Validasi nilai
        if not (0.0 <= self.unconditional_frac <= 1.0):
            raise ValueError("unconditional_frac must be between 0.0 and 1.0")
        if not (0.0 < self.train_split_ratio < 1.0):
            raise ValueError("train_split_ratio must be between 0.0 (exclusive) and 1.0 (exclusive)")


def setup_logging(level=logging.INFO, log_file: Optional[str] = None) -> None: #
    """
    Configures basic logging.
    Messages will be printed to the console and optionally to a log file.
    """
    handlers = [logging.StreamHandler()]
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            print(log_dir)
            os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s (%(module)s.%(funcName)s): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers
    )


def get_device(target_device_type: Optional[str] = None) -> torch.device:
    """
    Selects and returns the best available PyTorch device (CUDA, CPU or TPU).
    Prioritizes CUDA if available and no specific target is set.
    Allows specifying 'tpu', 'cuda', or 'cpu' as a target.

    Args:
        target_device_type (Optional[str]): Target device type ('cuda', 'cpu', 'tpu')

    Returns:
        torch.device: The selected PyTorch device
    """

    logger = logging.getLogger(__name__) # Get Logger instance

    if target_device_type:
        target_device_type = target_device_type.lower()
        logger.info(f"Attempting to use target_device_type: {target_device_type}")

    # TPU (PyTorch/XLA) Check
    # if target_device_type == 'tpu':
    #     try:
    #         import torch_xla.core.xla_model as xm
    #         device  = xm.xla_device()
    #         logger.info(f"Device: Successfully initialized TPU: {str(device)}")
    #         logger.info(f"  XLA World Size: {xm.xrt_world_size()}, XLA Ordinal: {xm.get_ordinal()}")
    #         return device
    #     except ImportError:
    #         logger.warning("torch_xla not found. Cannot initialize TPU. Falling back to CUDA/CPU.")
    #     except Exception as e:
    #         logger.error(f"Failed to initialize TPU with torch_xla: {e}. Falling back to CUDA/CPU.")

    # CUDA (GPU) Check
    elif target_device_type is None or target_device_type == 'cuda':
        if torch.cuda.is_available():
            device = torch.device("cuda")
            selected_gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"Device: Using CUDA GPU: {selected_gpu_name}")

            # Empty CUDA cache
            torch.cuda.empty_cache() 
            logger.info("CUDA cache emptied.")
        elif target_device_type == 'cuda':
            logger.warning("CUDA explicitly requested but not available. Falling back to CPU.")

    # CPU
    else:
        device = torch.device("cpu")
        logger.info("Device: Using CPU.")
    return device

def parse_run_args() -> argparse.Namespace: # Diadaptasi dari trainer.py Anda
    """
    Parses command-line arguments for the training run,
    using fields and defaults from SabdaRunConfig.
    """
    parser = argparse.ArgumentParser(description="Train the Sabda TTS model.")
    
    # Mengambil field dari SabdaRunConfig untuk membuat argumen secara dinamis
    for f_info in dataclass_fields(SabdaRunConfig):
        arg_name = f"--{f_info.name}"
        kwargs = {'default': f_info.default} # Selalu ada default dari dataclass
        
        if f_info.metadata and 'help' in f_info.metadata:
            kwargs['help'] = f_info.metadata['help']

        # Menentukan tipe argumen
        # Untuk tipe bool, argparse menanganinya secara khusus (store_true/store_false)
        # atau kita bisa menggunakan type=lambda x: (str(x).lower() == 'true')
        if f_info.type == bool:
            # Jika defaultnya True, buat flag untuk membuatnya False, dan sebaliknya
            if f_info.default is True:
                kwargs['action'] = 'store_false' # --arg_name akan set ke False
                # Nama argumen diubah agar lebih intuitif, misal --no-use_amp
                # arg_name = f"--no-{f_info.name.replace('_', '-')}" # Ini bisa jadi rumit
                # Untuk konsistensi, kita tetap pakai nama field, dan user input 'True'/'False'
                kwargs.pop('action', None) # Hapus action jika ada
                kwargs['type'] = lambda x: (str(x).lower() == 'true')
            else: # Default False
                kwargs['type'] = lambda x: (str(x).lower() == 'true')
        elif f_info.type == Optional[str] or f_info.type == str : # Pydantic sering pakai str untuk Path juga
             kwargs['type'] = str
        elif f_info.type == Optional[int] or f_info.type == int:
             kwargs['type'] = int
        elif f_info.type == Optional[float] or f_info.type == float:
             kwargs['type'] = float
        else:
            # Fallback jika tipe tidak dikenali secara eksplisit, coba gunakan tipe aslinya
            kwargs['type'] = f_info.type 
            
        # Untuk field yang opsional dan defaultnya None, argparse menghandle dengan baik
        # jika type di-set (misal type=str). Jika tidak ada input, akan jadi None.
        if f_info.default is None and 'type' not in kwargs:
             kwargs['type'] = str # Default ke str untuk argumen opsional yang bisa None

        # Tambahkan choices jika ada di metadata (belum kita pakai di SabdaRunConfig)
        # if 'choices' in f_info.metadata:
        #     kwargs['choices'] = f_info.metadata['choices']
            
        parser.add_argument(arg_name, **kwargs)
        
    return parser.parse_args()

def set_seed_for_reproducibility(seed: int): #
    """Sets the seed for random, numpy, and torch for reproducibility."""
    logger = logging.getLogger(__name__)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed} for reproducibility.")