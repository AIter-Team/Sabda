# In SabdaTTS/trainer/training_utils.py

import argparse
import logging
import os
import random
from pathlib import Path
# Corrected import from dataclasses
from dataclasses import dataclass, field, fields as dataclass_fields, MISSING
from typing import Optional, List, Union
from datetime import datetime

import torch
# torch_xla import is handled within get_device to avoid ImportError if not in TPU env

logger_utils = logging.getLogger(__name__)


@dataclass
class SabdaRunConfig:
    """
    Configuration for a specific training run.
    """
    # Paths and Naming
    config_path: str = field(
        default="training_configs/sabda_v1_config.json",
        metadata={"help": "Path to the SabdaConfig JSON file (e.g., training_configs/sabda_v1_config.json)"}
    )
    output_dir: str = field(
        default="checkpoints/sabda_runs",
        metadata={"help": "Directory to save checkpoints and logs."}
    )
    run_name: str = field(
        default=datetime.now().strftime("%f%S%M%H%y%U%j%Z"),
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
    warmup_steps: int = field(default=500, metadata={"help": "Number of warmup steps (optimizer updates) for the LR scheduler."})
    grad_accum_steps: int = field(default=1, metadata={"help": "Number of steps to accumulate gradients before an optimizer update."})
    
    # Logging, Saving, and Evaluation Frequency
    eval_steps: int = field(default=200, metadata={"help": "Frequency (in global steps) for evaluation. Set to 0 to disable step-based eval."})
    save_steps: int = field(default=1000, metadata={"help": "Frequency (in global steps) to save a checkpoint."})
    log_steps: int = field(default=50, metadata={"help": "Frequency (in global steps) to log training metrics."})

    # Evaluation Audio Generation Config
    eval_prompts: Optional[List[str]] = field(
        default_factory=lambda: [
            "Halo Sabda TTS, ini adalah suara percobaan.",
            "Selamat datang di Indonesia, negeri yang indah dan mempesona."
        ],
        metadata={"help": "List of sentences to generate audio for during evaluation."}
    )
    eval_gen_max_new_tokens: Optional[int] = field(
        default=None, 
        metadata={"help": "Max new tokens for eval generation (if None, SabdaConfig.data.audio_len is used)."}
    )
    eval_gen_temperature: float = field(
        default=0.7, 
        metadata={"help": "Temperature for evaluation generation."}
    )
    eval_gen_top_p: Optional[float] = field(
        default=0.9, 
        metadata={"help": "Top_p for evaluation generation (set to None or >=1.0 to effectively disable)."}
    )
    eval_gen_cfg_scale: float = field(
        default=3.0, 
        metadata={"help": "CFG scale for evaluation generation."}
    )

    # Dataset and DataLoader
    train_split_ratio: float = field(default=0.9, metadata={"help": "Ratio for training set split."})
    max_samples_dataset: Optional[int] = field(default=None, metadata={"help": "Max samples from dataset (for debugging)."})
    num_workers_dataloader: int = field(default=0, metadata={"help": "Number of worker processes for DataLoader."})
    pin_memory_dataloader: bool = field(default=True, metadata={"help": "Use pinned memory for DataLoader."})
    
    # Reproducibility and Device
    seed: int = field(default=42, metadata={"help": "Random seed for reproducibility."})
    device_target: Optional[str] = field(default=None, metadata={"help": "Target device: 'cuda', 'cpu', 'tpu', or None for auto."})

    # Mixed Precision
    use_amp: bool = field(default=False, metadata={"help": "Use Automatic Mixed Precision (AMP) for CUDA. For XLA, bfloat16 often handled via model/data dtypes."})
    
    # Unconditional Training
    unconditional_frac: float = field(default=0.0, metadata={"help": "Fraction of batches for unconditional training (0.0 to disable)."})

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        if self.resume_checkpoint_path:
            self.resume_checkpoint_path = Path(self.resume_checkpoint_path)
        
        if not (0.0 <= self.unconditional_frac <= 1.0):
            raise ValueError("unconditional_frac must be between 0.0 and 1.0")
        if not (0.0 < self.train_split_ratio <= 1.0):
            raise ValueError("train_split_ratio must be > 0.0 and <= 1.0")
        if self.grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be at least 1.")

def setup_logging(level=logging.INFO, log_file: Optional[str] = None) -> None:
    handlers = [logging.StreamHandler()]
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s (%(filename)s.%(funcName)s:%(lineno)d): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True # Force re-configuration if already configured (e.g. in notebook)
    )
    logging.getLogger(__name__).info("Logging configured.")

def get_device(target_device_type: Optional[str] = None) -> torch.device:
    logger = logging.getLogger(__name__)
    effective_target = target_device_type.lower() if target_device_type else None
    logger.info(f"Attempting to get device. Target: {effective_target or 'auto'}")

    # if effective_target == 'tpu':
    #     try:
    #         import torch_xla.core.xla_model as xm
    #         device = xm.xla_device()
    #         logger.info(f"Successfully initialized TPU: {str(device)}")
    #         logger.info(f"  XLA World Size: {xm.xrt_world_size()}, XLA Ordinal: {xm.get_ordinal()}")
    #         return device
    #     except ImportError:
    #         logger.warning("torch_xla not found. Cannot initialize TPU. Falling back to CUDA/CPU.")
    #     except Exception as e:
    #         logger.error(f"Failed to initialize TPU with torch_xla: {e}. Falling back to CUDA/CPU.", exc_info=True)

    if effective_target == 'cuda' or (effective_target is None and torch.cuda.is_available()):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            try:
                selected_gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"Using CUDA GPU: {selected_gpu_name}")
                torch.cuda.empty_cache() 
                logger.info("CUDA cache emptied.")
            except Exception as e:
                logger.error(f"CUDA available, but error during GPU info/cache empty: {e}. Using generic cuda device.", exc_info=True)
            return device
        elif effective_target == 'cuda':
            logger.warning("CUDA explicitly requested but not available. Falling back to CPU.")
    
    logger.info("Using CPU.")
    return torch.device("cpu")

def parse_run_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Sabda TTS model.")
    
    # Use the imported alias for fields from dataclasses
    config_fields_info = dataclass_fields(SabdaRunConfig) 
    
    for f_info in config_fields_info:
        arg_name = f"--{f_info.name.replace('_', '-')}" # Use hyphens for CLI args
        kwargs = {} 
        
        # Set default from dataclass field if not using default_factory
        # This uses the imported MISSING constant correctly
        if f_info.default is not MISSING: 
             kwargs['default'] = f_info.default
        elif f_info.default_factory is not MISSING:
             # For argparse, if default_factory is used, the CLI default is often None.
             # The actual default_factory logic is handled by dataclass __init__
             # or by the manual check in train.py if None is passed.
             kwargs['default'] = None 
        else:
            # No default or default_factory specified in dataclass field
            kwargs['required'] = True # Make it a required argument if no default

        if f_info.metadata and 'help' in f_info.metadata:
            # Append default value to help string if not already there
            default_val_str = str(kwargs.get('default', ' (handled by factory)' if f_info.default_factory is not MISSING else ''))
            kwargs['help'] = f_info.metadata['help'] + f" (default: {default_val_str})"
        else:
            kwargs['help'] = f"Set the {f_info.name}"


        # Type handling for argparse
        field_type = f_info.type
        field_type_origin = getattr(field_type, "__origin__", None)
        field_type_args = getattr(field_type, "__args__", tuple())

        if field_type == bool:
            # To allow --my_bool True or --my_bool False
            kwargs['type'] = lambda x: (str(x).lower() == 'true')
            # For flags, if default is False, action='store_true', if default is True, action='store_false'
            # This makes more intuitive flags like --use-amp (sets to True) or --no-use-amp (sets to False)
            # The current lambda approach is fine for explicit True/False.
        elif field_type_origin is Union and type(None) in field_type_args: # Optional[X]
            actual_type = next(t for t in field_type_args if t is not type(None))
            if actual_type == str: kwargs['type'] = str
            elif actual_type == int: kwargs['type'] = int
            elif actual_type == float: kwargs['type'] = float
            elif getattr(actual_type, "__origin__", None) is list and actual_type.__args__[0] == str: # Optional[List[str]]
                kwargs['type'] = str
                kwargs['nargs'] = '*' 
            else: 
                 kwargs['type'] = actual_type 
        elif field_type == str: kwargs['type'] = str
        elif field_type == int: kwargs['type'] = int
        elif field_type == float: kwargs['type'] = float
        elif field_type_origin is list and field_type_args and field_type_args[0] == str: # List[str]
            kwargs['type'] = str
            kwargs['nargs'] = '*'
            
        parser.add_argument(arg_name, **kwargs)
        
    return parser.parse_args()

def set_seed_for_reproducibility(seed: int):
    logger = logging.getLogger(__name__)
    random.seed(seed)
    # import numpy as np # Uncomment if you use numpy.random
    # np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Optional: for full determinism, but can impact performance
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False 
    logger.info(f"Random seed set to {seed} for reproducibility across PyTorch, random.")