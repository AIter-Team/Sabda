# In SabdaTTS/trainer/training_utils.py

import argparse
import logging
import os
import random
from pathlib import Path
from dataclasses import dataclass, field, fields as dataclass_fields # Ensure 'fields' is imported
from typing import Optional, List # Ensure List is imported

import torch
# Ensure torch_xla is only imported within get_device if that's the strategy,
# or handle potential ImportError if it's at the top level and not always available.
# For now, get_device handles its own torch_xla import.

# Logger untuk file utilitas ini
logger_utils = logging.getLogger(__name__) # Changed from logger_utils to follow convention

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
        default="sabda_run",
        metadata={"help": "Name for the current training run (used for TensorBoard and checkpoint subdirectories)."}
    )
    resume_checkpoint_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a checkpoint file to resume training from."}
    )

    # Core Training Hyperparameters
    epochs: int = field(default=10, metadata={"help": "Total number of training epochs."})
    batch_size: int = field(default=2, metadata={"help": "Batch size for DataLoader."}) # Effective batch size will be batch_size * grad_accum_steps
    learning_rate: float = field(default=1e-4, metadata={"help": "Peak learning rate for the optimizer."})
    warmup_steps: int = field(default=500, metadata={"help": "Number of warmup steps (optimizer updates) for the LR scheduler."})
    grad_accum_steps: int = field(default=1, metadata={"help": "Number of steps to accumulate gradients before an optimizer update."})
    
    # Logging, Saving, and Evaluation Frequency (in terms of global_step, i.e., batches processed)
    eval_steps: int = field(default=200, metadata={"help": "Frequency (in global steps) for evaluation. Set to 0 to disable step-based eval (eval at end of epoch only)."})
    save_steps: int = field(default=1000, metadata={"help": "Frequency (in global steps) to save a checkpoint."})
    log_steps: int = field(default=50, metadata={"help": "Frequency (in global steps) to log training metrics."})

    # NEW: Evaluation Audio Generation Config
    eval_prompts: Optional[List[str]] = field(
        default_factory=lambda: [
            "Halo Sabda TTS, ini adalah suara percobaan.", # Default Indonesian prompts
            "Selamat datang di Indonesia, negeri yang indah dan mempesona."
        ],
        metadata={"help": "List of sentences to generate audio for during evaluation."}
    )
    eval_gen_max_new_tokens: Optional[int] = field(
        default=None, 
        metadata={"help": "Max new tokens for eval generation (if None, SabdaConfig.data.audio_len is used by synthesizer)."}
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
    train_split_ratio: float = field(default=0.9, metadata={"help": "Ratio for training set split (e.g., 0.9 for 90% train)."})
    max_samples_dataset: Optional[int] = field(default=None, metadata={"help": "Max samples from dataset (for debugging)."})
    num_workers_dataloader: int = field(default=0, metadata={"help": "Number of worker processes for DataLoader (set >0 for parallel loading)."})
    pin_memory_dataloader: bool = field(default=True, metadata={"help": "Use pinned memory for DataLoader (typically for CUDA)."})
    
    # Reproducibility and Device
    seed: int = field(default=42, metadata={"help": "Random seed for reproducibility."})
    device_target: Optional[str] = field(default=None, metadata={"help": "Target device: 'cuda', 'cpu', 'tpu', or None for auto-detection."})

    # Mixed Precision (specific to CUDA, XLA handles bfloat16 based on model/data dtype)
    use_amp: bool = field(default=False, metadata={"help": "Use Automatic Mixed Precision (AMP) for CUDA. For XLA, bfloat16 is often handled via model/data dtypes."})
    
    # Unconditional Training
    unconditional_frac: float = field(default=0.0, metadata={"help": "Fraction of batches for unconditional training (0.0 to disable)."})

    def __post_init__(self):
        # Ensure paths are Path objects for easier manipulation
        self.output_dir = Path(self.output_dir)
        if self.resume_checkpoint_path:
            self.resume_checkpoint_path = Path(self.resume_checkpoint_path)
        
        # Validasi nilai
        if not (0.0 <= self.unconditional_frac <= 1.0):
            raise ValueError("unconditional_frac must be between 0.0 and 1.0")
        if not (0.0 < self.train_split_ratio <= 1.0): # train_split_ratio can be 1.0 if no validation set
            raise ValueError("train_split_ratio must be between 0.0 (exclusive) and 1.0 (inclusive)")
        if self.grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be at least 1.")


def setup_logging(level=logging.INFO, log_file: Optional[str] = None) -> None:
    """
    Configures basic logging.
    Messages will be printed to the console and optionally to a log file.
    """
    # Make sure logger_utils is defined if used, or use a local logger.
    # For simplicity, this function configures the root logger.
    handlers = [logging.StreamHandler()]
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            # This print might not be visible if logging is configured before this print occurs.
            # Consider logger_utils.info(f"Creating log directory: {log_dir}") after basicConfig
            os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(log_file)) # Add file handler

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s (%(filename)s.%(funcName)s:%(lineno)d): %(message)s", # Changed format for more detail
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers
    )
    # After basicConfig, loggers can be retrieved and used.
    logging.getLogger(__name__).info("Logging configured.")


def get_device(target_device_type: Optional[str] = None) -> torch.device:
    """
    Selects and returns the best available PyTorch device.
    Prioritizes CUDA if available and no specific target is set.
    Allows specifying 'tpu', 'cuda', or 'cpu' as a target.
    """
    logger = logging.getLogger(__name__) # Get logger for this function

    effective_target = target_device_type.lower() if target_device_type else None
    logger.info(f"Attempting to get device. Target: {effective_target or 'auto'}")

    if effective_target == 'tpu':
        try:
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            logger.info(f"Successfully initialized TPU: {str(device)}")
            # xm.xrt_world_size() and xm.get_ordinal() useful for distributed, good to log
            logger.info(f"  XLA World Size: {xm.xrt_world_size()}, XLA Ordinal: {xm.get_ordinal()}")
            return device
        except ImportError:
            logger.warning("torch_xla not found. Cannot initialize TPU. Falling back to CUDA/CPU.")
        except Exception as e: # Catch more general exceptions during XLA init
            logger.error(f"Failed to initialize TPU with torch_xla: {e}. Falling back to CUDA/CPU.", exc_info=True)
        # Fall through to CUDA/CPU if TPU fails or wasn't the primary target from here

    if effective_target == 'cuda' or (effective_target is None and torch.cuda.is_available()):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            try:
                selected_gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"Using CUDA GPU: {selected_gpu_name}")
                torch.cuda.empty_cache() 
                logger.info("CUDA cache emptied.")
            except Exception as e: # Handle cases where get_device_name might fail after is_available
                logger.error(f"CUDA available, but error during GPU info/cache empty: {e}. Using generic cuda device.", exc_info=True)
            return device
        elif effective_target == 'cuda': # CUDA explicitly requested but not available
            logger.warning("CUDA explicitly requested but not available. Falling back to CPU.")
    
    # Default to CPU if no other device is selected or available
    logger.info("Using CPU.")
    return torch.device("cpu")


def parse_run_args() -> argparse.Namespace:
    """
    Parses command-line arguments for the training run,
    using fields and defaults from SabdaRunConfig.
    """
    parser = argparse.ArgumentParser(description="Train the Sabda TTS model.")
    
    config_fields = dataclass_fields(SabdaRunConfig) # Use the imported 'fields'
    
    for f_info in config_fields:
        arg_name = f"--{f_info.name}"
        kwargs = {} 
        
        # Set default from dataclass field if not using default_factory
        if f_info.default is not field.MISSING: # field.MISSING from dataclasses
             kwargs['default'] = f_info.default
        elif f_info.default_factory is not field.MISSING:
             # For argparse, if default_factory is used, often the default CLI value is None,
             # letting the dataclass __init__ handle the factory.
             # Or, you might invoke factory here if it's simple and serializable.
             # For now, let's make it None so it's not explicitly passed unless user specifies it.
             kwargs['default'] = None 
        else:
            kwargs['default'] = None # General fallback for argparse if no default specified

        if f_info.metadata and 'help' in f_info.metadata:
            kwargs['help'] = f_info.metadata['help']
        else: # Add a default help string
            kwargs['help'] = f"Set the {f_info.name} (default: {kwargs.get('default', 'None from factory')})"

        # Type handling for argparse
        field_type_origin = getattr(f_info.type, "__origin__", None)
        field_type_args = getattr(f_info.type, "__args__", tuple())

        if f_info.type == bool:
            # For bools, if default is True, action is 'store_false', else 'store_true'
            # This makes it a flag: --my_bool sets to True (if default False), or --no-my_bool sets to False
            # However, to allow --my_bool True/False, type=lambda x: x.lower() == 'true' is better
            if kwargs.get('default') is True: # Check actual default value
                 kwargs['action'] = 'store_false'
                 arg_name = f"--no-{f_info.name.replace('_', '-')}" # e.g. --use-amp becomes --no-use-amp
                 kwargs['dest'] = f_info.name # Ensure it maps back to the correct field name
            else: # Default is False or None treated as False for flag
                 kwargs['action'] = 'store_true'
        elif field_type_origin is Union and type(None) in field_type_args: # Optional[X]
            # Get the actual type from Optional[X], e.g., X
            actual_type = next(t for t in field_type_args if t is not type(None))
            if actual_type == str: kwargs['type'] = str
            elif actual_type == int: kwargs['type'] = int
            elif actual_type == float: kwargs['type'] = float
            elif getattr(actual_type, "__origin__", None) is list and actual_type.__args__[0] == str: # Optional[List[str]]
                kwargs['type'] = str
                kwargs['nargs'] = '*' # Allows multiple string arguments for a list
                kwargs['help'] += " (space-separated list of strings)"
            else: # Fallback for other Optional types
                 kwargs['type'] = actual_type 
        elif f_info.type == str: kwargs['type'] = str
        elif f_info.type == int: kwargs['type'] = int
        elif f_info.type == float: kwargs['type'] = float
        # Note: Path type is not directly supported by argparse, usually handled as str.
        
        # Override default to None for nargs='*' if default was from factory to avoid argparse error
        if kwargs.get('nargs') == '*' and kwargs.get('default') is None and f_info.default_factory is not field.MISSING:
            pass # Keep default as None, factory handled post-init
            
        parser.add_argument(arg_name, **kwargs)
        
    return parser.parse_args()

def set_seed_for_reproducibility(seed: int):
    logger = logging.getLogger(__name__) # Get logger for this function
    random.seed(seed)
    torch.manual_seed(seed)
    # numpy.random.seed(seed) # If using numpy directly for randomness
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True # Can slow down training
        # torch.backends.cudnn.benchmark = False    # Can slow down training
    logger.info(f"Random seed set to {seed} for reproducibility across PyTorch, random.")