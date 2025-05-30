import torch
from torch import nn
from typing import Tuple

def _normalize_axes(axes: Tuple[int, ...], ndim: int) -> Tuple[int, ...]:
    return tuple(ax if ax >= 0 else ndim + ax for ax in axes)


def _str_to_dtype(dtype_str: str) -> torch.dtype | None:
    # Allow None for default behavior
    if dtype_str is None or dtype_str.lower() == "none":
        return None
    if dtype_str == "float32":
        return torch.float32
    elif dtype_str == "float16":
        return torch.float16
    elif dtype_str == "bfloat16":
        return torch.bfloat16
    else:
        raise ValueError(f"Unsupported dtype string: {dtype_str}")
    

def get_activation_fn(activation_string: str) -> nn.Module:  # Return Module instance
    """Maps activation string to PyTorch activation function module."""
    
    if activation_string == "gelu":
        return nn.GELU()
    elif activation_string == "relu":
        return nn.ReLU()
    elif activation_string == "silu" or activation_string == "swish":
        return nn.SiLU()
    elif activation_string == "linear":
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported activation function: {activation_string}")