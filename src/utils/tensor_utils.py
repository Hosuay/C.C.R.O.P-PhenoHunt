"""
Tensor utility functions for safe conversions and handling.
Ensures compatibility with PyTorch 2.0+ and handles edge cases.
"""

import torch
import numpy as np
import pandas as pd
from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)


def safe_to_tensor(
    data: Union[np.ndarray, pd.DataFrame, pd.Series, list],
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Safely convert various data types to PyTorch tensors.

    Handles:
    - NaN/Inf values
    - numpy.object_ types (deprecated in numpy 1.20+)
    - Mixed data types
    - Memory-efficient conversion

    Args:
        data: Input data to convert
        dtype: Target PyTorch dtype
        device: Target device (None = CPU)

    Returns:
        PyTorch tensor with safe conversions applied

    Raises:
        ValueError: If data cannot be safely converted
    """
    # Handle None
    if data is None:
        raise ValueError("Cannot convert None to tensor")

    # Convert pandas to numpy
    if isinstance(data, (pd.DataFrame, pd.Series)):
        data = data.values

    # Convert to numpy array if list
    if isinstance(data, list):
        data = np.array(data)

    # Ensure it's a numpy array
    if not isinstance(data, np.ndarray):
        try:
            data = np.array(data)
        except Exception as e:
            raise ValueError(f"Cannot convert data to numpy array: {e}")

    # Handle object dtype (deprecated numpy.object_)
    if data.dtype == np.object_ or data.dtype == 'O':
        logger.warning("Detected object dtype - attempting to convert to float")
        try:
            data = data.astype(np.float64)
        except Exception as e:
            raise ValueError(f"Cannot convert object dtype to numeric: {e}")

    # Replace NaN and Inf with safe values
    if np.any(np.isnan(data)):
        nan_count = np.sum(np.isnan(data))
        logger.warning(f"Found {nan_count} NaN values - replacing with 0.0")
        data = np.nan_to_num(data, nan=0.0)

    if np.any(np.isinf(data)):
        inf_count = np.sum(np.isinf(data))
        logger.warning(f"Found {inf_count} Inf values - clipping to finite range")
        data = np.nan_to_num(data, posinf=1e10, neginf=-1e10)

    # Convert to tensor
    try:
        tensor = torch.tensor(data, dtype=dtype)
    except Exception as e:
        # Fallback: convert to float64 first
        logger.warning(f"Direct conversion failed, using float64 intermediate: {e}")
        data = data.astype(np.float64)
        tensor = torch.tensor(data, dtype=dtype)

    # Move to device if specified
    if device is not None:
        tensor = tensor.to(device)

    return tensor


def validate_tensor(
    tensor: torch.Tensor,
    name: str = "tensor",
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    allow_nan: bool = False,
    allow_inf: bool = False
) -> bool:
    """
    Validate tensor for common issues.

    Args:
        tensor: Tensor to validate
        name: Name for error messages
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        allow_nan: Whether to allow NaN values
        allow_inf: Whether to allow Inf values

    Returns:
        True if valid, raises ValueError if invalid
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{name} must be a PyTorch tensor, got {type(tensor)}")

    # Check for NaN
    if not allow_nan and torch.any(torch.isnan(tensor)):
        raise ValueError(f"{name} contains NaN values")

    # Check for Inf
    if not allow_inf and torch.any(torch.isinf(tensor)):
        raise ValueError(f"{name} contains Inf values")

    # Check range
    if min_val is not None and torch.any(tensor < min_val):
        raise ValueError(f"{name} contains values below minimum {min_val}")

    if max_val is not None and torch.any(tensor > max_val):
        raise ValueError(f"{name} contains values above maximum {max_val}")

    return True


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Get the best available device (GPU if available, else CPU).

    Args:
        prefer_gpu: Whether to prefer GPU if available

    Returns:
        torch.device object
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif prefer_gpu and torch.backends.mps.is_available():
        # Apple Silicon support
        device = torch.device('mps')
        logger.info("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")

    return device


def set_seed(seed: int = 369, deterministic: bool = True):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed (default: 369 - sacred geometry)
        deterministic: Whether to use deterministic algorithms (slower but reproducible)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f"Set random seed to {seed} (deterministic={deterministic})")


def batch_to_device(batch: Union[torch.Tensor, list, tuple, dict], device: torch.device):
    """
    Move batch to device, handling various data structures.

    Args:
        batch: Data to move
        device: Target device

    Returns:
        Data on target device
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, (list, tuple)):
        return type(batch)(batch_to_device(item, device) for item in batch)
    elif isinstance(batch, dict):
        return {key: batch_to_device(val, device) for key, val in batch.items()}
    else:
        return batch


class TensorScaler:
    """
    Normalize and denormalize tensors for better training.
    Stores normalization parameters for inverse transform.
    """

    def __init__(self, method: str = 'minmax'):
        """
        Args:
            method: 'minmax' or 'standard'
        """
        self.method = method
        self.min = None
        self.max = None
        self.mean = None
        self.std = None
        self.fitted = False

    def fit(self, data: torch.Tensor):
        """Compute normalization parameters."""
        if self.method == 'minmax':
            self.min = data.min(dim=0, keepdim=True)[0]
            self.max = data.max(dim=0, keepdim=True)[0]
            # Avoid division by zero
            self.max = torch.where(self.max == self.min, self.min + 1.0, self.max)
        elif self.method == 'standard':
            self.mean = data.mean(dim=0, keepdim=True)
            self.std = data.std(dim=0, keepdim=True)
            # Avoid division by zero
            self.std = torch.where(self.std == 0, torch.ones_like(self.std), self.std)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.fitted = True
        logger.info(f"Fitted {self.method} scaler")

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """Normalize data."""
        if not self.fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")

        if self.method == 'minmax':
            return (data - self.min) / (self.max - self.min)
        else:  # standard
            return (data - self.mean) / self.std

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Denormalize data."""
        if not self.fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")

        if self.method == 'minmax':
            return data * (self.max - self.min) + self.min
        else:  # standard
            return data * self.std + self.mean

    def fit_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Fit and transform in one step."""
        self.fit(data)
        return self.transform(data)
