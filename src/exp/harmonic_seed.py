"""
Harmonic Seeding Module
========================

Deterministic seed generation based on sacred geometry numerology.

Sacred Numbers:
    - 3, 6, 9: Tesla's divine numbers
    - 27: 3³ cubic harmony
    - 33: Master number
    - 369: Ultimate harmonic seed

Important: This is an experimental deterministic seeding convention,
NOT a claim of biological causation. Used for reproducibility and ablation studies.

Author: C.C.R.O.P-PhenoHunt Team
Version: 1.0.0
"""

import hashlib
import logging
import numpy as np
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


# Sacred geometry constants
SACRED_NUMBERS = {
    'trinity': 3,
    'hexagon': 6,
    'completion': 9,
    'cube': 27,
    'master': 33,
    'divine': 369
}


def get_seed_from_numerology(
    params_dict: Dict[str, Any],
    base_seed: int = 369  # Default divine seed
) -> int:
    """
    Generate deterministic seed from parameter dictionary using harmonic numerology.

    Args:
        params_dict: Dictionary of experiment parameters
        base_seed: Base harmonic seed (default: 369)

    Returns:
        32-bit integer seed

    Example:
        >>> params = {'latent_dim': 27, 'epochs': 369, 'batch_size': 9}
        >>> seed = get_seed_from_numerology(params)
        >>> print(f"Harmonic seed: {seed}")

    Note:
        This is a deterministic convention for reproducibility,
        NOT a claim of biological causation.
    """
    logger.info(f"Generating harmonic seed from {len(params_dict)} parameters")

    # Sort parameters for determinism
    sorted_params = sorted(params_dict.items())

    # Create string representation
    param_str = '_'.join([f"{k}_{v}" for k, v in sorted_params])
    param_str = f"{base_seed}_{param_str}"

    # Hash to get seed
    hash_obj = hashlib.sha256(param_str.encode())
    hash_int = int(hash_obj.hexdigest(), 16)

    # Reduce to 32-bit seed
    seed = hash_int % (2**32)

    logger.info(f"Generated seed: {seed} from base {base_seed}")
    logger.debug(f"Parameter string: {param_str}")

    return seed


def set_harmonic_seed(
    seed: int,
    deterministic: bool = True
) -> None:
    """
    Set random seeds across all libraries using harmonic seed.

    Args:
        seed: Harmonic seed value
        deterministic: Whether to enforce full determinism

    Example:
        >>> set_harmonic_seed(369)  # Divine seed
        >>> # All subsequent random operations will be reproducible
    """
    logger.info(f"Setting harmonic seed: {seed}")

    # Set Python random seed
    import random
    random.seed(seed)

    # Set NumPy seed
    np.random.seed(seed % (2**32))

    # Set PyTorch seed if available
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.info("PyTorch deterministic mode enabled")

    except ImportError:
        logger.debug("PyTorch not available")

    # Set TensorFlow seed if available
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        logger.debug("TensorFlow seed set")
    except ImportError:
        logger.debug("TensorFlow not available")

    logger.info("Harmonic seed applied to all available libraries")


def get_harmonic_hyperparameters(
    task: str = 'vae',
    scale: str = 'standard'
) -> Dict[str, Any]:
    """
    Get harmonic hyperparameter presets aligned with sacred geometry.

    Args:
        task: Task type ('vae', 'classifier', 'regression')
        scale: Scale ('quick', 'standard', 'extended')

    Returns:
        Dictionary of hyperparameters

    Example:
        >>> params = get_harmonic_hyperparameters('vae', 'standard')
        >>> print(params['latent_dim'], params['epochs'])
        27 369
    """
    base_params = {
        'seed': SACRED_NUMBERS['divine'],
        'batch_size': SACRED_NUMBERS['trinity'],
    }

    if task == 'vae':
        task_params = {
            'latent_dim': SACRED_NUMBERS['cube'],  # 27
            'encoder_layers': [128, 64, 32],  # 3 layers
            'decoder_layers': [32, 64, 128],  # 3 layers
        }
    elif task == 'classifier':
        task_params = {
            'hidden_dims': [128, 64, 32],  # 3 layers
            'n_classes': SACRED_NUMBERS['hexagon']  # 6 effects
        }
    elif task == 'regression':
        task_params = {
            'hidden_dims': [64, 32, 16],  # 3 layers
        }
    else:
        task_params = {}

    if scale == 'quick':
        scale_params = {
            'epochs': SACRED_NUMBERS['cube'],  # 27
            'patience': SACRED_NUMBERS['trinity'],  # 3
        }
    elif scale == 'standard':
        scale_params = {
            'epochs': SACRED_NUMBERS['divine'],  # 369
            'patience': SACRED_NUMBERS['completion'],  # 9
        }
    elif scale == 'extended':
        scale_params = {
            'epochs': 999,  # 3 × 333
            'patience': SACRED_NUMBERS['cube'],  # 27
        }
    else:
        scale_params = {'epochs': 100, 'patience': 10}

    return {**base_params, **task_params, **scale_params}


def harmonic_learning_rate_schedule(
    base_lr: float = 1e-3,
    epochs: int = 369,
    cycle_epochs: Optional[list] = None
) -> list:
    """
    Create learning rate schedule with harmonic cycles.

    Sacred Geometry: Cycles at 3, 6, 9 intervals.

    Args:
        base_lr: Base learning rate
        epochs: Total epochs
        cycle_epochs: Custom cycle epochs (default: [3, 6, 9])

    Returns:
        List of learning rates per epoch

    Example:
        >>> lr_schedule = harmonic_learning_rate_schedule(1e-3, 27)
        >>> print(f"Epoch 0 LR: {lr_schedule[0]:.6f}")
    """
    if cycle_epochs is None:
        cycle_epochs = [3, 6, 9]

    lr_schedule = []

    for epoch in range(epochs):
        # Determine cycle position
        cycle_length = sum(cycle_epochs)
        position_in_cycle = epoch % cycle_length

        # Cosine annealing within cycle
        progress = position_in_cycle / cycle_length
        lr = base_lr * (0.5 * (1 + np.cos(np.pi * progress)))

        lr_schedule.append(lr)

    return lr_schedule


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("Harmonic Seeding Demo")
    print("=" * 60)

    # Get seed from parameters
    params = {
        'latent_dim': 27,
        'epochs': 369,
        'batch_size': 9,
        'learning_rate': 0.001
    }

    seed = get_seed_from_numerology(params, base_seed=369)
    print(f"\nGenerated seed: {seed}")

    # Set harmonic seed
    set_harmonic_seed(seed)
    print(f"Random sample after seeding: {np.random.rand()}")

    # Reset and verify reproducibility
    set_harmonic_seed(seed)
    print(f"Random sample (should match): {np.random.rand()}")

    # Get harmonic hyperparameters
    print("\nHarmonic VAE hyperparameters:")
    vae_params = get_harmonic_hyperparameters('vae', 'standard')
    for k, v in vae_params.items():
        print(f"  {k}: {v}")

    # Learning rate schedule
    print("\nHarmonic learning rate schedule (first 27 epochs):")
    lr_schedule = harmonic_learning_rate_schedule(1e-3, 27)
    for i in [0, 3, 6, 9, 12, 15, 18, 21, 24]:
        print(f"  Epoch {i:2d}: {lr_schedule[i]:.6f}")

    print("\n✓ Harmonic seeding demo complete")
    print("\nNote: This is a deterministic experimental convention,")
    print("not a claim of biological causation.")
