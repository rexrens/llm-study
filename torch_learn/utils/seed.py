"""
Reproducibility utilities for PyTorch.
Sets random seeds for reproducible experiments.
"""

import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
        deterministic: If True, enables deterministic algorithms (may impact performance)

    Note:
        Setting deterministic=True may impact performance and may not guarantee
        complete reproducibility across all operations due to CUDA non-determinism.

    Examples:
        >>> set_seed(42)  # Basic reproducibility
        >>> set_seed(42, deterministic=True)  # Max reproducibility (slower)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Use deterministic algorithms where possible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Allow non-deterministic algorithms for better performance
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_seed() -> int:
    """
    Get a random seed for the current run.

    Returns:
        int: A random seed value based on current time

    Examples:
        >>> seed = get_seed()
        >>> set_seed(seed)  # Use this seed for reproducibility
    """
    return random.randint(0, 2**32 - 1)


if __name__ == "__main__":
    # Test reproducibility
    seed = 123
    print(f"Setting seed: {seed}")

    set_seed(seed)

    # Generate some random values
    print(f"Random: {random.random()}")
    print(f"NumPy: {np.random.rand()}")
    print(f"PyTorch: {torch.rand(1).item()}")

    # Test that same seed produces same results
    print("\nTesting reproducibility:")
    for i in range(3):
        set_seed(seed)
        val = torch.rand(3).tolist()
        print(f"  Run {i+1}: {val}")
