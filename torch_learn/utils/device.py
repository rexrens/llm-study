"""
Device selection utilities for PyTorch.
Provides robust device selection across CPU, CUDA, and MPS (Apple Silicon).
"""

import torch


def get_device(device: str | None = None) -> torch.device:
    """
    Get the appropriate torch device.

    Args:
        device: Optional device string ("cpu", "cuda", "mps", "auto").
                If None or "auto", automatically selects the best available device.

    Returns:
        torch.device: The selected device

    Examples:
        >>> device = get_device()
        >>> device = get_device("cpu")
        >>> device = get_device("cuda:0")
    """
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device)


def print_device_info(device: torch.device | None = None) -> None:
    """
    Print detailed device information.

    Args:
        device: The device to check. If None, uses the auto-selected device.
    """
    device = device or get_device()

    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"  CUDA available: {torch.cuda.is_available()}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")
        print(f"  Current GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    elif device.type == "mps":
        print(f"  MPS available: {torch.backends.mps.is_available()}")
        print(f"  MPS built: {torch.backends.mps.is_built()}")


if __name__ == "__main__":
    # Test device selection
    device = get_device()
    print_device_info(device)

    # Test specific device
    print("\nSpecific device:")
    print(f"CPU device: {get_device('cpu')}")
