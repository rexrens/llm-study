"""
Utility modules for PyTorch tutorials.
"""

from .device import get_device, print_device_info
from .seed import get_seed, set_seed

__all__ = ["get_device", "print_device_info", "get_seed", "set_seed"]
