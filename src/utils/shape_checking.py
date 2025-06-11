import torch
from typing import Tuple


def is_valid_view(tensor: torch.Tensor, new_shape: Tuple[int, ...]) -> bool:
    """Check if view is valid by attempting it"""
    try:
        tensor.view(*new_shape)
        return True
    except RuntimeError:
        return False


def test_is_valid_view():
    # Example
    tensor = torch.randn(2, 16, 64)
    print(is_valid_view(tensor, (32, 64)))  # True
    print(is_valid_view(tensor, (10, 10)))  # False
