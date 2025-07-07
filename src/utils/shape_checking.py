import torch
from typing import Tuple, Union

from step_py.dyndim import DynDim


def is_valid_view(
    tensor: torch.Tensor, new_shape: Tuple[Union[int, DynDim], ...]
) -> bool:
    """Check if view is valid by attempting it"""
    if any(isinstance(dim, DynDim) for dim in new_shape):
        print("DynDim in shape")
        return False

    try:
        tensor.view(*new_shape)  # type: ignore
        return True
    except RuntimeError:
        return False


def test_is_valid_view():
    # Example
    tensor = torch.randn(2, 16, 64)
    print(is_valid_view(tensor, (32, 64)))  # True
    print(is_valid_view(tensor, (10, 10)))  # False
