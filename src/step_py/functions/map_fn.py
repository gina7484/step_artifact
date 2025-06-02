from abc import ABC, abstractmethod
from typing import Tuple
from step_py.datatype import Tile


class MapFn(ABC):
    """
    The parent class for functions that will be used in higher-order function operators
    such as Map.

    The apply function specifies the input type and the output type of the function.
    The functional behavior is identified through its name and additional arguments.
    """

    @abstractmethod
    def apply(self, input_tp: Tuple) -> Tile:
        pass


class Matmul(MapFn):
    """
    A function that performs matrix multiplication.
    """

    def apply(self, input_tp: Tuple) -> Tile:
        if len(input_tp) != 2:
            raise ValueError("Matmul requires exactly two input types.")

        # Assuming input_tp[0] and input_tp[1] are both Tile types
        tile_a, tile_b = input_tp[0], input_tp[1]

        if not (isinstance(tile_a, Tile) and isinstance(tile_b, Tile)):
            raise TypeError("Both inputs to Matmul must be of type Tile.")

        if tile_a.shape[-1] != tile_b.shape[-2]:
            raise ValueError("Incompatible shapes for matrix multiplication.")

        # The resulting shape will be (tile_a.shape[:-1], tile_b.shape[-1])
        result_shape = tile_a.shape[:-1] + (tile_b.shape[-1],)

        return Tile(
            dtype=tile_a.dtype, shape=result_shape
        )  # Return the resulting Tile type
