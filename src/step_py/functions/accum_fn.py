from abc import ABC, abstractmethod
from typing import Tuple
from step_py.datatype import DynTile, Tile, MultiHot, Index, Uint64


class AccumFn(ABC):
    """
    The parent class for functions that will be used in higher-order function operators
    such as Accum.

    The apply function specifies the input type and the output type of the function.
    The functional behavior is identified through its name and additional arguments.
    """

    @abstractmethod
    def apply(self, input_tp: Tuple) -> Tile:
        pass

    def __str__(self) -> str:
        return self.__class__.__name__


class Mul(AccumFn):
    """
    A function that performs element-wise multiplication.
    """

    def __init__(self):
        super().__init__()

    def apply(self, input_tp: Tuple) -> Tile:
        if len(input_tp) != 2:
            raise ValueError("Mul requires exactly two input types.")

        tile_a, tile_b = input_tp[0], input_tp[1]

        if not (isinstance(tile_a, Tile) and isinstance(tile_b, Tile)):
            raise TypeError("Both inputs to Mul must be of type Tile.")

        # Check if the shapes are broadcastable for element-wise multiplication
        tile_a_0, tile_a_1 = tile_a.shape
        tile_b_0, tile_b_1 = tile_b.shape

        # Check broadcastability for dimension 0
        if not ((tile_a_0 == tile_b_0) or (tile_a_0 == 1) or (tile_b_0 == 1)):
            raise ValueError(
                f"Shapes are not broadcastable: {tile_a.shape} and {tile_b.shape}"
            )

        # Check broadcastability for dimension 1
        if not ((tile_a_1 == tile_b_1) or (tile_a_1 == 1) or (tile_b_1 == 1)):
            raise ValueError(
                f"Shapes are not broadcastable: {tile_a.shape} and {tile_b.shape}"
            )

        # Calculate the output shape according to broadcast rules
        output_shape = (max(tile_a_0, tile_b_0), max(tile_a_1, tile_b_1))

        return Tile(tile_dtype=tile_a.tile_dtype, shape=output_shape)


class Add(AccumFn):
    """
    A function that performs element-wise addition.
    """

    def __init__(self):
        super().__init__()

    def apply(self, input_tp: Tuple) -> Tile:
        if len(input_tp) != 2:
            raise ValueError("All requires exactly two input types.")

        tile_a, tile_b = input_tp[0], input_tp[1]

        if not (isinstance(tile_a, Tile) and isinstance(tile_b, Tile)):
            raise TypeError("Both inputs to Add must be of type Tile.")

        # Check if the shapes are broadcastable for element-wise multiplication
        tile_a_0, tile_a_1 = tile_a.shape
        tile_b_0, tile_b_1 = tile_b.shape

        # Check broadcastability for dimension 0
        if not ((tile_a_0 == tile_b_0) or (tile_a_0 == 1) or (tile_b_0 == 1)):
            raise ValueError(
                f"Shapes are not broadcastable: {tile_a.shape} and {tile_b.shape}"
            )

        # Check broadcastability for dimension 1
        if not ((tile_a_1 == tile_b_1) or (tile_a_1 == 1) or (tile_b_1 == 1)):
            raise ValueError(
                f"Shapes are not broadcastable: {tile_a.shape} and {tile_b.shape}"
            )

        # Calculate the output shape according to broadcast rules
        output_shape = (max(tile_a_0, tile_b_0), max(tile_a_1, tile_b_1))

        return Tile(tile_dtype=tile_a.tile_dtype, shape=output_shape)


class RetileRow(AccumFn):
    def __init__(self):
        super().__init__()

    def apply(self, input_tp: Tuple) -> Tile:
        in_tile, accum_tile = input_tp[0], input_tp[1]

        if not (isinstance(in_tile, Tile) and isinstance(accum_tile, (Tile, DynTile))):
            raise TypeError("Both inputs must be of type Tile.")
        assert in_tile.shape[1] == accum_tile.shape[1]
        return accum_tile


class SignalReqAllRead(AccumFn):
    def __init__(self):
        super().__init__()

    def apply(self, input_tp: Tuple) -> Tile:
        assert len(input_tp) == 1

        return Tile(tile_dtype=Uint64(), shape=(1, 1))
