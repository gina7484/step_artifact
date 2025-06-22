from abc import ABC, abstractmethod
from typing import Tuple
from step_py.datatype import Tile, MultiHot, Index


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
    If `weight_transposed` is False, the tile shapes should be [M,K], [K,N]
    """

    # [Genghan] We need output dtype?
    weight_transposed: bool

    def __init__(self, weight_transposed: bool = False):
        super().__init__()
        self.weight_transposed = weight_transposed

    def apply(self, input_tp: Tuple) -> Tile:
        if len(input_tp) != 2:
            raise ValueError("Matmul requires exactly two input types.")

        # Assuming input_tp[0] and input_tp[1] are both Tile types
        tile_a, tile_b = input_tp[0], input_tp[1]

        if not (isinstance(tile_a, Tile) and isinstance(tile_b, Tile)):
            raise TypeError("Both inputs to Matmul must be of type Tile.")

        if not self.weight_transposed:  # [M,K] x [K,N]
            if tile_a.shape[1] != tile_b.shape[0]:
                raise ValueError("Incompatible shapes for matrix multiplication.")

            # The resulting shape will be (tile_a.shape[:-1], tile_b.shape[1])
            result_shape = (
                tile_a.shape[0],
                tile_b.shape[1],
            )
        else:  # [M,K] x [N,K]
            if tile_a.shape[1] != tile_b.shape[1]:
                raise ValueError("Incompatible shapes for matrix multiplication.")

            # The resulting shape will be (tile_a.shape[:-1], tile_b.shape[-1])
            result_shape = (
                tile_a.shape[0],
                tile_b.shape[0],
            )

        return Tile(
            tile_dtype=tile_a.tile_dtype, shape=result_shape
        )  # Return the resulting Tile type


class Mul(MapFn):
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


class Add(MapFn):
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


class Silu(MapFn):
    """
    A function that applies the SiLU activation function.
    """

    def __init__(self):
        super().__init__()

    def apply(self, input_tp: Tuple) -> Tile:
        if len(input_tp) != 1:
            raise ValueError("SiLU requires exactly one input type.")

        in_tile = input_tp[0]

        if not isinstance(in_tile, Tile):
            raise TypeError("Input to SiLU must be of type Tile.")

        return Tile(
            tile_dtype=in_tile.tile_dtype, shape=in_tile.shape
        )  # SiLU does not change the shape

class RetileRow(MapFn):
    def __init__(self):
        super().__init__()

    def apply(self, input_tp: Tuple) -> Tile:
        in_tile, accum_tile = input_tp[0], input_tp[1]

        if not (isinstance(in_tile, Tile) and isinstance(accum_tile, Tile)):
            raise TypeError("Both inputs must be of type Tile.")
        assert in_tile.shape[1] == accum_tile.shape[1]
        return Tile(
            tile_dtype=in_tile.tile_dtype,
            shape=(in_tile.shape[0] + accum_tile.shape[0], accum_tile.shape[1]),
        )