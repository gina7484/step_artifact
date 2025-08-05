from abc import ABC, abstractmethod
from typing import Tuple, Union
from step_py.datatype import DynTile, Tile, MultiHot, Index, Uint64


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

    def __str__(self) -> str:
        return self.__class__.__name__


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


class DynMatmul(MapFn):
    """
    A function that performs matrix multiplication.
    If `weight_transposed` is False, the tile shapes should be [M,K], [K,N]
    """

    # [Genghan] We need output dtype?
    weight_transposed: bool

    def __init__(self, weight_transposed: bool = False):
        super().__init__()
        self.weight_transposed = weight_transposed

    def apply(self, input_tp: Tuple) -> Union[Tile, DynTile]:
        if len(input_tp) != 2:
            raise ValueError("Matmul requires exactly two input types.")

        # Assuming input_tp[0] and input_tp[1] are both Tile types
        tile_a, tile_b = input_tp[0], input_tp[1]

        if not (
            isinstance(tile_a, (Tile, DynTile)) and isinstance(tile_b, (Tile, DynTile))
        ):
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

        if isinstance(result_shape[0], int) and isinstance(result_shape[1], int):
            return Tile(
                tile_dtype=tile_a.tile_dtype, shape=result_shape
            )  # Return the resulting Tile type
        else:
            return DynTile(
                tile_dtype=tile_a.tile_dtype, shape=result_shape
            )  # Return the resulting DynTile type


class Mul(MapFn):
    """
    A function that performs element-wise multiplication.
    """

    def __init__(self):
        super().__init__()

    def apply(self, input_tp: Tuple) -> Union[Tile, DynTile]:
        if len(input_tp) != 2:
            raise ValueError("Mul requires exactly two input types.")

        tile_a, tile_b = input_tp[0], input_tp[1]

        if not (
            isinstance(tile_a, (Tile, DynTile)) and isinstance(tile_b, (Tile, DynTile))
        ):
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
        output_shape = ()
        if tile_a_0 == tile_b_0 and tile_a_1 == tile_b_1:
            output_shape = tile_a.shape
        else:
            output_shape = (max(tile_a_0, tile_b_0), max(tile_a_1, tile_b_1))

        if isinstance(output_shape[0], int) and isinstance(output_shape[1], int):
            return Tile(tile_dtype=tile_a.tile_dtype, shape=output_shape)
        else:
            return DynTile(tile_dtype=tile_a.tile_dtype, shape=output_shape)


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


class Div(MapFn):
    """
    A function that performs element-wise division.
    """

    def __init__(self):
        super().__init__()

    def apply(self, input_tp: Tuple) -> Tile:
        if len(input_tp) != 2:
            raise ValueError("All requires exactly two input types.")

        tile_a, tile_b = input_tp[0], input_tp[1]

        if not (isinstance(tile_a, Tile) and isinstance(tile_b, Tile)):
            raise TypeError("Both inputs to Div must be of type Tile.")

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

    def apply(self, input_tp: Tuple) -> Union[Tile, DynTile]:
        if len(input_tp) != 1:
            raise ValueError("SiLU requires exactly one input type.")

        in_tile = input_tp[0]

        if isinstance(in_tile, Tile):
            return Tile(tile_dtype=in_tile.tile_dtype, shape=in_tile.shape)
        elif isinstance(in_tile, DynTile):
            return DynTile(tile_dtype=in_tile.tile_dtype, shape=in_tile.shape)
        else:
            raise TypeError("Input to SiLU must be of type Tile.")


class RowWiseSum(MapFn):
    """
    A function that performs reduction (add) between columns within a tile.
    This is used to do a row-wise reduction.
    It uses the initial value as the first accumulator.
    """

    def __init__(self):
        super().__init__()

    def apply(self, input_tp: Tuple) -> Union[Tile, DynTile]:
        if len(input_tp) != 1:
            raise ValueError("RowWiseReduction requires exactly one input type.")

        in_tile = input_tp[0]

        if isinstance(in_tile, Tile):
            return Tile(tile_dtype=in_tile.tile_dtype, shape=(in_tile.shape[0], 1))
        elif isinstance(in_tile, DynTile):
            return DynTile(tile_dtype=in_tile.tile_dtype, shape=(in_tile.shape[0], 1))
        else:
            raise TypeError("Input to RowWiseReduction must be of type Tile | DynTile.")


class Exp(MapFn):
    """
    A function that applies the exponential function.
    """

    def __init__(self):
        super().__init__()

    def apply(self, input_tp: Tuple) -> Union[Tile, DynTile]:
        if len(input_tp) != 1:
            raise ValueError("Exp requires exactly one input type.")

        in_tile = input_tp[0]

        if isinstance(in_tile, Tile):
            return Tile(tile_dtype=in_tile.tile_dtype, shape=in_tile.shape)
        elif isinstance(in_tile, DynTile):
            return DynTile(tile_dtype=in_tile.tile_dtype, shape=in_tile.shape)
        else:
            raise TypeError("Input to Exp must be of type Tile.")


class SetOffset(MapFn):
    """
    A function that sets the offset of a specific tile.
    - lhs: Tile
    - rhs: Offset ([1,1] tile)
    """

    def __init__(self):
        super().__init__()

    def apply(self, input_tp: Tuple) -> Union[Tile, DynTile]:
        if len(input_tp) != 2:
            raise ValueError("SetOffset requires exactly two input types.")

        in_tile, offset = input_tp[0], input_tp[1]

        if isinstance(in_tile, Tile):
            return Tile(tile_dtype=in_tile.tile_dtype, shape=in_tile.shape)
        elif isinstance(in_tile, DynTile):
            return DynTile(tile_dtype=in_tile.tile_dtype, shape=in_tile.shape)
        else:
            raise TypeError("Input to SetOffset must be of type Tile | DynTile.")


class RowWiseAppend(MapFn):
    """
    A function that appends a row to a tile.
    This does not change the shape of the tile, but increments the offset by 1 and
    writes the row in the new position.
    - lhs: Tile (N,D)
    - rhs: Tile to append (1,D)
    """

    def __init__(self):
        super().__init__()

    def apply(self, input_tp: Tuple) -> Union[Tile, DynTile]:
        if len(input_tp) != 2:
            raise ValueError("RowWiseAppend requires exactly two input types.")

        in_tile, new_row = input_tp[0], input_tp[1]

        if isinstance(in_tile, Tile):
            return Tile(tile_dtype=in_tile.tile_dtype, shape=in_tile.shape)
        elif isinstance(in_tile, DynTile):
            return DynTile(tile_dtype=in_tile.tile_dtype, shape=in_tile.shape)
        else:
            raise TypeError("Input to SetOffset must be of type Tile | DynTile.")


class CacheWriteAddrGen(MapFn):
    """
    A function that generates the address for the cache write.
    Can be decomposed into a Map doing a MAC operation

    """

    row_offset: int

    def __init__(self, row_offset: int):
        super().__init__()
        self.row_offset = row_offset

    def apply(self, input_tp: Tuple) -> Tile:
        if len(input_tp) != 2:
            raise ValueError("CacheWriteAddrGen requires exactly two input types.")

        in_tile, seq_len = input_tp[0], input_tp[1]

        assert (
            isinstance(in_tile, Tile)
            and in_tile.shape == (1, 1)
            and in_tile.tile_dtype == Uint64()
        )
        assert (
            isinstance(seq_len, Tile)
            and seq_len.shape == (1, 1)
            and seq_len.tile_dtype == Uint64()
        )

        return Tile(tile_dtype=Uint64(), shape=(1, 1))
