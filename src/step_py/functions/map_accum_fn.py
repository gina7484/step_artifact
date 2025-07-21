from abc import ABC, abstractmethod
from typing import Tuple, Union
from step_py.datatype import DynTile, Tile, MultiHot, Index


class MapAccumFn(ABC):
    @abstractmethod
    def apply(self, input_tp: Tuple) -> Tile:
        pass

    def __str__(self) -> str:
        return self.__class__.__name__


class DynMatmul(MapAccumFn):
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


class Matmul(MapAccumFn):
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
