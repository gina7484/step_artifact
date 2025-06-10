from dataclasses import dataclass
from typing import List, Tuple, Union
from abc import ABC

from step_py.dyndim import DynDim


class ElementTP(ABC):
    pass


class Float16(ElementTP):
    pass


class Float32(ElementTP):
    pass


@dataclass
class Tile:
    tile_dtype: ElementTP
    shape: Tuple[int, int]


@dataclass
class Buffer:
    buff_dtype: Tile
    shape: Tuple[int, ...]

    @property
    def rank(self) -> int:
        return len(self.shape)


class Select(ABC):
    pass


class MultiHot(Select):
    pass


class Index(Select):
    pass


@dataclass
class Stream:
    stream_dtype: Union[Tile, Buffer, Select]
    shape: Tuple[Union[int, DynDim], ...]

    @property
    def rank(self) -> int:
        return len(self.shape) - 1
