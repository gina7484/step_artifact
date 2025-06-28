from dataclasses import dataclass
from typing import List, Tuple, Union
from abc import ABC, abstractmethod

from step_py.dyndim import DynDim


class ElementTP(ABC):
    @abstractmethod
    def size_in_bytes(self) -> int:
        """Return the size of this element type in bytes."""
        pass


class Float16(ElementTP):
    def __eq__(self, value):
        if isinstance(value, Float16):
            return True
        return False

    def size_in_bytes(self) -> int:
        """Return the size of Float16 in bytes."""
        return 2


class Float32(ElementTP):
    def __eq__(self, value):
        if isinstance(value, Float32):
            return True
        return False

    def size_in_bytes(self) -> int:
        """Return the size of Float32 in bytes."""
        return 4


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
