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
    dtype: ElementTP
    shape: Tuple[int, int]


@dataclass
class Stream:
    dtype: Tile
    shape: Tuple[Union[int, DynDim], ...]

    @property
    def rank(self) -> int:
        return len(self.shape) - 1
