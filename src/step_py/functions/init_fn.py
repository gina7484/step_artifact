from abc import ABC, abstractmethod
from typing import Tuple, Union
from dataclasses import dataclass
from step_py.datatype import ElementTP, Tile, MultiHot, Index, Buffer


@dataclass
class InitFn(ABC):
    """
    The parent class for functions that will be used in higher-order function operators.

    The apply function specifies the input type and the output type of the function.
    The functional behavior is identified through its name and additional arguments.
    """

    shape: Tuple[int, int]
    dtype: ElementTP

    @abstractmethod
    def apply(self) -> Tile:
        pass

    def __str__(self) -> str:
        return self.__class__.__name__


@dataclass
class Zero(InitFn):
    """
    A function that initializes a Tile with zeros.
    """

    def apply(self) -> Tile:
        return Tile(shape=self.shape, tile_dtype=self.dtype)


@dataclass
class Empty(InitFn):
    """
    A function that initializes a Tile with zeros.
    """

    def apply(self) -> Tile:
        return Tile(shape=self.shape, tile_dtype=self.dtype)
