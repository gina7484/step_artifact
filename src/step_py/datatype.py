from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from abc import ABC, abstractmethod

import sympy

from step_py.dyndim import DynDim


class ElementTP(ABC):
    @abstractmethod
    def size_in_bytes(self) -> sympy.Expr:
        """Return the size of this element type in bytes."""
        pass


class Float16(ElementTP):
    def __eq__(self, value):
        if isinstance(value, Float16):
            return True
        return False

    def size_in_bytes(self) -> sympy.Expr:
        """Return the size of Float16 in bytes."""
        return sympy.Integer(2)

    def __str__(self) -> str:
        return "Float16"


class Float32(ElementTP):
    def __eq__(self, value):
        if isinstance(value, Float32):
            return True
        return False

    def size_in_bytes(self) -> sympy.Expr:
        """Return the size of Float32 in bytes."""
        return sympy.Integer(4)

    def __str__(self) -> str:
        return "Float32"


class Uint32(ElementTP):
    def __eq__(self, value):
        if isinstance(value, Uint32):
            return True
        return False

    def size_in_bytes(self) -> sympy.Expr:
        """Return the size of Uint32 in bytes."""
        return sympy.Integer(4)

    def __str__(self) -> str:
        return "Uint32"


class Uint64(ElementTP):
    def __eq__(self, value):
        if isinstance(value, Uint64):
            return True
        return False

    def size_in_bytes(self) -> sympy.Expr:
        """Return the size of Uint64 in bytes."""
        return sympy.Integer(8)

    def __str__(self) -> str:
        return "Uint64"


class Bool(ElementTP):
    def __eq__(self, value):
        if isinstance(value, Bool):
            return True
        return False

    def size_in_bytes(self) -> sympy.Expr:
        """Return the size of Bool in bytes."""
        return sympy.Integer(1)

    def __str__(self) -> str:
        return "Bool"


@dataclass
class DynTile:
    tile_dtype: ElementTP
    shape: Tuple[Union[int, DynDim], Union[int, DynDim]]

    def size_in_bytes(self) -> sympy.Expr:
        """Return the total size of this tile in bytes."""
        total_elements = sympy.Integer(1)
        for dim in self.shape:
            if isinstance(dim, DynDim):
                total_elements = total_elements * dim.expr
            else:
                total_elements = total_elements * dim
        return total_elements * self.tile_dtype.size_in_bytes()

    def __str__(self) -> str:
        return f"DynTile({self.tile_dtype}, {self.shape})"


@dataclass
class Tile:
    tile_dtype: ElementTP
    shape: Tuple[int, int]

    def size_in_bytes(self) -> sympy.Expr:
        """Return the total size of this tile in bytes."""
        tile_size = sympy.Integer(self.shape[0] * self.shape[1])
        return tile_size * self.tile_dtype.size_in_bytes()

    def __str__(self) -> str:
        return f"Tile({self.tile_dtype}, {self.shape})"


@dataclass
class Buffer:
    buff_dtype: Tile
    shape: Tuple[Union[int, DynDim], ...]

    def __post_init__(self):
        assert isinstance(
            self.shape[0], (int, DynDim)
        ), "First dimension of buffer shape must be int or DynDim"
        # Check that only the first dimension can be DynDim, others must be int
        if len(self.shape) > 1:
            for i, dim in enumerate(self.shape[1:], 1):
                if not isinstance(dim, int):
                    raise ValueError(
                        f"Buffer dimension {i} must be int, got {type(dim)}"
                    )

    @property
    def rank(self) -> int:
        return len(self.shape)

    def size_in_bytes(self) -> sympy.Expr:
        """Return the total size of this buffer in bytes."""
        total_elements = sympy.Integer(1)
        for dim in self.shape:
            if isinstance(dim, DynDim):
                total_elements = total_elements * dim.expr
            else:
                total_elements = total_elements * dim
        return total_elements * self.buff_dtype.size_in_bytes()

    def __str__(self) -> str:
        return f"Buffer({self.buff_dtype}, {self.shape})"


class Select(ABC):
    @abstractmethod
    def size_in_bytes(self) -> sympy.Expr:
        """Return the size of this select type in bytes."""
        pass


@dataclass
class MultiHot(Select):
    total_n: int

    def size_in_bytes(self) -> sympy.Expr:
        """Return the size of MultiHot in bytes."""
        return sympy.Integer(self.total_n)


@dataclass
class Index(Select):
    active_n: int

    def size_in_bytes(self) -> sympy.Expr:
        """Return the size of Index in bytes."""
        return sympy.Integer(2 * self.active_n)


@dataclass
class Stream:
    stream_dtype: Union[Tile, Buffer, Select]
    shape: Tuple[Union[int, DynDim], ...]

    @property
    def rank(self) -> int:
        return len(self.shape) - 1

    def total_elements(self) -> sympy.Expr:
        """Return the total number of elements in this stream."""
        total_elements = sympy.Integer(1)
        for dim in self.shape:
            if isinstance(dim, DynDim):
                total_elements = total_elements * dim.expr
            else:
                total_elements = total_elements * dim
        return total_elements
