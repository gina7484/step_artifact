from dataclasses import dataclass
from typing import Optional
import sympy


@dataclass
class DynDim:
    """
    A wrapper class around the dynamic dimensions.
    This wrapper class is defined to decouple the dynamic dimension expression
    with the underlying class used (e.g., `torch.SymInt`, Sympy's `Expr`)

    Currently we are using `torch.SymInt` class to express the dynamic dimension's shape
    symbolically.
    """

    expr: sympy.Symbol

    def __add__(self, other):
        return DynDim(self.expr + other)

    def __mul__(self, other):
        return DynDim(self.expr * other)

    def __sub__(self, other):
        return DynDim(self.expr - other)

    def __floordiv__(self, other):
        return DynDim(self.expr // other)

    def __mod__(self, other):
        return DynDim(self.expr % other)

    def __iadd__(self, other):
        self.expr += other
        return self

    def __isub__(self, other):
        self.expr -= other
        return self

    def __imul__(self, other):
        self.expr *= other
        return self

    def __repr__(self):
        return str(self.expr)
