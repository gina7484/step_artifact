from typing import Optional, Union
import sympy


class DynDim:
    """
    A wrapper class around the dynamic dimensions.
    This wrapper class is defined to decouple the dynamic dimension expression
    with the underlying class used (e.g., `torch.SymInt`, Sympy's `Expr`)

    Currently we are using `torch.SymInt` class to express the dynamic dimension's shape
    symbolically.
    """

    expr: sympy.Expr

    def __init__(self, name_or_expr: Union[str, sympy.Expr]):
        """
        Initialize a DynDim with either a name (string) or an existing sympy.Symbol.

        Args:
            name_or_expr: Either a string name for the symbol, or an existing sympy.Symbol
        """
        if isinstance(name_or_expr, str):
            # Create a new integer symbol with the given name
            self.expr = sympy.Symbol(name_or_expr, integer=True, nonnegative=True)
        elif isinstance(name_or_expr, (sympy.Expr)):
            # Use the existing symbol (assume it has proper assumptions)
            self.expr = name_or_expr
        else:
            raise TypeError(f"Expected str or sympy.Expr, got {type(name_or_expr)}")

    def __add__(self, other: Union[int, "DynDim"]) -> "DynDim":
        if isinstance(other, DynDim):
            return DynDim(self.expr + other.expr)  # type: ignore
        return DynDim(self.expr + other)  # type: ignore

    def __mul__(self, other: Union[int, "DynDim"]) -> "DynDim":
        if isinstance(other, DynDim):
            return DynDim(self.expr * other.expr)  # type: ignore
        return DynDim(self.expr * other)  # type: ignore

    def __sub__(self, other: Union[int, "DynDim"]) -> "DynDim":
        if isinstance(other, DynDim):
            return DynDim(self.expr - other.expr)  # type: ignore
        return DynDim(self.expr - other)  # type: ignore

    def __floordiv__(self, other: Union[int, "DynDim"]) -> "DynDim":
        if isinstance(other, DynDim):
            return DynDim(self.expr // other.expr)  # type: ignore
        return DynDim(self.expr // other)  # type: ignore

    def __truediv__(self, other: Union[int, "DynDim"]) -> "DynDim":
        if isinstance(other, DynDim):
            return DynDim(self.expr / other.expr)  # type: ignore
        return DynDim(self.expr / other)  # type: ignore

    def __mod__(self, other: Union[int, "DynDim"]) -> "DynDim":
        if isinstance(other, DynDim):
            return DynDim(self.expr % other.expr)  # type: ignore
        return DynDim(self.expr % other)  # type: ignore

    def __iadd__(self, other: Union[int, "DynDim"]) -> "DynDim":
        if isinstance(other, DynDim):
            self.expr += other.expr  # type: ignore
            return self
        self.expr += other  # type: ignore
        return self

    def __isub__(self, other: Union[int, "DynDim"]) -> "DynDim":
        if isinstance(other, DynDim):
            self.expr -= other.expr  # type: ignore
            return self
        self.expr -= other  # type: ignore
        return self

    def __imul__(self, other: Union[int, "DynDim"]) -> "DynDim":
        if isinstance(other, DynDim):
            self.expr *= other.expr  # type: ignore
            return self
        self.expr *= other  # type: ignore
        return self

    def __ifloordiv__(self, other: Union[int, "DynDim"]) -> "DynDim":
        if isinstance(other, DynDim):
            self.expr //= other.expr  # type: ignore
            return self
        self.expr //= other  # type: ignore
        return self

    def __imod__(self, other: Union[int, "DynDim"]) -> "DynDim":
        if isinstance(other, DynDim):
            self.expr %= other.expr  # type: ignore
            return self
        self.expr %= other  # type: ignore
        return self

    def __repr__(self) -> str:
        return str(self.expr)

    def __eq__(self, other):
        if not isinstance(other, DynDim):
            return NotImplemented
        return self.expr.equals(other.expr)
