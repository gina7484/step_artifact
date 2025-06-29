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

    def __init__(self, name_or_expr: Union[str, sympy.Symbol]):
        """
        Initialize a DynDim with either a name (string) or an existing sympy.Symbol.

        Args:
            name_or_expr: Either a string name for the symbol, or an existing sympy.Symbol
        """
        if isinstance(name_or_expr, str):
            # Create a new integer symbol with the given name
            self.expr = sympy.Symbol(name_or_expr, integer=True, positive=True)
        elif isinstance(name_or_expr, sympy.Symbol):
            # Use the existing symbol (assume it has proper assumptions)
            self.expr = name_or_expr
        else:
            raise TypeError(f"Expected str or sympy.Symbol, got {type(name_or_expr)}")

    def __add__(self, other: Union[int, "DynDim"]) -> "DynDim":
        if isinstance(other, DynDim):
            return DynDim(self.expr + other.expr).simplify()  # type: ignore
        return DynDim(self.expr + other).simplify()  # type: ignore

    def __mul__(self, other: Union[int, "DynDim"]) -> "DynDim":
        if isinstance(other, DynDim):
            return DynDim(self.expr * other.expr).simplify()  # type: ignore
        return DynDim(self.expr * other).simplify()  # type: ignore

    def __sub__(self, other: Union[int, "DynDim"]) -> "DynDim":
        if isinstance(other, DynDim):
            return DynDim(self.expr - other.expr).simplify()  # type: ignore
        return DynDim(self.expr - other).simplify()  # type: ignore

    def __floordiv__(self, other: Union[int, "DynDim"]) -> "DynDim":
        if isinstance(other, DynDim):
            return DynDim(self.expr // other.expr).simplify()  # type: ignore
        return DynDim(self.expr // other).simplify()  # type: ignore

    def __mod__(self, other: Union[int, "DynDim"]) -> "DynDim":
        if isinstance(other, DynDim):
            return DynDim(self.expr % other.expr).simplify()  # type: ignore
        return DynDim(self.expr % other).simplify()  # type: ignore

    def __iadd__(self, other: Union[int, "DynDim"]) -> "DynDim":
        if isinstance(other, DynDim):
            self.expr += other.expr  # type: ignore
            return self.simplify()
        self.expr += other  # type: ignore
        return self.simplify()

    def __isub__(self, other: Union[int, "DynDim"]) -> "DynDim":
        if isinstance(other, DynDim):
            self.expr -= other.expr  # type: ignore
            return self.simplify()
        self.expr -= other  # type: ignore
        return self.simplify()

    def __imul__(self, other: Union[int, "DynDim"]) -> "DynDim":
        if isinstance(other, DynDim):
            self.expr *= other.expr  # type: ignore
            return self.simplify()
        self.expr *= other  # type: ignore
        return self.simplify()

    def __ifloordiv__(self, other: Union[int, "DynDim"]) -> "DynDim":
        if isinstance(other, DynDim):
            self.expr //= other.expr  # type: ignore
            return self.simplify()
        self.expr //= other  # type: ignore
        return self.simplify()

    def __imod__(self, other: Union[int, "DynDim"]) -> "DynDim":
        if isinstance(other, DynDim):
            self.expr %= other.expr  # type: ignore
            return self.simplify()
        self.expr %= other  # type: ignore
        return self.simplify()

    def __repr__(self) -> str:
        return str(self.expr)

    def simplify(self) -> "DynDim":
        self.expr = sympy.simplify(self.expr)
        return self
