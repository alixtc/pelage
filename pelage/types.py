"""Module containing the type definitions for pelage."""

from typing import Iterable, Tuple, TypeVar, Union

import polars as pl

try:
    from polars._typing import ClosedInterval, IntoExpr, PolarsDataType

except ImportError:
    from polars.type_aliases import ClosedInterval, IntoExpr, PolarsDataType

# Use typevar to make sure that the input and return types are the same
PolarsLazyOrDataFrame = TypeVar("PolarsLazyOrDataFrame", pl.DataFrame, pl.LazyFrame)

PolarsColumnBounds = Union[
    Tuple[IntoExpr, IntoExpr], Tuple[IntoExpr, IntoExpr, ClosedInterval]
]

PolarsColumnType = Union[
    str, Iterable[str], PolarsDataType, Iterable[PolarsDataType], pl.Expr
]

IntOrNone = Union[int, None]

PolarsOverClauseInput = Union[IntoExpr, Iterable[IntoExpr]]


class PolarsAssertError(Exception):
    """Custom Error providing detailed information about the failed check.

    To investigate the last error in a jupyter notebook you can use:

    Examples
    --------
    >>> from pelage import PolarsAssertError # doctest: +SKIP
    >>> raise PolarsAssertError # doctest: +SKIP
    >>> import sys # doctest: +SKIP
    >>> error = sys.last_value # doctest: +SKIP
    >>> print(error) # prints the string representation # doctest: +SKIP
    >>> error.df # access the dataframe object # doctest: +SKIP

    Attributes
    ----------
    df : pl.DataFrame, optional,  by default pl.DataFrame()
        A subset of the original dataframe passed to the check function with a highlight
        on the values that caused the check to fail,
    supp_message : str, optional
        A human readable description of the check failure, and when available a possible
        way to solve the issue,
        by default ""
    """

    def __init__(self, df: pl.DataFrame | None = None, supp_message: str = "") -> None:
        self.supp_message = supp_message
        self.df = df if df is not None else pl.DataFrame()

    def __str__(self) -> str:
        base_message = "Error with the DataFrame passed to the check function:"

        if not self.df.is_empty():
            base_message = f"{self.df}\n{base_message}"

        return f"Details\n{base_message}\n--> {self.supp_message}"
