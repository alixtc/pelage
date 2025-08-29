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
