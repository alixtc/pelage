"""Utility functions for pelage."""

import polars as pl

from pelage.types import PolarsColumnType


def _has_sufficient_polars_version(version_number: str = "0.20.0") -> bool:
    required_version = tuple(map(int, (version_number.split("."))))
    polars_version = tuple(map(int, (pl.__version__.split("."))))
    return polars_version >= required_version


def _sanitize_column_inputs(
    columns: PolarsColumnType | None = None,
) -> pl.Expr:
    """Ensure that input can be converted to a `pl.col()` expression"""
    if columns is None:
        return pl.all()
    elif isinstance(columns, pl.Expr):
        return columns
    else:
        return pl.col(columns)
