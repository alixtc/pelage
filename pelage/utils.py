"""Utility functions for pelage."""

import polars as pl

from pelage.types import PolarsColumnType


def compare_schema(data_schema: dict, expected_schema: dict) -> str:
    """Returns a list of mismatched dtypes: (column, expected_type, actual_type)"""
    unmatched_colum_dtypes = [
        (key, value, data_schema[key])
        for key, value in expected_schema.items()
        if value != data_schema[key]
    ]
    messages = [
        f"{column=}, {expected_type=}, {real_dtype=}"
        for (column, expected_type, real_dtype) in unmatched_colum_dtypes
    ]

    return "\n".join(messages)


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
