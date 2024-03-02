from typing import Iterable

import polars as pl
from polars.type_aliases import IntoExpr


class PolarsCheckError(Exception):
    pass


def has_no_nulls(data: pl.DataFrame) -> pl.DataFrame:
    if data.null_count().sum_horizontal().item() > 0:
        raise PolarsCheckError
    return data


def unique(data: pl.DataFrame, columns: IntoExpr | Iterable[IntoExpr]) -> pl.DataFrame:
    if data.select(columns).is_duplicated().any():
        raise PolarsCheckError
    return data
