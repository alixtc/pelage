from typing import Iterable

import polars as pl


class PolarsCheckError(Exception):
    def __init__(self, df: pl.DataFrame = pl.DataFrame()) -> None:
        self.df = df

    def __str__(self) -> str:
        non_empty_df = self.df if not self.df.is_empty() else ""
        return f"There was an improper value in the passed DataFrame:\n{non_empty_df}"


def has_no_nulls(data: pl.DataFrame) -> pl.DataFrame:
    if data.null_count().sum_horizontal().item() > 0:
        raise PolarsCheckError
    return data


def unique(data: pl.DataFrame, columns: str | Iterable[str] | pl.Expr) -> pl.DataFrame:
    cols = pl.col(columns) if not isinstance(columns, pl.Expr) else columns
    improper_data = data.filter(pl.any_horizontal(cols.is_duplicated()))
    if not improper_data.is_empty():
        raise PolarsCheckError(improper_data)
    return data


def accepted_values(data: pl.DataFrame, items: dict[str, list]) -> pl.DataFrame:
    mask_for_improper_values = [
        ~pl.col(col).is_in(values) for col, values in items.items()
    ]
    improper_data = data.filter(pl.Expr.or_(*mask_for_improper_values))
    if not improper_data.is_empty():
        raise PolarsCheckError(improper_data)
    return data
