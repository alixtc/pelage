from typing import Iterable

import polars as pl


class PolarsCheckError(Exception):
    def __init__(self, df: pl.DataFrame = pl.DataFrame()) -> None:
        self.df = df

    def __str__(self) -> str:
        non_empty_df = self.df if not self.df.is_empty() else ""
        return f"There was an improper value in the passed DataFrame:\n{non_empty_df}"


def has_shape(data: pl.DataFrame, shape: tuple[int, int]) -> pl.DataFrame:
    if data.shape != shape:
        raise PolarsCheckError
    return data


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
    """Raises error if columns contains values not specified in `items`

    Parameters
    ----------
    data : pl.DataFrame
    items : dict[str, list]
        A dictionnary where keys are a string compatible with a pl.Expr, to be used with
        pl.col(). The value for each key is a list of all authorized values in the
        dataframe.
    """
    mask_for_improper_values = [
        ~pl.col(col).is_in(values) for col, values in items.items()
    ]
    improper_data = data.filter(pl.Expr.or_(*mask_for_improper_values))
    if not improper_data.is_empty():
        raise PolarsCheckError(improper_data)
    return data


def not_accepted_values(data: pl.DataFrame, items: dict[str, list]) -> pl.DataFrame:
    """Raises error if columns contains values specified in list of forbbiden `items`

    Parameters
    ----------
    data : pl.DataFrame
    items : dict[str, list]
        A dictionnary where keys are a string compatible with a pl.Expr, to be used with
        pl.col(). The value for each key is a list of all forbidden values in the
        dataframe.
    """
    mask_for_improper_values = [
        pl.col(col).is_in(values) for col, values in items.items()
    ]
    improper_data = data.filter(pl.Expr.or_(*mask_for_improper_values))
    if not improper_data.is_empty():
        raise PolarsCheckError(improper_data)
    return data


def not_null_proportion(
    data: pl.DataFrame, items: dict[str, float | tuple[float, float]]
) -> pl.DataFrame:
    """Asserts that the proportion of non-null values present in a column is between
    a specified range [at_least, at_most] where at_most is an optional argument
    (default: 1.0).

    Returns
    -------
    _type_
        _description_
    """

    pl_ranges = _format_ranges_by_columns(items)

    out_of_range_null_proportions = (
        (data.null_count() / len(data))
        .melt(variable_name="column", value_name="null_proportion")
        .with_columns(not_null_proportion=1 - pl.col("null_proportion"))
        .join(pl_ranges, on="column", how="inner")
        .filter(
            pl.col("not_null_proportion")
            .is_between(
                pl.col("min_prop"),
                pl.col("max_prop"),
            )
            .not_()
        )
        .drop("null_proportion")
    )
    if not out_of_range_null_proportions.is_empty():
        raise PolarsCheckError(out_of_range_null_proportions)
    return data


def _format_ranges_by_columns(
    items: dict[str, float | tuple[float, float]]
) -> pl.DataFrame:
    ranges = {k: (v if isinstance(v, tuple) else (v, 1)) for k, v in items.items()}
    pl_ranges = pl.DataFrame(
        [(k, v[0], v[1]) for k, v in ranges.items()],
        schema=["column", "min_prop", "max_prop"],
    )
    return pl_ranges
