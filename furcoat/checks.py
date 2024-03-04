from typing import Iterable, Optional, Union

import polars as pl
from polars.type_aliases import ClosedInterval, IntoExpr, PolarsDataType

PolarColumnType = Union[PolarsDataType, Iterable[PolarsDataType], pl.Expr]


class PolarsAssertError(Exception):
    def __init__(
        self, df: pl.DataFrame = pl.DataFrame(), supp_message: str = ""
    ) -> None:
        self.supp_message = supp_message
        self.df = df

    def __str__(self) -> str:
        base_message = "Error with the DataFrame passed to the check function:"

        if not self.df.is_empty():
            base_message = f"\n{self.df}\n{base_message}"

        return f"{base_message}\n-->{self.supp_message}"


def has_shape(data: pl.DataFrame, shape: tuple[int, int]) -> pl.DataFrame:
    """Check if a DataFrame has the specified shape"""
    if data.shape != shape:
        raise PolarsAssertError
    return data


def has_no_nulls(
    data: pl.DataFrame,
    columns: Optional[str | Iterable[str] | PolarColumnType] = None,
) -> pl.DataFrame:
    """Check if a DataFrame has any null (missing) values.

    Parameters
    ----------
    data : pl.DataFrame
        The input DataFrame to check for null values.
    columns : Optional[str | Iterable[str] | PolarColumnType], optional
        Columns to consider for null value check. By default, all columns are checked.

    Examples
    --------
    >>> import polars as pl
    >>> from furcoat import checks
    >>> df = pl.DataFrame({
    ...     "A": [1, 2],
    ...     "B": [None, 5]
    ... })
    >>> df
    shape: (4, 2)
    ┌─────┬─────┐
    │ A   ┆ B   │
    │ --- ┆ --- │
    │ i64 ┆ i64 │
    ╞═════╪═════╡
    │ 1   ┆     │
    ├─────┼─────┤
    │ 2   ┆ 5   │
    └─────┴─────┘
    >>> checks.has_no_nulls(df)
    PolarsAssertError: DataFrame contains null values in column(s):
    shape: (4, 2)
    ┌─────┬─────┐
    │ A   ┆ B   │
    │ --- ┆ --- │
    │ i64 ┆ i64 │
    ╞═════╪═════╡
    │ 1   ┆     │
    ├─────┼─────┤
    │ 2   ┆ 5   │
    └─────┴─────┘
    """
    selected_columns = _sanitize_column_inputs(columns)
    null_count = (
        data.select(selected_columns.null_count())
        .melt(variable_name="column", value_name="null_count")
        .filter(pl.col("null_count") > 0)
    )
    if not null_count.is_empty():
        raise PolarsAssertError(
            null_count, "There were unexpected nulls in the columns above"
        )
    return data


def _sanitize_column_inputs(
    columns: Optional[str | Iterable[str] | PolarColumnType] = None,
) -> pl.Expr:
    if columns is None:
        return pl.all()
    elif isinstance(columns, pl.Expr):
        return columns
    else:
        return pl.col(columns)


def has_no_infs(
    data: pl.DataFrame,
    columns: Optional[str | Iterable[str] | PolarColumnType] = None,
) -> pl.DataFrame:
    """Check if a DataFrame has any infinite (inf) values.

    Parameters
    ----------
    data : pl.DataFrame
        The input DataFrame to check for null values.
    columns : Optional[str | Iterable[str] | PolarColumnType], optional
        Columns to consider for null value check. By default, all columns are checked.
    """
    selected_columns = _sanitize_column_inputs(columns)
    inf_values = data.filter(pl.any_horizontal(selected_columns.is_infinite()))
    if not inf_values.is_empty():
        raise PolarsAssertError(inf_values)
    return data


def unique(
    data: pl.DataFrame,
    columns: Optional[str | Iterable[str] | PolarColumnType] = None,
) -> pl.DataFrame:
    """Check if a DataFrame columns have unique values.

    Parameters
    ----------
    data : pl.DataFrame
        The input DataFrame to check for null values.
    columns : Optional[str | Iterable[str] | PolarColumnType], optional
        Columns to consider for null value check. By default, all columns are checked.
    """
    selected_cols = _sanitize_column_inputs(columns)
    improper_data = data.filter(pl.any_horizontal(selected_cols.is_duplicated()))
    if not improper_data.is_empty():
        raise PolarsAssertError(improper_data)
    return data


def not_constant(
    data: pl.DataFrame,
    columns: Optional[str | Iterable[str] | PolarColumnType] = None,
) -> pl.DataFrame:
    """Check if a DataFrame has constant columns.

    Parameters
    ----------
    data : pl.DataFrame
        The input DataFrame to check for null values.
    columns : Optional[str | Iterable[str] | PolarColumnType], optional
        Columns to consider for null value check. By default, all columns are checked.
    """
    selected_cols = _sanitize_column_inputs(columns)
    constant_columns = (
        data.select(selected_cols.n_unique())
        .melt(variable_name="column", value_name="n_distinct")
        .filter(pl.col("n_distinct") == 1)
    )

    if not constant_columns.is_empty():
        raise PolarsAssertError(constant_columns)

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
        raise PolarsAssertError(
            improper_data,
            "It contains values that have not been white-listed in `items`",
        )
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
        raise PolarsAssertError(
            improper_data, "This DataFrame contains values marked as forbidden"
        )
    return data


def not_null_proportion(
    data: pl.DataFrame, items: dict[str, float | tuple[float, float]]
) -> pl.DataFrame:
    """sserts that the proportion of non-null values present in a column is between
    a specified range [at_least, at_most] where at_most is an optional argument
    (default: 1.0).

    Parameters
    ----------
    data : pl.DataFrame
        _description_
    items : dict[str, float  |  tuple[float, float]]
        Limit ranges for the proportion of not null value in the format:
            column_name : 0.333,
            column_name : (0.25, 0.44)
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
        raise PolarsAssertError(
            out_of_range_null_proportions,
            "Some columns contains a proportion of nulls beyond specified limits",
        )
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


def accepted_range(
    data: pl.DataFrame,
    items: dict[
        str, tuple[IntoExpr, IntoExpr] | tuple[IntoExpr, IntoExpr, ClosedInterval]
    ],
) -> pl.DataFrame:
    """Check that all the values from specifed columns in the dict `items` are within
    the indicated range.

    Parameters
    ----------
    data : pl.DataFrame
    items : dict[
            str, tuple[IntoExpr, IntoExpr]  |  tuple[IntoExpr, IntoExpr, ClosedInterval]
        ]
        Any type of inputs that match the following signature:
        `column_name: (boundaries)` where `boundaries is compatible with the Polars
        method `is_between()` syntax.

    """
    closed_boundaries = {
        k: (v if len(v) == 3 else (*v, "both")) for k, v in items.items()
    }
    forbidden_ranges = [
        pl.col(k).is_between(*v).not_() for k, v in closed_boundaries.items()
    ]
    out_of_range = data.filter(pl.Expr.or_(*forbidden_ranges))
    if not out_of_range.is_empty():
        raise PolarsAssertError(
            out_of_range, "Some values are beyond the acceptable ranges defined"
        )
    return data


def maintains_relationships(
    data: pl.DataFrame, other_df: pl.DataFrame, column: str
) -> pl.DataFrame:
    """Function to help ensuring that set of values in selected column remains  the
    same in both DataFrames. This helps to maintain referential integrity.

    Parameters
    ----------
    data : pl.DataFrame
        Dataframe after transformation
    other_df : pl.DataFrame
        Distant dataframe usually the one before transformation
    column : str
        Column to check for keys/ids
    """

    local_keys = set(data.get_column(column))
    other_keys = set(other_df.get_column(column))

    if local_keys != other_keys:
        if local_keys > other_keys:
            set_diff = sorted(list(local_keys - other_keys)[:5])
            msg = f"Some values were added to col '{column}', for ex: {*set_diff,}"
            raise PolarsAssertError(supp_message=msg)
        else:
            set_diff = sorted(list(other_keys - local_keys)[:5])
            msg = f"Some values were removed from col '{column}', for ex: {*set_diff,}"
            raise PolarsAssertError(supp_message=msg)

    return data
