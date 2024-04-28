"""Module containing the main checks for pelage.

Use the syntax `import pelage as plg` rather than `from pelage import checks`
"""

from typing import Dict, Iterable, List, Optional, Tuple, Union

import polars as pl
from packaging import version
from polars.type_aliases import ClosedInterval, IntoExpr, PolarsDataType

from pelage import utils

PolarsColumnBounds = Union[
    Tuple[IntoExpr, IntoExpr], Tuple[IntoExpr, IntoExpr, ClosedInterval]
]

PolarsColumnType = Union[
    str, Iterable[str], PolarsDataType, Iterable[PolarsDataType], pl.Expr
]

PolarsOverClauseInput = Union[IntoExpr, Iterable[IntoExpr]]


class PolarsAssertError(Exception):
    def __init__(
        self, df: pl.DataFrame = pl.DataFrame(), supp_message: str = ""
    ) -> None:
        self.supp_message = supp_message
        self.df = df

    def __str__(self) -> str:
        base_message = "Error with the DataFrame passed to the check function:"

        if not self.df.is_empty():
            base_message = f"{self.df}\n{base_message}"

        return f"Details\n{base_message}\n-->{self.supp_message}"


def has_shape(data: pl.DataFrame, shape: Tuple[int, int]) -> pl.DataFrame:
    """Check if a DataFrame has the specified shape

    Parameters
    ----------
    data : pl.DataFrame
        Input data
    shape : Tuple[int, int]
        Tuple with the expected dataframe shape, as from the `.shape()` method

    Returns
    -------
    pl.DataFrame
        The original polars DataFrame when the check passes

    Examples
    --------
    >>> import polars as pl
    >>> import pelage as plg
    >>> df = pl.DataFrame({"a": [1, 2], "b": ["a", "b"]})
    >>> df.pipe(plg.has_shape, (2, 2))
    shape: (2, 2)
    ┌─────┬─────┐
    │ a   ┆ b   │
    │ --- ┆ --- │
    │ i64 ┆ str │
    ╞═════╪═════╡
    │ 1   ┆ a   │
    │ 2   ┆ b   │
    └─────┴─────┘
    >>> df.pipe(plg.has_shape, (1, 2))
    Traceback (most recent call last):
    ...
    pelage.checks.PolarsAssertError: Details
    Error with the DataFrame passed to the check function:
    -->The data has not the expected shape
    """
    if data.shape != shape:
        raise PolarsAssertError(supp_message="The data has not the expected shape")
    return data


def has_columns(data: pl.DataFrame, names: Union[str, List[str]]) -> pl.DataFrame:
    """Check if a DataFrame has the specified

    Parameters
    ----------
    data : pl.DataFrame
        The DataFrame to check for column presence.
    names : Union[str, List[str]]
        The names of the columns to check.

    Returns
    -------
    pl.DataFrame
        The original polars DataFrame when the check passes

    Examples
    --------
    >>> import polars as pl
    >>> import pelage as plg
    >>> df = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})
    >>> df.pipe(plg.has_columns, "b")
    shape: (3, 2)
    ┌─────┬─────┐
    │ a   ┆ b   │
    │ --- ┆ --- │
    │ i64 ┆ str │
    ╞═════╪═════╡
    │ 1   ┆ a   │
    │ 2   ┆ b   │
    │ 3   ┆ c   │
    └─────┴─────┘
    >>> df.pipe(plg.has_columns, "c")
    Traceback (most recent call last):
        ...
    pelage.checks.PolarsAssertError: Details
    Error with the DataFrame passed to the check function:
    -->
    >>> df.pipe(plg.has_columns, ["a", "b"])
    shape: (3, 2)
    ┌─────┬─────┐
    │ a   ┆ b   │
    │ --- ┆ --- │
    │ i64 ┆ str │
    ╞═════╪═════╡
    │ 1   ┆ a   │
    │ 2   ┆ b   │
    │ 3   ┆ c   │
    └─────┴─────┘
    >>>
    """  # noqa: E501
    if isinstance(names, str):
        # Because set(str) explodes the string
        names = [names]
    mising_columns = set(names) - set(data.columns)
    if mising_columns:
        raise PolarsAssertError
    return data


def has_dtypes(data: pl.DataFrame, items: Dict[str, PolarsDataType]) -> pl.DataFrame:
    """Check that the columns have the expected types

    Parameters
    ----------
    data : pl.DataFrame
        To check
    items : Dict[str, PolarsDataType]
        A dictionnary of column name with their expected polars data type :
        `{
            "col_a": pl.String,
            "col_b": pl.Int64,
            "col_c": pl.Float64,
            ...
        }`

    Returns
    -------
    pl.DataFrame
        The original polars DataFrame when the check passes


    Examples
    --------
    >>> import polars as pl
    >>> from pelage import checks
    >>> df = pl.DataFrame({
    ...     "name": ["Alice", "Bob", "Charlie"],
    ...     "age": [20, 30, 40],
    ...     "city": ["New York", "London", "Paris"],
    ... })
    >>> checks.has_dtypes(df, {
    ...     "name": pl.String,
    ...     "age": pl.Int64,
    ...     "city": pl.String,
    ... })
    shape: (3, 3)
    ┌─────────┬─────┬──────────┐
    │ name    ┆ age ┆ city     │
    │ ---     ┆ --- ┆ ---      │
    │ str     ┆ i64 ┆ str      │
    ╞═════════╪═════╪══════════╡
    │ Alice   ┆ 20  ┆ New York │
    │ Bob     ┆ 30  ┆ London   │
    │ Charlie ┆ 40  ┆ Paris    │
    └─────────┴─────┴──────────┘
    >>> checks.has_dtypes(df, {
    ...     "age": pl.String,
    ...     "city": pl.Int64,
    ... })
    Traceback (most recent call last):
        ...
    pelage.checks.PolarsAssertError: Details
    Error with the DataFrame passed to the check function:
    -->Some columns don't have the expected type:
    column='age', expected_type=String, real_dtype=Int64
    column='city', expected_type=Int64, real_dtype=String
    """
    missing_columns = set(items.keys()) - set(data.columns)
    if missing_columns:
        message = f"Dtype check, some expected columns are missing:{missing_columns}"
        raise PolarsAssertError(supp_message=message)

    bad_column_type_requirement = set(items.items()) - set(data.schema.items())
    if bad_column_type_requirement:
        message = utils.compare_schema(data.schema, items)
        message = f"Some columns don't have the expected type:\n{message}"
        raise PolarsAssertError(supp_message=message)
    return data


def has_no_nulls(
    data: pl.DataFrame,
    columns: Optional[PolarsColumnType] = None,
) -> pl.DataFrame:
    """Check if a DataFrame has any null (missing) values.

    Parameters
    ----------
    data : pl.DataFrame
        The input DataFrame to check for null values.
    columns : Optional[PolarsColumnType] , optional
        Columns to consider for null value check. By default, all columns are checked.

    Returns
    -------
    pl.DataFrame
        The original polars DataFrame when the check passes


    Examples
    --------
    >>> import polars as pl
    >>> from pelage import checks
    >>> df = pl.DataFrame({
    ...     "A": [1, 2],
    ...     "B": [None, 5]
    ... })
    >>> df
    shape: (2, 2)
    ┌─────┬──────┐
    │ A   ┆ B    │
    │ --- ┆ ---  │
    │ i64 ┆ i64  │
    ╞═════╪══════╡
    │ 1   ┆ null │
    │ 2   ┆ 5    │
    └─────┴──────┘
    >>> checks.has_no_nulls(df)
    Traceback (most recent call last):
        ...
    pelage.checks.PolarsAssertError: Details
    shape: (1, 2)
    ┌────────┬────────────┐
    │ column ┆ null_count │
    │ ---    ┆ ---        │
    │ str    ┆ u32        │
    ╞════════╪════════════╡
    │ B      ┆ 1          │
    └────────┴────────────┘
    Error with the DataFrame passed to the check function:
    -->There were unexpected nulls in the columns above
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
    columns: Optional[PolarsColumnType] = None,
) -> pl.Expr:
    """Ensure that input can be converted to a `pl.col()` expression"""
    if columns is None:
        return pl.all()
    elif isinstance(columns, pl.Expr):
        return columns
    else:
        return pl.col(columns)


def has_no_infs(
    data: pl.DataFrame,
    columns: Optional[PolarsColumnType] = None,
) -> pl.DataFrame:
    """Check if a DataFrame has any infinite (inf) values.

    Parameters
    ----------
    data : pl.DataFrame
        The input DataFrame to check for null values.
    columns : Optional[PolarsColumnType] , optional
        Columns to consider for null value check. By default, all columns are checked.

    Returns
    -------
    pl.DataFrame
        The original polars DataFrame when the check passes


    Examples
    --------
    >>> import polars as pl
    >>> import pelage as plg
    >>> df = pl.DataFrame(
    ...     {
    ...         "a": [1, 2],
    ...         "b": [1.0, float("inf")],
    ...     }
    ... )
    >>> plg.has_no_infs(df)
    Traceback (most recent call last):
      ...
    pelage.checks.PolarsAssertError: Details
    shape: (1, 2)
    ┌─────┬─────┐
    │ a   ┆ b   │
    │ --- ┆ --- │
    │ i64 ┆ f64 │
    ╞═════╪═════╡
    │ 2   ┆ inf │
    └─────┴─────┘
    Error with the DataFrame passed to the check function:
    -->
    >>> plg.has_no_infs(df, ["a"])  # or  plg.has_no_infs(df, "a")
    shape: (2, 2)
    ┌─────┬─────┐
    │ a   ┆ b   │
    │ --- ┆ --- │
    │ i64 ┆ f64 │
    ╞═════╪═════╡
    │ 1   ┆ 1.0 │
    │ 2   ┆ inf │
    └─────┴─────┘
    """
    selected_columns = _sanitize_column_inputs(columns)
    inf_values = data.filter(pl.any_horizontal(selected_columns.is_infinite()))
    if not inf_values.is_empty():
        raise PolarsAssertError(inf_values)
    return data


def unique(
    data: pl.DataFrame,
    columns: Optional[PolarsColumnType] = None,
) -> pl.DataFrame:
    """Check if there are no duplicated values in each one of the selected columns
    independently, i.e. it is a column oriented check.

    Parameters
    ----------
    data : pl.DataFrame
        The input DataFrame to check for unique values.
    columns : Optional[PolarsColumnType] , optional
        Columns to consider for uniqueness check. By default, all columns are checked.

    Returns
    -------
    pl.DataFrame
        The original polars DataFrame when the check passes


    Examples
    --------
    >>> import polars as pl
    >>> import pelage as plg
    >>> df = pl.DataFrame({"a": [1, 2]})
    >>> df.pipe(plg.unique, "a")  # Can also use ["a", ...], pl.col("a)
    shape: (2, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 1   │
    │ 2   │
    └─────┘
    >>> df = pl.DataFrame({"a": [1, 1, 2]})
    >>> df.pipe(plg.unique, "a")
    Traceback (most recent call last):
    ...
    pelage.checks.PolarsAssertError: Details
    shape: (2, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 1   │
    │ 1   │
    └─────┘
    Error with the DataFrame passed to the check function:
    -->Somes values are duplicated within the specified columns
    """
    selected_cols = _sanitize_column_inputs(columns)
    improper_data = data.filter(pl.any_horizontal(selected_cols.is_duplicated()))
    if not improper_data.is_empty():
        raise PolarsAssertError(
            df=improper_data,
            supp_message="Somes values are duplicated within the specified columns",
        )
    return data


def _non_unique_comibation(data: pl.DataFrame, columns: pl.Expr) -> pl.DataFrame:
    if version.parse(pl.__version__) < version.parse("0.20.0"):
        return (
            data.group_by(columns)
            .agg(pl.count().alias("len"))
            .filter(pl.col("len") > 1)
        )
    else:
        return data.group_by(columns).len().filter(pl.col("len") > 1)


def unique_combination_of_columns(
    data: pl.DataFrame,
    columns: Optional[PolarsColumnType] = None,
) -> pl.DataFrame:
    """Ensure that the selected column have a unique combination per row.
    This function is particularly helpful to establish the granularity of a dataframe,
    i.e. this is a row oriented check.

    Parameters
    ----------
    data : pl.DataFrame
        _description_
    columns : Optional[PolarsColumnType] , optional
        Columns to consider for row unicity. By default, all columns are checked.

    Returns
    -------
    pl.DataFrame
        The original polars DataFrame when the check passes


    Examples
    --------
    >>> import polars as pl
    >>> import pelage as plg
    >>> df = pl.DataFrame({"a": ["a", "a"], "b": [1, 2]})
    >>> df.pipe(plg.unique_combination_of_columns, ["a", "b"])
    shape: (2, 2)
    ┌─────┬─────┐
    │ a   ┆ b   │
    │ --- ┆ --- │
    │ str ┆ i64 │
    ╞═════╪═════╡
    │ a   ┆ 1   │
    │ a   ┆ 2   │
    └─────┴─────┘
    >>> bad = pl.DataFrame({"a": ["X", "X"]})
    >>> bad.pipe(plg.unique_combination_of_columns, "a")
    Traceback (most recent call last):
    ...
    pelage.checks.PolarsAssertError: Details
    shape: (1, 2)
    ┌─────┬─────┐
    │ a   ┆ len │
    │ --- ┆ --- │
    │ str ┆ u32 │
    ╞═════╪═════╡
    │ X   ┆ 2   │
    └─────┴─────┘
    Error with the DataFrame passed to the check function:
    -->Some combinations of columns are not unique. See above, selected: col("a")
    """
    cols = _sanitize_column_inputs(columns)
    non_unique_combinations = _non_unique_comibation(data, cols)
    if not non_unique_combinations.is_empty():
        raise PolarsAssertError(
            non_unique_combinations,
            f"Some combinations of columns are not unique. See above, selected: {cols}",
        )
    return data


def not_constant(
    data: pl.DataFrame,
    columns: Optional[PolarsColumnType] = None,
) -> pl.DataFrame:
    """Check if a DataFrame has constant columns.

    Parameters
    ----------
    data : pl.DataFrame
        The input DataFrame to check for null values.
    columns : Optional[PolarsColumnType] , optional
        Columns to consider for null value check. By default, all columns are checked.

    Returns
    -------
    pl.DataFrame
        The original polars DataFrame when the check passes


    Examples
    --------

    >>> import polars as pl
    >>> import pelage as plg
    >>> df = pl.DataFrame({"a": [1, 2]})
    >>> df.pipe(plg.not_constant, "a")
    shape: (2, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 1   │
    │ 2   │
    └─────┘
    >>> df = pl.DataFrame({"b": [1, 1]})
    >>> df.pipe(plg.not_constant)
    Traceback (most recent call last):
    ...
    pelage.checks.PolarsAssertError: Details
    shape: (1, 2)
    ┌────────┬────────────┐
    │ column ┆ n_distinct │
    │ ---    ┆ ---        │
    │ str    ┆ u32        │
    ╞════════╪════════════╡
    │ b      ┆ 1          │
    └────────┴────────────┘
    Error with the DataFrame passed to the check function:
    -->Some columns are constant
    """
    selected_cols = _sanitize_column_inputs(columns)
    constant_columns = (
        data.select(selected_cols.n_unique())
        .melt(variable_name="column", value_name="n_distinct")
        .filter(pl.col("n_distinct") == 1)
    )

    if not constant_columns.is_empty():
        raise PolarsAssertError(
            constant_columns,
            supp_message="Some columns are constant",
        )

    return data


def accepted_values(data: pl.DataFrame, items: Dict[str, List]) -> pl.DataFrame:
    """Raises error if columns contains values not specified in `items`

    Parameters
    ----------
    data : pl.DataFrame
    items : Dict[str, List]
        A dictionnary where keys are a string compatible with a pl.Expr, to be used with
        pl.col(). The value for each key is a List of all authorized values in the
        dataframe.

    Returns
    -------
    pl.DataFrame
        The original polars DataFrame when the check passes


    Examples
    --------

    >>> import polars as pl
    >>> import pelage as plg
    >>> items = {"a": [1, 2, 3], "b": ["a", "b", "c"]}
    >>> df = pl.DataFrame(items)
    >>> df.pipe(plg.accepted_values, {"a": [1, 2, 3]})
    shape: (3, 2)
    ┌─────┬─────┐
    │ a   ┆ b   │
    │ --- ┆ --- │
    │ i64 ┆ str │
    ╞═════╪═════╡
    │ 1   ┆ a   │
    │ 2   ┆ b   │
    │ 3   ┆ c   │
    └─────┴─────┘
    >>> df.pipe(plg.accepted_values, {"a": [1, 2]})
    Traceback (most recent call last):
    ...
    pelage.checks.PolarsAssertError: Details
    shape: (1, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 3   │
    └─────┘
    Error with the DataFrame passed to the check function:
    -->It contains values that have not been white-Listed in `items`.
    Showing problematic columns only.
    """
    mask_for_improper_values = [
        ~pl.col(col).is_in(values) for col, values in items.items()
    ]
    improper_data = data.filter(pl.Expr.or_(*mask_for_improper_values))

    if not improper_data.is_empty():
        bad_column_names = [
            col.name
            for col in improper_data.select(mask_for_improper_values)
            if col.any()
        ]
        raise PolarsAssertError(
            improper_data.select(bad_column_names),
            "It contains values that have not been white-Listed in `items`."
            + "\nShowing problematic columns only.",
        )
    return data


def not_accepted_values(data: pl.DataFrame, items: Dict[str, List]) -> pl.DataFrame:
    """Raises error if columns contains values specified in List of forbbiden `items`

    Parameters
    ----------
    data : pl.DataFrame
    items : Dict[str, List]
        A dictionnary where keys are a string compatible with a pl.Expr, to be used with
        pl.col(). The value for each key is a List of all forbidden values in the
        dataframe.

    Returns
    -------
    pl.DataFrame
        The original polars DataFrame when the check passes


    Examples
    --------

    >>> import polars as pl
    >>> import pelage as plg
    >>> df = pl.DataFrame(
    ...     {"a": [1, 2, 3], "b": ["a", "b", "c"]}
    ... )
    >>> df.pipe(plg.not_accepted_values, {"a": [4, 5]})
    shape: (3, 2)
    ┌─────┬─────┐
    │ a   ┆ b   │
    │ --- ┆ --- │
    │ i64 ┆ str │
    ╞═════╪═════╡
    │ 1   ┆ a   │
    │ 2   ┆ b   │
    │ 3   ┆ c   │
    └─────┴─────┘
    >>> df.pipe(plg.not_accepted_values, {"b": ["a", "b"]})
    Traceback (most recent call last):
    ...
    pelage.checks.PolarsAssertError: Details
    shape: (2, 1)
    ┌─────┐
    │ b   │
    │ --- │
    │ str │
    ╞═════╡
    │ a   │
    │ b   │
    └─────┘
    Error with the DataFrame passed to the check function:
    -->This DataFrame contains values marked as forbidden
    """
    mask_for_improper_values = [
        pl.col(col).is_in(values) for col, values in items.items()
    ]
    improper_data = data.filter(pl.Expr.or_(*mask_for_improper_values))
    if not improper_data.is_empty():
        bad_column_names = [
            col.name
            for col in improper_data.select(mask_for_improper_values)
            if col.any()
        ]
        raise PolarsAssertError(
            improper_data.select(bad_column_names),
            "This DataFrame contains values marked as forbidden",
        )
    return data


def has_mandatory_values(data: pl.DataFrame, items: Dict[str, list]) -> pl.DataFrame:
    """Ensure that all specified values are present in their respective column.

    Parameters
    ----------
    data : pl.DataFrame
        To check
    items : Dict[str, list]
        _description_

    Returns
    -------
    pl.DataFrame
        The original polars DataFrame when the check passes

    Examples
    --------
    >>> import polars as pl
    >>> import pelage as plg
    >>> df = pl.DataFrame({"a": [1, 2]})
    >>> df.pipe(plg.has_mandatory_values, {"a": [1, 2]})
    shape: (2, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 1   │
    │ 2   │
    └─────┘
    >>> df.pipe(plg.has_mandatory_values, {"a": [3, 4]})
    Traceback (most recent call last):
    ...
    pelage.checks.PolarsAssertError: Details
    Error with the DataFrame passed to the check function:
    -->Missing mandatory values in the following columns: {'a': [3, 4]}
    """
    selected_data = data.select(pl.col(items.keys())).unique()
    missing = {}
    for key in items:
        required_values = set(items[key])
        present_values = set(selected_data.get_column(key))
        should_be_present = required_values - present_values
        if should_be_present:
            missing[key] = sorted(should_be_present)

    if missing:
        raise PolarsAssertError(
            supp_message=f"Missing mandatory values in the following columns: {missing}"
        )
    return data


def not_null_proportion(
    data: pl.DataFrame, items: Dict[str, Union[float, Tuple[float, float]]]
) -> pl.DataFrame:
    """sserts that the proportion of non-null values present in a column is between
    a specified range [at_least, at_most] where at_most is an optional argument
    (default: 1.0).

    Parameters
    ----------
    data : pl.DataFrame
        _description_
    items : Dict[str, float  |  Tuple[float, float]]
        Limit ranges for the proportion of not null value for selected columns.
        Any of the following formats is valid:
        {
            "column_name_a" : 0.333,
            "column_name_b" : (0.25, 0.44),
        }
        When specifying a single float, the higher bound of the range will automatically
        be set to 1.0, i.e. (given_float, 1.0)

    Returns
    -------
    pl.DataFrame
        The original polars DataFrame when the check passes

    Examples
    --------

    >>> import polars as pl
    >>> import pelage as plg
    >>> df = pl.DataFrame(
    ...         {
    ...             "a": [1, None, None],
    ...             "b": [1, 2, None],
    ...         }
    ...     )
    >>> df.pipe(plg.not_null_proportion, {"a": 0.33, "b": 0.66})
    shape: (3, 2)
    ┌──────┬──────┐
    │ a    ┆ b    │
    │ ---  ┆ ---  │
    │ i64  ┆ i64  │
    ╞══════╪══════╡
    │ 1    ┆ 1    │
    │ null ┆ 2    │
    │ null ┆ null │
    └──────┴──────┘
    >>> df.pipe(plg.not_null_proportion, {"a": 0.7})
    Traceback (most recent call last):
    ...
    pelage.checks.PolarsAssertError: Details
    shape: (1, 4)
    ┌────────┬─────────────────────┬──────────┬──────────┐
    │ column ┆ not_null_proportion ┆ min_prop ┆ max_prop │
    │ ---    ┆ ---                 ┆ ---      ┆ ---      │
    │ str    ┆ f64                 ┆ f64      ┆ i64      │
    ╞════════╪═════════════════════╪══════════╪══════════╡
    │ a      ┆ 0.333333            ┆ 0.7      ┆ 1        │
    └────────┴─────────────────────┴──────────┴──────────┘
    Error with the DataFrame passed to the check function:
    -->Some columns contains a proportion of nulls beyond specified limits
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
    items: Dict[str, Union[float, Tuple[float, float]]],
) -> pl.DataFrame:
    ranges = {k: (v if isinstance(v, tuple) else (v, 1)) for k, v in items.items()}
    pl_ranges = pl.DataFrame(
        [(k, v[0], v[1]) for k, v in ranges.items()],
        schema=["column", "min_prop", "max_prop"],
    )
    return pl_ranges


def at_least_one(
    data: pl.DataFrame,
    columns: Optional[PolarsColumnType] = None,
) -> pl.DataFrame:
    """Ensure that there is at least one not null value in the designated columns.

    Parameters
    ----------
    data : pl.DataFrame
        To check
    columns : Optional[PolarsColumnType], optional
        Columns to consider to check the presence of at least one value.
        By default, all columns are checked.

    Returns
    -------
    pl.DataFrame
        The original polars DataFrame when the check passes

    Examples
    --------

    >>> import polars as pl
    >>> import pelage as plg
    >>> df = pl.DataFrame({"a": [None, None], "b": [1, None]})
    >>> df.pipe(plg.at_least_one, "b")
    shape: (2, 2)
    ┌──────┬──────┐
    │ a    ┆ b    │
    │ ---  ┆ ---  │
    │ null ┆ i64  │
    ╞══════╪══════╡
    │ null ┆ 1    │
    │ null ┆ null │
    └──────┴──────┘

    >>> df.pipe(plg.at_least_one)
    Traceback (most recent call last):
    ...
    pelage.checks.PolarsAssertError: Details
    Error with the DataFrame passed to the check function:
    -->Some columns contains only null values: ['a']
    """
    selected_columns = _sanitize_column_inputs(columns)

    are_column_nulls = data.select(selected_columns).null_count() == len(data)

    null_columns = [col.name for col in are_column_nulls if col.all()]

    if null_columns:
        raise PolarsAssertError(
            supp_message=f"Some columns contains only null values: {null_columns}"
        )
    return data


def accepted_range(
    data: pl.DataFrame, items: Dict[str, PolarsColumnBounds]
) -> pl.DataFrame:
    """Check that all the values from specifed columns in the dict `items` are within
        the indicated range.

    Parameters
    ----------
    data : pl.DataFrame
    items : Dict[str, PolarsColumnBounds]
        Any type of inputs that match the following signature:
        `column_name: (boundaries)` where `boundaries is compatible with the Polars
        method `is_between()` syntax.

        For example: {
            "col_a": (low, high),
            "col_b", (low_b, high_b, "right"),
            "col_c", (low_c, high_c, "none"),
            ...
            }

    Returns
    -------
    pl.DataFrame
        The original polars DataFrame when the check passes

    Examples
    --------
    >>> import polars as pl
    >>> import pelage as plg
    >>> df = pl.DataFrame({"a": [1, 2, 3]})
    >>> df.pipe(plg.accepted_range, {"a": (0, 2)})
    Traceback (most recent call last):
    ...
    pelage.checks.PolarsAssertError: Details
    shape: (1, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 3   │
    └─────┘
    Error with the DataFrame passed to the check function:
    -->Some values are beyond the acceptable ranges defined
    >>> df.pipe(plg.accepted_range, {"a": (1, 3)})
    shape: (3, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 1   │
    │ 2   │
    │ 3   │
    └─────┘
    >>> df = pl.DataFrame({"a": ["b", "c"]})
    >>> df.pipe(plg.accepted_range, {"a": (pl.lit("a"), pl.lit("d"), "right")})
    shape: (2, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ str │
    ╞═════╡
    │ b   │
    │ c   │
    └─────┘
    >>> df.pipe(plg.accepted_range, {"a": (pl.lit("a"), pl.lit("d"), "left")})
    shape: (2, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ str │
    ╞═════╡
    │ b   │
    │ c   │
    └─────┘
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

    Returns
    -------
    pl.DataFrame
        The original polars DataFrame when the check passes

    Examples
    --------

    >>> import polars as pl
    >>> import pelage as plg
    >>> initial_df = pl.DataFrame({"a": ["a", "b"]})
    >>> final_df = pl.DataFrame({"a": ["a", "b"]})
    >>> final_df.pipe(plg.maintains_relationships, initial_df, "a")
    shape: (2, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ str │
    ╞═════╡
    │ a   │
    │ b   │
    └─────┘
    >>> final_df = pl.DataFrame({"a": ["a"]})
    >>> final_df.pipe(plg.maintains_relationships, initial_df, "a")
    Traceback (most recent call last):
    ...
    pelage.checks.PolarsAssertError: Details
    Error with the DataFrame passed to the check function:
    -->Some values were removed from col 'a', for ex: ('b',)
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


def is_monotonic(
    data: pl.DataFrame,
    column: str,
    decreasing: bool = False,
    strict: bool = True,
    interval: Optional[Union[int, float, str]] = None,
    group_by: Optional[PolarsOverClauseInput] = None,
) -> pl.DataFrame:
    """Verify that values in a column are consecutively increasing or decreasing

    Parameters
    ----------
    data : pl.DataFrame
        To check
    column : str
        Name of the column that should be monotonic.
    decreasing : bool, optional
        Should the column be decreasing, by default False
    strict : bool, optional
        The series must be stricly increasing or decreasing, no consecutive equal values
        are allowed, by default True
    interval : Optional[Union[int, float, str, pl.Duration]], optional, by default None
        For time-based column, the interval can be specified as a string as in the
        function `dt.offset_by` or `pl.DataFrame().rolling`. It can also be specified
        with the `pl.duration()` function directly in a more explicit manner.

        When using a string, the interval is dictated by the following string language:
            - 1ns (1 nanosecond)
            - 1us (1 microsecond)
            - 1ms (1 millisecond)
            - 1s (1 second)
            - 1m (1 minute)
            - 1h (1 hour)
            - 1d (1 calendar day)
            - 1w (1 calendar week)
            - 1mo (1 calendar month)
            - 1q (1 calendar quarter)
            - 1y (1 calendar year)
            - 1i (1 index count)
        By "calendar day", we mean the corresponding time on the next day (which may
        not be 24 hours, due to daylight savings). Similarly for "calendar week",
        "calendar month", "calendar quarter", and "calendar year".,

    group_by : Optional[PolarsOverClauseInput], optional, by default None
        When specified, the monotonic characteristics and intervals are estimated for
        each group independently.

    Returns
    -------
    pl.DataFrame
        The original polars DataFrame when the check passes


    Examples
    --------

    >>> import polars as pl
    >>> import pelage as plg
    >>> df =     given = pl.DataFrame({"int": [1, 2, 1]})
    >>> df = pl.DataFrame({"int": [1, 2, 3], "str": ["x", "y", "z"]})
    >>> df.pipe(plg.is_monotonic, "int")
    shape: (3, 2)
    ┌─────┬─────┐
    │ int ┆ str │
    │ --- ┆ --- │
    │ i64 ┆ str │
    ╞═════╪═════╡
    │ 1   ┆ x   │
    │ 2   ┆ y   │
    │ 3   ┆ z   │
    └─────┴─────┘
    >>> bad = pl.DataFrame({"int": [1, 2, 1], "str": ["x", "y", "z"]})
    >>> bad.pipe(plg.is_monotonic, "int")
    Traceback (most recent call last):
    ...
    pelage.checks.PolarsAssertError: Details
    Error with the DataFrame passed to the check function:
    -->Column "int" expected to be monotonic but is not, try .sort("int")
    >>> given = pl.DataFrame(
    ...     [
    ...         ("2020-01-01 01:42:00", "A"),
    ...         ("2020-01-01 01:43:00", "A"),
    ...         ("2020-01-01 01:44:00", "A"),
    ...         ("2021-12-12 01:43:00", "B"),
    ...         ("2021-12-12 01:44:00", "B"),
    ...     ],
    ...     schema=["dates", "group"],
    ... ).with_columns(pl.col("dates").str.to_datetime())
    >>> given.pipe(plg.is_monotonic, "dates", interval="1m", group_by="group")
    shape: (5, 2)
    ┌─────────────────────┬───────┐
    │ dates               ┆ group │
    │ ---                 ┆ ---   │
    │ datetime[μs]        ┆ str   │
    ╞═════════════════════╪═══════╡
    │ 2020-01-01 01:42:00 ┆ A     │
    │ 2020-01-01 01:43:00 ┆ A     │
    │ 2020-01-01 01:44:00 ┆ A     │
    │ 2021-12-12 01:43:00 ┆ B     │
    │ 2021-12-12 01:44:00 ┆ B     │
    └─────────────────────┴───────┘
    >>> given.pipe(plg.is_monotonic, "dates", interval="3m", group_by="group")
    Traceback (most recent call last):
    ...
    pelage.checks.PolarsAssertError: Details
    Error with the DataFrame passed to the check function:
    -->Intervals differ from the specified 3m interval. Unexpected: {datetime.timedelta(seconds=60)}
    """  # noqa: E501
    select_diff_expression = (
        pl.col(column).diff()
        if group_by is None
        else pl.col(column).diff().over(group_by)
    )

    # Cast necessary for dates and datetimes
    diff_column = data.select(select_diff_expression).get_column(column)
    diff_column_sign = diff_column.cast(int)

    if not decreasing and not strict:
        comparisons = (diff_column_sign >= 0).all()
    if not decreasing and strict:
        comparisons = (diff_column_sign > 0).all()
    if decreasing and not strict:
        comparisons = (diff_column_sign <= 0).all()
    if decreasing and strict:
        comparisons = (diff_column_sign < 0).all()

    if not comparisons:
        error_msg = (
            f'Column "{column}" expected to be monotonic but is not,'
            + f' try .sort("{column}")'
        )
        raise PolarsAssertError(supp_message=error_msg)

    if interval is None:
        return data

    if diff_column.dtype == pl.Duration:
        assert isinstance(interval, str)
        dummy_time = pl.Series(["1970-01-01 00:00:00"]).str.to_datetime()
        expected_timedelta = dummy_time.dt.offset_by(interval) - dummy_time
        actual_timedelta = diff_column.drop_nulls().unique()
        bad_intervals = set(actual_timedelta) - set(expected_timedelta)

    else:
        bad_intervals = (diff_column != interval).any()

    if bad_intervals:
        raise PolarsAssertError(
            supp_message=f"Intervals differ from the specified {interval} interval."
            + f" Unexpected: {bad_intervals}"
        )
    return data


def custom_check(data: pl.DataFrame, expresion: pl.Expr) -> pl.DataFrame:
    """Use custom Polars expression to check the DataFrame, based on `.filter()`.
    The expression when used through the dataframe method `.filter()` should return an
    empty dataframe.
    This expression should express the requierement for values that are not wanted
    in the dataframe. For instance, if a column should not contain the value `4`,
    use the expression `pl.col("column") != 4`.

    Analog to DBT utils fonction: `expression_is_true`

    Parameters
    ----------
    data : pl.DataFrame
        To check
    expresion : pl.Expr
        Polar Expression that can be passed to the `.filter()` method. As describe
        above, use an expression that should keep forbidden values when passed to the
        filter

    Returns
    -------
    pl.DataFrame
        The original polars DataFrame when the check passes.

    Examples
    --------

    >>> import polars as pl
    >>> import pelage as plg
    >>> df = pl.DataFrame({"a": [1, 2, 3]})
    >>> df.pipe(plg.custom_check, pl.col("a") < 4)
    shape: (3, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 1   │
    │ 2   │
    │ 3   │
    └─────┘
    >>> df.pipe(plg.custom_check, pl.col("a") != 3)
    Traceback (most recent call last):
    ...
    pelage.checks.PolarsAssertError: Details
    shape: (1, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 3   │
    └─────┘
    Error with the DataFrame passed to the check function:
    -->Unexpected data in `Custom Check`: [(col("a")) != (3)]
    """
    bad_data = data.filter(expresion.not_())
    if not bad_data.is_empty():
        columns_in_expression = set(expresion.meta.root_names())
        raise PolarsAssertError(
            df=bad_data.select(columns_in_expression),
            supp_message=f"Unexpected data in `Custom Check`: {str(expresion)}",
        )
    return data


def mutually_exclusive_ranges(
    data: pl.DataFrame,
    low_bound: str,
    high_bound: str,
    partition_by: Optional[PolarsOverClauseInput] = None,
) -> pl.DataFrame:
    """Ensure that the specified columns contains no overlapping intervals.

    Parameters
    ----------
    data : pl.DataFrame
        Data to check
    low_bound : str
        Name of column containing the lower bound of the interval
    high_bound : str
        Name of column containing the higher bound of the interval
    partition_by : IntoExpr | Iterable[IntoExpr], optional
        Parameter compatible with `.over()` function to split the check by groups,
        by default None

    Returns
    -------
    pl.DataFrame
        The original polars DataFrame when the check passes

    Examples
    --------

    >>> import polars as pl
    >>> import pelage as plg
    >>> df = pl.DataFrame(
    ...     [
    ...         [1, 2],
    ...         [3, 4],
    ...     ],
    ...     schema=["a", "b"], orient="row"
    ... )
    >>> df.pipe(plg.mutually_exclusive_ranges, low_bound="a", high_bound="b")
    shape: (2, 2)
    ┌─────┬─────┐
    │ a   ┆ b   │
    │ --- ┆ --- │
    │ i64 ┆ i64 │
    ╞═════╪═════╡
    │ 1   ┆ 2   │
    │ 3   ┆ 4   │
    └─────┴─────┘
    >>> df = pl.DataFrame(
    ...     [
    ...         [1, 3],
    ...         [2, 4],
    ...         [5, 7],
    ...         [6, 8],
    ...         [9, 9],
    ...     ],
    ...     schema=["a", "b"],
    ...     orient="row",
    ... )
    >>> df.pipe(plg.mutually_exclusive_ranges, low_bound="a", high_bound="b")
    Traceback (most recent call last):
    ...
    pelage.checks.PolarsAssertError: Details
    shape: (4, 3)
    ┌───────┬─────┬─────┐
    │ index ┆ a   ┆ b   │
    │ ---   ┆ --- ┆ --- │
    │ u32   ┆ i64 ┆ i64 │
    ╞═══════╪═════╪═════╡
    │ 0     ┆ 1   ┆ 3   │
    │ 1     ┆ 2   ┆ 4   │
    │ 2     ┆ 5   ┆ 7   │
    │ 3     ┆ 6   ┆ 8   │
    └───────┴─────┴─────┘
    Error with the DataFrame passed to the check function:
    -->There were overlapping intervals:
    DataFrame was sorted by: ['a', 'b'],
    Interval columns: low_bound='a', high_bound='b'
    """
    is_overlapping_interval = pl.col(low_bound) <= pl.col(high_bound).shift()
    sorting_columns = [low_bound, high_bound]

    if partition_by is not None:
        is_overlapping_interval = is_overlapping_interval.over(partition_by)
        sorting_columns = [partition_by, low_bound, high_bound]

    indexes_of_overlaps = is_overlapping_interval.arg_true()

    overlapping_ranges = (
        data.sort(*sorting_columns)
        .pipe(_add_row_index)
        .filter(
            pl.col("index").is_in(indexes_of_overlaps)
            | pl.col("index").is_in(indexes_of_overlaps - 1)
        )
    )

    if len(overlapping_ranges) > 0:
        message = (
            "There were overlapping intervals:\n"
            + f"DataFrame was sorted by: {sorting_columns},\n"
            + f"Interval columns: {low_bound=}, {high_bound=}"
        )
        raise PolarsAssertError(
            df=overlapping_ranges,
            supp_message=message,
        )
    return data


def _add_row_index(data: pl.DataFrame) -> pl.DataFrame:
    if version.parse(pl.__version__) < version.parse("0.20.0"):
        return data.with_row_count().rename({"row_nr": "index"})
    else:
        return data.with_row_index()


def column_is_within_n_std(
    data: pl.DataFrame,
    items: Tuple[PolarsColumnType, int],
    *args: Tuple[PolarsColumnType, int],
) -> pl.DataFrame:
    """Function asserting values are within a given STD range, thus ensuring the absence
    of outliers.

    Parameters
    ----------
    data : pl.DataFrame
        To check.
    items : Tuple[PolarsColumnType, int]
        A column name / column type with the number of STD authorized for the values
        within. Must be of the following form: `(col_name, n_std)`

    Returns
    -------
    pl.DataFrame
        The original polars DataFrame when the check passes

    Examples
    --------

    >>> import polars as pl
    >>> import pelage as plg
    >>> df = pl.DataFrame(
    ...     {
    ...         "a": list(range(0, 11)),
    ...         "b": list(range(0, 11)),
    ...         "c": list(range(0, 10)) + [5000],
    ...     }
    ... )
    >>> df.pipe(plg.column_is_within_n_std, ("a", 2), ("b", 3))
    shape: (11, 3)
    ┌─────┬─────┬──────┐
    │ a   ┆ b   ┆ c    │
    │ --- ┆ --- ┆ ---  │
    │ i64 ┆ i64 ┆ i64  │
    ╞═════╪═════╪══════╡
    │ 0   ┆ 0   ┆ 0    │
    │ 1   ┆ 1   ┆ 1    │
    │ 2   ┆ 2   ┆ 2    │
    │ 3   ┆ 3   ┆ 3    │
    │ 4   ┆ 4   ┆ 4    │
    │ …   ┆ …   ┆ …    │
    │ 6   ┆ 6   ┆ 6    │
    │ 7   ┆ 7   ┆ 7    │
    │ 8   ┆ 8   ┆ 8    │
    │ 9   ┆ 9   ┆ 9    │
    │ 10  ┆ 10  ┆ 5000 │
    └─────┴─────┴──────┘
    >>> df.pipe(plg.column_is_within_n_std, ("b", 2), ("c", 2))
    Traceback (most recent call last):
    ...
    pelage.checks.PolarsAssertError: Details
    shape: (1, 1)
    ┌──────┐
    │ c    │
    │ ---  │
    │ i64  │
    ╞══════╡
    │ 5000 │
    └──────┘
    Error with the DataFrame passed to the check function:
    -->There are some outliers outside the specified mean±std range
    Impacted columns: ['c']
    """
    check_items = [items, *args]

    pairs_to_check = [
        (_sanitize_column_inputs(col), n_std) for col, n_std in check_items
    ]

    columns_not_within_stds = [
        col.is_between(
            col.mean() - n_std * col.std(),
            col.mean() + n_std * col.std(),
        ).not_()
        for col, n_std in pairs_to_check
    ]

    outliers = data.filter(pl.Expr.or_(*columns_not_within_stds))
    if len(outliers) > 0:
        bad_column_names = [
            col.name for col in data.select(columns_not_within_stds) if col.any()
        ]

        raise PolarsAssertError(
            df=outliers.select(bad_column_names),
            supp_message=(
                "There are some outliers outside the specified mean±std range"
                + "\n"
                + f"Impacted columns: {bad_column_names}"
            ),
        )

    return data
