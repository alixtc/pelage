import polars as pl

from pelage.types import (
    PolarsAssertError,
    PolarsColumnType,
    PolarsLazyOrDataFrame,
    PolarsOverClauseInput,
)
from pelage.utils import _sanitize_column_inputs


def unique(
    data: PolarsLazyOrDataFrame,
    columns: PolarsColumnType | None = None,
    group_by: PolarsOverClauseInput | None = None,
) -> PolarsLazyOrDataFrame:
    """Check if there are no duplicated values in each one of the selected columns.

    This is a column oriented check, where each columns are search independently for
    duplicated values. For a row oriented check see `unique_combination_of_columns`

    Parameters
    ----------
    data : PolarsLazyOrDataFrame
        The input DataFrame to check for unique values.
    columns : Optional[PolarsColumnType] , optional
        Columns to consider for uniqueness check. By default, all columns are checked.
    group_by : Optional[PolarsOverClauseInput], optional
        Use this option to ensure uniqueness with data segmented by group.
        by default None

    Returns
    -------
    PolarsLazyOrDataFrame
        The original polars DataFrame or LazyFrame when the check passes


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
    pelage.types.PolarsAssertError: Details
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
    --> Somes values are duplicated within the specified columns

    Below are examples with `group_by` option:
    >>> df = pl.DataFrame(
    ...     [
    ...         [1, 1, 1],
    ...         [1, 1, 2],
    ...     ],
    ...     schema=["col1", "col2", "group"],
    ...     orient="row",
    ... )
    >>> df.pipe(plg.unique, ["col1", "col2"], group_by="group")
    shape: (2, 3)
    ┌──────┬──────┬───────┐
    │ col1 ┆ col2 ┆ group │
    │ ---  ┆ ---  ┆ ---   │
    │ i64  ┆ i64  ┆ i64   │
    ╞══════╪══════╪═══════╡
    │ 1    ┆ 1    ┆ 1     │
    │ 1    ┆ 1    ┆ 2     │
    └──────┴──────┴───────┘

    >>> df = pl.DataFrame(
    ...     [
    ...         [1, 1, 1],
    ...         [1, 1, 1],
    ...         [1, 1, 2],
    ...     ],
    ...     schema=["col1", "col2", "group"],
    ...     orient="row",
    ... )
    >>> df.pipe(plg.unique, ["col1", "col2"], group_by="group")
    Traceback (most recent call last):
    ...
    pelage.types.PolarsAssertError: Details
    shape: (2, 3)
    ┌──────┬──────┬───────┐
    │ col1 ┆ col2 ┆ group │
    │ ---  ┆ ---  ┆ ---   │
    │ i64  ┆ i64  ┆ i64   │
    ╞══════╪══════╪═══════╡
    │ 1    ┆ 1    ┆ 1     │
    │ 1    ┆ 1    ┆ 1     │
    └──────┴──────┴───────┘
    Error with the DataFrame passed to the check function:
    --> Somes values are duplicated within the specified columns
    """
    selected_cols = _sanitize_column_inputs(columns)
    highlight_columns_with_duplication = (
        selected_cols.is_duplicated()
        if group_by is None
        else selected_cols.is_duplicated().over(group_by)
    )
    improper_data = (
        data.lazy()
        .filter(pl.any_horizontal(highlight_columns_with_duplication))
        .collect()
    )

    if not improper_data.is_empty():
        raise PolarsAssertError(
            df=improper_data,
            supp_message="Somes values are duplicated within the specified columns",
        )
    return data
