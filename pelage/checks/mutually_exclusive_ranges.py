from typing import Optional

import polars as pl

from pelage.types import (
    PolarsAssertError,
    PolarsLazyOrDataFrame,
    PolarsOverClauseInput,
)
from pelage.utils import _has_sufficient_polars_version


def _add_row_index(data: PolarsLazyOrDataFrame) -> PolarsLazyOrDataFrame:
    if _has_sufficient_polars_version():
        return data.with_row_index()
    else:
        return data.with_row_count().rename({"row_nr": "index"})


def mutually_exclusive_ranges(
    data: PolarsLazyOrDataFrame,
    low_bound: str,
    high_bound: str,
    group_by: Optional[PolarsOverClauseInput] = None,
) -> PolarsLazyOrDataFrame:
    """Ensure that the specified columns contains no overlapping intervals.

    Parameters
    ----------
    data : PolarsLazyOrDataFrame
        Data to check
    low_bound : str
        Name of column containing the lower bound of the interval
    high_bound : str
        Name of column containing the higher bound of the interval
    group_by : IntoExpr | Iterable[IntoExpr], optional
        Parameter compatible with `.over()` function to split the check by groups,
        by default None


    Returns
    -------
    PolarsLazyOrDataFrame
        The original polars DataFrame or LazyFrame when the check passes

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
    pelage.types.PolarsAssertError: Details
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
    --> There were overlapping intervals:
    DataFrame was sorted by: ['a', 'b'],
    Interval columns: low_bound='a', high_bound='b'
    """
    is_overlapping_interval = pl.col(low_bound) <= pl.col(high_bound).shift()
    sorting_columns = [low_bound, high_bound]

    if group_by is not None:
        is_overlapping_interval = is_overlapping_interval.over(group_by)
        sorting_columns = [group_by, low_bound, high_bound]

    indexes_of_overlaps = is_overlapping_interval.arg_true()

    overlapping_ranges = (
        data.lazy()
        .sort(*sorting_columns)
        .pipe(_add_row_index)
        .filter(
            pl.col("index").is_in(indexes_of_overlaps.implode())
            | pl.col("index").is_in((indexes_of_overlaps - 1).implode())
            if _has_sufficient_polars_version("1.30.0")
            else pl.col("index").is_in(indexes_of_overlaps)
            | pl.col("index").is_in(indexes_of_overlaps - 1)
        )
        .collect()
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
