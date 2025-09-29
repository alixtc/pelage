from typing import Optional, Tuple

import polars as pl

from pelage.types import (
    IntOrNone,
    PolarsAssertError,
    PolarsLazyOrDataFrame,
    PolarsOverClauseInput,
)
from pelage.utils import _has_sufficient_polars_version


def has_shape(
    data: PolarsLazyOrDataFrame,
    shape: Tuple[IntOrNone, IntOrNone],
    group_by: Optional[PolarsOverClauseInput] = None,
) -> PolarsLazyOrDataFrame:
    """Check if a DataFrame has the specified shape.

    When used with the group_by option, this can be used to get the row count per group.

    Parameters
    ----------
    data : PolarsLazyOrDataFrame
        Input data
    shape : Tuple[IntOrNone, IntOrNone]
        Tuple with the expected dataframe shape, as from the `.shape()` method.
        You can use `None` for one of the two elements of the shape tuple if you do not
        want to check this dimension.

        Ex: `(5, None)` will ensure that the dataframe has 5 rows regardless of the
        number of columns.

    group_by : Optional[PolarsOverClauseInput], optional
        When specified compares the number of lines per group with the expected value,
        by default None

    Returns
    -------
    PolarsLazyOrDataFrame
        The original polars DataFrame or LazyFrame when the check passes

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

    Without specifying the number of columns:
    >>> df.pipe(plg.has_shape, (2, None))
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
    pelage.types.PolarsAssertError: Details
    Error with the DataFrame passed to the check function:
    --> The data has not the expected shape: (1, 2)

    Checking the number of rows per group:
    >>> group_example_df = pl.DataFrame(
    ...     {
    ...         "a": [1, 2, 3],
    ...         "b": ["a", "b", "b"],
    ...     }
    ... )
    >>> group_example_df.pipe(plg.has_shape, (1, None), group_by="b")
    Traceback (most recent call last):
    ...
    pelage.types.PolarsAssertError: Details
    shape: (1, 2)
    ┌─────┬─────┐
    │ b   ┆ len │
    │ --- ┆ --- │
    │ str ┆ u32 │
    ╞═════╪═════╡
    │ b   ┆ 2   │
    └─────┴─────┘
    Error with the DataFrame passed to the check function:
    --> The number of rows per group does not match the specified value: 1
    """

    if shape[0] is None and shape[1] is None:
        raise ValueError(
            "Both dimensions for expected shape cannot be set None simultaneously"
        )

    pl_len = (
        pl.len()
        if _has_sufficient_polars_version("0.20.0")
        else pl.count().alias("len")
    )

    if group_by is not None:
        non_matching_row_count = (
            data.lazy()
            .group_by(group_by)
            .agg(pl_len)
            .filter(pl.col("len") != shape[0])
            .collect()
        )

        if len(non_matching_row_count) > 0:
            raise PolarsAssertError(
                df=non_matching_row_count,
                supp_message=f"The number of rows per group does not match the specified value: {shape[0]}",  # noqa: E501
            )
        return data

    actual_shape = _get_frame_shape(data)

    if shape[1] is None:
        actual_shape = actual_shape[0], None

    if shape[0] is None:
        actual_shape = None, actual_shape[1]

    if actual_shape != shape:
        raise PolarsAssertError(
            supp_message=f"The data has not the expected shape: {shape}"
        )
    return data


def _get_frame_shape(data: PolarsLazyOrDataFrame) -> Tuple[int, int]:
    """Convenience function to get shape of Lazyframe given available methods"""
    if isinstance(data, pl.DataFrame):
        return data.shape

    pl_len = pl.len() if _has_sufficient_polars_version("0.20.0") else pl.count()

    if _has_sufficient_polars_version("1.0.0"):
        return (
            data.select(pl_len).collect().item(),
            len(data.collect_schema()),
        )
    else:
        return (
            data.select(pl_len).collect().item(),
            len(data.columns),
        )
