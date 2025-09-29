from typing import Optional

import polars as pl

from pelage.types import (
    PolarsAssertError,
    PolarsColumnType,
    PolarsLazyOrDataFrame,
)
from pelage.utils import (
    _has_sufficient_polars_version,
    _sanitize_column_inputs,
)


def unique_combination_of_columns(
    data: PolarsLazyOrDataFrame,
    columns: Optional[PolarsColumnType] = None,
) -> PolarsLazyOrDataFrame:
    """Ensure that the selected column have a unique combination per row.

    This function is particularly helpful to establish the granularity of a dataframe,
    i.e. this is a row oriented check.

    Parameters
    ----------
    data : PolarsLazyOrDataFrame
        _description_
    columns : Optional[PolarsColumnType] , optional
        Columns to consider for row unicity. By default, all columns are checked.

    Returns
    -------
    PolarsLazyOrDataFrame
        The original polars DataFrame or LazyFrame when the check passes


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
    pelage.types.PolarsAssertError: Details
    shape: (1, 2)
    ┌─────┬─────┐
    │ a   ┆ len │
    │ --- ┆ --- │
    │ str ┆ u32 │
    ╞═════╪═════╡
    │ X   ┆ 2   │
    └─────┴─────┘
    Error with the DataFrame passed to the check function:
    --> Some combinations of columns are not unique. See above, selected: col("a")
    """
    cols = _sanitize_column_inputs(columns)
    pl_len = (
        pl.len()
        if _has_sufficient_polars_version("0.20.0")
        else pl.count().alias("len")
    )
    non_unique_combinations = (
        data.lazy().group_by(cols).agg(pl_len).filter(pl.col("len") > 1).collect()
    )

    if not non_unique_combinations.is_empty():
        raise PolarsAssertError(
            non_unique_combinations,
            f"Some combinations of columns are not unique. See above, selected: {cols}",
        )
    return data
