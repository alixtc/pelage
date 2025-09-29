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


def has_no_nulls(
    data: PolarsLazyOrDataFrame,
    columns: Optional[PolarsColumnType] = None,
) -> PolarsLazyOrDataFrame:
    """Check if a DataFrame has any null (missing) values.

    Parameters
    ----------
    data : PolarsLazyOrDataFrame
        The input DataFrame to check for null values.
    columns : Optional[PolarsColumnType] , optional
        Columns to consider for null value check. By default, all columns are checked.

    Returns
    -------
    PolarsLazyOrDataFrame
        The original polars DataFrame or LazyFrame when the check passes


    Examples
    --------
    >>> import polars as pl
    >>> import pelage as plg
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
    >>> df.pipe(plg.has_no_nulls)
    Traceback (most recent call last):
        ...
    pelage.types.PolarsAssertError: Details
    shape: (1, 2)
    ┌────────┬────────────┐
    │ column ┆ null_count │
    │ ---    ┆ ---        │
    │ str    ┆ u32        │
    ╞════════╪════════════╡
    │ B      ┆ 1          │
    └────────┴────────────┘
    Error with the DataFrame passed to the check function:
    --> There were unexpected nulls in the columns above
    """
    selected_columns = _sanitize_column_inputs(columns)
    null_count = (
        (
            data.lazy()
            .select(selected_columns.null_count())
            .unpivot(variable_name="column", value_name="null_count")
            .filter(pl.col("null_count") > 0)
            .collect()
        )
        if _has_sufficient_polars_version("1.1.0")
        else (
            data.lazy()
            .select(selected_columns.null_count())
            .melt(variable_name="column", value_name="null_count")
            .filter(pl.col("null_count") > 0)
            .collect()
        )
    )

    if not null_count.is_empty():
        raise PolarsAssertError(
            null_count, "There were unexpected nulls in the columns above"
        )
    return data
