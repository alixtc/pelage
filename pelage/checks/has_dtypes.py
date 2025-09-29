from typing import Dict

import polars as pl

from pelage import utils
from pelage.types import (
    PolarsAssertError,
    PolarsDataType,
    PolarsLazyOrDataFrame,
)
from pelage.utils import _has_sufficient_polars_version


def _get_frame_schema(data: PolarsLazyOrDataFrame):
    if isinstance(data, pl.DataFrame):
        return data.schema
    if _has_sufficient_polars_version("1.0.0"):
        return data.collect_schema()

    return data.schema


def has_dtypes(
    data: PolarsLazyOrDataFrame,
    items: Dict[str, PolarsDataType],  # type: ignore
) -> PolarsLazyOrDataFrame:
    """Check that the columns have the expected types

    Parameters
    ----------
    data : PolarsLazyOrDataFrame
        Polars DataFrame or LazyFrame containing data to check.
    items : Dict[str, PolarsDataType]
        A dictionnary of column name with their expected polars data type:
        ```
        {
            "col_a": pl.String,
            "col_b": pl.Int64,
            "col_c": pl.Float64,
            ...
        }
        ```
    Returns
    -------
    PolarsLazyOrDataFrame
        The original polars DataFrame or LazyFrame when the check passes


    Examples
    --------
    >>> import polars as pl
    >>> import pelage as plg
    >>> df = pl.DataFrame({
    ...     "name": ["Alice", "Bob", "Charlie"],
    ...     "age": [20, 30, 40],
    ...     "city": ["New York", "London", "Paris"],
    ... })
    >>> df.pipe(plg.has_dtypes, {
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

    >>> df.pipe(plg.has_dtypes, {
    ...     "age": pl.String,
    ...     "city": pl.Int64,
    ... })
    Traceback (most recent call last):
        ...
    pelage.types.PolarsAssertError: Details
    Error with the DataFrame passed to the check function:
    --> Some columns don't have the expected type:
    column='age', expected_type=String, real_dtype=Int64
    column='city', expected_type=Int64, real_dtype=String
    """

    schema = _get_frame_schema(data)

    missing_columns = set(items.keys()) - set(schema.keys())
    if missing_columns:
        message = f"Dtype check, some expected columns are missing:{missing_columns}"
        raise PolarsAssertError(supp_message=message)

    bad_column_type_requirement = set(items.items()) - set(schema.items())
    if bad_column_type_requirement:
        message = utils.compare_schema(schema, items)
        message = f"Some columns don't have the expected type:\n{message}"
        raise PolarsAssertError(supp_message=message)
    return data
