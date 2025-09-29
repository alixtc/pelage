from typing import List, Set, Union

import polars as pl

from pelage.types import PolarsAssertError, PolarsLazyOrDataFrame
from pelage.utils import _has_sufficient_polars_version


def _get_lazyframe_columns(data: pl.LazyFrame) -> Set[str]:
    if _has_sufficient_polars_version("1.0.0"):
        return set(data.collect_schema().names())
    else:
        return set(data.columns)


def has_columns(
    data: PolarsLazyOrDataFrame, names: Union[str, List[str]]
) -> PolarsLazyOrDataFrame:
    """Check if a DataFrame has the specified

    Parameters
    ----------
    data : PolarsLazyOrDataFrame
        The DataFrame to check for column presence.
    names : Union[str, List[str]]
        The names of the columns to check.

    Returns
    -------
    PolarsLazyOrDataFrame
        The original polars DataFrame or LazyFrame when the check passes

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
    pelage.types.PolarsAssertError: Details
    Error with the DataFrame passed to the check function:
    --> Missing columns if the dataframe: {'c'}

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
    """
    if isinstance(names, str):
        # Because set(str) explodes the string
        names = [names]
    column_names_set = (
        _get_lazyframe_columns(data)
        if isinstance(data, pl.LazyFrame)
        else set(data.columns)
    )
    mising_columns = set(names) - column_names_set
    if mising_columns:
        raise PolarsAssertError(
            supp_message=f"Missing columns if the dataframe: {mising_columns}"
        )
    return data
