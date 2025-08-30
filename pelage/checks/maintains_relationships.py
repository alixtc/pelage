from typing import Union

import polars as pl

from pelage.checks.utils.checks import PolarsAssertError
from pelage.checks.utils.types import PolarsLazyOrDataFrame


def maintains_relationships(
    data: PolarsLazyOrDataFrame,
    other_df: Union[pl.DataFrame, pl.LazyFrame],
    column: str,
) -> PolarsLazyOrDataFrame:
    """Function to help ensuring that set of values in selected column remains  the
        same in both DataFrames. This helps to maintain referential integrity.

    Parameters
    ----------
    data : PolarsLazyOrDataFrame
        Dataframe after transformation
    other_df : Union[pl.DataFrame, pl.LazyFrame]
        Distant dataframe usually the one before transformation
    column : str
        Column to check for keys/ids

    Returns
    -------
    PolarsLazyOrDataFrame
        The original polars DataFrame or LazyFrame when the check passes

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
    --> Some values were removed from col 'a', for ex: ('b',)
    """

    local_keys = set(data.lazy().select(column).collect().get_column(column))
    other_keys = set(other_df.lazy().select(column).collect().get_column(column))

    if local_keys != other_keys:
        if local_keys > other_keys:
            set_diff = sorted(list(local_keys - other_keys)[:5])
            msg = f"Some values were added to col '{column}', for ex: {(*set_diff,)}"
            raise PolarsAssertError(supp_message=msg)
        else:
            set_diff = sorted(list(other_keys - local_keys)[:5])
            msg = (
                f"Some values were removed from col '{column}', for ex: {(*set_diff,)}"
            )
            raise PolarsAssertError(supp_message=msg)

    return data
