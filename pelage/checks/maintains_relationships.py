from typing import Union

import polars as pl

from pelage.types import PolarsAssertError, PolarsLazyOrDataFrame


def maintains_relationships(
    data: PolarsLazyOrDataFrame,
    other_df: Union[pl.DataFrame, pl.LazyFrame],
    column: Union[str, list[str]],
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
    pelage.types.PolarsAssertError: Details
    shape: (1, 2)
    ┌──────┬──────────┐
    │ a    ┆ a_in_ref │
    │ ---  ┆ ---      │
    │ str  ┆ str      │
    ╞══════╪══════════╡
    │ null ┆ b        │
    └──────┴──────────┘
    Error with the DataFrame passed to the check function:
    --> Some values were removed from col 'a', see above!
    """
    current_df = (
        data.lazy().select(column).unique().with_columns(_current=pl.lit("_current"))
    )

    reference_df = (
        other_df.lazy()
        .select(column)
        .unique()
        .with_columns(_reference=pl.lit("_reference"))
    )
    key_mismatches = (
        current_df.join(
            reference_df,
            on=column,
            how="full",
            suffix="_in_ref",
            # coalesce=True,
        )
        .filter(pl.col("_reference").is_null() | pl.col("_current").is_null())
        .head(200)
        .collect()
    )

    added_keys_from_ref = key_mismatches.filter(pl.col("_reference").is_null()).drop(
        "_current", "_reference"
    )
    missing_keys_from_ref = key_mismatches.filter(pl.col("_current").is_null()).drop(
        "_current", "_reference"
    )
    if not added_keys_from_ref.is_empty():
        msg = f"Some values were added to col '{column}', see above!"
        raise PolarsAssertError(supp_message=msg, df=added_keys_from_ref)

    if not missing_keys_from_ref.is_empty():
        msg = f"Some values were removed from col '{column}', see above!"
        raise PolarsAssertError(supp_message=msg, df=missing_keys_from_ref)

    return data
