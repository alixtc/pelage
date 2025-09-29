from typing import Dict, List

import polars as pl

from pelage.types import PolarsAssertError, PolarsLazyOrDataFrame


def not_accepted_values(
    data: PolarsLazyOrDataFrame, items: Dict[str, List]
) -> PolarsLazyOrDataFrame:
    """Raises error if columns contains values specified in List of forbbiden `items`

    Parameters
    ----------
    data : PolarsLazyOrDataFrame
    items : Dict[str, List]
        A dictionnary where keys are a string compatible with a pl.Expr, to be used with
        pl.col(). The value for each key is a List of all forbidden values in the
        dataframe.

    Returns
    -------
    PolarsLazyOrDataFrame
        The original polars DataFrame or LazyFrame when the check passes


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
    pelage.types.PolarsAssertError: Details
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
    --> This DataFrame contains values marked as forbidden
    """
    mask_for_forbidden_values = [
        pl.col(col).is_in(values) for col, values in items.items()
    ]
    forbidden_values = (
        data.lazy().filter(pl.Expr.or_(*mask_for_forbidden_values)).collect()
    )

    if not forbidden_values.is_empty():
        bad_column_names = [
            col.name
            for col in forbidden_values.select(mask_for_forbidden_values)
            if col.any()
        ]
        raise PolarsAssertError(
            forbidden_values.select(bad_column_names),
            "This DataFrame contains values marked as forbidden",
        )
    return data
