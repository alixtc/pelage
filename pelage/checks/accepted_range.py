from typing import Dict

import polars as pl

from pelage.types import (
    PolarsAssertError,
    PolarsColumnBounds,
    PolarsLazyOrDataFrame,
)


def accepted_range(
    data: PolarsLazyOrDataFrame, items: Dict[str, PolarsColumnBounds]
) -> PolarsLazyOrDataFrame:
    """Check that all the values from specifed columns in the dict `items` are within
        the indicated range.

    Parameters
    ----------
    data : PolarsLazyOrDataFrame
    items : Dict[str, PolarsColumnBounds]
        Any type of inputs that match the following signature:
        `column_name: (boundaries)` where boundaries is compatible with the Polars
        method `is_between()` syntax.

        For example:
        ```
        {
        "col_a": (low, high),
        "col_b", (low_b, high_b, "right"),
        "col_c", (low_c, high_c, "none"),
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
    >>> df = pl.DataFrame({"a": [1, 2, 3]})
    >>> df.pipe(plg.accepted_range, {"a": (0, 2)})
    Traceback (most recent call last):
    ...
    pelage.types.PolarsAssertError: Details
    shape: (1, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 3   │
    └─────┘
    Error with the DataFrame passed to the check function:
    --> Some values are beyond the acceptable ranges defined

    >>> df.pipe(plg.accepted_range, {"a": (1, 3)})
    shape: (3, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 1   │
    │ 2   │
    │ 3   │
    └─────┘

    Examples with string containing columns and specifying boundaries:
    >>> df = pl.DataFrame({"a": ["b", "c"]})
    >>> df.pipe(plg.accepted_range, {"a": (pl.lit("a"), pl.lit("d"), "right")})
    shape: (2, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ str │
    ╞═════╡
    │ b   │
    │ c   │
    └─────┘
    >>> df.pipe(plg.accepted_range, {"a": (pl.lit("a"), pl.lit("d"), "left")})
    shape: (2, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ str │
    ╞═════╡
    │ b   │
    │ c   │
    └─────┘
    """
    closed_boundaries = {
        k: (v if len(v) == 3 else (*v, "both")) for k, v in items.items()
    }
    forbidden_ranges = [
        pl.col(k).is_between(*v).not_()  # type: ignore
        for k, v in closed_boundaries.items()
    ]
    out_of_range = data.lazy().filter(pl.Expr.or_(*forbidden_ranges)).collect()

    if not out_of_range.is_empty():
        raise PolarsAssertError(
            out_of_range, "Some values are beyond the acceptable ranges defined"
        )
    return data
