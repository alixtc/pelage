from typing import Tuple

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


def column_is_within_n_std(
    data: PolarsLazyOrDataFrame,
    items: Tuple[PolarsColumnType, int],
    *args: Tuple[PolarsColumnType, int],
) -> PolarsLazyOrDataFrame:
    """Function asserting values are within a given STD range, thus ensuring the absence
    of outliers.

    Parameters
    ----------
    data : PolarsLazyOrDataFrame
        Polars DataFrame or LazyFrame containing data to check.
    items : Tuple[PolarsColumnType, int]
        A column name / column type with the number of STD authorized for the values
        within. Must be of the following form: `(col_name, n_std)`

    Returns
    -------
    PolarsLazyOrDataFrame
        The original polars DataFrame or LazyFrame when the check passes

    Examples
    --------

    >>> import polars as pl
    >>> import pelage as plg
    >>> df = pl.DataFrame(
    ...     {
    ...         "a": list(range(0, 11)),
    ...         "b": list(range(0, 11)),
    ...         "c": list(range(0, 10)) + [5000],
    ...     }
    ... )
    >>> df.pipe(plg.column_is_within_n_std, ("a", 2), ("b", 3))
    shape: (11, 3)
    ┌─────┬─────┬──────┐
    │ a   ┆ b   ┆ c    │
    │ --- ┆ --- ┆ ---  │
    │ i64 ┆ i64 ┆ i64  │
    ╞═════╪═════╪══════╡
    │ 0   ┆ 0   ┆ 0    │
    │ 1   ┆ 1   ┆ 1    │
    │ 2   ┆ 2   ┆ 2    │
    │ 3   ┆ 3   ┆ 3    │
    │ 4   ┆ 4   ┆ 4    │
    │ …   ┆ …   ┆ …    │
    │ 6   ┆ 6   ┆ 6    │
    │ 7   ┆ 7   ┆ 7    │
    │ 8   ┆ 8   ┆ 8    │
    │ 9   ┆ 9   ┆ 9    │
    │ 10  ┆ 10  ┆ 5000 │
    └─────┴─────┴──────┘

    >>> df.pipe(plg.column_is_within_n_std, ("b", 2), ("c", 2))
    Traceback (most recent call last):
    ...
    pelage.types.PolarsAssertError: Details
    shape: (1, 1)
    ┌──────┐
    │ c    │
    │ ---  │
    │ i64  │
    ╞══════╡
    │ 5000 │
    └──────┘
    Error with the DataFrame passed to the check function:
    --> There are some outliers outside the specified mean±std range
    Impacted columns: ['c']
    """
    check_items = [items, *args]

    pairs_to_check = [
        (_sanitize_column_inputs(col), n_std) for col, n_std in check_items
    ]

    keep_outlier_nullify_others = [
        pl.when(
            col.is_between(
                col.mean() - n_std * col.std(),
                col.mean() + n_std * col.std(),
            ).not_()
        )
        .then(col)  # Then `col` must come first to propagate column name for suffix
        .otherwise(None)
        .name.suffix("_out__")
        for col, n_std in pairs_to_check
    ]

    tagged_outliers = (
        data.lazy()
        .select(*keep_outlier_nullify_others)
        .filter(pl.any_horizontal(pl.all().is_not_null()))
        .collect()
    )
    if _has_sufficient_polars_version("0.20.0"):
        tagged_outliers = tagged_outliers.rename(lambda col: col.replace("_out__", ""))
    else:
        tagged_outliers = tagged_outliers.rename(
            {col: col.replace("_out__", "") for col in tagged_outliers.columns}
        )

    columns_with_null = [col.name for col in tagged_outliers if col.is_not_null().any()]

    if columns_with_null:
        bad_examples = tagged_outliers.select(columns_with_null)
        raise PolarsAssertError(
            df=bad_examples,
            supp_message=(
                "There are some outliers outside the specified mean±std range"
                + "\n"
                + f"Impacted columns: {columns_with_null}"
            ),
        )

    return data
