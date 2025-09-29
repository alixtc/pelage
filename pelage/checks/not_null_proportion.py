from typing import Dict, Optional, Tuple, Union

import polars as pl

from pelage.types import (
    PolarsAssertError,
    PolarsLazyOrDataFrame,
    PolarsOverClauseInput,
)
from pelage.utils import (
    _format_ranges_by_columns,
    _has_sufficient_polars_version,
)


def not_null_proportion(
    data: PolarsLazyOrDataFrame,
    items: Dict[str, Union[float, Tuple[float, float]]],
    group_by: Optional[PolarsOverClauseInput] = None,
) -> PolarsLazyOrDataFrame:
    """Checks that the proportion of non-null values in a column is within a
    a specified range [at_least, at_most] where at_most is an optional argument
    (default: 1.0).

    Parameters
    ----------
    data : PolarsLazyOrDataFrame
        _description_
    items : Dict[str, float  |  Tuple[float, float]]
        Ranges for the proportion of not null values for selected columns.

        Any of the following formats is valid:
        ```
        {
            "column_name_a" : 0.33,
            "column_name_b" : (0.25, 0.44),
            "column_name_c" : (0.25, 1.0),
            ...
        }
        ```
        When specifying a single float, the higher bound of the range will automatically
        be set to 1.0, i.e. (given_float, 1.0)

    group_by : Optional[PolarsOverClauseInput], optional
        When specified perform the check per group instead of the whole column,
        by default None

    Returns
    -------
    PolarsLazyOrDataFrame
        The original polars DataFrame or LazyFrame when the check passes

    Examples
    --------

    >>> import polars as pl
    >>> import pelage as plg
    >>> df = pl.DataFrame(
    ...         {
    ...             "a": [1, None, None],
    ...             "b": [1, 2, None],
    ...         }
    ...     )
    >>> df.pipe(plg.not_null_proportion, {"a": 0.33, "b": 0.66})
    shape: (3, 2)
    ┌──────┬──────┐
    │ a    ┆ b    │
    │ ---  ┆ ---  │
    │ i64  ┆ i64  │
    ╞══════╪══════╡
    │ 1    ┆ 1    │
    │ null ┆ 2    │
    │ null ┆ null │
    └──────┴──────┘

    >>> df.pipe(plg.not_null_proportion, {"a": 0.7})
    Traceback (most recent call last):
    ...
    pelage.types.PolarsAssertError: Details
    shape: (1, 4)
    ┌────────┬───────────────────┬──────────┬──────────┐
    │ column ┆ not_null_fraction ┆ min_prop ┆ max_prop │
    │ ---    ┆ ---               ┆ ---      ┆ ---      │
    │ str    ┆ f64               ┆ f64      ┆ i64      │
    ╞════════╪═══════════════════╪══════════╪══════════╡
    │ a      ┆ 0.333333          ┆ 0.7      ┆ 1        │
    └────────┴───────────────────┴──────────┴──────────┘
    Error with the DataFrame passed to the check function:
    --> Some columns contains a proportion of nulls beyond specified limits

    The following example details how to perform this checks for groups:
    >>> group_df = pl.DataFrame(
    ...     {
    ...         "a": [1, 1, None, None],
    ...         "group": ["A", "A", "B", "B"],
    ...     }
    ... )
    >>> group_df.pipe(plg.not_null_proportion, {"a": 0.5})
    shape: (4, 2)
    ┌──────┬───────┐
    │ a    ┆ group │
    │ ---  ┆ ---   │
    │ i64  ┆ str   │
    ╞══════╪═══════╡
    │ 1    ┆ A     │
    │ 1    ┆ A     │
    │ null ┆ B     │
    │ null ┆ B     │
    └──────┴───────┘

    >>> group_df.pipe(plg.not_null_proportion, {"a": 0.5}, group_by="group")
    Traceback (most recent call last):
    ...
    pelage.types.PolarsAssertError: Details
    shape: (1, 5)
    ┌───────┬────────┬───────────────────┬──────────┬──────────┐
    │ group ┆ column ┆ not_null_fraction ┆ min_prop ┆ max_prop │
    │ ---   ┆ ---    ┆ ---               ┆ ---      ┆ ---      │
    │ str   ┆ str    ┆ f64               ┆ f64      ┆ i64      │
    ╞═══════╪════════╪═══════════════════╪══════════╪══════════╡
    │ B     ┆ a      ┆ 0.0               ┆ 0.5      ┆ 1        │
    └───────┴────────┴───────────────────┴──────────┴──────────┘
    Error with the DataFrame passed to the check function:
    --> Some columns contains a proportion of nulls beyond specified limits
    """

    pl_ranges = _format_ranges_by_columns(items)

    # Trick with to have the same implementation between group_by and direct version
    # Also simplifies LazyFrame logic which does not have the len() except in group_by
    if group_by is None:
        formatted_data = data.with_columns(constant__=0)
        group_by = "constant__"
    else:
        formatted_data = data

    pl_len = pl.len() if _has_sufficient_polars_version("0.20.0") else pl.count()
    if _has_sufficient_polars_version("1.0.0"):
        null_proportions = (
            formatted_data.lazy()
            .group_by(group_by)
            .agg(pl.all().null_count() / pl_len)
            .unpivot(
                index=group_by,  # type: ignore
                variable_name="column",
                value_name="null_proportion",
            )
            .with_columns(not_null_fraction=1 - pl.col("null_proportion"))
            .collect()
        )
    else:
        null_proportions = (
            formatted_data.lazy()
            .group_by(group_by)
            .agg(pl.all().null_count() / pl_len)
            .melt(
                id_vars=group_by,  # type: ignore
                variable_name="column",
                value_name="null_proportion",
            )
            .with_columns(not_null_fraction=1 - pl.col("null_proportion"))
            .collect()
        )

    if "constant__" in null_proportions.columns:
        null_proportions = null_proportions.drop("constant__")

    out_of_range_null_proportions = (
        null_proportions.join(pl_ranges, on="column", how="inner")
        .filter(
            ~pl.col("not_null_fraction").is_between(
                pl.col("min_prop"), pl.col("max_prop")
            )
        )
        .drop("null_proportion")
    )
    if not out_of_range_null_proportions.is_empty():
        raise PolarsAssertError(
            out_of_range_null_proportions,
            "Some columns contains a proportion of nulls beyond specified limits",
        )
    return data
