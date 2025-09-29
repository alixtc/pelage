from typing import Optional

import polars as pl

from pelage.types import (
    PolarsAssertError,
    PolarsColumnType,
    PolarsLazyOrDataFrame,
    PolarsOverClauseInput,
)
from pelage.utils import _has_sufficient_polars_version, _sanitize_column_inputs


def at_least_one(
    data: PolarsLazyOrDataFrame,
    columns: Optional[PolarsColumnType] = None,
    group_by: Optional[PolarsOverClauseInput] = None,
) -> PolarsLazyOrDataFrame:
    """Ensure that there is at least one not null value in the designated columns.

    Parameters
    ----------
    data : PolarsLazyOrDataFrame
        Polars DataFrame or LazyFrame containing data to check.
    columns : Optional[PolarsColumnType], optional
        Columns to consider to check the presence of at least one value.
        By default, all columns are checked.
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
    >>> df = pl.DataFrame({"a": [None, None], "b": [1, None]})
    >>> df.pipe(plg.at_least_one, "b")
    shape: (2, 2)
    ┌──────┬──────┐
    │ a    ┆ b    │
    │ ---  ┆ ---  │
    │ null ┆ i64  │
    ╞══════╪══════╡
    │ null ┆ 1    │
    │ null ┆ null │
    └──────┴──────┘

    >>> df.pipe(plg.at_least_one)
    Traceback (most recent call last):
    ...
    pelage.types.PolarsAssertError: Details
    Error with the DataFrame passed to the check function:
    --> Some columns contains only null values: ['a']

    The folloing example details how to perform this checks for groups:
    >>> df = pl.DataFrame(
    ...         {
    ...             "a": [None, None, None, 2],
    ...             "group": ["G1", "G1", "G2", "G2"],
    ...         }
    ...     )
    >>> df.pipe(plg.at_least_one, "a")
    shape: (4, 2)
    ┌──────┬───────┐
    │ a    ┆ group │
    │ ---  ┆ ---   │
    │ i64  ┆ str   │
    ╞══════╪═══════╡
    │ null ┆ G1    │
    │ null ┆ G1    │
    │ null ┆ G2    │
    │ 2    ┆ G2    │
    └──────┴───────┘

    >>> df.pipe(plg.at_least_one, "a", group_by="group")
    Traceback (most recent call last):
    ...
    pelage.types.PolarsAssertError: Details
    shape: (1, 3)
    ┌───────┬─────────┬──────────────┐
    │ group ┆ columns ┆ at_least_one │
    │ ---   ┆ ---     ┆ ---          │
    │ str   ┆ str     ┆ bool         │
    ╞═══════╪═════════╪══════════════╡
    │ G1    ┆ a       ┆ false        │
    └───────┴─────────┴──────────────┘
    Error with the DataFrame passed to the check function:
    --> Some columns contains only null values per group
    """

    selected_columns = _sanitize_column_inputs(columns)

    pl_len = pl.len() if _has_sufficient_polars_version() else pl.count()

    if group_by is not None:
        if _has_sufficient_polars_version("1.0.0"):
            only_nulls_per_group = (
                data.lazy()
                .group_by(group_by)
                .agg(selected_columns.null_count() < pl_len)
                .unpivot(
                    index=group_by,  # type: ignore
                    variable_name="columns",
                    value_name="at_least_one",
                )
                .filter(pl.col("at_least_one").not_())
                .collect()
            )
        else:
            only_nulls_per_group = (
                data.lazy()
                .group_by(group_by)
                .agg(selected_columns.null_count() < pl_len)
                .melt(
                    id_vars=group_by,  # type: ignore
                    variable_name="columns",
                    value_name="at_least_one",
                )
                .filter(pl.col("at_least_one").not_())
                .collect()
            )

        if len(only_nulls_per_group) > 0:
            raise PolarsAssertError(
                df=only_nulls_per_group,
                supp_message="Some columns contains only null values per group",
            )
        return data

    # Implementation that work for both Dataframe and Lazyframe
    are_column_nulls = (
        data.lazy()
        .select(selected_columns)
        .with_columns(constant__=1)
        .group_by("constant__")
        .agg(pl.all().null_count() == pl_len)
        .drop("constant__")
        .collect()
    )

    null_columns = [col.name for col in are_column_nulls if col.all()]

    if null_columns:
        raise PolarsAssertError(
            supp_message=f"Some columns contains only null values: {null_columns}"
        )
    return data
