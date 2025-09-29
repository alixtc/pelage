from typing import List, Optional, Union

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


def not_constant(
    data: PolarsLazyOrDataFrame,
    columns: Optional[PolarsColumnType] = None,
    group_by: Optional[Union[str, List[str]]] = None,
) -> PolarsLazyOrDataFrame:
    """Check if a DataFrame has constant columns.

    Parameters
    ----------
    data : PolarsLazyOrDataFrame
        The input DataFrame to check for null values.
    columns : Optional[PolarsColumnType] , optional
        Columns to consider for null value check. By default, all columns are checked.
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
    >>> df = pl.DataFrame({"a": [1, 2]})
    >>> df.pipe(plg.not_constant, "a")
    shape: (2, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 1   │
    │ 2   │
    └─────┘
    >>> df = pl.DataFrame({"b": [1, 1]})

    >>> df.pipe(plg.not_constant)
    Traceback (most recent call last):
    ...
    pelage.types.PolarsAssertError: Details
    shape: (1, 2)
    ┌────────┬────────────┐
    │ column ┆ n_distinct │
    │ ---    ┆ ---        │
    │ str    ┆ u32        │
    ╞════════╪════════════╡
    │ b      ┆ 1          │
    └────────┴────────────┘
    Error with the DataFrame passed to the check function:
    --> Some columns are constant

    The folloing example details how to perform this checks for groups:
    >>> import polars as pl
    >>> import pelage as plg
    >>> df = pl.DataFrame(
    ...     {
    ...         "a": [1, 2, 1, 1],
    ...         "b": ["A", "A", "B", "B"],
    ...     }
    ... )
    >>> df.pipe(plg.not_constant, "a")
    shape: (4, 2)
    ┌─────┬─────┐
    │ a   ┆ b   │
    │ --- ┆ --- │
    │ i64 ┆ str │
    ╞═════╪═════╡
    │ 1   ┆ A   │
    │ 2   ┆ A   │
    │ 1   ┆ B   │
    │ 1   ┆ B   │
    └─────┴─────┘

    >>> df.pipe(plg.not_constant, "a", group_by="b")
    Traceback (most recent call last):
    ...
    pelage.types.PolarsAssertError: Details
    shape: (1, 3)
    ┌─────┬────────┬────────────┐
    │ b   ┆ column ┆ n_distinct │
    │ --- ┆ ---    ┆ ---        │
    │ str ┆ str    ┆ u32        │
    ╞═════╪════════╪════════════╡
    │ B   ┆ a      ┆ 1          │
    └─────┴────────┴────────────┘
    Error with the DataFrame passed to the check function:
    --> Some columns are constant within a given group
    """
    selected_cols = _sanitize_column_inputs(columns)

    if group_by is None:
        if _has_sufficient_polars_version("1.0.0"):
            constant_columns = (
                data.lazy()
                .select(selected_cols.n_unique())
                .unpivot(variable_name="column", value_name="n_distinct")
                .filter(pl.col("n_distinct") == 1)
                .collect()
            )
        else:
            constant_columns = (
                data.lazy()
                .select(selected_cols.n_unique())
                .melt(variable_name="column", value_name="n_distinct")
                .filter(pl.col("n_distinct") == 1)
                .collect()
            )
    else:
        if _has_sufficient_polars_version("1.0.0"):
            constant_columns = (
                data.lazy()
                .group_by(group_by)
                .agg(selected_cols.n_unique())
                .unpivot(
                    index=group_by, variable_name="column", value_name="n_distinct"
                )
                .filter(pl.col("n_distinct") == 1)
                .collect()
            )
        else:
            constant_columns = (
                data.lazy()
                .group_by(group_by)
                .agg(selected_cols.n_unique())
                .melt(id_vars=group_by, variable_name="column", value_name="n_distinct")
                .filter(pl.col("n_distinct") == 1)
                .collect()
            )

    if not constant_columns.is_empty():
        group_message = " within a given group" if group_by is not None else ""
        raise PolarsAssertError(
            constant_columns,
            supp_message=f"Some columns are constant{group_message}",
        )

    return data
