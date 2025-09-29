from typing import Dict, Optional

import polars as pl

from pelage.types import (
    PolarsAssertError,
    PolarsLazyOrDataFrame,
    PolarsOverClauseInput,
)
from pelage.utils import _has_sufficient_polars_version


def _format_missing_elements(selected_data: pl.DataFrame, items: Dict):
    missing = {}
    for key in items:
        required_values = set(items[key])
        present_values = set(selected_data.get_column(key))
        should_be_present = required_values - present_values
        if should_be_present:
            missing[key] = sorted(should_be_present)
    return missing


def compare_sets_per_column(
    data: PolarsLazyOrDataFrame, items: dict
) -> PolarsLazyOrDataFrame:
    if _has_sufficient_polars_version():
        expected_sets = {f"{k}_expected_set": pl.lit(v) for k, v in items.items()}
    else:
        expected_sets = {f"{k}_expected_set": v for k, v in items.items()}

    return data.with_columns(**expected_sets).filter(
        pl.Expr.or_(
            *[
                pl.col(f"{k}_expected_set").list.set_difference(pl.col(k)).list.len()
                != 0
                for k in items
            ]
        )
    )


def has_mandatory_values(
    data: PolarsLazyOrDataFrame,
    items: Dict[str, list],
    group_by: Optional[PolarsOverClauseInput] = None,
) -> PolarsLazyOrDataFrame:
    """Ensure that all specified values are present in their respective column.

    Parameters
    ----------
    data :  PolarsLazyOrDataFrame
        Polars DataFrame or LazyFrame containing data to check.
    items : Dict[str, list]
        A dictionnary where the keys are the columns names and the values are lists that
        contains all the required values for a given column.
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
    >>> df.pipe(plg.has_mandatory_values, {"a": [1, 2]})
    shape: (2, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 1   │
    │ 2   │
    └─────┘
    >>> df.pipe(plg.has_mandatory_values, {"a": [3, 4]})
    Traceback (most recent call last):
    ...
    pelage.types.PolarsAssertError: Details
    Error with the DataFrame passed to the check function:
    --> Missing mandatory values in the following columns: {'a': [3, 4]}

    The folloing example details how to perform this checks for groups:
    >>> group_df_example = pl.DataFrame(
    ...     {
    ...         "a": [1, 1, 1, 2],
    ...         "group": ["G1", "G1", "G2", "G2"],
    ...     }
    ... )
    >>> group_df_example.pipe(plg.has_mandatory_values, {"a": [1, 2]})
    shape: (4, 2)
    ┌─────┬───────┐
    │ a   ┆ group │
    │ --- ┆ ---   │
    │ i64 ┆ str   │
    ╞═════╪═══════╡
    │ 1   ┆ G1    │
    │ 1   ┆ G1    │
    │ 1   ┆ G2    │
    │ 2   ┆ G2    │
    └─────┴───────┘

    >>> group_df_example.pipe(plg.has_mandatory_values, {"a": [1, 2]}, group_by="group")
    Traceback (most recent call last):
    ...
    pelage.types.PolarsAssertError: Details
    shape: (1, 3)
    ┌───────┬───────────┬────────────────┐
    │ group ┆ a         ┆ a_expected_set │
    │ ---   ┆ ---       ┆ ---            │
    │ str   ┆ list[i64] ┆ list[i64]      │
    ╞═══════╪═══════════╪════════════════╡
    │ G1    ┆ [1]       ┆ [1, 2]         │
    └───────┴───────────┴────────────────┘
    Error with the DataFrame passed to the check function:
    --> Some groups are missing mandatory values
    """
    if group_by is not None:
        groups_missing_mandatory = (
            data.lazy()
            .group_by(group_by)
            .agg(pl.col(k).unique() for k in items)
            .pipe(compare_sets_per_column, items)
            .collect()
        )

        if len(groups_missing_mandatory) > 0:
            raise PolarsAssertError(
                df=groups_missing_mandatory,
                supp_message="Some groups are missing mandatory values",
            )
        return data

    selected_data = data.lazy().select(pl.col(items.keys())).unique().collect()

    missing = _format_missing_elements(selected_data, items)

    if missing:
        raise PolarsAssertError(
            supp_message=f"Missing mandatory values in the following columns: {missing}"
        )
    return data
