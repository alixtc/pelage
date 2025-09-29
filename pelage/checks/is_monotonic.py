from typing import Optional, Union

import polars as pl

from pelage.types import (
    PolarsAssertError,
    PolarsLazyOrDataFrame,
    PolarsOverClauseInput,
)


def is_monotonic(
    data: PolarsLazyOrDataFrame,
    column: str,
    decreasing: bool = False,
    strict: bool = True,
    interval: Optional[Union[int, float, str]] = None,
    group_by: Optional[PolarsOverClauseInput] = None,
) -> PolarsLazyOrDataFrame:
    """Verify that values in a column are consecutively increasing or decreasing.

    Parameters
    ----------
    data : PolarsLazyOrDataFrame
        Polars DataFrame or LazyFrame containing data to check.
    column : str
        Name of the column that should be monotonic.
    decreasing : bool, optional
        Should the column be decreasing, by default False
    strict : bool, optional
        The series must be stricly increasing or decreasing, no consecutive equal values
        are allowed, by default True
    interval : Optional[Union[int, float, str, pl.Duration]], optional
        For time-based column, the interval can be specified as a string as in the
        function `dt.offset_by` or `pl.DataFrame().rolling`. It can also be specified
        with the `pl.duration()` function directly in a more explicit manner.

        When using a string, the interval is dictated by the following string language:

            - 1ns (1 nanosecond)
            - 1us (1 microsecond)
            - 1ms (1 millisecond)
            - 1s (1 second)
            - 1m (1 minute)
            - 1h (1 hour)
            - 1d (1 calendar day)
            - 1w (1 calendar week)
            - 1mo (1 calendar month)
            - 1q (1 calendar quarter)
            - 1y (1 calendar year)
            - 1i (1 index count)

        By "calendar day", we mean the corresponding time on the next day (which may
        not be 24 hours, due to daylight savings). Similarly for "calendar week",
        "calendar month", "calendar quarter", and "calendar year".

        By default None
    group_by : Optional[PolarsOverClauseInput], optional
        When specified, the monotonic characteristics and intervals are estimated for
        each group independently.

        by default None

    Returns
    -------
    PolarsLazyOrDataFrame
        The original polars DataFrame or LazyFrame when the check passes


    Examples
    --------

    >>> import polars as pl
    >>> import pelage as plg
    >>> df =     given = pl.DataFrame({"int": [1, 2, 1]})
    >>> df = pl.DataFrame({"int": [1, 2, 3], "str": ["x", "y", "z"]})
    >>> df.pipe(plg.is_monotonic, "int")
    shape: (3, 2)
    ┌─────┬─────┐
    │ int ┆ str │
    │ --- ┆ --- │
    │ i64 ┆ str │
    ╞═════╪═════╡
    │ 1   ┆ x   │
    │ 2   ┆ y   │
    │ 3   ┆ z   │
    └─────┴─────┘

    >>> bad = pl.DataFrame({"data": [1, 2, 3, 1]})
    >>> bad.pipe(plg.is_monotonic, "data")
    Traceback (most recent call last):
    ...
    pelage.types.PolarsAssertError: Details
    shape: (2, 1)
    ┌──────┐
    │ data │
    │ ---  │
    │ i64  │
    ╞══════╡
    │ 3    │
    │ 1    │
    └──────┘
    Error with the DataFrame passed to the check function:
    --> Column "data" expected to be monotonic but is not, try .sort("data")

    The folloing example details how to perform this checks for groups:
    >>> given = pl.DataFrame(
    ...     [
    ...         ("2020-01-01 01:42:00", "A"),
    ...         ("2020-01-01 01:43:00", "A"),
    ...         ("2020-01-01 01:44:00", "A"),
    ...         ("2021-12-12 01:43:00", "B"),
    ...         ("2021-12-12 01:44:00", "B"),
    ...     ],
    ...     schema=["dates", "group"],
    ...     orient="row",
    ... ).with_columns(pl.col("dates").str.to_datetime())
    >>> given.pipe(plg.is_monotonic, "dates", interval="1m", group_by="group")
    shape: (5, 2)
    ┌─────────────────────┬───────┐
    │ dates               ┆ group │
    │ ---                 ┆ ---   │
    │ datetime[μs]        ┆ str   │
    ╞═════════════════════╪═══════╡
    │ 2020-01-01 01:42:00 ┆ A     │
    │ 2020-01-01 01:43:00 ┆ A     │
    │ 2020-01-01 01:44:00 ┆ A     │
    │ 2021-12-12 01:43:00 ┆ B     │
    │ 2021-12-12 01:44:00 ┆ B     │
    └─────────────────────┴───────┘
    >>> given.pipe(plg.is_monotonic, "dates", interval="3m", group_by="group")
    Traceback (most recent call last):
    ...
    pelage.types.PolarsAssertError: Details
    shape: (3, 3)
    ┌─────────────────────┬───────┬────────────────────────────────┐
    │ dates               ┆ group ┆ _previous_entry_with_3m_offset │
    │ ---                 ┆ ---   ┆ ---                            │
    │ datetime[μs]        ┆ str   ┆ datetime[μs]                   │
    ╞═════════════════════╪═══════╪════════════════════════════════╡
    │ 2020-01-01 01:43:00 ┆ A     ┆ 2020-01-01 01:45:00            │
    │ 2020-01-01 01:44:00 ┆ A     ┆ 2020-01-01 01:46:00            │
    │ 2021-12-12 01:44:00 ┆ B     ┆ 2021-12-12 01:46:00            │
    └─────────────────────┴───────┴────────────────────────────────┘
    Error with the DataFrame passed to the check function:
    --> Intervals differ from the specified one: 3m.
    """
    if group_by is None:
        # with version >= 0.20 .over(None) does nothing, but before it fails, use dummy.
        group_by = 1

    select_diff_expr = pl.col(column).diff().over(group_by)

    diff_column = data.lazy().select(select_diff_expr).collect().get_column(column)

    # pl.Duration does not have .sign() method out of the blue.
    # Cast necessary for dates and datetimes
    diff_column_sign = diff_column.cast(int).sign()

    if not decreasing and not strict:
        comparisons = (diff_column_sign >= 0).all()
        previous_and_current_match_expr = (select_diff_expr >= 0) & (
            pl.col(column).diff().shift(-1).over(group_by) >= 0
        )
    elif not decreasing and strict:
        comparisons = (diff_column_sign > 0).all()
        previous_and_current_match_expr = (select_diff_expr > 0) & (
            pl.col(column).diff().shift(-1).over(group_by) > 0
        )
    elif decreasing and not strict:
        comparisons = (diff_column_sign <= 0).all()
        previous_and_current_match_expr = (select_diff_expr <= 0) & (
            pl.col(column).diff().shift(-1).over(group_by) <= 0
        )
    else:
        comparisons = (diff_column_sign < 0).all()
        previous_and_current_match_expr = (select_diff_expr < 0) & (
            pl.col(column).diff().shift(-1).over(group_by) < 0
        )

    if not comparisons:
        consecutive_bad_lines = (
            data.lazy().filter(previous_and_current_match_expr.not_()).collect()
        )
        error_msg = (
            f'Column "{column}" expected to be monotonic but is not,'
            + f' try .sort("{column}")'
        )
        raise PolarsAssertError(df=consecutive_bad_lines, supp_message=error_msg)

    if interval is None:
        return data

    if diff_column.dtype == pl.Duration:
        assert isinstance(interval, str), (
            "The interval should be a string compatible with polars time definitions, "
            + f"but was {interval}."
        )

        bad_intervals = (
            data.lazy()
            .with_columns(
                pl.col(column)
                .shift()
                .over(group_by)
                .dt.offset_by(interval)
                .alias(f"_previous_entry_with_{interval}_offset")
            )
            .drop_nulls()
            .filter(pl.col(column) != pl.col(f"_previous_entry_with_{interval}_offset"))
            .collect()
        )

        if not bad_intervals.is_empty():
            raise PolarsAssertError(
                supp_message=f"Intervals differ from the specified one: {interval}.",
                df=bad_intervals,
            )
        return data

    else:
        bad_intervals = (diff_column != interval).any()

    if bad_intervals:
        highlight_bad_intervals = (
            data.lazy()
            .with_columns(_previous_delta=select_diff_expr)
            .filter(select_diff_expr != interval)
            .collect()
        )
        raise PolarsAssertError(
            df=highlight_bad_intervals,
            supp_message=f"Intervals differ from the specified {interval} interval."
            + f" Unexpected: {bad_intervals}",
        )
    return data
