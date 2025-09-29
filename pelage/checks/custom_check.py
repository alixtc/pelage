import polars as pl

from pelage.types import PolarsAssertError, PolarsLazyOrDataFrame


def custom_check(
    data: PolarsLazyOrDataFrame, expression: pl.Expr
) -> PolarsLazyOrDataFrame:
    """Use custom Polars expression to check the DataFrame, based on `.filter()`.

    The expression when used through the dataframe method `.filter()` should return an
    empty dataframe.
    This expression should express the requierement for values that are not wanted
    in the dataframe. For instance, if a column should not contain the value `4`,
    use the expression `pl.col("column") != 4`.

    Analog to dbt-utils fonction: `expression_is_true`

    Parameters
    ----------
    data : PolarsLazyOrDataFrame
        Polars DataFrame or LazyFrame containing data to check.
    expression : pl.Expr
        Polar Expression that can be passed to the `.filter()` method. As describe
        above, use an expression that should keep forbidden values when passed to the
        filter

    Returns
    -------
    PolarsLazyOrDataFrame
        The original polars DataFrame or LazyFrame when the check passes.

    Examples
    --------

    >>> import polars as pl
    >>> import pelage as plg
    >>> df = pl.DataFrame({"a": [1, 2, 3]})
    >>> df.pipe(plg.custom_check, pl.col("a") < 4)
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

    >>> df.pipe(plg.custom_check, pl.col("a") != 3)
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
    --> Unexpected data in `Custom Check`: [(col("a")) != (dyn int: 3)]
    """
    columns_in_expr = set(expression.meta.root_names())
    bad_data = data.lazy().select(columns_in_expr).filter(expression.not_()).collect()

    if not bad_data.is_empty():
        raise PolarsAssertError(
            df=bad_data,
            supp_message=f"Unexpected data in `Custom Check`: {str(expression)}",
        )
    return data
