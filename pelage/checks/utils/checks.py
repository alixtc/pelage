"""Module containing the main checks for pelage.

Use the syntax `import pelage as plg` rather than `from pelage import checks`
"""

from typing import Dict, Optional, Tuple, Union

import polars as pl

from pelage.checks.utils.types import PolarsColumnType


def _has_sufficient_polars_version(version_number: str = "0.20.0") -> bool:
    required_version = tuple(map(int, (version_number.split("."))))
    polars_version = tuple(map(int, (pl.__version__.split("."))))
    return polars_version >= required_version


def _sanitize_column_inputs(
    columns: Optional[PolarsColumnType] = None,
) -> pl.Expr:
    """Ensure that input can be converted to a `pl.col()` expression"""
    if columns is None:
        return pl.all()
    elif isinstance(columns, pl.Expr):
        return columns
    else:
        return pl.col(columns)


def _format_ranges_by_columns(
    items: Dict[str, Union[float, Tuple[float, float]]],
) -> pl.DataFrame:
    ranges = {k: (v if isinstance(v, tuple) else (v, 1)) for k, v in items.items()}
    pl_ranges = pl.DataFrame(
        [(k, v[0], v[1]) for k, v in ranges.items()],
        schema=["column", "min_prop", "max_prop"],
        orient="row",
    )
    return pl_ranges


class PolarsAssertError(Exception):
    """Custom Error providing detailed information about the failed check.

    To investigate the last error in a jupyter notebook you can use:

    Examples
    --------
    >>> from pelage import PolarsAssertError # doctest: +SKIP
    >>> raise PolarsAssertError # doctest: +SKIP
    >>> import sys # doctest: +SKIP
    >>> error = sys.last_value # doctest: +SKIP
    >>> print(error) # prints the string representation # doctest: +SKIP
    >>> error.df # access the dataframe object # doctest: +SKIP

    Attributes
    ----------
    df : pl.DataFrame, optional,  by default pl.DataFrame()
        A subset of the original dataframe passed to the check function with a highlight
        on the values that caused the check to fail,
    supp_message : str, optional
        A human readable description of the check failure, and when available a possible
        way to solve the issue,
        by default ""
    """

    def __init__(self, df: pl.DataFrame | None = None, supp_message: str = "") -> None:
        self.supp_message = supp_message
        self.df = df if df is not None else pl.DataFrame()

    def __str__(self) -> str:
        base_message = "Error with the DataFrame passed to the check function:"

        if not self.df.is_empty():
            base_message = f"{self.df}\n{base_message}"

        return f"Details\n{base_message}\n--> {self.supp_message}"
