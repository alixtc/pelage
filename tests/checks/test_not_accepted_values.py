from typing import Union

import polars as pl
import pytest
from polars import testing

import pelage as plg


def test_not_accepted_values():
    given_df = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})
    when = given_df.pipe(plg.not_accepted_values, {"a": [4, 5]})
    testing.assert_frame_equal(given_df, when)


def test_not_accepted_values_should_accept_pl_expr():
    given_df = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})
    when = given_df.pipe(plg.not_accepted_values, {"^b$": ["d", ""]})
    testing.assert_frame_equal(given_df, when)


@pytest.mark.parametrize(
    "given_df",
    [
        pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]}),
        pl.LazyFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]}),
    ],
)
def test_not_accepted_values_should_error_on_forbidden_values(
    given_df: Union[pl.DataFrame, pl.LazyFrame],
):
    items = {"a": [1], "b": ["a", "c"]}

    with pytest.raises(plg.PolarsAssertError):
        given_df.pipe(plg.not_accepted_values, items)
