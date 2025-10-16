from typing import Union

import polars as pl
import pytest
from polars import testing

import pelage as plg


@pytest.mark.parametrize(
    "given_df",
    [
        pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]}),
        pl.LazyFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]}),
    ],
)
def test_accepted_values(given_df: Union[pl.DataFrame, pl.LazyFrame]):
    items = {"a": [1, 2, 3], "b": ["a", "b", "c"]}
    when = given_df.pipe(plg.accepted_values, items)
    testing.assert_frame_equal(given_df, when)


def test_accepted_values_should_accept_pl_expr():
    items = {"a": [1, 2, 3], "b": ["a", "b", "c"]}
    given_df = pl.DataFrame(items)
    when = given_df.pipe(plg.accepted_values, {"^a$": [1, 2, 3]})
    testing.assert_frame_equal(given_df, when)


@pytest.mark.parametrize(
    "given_df",
    [
        pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]}),
        pl.LazyFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]}),
    ],
)
def test_accepted_values_should_error_on_out_of_range_values(
    given_df: Union[pl.DataFrame, pl.LazyFrame],
):
    items = {"a": [1, 2], "b": ["a", "b", "c"]}

    with pytest.raises(plg.PolarsAssertError) as err:
        given_df.pipe(plg.accepted_values, items)

    expected = pl.DataFrame({"a": [3]})
    testing.assert_frame_equal(err.value.df, expected)
