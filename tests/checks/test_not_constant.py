from typing import Union

import polars as pl
import pytest
from polars import testing

import pelage as plg


def test_not_constant():
    given_df = pl.DataFrame({"a": [1, 2]})
    when = given_df.pipe(plg.not_constant, "a")
    testing.assert_frame_equal(given_df, when)


@pytest.mark.parametrize(
    "given_df", [pl.DataFrame({"b": [1, 1]}), pl.LazyFrame({"b": [1, 1]})]
)
def test_not_constant_throws_error_on_constant_columns(
    given_df: Union[pl.DataFrame, pl.LazyFrame],
):
    with pytest.raises(plg.PolarsAssertError):
        given_df.pipe(plg.not_constant, "b")


def test_not_constant_accept_different_types_of_input():
    given_df = pl.DataFrame({"a": [1, 2], "b": ["A", "B"]})
    when = given_df.pipe(plg.not_constant, ["a", "b"])
    testing.assert_frame_equal(given_df, when)

    given_df = pl.DataFrame({"a": [1, 2], "b": ["A", "B"]})
    when = given_df.pipe(plg.not_constant, pl.col("a"))
    testing.assert_frame_equal(given_df, when)


def test_not_constant_accepts_group_by_option():
    given_df = pl.DataFrame(
        {
            "a": [1, 2, 1, 2],
            "b": ["A", "A", "B", "B"],
        }
    )
    when = given_df.pipe(plg.not_constant, "a", group_by="b")
    testing.assert_frame_equal(given_df, when)


def test_not_constant_should_error_when_values_for_one_group_are_constant():
    given_df = pl.DataFrame(
        {
            "a": [1, 2, 1, 1],
            "b": ["A", "A", "B", "B"],
        }
    )
    with pytest.raises(plg.PolarsAssertError) as err:
        given_df.pipe(plg.not_constant, "a", group_by="b")

    assert "b" in err.value.df.columns
