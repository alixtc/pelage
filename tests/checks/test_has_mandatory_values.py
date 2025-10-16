from typing import Union

import polars as pl
import pytest
from polars import testing

import pelage as plg


@pytest.mark.parametrize(
    "given_df",
    [
        pl.DataFrame({"a": [1, 2]}),
        pl.LazyFrame({"a": [1, 2]}),
    ],
)
def test_has_mandatory_values(given_df: Union[pl.DataFrame, pl.LazyFrame]):
    when = given_df.pipe(plg.has_mandatory_values, {"a": [1, 2]})
    testing.assert_frame_equal(given_df, when)


@pytest.mark.parametrize(
    "given_df",
    [
        pl.DataFrame({"a": [1, 2]}),
        pl.LazyFrame({"a": [1, 2]}),
    ],
)
def test_has_mandatory_values_should_error_on_missing_elements(
    given_df: Union[pl.DataFrame, pl.LazyFrame],
):
    with pytest.raises(plg.PolarsAssertError):
        given_df.pipe(plg.has_mandatory_values, {"a": [3]})


def test_has_mandatory_values_should_give_feedback_on_missing_values():
    given_df = pl.DataFrame({"a": [1, 1], "b": ["x", "y"]})
    with pytest.raises(plg.PolarsAssertError) as err:
        given_df.pipe(plg.has_mandatory_values, {"a": [1, 2], "b": ["s", "t"]})

    expected = {"a": [2], "b": ["s", "t"]}
    assert str(expected) in str(err.value)


@pytest.mark.parametrize(
    "given_df",
    [
        pl.DataFrame({"a": [1, 1, 1, 2], "group": ["G1", "G1", "G2", "G2"]}),
        pl.LazyFrame({"a": [1, 1, 1, 2], "group": ["G1", "G1", "G2", "G2"]}),
    ],
)
def test_has_mandatory_values_should_accept_group_by_option(
    given_df: Union[pl.DataFrame, pl.LazyFrame],
):
    when = given_df.pipe(plg.has_mandatory_values, {"a": [1]}, group_by="group")
    testing.assert_frame_equal(given_df, when)


@pytest.mark.parametrize(
    "given_df",
    [
        pl.DataFrame({"a": [1, 1, 1, 2], "group": ["G1", "G1", "G2", "G2"]}),
        pl.LazyFrame({"a": [1, 1, 1, 2], "group": ["G1", "G1", "G2", "G2"]}),
    ],
)
def test_has_mandatory_values_by_group_should_error_when_not_all_values_are_present(
    given_df: Union[pl.DataFrame, pl.LazyFrame],
):
    with pytest.raises(plg.PolarsAssertError):
        given_df.pipe(plg.has_mandatory_values, {"a": [1, 2]}, group_by="group")
