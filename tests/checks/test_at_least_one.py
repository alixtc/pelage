import polars as pl
import pytest
from polars import testing

import pelage as plg


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_at_least_one_should_return_df_is_one_value_is_not_null(
    frame: type[pl.DataFrame | pl.LazyFrame],
):
    given_df = frame({"a": [None, 1]})
    when = given_df.pipe(plg.at_least_one)
    testing.assert_frame_equal(given_df, when)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_at_least_one_should_error_if_all_values_are_nulls(
    frame: type[pl.DataFrame | pl.LazyFrame],
):
    given_df = frame({"a": [None, None]})
    with pytest.raises(plg.PolarsAssertError):
        given_df.pipe(plg.at_least_one)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_at_least_one_should_accept_column_selection(
    frame: type[pl.DataFrame | pl.LazyFrame],
):
    given_df = frame({"a": [None, None], "b": [1, None]})

    when = given_df.pipe(plg.at_least_one, "b")
    testing.assert_frame_equal(given_df, when)

    with pytest.raises(plg.PolarsAssertError):
        given_df.pipe(plg.at_least_one)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_at_least_one_should_accept_group_by_option(
    frame: type[pl.DataFrame | pl.LazyFrame],
):
    given_df = frame({"a": [None, 1, None, 2], "group": ["G1", "G1", "G2", "G2"]})
    when = given_df.pipe(plg.at_least_one, group_by="group")
    testing.assert_frame_equal(given_df, when)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_at_least_one_should_error_when_only_null_for_given_group(
    frame: type[pl.DataFrame | pl.LazyFrame],
):
    given_df = frame(
        {
            "a": [None, None, None, 2],
            "group": ["G1", "G1", "G2", "G2"],
        }
    )
    with pytest.raises(plg.PolarsAssertError):
        given_df.pipe(plg.at_least_one, "a", group_by="group")
