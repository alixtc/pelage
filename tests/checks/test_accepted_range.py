from typing import Type, Union

import polars as pl
import pytest
from polars import testing

import pelage as plg
import pelage.utils


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_accepted_range(frame: Type[Union[pl.DataFrame, pl.LazyFrame]]):
    given_df = frame({"a": [1, 2, 3]})
    when = given_df.pipe(plg.accepted_range, {"a": (1, 3)})
    testing.assert_frame_equal(given_df, when)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_accepted_range_is_compatible_with_is_between_syntax(
    frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
):
    given_df = frame({"a": ["b", "c"]})
    when = given_df.pipe(plg.accepted_range, {"a": (pl.lit("a"), pl.lit("d"), "right")})
    testing.assert_frame_equal(given_df, when)
    when = given_df.pipe(plg.accepted_range, {"a": (pl.lit("a"), pl.lit("d"), "left")})
    testing.assert_frame_equal(given_df, when)
    when = given_df.pipe(plg.accepted_range, {"a": (pl.lit("a"), pl.lit("d"), "both")})
    testing.assert_frame_equal(given_df, when)
    when = given_df.pipe(plg.accepted_range, {"a": (pl.lit("a"), pl.lit("d"), "none")})
    testing.assert_frame_equal(given_df, when)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_accepted_range_erros_when_values_are_out_of_range(
    frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
):
    given_df = frame({"a": [1, 2, 3]})
    with pytest.raises(plg.PolarsAssertError) as err:
        given_df.pipe(plg.accepted_range, {"a": (0, 2)})
    testing.assert_frame_equal(err.value.df, pl.DataFrame({"a": [3]}))


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_accepted_range_errors_on_two_different_ranges(
    frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
):
    given_df = frame({"a": [1, 2, 3], "b": [1, 2, 3]})
    with pytest.raises(plg.PolarsAssertError) as err:
        given_df.pipe(plg.accepted_range, {"a": (0, 2), "b": (2, 3)})

    expected = pl.DataFrame({"a": [1, 3], "b": [1, 3]})
    testing.assert_frame_equal(err.value.df, expected)
