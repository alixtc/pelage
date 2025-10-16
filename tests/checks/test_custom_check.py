from typing import Type, Union

import polars as pl
import pytest
from polars import testing

import pelage as plg


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_custom_checks_works_for_simple_filter(
    frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
):
    given_df = frame({"int": [1, 1, 1]})
    when = given_df.pipe(plg.custom_check, pl.col("int") == 1)
    testing.assert_frame_equal(given_df, when)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_custom_checks_errors_the_condition_returns_a_non_empty(
    frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
):
    given_df = frame({"int": [1, 2, 1]})
    with pytest.raises(plg.PolarsAssertError):
        given_df.pipe(plg.custom_check, pl.col("int") != 2)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_custom_checks_accept_over_clauses(
    frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
):
    given_df = frame({"a": [1, 1, 2, 2], "b": [1, 2, 3, 4]})
    when = given_df.pipe(plg.custom_check, pl.col("b").max().over("a") <= 4)
    testing.assert_frame_equal(given_df, when)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_custom_checks_should_select_only_affected_columns(
    frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
):
    given_df = frame(
        {
            "a": [1, 1, 2, 2],
            "b": [1, 2, 3, 4],
            "c": [1, 2, 3, 4],
        }
    )
    with pytest.raises(plg.PolarsAssertError) as err:
        given_df.pipe(plg.custom_check, pl.col("b").max().over("a") <= 3)

    assert {"a", "b"} == set(err.value.df.columns)
