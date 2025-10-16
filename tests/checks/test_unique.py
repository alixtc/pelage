from typing import Union

import polars as pl
import pytest
from polars import testing

import pelage as plg


@pytest.mark.parametrize(
    "given_df", [pl.DataFrame({"a": [1, 2]}), pl.LazyFrame({"a": [1, 2]})]
)
def test_unique_should_return_df_if_column_has_unique_values(
    given_df: Union[pl.DataFrame, pl.LazyFrame],
):
    when = given_df.pipe(plg.unique, "a")
    testing.assert_frame_equal(given_df, when)


def test_unique_should_should_accept_list_and_polars_select():
    given_df = pl.DataFrame({"a": [1, 2]})
    when = given_df.pipe(plg.unique, ["a"])
    testing.assert_frame_equal(given_df, when)

    given_df = pl.DataFrame({"a": [1, 2]})
    when = given_df.pipe(plg.unique, pl.col("a"))
    testing.assert_frame_equal(given_df, when)


def test_unique_should_throw_error_on_duplicates():
    given_df = pl.DataFrame({"a": [1, 1, 2]})
    with pytest.raises(plg.PolarsAssertError) as err:
        given_df.pipe(plg.unique, "a")
    testing.assert_frame_equal(err.value.df, pl.DataFrame({"a": [1, 1]}))
