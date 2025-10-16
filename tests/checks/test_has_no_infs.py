from typing import Union

import polars as pl
import pytest
from polars import testing

import pelage as plg


@pytest.mark.parametrize(
    "given_df", [pl.DataFrame({"a": [1, 2]}), pl.LazyFrame({"a": [1, 2]})]
)
def test_has_no_infs_returns_df_when_all_values_defined(
    given_df: Union[pl.DataFrame, pl.LazyFrame],
):
    when = given_df.pipe(plg.has_no_infs)
    testing.assert_frame_equal(given_df, when)


@pytest.mark.parametrize(
    "given_df",
    [
        pl.DataFrame({"a": [1.0, None, float("inf")]}),
        pl.LazyFrame({"a": [1.0, None, float("inf")]}),
    ],
)
def test_has_no_infs_throws_error_on_inf_values(
    given_df: Union[pl.DataFrame, pl.LazyFrame],
):
    with pytest.raises(plg.PolarsAssertError) as err:
        given_df.pipe(plg.has_no_infs)
    expected = pl.DataFrame({"a": [float("inf")]})
    testing.assert_frame_equal(err.value.df, expected)
