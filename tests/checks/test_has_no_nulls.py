from typing import Union

import polars as pl
import pytest
from polars import testing

import pelage as plg


@pytest.mark.parametrize(
    "given_df", [pl.DataFrame({"a": [1, 2]}), pl.LazyFrame({"a": [1, 2]})]
)
def test_has_no_nulls_returns_df_when_all_values_defined(
    given_df: Union[pl.DataFrame, pl.LazyFrame],
):
    when = given_df.pipe(plg.has_no_nulls)
    testing.assert_frame_equal(given_df, when)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_has_no_nulls_throws_error_on_null_values(frame):
    given_df = frame({"a": [1, None]})
    with pytest.raises(plg.PolarsAssertError):
        given_df.pipe(plg.has_no_nulls)


@pytest.mark.parametrize(
    "given_df",
    [
        pl.DataFrame({"a": [1, None]}),
        pl.LazyFrame({"a": [1, None]}),
    ],
)
def test_has_no_nulls_indicates_columns_with_nulls_in_error_message(
    given_df: Union[pl.DataFrame, pl.LazyFrame],
):
    expected = pl.DataFrame(
        {"column": ["a"], "null_count": [1]},
        schema={"column": pl.Utf8, "null_count": pl.UInt32},
    )
    with pytest.raises(plg.PolarsAssertError) as err:
        given_df.pipe(plg.has_no_nulls)
    testing.assert_frame_equal(err.value.df, expected)
