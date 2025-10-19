import polars as pl
import pytest
from polars import testing

import pelage as plg


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_has_columns(frame: type[pl.DataFrame | pl.LazyFrame]):
    given_df = frame({"a": [1, 2, 3]})
    when = given_df.pipe(plg.has_columns, "a")
    testing.assert_frame_equal(given_df, when)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_has_columns_accepts_lists(frame: type[pl.DataFrame | pl.LazyFrame]):
    given_df = frame({"a": [1, 2, 3], "b": ["a", "b", "c"]})
    when = given_df.pipe(plg.has_columns, ["a", "b"])
    testing.assert_frame_equal(given_df, when)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_has_columns_should_error_on_missing_column(
    frame: type[pl.DataFrame | pl.LazyFrame],
):
    given_df = frame({"a": [1, 2, 3]})
    with pytest.raises(plg.PolarsAssertError):
        given_df.pipe(plg.has_columns, "b")

    given_df = frame({"a": [1, 2, 3], "b": ["a", "b", "c"]})
    with pytest.raises(plg.PolarsAssertError):
        given_df.pipe(plg.has_columns, ["a", "b", "c"])
