import polars as pl
import pytest
from polars import testing

import pelage as plg


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_has_dtypes_accepts_dict(frame: type[pl.DataFrame | pl.LazyFrame]):
    given_df = frame({"a": [1, 2, 3]})
    when = given_df.pipe(plg.has_dtypes, {"a": pl.Int64})
    testing.assert_frame_equal(given_df, when)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_has_dtypes_accepts_dict_should_error_if_types_are_not_matched(
    frame: type[pl.DataFrame | pl.LazyFrame],
):
    given_df = frame({"a": [1, 2, 3]})
    with pytest.raises(plg.PolarsAssertError) as err:
        given_df.pipe(plg.has_dtypes, {"b": pl.Int64})

    assert "Dtype check, some expected columns are missing:" in str(err.value)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_has_dtypes_accepts_dict_should_error_if_columns_are_missing(
    frame: type[pl.DataFrame | pl.LazyFrame],
):
    given_df = frame({"a": [1, 2, 3]})
    with pytest.raises(plg.PolarsAssertError):
        given_df.pipe(plg.has_dtypes, {"a": pl.Utf8})


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_has_dtypes_accepts_dict_should_indicate_mismatched_dtypes(
    frame: type[pl.DataFrame | pl.LazyFrame],
):
    given_df = frame({"a": [1, 2, 3]})
    with pytest.raises(plg.PolarsAssertError) as err:
        given_df.pipe(plg.has_dtypes, {"a": pl.Utf8})

    base_message = "Some columns don't have the expected type:\n"
    assert base_message in str(err.value)
    assert "column='a'" in str(err.value)
