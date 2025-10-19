import polars as pl
import pytest
from polars import testing

import pelage as plg


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_column_is_within_n_std_accepts_tuple_args(
    frame: type[pl.DataFrame | pl.LazyFrame],
):
    given_df = frame({"a": [1, 2, 2, 1]})

    when = given_df.pipe(plg.column_is_within_n_std, ("a", 2))

    testing.assert_frame_equal(given_df, when)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_column_is_within_n_std_shoud_error_on_outliers(
    frame: type[pl.DataFrame | pl.LazyFrame],
):
    given_df = frame({"a": list(range(0, 10)) + [5000]})

    with pytest.raises(plg.PolarsAssertError):
        given_df.pipe(plg.column_is_within_n_std, ("a", 2))


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_column_is_within_n_std_accepts_list_of_tuple_args(
    frame: type[pl.DataFrame | pl.LazyFrame],
):
    given_df = frame(
        {
            "a": list(range(0, 11)),
            "b": list(range(0, 11)),
            "c": list(range(0, 10)) + [5000],
        }
    )

    when = given_df.pipe(plg.column_is_within_n_std, ("a", 2), ("b", 3))
    testing.assert_frame_equal(given_df, when)

    with pytest.raises(plg.PolarsAssertError):
        given_df.pipe(plg.column_is_within_n_std, ("b", 2), ("c", 2))
