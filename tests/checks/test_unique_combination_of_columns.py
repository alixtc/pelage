import polars as pl
import pytest
from polars import testing

import pelage as plg
from pelage import types


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_unique_combination_of_columns(frame: type[pl.DataFrame | pl.LazyFrame]):
    given_df = frame({"a": ["a", "b"]})
    when = given_df.pipe(plg.unique_combination_of_columns, "a")
    testing.assert_frame_equal(given_df, when)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_unique_combination_of_columns_use_all_columns_by_default(
    frame: type[pl.DataFrame | pl.LazyFrame],
):
    given_df = frame({"a": ["a", "b"]})
    when = given_df.pipe(plg.unique_combination_of_columns)
    testing.assert_frame_equal(given_df, when)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_unique_combination_of_columns_accepts_list_as_input(
    frame: type[pl.DataFrame | pl.LazyFrame],
):
    given_df = frame({"a": ["a", "a"], "b": [1, 2]})
    when = given_df.pipe(plg.unique_combination_of_columns, ["a", "b"])
    testing.assert_frame_equal(given_df, when)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_unique_combination_of_columns_should_err_for_non_unicity(
    frame: type[pl.DataFrame | pl.LazyFrame],
):
    given_df = frame({"a": ["a", "a"]})
    with pytest.raises(plg.PolarsAssertError):
        given_df.pipe(plg.unique_combination_of_columns, "a")


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_unique_combination_of_columns_base_error_message_format(
    frame: type[pl.DataFrame | pl.LazyFrame],
):
    given_df = frame({"a": ["a", "a"]})
    with pytest.raises(plg.PolarsAssertError) as err:
        given_df.pipe(plg.unique_combination_of_columns, "a")

    base_message = "Some combinations of columns are not unique."
    assert base_message in str(err.value)


@pytest.mark.parametrize(
    "columns",
    [
        "a",
        ["a", "b"],
        pl.Utf8,
        None,
    ],
)
def test_unique_combination_of_columns_error_message_format(
    columns: types.PolarsColumnType,
):
    given_df = pl.DataFrame({"a": ["a", "a"], "b": [1, 1]})
    with pytest.raises(plg.PolarsAssertError) as err:
        given_df.pipe(plg.unique_combination_of_columns, columns)

    base_message = "Some combinations of columns are not unique."

    assert base_message in str(err.value)
