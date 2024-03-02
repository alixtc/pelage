import polars as pl
import pytest
from polars import testing

from furcoat import checks


def test_polar_check_error_has_simple_message_when_used_directly():
    error = checks.PolarsCheckError()
    assert error.df.is_empty()
    assert "There was an improper value in the passed DataFrame:\n" == str(error)


def test_polar_check_error_should_accept_df_as_input():
    error = checks.PolarsCheckError(pl.DataFrame({"a": [1]}))
    testing.assert_frame_equal(error.df, pl.DataFrame({"a": [1]}))


def test_polar_check_error_should_have_clearer_error_message_with_dataframe_input():
    error = checks.PolarsCheckError(pl.DataFrame({"a": [1]}))
    assert "There was an improper value in the passed DataFrame:" in str(error)
    assert str(error.df) in str(error)


def test_has_no_nulls_returns_df_when_all_values_defined():
    given = pl.DataFrame({"a": [1, 2]})
    when = given.pipe(checks.has_no_nulls)
    testing.assert_frame_equal(given, when)


def test_has_no_nulls_throws_error_on_null_values():
    given = pl.DataFrame({"a": [1, None]})
    with pytest.raises(checks.PolarsCheckError):
        given.pipe(checks.has_no_nulls)


def test_unique_should_return_df_if_column_has_unique_values():
    given = pl.DataFrame({"a": [1, 2]})
    when = given.pipe(checks.unique, "a")
    testing.assert_frame_equal(given, when)


def test_unique_should_should_accept_list_and_polars_select():
    given = pl.DataFrame({"a": [1, 2]})
    when = given.pipe(checks.unique, ["a"])
    testing.assert_frame_equal(given, when)

    given = pl.DataFrame({"a": [1, 2]})
    when = given.pipe(checks.unique, pl.col("a"))
    testing.assert_frame_equal(given, when)


def test_unique_should_throw_error_on_duplicates():
    given = pl.DataFrame({"a": [1, 1, 2]})
    with pytest.raises(checks.PolarsCheckError) as err:
        given.pipe(checks.unique, "a")
    testing.assert_frame_equal(err.value.df, pl.DataFrame({"a": [1, 1]}))


def test_accepted_values():
    items = {"a": [1, 2, 3], "b": ["a", "b", "c"]}
    given = pl.DataFrame(items)
    when = given.pipe(checks.accepted_values, items)
    testing.assert_frame_equal(given, when)

    short_items = {"b": ["a", "b", "c"]}
    when = given.pipe(checks.accepted_values, short_items)
    testing.assert_frame_equal(given, when)


def test_accepted_values_should_error_on_out_of_range_values():
    items = {"a": [1, 2, 3], "b": ["a", "b", "c"]}
    given = pl.DataFrame(items)
    items = {"a": [1, 2], "b": ["a", "b", "c"]}

    with pytest.raises(checks.PolarsCheckError) as err:
        given.pipe(checks.accepted_values, items)
    assert err.value.df.shape == (1, 2)
