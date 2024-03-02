import polars as pl
import pytest
from polars import testing

from furcoat import checks


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
    given = pl.DataFrame({"a": [1, 1]})
    with pytest.raises(checks.PolarsCheckError):
        given.pipe(checks.unique, "a")


def test_accepted_values():
    items = {"a": [1, 2, 3], "b": ["a", "b", "c"]}
    given = pl.DataFrame(items)
    when = given.pipe(checks.accepted_values, items)
    testing.assert_frame_equal(given, when)

    short_items = {"b": ["a", "b", "c"]}
    when = given.pipe(checks.accepted_values, short_items)
    testing.assert_frame_equal(given, when)
    # items["A"] = [1, 2]
    # with pytest.raises(AssertionError):
    #     ck.has_vals_within_set(df, items)


def test_accepted_values_should_error_on_out_of_range_values():
    items = {"a": [1, 2, 3], "b": ["a", "b", "c"]}
    given = pl.DataFrame(items)

    items = {
        "a": [
            1,
            2,
        ],
        "b": ["a", "b", "c"],
    }
    with pytest.raises(checks.PolarsCheckError):
        given.pipe(checks.accepted_values, items)
