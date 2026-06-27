import polars as pl
import pytest
from polars import testing

import pelage as plg
from pelage.checks.not_null_proportion import _format_ranges_by_columns


@pytest.mark.parametrize(
    "given_df",
    [
        pl.DataFrame({"a": [1, 2, 3]}),
        pl.LazyFrame({"a": [1, 2, 3]}),
    ],
)
def test_not_null_proportion_accept_proportion_range_or_single_input(
    given_df: pl.DataFrame | pl.LazyFrame,
):
    when = given_df.pipe(plg.not_null_proportion, {"a": (0.1, 1.0)})
    testing.assert_frame_equal(given_df, when)

    when = given_df.pipe(plg.not_null_proportion, {"a": 0.1})
    testing.assert_frame_equal(given_df, when)


def test_not_null_proportion_accept_multiple_input():
    given_df = pl.DataFrame({"a": [1, None, None], "b": [1, 2, None]})
    when = given_df.pipe(plg.not_null_proportion, {"a": 0.1, "b": 0.1})
    testing.assert_frame_equal(given_df, when)


def test_not_null_proportion_accept_group_by_option():
    given_df = pl.DataFrame({"a": [1, None, None, 1], "group": ["A", "A", "B", "B"]})
    when = given_df.pipe(plg.not_null_proportion, {"a": 0.5}, group_by="group")
    testing.assert_frame_equal(given_df, when)


@pytest.mark.parametrize(
    "given_df",
    [
        pl.DataFrame({"a": [1, 1, None, None], "group": ["A", "A", "B", "B"]}),
        pl.LazyFrame({"a": [1, 1, None, None], "group": ["A", "A", "B", "B"]}),
    ],
)
def test_not_null_proportion_should_error_with_too_many_nulls_per_group(
    given_df: pl.DataFrame | pl.LazyFrame,
):
    with pytest.raises(plg.PolarsAssertError):
        given_df.pipe(plg.not_null_proportion, {"a": 0.5}, group_by="group")


@pytest.mark.parametrize(
    "given_df",
    [
        pl.DataFrame({"a": [1, None]}),
        pl.LazyFrame({"a": [1, None]}),
    ],
)
def test_not_null_proportion_errors_with_too_many_nulls(
    given_df: pl.DataFrame | pl.LazyFrame,
):
    with pytest.raises(plg.PolarsAssertError) as err:
        given_df.pipe(plg.not_null_proportion, {"a": 0.9})

    expected_df_columns = ["not_null_fraction", "min_prop", "max_prop"]
    assert all([col in err.value.df.columns for col in expected_df_columns])


def test_format_ranges_by_has_columns_and_min_max():
    items = {"a": 0.5, "b": (0.9, 0.95)}
    given_df = _format_ranges_by_columns(items)

    expected = pl.DataFrame(
        [
            ("a", 0.5, 1.0),
            ("b", 0.9, 0.95),
        ],
        schema=["column", "min_prop", "max_prop"],
        orient="row",
    )
    testing.assert_frame_equal(given_df, expected)
