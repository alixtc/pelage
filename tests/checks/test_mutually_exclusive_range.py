import polars as pl
import pytest
from polars import testing

import pelage as plg


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_mutually_exclusive_ranges_allows_to_specify_low_and_high_bounds(
    frame: type[pl.DataFrame | pl.LazyFrame],
):
    given_df = frame(
        [
            [1, 2],
            [3, 4],
        ],
        schema=["a", "b"],
        orient="row",
    )
    when = given_df.pipe(plg.mutually_exclusive_ranges, low_bound="a", high_bound="b")
    testing.assert_frame_equal(given_df, when)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_mutually_exclusive_ranges_should_error_on_overlapping_intervals(
    frame: type[pl.DataFrame | pl.LazyFrame],
):
    given_df = frame(
        [
            [1, 3],
            [2, 4],
        ],
        schema=["a", "b"],
        orient="row",
    )
    with pytest.raises(plg.PolarsAssertError):
        given_df.pipe(plg.mutually_exclusive_ranges, low_bound="a", high_bound="b")


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_mutually_exclusive_ranges_should_return_both_overlapping_intervals_and_index(
    frame: type[pl.DataFrame | pl.LazyFrame],
):
    given_df = frame(
        [
            [1, 3],
            [2, 4],
            [5, 7],
            [6, 8],
            [9, 9],
        ],
        schema=["a", "b"],
        orient="row",
    )

    with pytest.raises(plg.PolarsAssertError) as err:
        given_df.pipe(plg.mutually_exclusive_ranges, low_bound="a", high_bound="b")

    expected = given_df.lazy().head(4).collect()
    result = err.value.df.drop("index")
    testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_mutually_exclusive_ranges_allows_to_group_by_anoterh_column(
    frame: type[pl.DataFrame | pl.LazyFrame],
):
    given_df = frame(
        [
            ["A", 1, 3],
            ["B", 2, 4],
        ],
        schema=["group", "a", "b"],
        orient="row",
    )
    when = given_df.pipe(
        plg.mutually_exclusive_ranges,
        low_bound="a",
        high_bound="b",
        group_by="group",
    )
    testing.assert_frame_equal(given_df, when)
