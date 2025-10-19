import polars as pl
import pytest
from polars import testing

import pelage as plg
import pelage.utils


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_is_shape(frame: type[pl.DataFrame | pl.LazyFrame]):
    given_df = frame({"a": [1, 2, 3], "b": ["a", "b", "c"]})
    when = given_df.pipe(plg.has_shape, (3, 2))
    testing.assert_frame_equal(given_df, when)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_is_shape_should_error_when_expected_shape_has_only_nones(
    frame: type[pl.DataFrame | pl.LazyFrame],
):
    given_df = frame(
        {
            "a": [1, 2, 3],
            "b": ["a", "b", "c"],
        }
    )
    with pytest.raises(ValueError):
        given_df.pipe(plg.has_shape, (None, None))


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
@pytest.mark.parametrize("expected_shape", [(3, 2), (3, None), (None, 2)])
def test_is_shape_should_accept_none_values_to_facilitate_comparison(
    frame: type[pl.DataFrame | pl.LazyFrame], expected_shape
):
    given_df = frame(
        {
            "a": [1, 2, 3],
            "b": ["a", "b", "c"],
        }
    )
    when = given_df.pipe(plg.has_shape, expected_shape)
    testing.assert_frame_equal(given_df, when)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
@pytest.mark.parametrize("expected_shape", [(4, 2), (4, None), (None, 3)])
def test_is_shape_should__error_with_wrong_expected_dimensions(
    frame: type[pl.DataFrame | pl.LazyFrame], expected_shape
):
    given_df = frame(
        {
            "a": [1, 2, 3],
            "b": ["a", "b", "c"],
        }
    )
    with pytest.raises(plg.PolarsAssertError):
        given_df.pipe(plg.has_shape, expected_shape)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_is_shape_should_accept_group_by_option(
    frame: type[pl.DataFrame | pl.LazyFrame],
):
    given_df = frame(
        {
            "a": [1, 2, 3],
            "b": ["a", "b", "c"],
        }
    )

    expected = given_df.pipe(plg.has_shape, (1, None), group_by="b")
    testing.assert_frame_equal(given_df, expected)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_is_shape_should_should_error_when_row_count_per_group_does_not_match(
    frame: type[pl.DataFrame | pl.LazyFrame],
):
    given_df = frame(
        {
            "a": [1, 2, 3],
            "b": ["a", "b", "b"],
        }
    )
    with pytest.raises(plg.PolarsAssertError):
        given_df.pipe(plg.has_shape, (1, None), group_by="b")
