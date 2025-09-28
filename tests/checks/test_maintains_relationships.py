from typing import Type, Union

import polars as pl
import pytest
from polars import testing

import pelage as plg


@pytest.mark.parametrize(
    "first_frame,second_frame",
    [
        (pl.DataFrame, pl.DataFrame),
        (pl.LazyFrame, pl.LazyFrame),
        (pl.DataFrame, pl.LazyFrame),
        (pl.LazyFrame, pl.DataFrame),
    ],
)
def test_maintains_relationships(
    first_frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
    second_frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
):
    initial_df = first_frame({"a": ["a", "b"]})
    final_df = second_frame({"a": ["a", "b"]})
    when = final_df.pipe(plg.maintains_relationships, initial_df, "a")
    testing.assert_frame_equal(when, final_df)


@pytest.mark.parametrize(
    "first_frame,second_frame",
    [
        (pl.DataFrame, pl.DataFrame),
        (pl.LazyFrame, pl.LazyFrame),
        (pl.DataFrame, pl.LazyFrame),
        (pl.LazyFrame, pl.DataFrame),
    ],
)
def test_maintains_relationships_accepts_multiple_columns_as_input(
    first_frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
    second_frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
):
    initial_df = first_frame({"col1": ["a", "b"], "col2": [1, 2]})
    final_df = second_frame({"col1": ["a", "b"], "col2": [1, 2]})
    when = final_df.pipe(plg.maintains_relationships, initial_df, ["col1", "col2"])
    testing.assert_frame_equal(when, final_df)


@pytest.mark.parametrize(
    "first_frame,second_frame",
    [
        (pl.DataFrame, pl.DataFrame),
        (pl.LazyFrame, pl.LazyFrame),
        (pl.DataFrame, pl.LazyFrame),
        (pl.LazyFrame, pl.DataFrame),
    ],
)
def test_maintains_relationships_should_errors_if_some_values_are_dropped(
    first_frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
    second_frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
):
    initial_df = first_frame({"a": ["a", "b"]})
    final_df = second_frame({"a": ["a"]})
    with pytest.raises(plg.PolarsAssertError):
        final_df.pipe(plg.maintains_relationships, initial_df, "a")


@pytest.mark.parametrize(
    "first_frame,second_frame",
    [
        (pl.DataFrame, pl.DataFrame),
        (pl.LazyFrame, pl.LazyFrame),
        (pl.DataFrame, pl.LazyFrame),
        (pl.LazyFrame, pl.DataFrame),
    ],
)
def test_maintains_relationships_should_specify_some_changing_values(
    first_frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
    second_frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
):
    initial_df = first_frame({"a": ["a", "b", "c"]})
    final_df = second_frame({"a": ["a"]})
    with pytest.raises(plg.PolarsAssertError) as err:
        final_df.pipe(plg.maintains_relationships, initial_df, "a")
    assert "Some values were removed from col 'a', see above!" in str(err.value)

    initial_df = pl.DataFrame({"a": ["a"]})
    final_df = pl.DataFrame({"a": ["a", "b", "c"]})
    with pytest.raises(plg.PolarsAssertError) as err:
        final_df.pipe(plg.maintains_relationships, initial_df, "a")
    assert "Some values were added to col 'a', see above!" in str(err.value)
