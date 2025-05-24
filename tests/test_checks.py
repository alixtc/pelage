import datetime
from textwrap import dedent
from typing import Type, Union

import polars as pl
import pytest
from polars import testing

import pelage as plg
from pelage import checks


def test_dataframe_error_message_format():
    data = pl.DataFrame({"a": ["a", "b"], "b": [1, 2]})
    message = "Additional message"

    expected_message = """
    Details
    shape: (2, 2)
    ┌─────┬─────┐
    │ a   ┆ b   │
    │ --- ┆ --- │
    │ str ┆ i64 │
    ╞═════╪═════╡
    │ a   ┆ 1   │
    │ b   ┆ 2   │
    └─────┴─────┘
    Error with the DataFrame passed to the check function:
    --> Additional message
    """
    formatted_msg = str(plg.PolarsAssertError(data, message))
    assert dedent(expected_message).strip() == formatted_msg


def test_dataframe_error_message_format_accept_only_message():
    message = "Additional message"

    expected_message = """
    Details
    Error with the DataFrame passed to the check function:
    --> Additional message
    """

    formatted_msg = str(plg.PolarsAssertError(supp_message=message))
    assert dedent(expected_message).strip() == formatted_msg


def test_dataframe_error_message_format_accepts_no_arguments():
    expected_message = """
    Details
    Error with the DataFrame passed to the check function:
    -->
    """

    formatted_msg = str(plg.PolarsAssertError())
    assert formatted_msg == (dedent(expected_message).strip() + " ")


@pytest.mark.parametrize(
    "input,expected",
    [
        ("a", pl.col("a")),
        ("b", pl.col("b")),
        (pl.col("b"), pl.col("b")),
        (None, pl.all()),
    ],
)
def test_sanitize_column_inputs_works_with_(input, expected):
    given_df = checks._sanitize_column_inputs(input)
    assert str(given_df) == str(expected)


def test_sanitize_column_inputs_for_type_checker_pl_dtypes():
    given_df = checks._sanitize_column_inputs(pl.Int64)
    assert str(given_df) == str(pl.col(pl.Int64))


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_is_shape(frame: Type[Union[pl.DataFrame, pl.LazyFrame]]):
    given_df = frame({"a": [1, 2, 3], "b": ["a", "b", "c"]})
    when = given_df.pipe(plg.has_shape, (3, 2))
    testing.assert_frame_equal(given_df, when)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_is_shape_should_error_when_expected_shape_has_only_nones(
    frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
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
    frame: Type[Union[pl.DataFrame, pl.LazyFrame]], expected_shape
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
    frame: Type[Union[pl.DataFrame, pl.LazyFrame]], expected_shape
):
    given_df = frame(
        {
            "a": [1, 2, 3],
            "b": ["a", "b", "c"],
        }
    )
    with pytest.raises(checks.PolarsAssertError):
        given_df.pipe(plg.has_shape, expected_shape)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_is_shape_should_accept_group_by_option(
    frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
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
    frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
):
    given_df = pl.DataFrame(
        {
            "a": [1, 2, 3],
            "b": ["a", "b", "b"],
        }
    )
    with pytest.raises(checks.PolarsAssertError):
        given_df.pipe(plg.has_shape, (1, None), group_by="b")


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_has_columns(frame: Type[Union[pl.DataFrame, pl.LazyFrame]]):
    given_df = frame({"a": [1, 2, 3]})
    when = given_df.pipe(plg.has_columns, "a")
    testing.assert_frame_equal(given_df, when)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_has_columns_accepts_lists(frame: Type[Union[pl.DataFrame, pl.LazyFrame]]):
    given_df = frame({"a": [1, 2, 3], "b": ["a", "b", "c"]})
    when = given_df.pipe(plg.has_columns, ["a", "b"])
    testing.assert_frame_equal(given_df, when)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_has_columns_should_error_on_missing_column(
    frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
):
    given_df = frame({"a": [1, 2, 3]})
    with pytest.raises(plg.PolarsAssertError):
        given_df.pipe(plg.has_columns, "b")

    given_df = frame({"a": [1, 2, 3], "b": ["a", "b", "c"]})
    with pytest.raises(plg.PolarsAssertError):
        given_df.pipe(plg.has_columns, ["a", "b", "c"])


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_has_dtypes_accepts_dict(frame: Type[Union[pl.DataFrame, pl.LazyFrame]]):
    given_df = frame({"a": [1, 2, 3]})
    when = given_df.pipe(plg.has_dtypes, {"a": pl.Int64})
    testing.assert_frame_equal(given_df, when)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_has_dtypes_accepts_dict_should_error_if_types_are_not_matched(
    frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
):
    given_df = frame({"a": [1, 2, 3]})
    with pytest.raises(plg.PolarsAssertError) as err:
        given_df.pipe(plg.has_dtypes, {"b": pl.Int64})

    assert "Dtype check, some expected columns are missing:" in str(err.value)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_has_dtypes_accepts_dict_should_error_if_columns_are_missing(
    frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
):
    given_df = frame({"a": [1, 2, 3]})
    with pytest.raises(plg.PolarsAssertError):
        given_df.pipe(plg.has_dtypes, {"a": pl.Utf8})


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_has_dtypes_accepts_dict_should_indicate_mismatched_dtypes(
    frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
):
    given_df = frame({"a": [1, 2, 3]})
    with pytest.raises(plg.PolarsAssertError) as err:
        given_df.pipe(plg.has_dtypes, {"a": pl.Utf8})

    base_message = "Some columns don't have the expected type:\n"
    assert base_message in str(err.value)
    assert "column='a'" in str(err.value)


def test_is_shape_should_error_on_wrong_shape():
    given_df = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})
    with pytest.raises(plg.PolarsAssertError):
        given_df.pipe(plg.has_shape, (2, 2))


@pytest.mark.parametrize(
    "given_df", [pl.DataFrame({"a": [1, 2]}), pl.LazyFrame({"a": [1, 2]})]
)
def test_has_no_nulls_returns_df_when_all_values_defined(
    given_df: Union[pl.DataFrame, pl.LazyFrame],
):
    when = given_df.pipe(plg.has_no_nulls)
    testing.assert_frame_equal(given_df, when)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_has_no_nulls_throws_error_on_null_values(frame):
    given_df = frame({"a": [1, None]})
    with pytest.raises(plg.PolarsAssertError):
        given_df.pipe(plg.has_no_nulls)


@pytest.mark.parametrize(
    "given_df",
    [
        pl.DataFrame({"a": [1, None]}),
        pl.LazyFrame({"a": [1, None]}),
    ],
)
def test_has_no_nulls_indicates_columns_with_nulls_in_error_message(
    given_df: Union[pl.DataFrame, pl.LazyFrame],
):
    expected = pl.DataFrame(
        {"column": ["a"], "null_count": [1]},
        schema={"column": pl.Utf8, "null_count": pl.UInt32},
    )
    with pytest.raises(plg.PolarsAssertError) as err:
        given_df.pipe(plg.has_no_nulls)
    testing.assert_frame_equal(err.value.df, expected)


@pytest.mark.parametrize(
    "given_df", [pl.DataFrame({"a": [1, 2]}), pl.LazyFrame({"a": [1, 2]})]
)
def test_has_no_infs_returns_df_when_all_values_defined(
    given_df: Union[pl.DataFrame, pl.LazyFrame],
):
    when = given_df.pipe(plg.has_no_infs)
    testing.assert_frame_equal(given_df, when)


@pytest.mark.parametrize(
    "given_df",
    [
        pl.DataFrame({"a": [1.0, None, float("inf")]}),
        pl.LazyFrame({"a": [1.0, None, float("inf")]}),
    ],
)
def test_has_no_infs_throws_error_on_inf_values(
    given_df: Union[pl.DataFrame, pl.LazyFrame],
):
    with pytest.raises(plg.PolarsAssertError) as err:
        given_df.pipe(plg.has_no_infs)
    expected = pl.DataFrame({"a": [float("inf")]})
    testing.assert_frame_equal(err.value.df, expected)


@pytest.mark.parametrize(
    "given_df", [pl.DataFrame({"a": [1, 2]}), pl.LazyFrame({"a": [1, 2]})]
)
def test_unique_should_return_df_if_column_has_unique_values(
    given_df: Union[pl.DataFrame, pl.LazyFrame],
):
    when = given_df.pipe(plg.unique, "a")
    testing.assert_frame_equal(given_df, when)


def test_unique_should_should_accept_list_and_polars_select():
    given_df = pl.DataFrame({"a": [1, 2]})
    when = given_df.pipe(plg.unique, ["a"])
    testing.assert_frame_equal(given_df, when)

    given_df = pl.DataFrame({"a": [1, 2]})
    when = given_df.pipe(plg.unique, pl.col("a"))
    testing.assert_frame_equal(given_df, when)


def test_unique_should_throw_error_on_duplicates():
    given_df = pl.DataFrame({"a": [1, 1, 2]})
    with pytest.raises(plg.PolarsAssertError) as err:
        given_df.pipe(plg.unique, "a")
    testing.assert_frame_equal(err.value.df, pl.DataFrame({"a": [1, 1]}))


def test_not_constant():
    given_df = pl.DataFrame({"a": [1, 2]})
    when = given_df.pipe(plg.not_constant, "a")
    testing.assert_frame_equal(given_df, when)


@pytest.mark.parametrize(
    "given_df", [pl.DataFrame({"b": [1, 1]}), pl.LazyFrame({"b": [1, 1]})]
)
def test_not_constant_throws_error_on_constant_columns(
    given_df: Union[pl.DataFrame, pl.LazyFrame],
):
    with pytest.raises(plg.PolarsAssertError):
        given_df.pipe(plg.not_constant, "b")


def test_not_constant_accept_different_types_of_input():
    given_df = pl.DataFrame({"a": [1, 2], "b": ["A", "B"]})
    when = given_df.pipe(plg.not_constant, ["a", "b"])
    testing.assert_frame_equal(given_df, when)

    given_df = pl.DataFrame({"a": [1, 2], "b": ["A", "B"]})
    when = given_df.pipe(plg.not_constant, pl.col("a"))
    testing.assert_frame_equal(given_df, when)


def test_not_constant_accepts_group_by_option():
    given_df = pl.DataFrame(
        {
            "a": [1, 2, 1, 2],
            "b": ["A", "A", "B", "B"],
        }
    )
    when = given_df.pipe(plg.not_constant, "a", group_by="b")
    testing.assert_frame_equal(given_df, when)


def test_not_constant_should_error_when_values_for_one_group_are_constant():
    given_df = pl.DataFrame(
        {
            "a": [1, 2, 1, 1],
            "b": ["A", "A", "B", "B"],
        }
    )
    with pytest.raises(plg.PolarsAssertError) as err:
        given_df.pipe(plg.not_constant, "a", group_by="b")

    assert "b" in err.value.df.columns


@pytest.mark.parametrize(
    "given_df",
    [
        pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]}),
        pl.LazyFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]}),
    ],
)
def test_accepted_values(given_df: Union[pl.DataFrame, pl.LazyFrame]):
    items = {"a": [1, 2, 3], "b": ["a", "b", "c"]}
    when = given_df.pipe(plg.accepted_values, items)
    testing.assert_frame_equal(given_df, when)


def test_accepted_values_should_accept_pl_expr():
    items = {"a": [1, 2, 3], "b": ["a", "b", "c"]}
    given_df = pl.DataFrame(items)
    when = given_df.pipe(plg.accepted_values, {"^a$": [1, 2, 3]})
    testing.assert_frame_equal(given_df, when)


@pytest.mark.parametrize(
    "given_df",
    [
        pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]}),
        pl.LazyFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]}),
    ],
)
def test_accepted_values_should_error_on_out_of_range_values(
    given_df: Union[pl.DataFrame, pl.LazyFrame],
):
    items = {"a": [1, 2], "b": ["a", "b", "c"]}

    with pytest.raises(plg.PolarsAssertError) as err:
        given_df.pipe(plg.accepted_values, items)

    expected = pl.DataFrame({"a": [3]})
    testing.assert_frame_equal(err.value.df, expected)


def test_not_accepted_values():
    given_df = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})
    when = given_df.pipe(plg.not_accepted_values, {"a": [4, 5]})
    testing.assert_frame_equal(given_df, when)


def test_not_accepted_values_should_accept_pl_expr():
    given_df = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})
    when = given_df.pipe(plg.not_accepted_values, {"^b$": ["d", ""]})
    testing.assert_frame_equal(given_df, when)


@pytest.mark.parametrize(
    "given_df",
    [
        pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]}),
        pl.LazyFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]}),
    ],
)
def test_not_accepted_values_should_error_on_forbidden_values(
    given_df: Union[pl.DataFrame, pl.LazyFrame],
):
    items = {"a": [1], "b": ["a", "c"]}

    with pytest.raises(plg.PolarsAssertError):
        given_df.pipe(plg.not_accepted_values, items)


@pytest.mark.parametrize(
    "given_df",
    [
        pl.DataFrame({"a": [1, 2, 3]}),
        pl.LazyFrame({"a": [1, 2, 3]}),
    ],
)
def test_not_null_proportion_accept_proportion_range_or_single_input(
    given_df: Union[pl.DataFrame, pl.LazyFrame],
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
    given_df: Union[pl.DataFrame, pl.LazyFrame],
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
    given_df: Union[pl.DataFrame, pl.LazyFrame],
):
    with pytest.raises(plg.PolarsAssertError) as err:
        given_df.pipe(plg.not_null_proportion, {"a": 0.9})

    expected_df_columns = ["not_null_fraction", "min_prop", "max_prop"]
    assert all([col in err.value.df.columns for col in expected_df_columns])


def test_format_ranges_by_has_columns_and_min_max():
    items = {"a": 0.5, "b": (0.9, 0.95)}
    given_df = checks._format_ranges_by_columns(items)

    expected = pl.DataFrame(
        [
            ("a", 0.5, 1.0),
            ("b", 0.9, 0.95),
        ],
        schema=["column", "min_prop", "max_prop"],
        orient="row",
    )
    testing.assert_frame_equal(given_df, expected)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_accepted_range(frame: Type[Union[pl.DataFrame, pl.LazyFrame]]):
    given_df = frame({"a": [1, 2, 3]})
    when = given_df.pipe(plg.accepted_range, {"a": (1, 3)})
    testing.assert_frame_equal(given_df, when)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_accepted_range_is_compatible_with_is_between_syntax(
    frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
):
    given_df = frame({"a": ["b", "c"]})
    when = given_df.pipe(plg.accepted_range, {"a": (pl.lit("a"), pl.lit("d"), "right")})
    testing.assert_frame_equal(given_df, when)
    when = given_df.pipe(plg.accepted_range, {"a": (pl.lit("a"), pl.lit("d"), "left")})
    testing.assert_frame_equal(given_df, when)
    when = given_df.pipe(plg.accepted_range, {"a": (pl.lit("a"), pl.lit("d"), "both")})
    testing.assert_frame_equal(given_df, when)
    when = given_df.pipe(plg.accepted_range, {"a": (pl.lit("a"), pl.lit("d"), "none")})
    testing.assert_frame_equal(given_df, when)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_accepted_range_erros_when_values_are_out_of_range(
    frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
):
    given_df = frame({"a": [1, 2, 3]})
    with pytest.raises(plg.PolarsAssertError) as err:
        given_df.pipe(plg.accepted_range, {"a": (0, 2)})
    testing.assert_frame_equal(err.value.df, pl.DataFrame({"a": [3]}))


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_accepted_range_errors_on_two_different_ranges(
    frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
):
    given_df = frame({"a": [1, 2, 3], "b": [1, 2, 3]})
    with pytest.raises(plg.PolarsAssertError) as err:
        given_df.pipe(plg.accepted_range, {"a": (0, 2), "b": (2, 3)})

    expected = pl.DataFrame({"a": [1, 3], "b": [1, 3]})
    testing.assert_frame_equal(err.value.df, expected)


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
    assert "Some values were removed from col 'a', for ex: ('b', 'c')" in str(err.value)

    initial_df = pl.DataFrame({"a": ["a"]})
    final_df = pl.DataFrame({"a": ["a", "b", "c"]})
    with pytest.raises(plg.PolarsAssertError) as err:
        final_df.pipe(plg.maintains_relationships, initial_df, "a")
    assert "Some values were added to col 'a', for ex: ('b', 'c')" in str(err.value)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_unique_combination_of_columns(frame: Type[Union[pl.DataFrame, pl.LazyFrame]]):
    given_df = frame({"a": ["a", "b"]})
    when = given_df.pipe(plg.unique_combination_of_columns, "a")
    testing.assert_frame_equal(given_df, when)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_unique_combination_of_columns_use_all_columns_by_default(
    frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
):
    given_df = frame({"a": ["a", "b"]})
    when = given_df.pipe(plg.unique_combination_of_columns)
    testing.assert_frame_equal(given_df, when)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_unique_combination_of_columns_accepts_list_as_input(
    frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
):
    given_df = frame({"a": ["a", "a"], "b": [1, 2]})
    when = given_df.pipe(plg.unique_combination_of_columns, ["a", "b"])
    testing.assert_frame_equal(given_df, when)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_unique_combination_of_columns_should_err_for_non_unicity(
    frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
):
    given_df = frame({"a": ["a", "a"]})
    with pytest.raises(plg.PolarsAssertError):
        given_df.pipe(plg.unique_combination_of_columns, "a")


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_unique_combination_of_columns_base_error_message_format(
    frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
):
    given_df = frame({"a": ["a", "a"]})
    with pytest.raises(plg.PolarsAssertError) as err:
        given_df.pipe(plg.unique_combination_of_columns, "a")

    base_message = "Some combinations of columns are not unique."
    assert base_message in str(err.value)


@pytest.mark.parametrize(
    "columns,custom_message",
    [
        ("a", 'See above, selected: col("a")'),
        (["a", "b"], 'See above, selected: cols(["a", "b"])'),
        (pl.Utf8, "See above, selected: dtype_columns"),
        (None, "See above, selected: *"),
    ],
)
def test_unique_combination_of_columns_error_message_format(columns, custom_message):
    given_df = pl.DataFrame({"a": ["a", "a"], "b": [1, 1]})
    with pytest.raises(plg.PolarsAssertError) as err:
        given_df.pipe(plg.unique_combination_of_columns, columns)

    base_message = "Some combinations of columns are not unique."

    assert base_message in str(err.value)
    assert custom_message in str(err.value)


@pytest.mark.parametrize(
    "frame,column",
    [
        (pl.DataFrame, "int"),
        (pl.DataFrame, "dates"),
        (pl.DataFrame, "datetimes"),
        (pl.LazyFrame, "int"),
        (pl.LazyFrame, "dates"),
        (pl.LazyFrame, "datetimes"),
    ],
)
def test_is_monotonic_works_with_int_float_dates_datetimes(
    frame: Type[Union[pl.DataFrame, pl.LazyFrame]], column
):
    given_df = frame(
        {
            "int": [1, 2, 3],
            "dates": ["2022-01-01", "2023-01-01", "2023-01-02"],
        }
    ).with_columns(
        dates=pl.col("dates").str.to_date(),
        datetimes=pl.col("dates").str.to_datetime(),
    )
    when = given_df.pipe(plg.is_monotonic, column)
    testing.assert_frame_equal(given_df, when)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_is_monotonic_error_when_not_monotonic(
    frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
):
    given_df = frame({"int": [1, 2, 1]})
    with pytest.raises(plg.PolarsAssertError):
        given_df.pipe(plg.is_monotonic, "int")

    given_df = frame({"int": [1, 2, 2]})
    with pytest.raises(plg.PolarsAssertError):
        given_df.pipe(plg.is_monotonic, "int")


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_is_monotonic_can_specify_decreasing_monotonic(
    frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
):
    given_df = frame({"int": [3, 2, 1]})
    when = given_df.pipe(plg.is_monotonic, "int", decreasing=True)
    testing.assert_frame_equal(given_df, when)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_is_monotonic_can_accept_non_strictly_monotonic(
    frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
):
    given_df = frame({"int": [1, 2, 2]})
    when = given_df.pipe(plg.is_monotonic, "int", strict=False)
    testing.assert_frame_equal(given_df, when)

    given_df = frame({"int": [3, 2, 2]})
    when = given_df.pipe(plg.is_monotonic, "int", decreasing=True, strict=False)
    testing.assert_frame_equal(given_df, when)


def test_is_monotonic_should_allow_to_specify_interval_between_each_row():
    given_df = pl.DataFrame({"int": [1, 2, 3]})
    when = given_df.pipe(plg.is_monotonic, "int", interval=1)
    testing.assert_frame_equal(given_df, when)

    given_df = pl.DataFrame({"int": [1, 2, 3]})
    with pytest.raises(plg.PolarsAssertError):
        given_df.pipe(plg.is_monotonic, "int", interval=2)


def test_is_monotonic_should_allow_to_specify_interval_compatible_with_timedelta():
    given_df = pl.DataFrame(
        pl.Series(
            "dates",
            [
                "2020-01-01 01:42:00",
                "2020-01-01 01:43:00",
                "2020-01-01 01:44:00",
            ],
        ).str.to_datetime()
    )
    when = given_df.pipe(plg.is_monotonic, "dates", interval="1m")
    testing.assert_frame_equal(given_df, when)


def test_is_monotonic_should_allow_to_specify_interval_compatible_with_group_by():
    given_df = pl.DataFrame(
        [
            ("2020-01-01 01:42:00", "A"),
            ("2020-01-01 01:43:00", "A"),
            ("2020-01-01 01:44:00", "A"),
            ("2021-12-12 01:43:00", "B"),
            ("2021-12-12 01:44:00", "B"),
        ],
        schema=["dates", "group"],
        orient="row",
    ).with_columns(pl.col("dates").str.to_datetime())

    when = given_df.pipe(plg.is_monotonic, "dates", interval="1m", group_by="group")
    testing.assert_frame_equal(given_df, when)

    with pytest.raises(plg.PolarsAssertError):
        given_df.pipe(plg.is_monotonic, "dates", interval="3m", group_by="group")


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_is_monotonic_should_handle_larger_intervals(
    frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
):
    given_df = frame(
        {
            "monthly_interval": pl.date_range(
                pl.date(2024, 1, 1), pl.date(2024, 6, 1), "1mo", eager=True
            )
        }
    )

    when = given_df.pipe(plg.is_monotonic, "monthly_interval", interval="1mo")
    testing.assert_frame_equal(given_df, when)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_is_monotonic_should_handle_larger_intervals_reversed(
    frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
):
    given_df = frame(
        {
            "monthly_interval": pl.date_range(
                pl.date(2024, 1, 1), pl.date(2024, 6, 1), "1mo", eager=True
            )
        }
    ).sort("monthly_interval", descending=True)

    when = given_df.pipe(
        plg.is_monotonic, "monthly_interval", decreasing=True, interval="-1mo"
    )
    testing.assert_frame_equal(given_df, when)


def test_is_monotonic_error_give_out_specifyic_error_message():
    given_df = pl.DataFrame({"int": [1, 2, 1]})
    with pytest.raises(plg.PolarsAssertError) as err:
        given_df.pipe(plg.is_monotonic, "int")
    expected_msg = 'Column "int" expected to be monotonic but is not, try .sort("int")'
    assert expected_msg in str(err.value)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_is_monotonic_should_work_on_dates(
    frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
):
    start_date = datetime.date(2025, 2, 8)
    dates = [start_date + datetime.timedelta(days=i) for i in range(10)]
    data = frame({"date": dates})

    given_df = data.sort("date")
    result = given_df.pipe(plg.is_monotonic, "date")
    testing.assert_frame_equal(result, given_df)

    given_df = data.sort("date", descending=True)
    result = given_df.pipe(plg.is_monotonic, "date", decreasing=True)
    testing.assert_frame_equal(result, given_df)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_is_monotonic_should_work_on_dates_with_duplicates(
    frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
):
    start_date = datetime.date(2025, 2, 8)
    dates = [start_date + datetime.timedelta(days=i) for i in range(10)]
    data = frame({"date": dates * 2})

    given_df = data.sort("date")
    result = given_df.pipe(plg.is_monotonic, "date", strict=False)
    testing.assert_frame_equal(result, given_df)

    given_df = data.sort("date", descending=True)
    result = given_df.pipe(plg.is_monotonic, "date", decreasing=True, strict=False)
    testing.assert_frame_equal(result, given_df)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_is_monotonic_should_work_on_generic_datetime(
    frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
):
    start_date = datetime.datetime(2020, 2, 8, 15, 32, 5)
    dates = [start_date + datetime.timedelta(days=i) for i in range(10)]
    data = frame({"datetime": dates})

    given_df = data.sort("datetime")
    result = given_df.pipe(plg.is_monotonic, "datetime")
    testing.assert_frame_equal(result, given_df)

    given_df = data.sort("datetime", descending=True)
    result = given_df.pipe(plg.is_monotonic, "datetime", decreasing=True)
    testing.assert_frame_equal(result, given_df)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_custom_checks_works_for_simple_filter(
    frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
):
    given_df = frame({"int": [1, 1, 1]})
    when = given_df.pipe(plg.custom_check, pl.col("int") == 1)
    testing.assert_frame_equal(given_df, when)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_custom_checks_errors_the_condition_returns_a_non_empty(
    frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
):
    given_df = frame({"int": [1, 2, 1]})
    with pytest.raises(plg.PolarsAssertError):
        given_df.pipe(plg.custom_check, pl.col("int") != 2)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_custom_checks_accept_over_clauses(
    frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
):
    given_df = frame({"a": [1, 1, 2, 2], "b": [1, 2, 3, 4]})
    when = given_df.pipe(plg.custom_check, pl.col("b").max().over("a") <= 4)
    testing.assert_frame_equal(given_df, when)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_custom_checks_should_select_only_affected_columns(
    frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
):
    given_df = frame(
        {
            "a": [1, 1, 2, 2],
            "b": [1, 2, 3, 4],
            "c": [1, 2, 3, 4],
        }
    )
    with pytest.raises(plg.PolarsAssertError) as err:
        given_df.pipe(plg.custom_check, pl.col("b").max().over("a") <= 3)

    assert {"a", "b"} == set(err.value.df.columns)


@pytest.mark.parametrize(
    "given_df",
    [
        pl.DataFrame({"a": [1, 2]}),
        pl.LazyFrame({"a": [1, 2]}),
    ],
)
def test_has_mandatory_values(given_df: Union[pl.DataFrame, pl.LazyFrame]):
    when = given_df.pipe(plg.has_mandatory_values, {"a": [1, 2]})
    testing.assert_frame_equal(given_df, when)


@pytest.mark.parametrize(
    "given_df",
    [
        pl.DataFrame({"a": [1, 2]}),
        pl.LazyFrame({"a": [1, 2]}),
    ],
)
def test_has_mandatory_values_should_error_on_missing_elements(
    given_df: Union[pl.DataFrame, pl.LazyFrame],
):
    with pytest.raises(plg.PolarsAssertError):
        given_df.pipe(plg.has_mandatory_values, {"a": [3]})


def test_has_mandatory_values_should_give_feedback_on_missing_values():
    given_df = pl.DataFrame({"a": [1, 1], "b": ["x", "y"]})
    with pytest.raises(plg.PolarsAssertError) as err:
        given_df.pipe(plg.has_mandatory_values, {"a": [1, 2], "b": ["s", "t"]})

    expected = {"a": [2], "b": ["s", "t"]}
    assert str(expected) in str(err.value)


@pytest.mark.parametrize(
    "given_df",
    [
        pl.DataFrame({"a": [1, 1, 1, 2], "group": ["G1", "G1", "G2", "G2"]}),
        pl.LazyFrame({"a": [1, 1, 1, 2], "group": ["G1", "G1", "G2", "G2"]}),
    ],
)
def test_has_mandatory_values_should_accept_group_by_option(
    given_df: Union[pl.DataFrame, pl.LazyFrame],
):
    when = given_df.pipe(plg.has_mandatory_values, {"a": [1]}, group_by="group")
    testing.assert_frame_equal(given_df, when)


@pytest.mark.parametrize(
    "given_df",
    [
        pl.DataFrame({"a": [1, 1, 1, 2], "group": ["G1", "G1", "G2", "G2"]}),
        pl.LazyFrame({"a": [1, 1, 1, 2], "group": ["G1", "G1", "G2", "G2"]}),
    ],
)
def test_has_mandatory_values_by_group_should_error_when_not_all_values_are_present(
    given_df: Union[pl.DataFrame, pl.LazyFrame],
):
    with pytest.raises(plg.PolarsAssertError):
        given_df.pipe(plg.has_mandatory_values, {"a": [1, 2]}, group_by="group")


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_mutually_exclusive_ranges_allows_to_specify_low_and_high_bounds(
    frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
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
    frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
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
    frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
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

    expected = given_df.pipe(plg.checks._add_row_index).head(4)
    if isinstance(expected, pl.LazyFrame):
        expected = expected.collect()
    testing.assert_frame_equal(err.value.df, expected)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_mutually_exclusive_ranges_allows_to_group_by_anoterh_column(
    frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
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


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_column_is_within_n_std_accepts_tuple_args(
    frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
):
    given_df = frame({"a": [1, 2, 2, 1]})

    when = given_df.pipe(plg.column_is_within_n_std, ("a", 2))

    testing.assert_frame_equal(given_df, when)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_column_is_within_n_std_shoud_error_on_outliers(
    frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
):
    given_df = frame({"a": list(range(0, 10)) + [5000]})

    with pytest.raises(plg.PolarsAssertError):
        given_df.pipe(plg.column_is_within_n_std, ("a", 2))


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_column_is_within_n_std_accepts_list_of_tuple_args(
    frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
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


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_at_least_one_should_return_df_is_one_value_is_not_null(
    frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
):
    given_df = frame({"a": [None, 1]})
    when = given_df.pipe(plg.at_least_one)
    testing.assert_frame_equal(given_df, when)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_at_least_one_should_error_if_all_values_are_nulls(
    frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
):
    given_df = frame({"a": [None, None]})
    with pytest.raises(plg.PolarsAssertError):
        given_df.pipe(plg.at_least_one)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_at_least_one_should_accept_column_selection(
    frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
):
    given_df = frame({"a": [None, None], "b": [1, None]})

    when = given_df.pipe(plg.at_least_one, "b")
    testing.assert_frame_equal(given_df, when)

    with pytest.raises(plg.PolarsAssertError):
        given_df.pipe(plg.at_least_one)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_at_least_one_should_accept_group_by_option(
    frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
):
    given_df = frame({"a": [None, 1, None, 2], "group": ["G1", "G1", "G2", "G2"]})
    when = given_df.pipe(plg.at_least_one, group_by="group")
    testing.assert_frame_equal(given_df, when)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_at_least_one_should_error_when_only_null_for_given_group(
    frame: Type[Union[pl.DataFrame, pl.LazyFrame]],
):
    given_df = frame(
        {
            "a": [None, None, None, 2],
            "group": ["G1", "G1", "G2", "G2"],
        }
    )
    with pytest.raises(plg.PolarsAssertError):
        given_df.pipe(plg.at_least_one, "a", group_by="group")
