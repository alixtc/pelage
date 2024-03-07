from textwrap import dedent

import polars as pl
import pytest
from polars import testing

import furcoat as plg
from furcoat import checks


def test_dataframe_error_message_format():
    data = pl.DataFrame({"a": ["a", "b"], "b": [1, 2]})
    message = "Additional message"

    expected_message = """
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
    -->Additional message
    """
    formatted_msg = str(plg.PolarsAssertError(data, message))
    assert formatted_msg in dedent(expected_message)


def test_dataframe_error_message_format_accept_only_message():
    message = "Additional message"

    expected_message = """
    Error with the DataFrame passed to the check function:
    -->Additional message
    """

    formatted_msg = str(plg.PolarsAssertError(supp_message=message))
    assert formatted_msg in dedent(expected_message)


def test_dataframe_error_message_format_accepts_no_arguments():
    expected_message = """
    Error with the DataFrame passed to the check function:
    -->"""

    formatted_msg = str(plg.PolarsAssertError())
    assert formatted_msg in dedent(expected_message)


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
    given = checks._sanitize_column_inputs(input)
    assert str(given) == str(expected)


def test_sanitize_column_inputs_for_type_checker_pl_dtypes():
    given = checks._sanitize_column_inputs(pl.Int64)
    assert str(given) == str(pl.col(pl.Int64))


def test_is_shape():
    given = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})
    when = given.pipe(plg.has_shape, (3, 2))
    testing.assert_frame_equal(given, when)


def test_is_shape_should_error_on_wrong_shape():
    given = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})
    with pytest.raises(plg.PolarsAssertError):
        given.pipe(plg.has_shape, (2, 2))


def test_has_no_nulls_returns_df_when_all_values_defined():
    given = pl.DataFrame({"a": [1, 2]})
    when = given.pipe(plg.has_no_nulls)
    testing.assert_frame_equal(given, when)


def test_has_no_nulls_throws_error_on_null_values():
    given = pl.DataFrame({"a": [1, None]})
    with pytest.raises(plg.PolarsAssertError):
        given.pipe(plg.has_no_nulls)


def test_has_no_nulls_indicates_columns_with_nulls_in_error_message():
    given = pl.DataFrame({"a": [1, None]})
    expected = pl.DataFrame(
        {"column": ["a"], "null_count": [1]},
        schema={"column": pl.Utf8, "null_count": pl.UInt32},
    )
    with pytest.raises(plg.PolarsAssertError) as err:
        given.pipe(plg.has_no_nulls)
    testing.assert_frame_equal(err.value.df, expected)


def test_has_no_infs_returns_df_when_all_values_defined():
    given = pl.DataFrame({"a": [1, 2]})
    when = given.pipe(plg.has_no_infs)
    testing.assert_frame_equal(given, when)


def test_has_no_infs_throws_error_on_inf_values():
    given = pl.DataFrame({"a": [1, None, float("inf")]})
    with pytest.raises(plg.PolarsAssertError) as err:
        given.pipe(plg.has_no_infs)
    expected = pl.DataFrame({"a": [float("inf")]})
    testing.assert_frame_equal(err.value.df, expected)


def test_unique_should_return_df_if_column_has_unique_values():
    given = pl.DataFrame({"a": [1, 2]})
    when = given.pipe(plg.unique, "a")
    testing.assert_frame_equal(given, when)


def test_unique_should_should_accept_list_and_polars_select():
    given = pl.DataFrame({"a": [1, 2]})
    when = given.pipe(plg.unique, ["a"])
    testing.assert_frame_equal(given, when)

    given = pl.DataFrame({"a": [1, 2]})
    when = given.pipe(plg.unique, pl.col("a"))
    testing.assert_frame_equal(given, when)


def test_unique_should_throw_error_on_duplicates():
    given = pl.DataFrame({"a": [1, 1, 2]})
    with pytest.raises(plg.PolarsAssertError) as err:
        given.pipe(plg.unique, "a")
    testing.assert_frame_equal(err.value.df, pl.DataFrame({"a": [1, 1]}))


def test_not_constant():
    given = pl.DataFrame({"a": [1, 2]})
    when = given.pipe(plg.not_constant, "a")
    testing.assert_frame_equal(given, when)


def test_not_constant_throws_error_on_constant_columns():
    given = pl.DataFrame({"b": [1, 1]})
    with pytest.raises(plg.PolarsAssertError):
        given.pipe(plg.not_constant, "b")


def test_not_constant_accept_different_types_of_input():
    given = pl.DataFrame({"a": [1, 2], "b": ["A", "B"]})
    when = given.pipe(plg.not_constant, ["a", "b"])
    testing.assert_frame_equal(given, when)

    given = pl.DataFrame({"a": [1, 2], "b": ["A", "B"]})
    when = given.pipe(plg.not_constant, pl.col("a"))
    testing.assert_frame_equal(given, when)


def test_accepted_values():
    items = {"a": [1, 2, 3], "b": ["a", "b", "c"]}
    given = pl.DataFrame(items)
    when = given.pipe(plg.accepted_values, items)
    testing.assert_frame_equal(given, when)


def test_accepted_values_should_accept_pl_expr():
    items = {"a": [1, 2, 3], "b": ["a", "b", "c"]}
    given = pl.DataFrame(items)
    when = given.pipe(plg.accepted_values, {"^a$": [1, 2, 3]})
    testing.assert_frame_equal(given, when)


def test_accepted_values_should_error_on_out_of_range_values():
    items = {"a": [1, 2, 3], "b": ["a", "b", "c"]}
    given = pl.DataFrame(items)
    items = {"a": [1, 2], "b": ["a", "b", "c"]}

    with pytest.raises(plg.PolarsAssertError) as err:
        given.pipe(plg.accepted_values, items)
    assert err.value.df.shape == (1, 2)


def test_not_accepted_values():
    given = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})
    when = given.pipe(plg.not_accepted_values, {"a": [4, 5]})
    testing.assert_frame_equal(given, when)


def test_not_accepted_values_should_accept_pl_expr():
    given = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})
    when = given.pipe(plg.not_accepted_values, {"^b$": ["d", ""]})
    testing.assert_frame_equal(given, when)


def test_not_accepted_values_should_error_on_out_of_range_values():
    given = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})
    items = {"a": [1], "b": ["a", "c"]}

    with pytest.raises(plg.PolarsAssertError) as err:
        given.pipe(plg.not_accepted_values, items)
    testing.assert_frame_equal(
        err.value.df, pl.DataFrame({"a": [1, 3], "b": ["a", "c"]})
    )


def test_not_null_proportion_accept_proportion_range_or_single_input():
    given = pl.DataFrame({"a": [1, 2, 3]})
    when = given.pipe(plg.not_null_proportion, {"a": (0.1, 1.0)})
    testing.assert_frame_equal(given, when)

    when = given.pipe(plg.not_null_proportion, {"a": 0.1})
    testing.assert_frame_equal(given, when)


def test_not_null_proportion_accept_multiple_inpute():
    given = pl.DataFrame(
        {
            "a": [1, None, None],
            "b": [1, 2, None],
        }
    )
    when = given.pipe(plg.not_null_proportion, {"a": 0.1, "b": 0.1})
    testing.assert_frame_equal(given, when)


def test_not_null_proportion_errors_with_too_many_nulls():
    given = pl.DataFrame({"a": [1, None]})
    with pytest.raises(plg.PolarsAssertError) as err:
        given.pipe(plg.not_null_proportion, {"a": 0.9})

    expected_err_df = pl.DataFrame(
        {
            "column": ["a"],
            "not_null_proportion": [0.5],
            "min_prop": [0.9],
            "max_prop": [1],
        }
    )
    testing.assert_frame_equal(err.value.df, expected_err_df)


def test_format_ranges_by_has_columns_and_min_max():
    items = {"a": 0.5, "b": (0.9, 0.95)}
    given = checks._format_ranges_by_columns(items)

    expected = pl.DataFrame(
        [
            ("a", 0.5, 1.0),
            ("b", 0.9, 0.95),
        ],
        schema=["column", "min_prop", "max_prop"],
    )
    testing.assert_frame_equal(given, expected)


def test_accepted_range():
    given = pl.DataFrame({"a": [1, 2, 3]})
    when = given.pipe(plg.accepted_range, {"a": (1, 3)})
    testing.assert_frame_equal(given, when)


def test_accepted_range_is_compatible_with_is_between_syntax():
    given = pl.DataFrame({"a": ["b", "c"]})
    when = given.pipe(plg.accepted_range, {"a": (pl.lit("a"), pl.lit("d"), "right")})
    testing.assert_frame_equal(given, when)
    when = given.pipe(plg.accepted_range, {"a": (pl.lit("a"), pl.lit("d"), "left")})
    testing.assert_frame_equal(given, when)
    when = given.pipe(plg.accepted_range, {"a": (pl.lit("a"), pl.lit("d"), "both")})
    testing.assert_frame_equal(given, when)
    when = given.pipe(plg.accepted_range, {"a": (pl.lit("a"), pl.lit("d"), "none")})
    testing.assert_frame_equal(given, when)


def test_accepted_range_erros_when_values_are_out_of_range():
    given = pl.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(plg.PolarsAssertError) as err:
        given.pipe(plg.accepted_range, {"a": (0, 2)})
    testing.assert_frame_equal(err.value.df, pl.DataFrame({"a": [3]}))


def test_accepted_range_errors_on_two_different_ranges():
    given = pl.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    with pytest.raises(plg.PolarsAssertError) as err:
        given.pipe(plg.accepted_range, {"a": (0, 2), "b": (2, 3)})

    expected = pl.DataFrame({"a": [1, 3], "b": [1, 3]})
    testing.assert_frame_equal(err.value.df, expected)


def test_maintains_relationships():
    initial_df = pl.DataFrame({"a": ["a", "b"]})
    final_df = pl.DataFrame({"a": ["a", "b"]})
    when = final_df.pipe(plg.maintains_relationships, initial_df, "a")
    testing.assert_frame_equal(when, final_df)


def test_maintains_relationships_should_errors_if_some_values_are_dropped():
    initial_df = pl.DataFrame({"a": ["a", "b"]})
    final_df = pl.DataFrame({"a": ["a"]})
    with pytest.raises(plg.PolarsAssertError):
        final_df.pipe(plg.maintains_relationships, initial_df, "a")


def test_maintains_relationships_should_specify_some_changing_values():
    initial_df = pl.DataFrame({"a": ["a", "b", "c"]})
    final_df = pl.DataFrame({"a": ["a"]})
    with pytest.raises(plg.PolarsAssertError) as err:
        final_df.pipe(plg.maintains_relationships, initial_df, "a")
    assert "Some values were removed from col 'a', for ex: ('b', 'c')" in str(err.value)

    initial_df = pl.DataFrame({"a": ["a"]})
    final_df = pl.DataFrame({"a": ["a", "b", "c"]})
    with pytest.raises(plg.PolarsAssertError) as err:
        final_df.pipe(plg.maintains_relationships, initial_df, "a")
    assert "Some values were added to col 'a', for ex: ('b', 'c')" in str(err.value)


def test_unique_combination_of_columns():
    given = pl.DataFrame({"a": ["a", "b"]})
    when = given.pipe(plg.unique_combination_of_columns, "a")
    testing.assert_frame_equal(given, when)


def test_unique_combination_of_columns_use_all_columns_by_default():
    given = pl.DataFrame({"a": ["a", "b"]})
    when = given.pipe(plg.unique_combination_of_columns)
    testing.assert_frame_equal(given, when)


def test_unique_combination_of_columns_accepts_list_as_input():
    given = pl.DataFrame({"a": ["a", "a"], "b": [1, 2]})
    when = given.pipe(plg.unique_combination_of_columns, ["a", "b"])
    testing.assert_frame_equal(given, when)


def test_unique_combination_of_columns_should_err_for_non_unicity():
    given = pl.DataFrame({"a": ["a", "a"]})
    with pytest.raises(plg.PolarsAssertError):
        given.pipe(plg.unique_combination_of_columns, "a")


def test_unique_combination_of_columns_base_error_message_format():
    given = pl.DataFrame({"a": ["a", "a"]})
    with pytest.raises(plg.PolarsAssertError) as err:
        given.pipe(plg.unique_combination_of_columns, "a")

    base_message = "Some combinations of columns are not unique."
    assert base_message in str(err.value)


@pytest.mark.parametrize(
    "columns,custom_message",
    [
        ("a", 'See above, selected: col("a")'),
        (["a", "b"], 'See above, selected: cols(["a", "b"])'),
        (pl.Utf8, "See above, selected: dtype_columns([String])"),
        (None, "See above, selected: *"),
    ],
)
def test_unique_combination_of_columns_error_message_format(columns, custom_message):
    given = pl.DataFrame({"a": ["a", "a"], "b": [1, 1]})
    with pytest.raises(plg.PolarsAssertError) as err:
        given.pipe(plg.unique_combination_of_columns, columns)

    base_message = "Some combinations of columns are not unique."

    assert base_message in str(err.value)
    assert custom_message in str(err.value)


@pytest.mark.parametrize("column,", ["int", "dates", "datetimes"])
def test_is_monotonic_works_with_int_float_dates_datetimes(column):
    given = pl.DataFrame(
        {
            "int": [1, 2, 3],
            "dates": ["2022-01-01", "2023-01-01", "2023-01-02"],
        }
    ).with_columns(
        dates=pl.col("dates").str.to_date(),
        datetimes=pl.col("dates").str.to_datetime(),
    )
    when = given.pipe(plg.is_monotonic, column)
    testing.assert_frame_equal(given, when)


def test_is_monotonic_error_when_not_monotonic():
    given = pl.DataFrame({"int": [1, 2, 1]})
    with pytest.raises(plg.PolarsAssertError):
        given.pipe(plg.is_monotonic, "int")

    given = pl.DataFrame({"int": [1, 2, 2]})
    with pytest.raises(plg.PolarsAssertError):
        given.pipe(plg.is_monotonic, "int")


def test_is_monotonic_can_specify_decreasing_monotonic():
    given = pl.DataFrame({"int": [3, 2, 1]})
    when = given.pipe(plg.is_monotonic, "int", decreasing=True)
    testing.assert_frame_equal(given, when)


def test_is_monotonic_can_accept_non_strictly_monotonic():
    given = pl.DataFrame({"int": [1, 2, 2]})
    when = given.pipe(plg.is_monotonic, "int", strict=False)
    testing.assert_frame_equal(given, when)

    given = pl.DataFrame({"int": [3, 2, 2]})
    when = given.pipe(plg.is_monotonic, "int", decreasing=True, strict=False)
    testing.assert_frame_equal(given, when)


def test_is_monotonic_error_give_out_specifyic_error_message():
    given = pl.DataFrame({"int": [1, 2, 1]})
    with pytest.raises(plg.PolarsAssertError) as err:
        given.pipe(plg.is_monotonic, "int")
    expected_msg = 'Column "int" expected to be monotonic but is not, try .sort("int")'
    assert expected_msg in str(err.value)


def test_custom_checks_works_for_simple_filter():
    given = pl.DataFrame({"int": [1, 1, 1]})
    when = given.pipe(plg.custom_check, pl.col("int") == 1)
    testing.assert_frame_equal(given, when)


def test_custom_checks_errors_the_condition_returns_a_non_empty():
    given = pl.DataFrame({"int": [1, 2, 1]})
    with pytest.raises(plg.PolarsAssertError):
        given.pipe(plg.custom_check, pl.col("int") != 2)


def test_custom_checks_accept_over_clauses():
    given = pl.DataFrame({"a": [1, 1, 2, 2], "b": [1, 2, 3, 4]})
    when = given.pipe(plg.custom_check, pl.col("b").max().over("b") <= 4)
    testing.assert_frame_equal(given, when)
