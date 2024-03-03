import polars as pl
import pytest
from polars import testing

from furcoat import checks


def test_polar_check_error_has_simple_message_when_used_directly():
    error = checks.PolarsAssertError()
    assert error.df.is_empty()
    assert "There was an improper value in the passed DataFrame:\n" == str(error)


def test_polar_check_error_should_accept_df_as_input():
    error = checks.PolarsAssertError(pl.DataFrame({"a": [1]}))
    testing.assert_frame_equal(error.df, pl.DataFrame({"a": [1]}))


def test_polar_check_error_should_have_clearer_error_message_with_dataframe_input():
    error = checks.PolarsAssertError(pl.DataFrame({"a": [1]}))
    assert "There was an improper value in the passed DataFrame:" in str(error)
    assert str(error.df) in str(error)


@pytest.mark.parametrize(
    "input,expected",
    [
        ("a", pl.col("a")),
        ("b", pl.col("b")),
        (pl.col("b"), pl.col("b")),
        (None, pl.all()),
    ],
)
def test_sanitize_column_inputs_works_with(input, expected):
    given = checks._sanitize_column_inputs(input)
    assert str(given) == str(expected)


def test_is_shape():
    given = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})
    when = given.pipe(checks.has_shape, (3, 2))
    testing.assert_frame_equal(given, when)


def test_is_shape_should_error_on_wrong_shape():
    given = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})
    with pytest.raises(checks.PolarsAssertError):
        given.pipe(checks.has_shape, (2, 2))


def test_has_no_nulls_returns_df_when_all_values_defined():
    given = pl.DataFrame({"a": [1, 2]})
    when = given.pipe(checks.has_no_nulls)
    testing.assert_frame_equal(given, when)


def test_has_no_nulls_throws_error_on_null_values():
    given = pl.DataFrame({"a": [1, None]})
    with pytest.raises(checks.PolarsAssertError):
        given.pipe(checks.has_no_nulls)


def test_has_no_nulls_indicates_columns_with_nulls_in_error_message():
    given = pl.DataFrame({"a": [1, None]})
    expected = pl.DataFrame(
        {"column": ["a"], "null_count": [1]},
        schema={"column": pl.Utf8, "null_count": pl.UInt32},
    )
    with pytest.raises(checks.PolarsAssertError) as err:
        given.pipe(checks.has_no_nulls)
    testing.assert_frame_equal(err.value.df, expected)


def test_has_no_infs_returns_df_when_all_values_defined():
    given = pl.DataFrame({"a": [1, 2]})
    when = given.pipe(checks.has_no_infs)
    testing.assert_frame_equal(given, when)


def test_has_no_infs_throws_error_on_inf_values():
    given = pl.DataFrame({"a": [1, None, float("inf")]})
    with pytest.raises(checks.PolarsAssertError) as err:
        given.pipe(checks.has_no_infs)
    expected = pl.DataFrame({"a": [float("inf")]})
    testing.assert_frame_equal(err.value.df, expected)


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
    with pytest.raises(checks.PolarsAssertError) as err:
        given.pipe(checks.unique, "a")
    testing.assert_frame_equal(err.value.df, pl.DataFrame({"a": [1, 1]}))


def test_not_constant():
    given = pl.DataFrame({"a": [1, 2]})
    when = given.pipe(checks.not_constant, "a")
    testing.assert_frame_equal(given, when)


def test_not_constant_throws_error_on_constant_columns():
    given = pl.DataFrame({"b": [1, 1]})
    with pytest.raises(checks.PolarsAssertError):
        given.pipe(checks.not_constant, "b")


def test_not_constant_accept_different_types_of_input():
    given = pl.DataFrame({"a": [1, 2], "b": ["A", "B"]})
    when = given.pipe(checks.not_constant, ["a", "b"])
    testing.assert_frame_equal(given, when)

    given = pl.DataFrame({"a": [1, 2], "b": ["A", "B"]})
    when = given.pipe(checks.not_constant, pl.col("a"))
    testing.assert_frame_equal(given, when)


def test_accepted_values():
    items = {"a": [1, 2, 3], "b": ["a", "b", "c"]}
    given = pl.DataFrame(items)
    when = given.pipe(checks.accepted_values, items)
    testing.assert_frame_equal(given, when)


def test_accepted_values_should_accept_pl_expr():
    items = {"a": [1, 2, 3], "b": ["a", "b", "c"]}
    given = pl.DataFrame(items)
    when = given.pipe(checks.accepted_values, {"^a$": [1, 2, 3]})
    testing.assert_frame_equal(given, when)


def test_accepted_values_should_error_on_out_of_range_values():
    items = {"a": [1, 2, 3], "b": ["a", "b", "c"]}
    given = pl.DataFrame(items)
    items = {"a": [1, 2], "b": ["a", "b", "c"]}

    with pytest.raises(checks.PolarsAssertError) as err:
        given.pipe(checks.accepted_values, items)
    assert err.value.df.shape == (1, 2)


def test_not_accepted_values():
    given = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})
    when = given.pipe(checks.not_accepted_values, {"a": [4, 5]})
    testing.assert_frame_equal(given, when)


def test_not_accepted_values_should_accept_pl_expr():
    given = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})
    when = given.pipe(checks.not_accepted_values, {"^b$": ["d", ""]})
    testing.assert_frame_equal(given, when)


def test_not_accepted_values_should_error_on_out_of_range_values():
    given = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})
    items = {"a": [1], "b": ["a", "c"]}

    with pytest.raises(checks.PolarsAssertError) as err:
        given.pipe(checks.not_accepted_values, items)
    testing.assert_frame_equal(
        err.value.df, pl.DataFrame({"a": [1, 3], "b": ["a", "c"]})
    )


def test_not_null_proportion_accept_proportion_range_or_single_input():
    given = pl.DataFrame({"a": [1, 2, 3]})
    when = given.pipe(checks.not_null_proportion, {"a": (0.1, 1.0)})
    testing.assert_frame_equal(given, when)

    when = given.pipe(checks.not_null_proportion, {"a": 0.1})
    testing.assert_frame_equal(given, when)


def test_not_null_proportion_accept_multiple_inpute():
    given = pl.DataFrame(
        {
            "a": [1, None, None],
            "b": [1, 2, None],
        }
    )
    when = given.pipe(checks.not_null_proportion, {"a": 0.1, "b": 0.1})
    testing.assert_frame_equal(given, when)


def test_not_null_proportion_errors_with_too_many_nulls():
    given = pl.DataFrame({"a": [1, None]})
    with pytest.raises(checks.PolarsAssertError) as err:
        given.pipe(checks.not_null_proportion, {"a": 0.9})

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
    when = given.pipe(checks.accepted_range, {"a": (1, 3)})
    testing.assert_frame_equal(given, when)


def test_accepted_range_is_compatible_with_is_between_syntax():
    given = pl.DataFrame({"a": ["b", "c"]})
    when = given.pipe(checks.accepted_range, {"a": (pl.lit("a"), pl.lit("d"), "right")})
    testing.assert_frame_equal(given, when)
    when = given.pipe(checks.accepted_range, {"a": (pl.lit("a"), pl.lit("d"), "left")})
    testing.assert_frame_equal(given, when)
    when = given.pipe(checks.accepted_range, {"a": (pl.lit("a"), pl.lit("d"), "both")})
    testing.assert_frame_equal(given, when)
    when = given.pipe(checks.accepted_range, {"a": (pl.lit("a"), pl.lit("d"), "none")})
    testing.assert_frame_equal(given, when)


def test_accepted_range_erros_when_values_are_out_of_range():
    given = pl.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(checks.PolarsAssertError) as err:
        given.pipe(checks.accepted_range, {"a": (0, 2)})
    testing.assert_frame_equal(err.value.df, pl.DataFrame({"a": [3]}))


def test_accepted_range_errors_on_two_different_ranges():
    given = pl.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    with pytest.raises(checks.PolarsAssertError) as err:
        given.pipe(checks.accepted_range, {"a": (0, 2), "b": (2, 3)})

    expected = pl.DataFrame({"a": [1, 3], "b": [1, 3]})
    testing.assert_frame_equal(err.value.df, expected)
