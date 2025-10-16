from textwrap import dedent

import polars as pl
import pytest
from polars import testing

import pelage as plg
from pelage import utils


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
    given_df = utils._sanitize_column_inputs(input)
    assert str(given_df) == str(expected)


def test_sanitize_column_inputs_for_type_checker_pl_dtypes():
    given_df = utils._sanitize_column_inputs(pl.Int64)
    assert str(given_df) == str(pl.col(pl.Int64))


def test_is_shape_should_error_on_wrong_shape():
    given_df = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})
    with pytest.raises(plg.PolarsAssertError):
        given_df.pipe(plg.has_shape, (2, 2))


def test_format_ranges_by_has_columns_and_min_max():
    items = {"a": 0.5, "b": (0.9, 0.95)}
    given_df = utils._format_ranges_by_columns(items)

    expected = pl.DataFrame(
        [
            ("a", 0.5, 1.0),
            ("b", 0.9, 0.95),
        ],
        schema=["column", "min_prop", "max_prop"],
        orient="row",
    )
    testing.assert_frame_equal(given_df, expected)
