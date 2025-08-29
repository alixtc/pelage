from collections import OrderedDict

import polars as pl

from pelage import utils


def test_compare_schema_should_be_empty_when_types_matche():
    given = pl.DataFrame({"a": [1, 2], "b": [1.0, 1.2]})
    expected_schema = OrderedDict(a=pl.Int64, b=pl.Float64)
    assert utils.compare_schema(given.schema, expected_schema) == ""


def test_compare_schema_returns_column_expected_and_real_dtypes():
    given = pl.DataFrame({"a": [1, 2], "b": [1.0, 1.2]})
    expected_schema = OrderedDict(a=pl.Int64, b=pl.Int64)
    message = "column='b', expected_type=Int64, real_dtype=Float64"
    assert utils.compare_schema(given.schema, expected_schema) == message


def test_compare_schema_returns_one_line_per_mismatch_column():
    given = pl.DataFrame({"a": [1, 2], "b": [1.0, 1.2]})
    expected_schema = OrderedDict(a=pl.Float32, b=pl.Int64)
    message = (
        "column='a', expected_type=Float32, real_dtype=Int64"
        + "\n"
        + "column='b', expected_type=Int64, real_dtype=Float64"
    )
    assert utils.compare_schema(given.schema, expected_schema) == message
