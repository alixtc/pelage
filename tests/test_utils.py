from collections import OrderedDict

import polars as pl

from pelage import utils


def test_list_defective_columns_should_be_empty_when_bad_df_is_empty():
    given = pl.DataFrame({"a": [1, 1, 2, 2], "b": [1, 2, 3, 4]})
    not_met_conditions = [(pl.col("a") >= 1), pl.col("b") >= 1]
    bad = given.filter(pl.Expr.or_(*not_met_conditions).not_())
    assert utils._list_defective_columns(bad, not_met_conditions) == []


def test_list_defective_columns_should_return_name_of_single_defective_column():
    given = pl.DataFrame({"a": [1, 1, 2, 2], "b": [1, 2, 3, 4]})
    check = [(pl.col("a") <= 3), pl.col("b").is_in([2])]
    not_met_conditions = [cond.not_() for cond in check]
    bad = given.filter(pl.Expr.or_(*not_met_conditions))
    assert utils._list_defective_columns(bad, not_met_conditions) == ["b"]


def test_list_defective_columns_should_return_all_name_of_defective_columns():
    given = pl.DataFrame({"a": [1, 1, 2, 2], "b": [1, 2, 3, 4]})
    check = [(pl.col("a") <= 1), pl.col("b").is_in([2])]
    not_met_conditions = [cond.not_() for cond in check]
    bad = given.filter(pl.Expr.or_(*not_met_conditions))
    assert utils._list_defective_columns(bad, not_met_conditions) == ["a", "b"]


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
