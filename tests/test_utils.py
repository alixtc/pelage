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
