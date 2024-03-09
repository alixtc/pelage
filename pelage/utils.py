from typing import List

import polars as pl


def _list_defective_columns(bad: pl.DataFrame, conditions: List[pl.Expr]) -> List[str]:
    is_column_defective = bad.select(cond.any() for cond in conditions)
    return [col.name for col in is_column_defective if col.all()]
