import polars as pl


class PolarsCheckError(Exception):
    pass


def has_no_nulls(data: pl.DataFrame):
    if data.null_count().sum_horizontal().item() > 0:
        raise PolarsCheckError
    return data
