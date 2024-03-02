import polars as pl
import pytest
from polars import testing

from furcoat import checks


def test_has_no_nulls_returns_df_when_all_values_defined():
    given = pl.DataFrame(
        {
            "a": [1, 2],
        }
    )
    when = given.pipe(checks.has_no_nulls)
    testing.assert_frame_equal(given, when)


def test_has_no_nulls_throws_error_on_null_values():
    given = pl.DataFrame(
        {
            "a": [1, None],
        }
    )
    with pytest.raises(checks.PolarsCheckError):
        given.pipe(checks.has_no_nulls)
