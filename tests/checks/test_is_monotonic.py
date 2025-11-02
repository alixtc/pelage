import datetime

import polars as pl
import pytest
from polars import testing

import pelage as plg


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
    frame: type[pl.DataFrame | pl.LazyFrame], column
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
    frame: type[pl.DataFrame | pl.LazyFrame],
):
    given_df = frame({"int": [1, 2, 1]})
    with pytest.raises(plg.PolarsAssertError):
        given_df.pipe(plg.is_monotonic, "int")

    given_df = frame({"int": [1, 2, 2]})
    with pytest.raises(plg.PolarsAssertError):
        given_df.pipe(plg.is_monotonic, "int")


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_is_monotonic_can_specify_decreasing_monotonic(
    frame: type[pl.DataFrame | pl.LazyFrame],
):
    given_df = frame({"int": [3, 2, 1]})
    when = given_df.pipe(plg.is_monotonic, "int", decreasing=True)
    testing.assert_frame_equal(given_df, when)


@pytest.mark.parametrize("frame", [pl.DataFrame, pl.LazyFrame])
def test_is_monotonic_can_accept_non_strictly_monotonic(
    frame: type[pl.DataFrame | pl.LazyFrame],
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
    frame: type[pl.DataFrame | pl.LazyFrame],
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
    frame: type[pl.DataFrame | pl.LazyFrame],
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
    frame: type[pl.DataFrame | pl.LazyFrame],
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
    frame: type[pl.DataFrame | pl.LazyFrame],
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
    frame: type[pl.DataFrame | pl.LazyFrame],
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
