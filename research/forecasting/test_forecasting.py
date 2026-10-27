"""Temporal integrity and units: failures here invalidate a forecast comparison."""
import numpy as np
import pandas as pd

from build_data import normalize_electricity, split_for, tabular_features, calendar


def sample_rows():
    rows = []
    for time in pd.date_range("2025-01-01T00:00Z", periods=12, freq="5min"):
        row = {"Minutes5UTC": time.isoformat(), "PriceArea": "DK2",
               "ProductionLt100MW": 100, "ProductionGe100MW": 100,
               "OffshoreWindPower": 200, "OnshoreWindPower": 100, "SolarPower": 0,
               "ExchangeGermany": -50, "ExchangeSweden": 20, "ExchangeGreatBelt": 10,
               "ExchangeNorway": None, "ExchangeNetherlands": None, "ExchangeGreatBritain": None,
               "BornholmSE4": 99}
        rows.append(row)
    return pd.DataFrame(rows)


def test_hourly_power_mean_and_structural_nulls():
    result = normalize_electricity(sample_rows())
    row = result.loc[result.area.eq("DK2")].iloc[0]
    assert row.time == pd.Timestamp("2025-01-01T01:00Z")
    assert row.wind == 300
    assert row.load == 480  # Great Belt included, duplicate Bornholm omitted.


def test_missing_measurement_does_not_become_zero_or_partial_hour():
    data = sample_rows()
    data.loc[0, "OffshoreWindPower"] = None
    row = normalize_electricity(data).query("area == 'DK2'").iloc[0]
    assert np.isnan(row.wind)
    assert np.isnan(row.load)
    assert row.solar == 0
    row = normalize_electricity(sample_rows().iloc[:-1]).query("area == 'DK2'").iloc[0]
    assert np.isnan(row.wind)


def test_split_boundaries_purge_overlapping_labels():
    assert split_for(pd.Timestamp("2025-12-28T12:00Z")) == "train"
    assert split_for(pd.Timestamp("2025-12-31T12:00Z")) == "purged"
    assert split_for(pd.Timestamp("2026-01-01T12:00Z")) == "validation"
    assert split_for(pd.Timestamp("2026-05-31T12:00Z")) == "purged"
    assert split_for(pd.Timestamp("2026-06-01T12:00Z")) == "test"


def test_multi_day_baseline_never_reads_future_labels():
    history = np.arange(168 * 3).reshape(1, 168, 3)
    future = np.zeros((1, 48, 1))
    x, names = tabular_features(history, future, 0)
    day = x[:, 1 + names.index("day_baseline")]
    week = x[:, 1 + names.index("week_baseline")]
    np.testing.assert_array_equal(day[:24], day[24:])
    np.testing.assert_array_equal(week, history[0, :48, 0])
    assert x[:, 1 + names.index("lag_1")].max() == history[0, -1, 0]


def test_calendar_handles_dst_without_duplicate_utc_time():
    times = pd.date_range("2026-03-29T00:00Z", periods=4, freq="1h")
    result = calendar(times)
    assert len(result) == 4
    assert result.iloc[0].hour_1 == 1
    assert result.iloc[1].hour_3 == 1
