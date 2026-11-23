"""Eligibility of preliminary Energinet readings for frozen trial outcomes."""

import math

import pandas as pd


def valid_wind(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def capture_time(row, at, minimum, maximum=None):
    value = row.get("retrievedAt")
    if not value:
        return None
    try:
        captured = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    target = pd.Timestamp(row["time"])
    if captured.tzinfo is None or captured > at:
        return None
    if captured < target + pd.Timedelta(hours=minimum):
        return None
    if maximum is not None and at > target + pd.Timedelta(hours=maximum):
        return None
    return captured.isoformat()
