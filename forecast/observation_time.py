"""Eligibility of preliminary Energinet readings for frozen trial outcomes."""

import pandas as pd


def capture_time(row, at, minimum, maximum):
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
    youngest = target + pd.Timedelta(hours=minimum)
    oldest = target + pd.Timedelta(hours=maximum)
    if youngest <= captured <= oldest and at <= oldest:
        return captured.isoformat()
    return None
