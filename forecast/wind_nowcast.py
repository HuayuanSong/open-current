"""Wind-only short-horizon adjustment using completed five-minute measurements."""

import numpy as np
import pandas as pd

from .features import AREAS


def recent_wind(records, at, buffer_minutes=10):
    """Return two complete 15-minute means strictly available before selection.

    Energinet timestamps mark the beginning of five-minute intervals. Keep a
    publication buffer after interval completion. Never fill a missing turbine
    component, accept future intervals, or interpolate missing measurements.
    """
    at = pd.Timestamp(at)
    if at.tzinfo is None:
        raise ValueError("Wind observation cutoff needs a timezone")
    cutoff = at - pd.Timedelta(minutes=buffer_minutes + 5)
    result = {}
    for area in AREAS:
        rows = {}
        for row in records:
            if row.get("PriceArea") != area:
                continue
            stamp = pd.to_datetime(row.get("Minutes5UTC"), utc=True)
            if pd.isna(stamp) or stamp > cutoff or stamp != stamp.floor("5min"):
                continue
            parts = [row.get("OnshoreWindPower"), row.get("OffshoreWindPower")]
            if any(v is None or not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0 for v in parts):
                continue
            value = float(sum(parts))
            if stamp in rows and rows[stamp] != value:
                raise ValueError("Conflicting wind measurements at one timestamp")
            rows[stamp] = value
        if not rows:
            raise ValueError(f"No recent wind observations for {area}")
        last = max(rows)
        if cutoff - last > pd.Timedelta(minutes=10):
            raise ValueError(f"Five-minute wind observations are stale for {area}")
        times = pd.date_range(end=last, periods=6, freq="5min")
        if any(t not in rows for t in times):
            raise ValueError(f"Incomplete recent wind observations for {area}")
        previous = float(np.mean([rows[t] for t in times[:3]]))
        level = float(np.mean([rows[t] for t in times[3:]]))
        # The latest mean covers [last-10min, last+5min); use its midpoint.
        result[area] = dict(levelMW=level, previousMW=previous,
                            trendMWPerHour=(level - previous) * 4,
                            midpoint=(last - pd.Timedelta(minutes=2, seconds=30)).isoformat(),
                            observationEnd=(last + pd.Timedelta(minutes=5)).isoformat(),
                            selectedAt=at.isoformat(), bufferMinutes=buffer_minutes)
    return result


def adjust(prediction, times, recent, rules, horizon_hours=6):
    """Blend the first six wind hours with a bounded level/trend extrapolation.

    Solar and demand and all longer-horizon wind values are copied unchanged.
    The trend is capped to one hour and +/- the current measured power level.
    """
    output = np.array(prediction, dtype=float, copy=True)
    if output.shape != (2, 48, 3) or not np.isfinite(output).all() or (output < 0).any():
        raise ValueError("Expected a finite, nonnegative two-area forecast")
    stamps = pd.to_datetime(times, utc=True)
    if len(stamps) != 48 or any((stamps[1:] - stamps[:-1]) != pd.Timedelta(hours=1)):
        raise ValueError("Expected 48 consecutive target hours")
    if not isinstance(horizon_hours, int) or not 1 <= horizon_hours <= 6:
        raise ValueError("Correction horizon must be between one and six hours")
    for ai, area in enumerate(AREAS):
        rule, reading = rules[area], recent[area]
        strength, half_life, trend_weight = (float(rule[k]) for k in ("strength", "halfLifeHours", "trendWeight"))
        if not (0 <= strength <= 1 and half_life > 0 and 0 <= trend_weight <= 1):
            raise ValueError("Invalid wind correction rule")
        if pd.Timestamp(reading["selectedAt"]) >= stamps[0]:
            raise ValueError("Correction must be selected before the first target ends")
        midpoint = pd.Timestamp(reading["midpoint"])
        if midpoint.tzinfo is None or any(not np.isfinite(reading[k]) for k in ("levelMW", "previousMW", "trendMWPerHour")) or reading["levelMW"] < 0:
            raise ValueError("Invalid recent wind measurements")
        lead = np.maximum(0, (stamps[:horizon_hours] - pd.Timedelta(minutes=30) - midpoint).total_seconds() / 3600)
        level = reading["levelMW"]
        change = np.clip(reading["trendMWPerHour"] * np.minimum(lead, 1), -level, level)
        anchor = np.maximum(0, level + trend_weight * change)
        weight = strength * np.exp2(-lead / half_life)
        output[ai, :horizon_hours, 0] = ((1 - weight) * output[ai, :horizon_hours, 0] + weight * anchor)
    return output
