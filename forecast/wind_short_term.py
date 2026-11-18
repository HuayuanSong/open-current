"""Small causal 1–6-hour wind challenger; never changes the public forecast."""

import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import AREAS
from .wind_nowcast import recent_wind

HERE = Path(__file__).resolve().parents[1] / "research/forecasting/wind_refinement"


def load_models():
    selection = json.loads((HERE / "selection.json").read_text())
    loaded = {}
    for area, entry in selection["models"].items():
        path = HERE / "models" / entry["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Ramp model checksum differs")
        loaded[area] = json.loads(path.read_text()) if entry["family"] == "ridge" else lgb.Booster(model_file=str(path))
    return loaded, selection


def require_observed_wind(mask):
    """The saved observation mask contains 0/1 values for each area, hour and target."""
    recent = np.asarray(mask)[:, -24:, 0]
    if recent.shape != (2, 24) or not (recent == 1).all():
        raise ValueError("Missing recent hourly wind input")


def measurements(records, at):
    """Three hours of complete five-minute readings with the live release buffer."""
    recent = recent_wind(records, at)
    frame = pd.DataFrame(records)
    frame["time"] = pd.to_datetime(frame.Minutes5UTC, utc=True)
    output = {}
    for area in AREAS:
        reading = recent[area]
        end = pd.Timestamp(reading["observationEnd"])
        times = pd.date_range(end=end - pd.Timedelta(minutes=5), periods=36, freq="5min")
        rows = frame[frame.PriceArea.eq(area) & frame.time.isin(times)].copy()
        cols = ["OnshoreWindPower", "OffshoreWindPower"]
        if rows.groupby("time")[cols].nunique().gt(1).any().any():
            raise ValueError("Conflicting five-minute wind inputs")
        values = rows.drop_duplicates("time").set_index("time").reindex(times)[cols].to_numpy(float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Incomplete three-hour wind history")
        # Same documented DST source incidents excluded by hourly normalization.
        if any((pd.Timestamp(start) <= t < pd.Timestamp(stop)) for t in times for start, stop in (
            ("2026-03-29T00:00Z", "2026-03-29T03:00Z"),
            ("2026-03-30T07:00Z", "2026-03-30T10:00Z"),
        )):
            raise ValueError("Known source timestamp ambiguity")
        wind = values.sum(axis=1)
        output[area] = {**reading, "mean1h": float(wind[-12:].mean()),
            "mean3h": float(wind.mean()), "std1h": float(wind[-12:].std()),
            "std3h": float(wind.std()), "change1h": float(wind[-3:].mean() - wind[-15:-12].mean()),
            "offshoreFraction": float(values[-3:, 1].sum() / max(values[-3:].sum(), 1)),
            "ageMinutes": float((pd.Timestamp(at) - end).total_seconds() / 60)}
    return output


def features(history, future, names, readings, times, areas=AREAS):
    """Target-free features shared by historical replay and prospective scoring."""
    times = pd.DatetimeIndex(times)
    if times.tz is None or len(times) < 6 or any(times[1:] - times[:-1] != pd.Timedelta(hours=1)):
        raise ValueError("Expected consecutive timezone-aware target hours")
    output = []
    for ai, area in enumerate(areas):
        reading = readings[area]
        if pd.Timestamp(reading["selectedAt"]) >= times[0]:
            raise ValueError("Measurements must be selected before first target ends")
        columns = {
            "lead": np.arange(1, 7, dtype=float),
            "hoursFromMeasurement": np.asarray((times[:6] - pd.Timedelta(minutes=30) -
                pd.Timestamp(reading["midpoint"])).total_seconds() / 3600),
        }
        for name in ("levelMW", "previousMW", "trendMWPerHour", "mean1h", "mean3h", "std1h",
                     "std3h", "change1h", "offshoreFraction", "ageMinutes"):
            columns[name] = np.full(6, reading[name])
        for lag in (1, 3, 6, 24):
            columns[f"lag{lag}h"] = np.full(6, history[ai, -lag, 0])
        for i in range(3):
            for suffix in ("wind_speed_100m", "wind_sin", "wind_cos", "power_curve"):
                name = f"wx_{i}_{suffix}"
                value = future[ai, :6, names.index(name)]
                columns[name] = value
                columns[name + "_change"] = value - value[0]
        for name in ("hour_sin", "hour_cos", "year_sin", "year_cos"):
            columns[name] = future[ai, :6, names.index(name)]
        output.append(np.column_stack(list(columns.values())))
    array = np.asarray(output, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("Missing short-term model feature")
    return array, list(columns)


def predict(model, matrix, level):
    if isinstance(model, dict):
        delta = ((matrix - model["mean"]) / model["scale"]) @ np.array(model["coef"]) + model["intercept"]
    else:
        delta = model.predict(matrix, num_threads=2)
    return np.maximum(0, level + delta)
