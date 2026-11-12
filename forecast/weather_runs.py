"""Latest complete ECMWF runs, with provider availability evidence and a fallback.

This cache is separate from the daily weather history used during model training.
Fetching a run never changes an issued forecast or the prospective trial.
"""

import time

import numpy as np
import pandas as pd
import requests

from .features import CONFIG

MODEL = "ecmwf_ifs"
METADATA_URL = f"https://api.open-meteo.com/data/{MODEL}/static/meta.json"
FORECAST_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
POLICY = "latest-ecmwf-v1"
CONSISTENCY_BUFFER = pd.Timedelta(minutes=10)
MAX_AGE = pd.Timedelta(hours=48)


def available_run(metadata, at):
    """A model's initialization timestamp alone does not establish availability."""
    run = pd.to_datetime(metadata["last_run_initialisation_time"], unit="s", utc=True)
    available = pd.to_datetime(metadata["last_run_availability_time"], unit="s", utc=True)
    if pd.isna(run) or pd.isna(available) or available < run:
        raise ValueError("Invalid weather availability metadata")
    if run.hour not in (0, 6, 12, 18) or run != run.floor("h"):
        raise ValueError("Unexpected ECMWF initialization time")
    if available + CONSISTENCY_BUFFER > at or at - run > MAX_AGE:
        return None
    return run, available


def parameters(run):
    return dict(
        latitude=",".join(str(p["latitude"]) for p in CONFIG["locations"]),
        longitude=",".join(str(p["longitude"]) for p in CONFIG["locations"]),
        hourly=",".join(CONFIG["variables"]),
        models=MODEL,
        run=run.strftime("%Y-%m-%dT%H:%M"),
        forecast_days=5,
        timezone="UTC",
        wind_speed_unit="ms",
        cell_selection="nearest",
    )


def validate_points(points, required_times=None):
    if not isinstance(points, list) or len(points) != len(CONFIG["locations"]):
        raise ValueError("Unexpected weather locations")
    for point in points:
        hourly = point["hourly"]
        times = pd.to_datetime(hourly["time"], utc=True)
        if not len(times) or times.has_duplicates or not times.is_monotonic_increasing:
            raise ValueError("Invalid weather time axis")
        if any(len(hourly[name]) != len(times) for name in CONFIG["variables"]):
            raise ValueError("Incomplete weather variables")
        if required_times is not None:
            frame = pd.DataFrame({name: hourly[name] for name in CONFIG["variables"]}, index=times)
            if not np.isfinite(frame.reindex(required_times).to_numpy(dtype=float)).all():
                raise ValueError("Incomplete weather forecast horizon")


def refresh(state, at, request):
    """Cache a published run once; failures leave previous complete runs intact."""
    cached = state.setdefault("weatherRuns", {})
    status = {"checkedAt": at.isoformat(), "policy": POLICY, "error": None}
    begin = time.monotonic()
    try:
        metadata = request(METADATA_URL)
        candidate = available_run(metadata, at)
        if candidate is not None:
            run, available = candidate
            key = run.isoformat()
            status["latestAvailableRun"] = key
            if key not in cached:
                points = request(FORECAST_URL, parameters(run))
                origin = (at - pd.Timedelta(minutes=15)).floor("h")
                required_times = pd.date_range(origin + pd.Timedelta(hours=1), periods=48, freq="h")
                validate_points(points, required_times)
                cached[key] = {
                    "model": MODEL,
                    "run": key,
                    "availableAt": available.isoformat(),
                    "retrievedAt": pd.Timestamp.now(tz="UTC").isoformat(),
                    "data": points,
                }
    except (requests.RequestException, ValueError, TypeError, KeyError) as error:
        status["error"] = str(error)[:240]
        print(f"Latest weather unavailable; retaining complete cached runs: {error}", flush=True)
    finally:
        time.sleep(max(0, 1.5 - (time.monotonic() - begin)))
    state["weatherRunRefresh"] = status
    retained = {}
    for key, value in cached.items():
        try:
            run = pd.Timestamp(key)
            if run.tzinfo is not None and at - MAX_AGE <= run <= at:
                retained[key] = value
        except (ValueError, TypeError):
            continue
    state["weatherRuns"] = retained
    return status


def candidates(state, at):
    """Only cached runs with recorded availability and retrieval before selection."""
    for key, entry in state.get("weatherRuns", {}).items():
        try:
            run = pd.Timestamp(key)
            available = pd.Timestamp(entry["availableAt"])
            retrieved = pd.Timestamp(entry["retrievedAt"])
            if any(t.tzinfo is None or pd.isna(t) for t in (run, available, retrieved)):
                continue
            if (
                entry.get("model") == MODEL and entry.get("run") == key
                and at - MAX_AGE <= run <= available
                and available + CONSISTENCY_BUFFER <= retrieved <= at
            ):
                yield run, entry
        except (ValueError, TypeError, KeyError):
            continue
