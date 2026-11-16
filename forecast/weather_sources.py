"""Availability-verified weather for the public wind update.

Separate from the frozen daily inputs and existing experimental ledgers. Every
selection retains its model, initialization, publication and retrieval times.
"""

import time

import numpy as np
import pandas as pd
import requests

from . import data, weather_runs
from .features import AREAS, CONFIG, calendar, weather_features

MODELS = {"ecmwf": "ecmwf_ifs", "dmi": "dmi_harmonie_arome_europe"}
BUFFER = pd.Timedelta(minutes=10)
MAX_AGE = pd.Timedelta(hours=48)


def parameters(source, run):
    return {**weather_runs.parameters(run), "models": MODELS[source],
            "forecast_days": 3 if source == "dmi" else 5}


def eligible(entry, source, at):
    try:
        run, available, retrieved = [pd.Timestamp(entry[k]) for k in ("run", "availableAt", "retrievedAt")]
        return (all(t.tzinfo is not None and not pd.isna(t) for t in (run, available, retrieved))
                and entry["model"] == MODELS[source]
                and at - MAX_AGE <= run <= available
                and available + BUFFER <= retrieved <= at)
    except (KeyError, ValueError, TypeError):
        return False


def refresh(state, source, at, times, request=data.request):
    cache = state.setdefault("availableWeather", {}).setdefault(source, {})
    # Share recorded ECMWF downloads without changing the earlier trial cache.
    if source == "ecmwf":
        for key, entry in state.get("weatherRuns", {}).items():
            if eligible(entry, source, at):
                cache.setdefault(key, entry)
    status = dict(checkedAt=at.isoformat(), model=MODELS[source], error=None)
    begin = time.monotonic()
    try:
        metadata = request(f"https://api.open-meteo.com/data/{MODELS[source]}/static/meta.json")
        run = pd.to_datetime(metadata["last_run_initialisation_time"], unit="s", utc=True)
        available = pd.to_datetime(metadata["last_run_availability_time"], unit="s", utc=True)
        interval = 3 if source == "dmi" else 6
        if (pd.isna(run) or pd.isna(available) or available < run or run != run.floor("h")
                or run.hour % interval):
            raise ValueError("Invalid model publication metadata")
        status.update(latestRun=run.isoformat(), availableAt=available.isoformat())
        if at - MAX_AGE <= run and available + BUFFER <= at and run.isoformat() not in cache:
            points = request(weather_runs.FORECAST_URL, parameters(source, run))
            weather_runs.validate_points(points, times)
            cache[run.isoformat()] = dict(model=MODELS[source], run=run.isoformat(),
                availableAt=available.isoformat(), retrievedAt=data.now().isoformat(), data=points)
    except (requests.RequestException, ValueError, TypeError, KeyError) as error:
        status["error"] = str(error)[:240]
    finally:
        time.sleep(max(0, 1.5 - (time.monotonic() - begin)))
    checked = data.now()
    state["availableWeather"][source] = {k: v for k, v in cache.items() if eligible(v, source, checked)}
    state.setdefault("availableWeatherStatus", {})[source] = status
    return status


def select(state, source, at, times):
    entries = state.get("availableWeather", {}).get(source, {}).values()
    for entry in sorted(entries, key=lambda e: e.get("run", ""), reverse=True):
        if not eligible(entry, source, at):
            continue
        try:
            weather_runs.validate_points(entry["data"], times)
            return entry
        except (ValueError, TypeError, KeyError):
            continue
    raise ValueError(f"No complete availability-verified {source.upper()} forecast")


def future_inputs(baseline, points, times, names):
    """Change only future weather; preserve the trained historical feature policy."""
    weather_runs.validate_points(points, times)
    frames = []
    for ai, _ in enumerate(AREAS):
        wx = weather_features(points[ai * 3:ai * 3 + 3], times)[0]
        frame = pd.concat([wx, calendar(times)], axis=1)
        if list(frame.columns) != names:
            raise ValueError("Weather feature schema differs from the fitted models")
        frames.append(frame.to_numpy(dtype=np.float32))
    future = np.asarray(frames)
    if future.shape != baseline["future"].shape or not np.isfinite(future).all():
        raise ValueError("Invalid future weather inputs")
    return {**baseline, "future": future}
