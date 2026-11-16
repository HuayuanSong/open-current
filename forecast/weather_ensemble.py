"""Archive current weather-ensemble means/spreads with honest availability times.

The provider's stitched historical series is not an issuance-time archive. Only
snapshots actually retrieved before a challenger is issued may be used here.
"""

import numpy as np
import pandas as pd
import requests

from . import data
from .features import CONFIG
from .trial.ledger import canonical_hash

URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
MODEL = "ecmwf_ifs025_ensemble_mean"
VARIABLES = ("wind_speed_100m", "wind_speed_100m_spread")


def parameters():
    locations = CONFIG["locations"]
    return dict(latitude=",".join(str(p["latitude"]) for p in locations),
                longitude=",".join(str(p["longitude"]) for p in locations),
                models=MODEL, hourly=",".join(VARIABLES), wind_speed_unit="ms",
                forecast_days=4, timezone="UTC", cell_selection="nearest")


def values(points, times):
    if not isinstance(points, list) or len(points) != len(CONFIG["locations"]):
        raise ValueError("Expected six ensemble weather locations")
    matrices = []
    for point, location in zip(points, CONFIG["locations"]):
        if any(abs(point[key] - location[key]) > .4 for key in ("latitude", "longitude")):
            raise ValueError("Ensemble weather location mismatch")
        if point.get("utc_offset_seconds") != 0 or any(point["hourly_units"].get(v) != "m/s" for v in VARIABLES):
            raise ValueError("Unexpected ensemble weather units")
        frame = pd.DataFrame(point["hourly"])
        frame.index = pd.to_datetime(frame.pop("time"), utc=True)
        if frame.index.has_duplicates:
            raise ValueError("Duplicate ensemble timestamps")
        matrix = frame.reindex(times)[list(VARIABLES)].to_numpy(float)
        if not np.isfinite(matrix).all() or (matrix < 0).any() or (matrix > 100).any():
            raise ValueError("Incomplete or invalid ensemble wind data")
        matrices.append(matrix)
    # Averaged local spreads are a model input, not the uncertainty of a
    # regional spatial average (member correlations are unavailable).
    combined = np.asarray(matrices).reshape(2, 3, len(times), 2).mean(axis=1)
    return {area: {"meanWindMS": combined[ai, :, 0].tolist(),
                   "meanLocalSpreadMS": combined[ai, :, 1].tolist()}
            for ai, area in enumerate(data.AREAS)}


def snapshot(state, times, at):
    cached = state.get("weatherEnsemble")
    if cached:
        retrieved = pd.Timestamp(cached["retrievedAt"])
        if at - pd.Timedelta(hours=1) <= retrieved <= at and cached.get("parameters") == parameters():
            values(cached["data"], times)
            return cached
    # A short, bounded optional request; a slow ensemble endpoint must not hold
    # up the main forecast or acquire a fabricated successful retrieval time.
    response = requests.get(URL, params=parameters(), timeout=(5, 20))
    response.raise_for_status()
    points = response.json()
    values(points, times)
    record = dict(model=MODEL, retrievedAt=data.now().isoformat(), modelRun=None,
        availabilityEvidence="retrieved-snapshot", parameters=parameters(), data=points,
        scope="Local ensemble means/spreads; no exact initialization time supplied by this endpoint.")
    record["sha256"] = canonical_hash(record)
    state["weatherEnsemble"] = record
    return record
