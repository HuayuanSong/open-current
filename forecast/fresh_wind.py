"""Publish an independently archived wind update using current weather runs.

The daily ensemble still supplies solar/demand and the frozen comparisons.
Updated wind has its own version, issue times, error history and source evidence.
"""

import argparse
import gzip
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, publish, weather_sources
from .score import combine, run as baseline_score
from .trial.ledger import canonical_hash
from .wind_models import predict_wind
from .wind_nowcast import adjust
from .wind_update import measurements, recent_wind
from research.forecasting.weather_retrain import compare
from research.forecasting.weather_retrain.prospective import ensure_models

HERE = data.ROOT / "research/forecasting/fresh_wind"
CACHE = data.ROOT / ".cache/fresh-wind/live"


def configuration():
    selection = json.loads((HERE / "selection.json").read_text())
    if selection["resultsHash"] != compare.digest(HERE / "results.json"):
        raise ValueError("Wind selection and evaluation report differ")
    names = ["forecast/fresh_wind.py", "forecast/weather_sources.py", "forecast/wind_models.py",
             "forecast/wind_nowcast.py", "forecast/features.py", "forecast/score.py",
             "research/forecasting/fresh_wind/selection.json",
             "research/forecasting/weather_retrain/model-manifest.json"]
    return selection, canonical_hash({name: compare.digest(data.ROOT / name) for name in names})


def model_members(arrays, work):
    """Reuse this refresh's CPU workers only for exactly matching input arrays."""
    paths = [
        (data.ROOT / ".cache/forecast-serving/run/inputs.npz", data.ROOT / ".cache/forecast-serving/run"),
        (data.ROOT / ".cache/wind-update/live/evidence/scoring/baseline-inputs.npz",
         data.ROOT / ".cache/wind-update/live/evidence/scoring/baseline"),
    ]
    for input_path, folder in paths:
        if not input_path.exists():
            continue
        with np.load(input_path) as saved:
            matches = set(saved.files) == set(arrays) and all(
                np.array_equal(saved[k], arrays[k], equal_nan=True) for k in arrays)
        outputs = [folder / f"{family}.npz" for family in ("trees", "neural", "ttm")]
        if not matches or not all(p.exists() and p.stat().st_mtime_ns >= input_path.stat().st_mtime_ns for p in outputs):
            continue
        members = {}
        for path in outputs:
            with np.load(path) as saved:
                members.update({k: saved[k] for k in saved.files})
        return members, dict(reusedWorkers=True)
    work.mkdir(parents=True, exist_ok=True)
    path = work / "inputs.npz"
    np.savez_compressed(path, **arrays)
    _, members, timings = baseline_score(path, work)
    return members, dict(reusedWorkers=False, workers=timings)


def weighted_wind(predictions, weights):
    original = predictions["baseline"]
    result = original.copy()
    for ai, area in enumerate(data.AREAS):
        weight = weights[area]
        if not np.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError("Invalid DMI wind weight")
        if weight == 0:
            result[ai, :, 0] = predictions["ecmwf"][ai, :, 0]
        elif weight == 1:
            result[ai, :, 0] = predictions["dmi"][ai, :, 0]
        else:
            result[ai, :, 0] = ((1 - weight) * predictions["ecmwf"][ai, :, 0]
                               + weight * predictions["dmi"][ai, :, 0])
    return result


def issue(state, selection, fingerprint, work, *, diagnostic=False):
    at = data.now()
    arrays, meta = data.inputs(state, at)
    if pd.Timestamp(meta["times"][0]) <= at:
        return "waiting-for-source-hour"
    records = state.setdefault("windForecasts", [])
    for record in records:
        if record.get("sha256") != canonical_hash({k: v for k, v in record.items() if k != "sha256"}):
            raise ValueError("An issued wind forecast was modified")
    if any(r["modelVersion"] == selection["version"] and r.get("inferenceHash") != fingerprint for r in records):
        raise ValueError("Wind inference changed; use a new model version and retain issued records")
    times = pd.DatetimeIndex(meta["times"])
    for source in weather_sources.MODELS:
        weather_sources.refresh(state, source, data.now(), times)
    selected_at = data.now()
    needed = set()
    for weight in selection["dmiWeights"].values():
        if weight < 1:
            needed.add("ecmwf")
        if weight > 0:
            needed.add("dmi")
    entries = {source: weather_sources.select(state, source, selected_at, times) for source in sorted(needed)}
    raw, retrieved = measurements(selected_at)
    recent = recent_wind(raw, data.now())
    identifier = canonical_hash(dict(version=selection["version"], origin=meta["origin"],
        runs={source: entry["run"] for source, entry in entries.items()},
        observations={area: r["observationEnd"] for area, r in recent.items()}))
    if any(r["id"] == identifier for r in records):
        return "already-recorded"
    ensure_models()
    manifest = compare.verify_models()
    blend = json.loads((compare.ensure() / "selection.json").read_text())
    members, runtime = model_members(arrays, work / "scoring")
    predictions = dict(baseline=combine(members, arrays["area_ids"], blend)[0])
    inputs = {}
    for source, entry in entries.items():
        inputs[source] = weather_sources.future_inputs(arrays, entry["data"], times, meta["featureNames"])
        predictions[source] = predict_wind(inputs[source], members, manifest["featureNames"],
                                           compare.CACHE / "models/latest", blend)
    mixed = weighted_wind(predictions, selection["dmiWeights"])
    corrected = adjust(mixed, meta["times"], recent, selection["correctionRules"])
    np.testing.assert_array_equal(corrected[:, :, 1:], predictions["baseline"][:, :, 1:])
    np.testing.assert_array_equal(corrected[:, 6:, 0], mixed[:, 6:, 0])
    if not np.isfinite(corrected).all() or (corrected < 0).any() or np.any(corrected[:, :, 0] > np.array([15000, 10000])[:, None]):
        raise ValueError("Wind update failed plausibility checks")
    work.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(work / "inputs.npz", **arrays, **{f"{s}_future": v["future"] for s, v in inputs.items()})
    with gzip.open(work / "sources.json.gz", "wt") as output:
        json.dump(dict(weather=entries, electricity=raw, electricityRetrievedAt=retrieved), output, allow_nan=False)
    issued = data.now()
    if pd.Timestamp(meta["times"][0]) <= issued:
        raise ValueError("Wind scoring missed the first target hour; no retrospective issue")
    record = dict(id=identifier, modelVersion=selection["version"], issuedAt=issued.isoformat(),
        **{k: v for k, v in meta.items() if k != "featureNames"},
        baselineWeatherRun=meta["weatherRun"],
        areas={area: {target: corrected[ai, :, ti].astype(float).tolist() for ti, target in enumerate(data.TARGETS)}
               for ai, area in enumerate(data.AREAS)},
        weatherSources={s: {k: v for k, v in e.items() if k != "data"} for s, e in entries.items()},
        dmiWeights=selection["dmiWeights"], recentWind=recent, runtime=runtime,
        scope="Wind only. Solar and demand use the separately archived daily ensemble.",
        ranges=dict(version=selection["version"], issuedAt=issued.isoformat(), areas={}),
        rangeUnavailableReason="Prediction ranges are unavailable for this wind update because calibration checks did not pass.",
        inferenceHash=fingerprint, inputHash=compare.digest(work / "inputs.npz"),
        sourceHash=compare.digest(work / "sources.json.gz"), diagnosticOnly=diagnostic,
        githubRunId=os.environ.get("GITHUB_RUN_ID"))
    primary_weather = entries.get("ecmwf", entries.get("dmi"))
    record["weatherRun"] = primary_weather["run"]
    record["weatherRetrievedAt"] = primary_weather["retrievedAt"]
    record["weatherAvailableAt"] = primary_weather["availableAt"]
    record["sha256"] = canonical_hash(record)
    records.append(record)
    compare.write_json(work / "record.json", record)
    if not diagnostic and os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            output.write("issued=true\n")
    return "issued"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true", help="Write a diagnostic only to local cache")
    parser.add_argument("--state", type=Path, help="Local snapshots are permitted only for probes")
    args = parser.parse_args()
    if args.state and not args.probe:
        raise ValueError("Local state cannot create a public issue")
    selection, fingerprint = configuration()
    state = data.read_state(args.state or data.STATE_PATH)
    if args.probe:
        state["windForecasts"] = []
    work = CACHE / ("probe" if args.probe else "evidence")
    try:
        status = issue(state, selection, fingerprint, work, diagnostic=args.probe)
        attempt = dict(status=status, error=None)
    except Exception as error:
        attempt = dict(status="unavailable", error=str(error)[:300])
    checked = data.now()
    attempt["at"] = checked.isoformat()
    state["windServing"] = dict(version=selection["version"], lastAttempt=attempt,
                                dmiWeights=selection["dmiWeights"])
    if not args.probe:
        state["updatedAt"] = checked.isoformat()
        cutoff = checked - pd.Timedelta(days=35)
        state["windForecasts"] = [r for r in state.get("windForecasts", []) if pd.Timestamp(r["issuedAt"]) >= cutoff]
        data.write_state(state)
        baseline_status = json.loads(data.PUBLIC_PATH.read_text())
        publish.write_public(state, checked, baseline_status["modelVersion"], baseline_status.get("message"))
    summary = dict(**attempt, version=selection["version"], forecastsRetained=len(state.get("windForecasts", [])),
                   diagnosticOnly=args.probe, weather=state.get("availableWeatherStatus", {}))
    compare.write_json(work / "status.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if attempt["error"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
