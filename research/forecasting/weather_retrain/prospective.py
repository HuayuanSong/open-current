"""Independent prospective evidence, stored as append-only GitHub artifacts."""

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile

import numpy as np
import pandas as pd
import requests

from forecast import data, weather_runs
from forecast.features import AREAS, TARGETS
from forecast.trial.ledger import canonical_hash
from .common import CACHE, HERE, ROOT, PROTOCOL, boundary, digest, write_json
from .compare import score_pair, verify_models


def eligible_origin(at):
    at = pd.Timestamp(at)
    if at.tzinfo is None:
        raise ValueError("Issue time requires a timezone")
    at = at.tz_convert("UTC")
    if not boundary("prospectiveStart") <= at < boundary("prospectiveEndExclusive"):
        return None
    low, high = PROTOCOL["issueWindowMinutes"]
    if at.hour in PROTOCOL["prospectiveHours"] and low <= at.minute < high:
        return at.floor("h")
    return None


def observation_snapshot(hourly, at):
    result = []
    for row in hourly:
        age = (at - pd.Timestamp(row["time"])).total_seconds() / 3600
        if not 24 <= age <= 72 or row["area"] not in AREAS:
            continue
        for target in TARGETS:
            value = row.get(target)
            if value is not None and np.isfinite(value) and value >= 0 and (target != "load" or value > 0):
                result.append(dict(time=row["time"], area=row["area"], target=target, value=float(value), capturedAt=at.isoformat()))
    return result


def ensure_models():
    manifest = CACHE / "models/manifest.json"
    if not manifest.exists():
        release = json.loads((HERE / "model-release.json").read_text())
        response = requests.get(release["url"], timeout=90)
        response.raise_for_status()
        if len(response.content) != release["bytes"] or hashlib.sha256(response.content).hexdigest() != release["sha256"]:
            raise ValueError("Refit release checksum mismatch")
        folder = CACHE / "models"
        folder.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as archive:
            archive.extractall(folder, filter="data")
    expected = json.loads((HERE / "model-manifest.json").read_text())
    if json.loads(manifest.read_text()) != expected:
        raise ValueError("Unexpected refitted model bundle")
    return verify_models()


def verify_inference():
    lock = HERE / "inference-lock.json"
    for name, expected in json.loads(lock.read_text()).items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Frozen prospective inference changed: {name}")
    return digest(lock)


def serialize(values):
    return {area: {target: values[ai, :, ti].astype(float).tolist()
                  for ti, target in enumerate(TARGETS)} for ai, area in enumerate(AREAS)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true", help="Score now for an explicitly non-prospective smoke test")
    parser.add_argument("--state", type=Path, help="Use a local state for a probe; never permitted for prospective issuance")
    args = parser.parse_args()
    if args.state and not args.probe:
        raise ValueError("Prospective runs require fresh online observations")
    work = CACHE / ("probe" if args.probe else "evidence")
    work.mkdir(parents=True, exist_ok=True)
    # Each workflow run starts a new directory. Never replace an issued local record.
    if (work / "record.json").exists() and not args.probe:
        raise ValueError("An evidence record already exists; use a clean checkout/work directory")
    at = data.now()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    attempt = dict(checkedAt=at.isoformat(), protocolHash=digest(HERE / "protocol.json"), diagnosticOnly=args.probe,
                   githubRunId=os.environ.get("GITHUB_RUN_ID"), codeRevision=revision,
                   workflowRevision=os.environ.get("GITHUB_SHA"))
    try:
        if args.state:
            state = data.read_state(args.state)
        else:
            response = requests.get("https://huayuansong.github.io/open-current/data/forecast-state.json.gz", timeout=60)
            response.raise_for_status()
            state = json.loads(gzip.decompress(response.content))
            data.electricity(state, at)
        captured = data.now()
        write_json(work / "observations.json", observation_snapshot(state["hourly"], captured))
        attempt["observationsHash"] = digest(work / "observations.json")
        origin = eligible_origin(captured)
        if origin is None and not args.probe:
            attempt["status"] = "outside-issue-window"
            return
        ensure_models()
        lock_hash = verify_inference()
        data.weather(state, captured)
        weather_status = weather_runs.refresh(state, data.now(), data.request)
        selected_at = data.now()
        base, base_meta = data.inputs(state, selected_at)
        fresh, fresh_meta = data.inputs(state, selected_at, latest_weather=True)
        if not args.probe and pd.Timestamp(base_meta["origin"]) != origin:
            raise ValueError("Complete electricity hour is not available for the planned origin")
        if fresh_meta["weatherSelection"] != "recorded-availability":
            raise ValueError("No availability-verified newer weather run; preserve missing issue")
        names = json.loads((HERE / "model-manifest.json").read_text())["featureNames"]["future"]
        if base_meta["featureNames"] != names or fresh_meta["featureNames"] != names:
            raise ValueError("Prospective feature schema changed")
        for key in base:
            if key != "future":
                np.testing.assert_array_equal(base[key], fresh[key])
        np.savez_compressed(work / "inputs.npz", **base, latest_future=fresh["future"])
        with gzip.open(work / "weather-source.json.gz", "wt") as output:
            json.dump(state["weatherRuns"], output, allow_nan=False)
        predictions, timings = score_pair(base, fresh, CACHE / ("probe-scoring" if args.probe else "prospective-scoring"))
        issued = data.now()
        if not args.probe and (eligible_origin(issued) != origin or pd.Timestamp(base_meta["times"][0]) <= issued):
            raise ValueError("Scoring finished after the issue deadline; cannot backfill")
        limits = np.array([[15000, 15000, 20000], [10000, 10000, 15000]])[:, None, :]
        if any(np.any(p < 0) or np.any(p > limits) or not np.isfinite(p).all() for p in predictions.values()):
            raise ValueError("Implausible candidate forecast")
        record = dict(**attempt, issuedAt=issued.isoformat(), origin=base_meta["origin"], times=base_meta["times"],
                      modelManifestHash=digest(HERE / "model-manifest.json"), inferenceLockHash=lock_hash,
                      inputHash=digest(work / "inputs.npz"), sourceHash=digest(work / "weather-source.json.gz"),
                      baseline=base_meta, latest=fresh_meta, weatherRefresh=weather_status,
                      predictions={name: serialize(values) for name, values in predictions.items()}, runtime=timings)
        record["sha256"] = canonical_hash(record)
        write_json(work / "record.json", record)
        attempt["status"] = "diagnostic-scored" if args.probe else "issued"
        attempt["origin"] = record["origin"]
    except Exception as error:
        attempt["status"] = "unavailable"
        attempt["error"] = str(error)[:400]
        raise
    finally:
        write_json(work / "attempt.json", attempt)
        print(json.dumps(attempt), flush=True)


if __name__ == "__main__":
    main()
