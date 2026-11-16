"""Record fresher-weather and five-minute wind candidates beside production.

Runs during ordinary refreshes, with no narrow daily scheduling window. Issued
comparisons, input hashes and first eligible outcomes are retained independently
of both existing trials. This module never promotes a model automatically.
"""

import argparse
import gzip
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, intervals, weather_runs
from .features import AREAS
from .score import combine
from .trial.ledger import canonical_hash
from .wind_nowcast import adjust, recent_wind
from research.forecasting.weather_retrain import compare
from research.forecasting.weather_retrain.models import predict
from research.forecasting.weather_retrain.prospective import ensure_models

HERE = data.ROOT / "research/forecasting/wind_update"
CACHE = data.ROOT / ".cache/wind-update/live"
PUBLIC = data.ROOT / "dist/data/wind-update.json"


def configuration():
    protocol = json.loads((HERE / "protocol.json").read_text())
    selection = json.loads((HERE / "selection.json").read_text())
    if selection["protocolHash"] != compare.digest(HERE / "protocol.json"):
        raise ValueError("Wind selection and protocol differ")
    sources = ["forecast/wind_update.py", "forecast/wind_nowcast.py", "forecast/weather_runs.py",
               "forecast/data.py", "forecast/intervals.py", "forecast/score.py", "forecast/features.py",
               "research/forecasting/wind_update/protocol.json", "research/forecasting/wind_update/selection.json",
               "research/forecasting/weather_retrain/model-manifest.json"]
    fingerprint = canonical_hash({name: compare.digest(data.ROOT / name) for name in sources})
    return protocol, selection, fingerprint


def initialize(state, protocol, fingerprint):
    trial = state.setdefault("windUpdate", dict(version=protocol["version"], fingerprint=fingerprint,
                                               records=[], observations={}, lastAttempt=None))
    if (trial["version"] == protocol["version"] and trial["fingerprint"] != fingerprint
            and not trial["records"] and not trial["observations"]):
        # A deployment repair before the first issue changes no predictions.
        # Keep its previous fingerprint and failed attempt as an audit trail.
        trial.setdefault("preIssueRepairs", []).append(dict(
            fingerprint=trial["fingerprint"], lastAttempt=trial["lastAttempt"],
            replacedAt=data.now().isoformat()))
        trial["fingerprint"] = fingerprint
    if trial["version"] != protocol["version"] or trial["fingerprint"] != fingerprint:
        raise ValueError("Wind candidate changed; retain existing evidence and start a new version")
    for record in trial["records"]:
        if record["sha256"] != canonical_hash({k: v for k, v in record.items() if k != "sha256"}):
            raise ValueError("An issued wind comparison was modified")
    return trial


def freeze_observations(trial, hourly, at):
    wanted = {t for r in trial["records"] for t in r["times"]}
    for row in hourly:
        if row["time"] not in wanted or row["area"] not in AREAS:
            continue
        age = (at - pd.Timestamp(row["time"])).total_seconds() / 3600
        value = row.get("wind")
        if 24 <= age <= 72 and value is not None and np.isfinite(value) and value >= 0:
            trial["observations"].setdefault(row["time"], {}).setdefault(
                row["area"], dict(value=float(value), capturedAt=at.isoformat()))


def measurements(at):
    start = (at - pd.Timedelta(hours=3)).tz_convert("Europe/Copenhagen").date()
    collected = data.collected_electricity(start, at)
    if collected is not None:
        return collected["records"], collected["_retrieved_at"]
    payload = data.request("https://api.energidataservice.dk/dataset/ElectricityProdex5MinRealtime",
                           dict(start="now-PT3H", end="now", limit=300, sort="Minutes5UTC asc"))
    if not payload.get("records") or payload.get("total") != len(payload["records"]):
        raise ValueError("Incomplete five-minute wind response")
    return payload["records"], data.now().isoformat()


def paired_predictions(baseline, fresh, work):
    """Reuse this refresh's expensive model outputs only for identical inputs."""
    manifest = compare.verify_models()
    production = data.ROOT / ".cache/forecast-serving/run"
    input_path = production / "inputs.npz"
    reusable = False
    if input_path.exists():
        with np.load(input_path) as saved:
            reusable = set(saved.files) == set(baseline) and all(
                np.array_equal(saved[k], baseline[k], equal_nan=True) for k in baseline)
    paths = [production / f"{family}.npz" for family in ("trees", "neural", "ttm")]
    # Worker outputs must have been written after the matching input file.
    reusable = reusable and all(p.exists() and p.stat().st_mtime_ns >= input_path.stat().st_mtime_ns for p in paths)
    if not reusable:
        outputs, timings = compare.score_pair(baseline, fresh, work)
        return outputs["baseline"], outputs["latest"], dict(reusedProduction=False, workers=timings)
    members = {}
    for path in paths:
        with np.load(path) as output:
            members.update({k: output[k] for k in output.files})
    blend = json.loads((compare.ensure() / "selection.json").read_text())
    original = combine(members, baseline["area_ids"], blend)[0]
    refitted = predict(fresh, manifest["featureNames"], compare.CACHE / "models/latest")
    changed = combine({**members, **refitted}, baseline["area_ids"], blend)[0]
    return original, changed, dict(reusedProduction=True)


def issue(state, trial, protocol, selection, work, *, diagnostic=False):
    at = data.now()
    if not pd.Timestamp(protocol["prospectiveStart"], tz="UTC") <= at < pd.Timestamp(protocol["prospectiveEndExclusive"], tz="UTC"):
        return "outside-recording-period"
    baseline, base_meta = data.inputs(state, at)
    if pd.Timestamp(base_meta["times"][0]) <= at:
        return "no-future-first-hour"
    if any(r["origin"] == base_meta["origin"] for r in trial["records"]):
        return "already-recorded"
    ensure_models()
    weather_status = weather_runs.refresh(state, data.now(), data.request)
    selected_at = data.now()
    fresh, fresh_meta = data.inputs(state, selected_at, latest_weather=True)
    if fresh_meta["weatherSelection"] != "recorded-availability":
        raise ValueError("No complete availability-verified weather run")
    if fresh_meta["origin"] != base_meta["origin"]:
        raise ValueError("Electricity origin changed while collecting inputs")
    for key in baseline:
        if key != "future":
            np.testing.assert_array_equal(baseline[key], fresh[key])
    raw, retrieved = measurements(selected_at)
    observation_at = data.now()
    recent = recent_wind(raw, observation_at, protocol["observationBufferMinutes"])
    work.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(work / "inputs.npz", **baseline, latest_future=fresh["future"])
    with gzip.open(work / "sources.json.gz", "wt") as output:
        json.dump(dict(weather=state["weatherRuns"], electricity=raw, retrievedAt=retrieved), output, allow_nan=False)
    original, refitted, runtime = paired_predictions(baseline, fresh, work / "scoring")
    wind_only = original.copy()
    wind_only[:, :, 0] = refitted[:, :, 0]
    corrected = adjust(wind_only, base_meta["times"], recent, selection["rules"])
    # These guards are invariants, not evaluation-dependent target selection.
    np.testing.assert_array_equal(corrected[:, :, 1:], original[:, :, 1:])
    np.testing.assert_array_equal(corrected[:, 6:, 0], wind_only[:, 6:, 0])
    issued = data.now()
    if pd.Timestamp(base_meta["times"][0]) <= issued:
        raise ValueError("Scoring missed the first target; no retrospective issue")
    if not np.isfinite(corrected).all() or (corrected < 0).any() or np.any(corrected[:, :, 0] > np.array([15000, 10000])[:, None]):
        raise ValueError("Wind candidate failed plausibility checks")
    predictions = {name: {area: values[ai, :, 0].astype(float).tolist() for ai, area in enumerate(AREAS)}
                   for name, values in [("baseline", original), ("fresh", wind_only), ("corrected", corrected)]}
    ranges = {}
    for ai, area in enumerate(AREAS):
        lo, hi = intervals.bounds(corrected[ai, :, 0], np.ones(48, bool), "wind", selection["ranges"][area])
        ranges[area] = dict(lower=lo.tolist(), upper=hi.tolist())
    baseline_bands = intervals.build(original, np.ones((2, 48), bool), issued.isoformat(), state, base_meta["origin"])
    record = dict(origin=base_meta["origin"], issuedAt=issued.isoformat(), times=base_meta["times"],
                  baselineWeatherRun=base_meta["weatherRun"], weatherRun=fresh_meta["weatherRun"],
                  weatherAvailableAt=fresh_meta["weatherAvailableAt"], weatherRetrievedAt=fresh_meta["weatherRetrievedAt"],
                  weatherStatus=weather_status, electricityRetrievedAt=retrieved, recent=recent,
                  predictions=predictions, ranges=ranges,
                  baselineRanges={area: baseline_bands["areas"].get(area, {}).get("wind") for area in AREAS},
                  nominalCoverage=protocol["nominalCoverage"], diagnosticOnly=diagnostic,
                  calibratedBefore=selection["calibratedBefore"], fingerprint=trial["fingerprint"],
                  inputHash=compare.digest(work / "inputs.npz"), sourceHash=compare.digest(work / "sources.json.gz"),
                  githubRunId=os.environ.get("GITHUB_RUN_ID"), runtime=runtime)
    record["sha256"] = canonical_hash(record)
    trial["records"].append(record)
    compare.write_json(work / "record.json", record)
    if not diagnostic and os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            output.write("issued=true\n")
    return "issued"


def report(trial, protocol, at):
    complete = [r for r in trial["records"] if not r.get("diagnosticOnly") and all(
        area in trial["observations"].get(t, {}) for t in r["times"] for area in AREAS)]
    result = dict(version=protocol["version"], checkedAt=at.isoformat(), automaticPromotion=False,
                  status="collecting", reviewAfter=protocol["reviewAfter"],
                  issuedOrigins=len(trial["records"]), completeOrigins=len(complete),
                  distinctDays=len({r["origin"][:10] for r in complete}), lastAttempt=trial["lastAttempt"],
                  latest=trial["records"][-1] if trial["records"] else None, metrics={},
                  interpretation="Separate experimental wind candidates, not the dashboard forecast. Scores use complete paired 48-hour trajectories and first observations captured 24–72 hours later. Overlapping origins are dependent.")
    for area in AREAS:
        if not complete:
            continue
        truth = np.array([[trial["observations"][t][area]["value"] for t in r["times"]] for r in complete])
        metrics = {}
        for variant in ("baseline", "fresh", "corrected"):
            error = np.abs(np.array([r["predictions"][variant][area] for r in complete]) - truth)
            metrics[variant] = dict(maeMW=float(error.mean()), leadMAE=[float(error[:, lo:hi].mean()) for lo, hi in intervals.BANDS])
        lower = np.array([r["ranges"][area]["lower"] for r in complete])
        upper = np.array([r["ranges"][area]["upper"] for r in complete])
        metrics["corrected"].update(rangeCoverage=float(np.mean((lower <= truth) & (truth <= upper))),
                                     intervalScore=float(intervals.interval_score(truth, lower, upper).mean()))
        if all(r.get("baselineRanges", {}).get(area) for r in complete):
            base_lower = np.array([r["baselineRanges"][area]["lower"] for r in complete])
            base_upper = np.array([r["baselineRanges"][area]["upper"] for r in complete])
            if np.isfinite(base_lower).all() and np.isfinite(base_upper).all():
                metrics["baseline"].update(intervalScore=float(intervals.interval_score(truth, base_lower, base_upper).mean()))
        result["metrics"][area] = metrics
    checks = {}
    for area, metrics in result["metrics"].items():
        base, candidate = metrics["baseline"], metrics["corrected"]
        checks[area] = dict(
            improvementPct=100 * (1 - candidate["maeMW"] / base["maeMW"]) if base["maeMW"] else None,
            pointErrorPass=bool(candidate["maeMW"] <= base["maeMW"] * (1 - protocol["minimumWindImprovementPct"] / 100)),
            leadBandsPass=bool(all(c <= b * (1 + protocol["maximumLeadRegressionPct"] / 100)
                                  for c, b in zip(candidate["leadMAE"], base["leadMAE"]))),
            rangeCoveragePass=candidate["rangeCoverage"] >= protocol["minimumRangeCoverage"],
            intervalScorePass=(candidate["intervalScore"] <= base["intervalScore"] * (1 + protocol["maximumIntervalScoreRegressionPct"] / 100)
                               if "intervalScore" in base else None))
    result["checks"] = checks
    if at >= pd.Timestamp(protocol["reviewAfter"]):
        enough = len(complete) >= protocol["minimumCompleteOrigins"] and result["distinctDays"] >= protocol["minimumDistinctDays"]
        result["status"] = "ready-for-manual-review" if enough else "insufficient-evidence"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true", help="Write diagnostics to cache only")
    parser.add_argument("--state", type=Path, help="Local state is allowed only for a diagnostic probe")
    args = parser.parse_args()
    if args.state and not args.probe:
        raise ValueError("A local snapshot cannot create a prospective forecast")
    protocol, selection, fingerprint = configuration()
    state = data.read_state(args.state or data.STATE_PATH)
    if args.probe:
        state.pop("windUpdate", None)
    trial = initialize(state, protocol, fingerprint)
    freeze_observations(trial, state["hourly"], data.now())
    work = CACHE / ("probe" if args.probe else "evidence")
    try:
        status = issue(state, trial, protocol, selection, work, diagnostic=args.probe)
        trial["lastAttempt"] = dict(at=data.now().isoformat(), status=status, diagnosticOnly=args.probe)
    except Exception as error:
        trial["lastAttempt"] = dict(at=data.now().isoformat(), status="unavailable", error=str(error)[:400], diagnosticOnly=args.probe)
    checked = data.now()
    summary = report(trial, protocol, checked)
    compare.write_json(work / "status.json", summary)
    if not args.probe:
        state["updatedAt"] = checked.isoformat()
        data.write_state(state)
        compare.write_json(PUBLIC, summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "latest"}, indent=2), flush=True)
    if trial["lastAttempt"]["status"] == "unavailable":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
