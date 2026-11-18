"""Record ramp and uncertainty challengers separately from all public forecasts."""

import argparse
import gzip
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, weather_ensemble, wind_ranges, wind_short_term
from .intervals import interval_score
from .trial.ledger import canonical_hash
from research.forecasting.weather_retrain.common import digest, write_json

HERE = wind_short_term.HERE
CACHE = data.ROOT / ".cache/wind-refinement/live"
WIND_EVIDENCE = data.ROOT / ".cache/fresh-wind/live/evidence"
METHODS = ("absolute", "signed", "power", "ensemble")


def configuration():
    protocol = json.loads((HERE / "protocol.json").read_text())
    paths = [HERE / "protocol.json", HERE / "selection.json"] + [data.ROOT / "forecast" / name for name in
        ("wind_refinement.py", "wind_short_term.py", "wind_ranges.py", "weather_ensemble.py")]
    return protocol, canonical_hash({str(p.relative_to(data.ROOT)): digest(p) for p in paths})


def completed(records, state, at, *, horizon=48):
    actual = {(r["area"], r["time"]): r["wind"] for r in state["hourly"]}
    unique = {}
    for record in sorted(records, key=lambda r: r["issuedAt"]):
        if pd.Timestamp(record["times"][horizon-1]) + pd.Timedelta(minutes=15) > at:
            continue
        values = np.array([[actual.get((area, t), np.nan) for t in record["times"][:horizon]] for area in data.AREAS], float)
        if not np.isfinite(values).all():
            continue
        # One first-issued record per six-hour origin bucket. Half-hour refreshes
        # must not masquerade as many independent evaluation cases.
        bucket = pd.Timestamp(record["origin"]).floor("6h").isoformat()
        unique.setdefault(bucket, (record, values))
    return list(unique.values())


def calibrate(record, previous, state, protocol):
    at = pd.Timestamp(record["issuedAt"])
    eligible = [r for r in previous if r["baselineVersion"] == record["baselineVersion"]
                and r["version"] == record["version"] and r["origin"] >= (at - pd.Timedelta(days=14)).isoformat()]
    prior = completed(eligible, state, at)
    result = {}
    for method in METHODS:
        matched = [(r, y) for r, y in prior if method != "ensemble" or r.get("weather")]
        if len(matched) < protocol["minimumCalibrationOrigins"] or (method == "ensemble" and not record.get("weather")):
            continue
        areas = {}
        for ai, area in enumerate(data.AREAS):
            predictions = [r["baseline"][area] for r, _ in matched]
            scales = [wind_ranges.scale(method, r["baseline"][area], area, weather=(r.get("weather") or {}).get(area)) for r, _ in matched]
            rule = wind_ranges.fit([y[ai] for _, y in matched], predictions, scales, method)
            current_scale = wind_ranges.scale(method, record["baseline"][area], area, weather=(record.get("weather") or {}).get(area))
            lower, upper = wind_ranges.bounds(record["baseline"][area], current_scale, rule)
            areas[area] = dict(lower=lower.tolist(), upper=upper.tolist(), rule=rule)
        result[method] = dict(areas=areas, calibratedBefore=record["issuedAt"],
            lastCalibrationTarget=max(r["times"][-1] for r, _ in matched),
            calibrationIds=[r["id"] for r, _ in matched])
    return result


def issue(state, work, protocol, fingerprint):
    if not (WIND_EVIDENCE / "record.json").exists():
        return "no-new-public-wind-issue"
    base = json.loads((WIND_EVIDENCE / "record.json").read_text())
    at = data.now()
    if base.get("diagnosticOnly") or not any(r["id"] == base["id"] for r in state.get("windForecasts", [])):
        return "no-matching-public-wind-issue"
    if base.get("sha256") != canonical_hash({k: v for k, v in base.items() if k != "sha256"}):
        raise ValueError("Public wind evidence checksum differs")
    if any(area not in base["areas"] for area in data.AREAS):
        return "waiting-for-both-regions"
    if pd.Timestamp(base["times"][0]) <= at or at - pd.Timestamp(base["issuedAt"]) > pd.Timedelta(minutes=15):
        return "public-wind-issue-too-old"
    ledger = state.setdefault("windResearch", {})
    previous = ledger.setdefault("records", [])
    for r in previous:
        if r.get("sha256") != canonical_hash({k: v for k, v in r.items() if k != "sha256"}):
            raise ValueError("An issued challenger was modified")
        if r["version"] == protocol["version"] and r["inferenceHash"] != fingerprint:
            raise ValueError("Challenger changed without a new version")
    identifier = f"{protocol['version']}/{base['id']}"
    if any(r["id"] == identifier for r in previous):
        return "already-recorded"
    if digest(WIND_EVIDENCE / "inputs.npz") != base["inputHash"] or digest(WIND_EVIDENCE / "sources.json.gz") != base["sourceHash"]:
        raise ValueError("Public wind inputs/source checksums differ")
    with np.load(WIND_EVIDENCE / "inputs.npz") as saved:
        arrays = dict(saved)
    with gzip.open(WIND_EVIDENCE / "sources.json.gz", "rt") as source:
        sources = json.load(source)
    baseline = {area: base["areas"][area]["wind"] for area in data.AREAS}
    record = dict(id=identifier, version=protocol["version"], inferenceHash=fingerprint,
        baselineId=base["id"], baselineVersion=base["modelVersion"], origin=base["origin"], times=base["times"],
        baseline=baseline, challenger={a: list(v) for a, v in baseline.items()}, shortTermError=None,
        weather=None, weatherError=None, sourceHash=base["sourceHash"], inputHash=base["inputHash"],
        githubRunId=os.environ.get("GITHUB_RUN_ID"))
    models, selection = wind_short_term.load_models()
    try:
        wind_short_term.require_observed_wind(arrays["observed512"])
        selected_at = pd.Timestamp(base["recentWind"]["DK1"]["selectedAt"])
        recent = wind_short_term.measurements(sources["electricity"], selected_at)
        names = json.loads((data.ROOT / "research/forecasting/weather_retrain/model-manifest.json").read_text())["featureNames"]["future"]
        x, columns = wind_short_term.features(arrays["history"], arrays["ecmwf_future"], names, recent, pd.DatetimeIndex(base["times"]))
        if columns != selection["featureNames"]:
            raise ValueError("Short-term feature schema differs")
        for ai, area in enumerate(data.AREAS):
            values = wind_short_term.predict(models[area], x[ai], recent[area]["levelMW"])
            if not np.isfinite(values).all() or (values > (15000 if area == "DK1" else 10000)).any():
                raise ValueError("Short-term challenger failed plausibility checks")
            record["challenger"][area][:6] = values.tolist()
    except ValueError as error:
        record["shortTermError"] = str(error)
        record["challenger"] = {a: list(v) for a, v in baseline.items()}
    try:
        snapshot = weather_ensemble.snapshot(state, pd.DatetimeIndex(base["times"]), data.now())
        record["weather"] = weather_ensemble.values(snapshot["data"], pd.DatetimeIndex(base["times"]))
        record["weatherProvenance"] = {k: v for k, v in snapshot.items() if k != "data"}
    except (ValueError, KeyError, OSError) as error:
        record["weatherError"] = str(error)[:250]
    record["issuedAt"] = data.now().isoformat()
    if pd.Timestamp(record["issuedAt"]) >= pd.Timestamp(record["times"][0]):
        return "missed-first-target"
    record["ranges"] = calibrate(record, previous, state, protocol)
    # Hash once and preserve the exact forecasts/bounds forever after issuance.
    record["sha256"] = canonical_hash(record)
    previous.append(record)
    work.mkdir(parents=True, exist_ok=True)
    write_json(work / "record.json", record)
    if state.get("weatherEnsemble"):
        write_json(work / "ensemble.json", state["weatherEnsemble"])
    return "issued"


def evaluate(state, protocol, at):
    records = [r for r in state.get("windResearch", {}).get("records", []) if r["version"] == protocol["version"]]
    baseline_version = records[-1]["baselineVersion"] if records else None
    records = [r for r in records if r["baselineVersion"] == baseline_version]
    short = completed(records, state, at, horizon=6)
    full = completed(records, state, at)
    days = len({r["origin"][:10] for r, _ in short})
    result = dict(checkedAt=at.isoformat(), records=len(records), completedOrigins=len(short), distinctDays=days,
                  minimumDays=protocol["minimumProspectiveDays"], baselineVersion=baseline_version,
                  publicPromotion=False, shortTerm={}, ranges={})
    if short:
        for ai, area in enumerate(data.AREAS):
            old = np.asarray([np.abs(y[ai] - np.array(r["baseline"][area][:6])) for r, y in short])
            new = np.asarray([np.abs(y[ai] - np.array(r["challenger"][area][:6])) for r, y in short])
            result["shortTerm"][area] = dict(baselineMAE=float(old.mean()), challengerMAE=float(new.mean()),
                accuracyPass=bool(new.mean() <= old.mean()*.95), leadPass=bool((new.mean(0) <= old.mean(0)*1.05).all()),
                enoughDays=days >= protocol["minimumProspectiveDays"], fallbackOrigins=sum(bool(r["shortTermError"]) for r, _ in short))
    for method in METHODS:
        matched = [(r, y) for r, y in full if method in r["ranges"] and "absolute" in r["ranges"]]
        if not matched:
            continue
        result["ranges"][method] = {}
        for ai, area in enumerate(data.AREAS):
            truth = np.array([y[ai] for _, y in matched])
            lo, hi = (np.array([r["ranges"][method]["areas"][area][key] for r, _ in matched]) for key in ("lower", "upper"))
            ref_lo, ref_hi = (np.array([r["ranges"]["absolute"]["areas"][area][key] for r, _ in matched]) for key in ("lower", "upper"))
            covered = (lo <= truth) & (truth <= hi)
            score = float(interval_score(truth, lo, hi).mean())
            bands = [float(covered[:, start:end].mean()) for start, end in wind_ranges.BANDS]
            result["ranges"][method][area] = dict(origins=len(matched), coverage=float(covered.mean()), coverageByLeadBand=bands,
                intervalScore=score, meanWidthMW=float((hi-lo).mean()),
                coveragePass=all(protocol["minimumCoverage"] <= v <= protocol["maximumCoverage"] for v in bands),
                scorePass=bool(score <= interval_score(truth, ref_lo, ref_hi).mean()*.95),
                widthPass=bool((hi-lo).mean() <= (ref_hi-ref_lo).mean()*protocol["maximumWidthMultiplier"]),
                enoughDays=len({r["origin"][:10] for r, _ in matched}) >= protocol["minimumProspectiveDays"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--state", type=Path, default=data.STATE_PATH)
    args = parser.parse_args()
    state = data.read_state(args.state)
    protocol, fingerprint = configuration()
    work = CACHE / ("probe" if args.probe else "evidence")
    error = None
    try:
        status = issue(state, work, protocol, fingerprint)
    except Exception as exc:
        status, error = "unavailable", str(exc)[:300]
    at = data.now()
    ledger = state.setdefault("windResearch", {})
    cutoff = at - pd.Timedelta(days=35)
    ledger["records"] = [r for r in ledger.get("records", []) if pd.Timestamp(r["issuedAt"]) >= cutoff]
    ledger["status"] = dict(at=at.isoformat(), status=status, error=error)
    ledger["evaluation"] = evaluate(state, protocol, at)
    if not args.probe:
        state["updatedAt"] = at.isoformat()
        data.write_state(state)
        if status == "issued" and os.environ.get("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as output:
                output.write("issued=true\n")
    write_json(work / "status.json", dict(**ledger["status"], evaluation=ledger["evaluation"], diagnosticOnly=args.probe))
    print(json.dumps(ledger["status"], indent=2), flush=True)
    if error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
