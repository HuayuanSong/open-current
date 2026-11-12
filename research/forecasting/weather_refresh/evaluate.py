"""Paired, fixed-weight comparison; never writes to the production archive."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from forecast import data, weather_runs
from forecast.features import AREAS, TARGETS
from forecast.score import run as score
from forecast.artifacts import CONFIG as MODEL_RELEASE

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CACHE = ROOT / ".cache/weather-refresh"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def weather(run):
    params = weather_runs.parameters(run)
    request_hash = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]
    path = CACHE / "weather" / f"{run:%Y%m%dT%H}-{request_hash}.json.gz"
    if path.exists():
        with gzip.open(path, "rt") as source:
            record = json.load(source)
        if record["run"] != run.isoformat() or record["requestParameters"] != params:
            raise ValueError("Cached weather request differs from replay")
        return record
    begin = time.monotonic()
    try:
        points = data.request(weather_runs.FORECAST_URL, params)
        weather_runs.validate_points(points)
        record = {"run": run.isoformat(), "retrievedAt": data.now().isoformat(),
                  "requestParameters": params, "data": points}
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with gzip.open(temporary, "wt") as target:
            json.dump(record, target, allow_nan=False)
        temporary.replace(path)
        return record
    finally:
        time.sleep(max(0, 1.5 - (time.monotonic() - begin)))


def summary(frame):
    error = frame.prediction - frame.actual
    return pd.Series({
        "origins": frame.origin.nunique(), "hours": len(frame),
        "maeMW": error.abs().mean(), "biasMW": error.mean(),
        "wapePct": 100 * error.abs().sum() / frame.actual.sum(),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((HERE / "protocol.json").read_text())
    if MODEL_RELEASE["version"] != protocol["modelVersion"]:
        raise ValueError("Replay protocol requires the original pinned model release")
    with gzip.open(args.state, "rt") as source:
        state = json.load(source)
    CACHE.mkdir(exist_ok=True)
    actual = {(r["area"], r["time"]): r for r in state["hourly"]}
    cases, excluded = [], []
    for day in pd.date_range(protocol["startDate"], protocol["endDate"], tz="UTC"):
        for hour in protocol["originHoursUTC"]:
            origin = day + pd.Timedelta(hours=hour)
            at = origin + pd.Timedelta(minutes=protocol["issueMinutesAfterOrigin"])
            run = (at - pd.Timedelta(hours=protocol["assumedWeatherAvailabilityHours"])).floor("6h")
            try:
                baseline, old_meta = data.inputs(state, at)
                if pd.Timestamp(old_meta["origin"]) != origin:
                    raise ValueError("Missing origin observations")
                truth = np.array([
                    [[actual[(area, stamp)][target] for target in TARGETS]
                     for stamp in old_meta["times"]] for area in AREAS
                ], dtype=float)
                if not np.isfinite(truth).all():
                    raise ValueError("Incomplete 48-hour outcome")
                downloaded = weather(run)
                # Synthetic availability is explicit and confined to this replay.
                available = (run + pd.Timedelta(hours=protocol["assumedWeatherAvailabilityHours"])
                             - weather_runs.CONSISTENCY_BUFFER)
                entry = {**downloaded, "model": weather_runs.MODEL,
                         "availableAt": available.isoformat(),
                         "retrievedAt": at.isoformat()}
                changed = {**state, "weatherRuns": {run.isoformat(): entry}}
                candidate, new_meta = data.inputs(changed, at, latest_weather=True)
                if pd.Timestamp(new_meta["weatherRun"]) != run:
                    raise ValueError("Candidate run lacks the full required horizon")
                for key in baseline:
                    if key != "future":
                        np.testing.assert_array_equal(baseline[key], candidate[key], err_msg=key)
                cases.append(dict(origin=origin.isoformat(), truth=truth, baseline=baseline,
                                  candidate=candidate, oldRun=old_meta["weatherRun"], newRun=new_meta["weatherRun"]))
                print(f"Prepared {len(cases)}/48: {origin.isoformat()} ({old_meta['weatherRun']} -> {new_meta['weatherRun']})", flush=True)
            except (ValueError, KeyError) as error:
                excluded.append({"origin": origin.isoformat(), "error": str(error)})
    if not cases:
        raise RuntimeError("No complete paired origins")
    rows = []
    for start in range(0, len(cases), protocol["batchOrigins"]):
        batch = cases[start:start + protocol["batchOrigins"]]
        work = CACHE / f"batch-{start:03}"
        work.mkdir(exist_ok=True)
        arrays = {key: np.concatenate([case[variant][key] for case in batch
                                      for variant in ("baseline", "candidate")])
                  for key in batch[0]["baseline"]}
        input_path = work / "inputs.npz"
        np.savez_compressed(input_path, **arrays)
        fingerprint = {
            "input": digest(input_path),
            "protocol": digest(HERE / "protocol.json"),
            "modelRelease": digest(ROOT / "forecast/model-release.json"),
            "scoringSources": {name: digest(ROOT / "forecast" / name) for name in (
                "score.py", "worker.py", "features.py", "weather_mlp.py",
                "requirements-main.txt", "requirements-ttm.txt",
            )},
        }
        saved = work / "predictions.npz"
        manifest = work / "fingerprint.json"
        if saved.exists() and manifest.exists() and json.loads(manifest.read_text()) == fingerprint:
            prediction = np.load(saved)["prediction"]
        else:
            prediction, _, timings = score(input_path, work)
            np.savez_compressed(saved, prediction=prediction)
            manifest.write_text(json.dumps(fingerprint, indent=2))
            (work / "timings.json").write_text(json.dumps(timings, indent=2))
        if (prediction.shape != (len(batch) * len(AREAS) * 2, protocol["horizonHours"], len(TARGETS))
                or not np.isfinite(prediction).all()):
            raise ValueError("Incomplete paired predictions; replay cannot pass")
        for i, case in enumerate(batch):
            for vi, variant in enumerate(("baseline", "candidate")):
                for ai, area in enumerate(AREAS):
                    for lead in range(48):
                        for ti, target in enumerate(TARGETS):
                            rows.append(dict(origin=case["origin"], variant=variant,
                                             area=area, target=target, lead=lead + 1,
                                             prediction=float(prediction[i * 4 + vi * 2 + ai, lead, ti]),
                                             actual=float(case["truth"][ai, lead, ti])))
        print(f"Scored {start + len(batch)}/{len(cases)} origins", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(CACHE / "predictions.csv", index=False)
    scores = frame.groupby(["area", "target", "variant"]).apply(summary, include_groups=False).reset_index()
    frame["band"] = pd.cut(frame.lead, [0, 6, 24, 48], labels=["1-6", "7-24", "25-48"])
    bands = frame.groupby(["area", "target", "band", "variant"], observed=True).apply(summary, include_groups=False).reset_index()
    frame["day"] = frame.origin.str[:10]
    daily = frame.groupby(["day", "area", "target", "variant"]).apply(summary, include_groups=False).reset_index()
    scores.to_csv(HERE / "metrics.csv", index=False)
    bands.to_csv(HERE / "lead-metrics.csv", index=False)
    daily.to_csv(HERE / "daily-metrics.csv", index=False)
    paired = scores.pivot(index=["area", "target"], columns="variant", values="maeMW")
    reduction = 100 * (1 - paired.candidate / paired.baseline)
    band_pairs = bands.pivot(index=["area", "target", "band"], columns="variant", values="maeMW")
    band_increase = 100 * (band_pairs.candidate / band_pairs.baseline - 1)
    gate = protocol["promotionGate"]
    passed = (len(cases) >= gate["minimumCompleteOrigins"]
              and reduction.min() >= -gate["maximumSeriesMAEIncreasePct"]
              and band_increase.max() <= gate["maximumLeadBandMAEIncreasePct"]
              and reduction.loc[("DK1", "wind")] >= gate["minimumDK1WindMAEReductionPct"])
    result = dict(protocolHash=digest(HERE / "protocol.json"), stateHash=digest(args.state),
                  modelRelease=MODEL_RELEASE, scoringSources=fingerprint["scoringSources"],
                  stateUpdatedAt=state["updatedAt"], completeOrigins=len(cases), excluded=excluded,
                  promotionGatePassed=bool(passed),
                  seriesReductionPct={f"{area}/{target}": float(v) for (area, target), v in reduction.items()},
                  worstLeadBandIncreasePct=float(band_increase.max()),
                  assumptions=["Eight-hour weather availability lag is assumed, not historically verified.",
                               "Revised observations are shared by both alternatives.",
                               "Overlapping horizons are dependent; this is a short retrospective diagnostic."])
    (HERE / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
