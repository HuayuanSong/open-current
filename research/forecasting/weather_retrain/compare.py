"""Compare refitted components within the unchanged ensemble, without publishing."""

import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "4"

import argparse
import json

import numpy as np
import pandas as pd

from forecast import data, weather_runs
from forecast.artifacts import CONFIG as BASELINE, ensure
from forecast.features import AREAS, TARGETS
from forecast.score import combine, run as baseline_score
from research.forecasting.weather_refresh.evaluate import weather, summary
from .common import CACHE, HERE, PROTOCOL, boundary, digest, write_json
from .models import predict


def verify_models():
    manifest = json.loads((CACHE / "models/manifest.json").read_text())
    if manifest["protocolHash"] != digest(HERE / "protocol.json") or BASELINE["version"] != PROTOCOL["baselineVersion"]:
        raise ValueError("Frozen protocol or baseline changed")
    if manifest["featureSourceHash"] != digest(HERE.parents[2] / "forecast/features.py"):
        raise ValueError("Frozen feature definitions changed")
    for name, expected in manifest["sourceHashes"].items():
        if digest(HERE / name) != expected:
            raise ValueError(f"Fitting/inference source changed: {name}")
    for name, expected in manifest["models"].items():
        if digest(CACHE / "models" / name) != expected:
            raise ValueError(f"Refitted model changed: {name}")
    return manifest


def score_pair(baseline, latest, work):
    manifest = verify_models()
    names = manifest["featureNames"]
    work.mkdir(parents=True, exist_ok=True)
    source = work / "baseline-inputs.npz"
    np.savez_compressed(source, **baseline)
    unchanged, members, timings = baseline_score(source, work / "baseline")
    selection = json.loads((ensure() / "selection.json").read_text())
    results = {"baseline": unchanged}
    for policy, arrays in [("daily", baseline), ("latest", latest)]:
        refitted = predict(arrays, names, CACHE / "models" / policy)
        results[policy] = combine({**members, **refitted}, baseline["area_ids"], selection)[0]
    return results, timings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, type=str)
    args = parser.parse_args()
    state = data.read_state(HERE.parents[2] / args.state)
    verify_models()
    cases, excluded = [], []
    actual = {(r["area"], r["time"]): r for r in state["hourly"]}
    for day in pd.date_range(boundary("diagnosticStart"), boundary("diagnosticEnd"), freq="D"):
        for hour in PROTOCOL["diagnosticHours"]:
            origin = day + pd.Timedelta(hours=hour)
            at = origin + pd.Timedelta(minutes=PROTOCOL["issueMinutesAfterOrigin"])
            run = (at - pd.Timedelta(hours=PROTOCOL["assumedAvailabilityHours"])).floor("6h")
            try:
                base, meta = data.inputs(state, at)
                assert pd.Timestamp(meta["origin"]) == origin
                points = weather(run)
                available = run + pd.Timedelta(hours=PROTOCOL["assumedAvailabilityHours"]) - weather_runs.CONSISTENCY_BUFFER
                entry = {**points, "model": weather_runs.MODEL, "availableAt": available.isoformat(), "retrievedAt": at.isoformat()}
                latest, new_meta = data.inputs({**state, "weatherRuns": {run.isoformat(): entry}}, at, latest_weather=True)
                if pd.Timestamp(new_meta["weatherRun"]) != run:
                    raise ValueError("Missing latest forecast horizon")
                for key in base:
                    if key != "future":
                        np.testing.assert_array_equal(base[key], latest[key])
                truth = np.asarray([[[actual[(area, t)][target] for target in TARGETS] for t in meta["times"]] for area in AREAS])
                if not np.isfinite(truth).all():
                    raise ValueError("Incomplete paired outcomes")
                cases.append(dict(origin=origin.isoformat(), baseline=base, latest=latest, truth=truth))
            except (ValueError, KeyError) as error:
                excluded.append(dict(origin=origin.isoformat(), error=str(error)))
    if not cases:
        raise ValueError("No paired diagnostic origins")
    rows = []
    for start in range(0, len(cases), 8):
        batch = cases[start:start + 8]
        inputs = {policy: {key: np.concatenate([c[policy][key] for c in batch]) for key in batch[0][policy]}
                  for policy in ("baseline", "latest")}
        work = CACHE / f"diagnostic/batch-{start:03}"
        outputs, timings = score_pair(inputs["baseline"], inputs["latest"], work)
        np.savez_compressed(work / "predictions.npz", **outputs)
        write_json(work / "runtime.json", timings)
        for i, case in enumerate(batch):
            for variant, predictions in outputs.items():
                for ai, area in enumerate(AREAS):
                    for ti, target in enumerate(TARGETS):
                        for lead in range(48):
                            rows.append(dict(origin=case["origin"], variant=variant, area=area, target=target, lead=lead + 1,
                                prediction=float(predictions[i * 2 + ai, lead, ti]), actual=float(case["truth"][ai, lead, ti])))
        print(f"Scored diagnostic origins {start + len(batch)}/{len(cases)}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(CACHE / "diagnostic/predictions.csv", index=False)
    metrics = frame.groupby(["area", "target", "variant"]).apply(summary, include_groups=False).reset_index()
    metrics.to_csv(HERE / "diagnostic-metrics.csv", index=False)
    frame["band"] = pd.cut(frame.lead, [0, 6, 24, 48], labels=["1-6", "7-24", "25-48"])
    bands = frame.groupby(["area", "target", "variant", "band"], observed=True).apply(summary, include_groups=False).reset_index()
    bands.to_csv(HERE / "diagnostic-lead-metrics.csv", index=False)
    write_json(HERE / "diagnostic-audit.json", dict(completeOrigins=len(cases), excluded=excluded,
        stateHash=digest(HERE.parents[2] / args.state), modelManifestHash=digest(CACHE / "models/manifest.json"),
        interpretation="Inspected historical development evidence; not the prospective promotion test."))
    print(metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
