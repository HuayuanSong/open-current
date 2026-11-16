"""Compare DMI and ECMWF using frozen power models and chronological selection."""

import gzip
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from forecast import data, weather_runs, weather_sources
from forecast.features import AREAS
from forecast.score import combine
from forecast.wind_models import predict_wind
from forecast.wind_nowcast import adjust, recent_wind
from research.forecasting.weather_retrain import compare, prepare
from research.forecasting.weather_retrain.common import digest, write_json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CACHE = ROOT / ".cache/fresh-wind"
PROTOCOL = json.loads((HERE / "protocol.json").read_text())
STATE = ROOT / ".cache/wind-update/state.json.gz"
RAW = ROOT / ".cache/wind-update/electricity.json.gz"


def archived(source, run, times, state):
    if source == "ecmwf":
        return prepare.weather(run, times, state=state, state_path=STATE)
    params = weather_sources.parameters(source, run)
    path = CACHE / "weather" / f"{source}-{run:%Y%m%dT%H}.json.gz"
    if path.exists():
        saved = prepare.read(path)
        if saved["parameters"] != params:
            raise ValueError("Archived request changed")
        weather_runs.validate_points(saved["data"], times)
        return saved["data"], path
    begin = time.monotonic()
    try:
        points = data.request(weather_runs.FORECAST_URL, params)
        weather_runs.validate_points(points, times)
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt") as output:
            json.dump(dict(parameters=params, retrievedAt=data.now().isoformat(), data=points), output)
    finally:
        time.sleep(max(0, prepare.REQUEST_INTERVAL_SECONDS - (time.monotonic() - begin)))
    return points, path


def cached_members():
    """Reuse expensive inference only where every baseline input matches exactly."""
    batches = []
    for folder in (ROOT / ".cache/weather-retrain/diagnostic", ROOT / ".cache/wind-update/replay"):
        for path in sorted(folder.glob("batch-*/baseline-inputs.npz")):
            with np.load(path) as saved:
                arrays = {k: saved[k] for k in saved.files}
            members = {}
            for family in ("trees", "neural", "ttm"):
                with np.load(path.parent / "baseline" / f"{family}.npz") as saved:
                    members.update({k: saved[k] for k in saved.files})
            batches.append((arrays, members))
    return batches


def matching_members(arrays, batches):
    for saved, members in batches:
        for start in range(0, len(saved["area_ids"]), 2):
            if all(np.array_equal(arrays[k], saved[k][start:start + 2], equal_nan=True) for k in arrays):
                return {k: v[start:start + 2].copy() for k, v in members.items()}
    raise ValueError("No matching cached baseline inference")


def main():
    manifest = compare.verify_models()
    blend = json.loads((compare.ensure() / "selection.json").read_text())
    correction = json.loads((ROOT / PROTOCOL["correctionSelection"]).read_text())["rules"]
    state = data.read_state(STATE)
    raw = prepare.read(RAW)["data"]["records"]
    actual = {(r["area"], r["time"]): r for r in state["hourly"]}
    batches = cached_members()
    cases, excluded, sources = [], [], {}
    dates = pd.date_range(PROTOCOL["developmentStart"], PROTOCOL["evaluationEndExclusive"], inclusive="left", tz="UTC")
    for day in dates:
        for hour in PROTOCOL["originHoursUTC"]:
            origin = day + pd.Timedelta(hours=hour)
            if PROTOCOL["developmentEndExclusive"] <= str(origin.date()) < PROTOCOL["evaluationStart"]:
                continue
            at = origin + pd.Timedelta(minutes=PROTOCOL["issueMinutesAfterOrigin"])
            try:
                base, metadata = data.inputs(state, at)
                if pd.Timestamp(metadata["origin"]) != origin:
                    raise ValueError("Missing source hour")
                times = pd.DatetimeIndex(metadata["times"])
                truth = np.array([[[actual[(a, t)][target] for target in data.TARGETS]
                                   for t in metadata["times"]] for a in AREAS], dtype=float)
                if not np.isfinite(truth).all():
                    raise ValueError("Incomplete outcomes")
                recent = recent_wind(raw, at)
                work = CACHE / "cases" / f"{origin:%Y%m%dT%H}"
                work.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(work / "baseline.npz", **base)
                signature = dict(input=digest(work / "baseline.npz"), protocol=digest(HERE / "protocol.json"),
                                 inference=digest(ROOT / "forecast/wind_models.py"), manifest=digest(compare.CACHE / "models/manifest.json"))
                if ((work / "signature.json").exists() and (work / "predictions.npz").exists()
                        and json.loads((work / "signature.json").read_text()) == signature):
                    outputs = dict(np.load(work / "predictions.npz"))
                    sources.update(json.loads((work / "sources.json").read_text()))
                else:
                    members = matching_members(base, batches)
                    outputs = dict(baseline=combine(members, base["area_ids"], blend)[0])
                    case_sources = {}
                    for source in ("ecmwf", "dmi"):
                        run = (at - pd.Timedelta(hours=PROTOCOL["assumedAvailabilityHours"][source])).floor("6h" if source == "ecmwf" else "3h")
                        points, path = archived(source, run, times, state)
                        case_sources[str(path.relative_to(ROOT))] = digest(path)
                        inputs = weather_sources.future_inputs(base, points, times, metadata["featureNames"])
                        outputs[source] = predict_wind(inputs, members, manifest["featureNames"], compare.CACHE / "models/latest", blend)
                        np.testing.assert_array_equal(outputs[source][:, :, 1:], outputs["baseline"][:, :, 1:])
                    np.savez_compressed(work / "predictions.npz", **outputs)
                    write_json(work / "signature.json", signature)
                    write_json(work / "sources.json", case_sources)
                    sources.update(case_sources)
                cases.append(dict(origin=origin, times=times, recent=recent, truth=truth, **outputs))
                print(f"Scored {origin.isoformat()} ({len(cases)} origins)", flush=True)
            except (ValueError, KeyError) as error:
                excluded.append(dict(origin=origin.isoformat(), error=str(error)))
                print("Excluded", origin, error, flush=True)
    development = [c for c in cases if str(c["origin"].date()) < PROTOCOL["developmentEndExclusive"]
                   and str(c["times"][-1].date()) < PROTOCOL["evaluationStart"]]
    evaluation = [c for c in cases if str(c["origin"].date()) >= PROTOCOL["evaluationStart"]]
    if len(development) < 24 or len(evaluation) < PROTOCOL["minimumEvaluationOrigins"]:
        raise ValueError("Insufficient chronological cases")
    weights, development_scores = {}, {}
    for ai, area in enumerate(AREAS):
        scores = []
        for weight in PROTOCOL["dmiWeights"]:
            error = []
            for c in development:
                candidate = c["ecmwf"].copy()
                candidate[:, :, 0] = (1 - weight) * c["ecmwf"][:, :, 0] + weight * c["dmi"][:, :, 0]
                corrected = adjust(candidate, c["times"], c["recent"], correction)
                error.append(np.abs(corrected[ai, :, 0] - c["truth"][ai, :, 0]))
            scores.append((float(np.mean(error)), weight))
        weights[area] = min(scores)[1]
        development_scores[area] = [dict(dmiWeight=w, maeMW=m) for m, w in scores]
    rows = []
    for c in evaluation:
        chosen = c["ecmwf"].copy()
        for ai, area in enumerate(AREAS):
            w = weights[area]
            chosen[ai, :, 0] = (1 - w) * c["ecmwf"][ai, :, 0] + w * c["dmi"][ai, :, 0]
        candidates = {name: adjust(c[name], c["times"], c["recent"], correction) for name in ("ecmwf", "dmi")}
        candidates.update(baseline=c["baseline"], selected=adjust(chosen, c["times"], c["recent"], correction))
        for name, prediction in candidates.items():
            for ai, area in enumerate(AREAS):
                for lead in range(48):
                    rows.append(dict(origin=c["origin"].isoformat(), area=area, lead=lead + 1, variant=name,
                                     prediction=float(prediction[ai, lead, 0]), actual=float(c["truth"][ai, lead, 0])))
    frame = pd.DataFrame(rows)
    frame["absoluteError"] = np.abs(frame.prediction - frame.actual)
    frame["band"] = pd.cut(frame.lead, [0, 6, 24, 48], labels=["1-6", "7-24", "25-48"])
    metrics = frame.groupby(["area", "variant"]).absoluteError.mean().unstack("variant")
    bands = frame.groupby(["area", "variant", "band"], observed=True).absoluteError.mean().unstack("variant")
    checks = {}
    for area in AREAS:
        checks[area] = dict(improvementPct=100 * (1 - metrics.loc[area, "selected"] / metrics.loc[area, "baseline"]),
            pointErrorPass=bool(metrics.loc[area, "selected"] <= metrics.loc[area, "baseline"] * .95),
            leadBandsPass=bool((bands.loc[area, "selected"] <= bands.loc[area, "baseline"] * 1.05).all()))
    result = dict(version=PROTOCOL["version"], protocolHash=digest(HERE / "protocol.json"),
                  developmentOrigins=len(development), evaluationOrigins=len(evaluation),
                  dmiWeights=weights, developmentScores=development_scores, checks=checks,
                  metrics=metrics.reset_index().to_dict("records"), bands=bands.reset_index().to_dict("records"),
                  excluded=excluded, sourceHashes=sources, stateHash=digest(STATE), rawHash=digest(RAW),
                  interpretation=PROTOCOL["interpretation"])
    frame.to_csv(CACHE / "predictions.csv", index=False)
    write_json(HERE / "results.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "sourceHashes"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
