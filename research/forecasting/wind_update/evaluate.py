"""Frozen wind-only development replay; no production or old-trial writes."""

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from forecast import data, intervals, weather_runs
from forecast.features import AREAS
from forecast.wind_nowcast import adjust, recent_wind
from research.forecasting.weather_retrain import compare, prepare
from research.forecasting.weather_retrain.common import digest, write_json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CACHE = ROOT / ".cache/wind-update"
PROTOCOL = json.loads((HERE / "protocol.json").read_text())


def boundary(key):
    return pd.Timestamp(PROTOCOL[key], tz="UTC")


def prepare_cases(state, raw):
    """Reuse inspected November development predictions; score later dates once."""
    compare.verify_models()
    saved = pd.read_csv(ROOT / ".cache/weather-retrain/diagnostic/predictions.csv")
    actual = {(r["area"], r["time"]): r for r in state["hourly"]}
    cases, excluded = [], []
    for day in pd.date_range(boundary("developmentStart"), boundary("evaluationEndExclusive") - pd.Timedelta(days=1)):
        for hour in PROTOCOL["originHoursUTC"]:
            origin = day + pd.Timedelta(hours=hour)
            if boundary("developmentEndExclusive") <= origin < boundary("evaluationStart"):
                continue
            at = origin + pd.Timedelta(minutes=PROTOCOL["issueMinutesAfterOrigin"])
            times = pd.date_range(origin + pd.Timedelta(hours=1), periods=48, freq="h")
            try:
                recent = recent_wind(raw, at, PROTOCOL["observationBufferMinutes"])
                truth = np.asarray([[[actual[(area, t.isoformat())][target] for target in data.TARGETS]
                                     for t in times] for area in AREAS], dtype=float)
                if not np.isfinite(truth).all():
                    raise ValueError("Incomplete observed outcome")
                case = dict(origin=origin.isoformat(), times=[t.isoformat() for t in times], recent=recent, truth=truth)
                rows = saved[saved.origin == origin.isoformat()]
                if origin < boundary("developmentEndExclusive"):
                    outputs = {}
                    for variant in ("baseline", "latest"):
                        outputs[variant] = np.asarray([
                            rows[(rows.variant == variant) & (rows.area == area)].pivot(index="lead", columns="target", values="prediction")
                            .reindex(index=range(1, 49), columns=data.TARGETS).to_numpy() for area in AREAS])
                    if not all(np.isfinite(v).all() for v in outputs.values()):
                        raise ValueError("Missing cached development predictions")
                    case.update(outputs)
                else:
                    baseline, meta = data.inputs(state, at)
                    if pd.Timestamp(meta["origin"]) != origin:
                        raise ValueError("Missing origin history")
                    run = (at - pd.Timedelta(hours=PROTOCOL["assumedWeatherAvailabilityHours"])).floor("6h")
                    points, _ = prepare.weather(run, times, state=state, state_path=CACHE / "state.json.gz")
                    available = run + pd.Timedelta(hours=PROTOCOL["assumedWeatherAvailabilityHours"]) - weather_runs.CONSISTENCY_BUFFER
                    entry = dict(model=weather_runs.MODEL, run=run.isoformat(), data=points,
                                 availableAt=available.isoformat(), retrievedAt=at.isoformat())
                    fresh, new_meta = data.inputs({**state, "weatherRuns": {run.isoformat(): entry}}, at, latest_weather=True)
                    if pd.Timestamp(new_meta["weatherRun"]) != run:
                        raise ValueError("Required fresh run unavailable")
                    for key in baseline:
                        if key != "future":
                            np.testing.assert_array_equal(baseline[key], fresh[key])
                    case.update(baselineInputs=baseline, freshInputs=fresh)
                cases.append(case)
                print("Prepared", case["origin"], flush=True)
            except (ValueError, KeyError) as error:
                excluded.append(dict(origin=origin.isoformat(), error=str(error)))
    pending = [c for c in cases if "baseline" not in c]
    for start in range(0, len(pending), 4):
        batch = pending[start:start + 4]
        arrays = {key: {name: np.concatenate([c[key][name] for c in batch]) for name in batch[0][key]}
                  for key in ("baselineInputs", "freshInputs")}
        work = CACHE / f"replay/batch-{start:03}"
        work.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(work / "inputs.npz", **arrays["baselineInputs"], latest_future=arrays["freshInputs"]["future"])
        fingerprint = dict(input=digest(work / "inputs.npz"), protocol=digest(HERE / "protocol.json"),
                           manifest=digest(compare.CACHE / "models/manifest.json"))
        if ((work / "fingerprint.json").exists() and (work / "predictions.npz").exists()
                and json.loads((work / "fingerprint.json").read_text()) == fingerprint):
            outputs = dict(np.load(work / "predictions.npz"))
        else:
            outputs, timings = compare.score_pair(arrays["baselineInputs"], arrays["freshInputs"], work)
            np.savez_compressed(work / "predictions.npz", **outputs)
            write_json(work / "fingerprint.json", fingerprint)
            write_json(work / "runtime.json", timings)
        for i, case in enumerate(batch):
            case.update({name: values[i * 2:i * 2 + 2] for name, values in outputs.items()})
            case.pop("baselineInputs"); case.pop("freshInputs")
        print(f"Scored {start + len(batch)}/{len(pending)} later origins", flush=True)
    return cases, excluded


def wind_only(case):
    output = case["baseline"].copy()
    output[:, :, 0] = case["latest"][:, :, 0]
    return output


def choose_rules(cases):
    dev = [c for c in cases if pd.Timestamp(c["origin"]) + pd.Timedelta(hours=6) < boundary("developmentEndExclusive")]
    if len(dev) < 24:
        raise ValueError("Insufficient chronological development examples")
    rules = {}
    for ai, area in enumerate(AREAS):
        trials = []
        for strength, half_life, trend in itertools.product(PROTOCOL["strengths"], PROTOCOL["halfLivesHours"], PROTOCOL["trendWeights"]):
            rule = dict(strength=strength, halfLifeHours=half_life, trendWeight=trend)
            errors = [np.abs(adjust(wind_only(c), c["times"], c["recent"], {a: rule for a in AREAS})[ai, :6, 0] - c["truth"][ai, :6, 0]) for c in dev]
            trials.append((float(np.mean(errors)), strength, half_life, trend, rule))
        best = min(trials, key=lambda x: x[:4])
        rules[area] = best[-1]
    return rules


def main():
    state = data.read_state(CACHE / "state.json.gz")
    raw = prepare.read(CACHE / "electricity.json.gz")["data"]["records"]
    cases, excluded = prepare_cases(state, raw)
    rules = choose_rules(cases)
    for case in cases:
        case["fresh_wind"] = wind_only(case)
        case["corrected"] = adjust(case["fresh_wind"], case["times"], case["recent"], rules)
    calibration_cases = [c for c in cases if pd.Timestamp(c["origin"]) < boundary("developmentEndExclusive")
                         and pd.Timestamp(c["times"][-1]) < boundary("evaluationStart")]
    range_rules = {}
    for ai, area in enumerate(AREAS):
        y = np.array([c["truth"][ai, :, 0] for c in calibration_cases])
        p = np.array([c["corrected"][ai, :, 0] for c in calibration_cases])
        range_rules[area] = intervals.fit(y, p, np.ones_like(p, bool), "wind", "absolute", 1)
    config = dict(version=PROTOCOL["version"], protocolHash=digest(HERE / "protocol.json"),
                  rules=rules, ranges=range_rules, calibrationOrigins=len(calibration_cases),
                  calibratedBefore=PROTOCOL["evaluationStart"], automaticPromotion=False)
    write_json(HERE / "selection.json", config)
    rows = []
    for case in cases:
        if pd.Timestamp(case["origin"]) < boundary("evaluationStart"):
            continue
        for ai, area in enumerate(AREAS):
            lo, hi = intervals.bounds(case["corrected"][ai, :, 0], np.ones(48, bool), "wind", range_rules[area])
            for variant in ("baseline", "fresh_wind", "corrected"):
                for i in range(48):
                    value = case[variant][ai, i, 0]
                    rows.append(dict(origin=case["origin"], area=area, lead=i + 1, variant=variant,
                                     prediction=float(value), actual=float(case["truth"][ai, i, 0]),
                                     lower=float(lo[i]) if variant == "corrected" else None,
                                     upper=float(hi[i]) if variant == "corrected" else None))
    frame = pd.DataFrame(rows)
    frame["absoluteError"] = np.abs(frame.prediction - frame.actual)
    frame["band"] = pd.cut(frame.lead, [0, 6, 24, 48], labels=["1-6", "7-24", "25-48"])
    metrics = frame.groupby(["area", "variant"]).absoluteError.mean().unstack("variant")
    bands = frame.groupby(["area", "variant", "band"], observed=True).absoluteError.mean().unstack("variant")
    range_rows = frame[frame.variant == "corrected"]
    coverage = range_rows.assign(covered=(range_rows.actual >= range_rows.lower) & (range_rows.actual <= range_rows.upper)).groupby("area").covered.mean()
    report = dict(version=PROTOCOL["version"], protocolHash=digest(HERE / "protocol.json"),
                  stateHash=digest(CACHE / "state.json.gz"), electricityHash=digest(CACHE / "electricity.json.gz"),
                  selectedRules=rules, developmentOrigins=sum(pd.Timestamp(c["origin"]) < boundary("developmentEndExclusive") for c in cases),
                  evaluationOrigins=int(frame.origin.nunique()), excluded=excluded,
                  metrics=metrics.reset_index().to_dict("records"), bands=bands.reset_index().to_dict("records"),
                  rangeCoverage=coverage.to_dict(), prospectivePromotionEligible=False,
                  interpretation=PROTOCOL["interpretation"])
    frame.to_csv(CACHE / "predictions.csv", index=False)
    write_json(HERE / "results.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
