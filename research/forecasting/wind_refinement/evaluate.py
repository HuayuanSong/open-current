"""Evaluate frozen ramp models and chronological conditional range candidates."""

import json

import numpy as np
import pandas as pd

from forecast import data, weather_sources, wind_ranges
from forecast.fresh_wind import weighted_wind
from forecast.intervals import interval_score
from forecast.wind_nowcast import adjust, recent_wind
from forecast.wind_short_term import measurements, features, predict, load_models
from research.forecasting.fresh_wind import evaluate as previous
from research.forecasting.weather_retrain.common import digest, write_json
from .prepare import CACHE, HERE, ROOT, raw_history


def cases(*, stress=False):
    state = data.read_state(previous.STATE)
    actual = {(r["area"], r["time"]): r["wind"] for r in state["hourly"]}
    raw, _ = raw_history()
    selection = json.loads((previous.HERE / "selection.json").read_text())
    names = json.loads((ROOT / ".cache/weather-retrain/feature-names.json").read_text())["future"]
    loaded, ramp_selection = load_models()
    output = []
    for path in sorted((previous.CACHE / ("stress" if stress else "cases")).glob("*/predictions.npz")):
        origin = pd.Timestamp(pd.to_datetime(path.parent.name, format="%Y%m%dT%H"), tz="UTC")
        at = origin + pd.Timedelta(minutes=30)
        times = pd.date_range(origin + pd.Timedelta(hours=1), periods=48, freq="h")
        records = raw.loc[at-pd.Timedelta(hours=4):at].to_dict("records")
        recent = recent_wind(records, at)
        with np.load(path) as saved:
            predictions = dict(saved)
        with np.load(path.parent / ("inputs.npz" if stress else "baseline.npz")) as saved:
            base = dict(saved)
        run = (at - pd.Timedelta(hours=8)).floor("6h")
        points, _ = previous.archived("ecmwf", run, times, state)
        future = weather_sources.future_inputs(base, points, times, names)["future"]
        selected = adjust(weighted_wind(predictions, selection["dmiWeights"]), times, recent, selection["correctionRules"])[:, :, 0]
        ramp = selected.copy()
        unavailable = None
        try:
            extended = measurements(records, at)
            matrix, columns = features(base["history"], future, names, extended, times)
            if columns != ramp_selection["featureNames"]:
                raise ValueError("Ramp training/serving feature mismatch")
            for ai, area in enumerate(data.AREAS):
                ramp[ai, :6] = predict(loaded[area], matrix[ai], extended[area]["levelMW"])
        except ValueError as error:
            unavailable = str(error)
        output.append(dict(origin=origin, issuedAt=at, times=times, prediction=selected, ramp=ramp,
            actual=np.array([[actual[(a, t.isoformat())] for t in times] for a in data.AREAS]),
            level=np.array([recent[a]["levelMW"] for a in data.AREAS]),
            disagreement=predictions["dmi"][:, :, 0] - predictions["ecmwf"][:, :, 0], rampUnavailable=unavailable))
    return output


def range_metrics(rows):
    y, lo, hi = (np.asarray([row[k] for row in rows]) for k in ("actual", "lower", "upper"))
    covered = (lo <= y) & (y <= hi)
    return dict(origins=len(rows), coverage=float(covered.mean()),
        intervalScore=float(interval_score(y, lo, hi).mean()), meanWidthMW=float((hi-lo).mean()),
        coverageByLeadBand={f"{start+1}-{end}": float(covered[:, start:end].mean()) for start, end in wind_ranges.BANDS})


def main():
    protocol = json.loads((HERE / "protocol.json").read_text())
    rows = cases()
    evaluation = [c for c in rows if pd.Timestamp(protocol["rampEvaluationStart"]) <= c["origin"] < pd.Timestamp(protocol["rampEvaluationEndExclusive"])]
    report = dict(protocolHash=digest(HERE / "protocol.json"), modelSelectionHash=digest(HERE / "selection.json"),
                  evaluationOrigins=len(evaluation), shortTerm={}, ranges={}, publicPromotion=False,
                  interpretation=protocol["interpretation"])
    report["rampFallbacks"] = [dict(origin=c["origin"].isoformat(), reason=c["rampUnavailable"]) for c in evaluation if c["rampUnavailable"]]
    for ai, area in enumerate(data.AREAS):
        truth, baseline, challenger = (np.array([c[k][ai, :6] for c in evaluation]) for k in ("actual", "prediction", "ramp"))
        old, new = np.abs(truth - baseline), np.abs(truth - challenger)
        levels = np.array([c["level"][ai] for c in evaluation])[:, None]
        ramps = np.abs(truth - np.column_stack([levels, truth[:, :-1]])) >= protocol["rampThresholdMW"][area]
        report["shortTerm"][area] = dict(baselineMAE=float(old.mean()), challengerMAE=float(new.mean()),
            improvementPct=float(100*(1-new.mean()/old.mean())), baselineByLead=old.mean(axis=0).tolist(),
            challengerByLead=new.mean(axis=0).tolist(),
            rampHours=int(ramps.sum()), rampBaselineMAE=float(old[ramps].mean()) if ramps.any() else None,
            rampChallengerMAE=float(new[ramps].mean()) if ramps.any() else None,
            accuracyPass=bool(new.mean() <= old.mean() * .95),
            leadPass=bool((new.mean(axis=0) <= old.mean(axis=0)*1.05).all()),
            longHorizonUnchanged=True)
        candidates = []
        for method in protocol["rangeMethods"]:
            for window in protocol["rangeWindowsDays"]:
                results = []
                for case in rows:
                    prior = [c for c in rows if case["origin"] - pd.Timedelta(days=window) <= c["origin"]
                        and c["times"][-1] + pd.Timedelta(minutes=15) <= case["issuedAt"]]
                    if len(prior) < protocol["minimumCalibrationOrigins"]:
                        continue
                    scales = [wind_ranges.scale(method, c["prediction"][ai], area, disagreement=c["disagreement"][ai]) for c in prior]
                    rule = wind_ranges.fit([c["actual"][ai] for c in prior], [c["prediction"][ai] for c in prior], scales, method)
                    current_scale = wind_ranges.scale(method, case["prediction"][ai], area, disagreement=case["disagreement"][ai])
                    lower, upper = wind_ranges.bounds(case["prediction"][ai], current_scale, rule)
                    results.append(dict(origin=case["origin"], lastLabel=case["times"][-1], lower=lower, upper=upper, actual=case["actual"][ai]))
                # All selection outcomes finish before evaluation begins. Method
                # choice never uses the November 10–13 scores below.
                dev = [r for r in results if r["origin"] < pd.Timestamp("2026-11-08T00:00Z")
                       and r["lastLabel"] < pd.Timestamp(protocol["rampEvaluationStart"])]
                test = [r for r in results if pd.Timestamp(protocol["rampEvaluationStart"]) <= r["origin"] < pd.Timestamp(protocol["rampEvaluationEndExclusive"])]
                if not dev or not test:
                    raise ValueError("No chronological range comparison")
                candidates.append(dict(method=method, windowDays=window, development=range_metrics(dev), evaluation=range_metrics(test)))
        eligible = [c for c in candidates if protocol["minimumCoverage"] <= c["development"]["coverage"] <= protocol["maximumCoverage"]]
        chosen = min(eligible, key=lambda c: c["development"]["intervalScore"]) if eligible else None
        reference = next(c for c in candidates if c["method"] == "absolute" and c["windowDays"] == 14)["evaluation"]
        checks = {}
        if chosen:
            metrics = chosen["evaluation"]
            checks = dict(coveragePass=bool(all(protocol["minimumCoverage"] <= v <= protocol["maximumCoverage"] for v in metrics["coverageByLeadBand"].values())),
                scorePass=bool(metrics["intervalScore"] <= reference["intervalScore"]*.95),
                widthPass=bool(metrics["meanWidthMW"] <= reference["meanWidthMW"]*protocol["maximumWidthMultiplier"]))
        report["ranges"][area] = dict(selected={k: chosen[k] for k in ("method", "windowDays")} if chosen else None,
            checks=checks, candidates=candidates, prospectiveValidationRequired=True)
    write_json(HERE / "results.json", report)
    print(json.dumps({k:v for k,v in report.items() if k!='ranges'}, indent=2), flush=True)
    for area, result in report["ranges"].items():
        print(area, 'ranges', result["selected"], result["checks"], flush=True)


if __name__ == "__main__":
    main()
