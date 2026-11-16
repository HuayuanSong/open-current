"""Additional recent cases after fixing source weights; never tune on this check."""

import json

import numpy as np
import pandas as pd

from forecast import data, weather_sources
from forecast.fresh_wind import weighted_wind
from forecast.score import run as baseline_score
from forecast.wind_models import predict_wind
from forecast.wind_nowcast import adjust, recent_wind
from research.forecasting.weather_retrain import compare, prepare
from research.forecasting.weather_retrain.common import digest, write_json
from .evaluate import archived, CACHE, HERE, ROOT, PROTOCOL, STATE, RAW


def main():
    chosen = json.loads((HERE / "results.json").read_text())["dmiWeights"]
    rules = json.loads((ROOT / PROTOCOL["correctionSelection"]).read_text())["rules"]
    manifest = compare.verify_models()
    blend = json.loads((compare.ensure() / "selection.json").read_text())
    state = data.read_state(STATE)
    actual = {(r["area"], r["time"]): r for r in state["hourly"]}
    raw = prepare.read(RAW)["data"]["records"]
    rows = []
    for origin in [pd.Timestamp(f"2026-11-14T{hour:02}:00Z") for hour in (3, 9)]:
        at = origin + pd.Timedelta(minutes=30)
        arrays, meta = data.inputs(state, at)
        if pd.Timestamp(meta["origin"]) != origin:
            raise ValueError("Incomplete stress-test history")
        times = pd.DatetimeIndex(meta["times"])
        truth = np.array([[actual[(a, t)]["wind"] for t in meta["times"]] for a in data.AREAS])
        if not np.isfinite(truth).all():
            raise ValueError("Incomplete stress-test outcomes")
        work = CACHE / "stress" / f"{origin:%Y%m%dT%H}"
        work.mkdir(parents=True, exist_ok=True)
        path = work / "inputs.npz"
        np.savez_compressed(path, **arrays)
        signature = dict(input=digest(path), selection=chosen, inference=digest(ROOT / "forecast/wind_models.py"))
        if (work / "signature.json").exists() and json.loads((work / "signature.json").read_text()) == signature:
            outputs = dict(np.load(work / "predictions.npz"))
        else:
            base, members, _ = baseline_score(path, work / "baseline")
            outputs = dict(baseline=base)
            for source in ("ecmwf", "dmi"):
                run = (at - pd.Timedelta(hours=PROTOCOL["assumedAvailabilityHours"][source])).floor("6h" if source == "ecmwf" else "3h")
                points, _ = archived(source, run, times, state)
                fresh = weather_sources.future_inputs(arrays, points, times, meta["featureNames"])
                outputs[source] = predict_wind(fresh, members, manifest["featureNames"], compare.CACHE / "models/latest", blend)
            np.savez_compressed(work / "predictions.npz", **outputs)
            write_json(work / "signature.json", signature)
        recent = recent_wind(raw, at)
        corrected = adjust(weighted_wind(outputs, chosen), times, recent, rules)
        for ai, area in enumerate(data.AREAS):
            rows.append(dict(origin=origin.isoformat(), area=area,
                baselineMAE=float(np.abs(outputs["baseline"][ai, :, 0] - truth[ai]).mean()),
                selectedMAE=float(np.abs(corrected[ai, :, 0] - truth[ai]).mean())))
        print("Checked", origin.isoformat(), flush=True)
    report = dict(dmiWeights=chosen, rows=rows, interpretation="Two additional recent stress cases, not a separate statistically sufficient validation set. No weights were selected using these outcomes.")
    write_json(HERE / "stress-results.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
