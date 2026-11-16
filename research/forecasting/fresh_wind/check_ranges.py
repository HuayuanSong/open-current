"""Check the selected wind model's ranges without tuning on evaluation outcomes."""

import json

import numpy as np
import pandas as pd

from forecast import data, intervals
from forecast.fresh_wind import weighted_wind
from forecast.wind_nowcast import adjust, recent_wind
from research.forecasting.weather_retrain import prepare
from research.forecasting.weather_retrain.common import write_json
from .evaluate import CACHE, HERE, ROOT, PROTOCOL, STATE, RAW


def main():
    result = json.loads((HERE / "results.json").read_text())
    state = data.read_state(STATE)
    actual = {(r["area"], r["time"]): r for r in state["hourly"]}
    raw = prepare.read(RAW)["data"]["records"]
    rules = json.loads((ROOT / PROTOCOL["correctionSelection"]).read_text())["rules"]
    cases = []
    for path in sorted((CACHE / "cases").glob("*/predictions.npz")):
        origin = pd.Timestamp(pd.to_datetime(path.parent.name, format="%Y%m%dT%H"), tz="UTC")
        times = pd.date_range(origin + pd.Timedelta(hours=1), periods=48, freq="h")
        if str(origin.date()) >= PROTOCOL["developmentEndExclusive"] or str(times[-1].date()) >= PROTOCOL["evaluationStart"]:
            continue
        mixed = weighted_wind(dict(np.load(path)), result["dmiWeights"])
        recent = recent_wind(raw, origin + pd.Timedelta(minutes=PROTOCOL["issueMinutesAfterOrigin"]))
        cases.append((adjust(mixed, times, recent, rules)[:, :, 0],
                      np.array([[actual[(a, t.isoformat())]["wind"] for t in times] for a in data.AREAS])))
    frame = pd.read_csv(CACHE / "predictions.csv")
    report = dict(calibrationOrigins=len(cases), calibratedBefore=PROTOCOL["evaluationStart"],
                  nominalCoverage=.8, series={})
    for ai, area in enumerate(data.AREAS):
        p = np.array([c[0][ai] for c in cases]); y = np.array([c[1][ai] for c in cases])
        rule = intervals.fit(y, p, np.ones_like(p, bool), "wind", "absolute", 1)
        later = frame[(frame.area == area) & (frame.variant == "selected")]
        pred = later.pivot(index="origin", columns="lead", values="prediction").to_numpy()
        truth = later.pivot(index="origin", columns="lead", values="actual").to_numpy()
        lower, upper = zip(*(intervals.bounds(row, np.ones(48, bool), "wind", rule) for row in pred))
        lower, upper = np.array(lower), np.array(upper)
        report["series"][area] = dict(coverage=float(np.mean((lower <= truth) & (truth <= upper))),
            intervalScore=float(intervals.interval_score(truth, lower, upper).mean()),
            meanWidthMW=float(np.mean(upper - lower)), evaluationOrigins=len(pred), rule=rule)
    write_json(HERE / "range-check.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
