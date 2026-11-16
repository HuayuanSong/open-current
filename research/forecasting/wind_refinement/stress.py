"""Check two later cached cases without changing settings or selection."""

import json

import numpy as np

from forecast.data import AREAS
from research.forecasting.weather_retrain.common import digest, write_json
from .evaluate import cases
from .prepare import HERE


def main():
    rows = []
    for case in cases(stress=True):
        for ai, area in enumerate(AREAS):
            truth = case["actual"][ai, :6]
            rows.append(dict(origin=case["origin"].isoformat(), area=area,
                baselineMAE=float(np.abs(truth - case["prediction"][ai, :6]).mean()),
                challengerMAE=float(np.abs(truth - case["ramp"][ai, :6]).mean()),
                fallback=case["rampUnavailable"]))
    result = dict(modelSelectionHash=digest(HERE / "selection.json"), rows=rows,
        interpretation="Two later November 14 origins; a small stress check, not independent proof of generalization. No model or setting was changed using these outcomes.")
    write_json(HERE / "stress-results.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
