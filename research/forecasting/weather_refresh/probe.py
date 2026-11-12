"""Download and score a live candidate locally, without publishing any forecast."""

import argparse
import json
from pathlib import Path

import numpy as np

from forecast import data, weather_runs
from forecast.features import AREAS, TARGETS
from forecast.score import run as score

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True, help="A recent downloaded forecast-state.json.gz")
    parser.add_argument("--score", action="store_true", help="Also run the fixed ensemble on both inputs")
    args = parser.parse_args()
    state = data.read_state(args.state)
    at = data.now()
    # Load and write a separate cache; never write the source state or dist/.
    work = ROOT / ".cache/weather-refresh/live"
    work.mkdir(parents=True, exist_ok=True)
    cache = work / "weather-runs.json"
    if cache.exists():
        state["weatherRuns"] = json.loads(cache.read_text())
    status = weather_runs.refresh(state, at, data.request)
    cache.write_text(json.dumps(state["weatherRuns"], allow_nan=False))
    selected_at = data.now()
    baseline, baseline_meta = data.inputs(state, selected_at)
    candidate, candidate_meta = data.inputs(state, selected_at, latest_weather=True)
    for key in baseline:
        if key != "future":
            np.testing.assert_array_equal(baseline[key], candidate[key], err_msg=key)
    result = {"diagnosticOnly": True, "checkedAt": selected_at.isoformat(), "weatherRefresh": status,
              "baseline": baseline_meta, "candidate": candidate_meta}
    if args.score:
        input_path = work / "inputs.npz"
        np.savez_compressed(input_path, **{
            key: np.concatenate([baseline[key], candidate[key]]) for key in baseline
        })
        prediction, _, timings = score(input_path, work)
        result["runtime"] = timings
        for vi, variant in enumerate(("baseline", "candidate")):
            result[variant]["predictions"] = {
                area: {target: prediction[vi * 2 + ai, :, ti].astype(float).tolist()
                       for ti, target in enumerate(TARGETS)}
                for ai, area in enumerate(AREAS)
            }
    (work / "comparison.json").write_text(json.dumps(result, allow_nan=False, indent=2) + "\n")
    print(json.dumps({"diagnosticOnly": True, "baselineRun": baseline_meta["weatherRun"],
                      "candidateRun": candidate_meta["weatherRun"],
                      "selection": candidate_meta["weatherSelection"],
                      "weatherError": status["error"], "scored": args.score,
                      "output": str(work / "comparison.json")}, indent=2))


if __name__ == "__main__":
    main()
