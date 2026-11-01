"""Refresh source history, score the fixed ensemble, retain issued forecasts."""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from . import data, intervals
from .artifacts import ensure, CONFIG, ROOT
from .publish import write_public
from .score import run as score


def _series_by_area(values):
    return {
        area: {
            target: values[area_index, :, target_index].astype(float).tolist()
            for target_index, target in enumerate(data.TARGETS)
        }
        for area_index, area in enumerate(data.AREAS)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    at = data.now()
    state = data.read_state()
    version = CONFIG["version"]
    if args.restore:
        # A restore failure must not reset the issued-forecast archive.
        state = data.restore(
            os.environ.get(
                "SNAPSHOT_URL", "https://huayuansong.github.io/open-current/"
            )
        )
    error = None
    try:
        if not args.offline:
            data.electricity(state, at)
            data.weather(state, at)
        intervals.refresh_history(state, at)
        arrays, metadata = data.inputs(state, at)
        bundle = ensure()
        feature_names = json.loads((bundle / "feature_names.json").read_text())
        if metadata.pop("featureNames") != feature_names["future"]:
            raise ValueError("Feature schema differs from trained models")
        range_version = json.loads(intervals.CALIBRATION.read_text())["version"]
        identifier = (
            f'{version}/{metadata["origin"]}/{metadata["weatherRun"]}/{range_version}'
        )
        if not any(r["id"] == identifier for r in state["forecasts"]):
            work_dir = ROOT / ".cache/forecast-serving/run"
            work_dir.mkdir(parents=True, exist_ok=True)
            input_path = work_dir / "inputs.npz"
            np.savez_compressed(input_path, **arrays)
            prediction, families, timings = score(input_path, work_dir, explain=True)
            # An operational plausibility guard, not a correction to predictions.
            limits = np.array([[15000, 15000, 20000], [10000, 10000, 15000]])
            if (prediction < 0).any() or (prediction > limits[:, None, :]).any():
                raise ValueError("Forecast failed the physical plausibility check")
            record = {
                "id": identifier,
                "modelVersion": version,
                "issuedAt": data.now().isoformat(),
                **metadata,
                "areas": _series_by_area(prediction),
                "treeControl": _series_by_area(families["tree_control"]),
                "runtime": timings,
            }
            radiation_columns = [
                index
                for index, name in enumerate(feature_names["future"])
                if name.endswith("terrestrial_radiation")
            ]
            daylight = arrays["future"][:, :, radiation_columns].mean(-1) > 5
            record["daylight"] = {
                area: daylight[area_index].tolist()
                for area_index, area in enumerate(data.AREAS)
            }
            record["ranges"] = intervals.build(
                prediction, daylight, record["issuedAt"], state, record["origin"]
            )
            # A late source hour must not acquire retrospective prediction ranges.
            issued_at = pd.Timestamp(record["issuedAt"])
            for index, stamp in enumerate(record["times"]):
                if pd.Timestamp(stamp) <= issued_at:
                    for targets in record["ranges"]["areas"].values():
                        for bounds in targets.values():
                            bounds["lower"][index] = None
                            bounds["upper"][index] = None
            record["explanations"] = json.loads(
                (work_dir / "trees.explanations.json").read_text()
            )
            state["forecasts"].append(record)
            print(f"Issued {identifier}", flush=True)
        else:
            print(
                "Forecast for this origin already issued; preserving its original values.",
                flush=True,
            )
    except Exception as exc:
        # Keep prior forecasts; never silently drop a model or change blend weights.
        error = str(exc)[:240]
        print(f"Forecast update unavailable: {error}", file=sys.stderr, flush=True)
    checked = data.now()
    state["updatedAt"] = checked.isoformat()
    retention_cutoff = checked - pd.Timedelta(days=35)
    state["forecasts"] = [
        r for r in state["forecasts"] if pd.Timestamp(r["issuedAt"]) >= retention_cutoff
    ]
    data.write_state(state)
    public = write_public(state, checked, version, error)
    print(
        json.dumps(
            {
                "status": public["status"],
                "forecastsRetained": len(state["forecasts"]),
                "origin": public["latest"]["origin"] if public["latest"] else None,
                "error": error,
            }
        ),
        flush=True,
    )
    if error and args.strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
