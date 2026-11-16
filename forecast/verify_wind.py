"""Verify the public wind-only refit against the pinned native-model fixture."""

import json

import numpy as np

from .fresh_wind import configuration
from .score import combine
from .wind_models import predict_wind
from research.forecasting.weather_retrain import compare
from research.forecasting.weather_retrain.prospective import ensure_models


def main():
    configuration()
    ensure_models()
    manifest = compare.verify_models()
    release = json.loads((compare.HERE / "model-release.json").read_text())
    for name, expected in release["fixtures"].items():
        if compare.digest(compare.CACHE / "models" / name) != expected:
            raise ValueError("Modified wind model export fixture")
    with np.load(compare.CACHE / "models/fixture.npz") as saved:
        arrays = {k: saved[k] for k in saved.files}
    arrays["future"] = arrays.pop("latest_future")
    with np.load(compare.CACHE / "models/expected.npz") as saved:
        expected = {k: saved[k] for k in saved.files}
    selection = json.loads((compare.ensure() / "selection.json").read_text())
    shape = (len(arrays["area_ids"]), 48, 3)
    names = set(selection["families"]) | {"original_trees", "original_neural", "tree_control"}
    members = {name: np.zeros(shape, np.float32) for name in names}
    reference_members = {k: v.copy() for k, v in members.items()}
    for family in ("tree_control", "ridge", "lightgbm", "xgboost"):
        reference_members[family][:, :, 0] = expected[f"latest__{family}"][:, :, 0]
    reference = combine(reference_members, arrays["area_ids"], selection)[0]
    result = predict_wind(arrays, members, manifest["featureNames"], compare.CACHE / "models/latest", selection)
    np.testing.assert_allclose(result[:, :, 0], reference[:, :, 0], rtol=2e-5, atol=.02)
    np.testing.assert_array_equal(result[:, :, 1:], np.zeros_like(result[:, :, 1:]))
    print(json.dumps(dict(windRefitParity="passed", maxDifferenceMW=float(np.max(np.abs(result - reference))))))


if __name__ == "__main__":
    main()
