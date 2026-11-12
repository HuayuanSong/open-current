"""Check native model exports against a fixed Mac-generated prediction fixture."""

import json

import numpy as np

from .common import CACHE, HERE, digest
from .models import predict
from .prospective import ensure_models, verify_inference


def main():
    manifest = ensure_models()
    verify_inference()
    release = json.loads((HERE / "model-release.json").read_text())
    for name, expected in release["fixtures"].items():
        if digest(CACHE / "models" / name) != expected:
            raise ValueError("Modified export fixture")
    arrays = dict(np.load(CACHE / "models/fixture.npz"))
    latest = arrays.pop("latest_future")
    expected = dict(np.load(CACHE / "models/expected.npz"))
    differences = {}
    for policy in ("daily", "latest"):
        inputs = arrays if policy == "daily" else {**arrays, "future": latest}
        predictions = predict(inputs, manifest["featureNames"], CACHE / "models" / policy)
        for family, values in predictions.items():
            name = f"{policy}__{family}"
            np.testing.assert_allclose(values, expected[name], rtol=2e-5, atol=.02, err_msg=name)
            differences[name] = float(np.abs(values - expected[name]).max())
    print(json.dumps(dict(status="passed", maxDifferenceMW=differences)), flush=True)


if __name__ == "__main__":
    main()
