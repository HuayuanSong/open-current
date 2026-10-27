"""Reload trained weights and verify they reproduce saved forecast predictions."""
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import subprocess
import sys

from build_data import AREAS, TARGETS, tabular_features
from fetch_data import CACHE


def main():
    data = dict(np.load(CACHE / "dataset.npz"))
    meta = pd.read_csv(CACHE / "samples.csv")
    expected = np.load(CACHE / "trained_predictions.npz")
    selected = json.loads((CACHE / "selection.json").read_text())
    names = json.loads((CACHE / "feature_names.json").read_text())["future"]
    indices = [np.flatnonzero(meta.area.eq(a) & meta.split.eq("test"))[0] for a in AREAS]
    maximum = 0
    for ti, target in enumerate(TARGETS):
        for ai, area in enumerate(AREAS):
            index = indices[ai]
            x, extra = tabular_features(data["history"][index:index + 1], data["future"][index:index + 1], ti)
            for family in ["ridge", "lightgbm", "lightgbm_no_weather"]:
                if family == "ridge":
                    model = joblib.load(CACHE / "models" / f"{family}_{area}_{target}.joblib")
                    prediction = model.predict(x)
                else:
                    model = lgb.Booster(model_file=str(CACHE / "models" / f"{family}_{area}_{target}.txt"))
                    columns = json.loads((CACHE / "models" / f"{family}_{area}_{target}_features.json").read_text())
                    all_names = names + extra
                    prediction = model.predict(x[:, [all_names.index(c) for c in columns]], num_threads=4)
                prediction = np.maximum(prediction, 0)
                reference = expected[family][index, :, ti]
                np.testing.assert_allclose(prediction, reference, rtol=1e-5, atol=0.001)
                maximum = max(maximum, float(np.max(np.abs(prediction - reference))))
    subprocess.run([sys.executable, str(Path(__file__).with_name("verify_neural.py"))], check=True)
    print(f"Reloaded 18 statistical/tree models; predictions agree (maximum absolute difference {maximum:.6f} MW).")


if __name__ == "__main__":
    main()
