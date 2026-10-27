"""Reload neural weights in an isolated process; no LightGBM runtime loaded."""
import json
import numpy as np
import pandas as pd
import torch
from build_data import AREAS, TARGETS
from fetch_data import CACHE
from neural import WeatherMLP

def main():
    data = dict(np.load(CACHE / "dataset.npz"))
    meta = pd.read_csv(CACHE / "samples.csv")
    expected = np.load(CACHE / "trained_predictions.npz")
    selected = json.loads((CACHE / "selection.json").read_text())
    indices = [np.flatnonzero(meta.area.eq(a) & meta.split.eq("test"))[0] for a in AREAS]
    maximum = 0
    for ti, target in enumerate(TARGETS):
        width = next(r["setting"] for r in selected if r["family"] == "weather_mlp" and r["target"] == target)
        artifact = torch.load(CACHE / "models" / f"mlp_{target}_{width}.pt", map_location="cpu", weights_only=True)
        model = WeatherMLP(artifact["features"], width, ti)
        model.load_state_dict(artifact["state_dict"])
        model.eval()
        means = np.asarray(artifact["means"])[:, None]
        scales = np.asarray(artifact["scales"])[:, None]
        history = torch.tensor((data["history"][indices, :, ti] - means) / scales, dtype=torch.float32)
        future = torch.tensor((data["future"][indices] - artifact["scaler_mean"]) / artifact["scaler_scale"], dtype=torch.float32)
        with torch.inference_mode():
            prediction = np.maximum(0, model(history, future, torch.eye(2)).numpy() * scales + means)
        reference = expected["weather_mlp"][indices, :, ti]
        np.testing.assert_allclose(prediction, reference, rtol=1e-5, atol=0.005)
        maximum = max(maximum, float(np.max(np.abs(prediction - reference))))
    print(f"Reloaded 3 selected neural models; maximum prediction difference {maximum:.6f} MW.")

if __name__ == "__main__":
    main()
