"""Evaluate pinned, publicly downloadable Chronos-2 on identical forecast origins."""
from __future__ import annotations

import os
from pathlib import Path
os.environ["HF_HOME"] = str(Path(__file__).resolve().parents[2] / ".cache/forecasting/huggingface")
for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[variable] = "4"

import json
import resource
import time

import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline

from fetch_data import CACHE, CONFIG, HERE


def main():
    torch.set_num_threads(CONFIG["threads"])
    torch.set_num_interop_threads(1)
    data = dict(np.load(CACHE / "dataset.npz"))
    meta = pd.read_csv(CACHE / "samples.csv")
    names = json.loads((CACHE / "feature_names.json").read_text())["compact"]
    revision = json.loads((HERE / "chronos_model.json").read_text())
    start = time.perf_counter()
    pipeline = Chronos2Pipeline.from_pretrained(revision["model"], revision=revision["revision"],
                                               device_map="cpu", torch_dtype=torch.float32)
    load_seconds = time.perf_counter() - start
    print(f"Loaded Chronos-2 on CPU in {load_seconds:.2f}s", flush=True)
    output = np.full((*data["y"].shape, 3), np.nan, dtype=np.float32)
    path = CACHE / "chronos_predictions.npz"
    if path.exists():
        output = np.load(path)["quantiles"]

    def example(index):
        return {"target": data["history"][index].T,
                "past_covariates": {name: data["past_cov"][index, :, j] for j, name in enumerate(names)},
                "future_covariates": {name: data["future_cov"][index, :, j] for j, name in enumerate(names)}}

    indices = np.flatnonzero(~meta.split.eq("train").to_numpy())
    # Disable cross-learning: different forecast origins must never exchange
    # information, since a later origin's history contains earlier test labels.
    def predict(batch):
        return pipeline.predict_quantiles(batch, prediction_length=CONFIG["horizon"],
            quantile_levels=[0.1, 0.5, 0.9], context_length=CONFIG["lookback"],
            batch_size=64, cross_learning=False)[0]

    timings = []
    warm = [example(int(i)) for i in indices[:2]]
    predict(warm)
    for _ in range(5):
        start = time.perf_counter()
        predict(warm)
        timings.append(time.perf_counter() - start)
    start = time.perf_counter()
    for offset in range(0, len(indices), 8):
        batch_indices = [i for i in indices[offset:offset + 8] if not np.isfinite(output[i]).all()]
        if not batch_indices:
            continue
        result = predict([example(int(i)) for i in batch_indices])
        for index, values in zip(batch_indices, result, strict=True):
            output[index] = np.maximum(values.numpy().transpose(1, 0, 2), 0)
        np.savez_compressed(path, quantiles=output)
        if offset % 40 == 0:
            print(f"Chronos-2 {min(offset + 8, len(indices))}/{len(indices)} area-origins", flush=True)
    info = {**revision, "device": "cpu", "threads": CONFIG["threads"],
            "load_seconds": load_seconds, "inference_seconds_two_areas": float(np.median(timings)),
            "evaluation_seconds": time.perf_counter() - start,
            "parameter_count": sum(p.numel() for p in pipeline.model.parameters()),
            "peak_process_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "cross_learning": False, "trained_locally": False,
            "covariates": names, "evaluated_area_origins": len(indices)}
    (CACHE / "chronos_timings.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2), flush=True)


if __name__ == "__main__":
    main()
