"""Isolated CPU neural training: avoids mixed LightGBM/PyTorch OpenMP runtimes."""
from __future__ import annotations
import os
for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[variable] = "4"
import argparse
import copy
import json
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from build_data import H, L, TARGETS
from fetch_data import CACHE, CONFIG
MODELS = CACHE / "models"
MODELS.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(CONFIG["threads"])
torch.set_num_interop_threads(1)

class StandardScaler:
    """Training-only standardization, without loading another OpenMP runtime."""
    def fit(self, values):
        values = values.astype(np.float64)
        self.mean_ = values.mean(axis=0)
        self.scale_ = values.std(axis=0)
        self.scale_[self.scale_ == 0] = 1
        return self
    def transform(self, values):
        return ((values.astype(np.float64) - self.mean_) / self.scale_).astype(np.float32)

def benchmark(function, repetitions=10):
    function()
    durations = []
    for _ in range(repetitions):
        start = time.perf_counter()
        function()
        durations.append(time.perf_counter() - start)
    return float(np.median(durations))

class ResidualBlock(nn.Module):
    def __init__(self, inputs, hidden, outputs, dropout=0.1):
        super().__init__()
        self.main = nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU(), nn.Linear(hidden, outputs), nn.Dropout(dropout))
        self.skip = nn.Linear(inputs, outputs)
        self.norm = nn.LayerNorm(outputs)

    def forward(self, x):
        return self.norm(self.main(x) + self.skip(x))


class WeatherMLP(nn.Module):
    """TiDE-inspired model adapted for origin-specific future NWP covariates.

    Unlike the paper, no past weather is fed to the encoder. Each sample has
    its own issued future weather; a stitched perfect-weather series is avoided.
    """
    def __init__(self, features, width, target):
        super().__init__()
        self.project = ResidualBlock(features, 32, 8)
        self.encoder = nn.Sequential(ResidualBlock(L + H * 8 + 2, width, width),
                                     ResidualBlock(width, width, width))
        self.decoder = ResidualBlock(width, width, H * 8)
        self.temporal = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 1))
        self.linear = nn.Linear(L, H)
        # Start from an appropriate deterministic baseline; learn corrections.
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.zero_()
            for lead in range(H):
                index = L - 1 if target == 0 else (L - 24 + lead % 24 if target == 1 else lead)
                self.linear.weight[lead, index] = 1

    def forward(self, history, future, area):
        projected = self.project(future)
        encoded = self.encoder(torch.cat([history, projected.flatten(1), area], dim=1))
        decoded = self.decoder(encoded).reshape(-1, H, 8)
        residual = self.temporal(torch.cat([decoded, projected], dim=-1)).squeeze(-1)
        return self.linear(history) + residual


def train_mlp(arrays, meta, target, width):
    torch.manual_seed(CONFIG["seed"])
    train = meta.split.eq("train").to_numpy()
    val = meta.split.eq("validation").to_numpy()
    area_ids = meta.area_id.to_numpy()
    means, scales = [], []
    for area in range(2):
        yt = arrays["y"][train & (area_ids == area), :, target]
        means.append(float(yt.mean()))
        scales.append(float(max(yt.std(), 1)))
    mean = np.array(means)[area_ids, None]
    scale = np.array(scales)[area_ids, None]
    history = torch.tensor((arrays["history"][:, :, target] - mean) / scale, dtype=torch.float32)
    target_values = torch.tensor((arrays["y"][:, :, target] - mean) / scale, dtype=torch.float32)
    scaler = StandardScaler().fit(arrays["future"][train].reshape(-1, arrays["future"].shape[-1]))
    future = torch.tensor(scaler.transform(arrays["future"].reshape(-1, arrays["future"].shape[-1])).reshape(arrays["future"].shape), dtype=torch.float32)
    areas = torch.tensor(np.eye(2)[area_ids], dtype=torch.float32)
    model = WeatherMLP(future.shape[-1], width, target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    indices = np.flatnonzero(train)
    best_score, stale, best_epoch, best_state = np.inf, 0, 0, None
    rng = np.random.default_rng(CONFIG["seed"])
    curve = []
    start = time.perf_counter()
    for epoch in range(160):
        model.train()
        losses = []
        shuffled = rng.permutation(indices)
        for offset in range(0, len(shuffled), 64):
            batch = shuffled[offset:offset + 64]
            optimizer.zero_grad()
            predicted = model(history[batch], future[batch], areas[batch])
            loss = nn.functional.l1_loss(predicted, target_values[batch])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            value = float(nn.functional.l1_loss(model(history[val], future[val], areas[val]), target_values[val]))
        curve.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation_loss": value})
        if value < best_score - 0.0001:
            best_score, stale, best_epoch = value, 0, epoch + 1
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= 20:
            break
    elapsed = time.perf_counter() - start
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        output = model(history, future, areas).numpy() * scale + mean
        seconds = benchmark(lambda: model(history[:2], future[:2], areas[:2]))
    path = MODELS / f"mlp_{TARGETS[target]}_{width}.pt"
    torch.save({"state_dict": best_state, "width": width, "target": target,
                "features": future.shape[-1], "means": means, "scales": scales,
                "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist()}, path)
    (MODELS / f"mlp_{TARGETS[target]}_{width}_curve.json").write_text(json.dumps(curve))
    info = {"family": "weather_mlp", "target": TARGETS[target], "width": width,
            "best_epoch": best_epoch, "epochs": len(curve), "training_seconds": elapsed,
            "inference_seconds_two_areas": seconds, "parameters": sum(p.numel() for p in model.parameters()),
            "bytes": path.stat().st_size}
    print("MLP", json.dumps(info), flush=True)
    return np.maximum(output, 0).astype(np.float32), info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=int)
    parser.add_argument("width", type=int)
    args = parser.parse_args()
    data = dict(np.load(CACHE / "dataset.npz"))
    meta = pd.read_csv(CACHE / "samples.csv")
    prediction, info = train_mlp(data, meta, args.target, args.width)
    name = f"mlp_{TARGETS[args.target]}_{args.width}"
    np.savez_compressed(MODELS / f"{name}_predictions.npz", prediction=prediction)
    (MODELS / f"{name}_info.json").write_text(json.dumps(info, indent=2))
