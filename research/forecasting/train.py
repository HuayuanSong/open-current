"""Small, predeclared model comparison; test targets never select settings."""
from __future__ import annotations

import os
for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[variable] = "4"

import datetime
import hashlib
import json
import platform
import time
import subprocess
import sys

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from build_data import AREAS, H, L, TARGETS, tabular_features
from fetch_data import CACHE, CONFIG, HERE

MODELS = CACHE / "models"
MODELS.mkdir(parents=True, exist_ok=True)


def load_data():
    arrays = dict(np.load(CACHE / "dataset.npz"))
    meta = pd.read_csv(CACHE / "samples.csv")
    names = json.loads((CACHE / "feature_names.json").read_text())
    return arrays, meta, names


def mae(y, prediction):
    return float(np.mean(np.abs(y - prediction)))


def benchmark(function, repetitions=10):
    function()
    durations = []
    for _ in range(repetitions):
        start = time.perf_counter()
        function()
        durations.append(time.perf_counter() - start)
    return float(np.median(durations))


def run_mlp(target, width):
    # A fresh process is required on macOS: the wheel runtimes for LightGBM
    # and PyTorch can deadlock when both OpenMP pools are active together.
    subprocess.run([sys.executable, str(HERE / "neural.py"), str(target), str(width)], check=True)
    name = f"mlp_{TARGETS[target]}_{width}"
    prediction = np.load(MODELS / f"{name}_predictions.npz")["prediction"]
    info = json.loads((MODELS / f"{name}_info.json").read_text())
    return prediction, info


def main():
    run_start = time.perf_counter()
    run_info = {"started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "platform": platform.platform(), "python": platform.python_version(),
                "threads": CONFIG["threads"],
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in HERE.iterdir() if p.suffix in [".py", ".json"] or p.name == "PROTOCOL.md"}}
    (CACHE / "training_run.json").write_text(json.dumps(run_info, indent=2))
    arrays, meta, names = load_data()
    y, history, future = arrays["y"], arrays["history"], arrays["future"]
    predictions = {
        "persistence": np.repeat(history[:, -1:, :], H, axis=1),
        "last_day": history[:, -24:, :][:, np.arange(H) % 24],
        "last_week": history[:, :H, :],
    }
    for name in ["ridge", "lightgbm", "lightgbm_no_weather", "weather_mlp", "blend", "lightgbm_p10", "lightgbm_p90", "lightgbm_cal_p10", "lightgbm_cal_p90"]:
        predictions[name] = np.full_like(y, np.nan)
    selection, timings, tuning = [], [], []
    all_feature_names = None
    settings = [{"num_leaves": 15, "n_estimators": 400, "min_child_samples": 100},
                {"num_leaves": 31, "n_estimators": 700, "min_child_samples": 60}]
    for ti, target in enumerate(TARGETS):
        x, extra_names = tabular_features(history, future, ti)
        feature_names = names["future"] + extra_names
        all_feature_names = feature_names
        no_weather = [i for i, name in enumerate(feature_names) if not name.startswith("wx_")]
        for ai, area in enumerate(AREAS):
            sample_mask = meta.area_id.eq(ai).to_numpy()
            mask = np.repeat(sample_mask, H)
            train = np.repeat(sample_mask & meta.split.eq("train").to_numpy(), H)
            val = np.repeat(sample_mask & meta.split.eq("validation").to_numpy(), H)
            truth = y[:, :, ti].flatten()
            ridges = []
            for alpha in [1.0, 100.0, 10000.0]:
                start = time.perf_counter()
                model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
                model.fit(x[train], truth[train])
                elapsed = time.perf_counter() - start
                score = mae(truth[val], np.maximum(model.predict(x[val]), 0))
                ridges.append((score, alpha, model, elapsed))
                tuning.append({"family": "ridge", "area": area, "target": target, "setting": alpha, "validation_mae": score})
            score, alpha, model, elapsed = min(ridges, key=lambda v: v[0])
            predictions["ridge"][sample_mask, :, ti] = np.maximum(model.predict(x[mask]).reshape(-1, H), 0)
            path = MODELS / f"ridge_{area}_{target}.joblib"
            joblib.dump(model, path)
            timings.append({"family": "ridge", "area": area, "target": target, "training_seconds": elapsed,
                            "inference_seconds_one_area": benchmark(lambda: model.predict(x[mask][:H])), "bytes": path.stat().st_size})
            selection.append({"family": "ridge", "area": area, "target": target, "setting": alpha, "validation_mae": score})
            best_tree_setting = None
            for family, columns in [("lightgbm", list(range(x.shape[1]))), ("lightgbm_no_weather", no_weather)]:
                xx = x[:, columns]
                candidates = []
                for number, setting in enumerate(settings):
                    start = time.perf_counter()
                    tree = lgb.LGBMRegressor(**setting, objective="regression_l1", learning_rate=0.04,
                        reg_lambda=10, n_jobs=CONFIG["threads"], random_state=CONFIG["seed"], verbosity=-1,
                        deterministic=True, force_col_wise=True)
                    tree.fit(xx[train], truth[train])
                    elapsed = time.perf_counter() - start
                    score = mae(truth[val], np.maximum(tree.predict(xx[val]), 0))
                    candidates.append((score, number, tree, elapsed))
                    tuning.append({"family": family, "area": area, "target": target, "setting": number, "validation_mae": score})
                score, number, tree, elapsed = min(candidates, key=lambda v: v[0])
                predictions[family][sample_mask, :, ti] = np.maximum(tree.predict(xx[mask]).reshape(-1, H), 0)
                path = MODELS / f"{family}_{area}_{target}.txt"
                tree.booster_.save_model(str(path))
                (MODELS / f"{family}_{area}_{target}_features.json").write_text(json.dumps([feature_names[c] for c in columns]))
                timings.append({"family": family, "area": area, "target": target, "training_seconds": elapsed,
                                "inference_seconds_one_area": benchmark(lambda: tree.predict(xx[mask][:H])), "bytes": path.stat().st_size})
                selection.append({"family": family, "area": area, "target": target, "setting": number, "validation_mae": score})
                if family == "lightgbm":
                    best_tree_setting = settings[number]
                    pd.DataFrame({"feature": feature_names, "gain": tree.booster_.feature_importance(importance_type="gain")}).sort_values("gain", ascending=False).to_csv(MODELS / f"importance_{area}_{target}.csv", index=False)
                print(f"{family} {area}/{target}: validation MAE {score:.2f}, selected setting {number}, fit {elapsed:.1f}s", flush=True)
            for quantile in [0.1, 0.9]:
                start = time.perf_counter()
                tree = lgb.LGBMRegressor(**best_tree_setting, objective="quantile", alpha=quantile, learning_rate=0.04,
                    reg_lambda=10, n_jobs=CONFIG["threads"], random_state=CONFIG["seed"], verbosity=-1,
                    deterministic=True, force_col_wise=True)
                tree.fit(x[train], truth[train])
                elapsed = time.perf_counter() - start
                family = f"lightgbm_p{int(quantile * 100)}"
                predictions[family][sample_mask, :, ti] = np.maximum(tree.predict(x[mask]).reshape(-1, H), 0)
                path = MODELS / f"{family}_{area}_{target}.txt"
                tree.booster_.save_model(str(path))
                timings.append({"family": family, "area": area, "target": target, "training_seconds": elapsed,
                                "inference_seconds_one_area": benchmark(lambda: tree.predict(x[mask][:H])), "bytes": path.stat().st_size})
        neural = [run_mlp(ti, width) for width in [128, 256]]
        for prediction, info in neural:
            timings.append(info)
        # Architecture selected per target on the mean of area-normalized MAEs.
        scores = []
        for prediction, info in neural:
            local = []
            for ai, area in enumerate(AREAS):
                valid = meta.area_id.eq(ai) & meta.split.eq("validation")
                training = meta.area_id.eq(ai) & meta.split.eq("train")
                score = mae(y[valid, :, ti], prediction[valid])
                local.append(score / y[training, :, ti].std())
                tuning.append({"family": "weather_mlp", "area": area, "target": target, "setting": info["width"], "validation_mae": score})
            scores.append(np.mean(local))
        ni = int(np.argmin(scores))
        predictions["weather_mlp"][:, :, ti] = neural[ni][0]
        selection.append({"family": "weather_mlp", "area": "pooled", "target": target, "setting": neural[ni][1]["width"], "validation_scaled_mae": float(scores[ni])})
        predictions["blend"][:, :, ti] = 0.5 * predictions["weather_mlp"][:, :, ti] + 0.5 * predictions["lightgbm"][:, :, ti]
        # Save progress after every target, keeping a usable local checkpoint.
        np.savez_compressed(CACHE / "trained_predictions.npz", **predictions)
    # Sort quantiles independently of observations. The median point prediction
    # is included to avoid a reported interval excluding its own point forecast.
    ordered = np.sort(np.stack([predictions["lightgbm_p10"], predictions["lightgbm"], predictions["lightgbm_p90"]]), axis=0)
    predictions["lightgbm_p10"], predictions["lightgbm_p90"] = ordered[0], ordered[2]
    calibration = []
    for ai, area in enumerate(AREAS):
        am = meta.area_id.eq(ai).to_numpy()
        cm = am & meta.split.eq("calibration").to_numpy()
        for ti, target in enumerate(TARGETS):
            for start, stop in [(0, 6), (6, 24), (24, 48)]:
                lower = predictions["lightgbm_p10"][cm, start:stop, ti]
                upper = predictions["lightgbm_p90"][cm, start:stop, ti]
                actual = y[cm, start:stop, ti]
                scores = np.maximum(lower - actual, actual - upper).flatten()
                # Empirical calibration under dependent data, not a claim of
                # exchangeable conformal finite-sample coverage guarantees.
                level = min(1, np.ceil((len(scores) + 1) * 0.8) / len(scores))
                correction = max(0, float(np.quantile(scores, level, method="higher")))
                predictions["lightgbm_cal_p10"][am, start:stop, ti] = np.maximum(0, predictions["lightgbm_p10"][am, start:stop, ti] - correction)
                predictions["lightgbm_cal_p90"][am, start:stop, ti] = predictions["lightgbm_p90"][am, start:stop, ti] + correction
                calibration.append({"area": area, "target": target, "lead_start": start + 1, "lead_end": stop, "expansion_mw": correction})
    np.savez_compressed(CACHE / "trained_predictions.npz", **predictions)
    (CACHE / "selection.json").write_text(json.dumps(selection, indent=2))
    (CACHE / "training_timings.json").write_text(json.dumps(timings, indent=2))
    (CACHE / "tuning.json").write_text(json.dumps(tuning, indent=2))
    (CACHE / "calibration.json").write_text(json.dumps(calibration, indent=2))
    run_info["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    run_info["total_wall_seconds"] = time.perf_counter() - run_start
    (CACHE / "training_run.json").write_text(json.dumps(run_info, indent=2))
    print("Training complete; settings selected using validation only.", flush=True)


if __name__ == "__main__":
    main()
