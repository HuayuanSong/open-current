"""Evaluate the frozen models, export tables/figures, and record limitations."""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_data import AREAS, H, TARGETS
from fetch_data import CACHE, CONFIG, HERE

LABELS = {"persistence": "Persistence", "last_day": "Last day", "last_week": "Last week",
          "ridge": "Ridge + weather", "lightgbm": "LightGBM + weather",
          "lightgbm_no_weather": "LightGBM, no weather", "weather_mlp": "Residual weather MLP",
          "blend": "Tree/MLP blend", "chronos2": "Chronos-2 zero-shot"}


def losses(actual, predicted):
    error = predicted - actual
    return {"mae_mw": float(np.mean(np.abs(error))), "rmse_mw": float(np.sqrt(np.mean(error ** 2))),
            "bias_mw": float(np.mean(error)), "n": int(actual.size)}


def interval_scores(actual, lower, median, upper):
    pinballs = []
    for q, values in [(0.1, lower), (0.5, median), (0.9, upper)]:
        error = actual - values
        pinballs.append(np.mean(np.maximum(q * error, (q - 1) * error)))
    return {"coverage_80": float(np.mean((actual >= lower) & (actual <= upper))),
            "mean_width_mw": float(np.mean(upper - lower)),
            "mean_pinball_mw": float(np.mean(pinballs)), "n": int(actual.size)}


def bootstrap_skill(actual, candidate, baseline, origins):
    # Calendar-week clusters preserve dependence across overlapping 48h horizons.
    frame = pd.DataFrame({"origin": pd.to_datetime(origins, utc=True),
        "candidate": np.abs(candidate - actual).mean(axis=1),
        "baseline": np.abs(baseline - actual).mean(axis=1)})
    frame["block"] = frame.origin.map(lambda x: x.value // (7 * 86400 * 10**9))
    sums = frame.groupby("block")[["candidate", "baseline"]].sum().to_numpy()
    rng = np.random.default_rng(CONFIG["seed"])
    indices = rng.integers(0, len(sums), size=(2000, len(sums)))
    boot = sums[indices].sum(axis=1)
    skill = 100 * (1 - boot[:, 0] / boot[:, 1])
    return np.quantile(skill, [0.025, 0.975]).tolist()


def main():
    data = dict(np.load(CACHE / "dataset.npz"))
    meta = pd.read_csv(CACHE / "samples.csv")
    predictions = dict(np.load(CACHE / "trained_predictions.npz"))
    chronos = np.load(CACHE / "chronos_predictions.npz")["quantiles"]
    predictions.update({"chronos2": chronos[..., 1], "chronos2_p10": chronos[..., 0], "chronos2_p90": chronos[..., 2]})
    point_models = list(LABELS)
    y = data["y"]
    selected, metric_rows, interval_rows, monthly_rows = [], [], [], []
    for ai, area in enumerate(AREAS):
        am = meta.area_id.eq(ai).to_numpy()
        val = am & meta.split.eq("validation").to_numpy()
        test = am & meta.split.eq("test").to_numpy()
        training = am & meta.split.eq("train").to_numpy()
        for ti, target in enumerate(TARGETS):
            scores = {name: float(np.abs(predictions[name][val, :, ti] - y[val, :, ti]).mean()) for name in point_models}
            if not all(np.isfinite(list(scores.values()))):
                raise ValueError(f"Incomplete validation predictions: {area}/{target}: {scores}")
            best_baseline = min(["persistence", "last_day", "last_week"], key=scores.get)
            winner = min(scores, key=scores.get)
            actual = y[test, :, ti]
            chosen = predictions[winner][test, :, ti]
            baseline = predictions[best_baseline][test, :, ti]
            ci = bootstrap_skill(actual, chosen, baseline, meta.loc[test, "origin"])
            selected.append({"area": area, "target": target, "model": winner, "baseline": best_baseline,
                "validation_mae_mw": scores[winner], "test_mae_mw": losses(actual, chosen)["mae_mw"],
                "baseline_test_mae_mw": losses(actual, baseline)["mae_mw"],
                "test_skill_pct": 100 * (1 - np.abs(chosen - actual).mean() / np.abs(baseline - actual).mean()),
                "skill_ci_lower": ci[0], "skill_ci_upper": ci[1]})
            peak_cutoff = float(np.quantile(y[training, :, ti], 0.9))
            for split in ["validation", "calibration", "test"]:
                mask = am & meta.split.eq(split).to_numpy()
                actual = y[mask, :, ti]
                base = predictions[best_baseline][mask, :, ti]
                for name in point_models:
                    prediction = predictions[name][mask, :, ti]
                    if not np.isfinite(prediction).all():
                        raise ValueError(f"Incomplete predictions: {name}/{area}/{target}/{split}")
                    slices = {"all": np.ones_like(actual, dtype=bool)}
                    if target == "solar":
                        slices["daylight"] = data["daylight"][mask].astype(bool)
                    if target == "load":
                        slices["high_load"] = actual >= peak_cutoff
                    for label, cells in slices.items():
                        for bucket, start, stop in [("1–48", 0, 48), ("1–6", 0, 6), ("7–24", 6, 24), ("25–48", 24, 48)]:
                            selected_cells = cells[:, start:stop]
                            if not selected_cells.any():
                                continue
                            values = losses(actual[:, start:stop][selected_cells], prediction[:, start:stop][selected_cells])
                            base_mae = np.abs(base[:, start:stop][selected_cells] - actual[:, start:stop][selected_cells]).mean()
                            values["skill_pct"] = float(100 * (1 - values["mae_mw"] / base_mae)) if base_mae else np.nan
                            metric_rows.append({"model": name, "area": area, "target": target, "split": split,
                                                "slice": label, "horizon": bucket, "baseline": best_baseline, **values})
                    previous = data["history"][mask, -1:, ti]
                    actual_ramp = np.diff(np.concatenate([previous, actual], axis=1), axis=1)
                    pred_ramp = np.diff(np.concatenate([previous, prediction], axis=1), axis=1)
                    metric_rows.append({"model": name, "area": area, "target": target, "split": split,
                                        "slice": "hourly_ramp", "horizon": "1–48", **losses(actual_ramp, pred_ramp)})
                    if split == "test":
                        months = pd.to_datetime(meta.loc[mask, "origin"], utc=True).dt.strftime("%Y-%m").to_numpy()
                        for month in sorted(set(months)):
                            rows = months == month
                            monthly_rows.append({"model": name, "area": area, "target": target, "month": month,
                                                 **losses(actual[rows], prediction[rows])})
                for family, lower_key, upper_key in [
                    ("lightgbm_raw", "lightgbm_p10", "lightgbm_p90"),
                    ("lightgbm_calibrated", "lightgbm_cal_p10", "lightgbm_cal_p90"),
                    ("chronos2_raw", "chronos2_p10", "chronos2_p90")]:
                    median_key = "chronos2" if family.startswith("chronos") else "lightgbm"
                    for slice_name, cells in [("all", np.ones_like(actual, bool))] + (
                        [("daylight", data["daylight"][mask].astype(bool))] if target == "solar" else []):
                        lower, upper, median = predictions[lower_key][mask, :, ti], predictions[upper_key][mask, :, ti], predictions[median_key][mask, :, ti]
                        interval_rows.append({"model": family, "area": area, "target": target, "split": split, "slice": slice_name,
                            **interval_scores(actual[cells], lower[cells], median[cells], upper[cells])})
    metrics, intervals, selected = pd.DataFrame(metric_rows), pd.DataFrame(interval_rows), pd.DataFrame(selected)
    metrics.to_csv(HERE / "metrics.csv", index=False)
    intervals.to_csv(HERE / "interval_metrics.csv", index=False)
    selected.to_csv(HERE / "selected_models.csv", index=False)
    pd.DataFrame(monthly_rows).to_csv(HERE / "monthly_metrics.csv", index=False)
    timings = json.loads((CACHE / "training_timings.json").read_text())
    pd.DataFrame(timings).to_csv(HERE / "runtime.csv", index=False)
    for file in ["selection.json", "tuning.json", "chronos_timings.json", "calibration.json", "training_run.json"]:
        (HERE / file).write_text((CACHE / file).read_text())
    plot_results(metrics, selected, data, meta, predictions)
    write_report(metrics, intervals, selected, timings, meta)
    print(selected.to_string(index=False), flush=True)


def plot_results(metrics, selected, data, meta, predictions):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#faf9f5", "axes.facecolor": "#faf9f5"})
    colors = ["#8f9b9a", "#a9b9b6", "#617c78", "#ca9b49", "#007b78", "#b86850", "#5684b0", "#785c99", "#3c465c"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 14), constrained_layout=True)
    for ai, area in enumerate(AREAS):
        for ti, target in enumerate(TARGETS):
            frame = metrics.query("split == 'test' and slice == 'all' and horizon == '1–48'")
            frame = frame[(frame.area == area) & (frame.target == target)].set_index("model").loc[list(LABELS)]
            ax = axes[ti, ai]
            ax.barh(range(len(frame)), frame.mae_mw, color=colors)
            ax.set_yticks(range(len(frame)), [LABELS[n] for n in frame.index])
            ax.invert_yaxis()
            ax.set_title(f"{area} · {target.capitalize()}", loc="left", fontweight="bold")
            ax.set_xlabel("Mean absolute error (MW) · lower is better")
    fig.suptitle("Open Current · 48-hour forecast benchmark\nHeld-out June–September 2026; identical eligible origins", fontsize=18)
    fig.savefig(HERE / "accuracy.png", dpi=160)
    fig.savefig(HERE / "accuracy.svg")
    plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), constrained_layout=True)
    for ai, area in enumerate(AREAS):
        indices = np.flatnonzero((meta.area == area) & (meta.split == "test") & (meta.origin >= "2026-07-15"))
        index = indices[0]
        for ti, target in enumerate(TARGETS):
            row = selected[(selected.area == area) & (selected.target == target)].iloc[0]
            ax = axes[ti, ai]
            ax.plot(np.arange(1, H + 1), data["y"][index, :, ti], color="#263534", label="Observed", linewidth=2)
            ax.plot(np.arange(1, H + 1), predictions[row.model][index, :, ti], color="#007b78", label=LABELS[row.model], linewidth=2)
            ax.plot(np.arange(1, H + 1), predictions[row.baseline][index, :, ti], color="#b6925a", label=LABELS[row.baseline], linestyle="--")
            ax.set_title(f"{area} {target} · origin {meta.iloc[index].origin[:10]}", loc="left")
            ax.set_xlabel("Forecast hour")
            ax.set_ylabel("MW")
            ax.legend(fontsize=9)
    fig.suptitle("One forecast example · first eligible origin on/after 15 July\nModel selected on validation months; example date fixed independently of error", fontsize=15)
    fig.savefig(HERE / "forecast_example.png", dpi=160)
    fig.savefig(HERE / "forecast_example.svg")
    plt.close(fig)


def write_report(metrics, intervals, selected, timings, meta):
    audit = json.loads((HERE / "data_audit.json").read_text())
    chrono = json.loads((CACHE / "chronos_timings.json").read_text())
    training_run = json.loads((CACHE / "training_run.json").read_text())
    report = ["# Local forecasting results", "", "Generated from saved predictions; no deployment or repository push.", "",
              "See [research review](RESEARCH.md) and [frozen protocol](PROTOCOL.md).", "",
              "## Interpretation", "",
              "The strongest practical starting point is a small family of weather-informed models: boosted trees for renewables, a compact tree/neural blend where validation supports it, and a calendar/weather regression for load. The tested Chronos-2 configuration did not win model selection. Its inference was still fast enough that CPU cost alone is not a reason to reject foundation models.", "",
              "Weather had a large effect on wind/solar accuracy. For load, adding weather to the tree model improved test MAE by only about 4–6%; the larger gains over naive forecasts also reflect calendar and history modeling. Validation favored ridge for load, while trees/blends did better in the summer test. That seasonal ranking change argues for rolling evaluation before deployment, rather than choosing a new winner using this already-inspected test.", "",
              "Point forecasts are promising; uncertainty is less mature. Some calibrated tree intervals under-covered substantially, especially DK1 wind. Do not present these as reliable 80% bands yet. Calibration retained only 28 origins per area after completeness checks.", "",
              "## Validation-selected models on the untouched test", "",
              "The model and baseline were chosen using January–March validation. These are not hindsight test winners.", "",
              "| Area | Target | Selected model | Test MAE (MW) | Baseline MAE (MW) | MAE reduction | 95% week-block bootstrap interval |",
              "|---|---|---|---:|---:|---:|---:|"]
    for r in selected.itertuples():
        report.append(f"| {r.area} | {r.target} | {LABELS[r.model]} | {r.test_mae_mw:.1f} | {r.baseline_test_mae_mw:.1f} ({LABELS[r.baseline]}) | {r.test_skill_pct:.1f}% | {r.skill_ci_lower:.1f}–{r.skill_ci_upper:.1f}% |")
    report += ["", "![Model comparison](accuracy.png)", "", "## Did weather help the boosted trees?", "",
               "Each ablation has its own validation-selected setting, using identical origins and lag/calendar inputs.", "",
               "| Area | Target | Weather MAE | No-weather MAE | Reduction |", "|---|---|---:|---:|---:|"]
    full = metrics.query("split == 'test' and slice == 'all' and horizon == '1–48'")
    for area in AREAS:
        for target in TARGETS:
            f = full[(full.area == area) & (full.target == target)].set_index("model")
            w, n = f.loc["lightgbm", "mae_mw"], f.loc["lightgbm_no_weather", "mae_mw"]
            report.append(f"| {area} | {target} | {w:.1f} MW | {n:.1f} MW | {100 * (1 - w / n):.1f}% |")
    report += ["", "## Intervals need empirical checking", "",
               "These are nominal 80% prediction intervals. Coverage is measured on test data; dependent observations and drift invalidate any automatic exchangeable-data guarantee.", "",
               "| Area | Target | Interval method | Coverage | Mean width (MW) |", "|---|---|---|---:|---:|"]
    for row in intervals.query("split == 'test' and slice == 'all'").itertuples():
        report.append(f"| {row.area} | {row.target} | {row.model} | {100 * row.coverage_80:.1f}% | {row.mean_width_mw:.1f} |")
    tree = [r for r in timings if r["family"] == "lightgbm"]
    tree_seconds = sum(r["inference_seconds_one_area"] for r in tree)
    tree_bytes = sum(r["bytes"] for r in tree)
    tree_training = sum(r["training_seconds"] for r in tree)
    report += ["", "## Runtime and artifacts", "",
        f"- Six selected LightGBM point models: **{tree_seconds:.4f} s** summed warm inference for both areas and all targets, **{tree_bytes / 1e6:.2f} MB** serialized, **{tree_training:.1f} s** summed selected-model fitting. Tuning and interval models cost extra.",
        f"- Chronos-2: **{chrono['inference_seconds_two_areas']:.3f} s** for two area tasks/all three targets; **{chrono['load_seconds']:.2f} s** cached model load (excluding Python imports/download); **{chrono['parameter_count']:,} parameters**. Public pretrained weights, not trained locally.",
        f"- The successful full training/tuning/calibration run took **{training_run['total_wall_seconds']:.1f} s**, excluding backfill, the separate Chronos evaluation and an earlier local runtime failure. Neural training runs in isolated processes to avoid a macOS OpenMP conflict between LightGBM and PyTorch.",
        f"- Local system: `{platform.platform()}`, Python `{platform.python_version()}`, four CPU threads per model. These timings exclude network/data preparation and are **not measurements on a GitHub runner**.",
        "- All model settings, selected neural widths/epochs, quantile timings and model sizes are in [runtime.csv](runtime.csv), [selection.json](selection.json) and [tuning.json](tuning.json).",
        "", "## Data and exclusions", "",
        f"- {audit['raw_records']:,} five-minute records, {audit['hourly_records']:,} area-hours before sample exclusions, {audit['samples']:,} eligible area-origins.",
        f"- Counts: `{json.dumps(audit['sample_counts'], sort_keys=True)}`.",
        f"- Exclusions: `{json.dumps(audit['excluded'], sort_keys=True)}`.",
        "- Complete 168-hour histories and complete 48-hour labels are required. Even one missing five-minute interval can exclude several neighboring origins; these exclusions do not imply continuous multi-day outages. See [hourly coverage](data_coverage.csv). Accuracy is conditional on available data, not a claim of 100% service coverage.",
        "- SCADA target values can contain uncorrected errors; load is derived system load including losses. The observed historical download is not a publication-vintage archive.",
        "- One origin per day, 12 UTC completed-hour inputs with an assumed 15-minute availability buffer. Weather is that day's 00 UTC ECMWF run. This does not validate half-hourly refreshes or actual historical delivery times.",
        "- One year of training and summer/early-autumn testing do not establish winter or pan-European performance. The small hyperparameter budget and single neural seed do not establish that an architecture cannot do better.",
        "- This compares practical pipelines, not architecture alone: trained models use weather at three separate points per area; Chronos uses area means/spreads and jointly sees all three target histories. Different feature representations and zero-shot versus local training limit family-wide conclusions.",
        f"- Chronos checkpoint `{chrono['revision']}`; repository last modified `{chrono['last_modified']}`. The [weight audit](chronos_weights_audit.json) confirms its weights match the 30 October 2025 release byte for byte, preceding the 2026 evaluation. The complete pretraining corpus remains unaudited.",
        "", "## Inspect the forecasts", "", "![Forecast example](forecast_example.png)", "",
        "Full model/area/horizon/daylight/peak/ramp results: [metrics.csv](metrics.csv). Monthly drift: [monthly_metrics.csv](monthly_metrics.csv). Interval coverage, width and pinball: [interval_metrics.csv](interval_metrics.csv).",
        "", "Saved raw data, prediction arrays and model weights are in `.cache/forecasting/` (ignored by Git). The raw-data manifest has SHA-256 checksums. No live UI, workflow or published dataset was modified."]
    (HERE / "RESULTS.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
