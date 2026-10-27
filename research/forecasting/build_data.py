"""Build forecast-origin samples. Never substitute observed future weather."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

from fetch_data import CACHE, CONFIG, HERE

TARGETS = ["wind", "solar", "load"]
AREAS = ["DK1", "DK2"]
H = CONFIG["horizon"]
L = CONFIG["lookback"]
HOLIDAYS = holidays.DK(years=range(2024, 2028))


def read_gzip(path):
    with gzip.open(path, "rt") as file:
        return json.load(file)["data"]


def normalize_electricity(raw):
    raw = raw.copy()
    raw["time"] = pd.to_datetime(raw.Minutes5UTC, utc=True)
    raw = raw.drop_duplicates(["time", "PriceArea"])
    generation = ["ProductionLt100MW", "ProductionGe100MW", "OffshoreWindPower",
                  "OnshoreWindPower", "SolarPower"]
    output = []
    for area in AREAS:
        frame = raw.loc[raw.PriceArea.eq(area)].set_index("time").sort_index()
        wind = frame[["OffshoreWindPower", "OnshoreWindPower"]].sum(axis=1, min_count=2)
        gen = frame[generation].sum(axis=1, min_count=len(generation))
        borders = ["ExchangeGermany", "ExchangeSweden", "ExchangeGreatBelt"]
        if area == "DK1":
            borders += ["ExchangeNorway", "ExchangeNetherlands", "ExchangeGreatBritain"]
        load = gen + frame[borders].sum(axis=1, min_count=len(borders))
        values = pd.DataFrame({"wind": wind, "solar": frame.SolarPower, "load": load})
        values[["wind", "solar"]] = values[["wind", "solar"]].where(values[["wind", "solar"]] >= 0)
        values["load"] = values["load"].where(values["load"] > 0)
        # Provider timestamps use CET ambiguously around the DST transition.
        # Mask a conservative UTC envelope around both documented incidents.
        bad = ((values.index >= "2026-03-29T00:00Z") & (values.index < "2026-03-29T03:00Z"))
        bad |= ((values.index >= "2026-03-30T07:00Z") & (values.index < "2026-03-30T10:00Z"))
        values.loc[bad, :] = np.nan
        hourly = values.resample("1h", closed="left", label="right").mean()
        counts = values.resample("1h", closed="left", label="right").count()
        hourly = hourly.where(counts.eq(12))
        hourly["area"] = area
        output.append(hourly.reset_index())
    return pd.concat(output, ignore_index=True)


def calendar(index):
    local = index.tz_convert("Europe/Copenhagen")
    hour = local.hour.to_numpy()
    day = local.day_of_week.to_numpy()
    year_angle = 2 * np.pi * (local.day_of_year.to_numpy() - 1) / 365.25
    features = {
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "year_sin": np.sin(year_angle), "year_cos": np.cos(year_angle),
        "weekend": (day >= 5).astype(float),
        "holiday": np.array([d in HOLIDAYS for d in local.date], float),
        "trend_years": np.asarray((index - pd.Timestamp("2025-01-01", tz="UTC")).total_seconds()) / (365.25 * 86400),
    }
    for weekday in range(7):
        features[f"weekday_{weekday}"] = (day == weekday).astype(float)
    for h in range(24):
        features[f"hour_{h}"] = (hour == h).astype(float)
        features[f"weekend_hour_{h}"] = ((hour == h) & (day >= 5)).astype(float)
    return pd.DataFrame(features, index=index)


def weather_features(points, valid_times):
    features = {}
    winds, radiation, temperature, terrestrial = [], [], [], []
    for i, point in enumerate(points):
        frame = pd.DataFrame(point["hourly"])
        frame.index = pd.to_datetime(frame.pop("time"), utc=True)
        frame = frame.reindex(valid_times).astype(float)
        for name in CONFIG["variables"]:
            if name != "wind_direction_100m":
                features[f"wx_{i}_{name}"] = frame[name].to_numpy()
        angle = np.radians(frame.wind_direction_100m.to_numpy())
        features[f"wx_{i}_wind_sin"] = np.sin(angle)
        features[f"wx_{i}_wind_cos"] = np.cos(angle)
        wind = frame.wind_speed_100m.to_numpy()
        temp = frame.temperature_2m.to_numpy()
        ghi = frame.shortwave_radiation.to_numpy()
        # Generic conversion features, not a calibrated turbine/PV plant model.
        features[f"wx_{i}_power_curve"] = np.where(
            (wind >= 3) & (wind < 25), np.clip((wind**3 - 3**3) / (12**3 - 3**3), 0, 1), 0)
        features[f"wx_{i}_heating"] = np.maximum(18 - temp, 0)
        features[f"wx_{i}_cooling"] = np.maximum(temp - 22, 0)
        features[f"wx_{i}_pv_temperature"] = ghi * (1 - 0.004 * (temp - 25))
        winds.append(wind)
        radiation.append(ghi)
        temperature.append(temp)
        terrestrial.append(frame.terrestrial_radiation.to_numpy())
    features["wx_wind_spread"] = np.std(winds, axis=0)
    features["wx_radiation_spread"] = np.std(radiation, axis=0)
    compact = {
        "wind": np.mean(winds, axis=0), "radiation": np.mean(radiation, axis=0),
        "temperature": np.mean(temperature, axis=0),
        "wind_spread": np.std(winds, axis=0), "radiation_spread": np.std(radiation, axis=0),
    }
    return pd.DataFrame(features, index=valid_times), compact, np.mean(terrestrial, axis=0) > 5


def split_for(origin):
    last_label = origin + pd.Timedelta(hours=H)
    for split, end in [("train", "train_end"), ("validation", "validation_end"),
                       ("calibration", "calibration_end"), ("test", "test_end")]:
        boundary = pd.Timestamp(CONFIG[end], tz="UTC")
        if origin < boundary:
            return split if last_label < boundary else "purged"
    return "outside"


def tabular_features(history, future, target):
    """One row per origin and lead; every target feature is from the history."""
    own = history[:, :, target]
    count, horizon, _ = future.shape
    extras = {"lead": np.broadcast_to(np.arange(1, horizon + 1), (count, horizon))}
    for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168]:
        extras[f"lag_{lag}"] = np.repeat(own[:, -lag, None], horizon, axis=1)
    for window in [6, 24, 168]:
        for name, fn in [("mean", np.mean), ("std", np.std), ("max", np.max)]:
            extras[f"{name}_{window}"] = np.repeat(fn(own[:, -window:], axis=1)[:, None], horizon, axis=1)
    extras["day_baseline"] = own[:, -24:][:, np.arange(horizon) % 24]
    extras["week_baseline"] = own[:, :horizon]
    extras["week_hour_mean"] = own.reshape(count, 7, 24).mean(axis=1)[:, np.arange(horizon) % 24]
    extra_array = np.stack(list(extras.values()), axis=-1)
    return np.concatenate([future, extra_array], axis=-1).reshape(count * horizon, -1), list(extras)


def main():
    raw = []
    for path in sorted((CACHE / "electricity").glob("20*.json.gz")):
        raw.extend(read_gzip(path)["records"])
    hourly = normalize_electricity(pd.DataFrame(raw))
    hourly.to_parquet(CACHE / "electricity_hourly.parquet", index=False)
    coverage = hourly.assign(month=hourly.time.dt.strftime("%Y-%m"))
    coverage_rows = [{"area": area, "month": month, "hours": len(group),
                      "incomplete_hours": int(group[TARGETS].isna().any(axis=1).sum())}
                     for (area, month), group in coverage.groupby(["area", "month"])]
    pd.DataFrame(coverage_rows).to_csv(HERE / "data_coverage.csv", index=False)
    observations = {a: hourly[hourly.area.eq(a)].set_index("time")[TARGETS] for a in AREAS}
    all_dates = pd.date_range(CONFIG["run_start"], CONFIG["run_end"], freq="1D", tz="UTC")
    # An archive of past covariates from forecasts that preceded their valid times.
    # Keep all weather days, including those excluded for missing electricity.
    wx, compact_history = {}, {area: [] for area in AREAS}
    for day in all_dates:
        path = CACHE / "weather" / f"{day.date()}.json.gz"
        if not path.exists():
            raise FileNotFoundError(f"Backfill incomplete: {path}")
        points = read_gzip(path)
        if isinstance(points, dict) and points.get("unavailable"):
            continue
        if len(points) != len(CONFIG["locations"]):
            raise ValueError(f"Wrong weather location count: {path}")
        origin = day + pd.Timedelta(hours=CONFIG["origin_hour_utc"])
        valid = pd.date_range(origin + pd.Timedelta(hours=1), periods=H, freq="1h")
        for ai, area in enumerate(AREAS):
            weather, compact, daylight = weather_features(points[ai * 3:ai * 3 + 3], valid)
            cal = calendar(valid)
            for key in ["hour_sin", "hour_cos", "weekend", "holiday", "year_sin", "year_cos"]:
                compact[key] = cal[key].to_numpy()
            cf = pd.DataFrame(compact, index=valid)
            compact_history[area].append(cf.iloc[:24])
            wx[(origin, area)] = (pd.concat([weather, cal], axis=1), cf, daylight)
    compact_history = {a: pd.concat(frames).sort_index() for a, frames in compact_history.items()}
    arrays = {key: [] for key in ["history", "future", "y", "past_cov", "future_cov", "daylight"]}
    metadata, exclusions = [], []
    for day in all_dates:
        origin = day + pd.Timedelta(hours=CONFIG["origin_hour_utc"])
        split = split_for(origin)
        for ai, area in enumerate(AREAS):
            if split in ["purged", "outside"]:
                exclusions.append({"origin": str(origin), "area": area, "reason": split})
                continue
            if (origin, area) not in wx:
                exclusions.append({"origin": str(origin), "area": area, "reason": "unavailable_weather_run"})
                continue
            history_times = pd.date_range(end=origin, periods=L, freq="1h")
            future_times = pd.date_range(origin + pd.Timedelta(hours=1), periods=H, freq="1h")
            history = observations[area].reindex(history_times).to_numpy()
            y = observations[area].reindex(future_times).to_numpy()
            future, fcov, daylight = wx[(origin, area)]
            if not np.isfinite(history).all() or not np.isfinite(y).all():
                exclusions.append({"origin": str(origin), "area": area, "reason": "missing_or_invalid_electricity"})
                continue
            if not np.isfinite(future.to_numpy()).all():
                exclusions.append({"origin": str(origin), "area": area, "reason": "missing_weather"})
                continue
            arrays["history"].append(history)
            arrays["y"].append(y)
            arrays["future"].append(future.to_numpy())
            arrays["daylight"].append(daylight)
            arrays["future_cov"].append(fcov.to_numpy())
            past = compact_history[area].reindex(history_times)
            # Calendar is known even before the first archived weather day.
            cal = calendar(history_times)
            for name in cal.columns.intersection(past.columns):
                past[name] = cal[name]
            arrays["past_cov"].append(past.to_numpy())
            metadata.append({"origin": origin.isoformat(), "area": area, "area_id": ai, "split": split,
                             "weather_run": day.isoformat(), "latest_observation_end": origin.isoformat(),
                             "issue_time": (origin + pd.Timedelta(minutes=15)).isoformat(),
                             "last_label": future_times[-1].isoformat()})
    np.savez_compressed(CACHE / "dataset.npz", **{k: np.array(v, dtype=np.float32) for k, v in arrays.items()})
    pd.DataFrame(metadata).to_csv(CACHE / "samples.csv", index=False)
    pd.DataFrame(exclusions).to_csv(CACHE / "exclusions.csv", index=False)
    (CACHE / "feature_names.json").write_text(json.dumps({"future": list(future.columns), "compact": list(fcov.columns)}, indent=2))
    summary = {"raw_records": len(raw), "hourly_records": len(hourly), "samples": len(metadata),
               "sample_counts": pd.DataFrame(metadata).groupby(["split", "area"]).size().to_dict(),
               "excluded": pd.DataFrame(exclusions).groupby("reason").size().to_dict()}
    summary["sample_counts"] = {"/".join(k): int(v) for k, v in summary["sample_counts"].items()}
    summary["excluded"] = {k: int(v) for k, v in summary["excluded"].items()}
    (HERE / "data_audit.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
