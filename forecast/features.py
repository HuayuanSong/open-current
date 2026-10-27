"""Inference features shared verbatim with the evaluated research pipeline."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import holidays
CONFIG = json.loads((Path(__file__).parent / "config.json").read_text())
AREAS = ["DK1", "DK2"]
TARGETS = ["wind", "solar", "load"]
H = 48
HOLIDAYS = holidays.DK(years=range(2024, 2032))

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

def features(a, names, ti, improved=True):
    x, extras = tabular_features(a['history'], a['future'], ti)
    if not improved:
        return x
    added = []
    for target in range(3):
        for hours in [24, 168]:
            added += [np.repeat(a['history'][:, -hours:, target].mean(1)[:,None], H, 1),
                      np.repeat(a['history'][:, -hours:, target].std(1)[:,None], H, 1)]
    # A rolling conversion from prior issued weather to observed output. Each
    # slope only uses the history available at this origin, never future labels.
    proxy = 'power_curve' if ti == 0 else 'pv_temperature'
    if ti in [0, 1]:
        columns = [i for i,n in enumerate(names['future']) if n.endswith(proxy)]
        oldwx = np.nanmean(a['past_full512'][:, -168:, columns], axis=-1)
        futurewx = a['future'][:, :, columns].mean(-1)
        for hours in [24, 168]:
            wx = oldwx[:, -hours:]
            y = a['history'][:, -hours:, ti]
            usable = np.isfinite(wx) & (wx > (0.05 if ti == 0 else 50))
            numerator = np.where(usable, y*wx, 0).sum(1)
            denominator = np.where(usable, wx*wx, 0).sum(1)
            slope = numerator / np.maximum(denominator, 1e-6)
            added.append(futurewx*slope[:,None])
            added.append(np.repeat(slope[:,None], H, 1))
    return np.concatenate([x, np.stack(added, -1).reshape(len(x), -1)], axis=1)

def causal_fill(obs):
    """Fill gaps from earlier timestamps only; retain a separate observed mask."""
    filled = obs.copy()
    for lag in [168, 336, 504, 24]:
        filled = filled.fillna(obs.shift(lag))
    return filled.ffill()
