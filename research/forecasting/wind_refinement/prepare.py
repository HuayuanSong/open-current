"""Reuse exact-run weather and five-minute electricity archives for ramp training."""

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from forecast.wind_short_term import measurements, features, require_observed_wind
from research.forecasting.weather_retrain.common import digest, write_json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CACHE = ROOT / ".cache/wind-refinement"


def raw_history():
    sources, frames = {}, []
    paths = sorted((ROOT / ".cache/forecasting/electricity").glob("20*.json.gz"))
    paths += sorted((ROOT / ".cache/forecasting/v2/electricity").glob("20*.json.gz"))
    paths += [ROOT / ".cache/wind-update/electricity.json.gz"]
    for path in paths:
        with gzip.open(path, "rt") as source:
            records = json.load(source)["data"]["records"]
        frames.append(pd.DataFrame(records)[["Minutes5UTC", "PriceArea", "OnshoreWindPower", "OffshoreWindPower"]])
        sources[str(path.relative_to(ROOT))] = digest(path)
    raw = pd.concat(frames).drop_duplicates(["Minutes5UTC", "PriceArea"], keep="last")
    raw.index = pd.to_datetime(raw.Minutes5UTC, utc=True)
    return raw.sort_index(), sources


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    raw, sources = raw_history()
    path = ROOT / ".cache/weather-retrain/dataset.npz"
    with np.load(path) as saved:
        arrays = {k: saved[k] for k in ("history", "fresh", "y", "observed512")}
    samples = pd.read_csv(ROOT / ".cache/weather-retrain/samples.csv")
    names = json.loads((ROOT / ".cache/weather-retrain/feature-names.json").read_text())["future"]
    sources[str(path.relative_to(ROOT))] = digest(path)
    matrices, truth, levels, metadata, excluded = [], [], [], [], []
    for start in range(0, len(samples), 2):
        origin = pd.Timestamp(samples.iloc[start].origin)
        at = origin + pd.Timedelta(minutes=30)
        times = pd.date_range(origin + pd.Timedelta(hours=1), periods=6, freq="h")
        try:
            require_observed_wind(arrays["observed512"][start:start+2])
            recent = measurements(raw.loc[at-pd.Timedelta(hours=4):at].to_dict("records"), at)
            x, columns = features(arrays["history"][start:start+2], arrays["fresh"][start:start+2], names, recent, times)
            matrices.extend(x)
            truth.extend(arrays["y"][start:start+2, :6, 0])
            levels.extend([recent[a]["levelMW"] for a in ("DK1", "DK2")])
            metadata.extend(samples.iloc[start:start+2].to_dict("records"))
        except (ValueError, KeyError) as error:
            excluded.append(dict(origin=origin.isoformat(), reason=str(error)))
        if start % 200 == 0:
            print(f"Prepared {start//2}/{len(samples)//2} archived origins", flush=True)
    np.savez_compressed(CACHE / "dataset.npz", x=matrices, y=truth, level=levels)
    pd.DataFrame(metadata).to_csv(CACHE / "samples.csv", index=False)
    write_json(CACHE / "feature-names.json", columns)
    write_json(HERE / "data-audit.json", dict(sourceHashes=sources, origins=len(metadata)//2,
        excluded=excluded, features=len(columns), datasetHash=digest(CACHE / "dataset.npz")))
    print(f"Prepared {len(metadata)//2} origins; excluded {len(excluded)}", flush=True)


if __name__ == "__main__":
    main()
