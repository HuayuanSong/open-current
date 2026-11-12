"""Build matched chronological training samples from exact issued weather runs."""

import argparse
import gzip
import json
import time
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import requests

from forecast import data, weather_runs
from forecast.features import AREAS, CONFIG, TARGETS, calendar, weather_features
from .common import CACHE, HERE, ROOT, PROTOCOL, boundary, digest, write_json


def read(path):
    with gzip.open(path, "rt") as source:
        return json.load(source)


def daily_cache(state):
    cached, sources = {}, {}
    for path in sorted((ROOT / ".cache/forecasting/v2/weather").glob("*.json.gz")):
        record = read(path)
        points = record.get("data")
        if isinstance(points, list):
            cached[path.name[:10]] = points
            sources[str(path.relative_to(ROOT))] = digest(path)
    for key, entry in state["weather"].items():
        if entry.get("data"):
            cached[key] = entry["data"]
    return cached, sources


def historical_weather(daily):
    frames = {area: [] for area in AREAS}
    for key, points in sorted(daily.items()):
        times = pd.date_range(pd.Timestamp(key, tz="UTC") + pd.Timedelta(hours=13), periods=24, freq="h")
        for ai, area in enumerate(AREAS):
            wx = weather_features(points[ai * 3:ai * 3 + 3], times)[0]
            frames[area].append(pd.concat([wx, calendar(times)], axis=1))
    return {area: pd.concat(values).sort_index() for area, values in frames.items()}


def matching_request(record, run):
    query = parse_qs(urlparse(record.get("url", "")).query)
    params = weather_runs.parameters(run)
    return all(query.get(k) == [str(v)] for k, v in params.items() if k != "forecast_days")


def weather(run, times, *, cached_only=False, state=None, state_path=None):
    path = CACHE / "weather" / f"{run:%Y%m%dT%H}.json.gz"
    params = weather_runs.parameters(run)
    if path.exists():
        record = read(path)
        if record["parameters"] != params:
            raise ValueError("Cached weather request parameters differ")
        weather_runs.validate_points(record["data"], times)
        return record["data"], path
    legacy = []
    if run.hour == 0:
        legacy.append(ROOT / f".cache/forecasting/v2/weather/{run:%Y-%m-%d}.json.gz")
    if run.hour == 6:
        legacy.append(ROOT / f".cache/forecasting/v3/weather/fresh/{run:%Y-%m-%d}.json.gz")
    for source in legacy:
        if source.exists():
            record = read(source)
            if matching_request(record, run):
                try:
                    weather_runs.validate_points(record["data"], times)
                    return record["data"], source
                except (ValueError, KeyError, TypeError):
                    pass
    for source in sorted((ROOT / ".cache/weather-refresh/weather").glob(f"{run:%Y%m%dT%H}-*.json.gz")):
        record = read(source)
        if record.get("requestParameters") == params and record.get("run") == run.isoformat():
            try:
                weather_runs.validate_points(record["data"], times)
                return record["data"], source
            except (ValueError, KeyError, TypeError):
                pass
    # The production snapshot's daily keys come from the pinned 00 UTC ECMWF
    # collector. Preserve its file hash as provenance, without inventing a URL.
    if state is not None and run.hour == 0:
        record = state["weather"].get(str(run.date()), {})
        if record.get("data"):
            weather_runs.validate_points(record["data"], times)
            return record["data"], state_path
    if cached_only:
        raise ValueError(f"Archived run {run.isoformat()} not cached; acquisition stopped at provider rate limit")
    start = time.monotonic()
    try:
        print(f"Downloading archived run {run.isoformat()}", flush=True)
        for attempt in range(3):
            try:
                points = data.request(weather_runs.FORECAST_URL, params)
                break
            except (requests.Timeout, requests.ConnectionError):
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
        weather_runs.validate_points(points, times)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with gzip.open(temporary, "wt") as target:
            json.dump(dict(parameters=params, retrievedAt=data.now().isoformat(), data=points), target, allow_nan=False)
        temporary.replace(path)
        return points, path
    finally:
        time.sleep(max(0, 1.5 - (time.monotonic() - start)))


def build_history(frame, past, origin):
    values, observed, gaps = data.fill_history(frame, origin)
    times = pd.date_range(end=origin, periods=512, freq="h")
    history_weather = past.reindex(times).copy()
    cal = calendar(times)
    history_weather[cal.columns] = cal
    return dict(history=values[-168:], history512=values, observed512=observed,
                past_full512=history_weather.to_numpy()), gaps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, type=str)
    parser.add_argument("--cached-only", action="store_true", help="Make no API requests; record missing runs as exclusions")
    args = parser.parse_args()
    state = data.read_state(ROOT / args.state)
    CACHE.mkdir(parents=True, exist_ok=True)
    state_path = ROOT / args.state
    hourly_path = ROOT / ".cache/forecasting/v2/electricity_hourly.parquet"
    hourly = pd.concat([pd.read_parquet(hourly_path), pd.DataFrame(state["hourly"])], ignore_index=True)
    hourly["time"] = pd.to_datetime(hourly.time, utc=True)
    hourly = hourly.drop_duplicates(["area", "time"], keep="last").sort_values("time")
    regions = {area: hourly.loc[hourly.area.eq(area)].set_index("time")[TARGETS] for area in AREAS}
    daily, sources = daily_cache(state)
    sources[str(state_path.relative_to(ROOT))] = digest(state_path)
    sources[str(hourly_path.relative_to(ROOT))] = digest(hourly_path)
    past = historical_weather(daily)
    names = list(past[AREAS[0]].columns)
    write_json(CACHE / "feature-names.json", {"future": names})
    collected, samples, exclusions = {}, [], []
    days = pd.date_range(boundary("startDate"), boundary("endDate"), freq="D")
    total = len(days) * 2
    done = 0
    for day in days:
        parity = (day - boundary("parityAnchor")).days % 2
        for hour in PROTOCOL["originHoursByDayParity"][parity]:
            done += 1
            origin = day + pd.Timedelta(hours=hour)
            at = origin + pd.Timedelta(minutes=PROTOCOL["issueMinutesAfterOrigin"])
            times = pd.date_range(origin + pd.Timedelta(hours=1), periods=PROTOCOL["horizonHours"], freq="h")
            run = (at - pd.Timedelta(hours=PROTOCOL["assumedAvailabilityHours"])).floor("6h")
            control_run = (at - pd.Timedelta(hours=12, minutes=15)).floor("D")
            try:
                histories = [build_history(regions[area], past[area], origin) for area in AREAS]
                truth = [regions[area].reindex(times)[TARGETS].to_numpy() for area in AREAS]
                if not np.isfinite(truth).all():
                    raise ValueError("Missing outcome")
                if times[-1] >= boundary("finalFitLabelsBefore"):
                    raise ValueError("Final fitting label boundary")
                kwargs = dict(cached_only=args.cached_only, state=state, state_path=state_path)
                control_points, control_path = weather(control_run, times, **kwargs)
                fresh_points, fresh_path = weather(run, times, **kwargs)
                for source in [control_path, fresh_path]:
                    sources[str(source.relative_to(ROOT))] = digest(source)
                rows = []
                for ai, area in enumerate(AREAS):
                    arrays, gaps = histories[ai]
                    arrays = {**arrays, "y": truth[ai], "area_ids": ai}
                    for policy, points in [("future", control_points), ("fresh", fresh_points)]:
                        wx = weather_features(points[ai * 3:ai * 3 + 3], times)[0]
                        arrays[policy] = pd.concat([wx, calendar(times)], axis=1)[names].to_numpy()
                    rows.append((arrays, dict(origin=origin.isoformat(), area=area, area_id=ai,
                                last_label=times[-1].isoformat(), control_run=control_run.isoformat(),
                                latest_run=run.isoformat(), filled_input_hours=gaps)))
                for arrays, meta in rows:
                    for key, value in arrays.items():
                        collected.setdefault(key, []).append(np.asarray(value, dtype=np.float32))
                    samples.append(meta)
            except requests.HTTPError as error:
                if error.response.status_code not in (400, 404):
                    raise
                exclusions.append(dict(origin=origin.isoformat(), reason=str(error)))
            except (requests.Timeout, requests.ConnectionError) as error:
                reason = f"Weather unavailable after bounded transport retries: {error}"
                exclusions.append(dict(origin=origin.isoformat(), reason=reason))
                print(f"Excluded {origin.isoformat()}: {reason}", flush=True)
            except (ValueError, KeyError) as error:
                exclusions.append(dict(origin=origin.isoformat(), reason=str(error)))
            if done % 20 == 0 or done == total:
                print(f"Prepared {done}/{total} origins through {origin.isoformat()}; "
                      f"{len(samples) // 2} usable, {len(exclusions)} excluded", flush=True)
    arrays = {key: np.stack(values) for key, values in collected.items()}
    arrays["area_ids"] = arrays["area_ids"].astype(int)
    np.savez_compressed(CACHE / "dataset.npz", **arrays)
    pd.DataFrame(samples).to_csv(CACHE / "samples.csv", index=False)
    pd.DataFrame(exclusions).to_csv(HERE / "exclusions.csv", index=False)
    write_json(CACHE / "source-manifest.json", sources)
    write_json(HERE / "data-audit.json", dict(protocolHash=digest(HERE / "protocol.json"),
        sourceManifestHash=digest(CACHE / "source-manifest.json"), datasetHash=digest(CACHE / "dataset.npz"),
        attemptedOrigins=total, completeOrigins=len(samples) // 2, exclusions=len(exclusions),
        features=len(names), shapes={k: list(v.shape) for k, v in arrays.items()},
        acquisitionMode="cached-only" if args.cached_only else "online",
        historicalAvailability="assumed eight-hour lag, not verified per-run publication times"))
    print("Dataset ready", flush=True)


if __name__ == "__main__":
    main()
