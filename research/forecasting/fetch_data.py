"""Resumable, unauthenticated research backfill. Never changes published data."""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / ".cache" / "forecasting"
CONFIG = json.loads((HERE / "config.json").read_text())


def fetch(url, params, path):
    if path.exists():
        return False
    for attempt in range(5):
        try:
            response = requests.get(url, params=params, timeout=120)
            if response.status_code == 429:
                time.sleep(min(60, float(response.headers.get("Retry-After", 60))))
                continue
            if response.status_code == 400 and "The requested model run is not available" in response.text:
                payload = {"unavailable": True, "reason": response.json()["reason"]}
                path.parent.mkdir(parents=True, exist_ok=True)
                with gzip.open(path, "wt") as output:
                    json.dump({"retrieved_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                               "url": response.url, "data": payload}, output)
                print(f"Archive gap recorded: {path.name}", flush=True)
                return True
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict) and payload.get("error"):
                raise ValueError(payload)
            # Metadata records retrieval time and exact request; this is not a
            # claim that historical publication timestamps are available.
            wrapped = {"retrieved_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                       "url": response.url, "data": payload}
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            with gzip.open(temporary, "wt") as output:
                json.dump(wrapped, output, separators=(",", ":"), allow_nan=False)
            temporary.replace(path)
            return True
        except (requests.RequestException, ValueError) as error:
            print(f"Retry {attempt + 1}: {path.name}: {str(error)[:180]}", flush=True)
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Rate limit did not clear: {path}")


def electricity():
    dataset = "ElectricityProdex5MinRealtime"
    fetch(f"https://api.energidataservice.dk/meta/dataset/{dataset}", {},
          CACHE / "electricity" / "metadata.json.gz")
    # Extra December provides history for the first training origin. October
    # provides the final horizon, although split boundary purging excludes it.
    start = dt.date(2024, 12, 1)
    end = dt.date(2026, 10, 3)
    while start < end:
        following = (start.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
        stop = min(following, end)
        path = CACHE / "electricity" / f"{start}.json.gz"
        fetch(f"https://api.energidataservice.dk/dataset/{dataset}",
              {"start": str(start), "end": str(stop), "limit": 30000,
               "sort": "Minutes5UTC asc"}, path)
        with gzip.open(path, "rt") as source:
            data = json.load(source)["data"]
        if data["total"] != len(data["records"]):
            raise ValueError(f"Truncated electricity response: {path}")
        print(f"Electricity {start}: {len(data['records'])} records", flush=True)
        start = following


def weather():
    start = dt.date.fromisoformat(CONFIG["run_start"])
    end = dt.date.fromisoformat(CONFIG["run_end"])
    locations = CONFIG["locations"]
    total = (end - start).days + 1
    # Six locations, six variables, three days: <=3,828 location-calls for
    # this backfill. 1.5s spacing also stays below the 600/min free allowance.
    # Resume uses cached responses and does not repeat successful requests.
    for index in range(total):
        day = start + dt.timedelta(days=index)
        path = CACHE / "weather" / f"{day}.json.gz"
        before = time.monotonic()
        downloaded = fetch("https://single-runs-api.open-meteo.com/v1/forecast", {
            "latitude": ",".join(str(p["latitude"]) for p in locations),
            "longitude": ",".join(str(p["longitude"]) for p in locations),
            "hourly": ",".join(CONFIG["variables"]),
            "models": CONFIG["weather_model"], "run": f"{day}T00:00",
            "forecast_days": 3, "timezone": "UTC", "wind_speed_unit": "ms",
            "cell_selection": "nearest",
        }, path)
        if index % 20 == 0 or index == total - 1:
            print(f"Weather {index + 1}/{total}: {day}", flush=True)
        if downloaded:
            time.sleep(max(0, 1.5 - (time.monotonic() - before)))


def manifest():
    files = []
    for directory in ["weather", "electricity"]:
        for path in sorted((CACHE / directory).glob("*.json.gz")):
            files.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (CACHE / "manifest.json").write_text(json.dumps({"config": CONFIG, "files": files}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", choices=["weather", "electricity", "manifest", "all"])
    args = parser.parse_args()
    if args.source in ["electricity", "all"]:
        electricity()
    if args.source in ["weather", "all"]:
        weather()
    manifest()
