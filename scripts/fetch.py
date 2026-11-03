"""Download public feeds. One paced queue per independently limited endpoint.

Raw responses stay outside the published site. A failure leaves the last valid
response intact. Normalization separately retains previous published records.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import json
import math
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request
from zoneinfo import ZoneInfo

from config import COUNTRIES, PRICE_ZONES, SOURCE

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
RETRYABLE_HTTP_STATUSES = (429, 500, 502, 503, 504)


def retry_after_seconds(header, fallback):
    """Read server backoff without turning invalid values into unbounded waits."""
    try:
        pause = float(header)
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(header)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            pause = max(0, (retry_at - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return fallback
    return pause if math.isfinite(pause) and pause >= 0 else fallback


def fetch_json(url, attempts=4, min_interval=0, before_request=None):
    for attempt in range(attempts):
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "OpenCurrent/1.0 (public educational electricity explorer)",
                "Accept": "application/json",
            },
        )
        try:
            if before_request is not None:
                before_request()
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code not in RETRYABLE_HTTP_STATUSES or attempt == attempts - 1:
                raise
            pause = retry_after_seconds(
                error.headers.get("Retry-After", ""), fallback=35 * (attempt + 1)
            )
            time.sleep(max(1, min_interval, pause) + 1)
        except (TimeoutError, urllib.error.URLError):
            if attempt == attempts - 1:
                raise
            time.sleep(max(min_interval, 35 * (attempt + 1)))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False))
    temporary.replace(path)


def queue(endpoint, keys, query_key, days, interval):
    now = datetime.now(timezone.utc)
    start = (now - timedelta(days=days)).strftime("%Y-%m-%dT%H:%MZ")
    end = (now + timedelta(days=1 if endpoint == "price" else 0)).strftime(
        "%Y-%m-%dT%H:%MZ"
    )
    failures = []
    next_request_at = 0

    def wait_for_slot():
        nonlocal next_request_at
        remaining = next_request_at - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        next_request_at = time.monotonic() + interval

    for key in keys:
        try:
            params = urllib.parse.urlencode(
                {query_key: key, "start": start, "end": end}
            )
            data = fetch_json(
                SOURCE + endpoint + "?" + params, before_request=wait_for_slot
            )
            if "CC BY 4.0" not in data.get("license", ""):
                raise ValueError(
                    "No explicit public CC BY 4.0 license; refusing to publish"
                )
            if not isinstance(data.get("data"), list):
                raise ValueError("Unexpected response schema")
            data["_retrieved_at"] = datetime.now(timezone.utc).isoformat()
            write_json(CACHE / endpoint / (key + ".json"), data)
            print(f'{endpoint}/{key}: {len(data["data"])} rows', flush=True)
        except Exception as error:
            if (
                isinstance(error, urllib.error.HTTPError)
                and error.code in RETRYABLE_HTTP_STATUSES
            ):
                pause = retry_after_seconds(
                    error.headers.get("Retry-After", ""), fallback=0
                )
                if pause > 0:
                    # An exhausted key must not let the next key bypass server backoff.
                    next_request_at = max(next_request_at, time.monotonic() + pause + 1)
            failures.append({"endpoint": endpoint, "key": key, "error": str(error)})
            print(f"{endpoint}/{key}: keeping previous data ({error})", flush=True)
    return failures


def energinet(days):
    failures = []
    for dataset in ["ElectricityProdex5MinRealtime", "CO2Emis"]:
        try:
            query = {"start": "now", "startOffset": f"-P{min(days, 7)}D", "limit": 5000}
            if dataset == "ElectricityProdex5MinRealtime":
                # Share the forecast's three-day revision window with collection,
                # avoiding a second request to the same five-minute dataset.
                start = (
                    datetime.now(timezone.utc) - timedelta(days=max(3, min(days, 7)))
                ).astimezone(ZoneInfo("Europe/Copenhagen")).date()
                query = {"start": str(start), "end": "now", "limit": 5000}
            params = urllib.parse.urlencode(query)
            data = fetch_json(
                "https://api.energidataservice.dk/dataset/" + dataset + "?" + params
            )
            if not isinstance(data.get("records"), list) or not data["records"]:
                raise ValueError("No records returned")
            if data.get("total") != len(data["records"]):
                raise ValueError("Incomplete electricity response")
            data["_retrieved_at"] = datetime.now(timezone.utc).isoformat()
            if dataset == "ElectricityProdex5MinRealtime":
                data["_start"] = query["start"]
            write_json(CACHE / "energinet" / (dataset + ".json"), data)
            print(f'Energinet/{dataset}: {len(data["records"])} rows', flush=True)
        except Exception as error:
            failures.append({"endpoint": dataset, "key": "dk", "error": str(error)})
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=1)
    parser.add_argument(
        "--interval",
        type=float,
        default=31,
        help="Seconds per request per endpoint (minimum 31)",
    )
    parser.add_argument("--countries", nargs="*")
    args = parser.parse_args()
    countries = args.countries or list(COUNTRIES)
    if any(code not in COUNTRIES for code in countries):
        parser.error("Unsupported country")
    prices = [
        code for code, (country, _) in PRICE_ZONES.items() if country in countries
    ]
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [
            pool.submit(
                queue,
                "public_power",
                countries,
                "country",
                args.days,
                max(31, args.interval),
            ),
            pool.submit(
                queue, "cbpf", countries, "country", args.days, max(31, args.interval)
            ),
            pool.submit(
                queue, "price", prices, "bzn", args.days, max(31, args.interval)
            ),
        ]
        failures = [failure for job in jobs for failure in job.result()]
    # Collect the faster Danish feeds last, so the European queue doesn't add
    # its full run time to the age of Denmark's five-minute readings.
    failures.extend(energinet(args.days))
    write_json(
        CACHE / "fetch-status.json",
        {"finished_at": datetime.now(timezone.utc).isoformat(), "failures": failures},
    )
    print(f"Collection finished: {len(failures)} unavailable feeds.", flush=True)


if __name__ == "__main__":
    main()
