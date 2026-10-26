"""Download public feeds. One paced queue per independently limited endpoint.

Raw responses stay outside the published site. A failure leaves the last valid
response intact. Normalization separately retains previous published records.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import json
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request

from config import COUNTRIES, PRICE_ZONES, SOURCE

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / '.cache'


def fetch_json(url, attempts=4):
    for attempt in range(attempts):
        request = urllib.request.Request(url, headers={
            'User-Agent': 'OpenCurrent/1.0 (public educational electricity explorer)',
            'Accept': 'application/json',
        })
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == attempts - 1:
                raise
            retry = error.headers.get('Retry-After', '')
            try:
                pause = float(retry)
            except ValueError:
                try:
                    pause = (parsedate_to_datetime(retry) - datetime.now(timezone.utc)).total_seconds()
                except (TypeError, ValueError):
                    pause = 35 * (attempt + 1)
            time.sleep(max(1, pause) + 1)
        except (TimeoutError, urllib.error.URLError):
            if attempt == attempts - 1:
                raise
            time.sleep(35 * (attempt + 1))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, separators=(',', ':'), allow_nan=False))
    temporary.replace(path)


def queue(endpoint, keys, query_key, days, interval):
    now = datetime.now(timezone.utc)
    start = (now - timedelta(days=days)).strftime('%Y-%m-%dT%H:%MZ')
    end = (now + timedelta(days=1 if endpoint == 'price' else 0)).strftime('%Y-%m-%dT%H:%MZ')
    failures = []
    for index, key in enumerate(keys):
        started = time.monotonic()
        try:
            params = urllib.parse.urlencode({query_key: key, 'start': start, 'end': end})
            data = fetch_json(SOURCE + endpoint + '?' + params)
            if 'CC BY 4.0' not in data.get('license', ''):
                raise ValueError('No explicit public CC BY 4.0 license; refusing to publish')
            if not isinstance(data.get('data'), list):
                raise ValueError('Unexpected response schema')
            data['_retrieved_at'] = datetime.now(timezone.utc).isoformat()
            write_json(CACHE / endpoint / (key + '.json'), data)
            print(f'{endpoint}/{key}: {len(data["data"])} rows', flush=True)
        except Exception as error:
            failures.append({'endpoint': endpoint, 'key': key, 'error': str(error)})
            print(f'{endpoint}/{key}: keeping previous data ({error})', flush=True)
        if index < len(keys) - 1:
            time.sleep(max(0, interval - (time.monotonic() - started)))
    return failures


def energinet(days):
    failures = []
    for dataset in ['ElectricityProdex5MinRealtime', 'CO2Emis']:
        try:
            # Use supported start + startOffset parameters, in Danish local time.
            params = urllib.parse.urlencode({'start': 'now', 'startOffset': f'-P{min(days, 7)}D', 'limit': 5000})
            data = fetch_json('https://api.energidataservice.dk/dataset/' + dataset + '?' + params)
            if not isinstance(data.get('records'), list) or not data['records']:
                raise ValueError('No records returned')
            data['_retrieved_at'] = datetime.now(timezone.utc).isoformat()
            write_json(CACHE / 'energinet' / (dataset + '.json'), data)
            print(f'Energinet/{dataset}: {len(data["records"])} rows', flush=True)
        except Exception as error:
            failures.append({'endpoint': dataset, 'key': 'dk', 'error': str(error)})
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=1)
    parser.add_argument('--interval', type=float, default=31, help='Seconds per request per endpoint (minimum 31)')
    parser.add_argument('--countries', nargs='*')
    args = parser.parse_args()
    countries = args.countries or list(COUNTRIES)
    if any(code not in COUNTRIES for code in countries):
        parser.error('Unsupported country')
    prices = [code for code, (country, _) in PRICE_ZONES.items() if country in countries]
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [
            pool.submit(queue, 'public_power', countries, 'country', args.days, max(31, args.interval)),
            pool.submit(queue, 'cbpf', countries, 'country', args.days, max(31, args.interval)),
            pool.submit(queue, 'price', prices, 'bzn', args.days, max(31, args.interval)),
        ]
        failures = [failure for job in jobs for failure in job.result()]
    # Collect the faster Danish feeds last, so the European queue doesn't add
    # its full run time to the age of Denmark's five-minute readings.
    failures.extend(energinet(args.days))
    write_json(CACHE / 'fetch-status.json', {'finished_at': datetime.now(timezone.utc).isoformat(), 'failures': failures})
    print(f'Collection finished: {len(failures)} unavailable feeds.', flush=True)


if __name__ == '__main__':
    main()
