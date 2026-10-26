"""Restore rolling history from the previous Pages deployment, without Git churn."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import urllib.error
import urllib.request

from config import COUNTRIES
from fetch import write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    base = os.environ.get('SNAPSHOT_URL', 'https://huayuansong.github.io/grid-in-motion/').rstrip('/') + '/data/'
    try:
        with urllib.request.urlopen(base + 'index.json', timeout=30) as response:
            previous = json.load(response)
        local = json.loads((ROOT / 'dist/data/index.json').read_text())
        if previous.get('schema') != 1:
            raise SystemExit('Unrecognized previous snapshot; refusing to overwrite it.')
        if previous['generatedAt'] <= local['generatedAt']:
            print('Using the bundled snapshot (at least as recent as the deployed snapshot).')
            return
    except urllib.error.HTTPError as error:
        if error.code == 404:
            print('First deployment: using bundled history.')
            return
        raise SystemExit(f'Previous snapshot is unavailable; leaving the deployment intact ({error}).')
    except (urllib.error.URLError, ValueError, KeyError, TimeoutError) as error:
        raise SystemExit(f'Could not verify the previous snapshot; leaving the deployment intact ({error}).')

    def restore(path):
        try:
            with urllib.request.urlopen(base + path, timeout=30) as response:
                payload = json.load(response)
            return (path, payload)
        except (urllib.error.URLError, ValueError, TimeoutError) as error:
            print(f'Retaining bundled {path}: {error}')
            return None

    paths = ['denmark.json'] + [f'countries/{code}.json' for code in COUNTRIES]
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(restore, paths))
    # The normalizer rebuilds the index from actual retained histories; do not
    # stamp the earlier index with a new collection time during a code-only deploy.
    if not all(results):
        raise SystemExit('Incomplete history restore; leaving the previous deployment intact.')
    for path, payload in results:
        write_json(ROOT / 'dist/data' / path, payload)
    write_json(ROOT / 'dist/data/index.json', previous)
    print(f'Restored {len(results)} published data files.')


if __name__ == '__main__':
    main()

