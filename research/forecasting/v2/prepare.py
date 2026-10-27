"""Extend immutable v1 data with a separately cached October holdout."""
import common
from common import CACHE, OLD, HERE
import datetime as dt
import hashlib
import json
import time
import numpy as np
import pandas as pd
import fetch_data as fetch
import build_data as build

def causal_fill(obs):
    """Fill gaps from earlier timestamps only; retain a separate observed mask."""
    filled = obs.copy()
    for lag in [168, 336, 504, 24]:
        filled = filled.fillna(obs.shift(lag))
    return filled.ffill()

def main():
    # Record hashes before doing any data/model work in v2.
    snapshot = HERE / 'original_artifacts.json'
    if not snapshot.exists():
        paths = list(common.V1.glob('*')) + list((OLD / 'models').glob('*'))
        snapshot.write_text(json.dumps({str(p.relative_to(common.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths if p.is_file()}, indent=2))
    for sub in ['weather', 'electricity']:
        (CACHE / sub).mkdir(exist_ok=True)
        for p in (OLD / sub).glob('*.json.gz'):
            if sub == 'electricity' and p.name == '2026-10-01.json.gz':
                continue
            link = CACHE / sub / p.name
            if not link.exists():
                link.symlink_to(p)
    fetch.CACHE = CACHE
    fetch.CONFIG = {**fetch.CONFIG, 'run_end': '2026-10-24'}
    fetch.fetch('https://api.energidataservice.dk/dataset/ElectricityProdex5MinRealtime',
                {'start': '2026-10-01', 'end': '2026-10-27', 'limit': 30000, 'sort': 'Minutes5UTC asc'},
                CACHE / 'electricity/2026-10-01.json.gz')
    raw = build.read_gzip(CACHE / 'electricity/2026-10-01.json.gz')
    assert raw['total'] == len(raw['records']), 'Truncated October response'
    fetch.weather()
    fetch.manifest()
    build.CACHE = CACHE
    build.HERE = HERE
    build.CONFIG = {**fetch.CONFIG, 'train_end': '2026-06-01', 'validation_end': '2026-08-01',
                    'calibration_end': '2026-10-01', 'test_end': '2026-10-27'}
    build.main()
    a = dict(np.load(CACHE / 'dataset.npz'))
    meta = pd.read_csv(CACHE / 'samples.csv')
    hourly = pd.read_parquet(CACHE / 'electricity_hourly.parquet')
    all_weather = {area: [] for area in build.AREAS}
    for path in sorted((CACHE / 'weather').glob('*.json.gz')):
        points = build.read_gzip(path)
        if isinstance(points, dict):
            continue
        day = pd.Timestamp(path.name[:10], tz='UTC')
        origin = day + pd.Timedelta(hours=12)
        times = pd.date_range(origin + pd.Timedelta(hours=1), periods=48, freq='h')
        for ai, area in enumerate(build.AREAS):
            w, _, _ = build.weather_features(points[ai*3:ai*3+3], times)
            all_weather[area].append(pd.concat([w, build.calendar(times)], axis=1).iloc[:24])
    all_weather = {k: pd.concat(v).sort_index() for k,v in all_weather.items()}
    extended, observed, past, missing_counts = [], [], [], []
    # Fill only with information preceding each gap. No backward interpolation.
    for area in build.AREAS:
        obs = hourly.loc[hourly.area.eq(area)].set_index('time')[build.TARGETS]
        filled = causal_fill(obs)
        all_weather[area] = all_weather[area].reindex(obs.index)
        for row in meta.loc[meta.area.eq(area)].itertuples():
            idx = pd.date_range(end=pd.Timestamp(row.origin), periods=512, freq='h')
            z = filled.reindex(idx).to_numpy()
            assert np.isfinite(z).all(), (row.origin, area, 'insufficient older observations')
            extended.append((row.Index, z))
            observed.append((row.Index, obs.reindex(idx).notna().to_numpy()))
            w = all_weather[area].reindex(idx)
            cal = build.calendar(idx)
            w[cal.columns] = cal
            past.append((row.Index, w.to_numpy()))
            missing_counts.append(int((~obs.reindex(idx).notna()).sum().sum()))
    for key, values in [('history512', extended), ('observed512', observed), ('past_full512', past)]:
        a[key] = np.stack([v for _,v in sorted(values)], axis=0).astype(np.float32)
    np.savez_compressed(CACHE / 'dataset.npz', **a)
    audit = json.loads((HERE / 'data_audit.json').read_text())
    audit.update({'extended_history_missing_cells': sum(missing_counts),
                  'extended_history_total_cells': int(a['history512'].size),
                  'created_at': dt.datetime.now(dt.timezone.utc).isoformat()})
    (HERE / 'data_audit.json').write_text(json.dumps(audit, indent=2))
    print('Extended data ready:', {k:v.shape for k,v in a.items()}, flush=True)

if __name__ == '__main__':
    main()
