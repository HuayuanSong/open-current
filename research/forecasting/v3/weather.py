"""Rate-limited immutable single-run downloads; no best-match/observed weather."""
from research.forecasting.v3.common import CACHE, HERE, load_v2, digest, write_json
import argparse
import datetime as dt
import gzip
import json
import time
import requests
import pandas as pd
from forecast.features import CONFIG

ENDPOINT='https://single-runs-api.open-meteo.com/v1/forecast'

def latest_available_run(metadata, at, consistency_minutes=10):
    """Use publication time, not initialization time, for live run eligibility.

    A not-yet-replicated newest run returns None; a caller must retain an older
    recorded complete run rather than guess its predecessor's availability.
    """
    at=pd.Timestamp(at)
    if at.tzinfo is None:raise ValueError('Issue time must be timezone-aware')
    run=pd.to_datetime(metadata['last_run_initialisation_time'],unit='s',utc=True)
    available=pd.to_datetime(metadata['last_run_availability_time'],unit='s',utc=True)
    if available<run:raise ValueError('Availability precedes model initialization')
    return run if available+pd.Timedelta(minutes=consistency_minutes)<=at else None

def read(path):
    with gzip.open(path,'rt') as file: return json.load(file)

def fetch(kind, day):
    points=json.loads((HERE/'weather-points.json').read_text()) if kind=='fleet' else CONFIG['locations']
    model='dmi_harmonie_arome_europe' if kind=='regional' else 'ecmwf_ifs'
    hour=0 if kind=='fleet' else 6
    run=pd.Timestamp(day,tz='UTC')+pd.Timedelta(hours=hour)
    path=CACHE/'weather'/kind/f'{day}.json.gz'
    if path.exists():
        cached=read(path)
        if cached['points']!=points or cached['model']!=model or pd.Timestamp(cached['run'])!=run:
            raise ValueError(f'Cache provenance mismatch: {path}; use a new experiment cache')
        return cached
    params=dict(latitude=','.join(str(p['latitude']) for p in points),
                longitude=','.join(str(p['longitude']) for p in points),
                models=model,run=run.strftime('%Y-%m-%dT%H:%M'),
                hourly=','.join(CONFIG['variables']),forecast_days=4,
                timezone='UTC',wind_speed_unit='ms',cell_selection='nearest')
    for attempt in range(4):
        start=time.monotonic()
        response=requests.get(ENDPOINT,params=params,timeout=60)
        # Six locations count as multiple API units. At most 60 requests/minute.
        time.sleep(max(0,1.05-(time.monotonic()-start)))
        if response.status_code==429:
            if 'daily' in response.text.lower() or 'day' in response.text.lower():
                raise RuntimeError('Daily provider quota reached; resume this cache another day')
            time.sleep(min(30, float(response.headers.get('Retry-After',15))))
            continue
        if response.status_code in [500,502,503,504] and attempt<3:
            time.sleep(3); continue
        if response.status_code==400 and any(x in response.text.lower() for x in ['not available','not found','no data']):
            data={'unavailable':True,'reason':response.text[:400]};break
        response.raise_for_status();data=response.json();break
    else: raise RuntimeError('Provider retries exhausted')
    if isinstance(data,list):
        if len(data)!=len(points): raise ValueError('Weather location count mismatch')
        for point in data:
            if 'hourly' not in point or not set(CONFIG['variables']).issubset(point['hourly']):
                raise ValueError('Weather variable mismatch')
    elif not data.get('unavailable'): raise ValueError('Unexpected weather response')
    payload=dict(model=model,run=run.isoformat(),retrieved_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                 url=response.url,points=points,data=data)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    with gzip.open(tmp,'wt') as file:json.dump(payload,file,allow_nan=False)
    tmp.replace(path)
    return payload

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true');args=parser.parse_args()
    _,m,_=load_v2();days=sorted(m.origin.dt.strftime('%Y-%m-%d').unique())
    if args.pilot: days=['2026-09-01']
    jobs=[(kind,day) for day in days for kind in ['fleet','fresh','regional'] if kind!='regional' or day>='2026-04-02']
    for i,(kind,day) in enumerate(jobs):
        data=fetch(kind,day)
        if i%20==0 or args.pilot: print(f'{i+1}/{len(jobs)} {kind} {day} ready',flush=True)
    paths=sorted((CACHE/'weather').glob('*/*.json.gz'))
    write_json(HERE/'weather-manifest.json',{str(p.relative_to(CACHE)):digest(p) for p in paths})
    print(f'Complete: {len(paths)} cached issued weather runs',flush=True)

if __name__=='__main__':main()
