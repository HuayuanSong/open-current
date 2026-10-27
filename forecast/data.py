"""Persistent public-source history and forecast-origin-safe live inputs."""
import datetime as dt
import gzip
import json
import time
from pathlib import Path
import requests
import numpy as np
import pandas as pd
from .features import AREAS,TARGETS,CONFIG,calendar,weather_features,normalize_electricity,causal_fill

ROOT=Path(__file__).resolve().parents[1]
STATE_PATH=ROOT/'dist/data/forecast-state.json.gz'
PUBLIC_PATH=ROOT/'dist/data/forecasts.json'

def now(): return pd.Timestamp.now(tz='UTC')

def request(url,params=None):
    for attempt in range(3):
        response=requests.get(url,params=params,timeout=60)
        if response.status_code in [429,500,502,503,504] and attempt<2:
            time.sleep(min(30,max(2,float(response.headers.get('Retry-After','5')))));continue
        response.raise_for_status();return response.json()
    raise RuntimeError('Source retries exhausted')

def empty_state(): return dict(schema=1,updatedAt=None,hourly=[],weather={},forecasts=[])

def read_state(path=STATE_PATH):
    if not path.exists():return empty_state()
    with gzip.open(path,'rt') as f: state=json.load(f)
    if state.get('schema')!=1:raise ValueError('Unrecognized forecast history')
    return state

def write_state(state,path=STATE_PATH):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp')
    with gzip.open(temporary,'wt') as f:json.dump(state,f,separators=(',',':'),allow_nan=False)
    temporary.replace(path)

def restore(base):
    """State is published with the site, not dependent on an evictable cache."""
    local=read_state();url=base.rstrip('/')+'/data/forecast-state.json.gz'
    response=requests.get(url,timeout=60)
    if response.status_code==404:return local
    response.raise_for_status();remote=json.loads(gzip.decompress(response.content))
    if remote.get('schema')!=1:raise ValueError('Unrecognized deployed forecast history')
    if (remote.get('updatedAt') or '')>(local.get('updatedAt') or ''):
        local=remote;write_state(local)
        # Preserve the corresponding last-good forecast on upstream failure.
        public=request(base.rstrip('/')+'/data/forecasts.json')
        PUBLIC_PATH.write_text(json.dumps(public,separators=(',',':'),allow_nan=False)+'\n')
    return local

def electricity(state,at):
    start=(at-pd.Timedelta(days=35 if not state['hourly'] else 3)).tz_convert('Europe/Copenhagen').date()
    payload=request('https://api.energidataservice.dk/dataset/ElectricityProdex5MinRealtime',
        {'start':str(start),'end':'now','limit':30000,'sort':'Minutes5UTC asc'})
    if payload.get('total')!=len(payload.get('records',[])) or not payload['records']:
        raise ValueError('Incomplete electricity response')
    raw=pd.DataFrame(payload['records']);hourly=normalize_electricity(raw)
    # Discard the still-open hour. Incomplete closed hours remain missing.
    hourly=hourly[hourly.time<=at.floor('h')]
    previous=pd.DataFrame(state['hourly'])
    if len(previous):
        previous['time']=pd.to_datetime(previous.time,utc=True)
        hourly=pd.concat([previous,hourly]).drop_duplicates(['time','area'],keep='last')
    hourly=hourly[hourly.time>=at-pd.Timedelta(days=42)].sort_values(['time','area'])
    serial=hourly.copy();serial['time']=serial.time.map(lambda x:x.isoformat())
    state['hourly']=json.loads(serial.to_json(orient='records'))
    state['electricityRetrievedAt']=now().isoformat()
    return hourly

def weather(state,at):
    # Same daily 00 UTC model vintage as training, conservatively available
    # after 12:15 UTC. Request five days to support origins throughout the day.
    latest=(at-pd.Timedelta(hours=12,minutes=15)).floor('D')
    days=pd.date_range(latest-pd.Timedelta(days=24),latest,freq='D')
    for day in days:
        key=str(day.date())
        cached=state['weather'].get(key)
        if cached and cached.get('data'):continue
        # An old missing run remains recorded; retry a recent missing run later.
        if cached and day<latest-pd.Timedelta(days=1):continue
        begin=time.monotonic()
        try:
            points=request('https://single-runs-api.open-meteo.com/v1/forecast',dict(
                latitude=','.join(str(p['latitude']) for p in CONFIG['locations']),
                longitude=','.join(str(p['longitude']) for p in CONFIG['locations']),
                hourly=','.join(CONFIG['variables']),models='ecmwf_ifs',run=f'{key}T00:00',
                forecast_days=5,timezone='UTC',wind_speed_unit='ms',cell_selection='nearest'))
            if not isinstance(points,list) or len(points)!=6:raise ValueError('Unexpected weather locations')
            state['weather'][key]={'retrievedAt':now().isoformat(),'data':points}
        except (requests.RequestException,ValueError) as error:
            state['weather'][key]={'retrievedAt':now().isoformat(),'error':str(error)[:180]}
        time.sleep(max(0,1.5-(time.monotonic()-begin)))
    state['weather']={k:v for k,v in state['weather'].items() if pd.Timestamp(k,tz='UTC')>=latest-pd.Timedelta(days=40)}

def fill_history(frame,origin):
    """Allow only isolated older gaps; no interpolation from later observations."""
    idx=pd.date_range(end=origin,periods=512,freq='h')
    unfilled=frame.reindex(idx)[TARGETS]
    recent=unfilled.iloc[-168:]
    if recent.iloc[-3:].isna().any().any():raise ValueError('Waiting for complete recent observations')
    missing=recent.isna().any(axis=1)
    max_run=int(missing.groupby((missing!=missing.shift()).cumsum()).sum().max())
    if int(missing.sum())>3 or max_run>2:
        raise ValueError('Too many missing input hours for a reliable update')
    if unfilled.isna().any(axis=1).sum()>25:raise ValueError('Insufficient 22-day history')
    filled=causal_fill(frame[TARGETS]).reindex(idx)
    if not np.isfinite(filled.to_numpy()).all():raise ValueError('History could not be filled causally')
    return filled.to_numpy(),unfilled.notna().to_numpy(),int(missing.sum())

def inputs(state,at):
    hourly=pd.DataFrame(state['hourly']);hourly['time']=pd.to_datetime(hourly.time,utc=True)
    # Latest jointly complete source hour, with a 15-minute publication buffer.
    cutoff=(at-pd.Timedelta(minutes=15)).floor('h')
    complete=hourly[hourly[TARGETS].notna().all(axis=1)&(hourly.time<=cutoff)].groupby('time').area.nunique()
    valid=complete[complete==2]
    if valid.empty:raise ValueError('No complete recent source hour')
    origin=valid.index.max()
    if cutoff-origin>pd.Timedelta(hours=2):raise ValueError('Recent electricity readings are delayed')
    times=pd.date_range(origin+pd.Timedelta(hours=1),periods=48,freq='h')
    future_run=None;future_points=None
    for key,entry in sorted(state['weather'].items(),reverse=True):
        run=pd.Timestamp(key,tz='UTC')
        if run+pd.Timedelta(hours=12,minutes=15)>at or not entry.get('data'):continue
        if at-run>pd.Timedelta(hours=48):break
        points=entry['data']
        check=[weather_features(points[ai*3:ai*3+3],times)[0].to_numpy() for ai in range(2)]
        if all(np.isfinite(x).all() for x in check):future_run=run;future_points=points;break
    if future_run is None:raise ValueError('No usable issued weather forecast')
    result={k:[] for k in ['history','future','history512','observed512','past_full512']}
    quality={};feature_names=None
    for ai,area in enumerate(AREAS):
        frame=hourly[hourly.area.eq(area)].set_index('time').sort_index()
        history,observed,gaps=fill_history(frame,origin);quality[area]={'filledInputHours':gaps}
        prior=[]
        for key,entry in sorted(state['weather'].items()):
            if not entry.get('data'):continue
            run=pd.Timestamp(key,tz='UTC')
            historical_origin=run+pd.Timedelta(hours=12)
            valid_times=pd.date_range(historical_origin+pd.Timedelta(hours=1),periods=24,freq='h')
            wx,_,_=weather_features(entry['data'][ai*3:ai*3+3],valid_times)
            prior.append(pd.concat([wx,calendar(valid_times)],axis=1))
        if not prior:raise ValueError('Missing historical weather')
        history_times=pd.date_range(end=origin,periods=512,freq='h')
        past=pd.concat(prior).sort_index().reindex(history_times)
        cal=calendar(history_times);past[cal.columns]=cal
        wx,_,_=weather_features(future_points[ai*3:ai*3+3],times)
        future=pd.concat([wx,calendar(times)],axis=1);feature_names=list(future.columns)
        result['history'].append(history[-168:]);result['history512'].append(history)
        result['observed512'].append(observed);result['past_full512'].append(past.to_numpy())
        result['future'].append(future.to_numpy())
    result={k:np.asarray(v,dtype=np.float32) for k,v in result.items()};result['area_ids']=np.array([0,1])
    metadata={'origin':origin.isoformat(),'weatherRun':future_run.isoformat(),
        'weatherRetrievedAt':state['weather'][str(future_run.date())]['retrievedAt'],
        'electricityRetrievedAt':state.get('electricityRetrievedAt'),'quality':quality,
        'featureNames':feature_names,'times':[t.isoformat() for t in times]}
    return result,metadata

def seed_from_research():
    """One-time local bootstrap using the existing licensed raw snapshots."""
    state=read_state()
    if state['hourly']:return
    path=ROOT/'.cache/forecasting/v2/electricity_hourly.parquet'
    if not path.exists():return
    at=now();frame=pd.read_parquet(path);frame=frame[frame.time>=at-pd.Timedelta(days=40)]
    frame['time']=frame.time.map(lambda t:t.isoformat());state['hourly']=json.loads(frame.to_json(orient='records'))
    # Re-fetch only recent weather with the longer operational horizon.
    for path in (ROOT/'.cache/forecasting/v2/weather').glob('*.json.gz'):
        day=pd.Timestamp(path.name[:10],tz='UTC')
        if not at-pd.Timedelta(days=28)<day<at-pd.Timedelta(days=3):continue
        with gzip.open(path,'rt') as f: raw=json.load(f)
        if isinstance(raw['data'],list):state['weather'][str(day.date())]={'retrievedAt':raw['retrieved_at'],'data':raw['data']}
    write_state(state)

if __name__=='__main__':seed_from_research()
