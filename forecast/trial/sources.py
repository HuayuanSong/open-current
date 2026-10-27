"""Fetch exact, currently available model runs; retain request-time evidence."""
import hashlib
import json
import time
import numpy as np
import pandas as pd
from forecast import data
from forecast.features import CONFIG
from .features import weather_array,fleet_array,regional_tail

def fetch(model,run,points):
    if run>=data.now():raise ValueError('Weather model run is in the future')
    begin=time.monotonic();retrieved=data.now()
    raw=data.request('https://single-runs-api.open-meteo.com/v1/forecast',dict(
        latitude=','.join(str(p['latitude']) for p in points),longitude=','.join(str(p['longitude']) for p in points),
        hourly=','.join(CONFIG['variables']),models=model,run=run.strftime('%Y-%m-%dT%H:%M'),
        forecast_days=4,timezone='UTC',wind_speed_unit='ms',cell_selection='nearest'))
    time.sleep(max(0,1.05-(time.monotonic()-begin)))
    if not isinstance(raw,list) or len(raw)!=len(points):raise ValueError('Unexpected weather response')
    return dict(model=model,run=run.isoformat(),requestStartedAt=retrieved.isoformat(),retrievedAt=data.now().isoformat(),
                points=points,data=raw,sha256=hashlib.sha256(json.dumps(raw,sort_keys=True,separators=(',',':')).encode()).hexdigest())

def inputs(arrays,meta,bundle,protocol):
    origin=pd.Timestamp(meta['origin']);day=origin.floor('D');times=pd.to_datetime(meta['times'],utc=True)
    if pd.Timestamp(meta['weatherRun'])!=day:raise ValueError('Trial requires the matching 00 UTC baseline weather run')
    # Runs are fixed by the trial, not selected retrospectively for their errors.
    # A successful exact-run request proves availability before issuance. Using
    # a later run would change the frozen comparison and is explicitly disallowed.
    run=day+pd.Timedelta(hours=protocol['ecmwfRunHourUTC'])
    snapshots={};errors={};out={k:v.copy() for k,v in arrays.items()}
    snapshots['fresh']=fetch('ecmwf_ifs',run,CONFIG['locations'])
    out['fresh']=weather_array(snapshots['fresh']['data'],times)
    if not np.isfinite(out['fresh']).all():raise ValueError('Incomplete ECMWF forecast')
    points=json.loads((bundle/'weather-points.json').read_text());capacity=json.loads((bundle/'capacity.json').read_text())
    available=['evening_control','newer_weather']
    try:
        snapshots['fleet']=fetch('ecmwf_ifs',day,points)
        out['fleet']=fleet_array(snapshots['fleet']['data'],points,capacity,times)
        available.append('fleet_weather')
    except Exception as exc:errors['fleet_weather']=str(exc)[:240]
    try:
        snapshots['regional']=fetch('dmi_harmonie_arome_europe',day+pd.Timedelta(hours=protocol['dmiRunHourUTC']),CONFIG['locations'])
        wx=weather_array(snapshots['regional']['data'],times)
        out['regional'],out['regional_native']=regional_tail(wx,out['fresh'],pd.Timestamp(snapshots['regional']['run']),times)
        available.append('regional_weather')
    except Exception as exc:errors['regional_weather']=str(exc)[:240]
    return out,snapshots,available,errors
