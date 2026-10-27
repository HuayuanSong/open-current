"""Append-only issued forecasts and first eligible observation snapshots."""
import hashlib
import json
import numpy as np
import pandas as pd
from forecast.features import AREAS,TARGETS

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def initialize(state,protocol):
    fingerprint=canonical_hash(protocol)
    if 'trial' not in state:
        state['trial']=dict(schema=1,protocol=protocol,protocolHash=fingerprint,records=[],observations={},attempts=[])
    trial=state['trial']
    if trial.get('schema')!=1 or trial.get('protocolHash')!=fingerprint or trial.get('protocol')!=protocol:
        raise ValueError('Trial protocol mismatch; preserving the existing archive')
    for record in trial['records']:
        if record.get('sha256')!=canonical_hash({k:v for k,v in record.items() if k!='sha256'}):
            raise ValueError('An issued trial forecast was modified')
    return trial

def dates(protocol):
    return pd.date_range(protocol['originStart'],pd.Timestamp(protocol['originEndExclusive'])-pd.Timedelta(days=1),freq='D',tz='UTC')

def bind_inference(trial,fingerprint):
    previous=trial.get('inferenceLockHash')
    if previous is None and not trial['records']:trial['inferenceLockHash']=fingerprint
    elif previous!=fingerprint:raise ValueError('Frozen inference lock changed during the trial')

def review_after(protocol):
    return dates(protocol)[-1]+pd.Timedelta(hours=protocol['originHourUTC']+protocol['horizonHours']+protocol['observationMaxAgeHours'])

def eligible_origin(at,protocol):
    at=pd.Timestamp(at)
    if at.tzinfo is None:raise ValueError('Trial issue time requires a timezone')
    at=at.tz_convert('UTC');origin=at.floor('D')+pd.Timedelta(hours=protocol['originHourUTC'])
    if not (pd.Timestamp(protocol['originStart'],tz='UTC')<=origin<pd.Timestamp(protocol['originEndExclusive'],tz='UTC')):return None
    if origin+pd.Timedelta(minutes=protocol['issueStartMinute'])<=at<origin+pd.Timedelta(minutes=protocol['issueDeadlineMinute']):return origin
    return None

def append(trial,record,at):
    p=trial['protocol'];origin=eligible_origin(at,p)
    if origin is None or pd.Timestamp(record['origin'])!=origin:raise ValueError('Outside prospective issue window; no backfill')
    if pd.Timestamp(record['issuedAt'])!=pd.Timestamp(at):raise ValueError('Issue timestamp is not the current scoring time')
    if record.get('trialVersion')!=p['version'] or record.get('baselineVersion')!=p['baselineVersion']:
        raise ValueError('Model version differs from frozen trial')
    expected=pd.date_range(origin+pd.Timedelta(hours=1),periods=p['horizonHours'],freq='h')
    if list(pd.to_datetime(record['times'],utc=True))!=list(expected) or expected[0]<=pd.Timestamp(at):
        raise ValueError('Forecast targets must follow issuance and match the fixed horizon')
    if any(pd.Timestamp(r['origin'])==origin for r in trial['records']):return False
    if any(len(record['daylight'][area])!=48 for area in AREAS):raise ValueError('Daylight mask must match forecast horizon')
    for candidate,areas in record['predictions'].items():
        for area in AREAS:
            for target in TARGETS:
                values=areas[area][target]
                if values is None:
                    if candidate=='baseline':raise ValueError('Baseline may not be missing')
                    continue
                if len(values)!=48 or not np.isfinite(values).all() or min(values)<0:raise ValueError('Invalid issued values')
    if 'baseline' not in record['predictions'] or 'primary' not in record['predictions']:raise ValueError('Missing paired models')
    saved=json.loads(json.dumps(record,allow_nan=False));saved['sha256']=canonical_hash(saved)
    trial['records'].append(saved)
    return True

def freeze_observations(trial,hourly,at):
    """Store the first complete reading 24–72h after its valid time, once only."""
    p=trial['protocol'];at=pd.Timestamp(at)
    wanted={pd.Timestamp(t).isoformat() for r in trial['records'] for t in r['times']}
    for row in hourly:
        stamp=pd.Timestamp(row['time']).isoformat();area=row['area']
        if area not in AREAS or stamp not in wanted:continue
        age=(at-pd.Timestamp(stamp)).total_seconds()/3600
        if not p['observationMinAgeHours']<=age<=p['observationMaxAgeHours']:continue
        # Freeze each series independently; missing solar must not suppress wind.
        slot=trial['observations'].setdefault(stamp,{})
        values=slot.setdefault(area,{})
        for target in TARGETS:
            value=row.get(target)
            if target in values or value is None or not np.isfinite(value):continue
            if value<0 or (target=='load' and value==0):continue
            values[target]={'value':float(value),'capturedAt':at.isoformat()}

def attempt(trial,at,status,message=None):
    # Successful/failed issuance attempts only, not every off-hours cron tick.
    entry=dict(at=pd.Timestamp(at).isoformat(),status=status,message=message)
    trial['attempts'].append(entry)
