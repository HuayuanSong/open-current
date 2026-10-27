"""Publish a compact panel payload and score only forecasts actually issued."""
import json
import numpy as np
import pandas as pd
from .features import AREAS,TARGETS
from .data import PUBLIC_PATH
from .intervals import validation

def performance(state,at,version):
    actual={(r['area'],r['time']):r for r in state['hourly']}
    totals={area:{target:dict(error=0.,actual=0.,count=0,origins=set(),range_count=0,covered=0,width=0.,range_origins=set()) for target in TARGETS} for area in AREAS}
    range_version=(validation() or {}).get('version')
    for record in state['forecasts']:
        if record['modelVersion']!=version or pd.Timestamp(record['issuedAt'])<at-pd.Timedelta(days=7):continue
        for area in AREAS:
            for i,stamp in enumerate(record['times']):
                valid_time=pd.Timestamp(stamp)
                if valid_time<=pd.Timestamp(record['issuedAt']) or valid_time>at-pd.Timedelta(minutes=15):continue
                observed=actual.get((area,stamp))
                if not observed:continue
                for target in TARGETS:
                    value=observed.get(target)
                    if value is None:continue
                    stat=totals[area][target];stat['error']+=abs(record['areas'][area][target][i]-value)
                    stat['actual']+=value;stat['count']+=1;stat['origins'].add(record['id'])
                    ranges=record.get('ranges',{})
                    band=ranges.get('areas',{}).get(area,{}).get(target)
                    if band and ranges.get('version')==range_version and ranges.get('issuedAt')==record['issuedAt']:
                        lo,hi=band['lower'][i],band['upper'][i]
                        if lo is not None and hi is not None:
                            stat['range_count']+=1;stat['covered']+=int(lo<=value<=hi);stat['width']+=hi-lo;stat['range_origins'].add(record['id'])
    return {area:{target:{'maeMW':s['error']/s['count'] if s['count'] else None,
        'wapePct':100*s['error']/s['actual'] if s['actual']>0 else None,
        'forecastHours':s['count'],'origins':len(s['origins']),
        'rangeCoverage':s['covered']/s['range_count'] if s['range_count'] else None,'rangeHours':s['range_count'],
        'rangeOrigins':len(s['range_origins']),'meanRangeWidthMW':s['width']/s['range_count'] if s['range_count'] else None} for target,s in targets.items()} for area,targets in totals.items()}

def payload(state,at,version,error=None):
    records=[r for r in state['forecasts'] if r['modelVersion']==version]
    latest=max(records,key=lambda r:(r['origin'],r['issuedAt'])) if records else None
    result={'schema':1,'checkedAt':at.isoformat(),'modelVersion':version,'experimental':True,
            'status':'delayed' if error and latest else 'unavailable' if not latest else 'ready',
            'message':error,'latest':latest,'comparison':None,'observations':{},
            'performance':performance(state,at,version),
            'rangeValidation':validation(),
            'methodologyUrl':'https://github.com/HuayuanSong/open-current/tree/main/research/forecasting/v2'}
    for area in AREAS:
        result['observations'][area]=[r for r in state['hourly'] if r['area']==area and pd.Timestamp(r['time'])>=at-pd.Timedelta(hours=72)]
    if latest:
        prior=[r for r in records if pd.Timestamp(r['origin'])<pd.Timestamp(latest['origin'])]
        if prior:
            # Prefer a forecast issued about 24 hours earlier; never reconstruct it.
            target=pd.Timestamp(latest['origin'])-pd.Timedelta(hours=24)
            result['comparison']=min(prior,key=lambda r:abs(pd.Timestamp(r['origin'])-target))
    return result

def write_public(state,at,version,error=None):
    public=payload(state,at,version,error)
    temporary=PUBLIC_PATH.with_suffix('.tmp')
    temporary.write_text(json.dumps(public,separators=(',',':'),allow_nan=False)+'\n');temporary.replace(PUBLIC_PATH)
    return public
