"""Empirical pointwise prediction ranges; no independence/coverage guarantee."""
import json
import gzip
from pathlib import Path
import numpy as np
import pandas as pd

BANDS=[(0,6),(6,24),(24,48)]
AREAS=['DK1','DK2'];TARGETS=['wind','solar','load']
CALIBRATION=Path(__file__).with_name('interval-calibration.json')
HISTORY=Path(__file__).with_name('interval-history.json.gz')

def groups(daylight,target):
    lead=np.broadcast_to(np.arange(48),np.asarray(daylight).shape)
    band=np.where(lead<6,0,np.where(lead<24,1,2))
    return band*2+(np.asarray(daylight,dtype=int) if target=='solar' else 0)

def fit(y,p,daylight,target,method,floor):
    y=np.asarray(y);p=np.asarray(p);g=groups(daylight,target)
    residual=y-p;scale=np.maximum(p,floor) if method=='scaled' else np.ones_like(p)
    score=residual/scale;cells={}
    for key in range(6):
        if target!='solar' and key%2:continue
        mask=g==key
        if mask.sum()<20:mask=(g%2==key%2) if target=='solar' else np.ones_like(g,dtype=bool)
        values=score[mask]
        if len(values)<20:raise ValueError('Insufficient range calibration observations')
        if method=='signed':
            lo=float(np.quantile(values,.1,method='lower'));hi=float(np.quantile(values,.9,method='higher'))
        else:
            rank=min(len(values)-1,int(np.ceil((len(values)+1)*.8))-1)
            radius=float(np.sort(np.abs(values))[rank]);lo=-radius;hi=radius
        cells[str(key)]={'low':min(0.,lo),'high':max(0.,hi),'observations':int(len(values))}
    return {'method':method,'floorMW':float(floor),'cells':cells}

def bounds(p,daylight,target,rule):
    p=np.asarray(p);g=groups(daylight,target);lo=np.zeros_like(p);hi=np.zeros_like(p)
    scale=np.maximum(p,rule['floorMW']) if rule['method']=='scaled' else np.ones_like(p)
    for key,cell in rule['cells'].items():
        mask=g==int(key);lo[mask]=p[mask]+cell['low']*scale[mask];hi[mask]=p[mask]+cell['high']*scale[mask]
    return np.maximum(0,lo),np.maximum(p,hi)

def interval_score(y,lo,hi):
    return hi-lo+10*np.maximum(0,lo-y)+10*np.maximum(0,y-hi)

def refresh_history(state,at):
    """Keep one mature forecast per region/day; all labels precede cutoff."""
    with gzip.open(HISTORY,'rt') as f:seed=json.load(f)
    candidates=seed+state.get('rangeHistory',[])
    actual={(r['area'],r['time']):r for r in state['hourly']}
    for record in state['forecasts']:
        if record['modelVersion']!='ensemble-v2-2026-10-27' or not record.get('daylight'):continue
        if pd.Timestamp(record['times'][-1])>=at-pd.Timedelta(minutes=15):continue
        if pd.Timestamp(record['issuedAt'])>=pd.Timestamp(record['times'][0]):continue
        for area in AREAS:
            readings=[actual.get((area,t)) for t in record['times']]
            if any(r is None or any(r.get(t) is None for t in TARGETS) for r in readings):continue
            candidates.append({'area':area,'origin':record['origin'],'lastLabel':record['times'][-1],
                'prediction':np.array([record['areas'][area][t] for t in TARGETS]).T.tolist(),
                'actual':[[r[t] for t in TARGETS] for r in readings],'daylight':record['daylight'][area],
                'source':'issued','issuedAt':record['issuedAt']})
    chosen={}
    def rank(r):
        origin=pd.Timestamp(r['origin'])
        return (r['source']!='issued',abs(origin.hour+origin.minute/60-12),r.get('issuedAt',''))
    for r in candidates:
        origin=pd.Timestamp(r['origin'])
        if origin<at-pd.Timedelta(days=64) or pd.Timestamp(r['lastLabel'])>=at-pd.Timedelta(minutes=15):continue
        key=(r['area'],str(origin.date()))
        if key not in chosen or rank(r)<rank(chosen[key]):chosen[key]=r
    state['rangeHistory']=list(chosen.values())
    return state['rangeHistory']

def mature_history(history,area,at,days):
    valid=[r for r in history if r['area']==area and pd.Timestamp(r['lastLabel'])<at-pd.Timedelta(minutes=15)]
    selected=[r for r in valid if pd.Timestamp(r['origin'])>=at-pd.Timedelta(days=days)]
    if len(selected)<7:selected=[r for r in valid if pd.Timestamp(r['origin'])>=at-pd.Timedelta(days=56)]
    if len(selected)<7:raise ValueError('Not enough mature daily calibration origins')
    return selected

def build(prediction,daylight,issued_at,state,origin=None):
    calibration=json.loads(CALIBRATION.read_text())
    # Use the same conservative origin-based cutoff as the historical audit.
    areas={};counts={};at=pd.Timestamp(origin or issued_at)
    for ai,area in enumerate(AREAS):
        areas[area]={};counts[area]={}
        for ti,target in enumerate(TARGETS):
            spec=calibration['series'][area][target]
            if not spec['enabled']:continue
            try:history=mature_history(state.get('rangeHistory',[]),area,at,spec['windowDays'])
            except ValueError:continue
            truth=np.array([r['actual'] for r in history])[:,:,ti]
            past=np.array([r['prediction'] for r in history])[:,:,ti]
            sun=np.array([r['daylight'] for r in history])
            rule=fit(truth,past,sun,target,spec['method'],spec['floorMW'])
            lo,hi=bounds(prediction[ai,:,ti],daylight[ai],target,rule)
            visible=daylight[ai].astype(bool) if spec['scope']=='daylight' else np.ones(48,dtype=bool)
            areas[area][target]={'lower':[float(v) if ok else None for v,ok in zip(lo,visible)],
                                'upper':[float(v) if ok else None for v,ok in zip(hi,visible)],'scope':spec['scope']}
            effective=56 if any(pd.Timestamp(r['origin'])<at-pd.Timedelta(days=spec['windowDays']) for r in history) else spec['windowDays']
            counts[area][target]={'origins':len(history),'lastLabel':max(r['lastLabel'] for r in history),'windowDays':spec['windowDays'],'effectiveWindowDays':effective}
    return {'version':calibration['version'],'modelVersion':calibration['modelVersion'],
            'nominalCoverage':.8,'issuedAt':issued_at,'calibratedBefore':(at-pd.Timedelta(minutes=15)).isoformat(),'calibration':counts,'areas':areas}

def validation():
    if not CALIBRATION.exists():return None
    c=json.loads(CALIBRATION.read_text())
    return {'version':c['version'],'nominalCoverage':.8,'scope':c['scope'],
            'series':{a:{t:{'enabled':s['enabled'],'scope':s['scope'],**s['audit']} for t,s in ts.items()} for a,ts in c['series'].items()},
            'url':'https://github.com/HuayuanSong/open-current/tree/main/research/forecasting/uncertainty'}
