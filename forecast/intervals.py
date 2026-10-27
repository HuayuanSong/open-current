"""Empirical pointwise prediction ranges; no independence/coverage guarantee."""
import json
from pathlib import Path
import numpy as np

BANDS=[(0,6),(6,24),(24,48)]
AREAS=['DK1','DK2'];TARGETS=['wind','solar','load']
CALIBRATION=Path(__file__).with_name('interval-calibration.json')

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

def build(prediction,daylight,issued_at):
    calibration=json.loads(CALIBRATION.read_text())
    areas={}
    for ai,area in enumerate(AREAS):
        areas[area]={}
        for ti,target in enumerate(TARGETS):
            rule=calibration['series'][area][target]
            if not rule['enabled']:continue
            lo,hi=bounds(prediction[ai,:,ti],daylight[ai],target,rule)
            areas[area][target]={'lower':lo.astype(float).tolist(),'upper':hi.astype(float).tolist()}
    return {'version':calibration['version'],'modelVersion':calibration['modelVersion'],
            'nominalCoverage':.8,'issuedAt':issued_at,'areas':areas}

def validation():
    if not CALIBRATION.exists():return None
    c=json.loads(CALIBRATION.read_text())
    return {'version':c['version'],'nominalCoverage':.8,'scope':c['scope'],
            'series':{a:{t:{'enabled':s['enabled'],**s['audit']} for t,s in ts.items()} for a,ts in c['series'].items()},
            'url':'https://github.com/HuayuanSong/open-current/tree/main/research/forecasting/uncertainty'}
