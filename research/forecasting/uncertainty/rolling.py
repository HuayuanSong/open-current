"""Rolling residual calibration selected without using October scores."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from ranges import data,predictions,ROOT,HERE,METHODS
from forecast.intervals import fit,bounds,interval_score,BANDS,AREAS,TARGETS
WINDOWS=[14,28,56]

def eligible(meta,valid,area,at,days):
    mature=valid&meta.area.eq(area).to_numpy()&(meta.last_label<at-pd.Timedelta(minutes=15))
    selected=mature&(meta.origin>=at-pd.Timedelta(days=days))
    if selected.sum()<7:selected=mature&(meta.origin>=at-pd.Timedelta(days=56))
    if selected.sum()<7:raise ValueError('Not enough mature daily calibration origins')
    return np.asarray(selected)

def select():
    a,m=data();p=predictions(m,'dev');m=m.iloc[:len(p)];a={k:v[:len(p)] for k,v in a.items()}
    valid=np.isfinite(p).all(axis=(1,2));old=json.loads((HERE/'range-selection.json').read_text())
    result={'version':'ranges-v2-2026-10-27','modelVersion':old['modelVersion'],'scope':old['scope']+' October was reused to diagnose the initial failed solar ranges.',
        'protocolSha256':hashlib.sha256((HERE/'ROLLING_PROTOCOL.md').read_bytes()).hexdigest(),'series':{}}
    rows=[]
    for area in AREAS:
        checks=np.flatnonzero(valid&m.area.eq(area).to_numpy()&(m.origin>=pd.Timestamp('2026-08-01',tz='UTC')));result['series'][area]={}
        for ti,target in enumerate(TARGETS):
            floor=old['series'][area][target]['floorMW'];candidates=[]
            for days in WINDOWS:
                for method in METHODS:
                    scores=[]
                    for i in checks:
                        keep=eligible(m,valid,area,m.origin.iloc[i],days)
                        rule=fit(a['y'][keep,:,ti],p[keep,:,ti],a['daylight'][keep],target,method,floor)
                        lo,hi=bounds(p[i,:,ti],a['daylight'][i],target,rule);scores.append(interval_score(a['y'][i,:,ti],lo,hi).mean())
                    row=dict(area=area,target=target,windowDays=days,method=method,intervalScore=float(np.mean(scores)))
                    rows.append(row);candidates.append(row)
            best=min(candidates,key=lambda r:r['intervalScore'])
            result['series'][area][target]={**best,'floorMW':floor,'climatology':old['series'][area][target]['climatology']}
    path=HERE/'rolling-selection.json'
    if path.exists():assert json.loads(path.read_text())==result,'Selection is frozen'
    else:path.write_text(json.dumps(result,indent=2)+'\n')
    pd.DataFrame(rows).to_csv(HERE/'rolling-development.csv',index=False)
    print(json.dumps({a:{t:{k:s[k] for k in ['method','windowDays','intervalScore']} for t,s in v.items()} for a,v in result['series'].items()},indent=2))

def audit():
    a,m=data();dev=predictions(m,'dev');final=predictions(m,'final');p=np.concatenate([dev,final[len(dev):]])
    valid=np.isfinite(p).all(axis=(1,2));checks=np.flatnonzero(m.split.eq('test'))
    selection=json.loads((HERE/'rolling-selection.json').read_text());rows=[];leadrows=[];originrows=[];lower=np.full_like(p,np.nan);upper=np.full_like(p,np.nan)
    for area in AREAS:
        ids=np.array([i for i in checks if m.area.iloc[i]==area])
        for ti,target in enumerate(TARGETS):
            spec=selection['series'][area][target]
            for i in ids:
                keep=eligible(m,valid,area,m.origin.iloc[i],spec['windowDays'])
                rule=fit(a['y'][keep,:,ti],p[keep,:,ti],a['daylight'][keep],target,spec['method'],spec['floorMW'])
                lo,hi=bounds(p[i,:,ti],a['daylight'][i],target,rule);lower[i,:,ti]=lo;upper[i,:,ti]=hi
                originrows.append(dict(area=area,target=target,origin=m.origin.iloc[i].isoformat(),calibrationOrigins=int(keep.sum()),
                    lastCalibrationLabel=m.last_label[keep].max().isoformat(),coverage=float(((a['y'][i,:,ti]>=lo)&(a['y'][i,:,ti]<=hi)).mean())))
            truth=a['y'][ids,:,ti];lo=lower[ids,:,ti];hi=upper[ids,:,ti];covered=(truth>=lo)&(truth<=hi);day=a['daylight'][ids].astype(bool)
            coverage=float(covered.mean());bands=[]
            for left,right in BANDS:
                rate=float(covered[:,left:right].mean());bands.append(rate);leadrows.append(dict(area=area,target=target,leadStart=left+1,leadEnd=right,coverage=rate))
            daylight=float(covered[day].mean()) if target=='solar' else None
            score=float(interval_score(truth,lo,hi).mean());clo,chi=bounds(np.zeros_like(truth),day,target,spec['climatology']);baseline=float(interval_score(truth,clo,chi).mean())
            audit={'coverage':coverage,'meanWidthMW':float((hi-lo).mean()),'intervalScore':score,'climatologyScore':baseline,
                'origins':len(ids),'forecastHours':int(truth.size),'daylightCoverage':daylight,'leadCoverage':bands}
            spec.update(audit=audit,enabled=bool(.7<=coverage<=.9 and min(bands)>=.65 and (daylight is None or daylight>=.65) and score<baseline))
            rows.append(dict(area=area,target=target,method=spec['method'],windowDays=spec['windowDays'],enabled=spec['enabled'],**audit))
    pd.DataFrame(rows).to_csv(HERE/'rolling-audit.csv',index=False);pd.DataFrame(leadrows).to_csv(HERE/'rolling-leads.csv',index=False)
    pd.DataFrame(originrows).to_csv(HERE/'rolling-origin-audit.csv',index=False)
    (ROOT/'forecast/interval-calibration.json').write_text(json.dumps(selection,indent=2)+'\n')
    records=[]
    for i in np.flatnonzero(valid):
        records.append({'area':m.area.iloc[i],'origin':m.origin.iloc[i].isoformat(),'lastLabel':m.last_label.iloc[i].isoformat(),
                        'prediction':p[i].tolist(),'actual':a['y'][i].tolist(),'daylight':a['daylight'][i].astype(bool).tolist(),'source':'historical'})
    with gzip.open(ROOT/'forecast/interval-history.json.gz','wt') as f:json.dump(records,f,separators=(',',':'),allow_nan=False)
    np.savez_compressed(ROOT/'.cache/forecasting/uncertainty/rolling-audit.npz',actual=a['y'][checks],prediction=p[checks],lower=lower[checks],upper=upper[checks])
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['select','audit']);args=parser.parse_args();globals()[args.stage]()
