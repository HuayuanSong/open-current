"""Select empirical range construction before auditing October coverage."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from forecast.intervals import fit,bounds,interval_score,groups,BANDS,AREAS,TARGETS
CACHE=ROOT/'.cache/forecasting/v2';HERE=Path(__file__).parent
METHODS=['absolute','scaled','signed']

def data():
    a=dict(np.load(CACHE/'dataset.npz'));m=pd.read_csv(CACHE/'samples.csv')
    for key in ['origin','last_label']:m[key]=pd.to_datetime(m[key],utc=True)
    return a,m

def predictions(meta,stage):
    s=json.loads((ROOT/'research/forecasting/v2/final_selection.json').read_text())
    families={f:np.load(CACHE/f'{f}_{stage}.npz')['prediction'] for f in s['families']}
    n=len(next(iter(families.values())));m=meta.iloc[:n];out=np.full((n,48,3),np.nan)
    for rule in s['selection']:
        mask=m.area.eq(rule['area']).to_numpy();ti=TARGETS.index(rule['target'])
        for (lo,hi),weights in zip(s['bands'],rule['weights']):
            out[mask,lo:hi,ti]=sum(w*families[f][mask,lo:hi,ti] for f,w in zip(s['families'],weights))
    return out

def select():
    a,m=data();p=predictions(m,'dev');m=m.iloc[:len(p)];a={k:v[:len(p)] for k,v in a.items()};valid=np.isfinite(p).all(axis=(1,2))
    result={'version':'ranges-v1-2026-10-27','modelVersion':'ensemble-v2-2026-10-27',
            'scope':'Daily 12 UTC October 2026 origins; complete recent inputs; descriptive audit, not guaranteed coverage.',
            'protocolSha256':hashlib.sha256((HERE/'PROTOCOL.md').read_bytes()).hexdigest(),'series':{}}
    rows=[]
    for area in AREAS:
        area_mask=m.area.eq(area).to_numpy();initial=valid&area_mask&(m.last_label<pd.Timestamp('2026-08-01',tz='UTC'))
        final=valid&area_mask&(m.last_label<pd.Timestamp('2026-10-01',tz='UTC'))
        checks=np.flatnonzero(valid&area_mask&(m.origin>=pd.Timestamp('2026-08-01',tz='UTC')))
        result['series'][area]={}
        for ti,target in enumerate(TARGETS):
            floor=max(1.,float(np.median(a['y'][initial,:,ti][a['daylight'][initial].astype(bool)] if target=='solar' else a['y'][initial,:,ti]))*.25)
            scores={}
            for method in METHODS:
                measured=[]
                for i in checks:
                    mature=valid&area_mask&(m.last_label<m.origin.iloc[i]-pd.Timedelta(minutes=15))
                    rule=fit(a['y'][:len(m),:,ti][mature],p[mature,:,ti],a['daylight'][:len(m)][mature],target,method,floor)
                    lo,hi=bounds(p[i,:,ti],a['daylight'][i],target,rule)
                    measured.append(interval_score(a['y'][i,:,ti],lo,hi).mean())
                scores[method]=float(np.mean(measured));rows.append(dict(area=area,target=target,method=method,interval_score=scores[method],origins=len(checks)))
            method=min(scores,key=scores.get)
            rule=fit(a['y'][:len(m),:,ti][final],p[final,:,ti],a['daylight'][:len(m)][final],target,method,floor)
            # Climatological comparator is frozen before October is examined.
            climate=fit(a['y'][:len(m),:,ti][final],np.zeros_like(p[final,:,ti]),a['daylight'][:len(m)][final],target,'signed',1.)
            bucket=groups(a['daylight'][:len(m)][final],target)
            for key,cell in climate['cells'].items():
                subset=bucket==int(key)
                if subset.sum()<20:subset=(bucket%2==int(key)%2) if target=='solar' else np.ones_like(bucket,dtype=bool)
                cell['low']=float(np.quantile(a['y'][:len(m),:,ti][final][subset],.1,method='lower'))
            rule.update(developmentScores=scores,calibrationOrigins=int(final.sum()),climatology=climate)
            result['series'][area][target]=rule
    path=HERE/'range-selection.json'
    if path.exists():assert json.loads(path.read_text())==result,'Refusing to replace frozen range selection'
    else:path.write_text(json.dumps(result,indent=2)+'\n')
    pd.DataFrame(rows).to_csv(HERE/'range-development.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))

def audit():
    a,m=data();keep=m.origin>=pd.Timestamp('2026-10-01',tz='UTC')
    p=predictions(m,'final')[keep];y=a['y'][keep];day=a['daylight'][keep];mt=m.loc[keep].reset_index(drop=True)
    selection=json.loads((HERE/'range-selection.json').read_text());rows=[];lead_rows=[]
    lower=np.empty_like(p);upper=np.empty_like(p)
    for area in AREAS:
        mask=mt.area.eq(area).to_numpy()
        for ti,target in enumerate(TARGETS):
            rule=selection['series'][area][target];truth=y[mask,:,ti];pred=p[mask,:,ti];daylight=day[mask]
            lo,hi=bounds(pred,daylight,target,rule);lower[mask,:,ti]=lo;upper[mask,:,ti]=hi
            covered=(truth>=lo)&(truth<=hi);score=float(interval_score(truth,lo,hi).mean())
            clo,chi=bounds(np.zeros_like(pred),daylight,target,rule['climatology'])
            baseline=float(interval_score(truth,clo,chi).mean())
            bands=[]
            for left,right in BANDS:
                coverage=float(covered[:,left:right].mean());bands.append(coverage)
                lead_rows.append(dict(area=area,target=target,lead_start=left+1,lead_end=right,coverage=coverage,widthMW=float((hi-lo)[:,left:right].mean())))
            coverage=float(covered.mean());day_coverage=float(covered[daylight.astype(bool)].mean()) if target=='solar' else None
            audit={'coverage':coverage,'meanWidthMW':float((hi-lo).mean()),'intervalScore':score,'climatologyScore':baseline,
                   'origins':int(mask.sum()),'forecastHours':int(truth.size),'daylightCoverage':day_coverage,'leadCoverage':bands}
            rule['enabled']=bool(.7<=coverage<=.9 and min(bands)>=.65 and (day_coverage is None or day_coverage>=.65) and score<baseline)
            rule['audit']=audit;rows.append(dict(area=area,target=target,method=rule['method'],enabled=rule['enabled'],**audit))
    pd.DataFrame(rows).to_csv(HERE/'range-audit.csv',index=False);pd.DataFrame(lead_rows).to_csv(HERE/'range-leads.csv',index=False)
    (ROOT/'forecast/interval-calibration.json').write_text(json.dumps(selection,indent=2)+'\n')
    np.savez_compressed(ROOT/'.cache/forecasting/uncertainty/audit.npz',actual=y,prediction=p,lower=lower,upper=upper)
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['select','audit']);args=parser.parse_args()
    globals()[args.stage]()
