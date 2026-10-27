"""Matched feature/run ablations, with development selection before October."""
from research.forecasting.v3.common import *
import argparse
import json
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from forecast.features import features

SETTINGS=[dict(n_estimators=400,num_leaves=15,min_child_samples=100),
          dict(n_estimators=800,num_leaves=31,min_child_samples=60)]
STUDIES={'fleet':(12,['control','fleet'],[0]),
         'evening':(18,['control','fresh','fresh_fleet'],[0,1,2]),
         'regional':(18,['control','regional','multi_weather'],[0,1,2])}

def load(study):
    hour,variants,targets=STUDIES[study]
    a=dict(np.load(CACHE/f'data-{hour}.npz'));m=pd.read_csv(CACHE/f'samples-{hour}.csv')
    for key in ['origin','last_label']:m[key]=pd.to_datetime(m[key],utc=True)
    if study=='regional':
        keep=np.isfinite(a['regional']).all((1,2));a={k:v[keep] for k,v in a.items()};m=m.loc[keep].reset_index(drop=True)
    names=json.loads((HERE/'feature-names.json').read_text())
    return a,m,names

def matrix(a,names,ti,study,variant):
    weather='future'
    if study=='evening' and variant in ['fresh','fresh_fleet']:weather='fresh'
    if study=='regional':weather='regional' if variant=='regional' else 'fresh'
    x=features({**a,'future':a[weather]},names,ti,True)
    if ti==0 and variant in ['fleet','fresh_fleet']:
        x=np.c_[x,a['fleet'].reshape(len(x),-1)]
    if variant=='multi_weather':
        cols=[i for i,n in enumerate(names['future']) if n.startswith('wx_')]
        x=np.c_[x,a['regional'][:,:,cols].reshape(len(x),-1)]
    if study=='regional' and variant in ['regional','multi_weather']:
        x=np.c_[x,a['regional_native'].ravel()]
    if not np.isfinite(x).all():raise ValueError('Missing input feature')
    return x

def main(study,stage):
    a,m,names=load(study);_,variants,targets=STUDIES[study]
    selection_path=HERE/f'{study}-selection.json'
    selections=json.loads(selection_path.read_text())['selection'] if stage=='evaluate' else None
    if stage=='select':
        # Remove October from the process, rather than relying on a fit mask.
        keep=(m.origin<pd.Timestamp('2026-10-01',tz='UTC')).to_numpy()
        a={k:v[keep] for k,v in a.items()};m=m.loc[keep].reset_index(drop=True)
    truth=a['y'].copy()
    if stage=='evaluate':a['y'][m.origin>=pd.Timestamp('2026-10-01',tz='UTC')]=np.nan
    outputs={v:np.full_like(truth,np.nan) for v in variants};choices=[];tuning=[];timings=[]
    folds=[('june_july','2026-06-01','2026-08-01'),('august_september','2026-08-01','2026-10-01')]
    if stage=='evaluate':folds=[('october','2026-10-01','2026-10-27')]
    for ti in targets:
        for variant in variants:
            if ti!=0 and variant=='fresh_fleet':continue
            x=matrix(a,names,ti,study,variant)
            for area in AREAS:
                area_mask=m.area.eq(area).to_numpy()
                choices_here=[0,1] if stage=='select' else [next(r['setting'] for r in selections if r['area']==area and r['target']==TARGETS[ti] and r['variant']==variant)]
                candidates=[]
                for setting in choices_here:
                    out=np.full((len(m),48),np.nan)
                    for fold,start,end in folds:
                        start=pd.Timestamp(start,tz='UTC');end=pd.Timestamp(end,tz='UTC')
                        tr=((m.last_label<start)&area_mask).to_numpy();va=((m.origin>=start)&(m.last_label<end)&area_mask).to_numpy()
                        assert m.loc[tr,'last_label'].max()<m.loc[va,'origin'].min()
                        if tr.sum()<10:raise ValueError(f'Too few training origins: {study}/{area}/{fold}')
                        model=lgb.LGBMRegressor(**SETTINGS[setting],learning_rate=.04,reg_lambda=10,
                            objective='regression_l1',n_jobs=4,random_state=42,verbosity=-1,
                            deterministic=True,force_col_wise=True)
                        begin=time.perf_counter();model.fit(x[np.repeat(tr,48)],a['y'][tr,:,ti].ravel())
                        train_seconds=time.perf_counter()-begin;begin=time.perf_counter()
                        out[va]=np.maximum(0,model.predict(x[np.repeat(va,48)]).reshape(-1,48))
                        inference_seconds=time.perf_counter()-begin
                        timings.append(dict(study=study,variant=variant,area=area,target=TARGETS[ti],setting=setting,fold=fold,
                                            train_origins=int(tr.sum()),test_origins=int(va.sum()),
                                            train_seconds=train_seconds,inference_seconds=inference_seconds))
                        if stage=='evaluate':
                            folder=CACHE/'weather-models';folder.mkdir(exist_ok=True)
                            path=folder/f'{study}_{variant}_{area}_{TARGETS[ti]}.txt'
                            model.booster_.save_model(str(path))
                            reloaded=lgb.Booster(model_file=str(path))
                            check=np.maximum(0,reloaded.predict(x[np.repeat(va,48)],num_threads=4).reshape(-1,48))
                            np.testing.assert_allclose(check,out[va],rtol=1e-6,atol=1e-5)
                    valid=np.isfinite(out)
                    score=float(np.abs(out[valid]-truth[:,:,ti][valid]).mean()) if stage=='select' else 0
                    candidates.append((score,setting,out));tuning.append(dict(area=area,target=TARGETS[ti],variant=variant,setting=setting,development_mae=score if stage=='select' else None))
                score,setting,out=min(candidates,key=lambda c:c[0])
                outputs[variant][area_mask,:,ti]=out[area_mask]
                choices.append(dict(area=area,target=TARGETS[ti],variant=variant,setting=setting))
                print(study,stage,area,TARGETS[ti],variant,'setting',setting,'dev_mae',score if stage=='select' else 'not scored yet',flush=True)
    if 'fresh_fleet' in outputs:
        outputs['fresh_fleet'][:,:,1:]=outputs['fresh'][:,:,1:]
    rows=[]
    for variant,out in outputs.items():
        for area in AREAS:
            for ti in targets:
                valid=np.isfinite(out[:,:,ti]).all(1)&m.area.eq(area).to_numpy()
                y=truth[valid,:,ti];p=out[valid,:,ti];base=outputs['control'][valid,:,ti]
                for lo,hi in [(0,48),*BANDS]:
                    error=np.abs(y[:,lo:hi]-p[:,lo:hi]);old=np.abs(y[:,lo:hi]-base[:,lo:hi])
                    record=dict(study=study,area=area,target=TARGETS[ti],variant=variant,lead_start=lo+1,lead_end=hi,
                        origins=int(valid.sum()),mae_mw=float(error.mean()),wape_pct=float(error.sum()/y[:,lo:hi].sum()*100),
                        improvement_vs_control_pct=float((1-error.sum()/old.sum())*100))
                    if ti==1:
                        day=a['daylight'][valid,lo:hi].astype(bool);record['daylight_mae_mw']=float(error[day].mean())
                    rows.append(record)
    pd.DataFrame(rows).to_csv(HERE/f'{study}-{stage}.csv',index=False)
    write_json(HERE/f'{study}-{stage}-runtime.json',timings)
    write_json(HERE/f'{study}-{stage}-tuning.json',tuning)
    if stage=='select':
        if selection_path.exists():assert json.loads(selection_path.read_text())['selection']==choices,'Frozen selection changed'
        else:write_json(selection_path,dict(selection=choices,frozen_at=pd.Timestamp.now(tz='UTC').isoformat(),
            protocol_sha256=digest(HERE/'PROTOCOL.md'),regional_addendum_sha256=digest(HERE/'REGIONAL_ADDENDUM.md')))
    valid=np.logical_or.reduce([np.isfinite(v).any((1,2)) for v in outputs.values()])
    np.savez_compressed(CACHE/f'{study}-{stage}.npz',actual=truth[valid],**{k:v[valid] for k,v in outputs.items()})
    m.loc[valid].to_csv(HERE/f'{study}-{stage}-origins.csv',index=False)
    print(pd.DataFrame(rows).query('lead_start==1 and lead_end==48').to_string(index=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('study',choices=STUDIES);parser.add_argument('stage',choices=['select','evaluate']);args=parser.parse_args();main(args.study,args.stage)
