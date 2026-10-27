"""Freeze development selection first; evaluate October in a separate call."""
from common import *
import argparse
import datetime as dt
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize

FAMILIES=['frozen','tree_control','ridge','lightgbm','xgboost','chronos_zero','chronos','ttm','ttm_scaled']
BANDS=[(0,6),(6,24),(24,48)]

def development_mask(m):
    return np.logical_or.reduce([masks(m,start,end)[1] for _,start,end in FOLDS])

def metrics(y,p):
    return dict(mae_mw=float(np.abs(y-p).mean()),wape_pct=float(np.abs(y-p).sum()/y.sum()*100))

def select():
    a,m,names=load_data()
    keep=(m.origin<pd.Timestamp('2026-10-01',tz='UTC')).to_numpy()
    y=a['y'][keep];m=m.loc[keep].reset_index(drop=True)
    pred={f:np.load(CACHE/f'{f}_dev.npz')['prediction'] for f in FAMILIES}
    valid=development_mask(m)
    for f,p in pred.items():
        assert p.shape==y.shape and np.isfinite(p[valid]).all(),f
    rows=[];selection=[];blended=np.full_like(y,np.nan)
    for area in AREAS:
        mask=valid & m.area.eq(area).to_numpy()
        for ti,target in enumerate(TARGETS):
            for f,p in pred.items():
                rows.append(dict(area=area,target=target,family=f,**metrics(y[mask,:,ti],p[mask,:,ti])))
            candidates=[r for r in rows if r['area']==area and r['target']==target]
            best=min(candidates,key=lambda r:r['mae_mw'])
            weights=[]
            for lo,hi in BANDS:
                x=np.stack([p[mask,lo:hi,ti].ravel() for p in pred.values()],axis=1)
                truth=y[mask,lo:hi,ti].ravel()
                scale=max(float(truth.mean()),1)
                equal=np.full(len(FAMILIES),1/len(FAMILIES))
                fit=minimize(lambda w:np.abs(x@w-truth).mean()/scale+.01*np.square(w-equal).sum(),
                    equal,method='SLSQP',bounds=[(0,1)]*len(FAMILIES),
                    constraints={'type':'eq','fun':lambda w:w.sum()-1},
                    options={'maxiter':500,'ftol':1e-9})
                assert fit.success,fit.message
                w=np.clip(fit.x,0,1);w/=w.sum();weights.append(w.tolist())
                blended[mask,lo:hi,ti]=(x@w).reshape(-1,hi-lo)
            bm=metrics(y[mask,:,ti],blended[mask,:,ti])
            rows.append(dict(area=area,target=target,family='blend',**bm))
            selection.append(dict(area=area,target=target,best_single=best['family'],
                selected='blend' if bm['mae_mw']<best['mae_mw'] else best['family'],
                single_mae=best['mae_mw'],blend_mae=bm['mae_mw'],weights=weights))
    pd.DataFrame(rows).to_csv(HERE/'development_metrics.csv',index=False)
    payload=dict(frozen_at=dt.datetime.now(dt.timezone.utc).isoformat(),families=FAMILIES,bands=BANDS,selection=selection,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')},
        prediction_sha256={f:hashlib.sha256((CACHE/f'{f}_dev.npz').read_bytes()).hexdigest() for f in FAMILIES})
    path=HERE/'final_selection.json'
    if path.exists():
        previous=json.loads(path.read_text())
        assert previous['selection']==selection,'Refusing to overwrite different frozen selection'
    else:
        path.write_text(json.dumps(payload,indent=2))
    print(pd.DataFrame(selection)[['area','target','best_single','selected','single_mae','blend_mae']].to_string(index=False))

def evaluate():
    frozen=json.loads((HERE/'final_selection.json').read_text())
    a,m,names=load_data()
    test=(m.origin>=pd.Timestamp('2026-10-01',tz='UTC')).to_numpy()
    y=a['y'][test];mt=m.loc[test].reset_index(drop=True)
    pred={f:np.load(CACHE/f'{f}_final.npz')['prediction'][test] for f in FAMILIES}
    pred['persistence']=np.repeat(a['history'][test,-1:,:],H,axis=1)
    pred['last_day']=a['history'][test,-24:,:][:,np.arange(H)%24]
    pred['last_week']=a['history'][test,:H,:]
    pred['blend']=np.zeros_like(y);pred['selected']=np.zeros_like(y)
    for s in frozen['selection']:
        mask=mt.area.eq(s['area']).to_numpy();ti=TARGETS.index(s['target'])
        for (lo,hi),weights in zip(BANDS,s['weights'],strict=True):
            pred['blend'][mask,lo:hi,ti]=sum(w*pred[f][mask,lo:hi,ti] for f,w in zip(FAMILIES,weights,strict=True))
        pred['selected'][mask,:,ti]=pred[s['selected']][mask,:,ti]
    for f,p in pred.items():
        assert np.isfinite(p).all(),f
    rows=[];summary=[];leadrows=[]
    rng=np.random.default_rng(42)
    for area in AREAS:
        mask=mt.area.eq(area).to_numpy()
        origins=mt.loc[mask,'origin']
        weeks=origins.dt.strftime('%G-%V').to_numpy();unique=np.unique(weeks)
        for ti,target in enumerate(TARGETS):
            truth=y[mask,:,ti]
            for f,p in pred.items():
                row=dict(area=area,target=target,family=f,origins=int(mask.sum()),**metrics(truth,p[mask,:,ti]))
                if ti==1:
                    day=a['daylight'][test][mask].astype(bool)
                    row['daylight_mae_mw']=float(np.abs(truth-p[mask,:,ti])[day].mean())
                rows.append(row)
                for lo,hi in BANDS:
                    leadrows.append(dict(area=area,target=target,family=f,lead_start=lo+1,lead_end=hi,**metrics(truth[:,lo:hi],p[mask,lo:hi,ti])))
            new=np.abs(truth-pred['selected'][mask,:,ti]);old=np.abs(truth-pred['frozen'][mask,:,ti])
            control=np.abs(truth-pred['tree_control'][mask,:,ti])
            bootstrap=[]
            for _ in range(5000):
                ids=np.concatenate([np.flatnonzero(weeks==w) for w in rng.choice(unique,len(unique),replace=True)])
                bootstrap.append((1-new[ids].sum()/old[ids].sum())*100)
            lo,hi=np.quantile(bootstrap,[.025,.975])
            s=next(r for r in frozen['selection'] if r['area']==area and r['target']==target)
            summary.append(dict(area=area,target=target,selected=s['selected'],best_single=s['best_single'],
                **metrics(truth,pred['selected'][mask,:,ti]),old_mae_mw=float(old.mean()),control_mae_mw=float(control.mean()),
                improvement_vs_original_pct=float((1-new.sum()/old.sum())*100),
                improvement_vs_refit_control_pct=float((1-new.sum()/control.sum())*100),
                approximate_skill_ci_low=float(lo),approximate_skill_ci_high=float(hi),weeks=len(unique)))
    frame=pd.DataFrame(rows);summ=pd.DataFrame(summary)
    frame.to_csv(HERE/'holdout_metrics.csv',index=False)
    summ.to_csv(HERE/'selected_metrics.csv',index=False)
    pd.DataFrame(leadrows).to_csv(HERE/'lead_metrics.csv',index=False)
    np.savez_compressed(CACHE/'evaluated_holdout.npz',actual=y,**pred)
    mt.to_csv(HERE/'holdout_origins.csv',index=False)
    (HERE/'evaluation_audit.json').write_text(json.dumps(dict(evaluated_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        frozen_selection_sha256=hashlib.sha256((HERE/'final_selection.json').read_bytes()).hexdigest(),
        first_origin=str(mt.origin.min()),last_origin=str(mt.origin.max()),area_origins=len(mt)),indent=2))
    plot(frame,summ,pred,y,mt)
    report(frame,summ,frozen,mt)
    print(summ.to_string(index=False))

def plot(frame,summary,pred,y,m):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    families=['frozen','tree_control','lightgbm','xgboost','chronos_zero','chronos','ttm','ttm_scaled','selected']
    labels=['Original','Refitted control','LightGBM','XGBoost','Chronos zero-shot','Chronos LoRA','TTM raw inputs','TTM scaled inputs','Selected forecast']
    for ai,area in enumerate(AREAS):
        for ti,target in enumerate(TARGETS):
            ax=axes[ai,ti]
            part=frame[frame.area.eq(area)&frame.target.eq(target)].set_index('family')
            values=part.loc[families,'mae_mw']
            ax.barh(labels,values,color=['#9ca3af','#6b7280','#05847c','#27a99b','#9a8dbe','#7555a4','#e8c286','#d89639','#102d47'])
            ax.invert_yaxis();ax.set_title(f'{area} · {target}');ax.set_xlabel('Mean absolute error (MW) ↓')
            if ti:
                ax.set_yticklabels([])
    fig.suptitle('October holdout: same 13 forecast dates per region',fontsize=16)
    fig.savefig(HERE/'comparison.png',dpi=160);fig.savefig(HERE/'comparison.svg');plt.close(fig)
    # Deterministic first available date, never chosen by forecast quality.
    fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
    for ai,area in enumerate(AREAS):
        i=np.flatnonzero(m.area.eq(area).to_numpy())[0]
        for ti,target in enumerate(TARGETS):
            ax=axes[ai,ti]
            ax.plot(np.arange(1,49),y[i,:,ti],color='#102d47',label='Observed')
            ax.plot(np.arange(1,49),pred['selected'][i,:,ti],color='#05847c',label='Selected')
            ax.plot(np.arange(1,49),pred['frozen'][i,:,ti],color='#9ca3af',ls='--',label='Original')
            ax.set_title(f'{area} · {target}');ax.set_xlabel('Hours ahead');ax.set_ylabel('MW')
    axes[0,0].legend();fig.suptitle(f'First eligible October origin: {m.origin.min().date()} (not cherry-picked)',fontsize=15)
    fig.savefig(HERE/'forecast_example.png',dpi=160);plt.close(fig)

def report(frame,summary,frozen,meta):
    def table(headers,rows):
        return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                         ['| '+' | '.join(str(c) for c in r)+' |' for r in rows])
    summary_table=table(['Area','Target','Original MAE','New MAE','New WAPE','Error reduction vs original','Gain vs refitted control'],
        [[r.area,r.target,f'{r.old_mae_mw:.1f} MW',f'{r.mae_mw:.1f} MW',f'{r.wape_pct:.1f}%',
          f'{r.improvement_vs_original_pct:+.1f}%',f'{r.improvement_vs_refit_control_pct:+.1f}%'] for r in summary.itertuples()])
    family_table=table(['Family','DK1 wind','DK1 solar','DK1 load','DK2 wind','DK2 solar','DK2 load'],
        [[f]+[f'{frame.loc[frame.family.eq(f)&frame.area.eq(area)&frame.target.eq(target),"mae_mw"].iloc[0]:.1f}'
        for area in AREAS for target in TARGETS] for f in frame.family.unique()])
    times=[]
    for family in ['tree_control','ridge','lightgbm','xgboost','chronos','ttm','ttm_scaled']:
        path=HERE/f'{family}_final_runtime.json'
        if not path.exists():continue
        data=json.loads(path.read_text())
        train=sum(r.get('seconds',r.get('training_seconds',0))+r.get('backbone_cache_seconds',0) for r in data)
        infer=sum(r.get('warm_inference_seconds',r.get('warm_inference_seconds_two_areas',0)) for r in data)
        times.append([family,f'{train:.1f}',f'{infer:.3f}'])
    csel=json.loads((HERE/'chronos_selection.json').read_text())
    tsel=json.loads((HERE/'ttm_selection.json').read_text())
    tscaled=json.loads((HERE/'ttm_scaled_selection.json').read_text())
    neural_table=table(['Candidate','Development mean WAPE'],
        [[f'Chronos LoRA {k} steps',f'{v*100:.2f}%'] for k,v in csel['scores'].items()]+
        [[f'TTM raw lr/epoch {k}',f'{v*100:.2f}%'] for k,v in tsel['scores'].items()]+
        [[f'TTM scaled lr/epoch {k}',f'{v*100:.2f}%'] for k,v in tscaled['scores'].items()])
    text=f'''# Second forecasting experiment

All three requested rounds were trained and evaluated locally: improved
LightGBM/XGBoost, weather-conditioned Chronos-2 LoRA, and pretrained
TinyTimeMixer with a fine-tuned weather/channel-mixing head.

## Fresh October holdout

Settings and ensemble weights were frozen at {frozen['frozen_at']} before
this evaluation. The original June–September test is now development data.
October has **13 eligible dates per region**, 48 forecast hours each. The
forecasts overlap, so those hours are not independent observations.

{summary_table}

MAE is mean absolute error in MW. WAPE is total absolute error divided by total
observed output/load; it is not classification accuracy. Positive gains mean
less error. The refitted control uses the original tree features with updated
training data, so it is a stronger comparison than the frozen original models.

## Every candidate, including losses

All entries below are October MAE in MW. The selected forecast uses development
selection, not whichever candidate happened to win October.

{family_table}

Development selected **{csel['steps']} Chronos steps** and **{tsel['epochs']} TTM
epochs at learning rate {tsel['learning_rate']}**. These scores average WAPE
equally across the six area/target combinations; they are not October scores.
The scaled-input TTM variant selected {tscaled['epochs']} epochs at learning
rate {tscaled['learning_rate']}; its addition during development is recorded
in the implementation notes.

{neural_table}

![Holdout model comparison](comparison.png)

![First eligible example](forecast_example.png)

## Local CPU cost

{table(['Family','Final training seconds','Warm inference seconds, all six series'],times)}

Measured on the local Mac with four CPU threads. Some training ran concurrently;
these are elapsed timings, not isolated hardware benchmarks. Inference excludes
Python imports, first model download and API requests. GitHub's Linux runners
have not been benchmarked. Tuning costs and memory are in the runtime JSON files.

## What this experiment can establish

- Two expanding validation folds: train through May for June–July; train through
  July for August–September. Labels crossing each boundary are excluded. Final
  models train on eligible labels before October. Selection is per region/target.
- Matched LightGBM/XGBoost absolute-error objectives and four configurations
  each; compare shallow/regularized vs larger trees and optional recency weights.
  Extra inputs summarize recent weather-to-output conversion and cross-target
  history. Ridge and the unchanged-feature tree are controls.
- Chronos sees the same per-location weather signals rather than the earlier
  averaged covariates. LoRA updates about 1.2 million parameters. Its fine-tuning
  keeps each forecast origin fixed. TinyTimeMixer uses 512 actual historical
  hours and freezes the pretrained backbone while training decoder/weather head.
- The convex ensemble has separate weights for 1–6, 7–24 and 25–48 hours.
  Weights and family selection reuse development predictions; the independent
  October holdout tests that decision. They are not selected on October errors.
- All future weather comes from archived model runs. Neither neural model is
  given future electricity observations as input. Reload and leakage checks run
  during training; original artifact hashes are retained.

## Limits

The small October sample does not prove a year-round improvement. Descriptive
paired calendar-week bootstrap intervals in `selected_metrics.csv` have very
few clusters and cannot establish robust statistical significance. Missing
SCADA hours exclude overlapping windows, so results are conditional on complete
input availability. Older TTM gaps are causally filled with observation masks.

Weather run publication timing is assumed, not historically timestamped. SCADA
is a later download of an operational feed, not a preserved real-time vintage;
derived system load includes losses. Weather locations remain the original six
points and are not generation-capacity weighted. This is one daily origin, not
a validation of half-hourly operation. Intervals have not been recalibrated.
Each setting has one seeded training run; the bootstrap does not measure
variation between training seeds. The original tree-versus-neural comparison
also differs in history representation (168 versus TTM's 512 hours), so this
compares practical pipelines rather than isolating architecture alone.

No models or site changes have been pushed or deployed. See [protocol](PROTOCOL.md),
[implementation notes](IMPLEMENTATION_NOTES.md), and [reproduction instructions](README.md).

Primary implementation references: [LightGBM](https://lightgbm.readthedocs.io/en/latest/Features.html),
[XGBoost](https://xgboost.readthedocs.io/en/stable/parameter.html),
[Chronos-2](https://huggingface.co/amazon/chronos-2),
[TinyTimeMixer R2](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2).
'''
    (HERE/'RESULTS.md').write_text(text)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['select','evaluate']);args=p.parse_args()
    select() if args.stage=='select' else evaluate()
