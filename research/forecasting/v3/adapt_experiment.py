"""Freeze Aug–Sep selections, then evaluate October separately."""
from research.forecasting.v3.common import *
import argparse
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from research.forecasting.v3.adaptation import fixed_weights,expand_weights,conditions,fit_gate,gate_predict,error_features,correction

def data():
    a,m,names=load_v2();p=np.full((*a['y'].shape,len(FAMILIES)),np.nan)
    pre=(m.origin<pd.Timestamp('2026-10-01',tz='UTC')).to_numpy()
    for j,family in enumerate(FAMILIES):
        p[pre,:,:,j]=np.load(V2/f'{family}_dev.npz')['prediction']
        p[~pre,:,:,j]=np.load(V2/f'{family}_final.npz')['prediction'][~pre]
    keep=np.isfinite(p).all((1,2,3))
    return {k:v[keep] for k,v in a.items()},m.loc[keep].reset_index(drop=True),names,p[keep]

def fit_corrector(option,features,actual,baseline,levels,train):
    if not option.startswith('ridge'):return None
    model=make_pipeline(StandardScaler(),Ridge(alpha=float(option[5:])))
    residual=(actual-baseline)/levels[:,None]
    model.fit(features[train].reshape(-1,4),residual[train].ravel())
    return model

def main(stage):
    a,m,names,p=data();condition=conditions(a['future'],names['future'])
    boundary=pd.Timestamp('2026-08-01' if stage=='select' else '2026-10-01',tz='UTC')
    end=pd.Timestamp('2026-10-01' if stage=='select' else '2026-10-27',tz='UTC')
    train=(m.last_label<boundary).to_numpy();test=((m.origin>=boundary)&(m.last_label<end)).to_numpy()
    if stage=='select':
        # No October labels enter any development process, even as unused arrays.
        a['y'][m.origin>=pd.Timestamp('2026-10-01',tz='UTC')]=np.nan
    prior=json.loads((V2_CODE/'final_selection.json').read_text())
    selected=json.loads((HERE/'adapt-selection.json').read_text()) if stage=='evaluate' else None
    options=['none','ewma2','ewma7','ridge10','ridge100']
    rows=[];choices=[];all_out={};runtimes=[]
    for area in AREAS:
        mask=m.area.eq(area).to_numpy();indices=np.flatnonzero(mask);mt=m.loc[mask].reset_index(drop=True)
        tr=train[mask];te=test[mask]
        for ti,target in enumerate(TARGETS):
            begin=time.perf_counter();families=p[mask,:,ti];truth=a['y'][mask,:,ti];c=condition[mask]
            levels=np.maximum(a['history'][mask,:,ti].mean(1),1)
            if stage=='select':weights=fixed_weights(families[tr],truth[tr])
            else:weights=np.asarray(next(r['weights'] for r in prior['selection'] if r['area']==area and r['target']==target))
            baseline=(expand_weights(weights,len(families))*families).sum(-1)
            inputs=error_features(mt.origin,baseline,truth,levels)
            outputs={'fixed':baseline};correctors={};gates={}
            chosen=next(r for r in selected['selection'] if r['area']==area and r['target']==target) if selected else None
            for option in options if not chosen else [chosen['correction']]:
                model=fit_corrector(option,inputs,truth,baseline,levels,tr);correctors[option]=model
                outputs[f'correction_{option}']=correction(baseline,inputs,levels,option,model)
            for penalty in [0.1,1.0] if not chosen else [chosen['penalty']]:
                gate=fit_gate(families[tr],truth[tr],c[tr],weights,penalty);gates[penalty]=gate
                outputs[f'gate_{penalty}']=gate_predict(gate,families,c,weights)
            if stage=='select':
                best_c=min(options,key=lambda x:np.abs(outputs[f'correction_{x}'][te]-truth[te]).mean())
                best_g=min(gates,key=lambda x:np.abs(outputs[f'gate_{x}'][te]-truth[te]).mean())
                chosen=dict(area=area,target=target,correction=best_c,penalty=best_g)
                choices.append(chosen)
            gate_output=outputs[f'gate_{chosen["penalty"]}']
            # Combined uses errors of its own gate, rather than correcting a
            # changed prediction using residuals of a different forecast.
            gate_inputs=error_features(mt.origin,gate_output,truth,levels)
            combo_model=fit_corrector(chosen['correction'],gate_inputs,truth,gate_output,levels,tr)
            outputs['combined']=correction(gate_output,gate_inputs,levels,chosen['correction'],combo_model)
            outputs['selected_correction']=outputs[f'correction_{chosen["correction"]}']
            outputs['selected_gate']=gate_output
            for family,out in outputs.items():
                if not np.isfinite(out[te]).all():raise ValueError('Nonfinite prediction')
                error=np.abs(out[te]-truth[te]);old=np.abs(baseline[te]-truth[te])
                for lo,hi in [(0,48),*BANDS]:
                    rows.append(dict(area=area,target=target,candidate=family,lead_start=lo+1,lead_end=hi,
                        origins=int(te.sum()),mae_mw=float(error[:,lo:hi].mean()),
                        wape_pct=float(error[:,lo:hi].sum()/truth[te,lo:hi].sum()*100),
                        improvement_pct=float((1-error[:,lo:hi].sum()/old[:,lo:hi].sum())*100)))
            if stage=='evaluate':
                folder=CACHE/'adapt-models';folder.mkdir(exist_ok=True)
                write_json(folder/f'{area}_{target}_gate.json',dict(gate=gates[chosen['penalty']],weights=weights.tolist(),selection=chosen))
                joblib.dump(correctors[chosen['correction']],folder/f'{area}_{target}_correction.joblib')
                joblib.dump(combo_model,folder/f'{area}_{target}_combined.joblib')
                for name in ['fixed','selected_correction','selected_gate','combined']:
                    if name not in all_out:all_out[name]=np.full_like(a['y'],np.nan)
                    all_out[name][indices[te],:,ti]=outputs[name][te]
            runtimes.append(dict(area=area,target=target,seconds=time.perf_counter()-begin))
            print(stage,area,target,chosen,flush=True)
    pd.DataFrame(rows).to_csv(HERE/f'adapt-{stage}.csv',index=False)
    write_json(HERE/f'adapt-{stage}-runtime.json',runtimes)
    if stage=='select':
        path=HERE/'adapt-selection.json'
        payload=dict(selection=choices,protocol_sha256=digest(HERE/'PROTOCOL.md'),
                     frozen_at=pd.Timestamp.now(tz='UTC').isoformat())
        if path.exists():assert json.loads(path.read_text())['selection']==choices,'Frozen selection changed'
        else:write_json(path,payload)
    else:
        np.savez_compressed(CACHE/'adapt-evaluation.npz',actual=a['y'][test],**{k:v[test] for k,v in all_out.items()})
        m.loc[test].to_csv(HERE/'adapt-origins.csv',index=False)
    print(pd.DataFrame(rows).query('lead_end==48 and lead_start==1').to_string(index=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['select','evaluate']);args=parser.parse_args();main(args.stage)
