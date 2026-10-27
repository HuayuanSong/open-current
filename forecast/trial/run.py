"""Record prospective daily challengers and update their paired evaluation."""
import argparse
import gzip
import json
import os
import sys
import numpy as np
import pandas as pd
from forecast import data
from forecast.artifacts import CONFIG as BASELINE
from forecast.score import run as baseline_score
from forecast.features import AREAS,TARGETS
from .artifacts import ROOT,ensure,digest
from .ledger import initialize,eligible_origin,append,freeze_observations,attempt
from .evaluate import metrics
from .sources import inputs as weather_inputs
from .score import predict

PROTOCOL=ROOT/'forecast/trial/protocol.json'
SUMMARY=ROOT/'dist/data/forecast-trial.json'
WORK=ROOT/'.cache/forecast-serving/trial'

def serialize(prediction):
    return {area:{target:prediction[ai,:,ti].astype(float).tolist() if np.isfinite(prediction[ai,:,ti]).all() else None
                  for ti,target in enumerate(TARGETS)} for ai,area in enumerate(AREAS)}

def issue(state,trial,at):
    p=trial['protocol'];origin=eligible_origin(at,p)
    if origin is None or any(pd.Timestamp(r['origin'])==origin for r in trial['records']):return False
    if BASELINE['version']!=p['baselineVersion']:raise ValueError('Public baseline model changed during the trial')
    arrays,meta=data.inputs(state,at)
    if pd.Timestamp(meta['origin'])!=origin:raise ValueError('Waiting for the complete 18 UTC electricity hour')
    bundle=ensure();names=json.loads((bundle/'feature-names.json').read_text())
    if meta.pop('featureNames')!=names['future']:raise ValueError('Trial feature schema changed')
    WORK.mkdir(parents=True,exist_ok=True)
    enriched,snapshots,available,errors=weather_inputs(arrays,meta,bundle,p)
    np.savez_compressed(WORK/'inputs.npz',**enriched)
    with gzip.open(WORK/'sources.json.gz','wt') as file:json.dump(snapshots,file,separators=(',',':'),allow_nan=False)
    # Recompute the unchanged production ensemble on this exact captured input
    # snapshot. A previously issued public forecast might have used older data.
    baseline,_,timings=baseline_score(WORK/'inputs.npz',WORK/'baseline',explain=False)
    predictions=predict(enriched,bundle,p,available);predictions['baseline']=baseline
    limits=np.array([[15000,15000,20000],[10000,10000,15000]])[:,None,:]
    for name,values in predictions.items():
        finite=np.isfinite(values)
        if np.any(np.where(finite,values,0)>limits) or np.any(values[finite]<0):raise ValueError(f'Implausible trial prediction: {name}')
    issued=data.now()
    source_meta={key:{k:v for k,v in raw.items() if k not in ['data','points']} for key,raw in snapshots.items()}
    sun=[i for i,n in enumerate(names['future']) if n.endswith('terrestrial_radiation')]
    daylight=arrays['future'][:,:,sun].mean(-1)>5
    record=dict(id=p['version']+'/'+origin.isoformat(),trialVersion=p['version'],baselineVersion=BASELINE['version'],
        issuedAt=issued.isoformat(),**meta,predictions={k:serialize(v) for k,v in predictions.items()},
        daylight={area:daylight[ai].tolist() for ai,area in enumerate(AREAS)},
        sources=source_meta,sourceErrors=errors,inputSha256=digest(WORK/'inputs.npz'),sourceSnapshotSha256=digest(WORK/'sources.json.gz'),
        modelRelease=json.loads((ROOT/'forecast/trial/model-release.json').read_text()),baselineRuntime=timings,
        githubRunId=os.environ.get('GITHUB_RUN_ID'),codeRevision=os.environ.get('GITHUB_SHA'))
    # Preserve the full baseline 00-run weather used to build past/future inputs
    # in a separate audit file. All values are public weather/electricity data.
    with gzip.open(WORK/'baseline-context.json.gz','wt') as file:
        json.dump({'weather':state['weather'],'electricityRetrievedAt':state.get('electricityRetrievedAt')},file,separators=(',',':'),allow_nan=False)
    record['baselineContextSha256']=digest(WORK/'baseline-context.json.gz')
    added=append(trial,record,issued)
    if added:(WORK/'record.json').write_text(json.dumps(trial['records'][-1],separators=(',',':'),allow_nan=False)+'\n')
    return added

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--strict',action='store_true');args=parser.parse_args()
    protocol=json.loads(PROTOCOL.read_text());state=data.read_state()
    if not state.get('updatedAt') or not state.get('hourly'):raise ValueError('Restore and refresh the production state before running the trial')
    trial=initialize(state,protocol);at=data.now();added=False;error=None
    freeze_observations(trial,state['hourly'],at)
    try:
        added=issue(state,trial,at)
        if added:attempt(trial,data.now(),'issued')
    except Exception as exc:
        error=str(exc)[:240];attempt(trial,data.now(),'unavailable',error)
        print('Trial issuance unavailable: '+error,file=sys.stderr,flush=True)
    checked=data.now();state['updatedAt']=checked.isoformat();data.write_state(state)
    summary=metrics(trial,checked)
    summary['protocolUrl']='https://github.com/HuayuanSong/open-current/blob/main/forecast/trial/README.md'
    summary['protocolHash']=trial['protocolHash']
    temp=SUMMARY.with_suffix('.tmp');temp.write_text(json.dumps(summary,separators=(',',':'),allow_nan=False)+'\n');temp.replace(SUMMARY)
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'],'a') as file:file.write(f'issued={str(added).lower()}\n')
    print(json.dumps(dict(status=summary['status'],issuedNow=added,issuedOrigins=summary['issuedOrigins'],reviewAfter=summary['reviewAfter'],error=error)),flush=True)
    if error and args.strict:raise SystemExit(1)

if __name__=='__main__':main()
