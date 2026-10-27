"""Refresh source history, score the fixed ensemble, retain issued forecasts."""
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from . import data
from . import intervals
from .artifacts import ensure,CONFIG,ROOT
from .publish import write_public
from .score import run as score

def main():
    p=argparse.ArgumentParser();p.add_argument('--restore',action='store_true');p.add_argument('--strict',action='store_true');p.add_argument('--offline',action='store_true')
    args=p.parse_args();at=data.now();state=data.read_state();version=CONFIG['version']
    if args.restore:
        # A restore failure must not reset the issued-forecast archive.
        state=data.restore(os.environ.get('SNAPSHOT_URL','https://huayuansong.github.io/open-current/'))
    error=None
    try:
        if not args.offline:
            data.electricity(state,at);data.weather(state,at)
        intervals.refresh_history(state,at)
        arrays,meta=data.inputs(state,at)
        bundle=ensure();names=json.loads((bundle/'feature_names.json').read_text())
        if meta.pop('featureNames')!=names['future']:raise ValueError('Feature schema differs from trained models')
        range_version=json.loads(intervals.CALIBRATION.read_text())['version']
        identifier=f'{version}/{meta["origin"]}/{meta["weatherRun"]}/{range_version}'
        if not any(r['id']==identifier for r in state['forecasts']):
            work=ROOT/'.cache/forecast-serving/run';work.mkdir(parents=True,exist_ok=True)
            path=work/'inputs.npz';np.savez_compressed(path,**arrays)
            prediction,families,timings=score(path,work,explain=True)
            # An operational plausibility guard, not a correction to predictions.
            limits=np.array([[15000,15000,20000],[10000,10000,15000]])
            if (prediction<0).any() or (prediction>limits[:,None,:]).any():
                raise ValueError('Forecast failed the physical plausibility check')
            record={'id':identifier,'modelVersion':version,'issuedAt':data.now().isoformat(),**meta,
                'areas':{area:{target:prediction[ai,:,ti].astype(float).tolist() for ti,target in enumerate(data.TARGETS)} for ai,area in enumerate(data.AREAS)},
                'treeControl':{area:{target:families['tree_control'][ai,:,ti].astype(float).tolist() for ti,target in enumerate(data.TARGETS)} for ai,area in enumerate(data.AREAS)},
                'runtime':timings}
            sun=[i for i,n in enumerate(names['future']) if n.endswith('terrestrial_radiation')]
            daylight=arrays['future'][:,:,sun].mean(-1)>5
            record['daylight']={area:daylight[ai].tolist() for ai,area in enumerate(data.AREAS)}
            record['ranges']=intervals.build(prediction,daylight,record['issuedAt'],state)
            # A late source hour must not acquire retrospective prediction ranges.
            for i,stamp in enumerate(record['times']):
                if pd.Timestamp(stamp)<=pd.Timestamp(record['issuedAt']):
                    for targets in record['ranges']['areas'].values():
                        for bounds in targets.values():bounds['lower'][i]=None;bounds['upper'][i]=None
            record['explanations']=json.loads((work/'trees.explanations.json').read_text())
            state['forecasts'].append(record)
            print(f'Issued {identifier}',flush=True)
        else:
            print('Forecast for this origin already issued; preserving its original values.',flush=True)
    except Exception as exc:
        # Keep prior forecasts; never silently drop a model or change blend weights.
        error=str(exc)[:240]
        print(f'Forecast update unavailable: {error}',file=sys.stderr,flush=True)
    checked=data.now();state['updatedAt']=checked.isoformat()
    state['forecasts']=[r for r in state['forecasts'] if pd.Timestamp(r['issuedAt'])>=checked-pd.Timedelta(days=35)]
    data.write_state(state);public=write_public(state,checked,version,error)
    print(json.dumps({'status':public['status'],'forecastsRetained':len(state['forecasts']),
        'origin':public['latest']['origin'] if public['latest'] else None,'error':error}),flush=True)
    if error and args.strict:raise SystemExit(1)

if __name__=='__main__':main()
