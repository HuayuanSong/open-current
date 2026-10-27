"""Matched-budget tree comparison with expanding temporal validation."""
from common import *
import argparse
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from build_data import tabular_features

SETTINGS = [dict(rounds=400, leaves=15, depth=4, minimum=100, half_life=None),
            dict(rounds=800, leaves=31, depth=5, minimum=60, half_life=None),
            dict(rounds=400, leaves=15, depth=4, minimum=100, half_life=180),
            dict(rounds=800, leaves=31, depth=5, minimum=60, half_life=180)]

def features(a, names, ti, improved=True):
    x, extras = tabular_features(a['history'], a['future'], ti)
    if not improved:
        return x
    added = []
    for target in range(3):
        for hours in [24, 168]:
            added += [np.repeat(a['history'][:, -hours:, target].mean(1)[:,None], H, 1),
                      np.repeat(a['history'][:, -hours:, target].std(1)[:,None], H, 1)]
    # A rolling conversion from prior issued weather to observed output. Each
    # slope only uses the history available at this origin, never future labels.
    proxy = 'power_curve' if ti == 0 else 'pv_temperature'
    if ti in [0, 1]:
        columns = [i for i,n in enumerate(names['future']) if n.endswith(proxy)]
        oldwx = np.nanmean(a['past_full512'][:, -168:, columns], axis=-1)
        futurewx = a['future'][:, :, columns].mean(-1)
        for hours in [24, 168]:
            wx = oldwx[:, -hours:]
            y = a['history'][:, -hours:, ti]
            usable = np.isfinite(wx) & (wx > (0.05 if ti == 0 else 50))
            numerator = np.where(usable, y*wx, 0).sum(1)
            denominator = np.where(usable, wx*wx, 0).sum(1)
            slope = numerator / np.maximum(denominator, 1e-6)
            added.append(futurewx*slope[:,None])
            added.append(np.repeat(slope[:,None], H, 1))
    return np.concatenate([x, np.stack(added, -1).reshape(len(x), -1)], axis=1)

def create(family, setting):
    if family == 'ridge':
        return make_pipeline(StandardScaler(), Ridge(alpha=[1,100,10000][setting]))
    s = SETTINGS[setting]
    if family in ['lightgbm', 'tree_control']:
        import lightgbm as lgb
        return lgb.LGBMRegressor(n_estimators=s['rounds'], num_leaves=s['leaves'],
            min_child_samples=s['minimum'], learning_rate=.04, reg_lambda=10,
            objective='regression_l1', n_jobs=4, random_state=42, verbosity=-1,
            deterministic=True, force_col_wise=True)
    import xgboost as xgb
    return xgb.XGBRegressor(n_estimators=s['rounds'], max_depth=s['depth'],
        min_child_weight=s['minimum'], learning_rate=.04, reg_lambda=10,
        objective='reg:absoluteerror', tree_method='hist', device='cpu', n_jobs=4,
        random_state=42)

def save(model, family, path):
    if family == 'xgboost':
        model.save_model(path.with_suffix('.ubj'))
    elif family == 'ridge':
        joblib.dump(model, path.with_suffix('.joblib'))
    else:
        model.booster_.save_model(str(path.with_suffix('.txt')))

def main(family, final):
    a, m, names = load_data()
    if not final:
        # Training/tuning process never receives the held-out targets.
        keep = (m.origin < pd.Timestamp('2026-10-01', tz='UTC')).to_numpy()
        a = {k:v[keep] for k,v in a.items()}; m=m.loc[keep].reset_index(drop=True)
    else:
        a['y'] = a['y'].copy()
        a['y'][m.origin >= pd.Timestamp('2026-10-01', tz='UTC')] = np.nan
    result = np.full_like(a['y'], np.nan)
    selection, tuning, runtime = [], [], []
    frozen = json.loads((HERE / f'{family}_selection.json').read_text()) if final else None
    for ti,target in enumerate(TARGETS):
        x = features(a, names, ti, improved=family not in ['tree_control','ridge'])
        truth = a['y'][:,:,ti].ravel()
        for ai,area in enumerate(AREAS):
            area_mask = m.area.eq(area).to_numpy()
            if final:
                options = [next(r['setting'] for r in frozen if r['area']==area and r['target']==target)]
                folds = [('final','2026-10-01','2026-10-27')]
            else:
                options = list(range(3 if family=='ridge' else (2 if family=='tree_control' else 4)))
                folds=FOLDS
            candidates=[]
            for number in options:
                out = np.full((len(m),H),np.nan,np.float32)
                for fold,start,end in folds:
                    tr, va = masks(m,start,end)
                    train=np.repeat(tr & area_mask,H); valid=va & area_mask
                    model=create(family,number)
                    kwargs={}
                    if family!='ridge' and SETTINGS[number]['half_life']:
                        days=(pd.Timestamp(start,tz='UTC')-m.origin).dt.total_seconds().to_numpy()/86400
                        kwargs['sample_weight']=np.repeat(np.power(.5,days/SETTINGS[number]['half_life']),H)[train]
                    begin=time.perf_counter();model.fit(x[train],truth[train],**kwargs)
                    seconds=time.perf_counter()-begin
                    out[valid]=np.maximum(0,model.predict(x[np.repeat(valid,H)]).reshape(-1,H))
                    runtime.append(dict(family=family,area=area,target=target,setting=number,fold=fold,seconds=seconds))
                    if final:
                        path=MODELS/f'{family}_{area}_{target}'
                        save(model,family,path)
                        t=[]
                        for _ in range(11):
                            begin=time.perf_counter();model.predict(x[np.repeat(valid,H)][:H]);t.append(time.perf_counter()-begin)
                        runtime[-1]['warm_inference_seconds']=float(np.median(t[1:]))
                finite=np.isfinite(out)
                score=float(np.abs(out[finite]-a['y'][:,:,ti][finite]).mean()) if not final else None
                candidates.append((score,number,out))
                tuning.append(dict(area=area,target=target,setting=number,mae=score))
                print(f'{family} {area}/{target} setting={number} dev_mae={score}',flush=True)
            score,number,out=min(candidates,key=lambda c:c[0]) if not final else candidates[0]
            result[area_mask,:,ti]=out[area_mask]
            selection.append(dict(area=area,target=target,setting=number,mae=score))
    stage='final' if final else 'dev'
    np.savez_compressed(CACHE/f'{family}_{stage}.npz',prediction=result)
    (HERE/f'{family}_{stage}_runtime.json').write_text(json.dumps(runtime,indent=2))
    if not final:
        (HERE/f'{family}_selection.json').write_text(json.dumps(selection,indent=2))
        (HERE/f'{family}_tuning.json').write_text(json.dumps(tuning,indent=2))
        m.to_csv(CACHE/'development_samples.csv',index=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('family',choices=['lightgbm','xgboost','tree_control','ridge']);p.add_argument('--final',action='store_true')
    args=p.parse_args();main(args.family,args.final)
