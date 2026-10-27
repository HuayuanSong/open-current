"""Isolated CPU inference workers; never mix LightGBM and PyTorch runtimes."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[name]='4'
from pathlib import Path
os.environ.setdefault('HF_HOME',str(Path(__file__).resolve().parents[1]/'.cache/forecast-serving/huggingface'))
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
import argparse
import json
import resource
import sys
import time
import numpy as np
from .artifacts import ensure
from .features import AREAS,TARGETS,features,tabular_features

def ridge(path,x):
    p=json.loads(path.read_text())
    return ((x-np.asarray(p['mean']))/np.asarray(p['scale']))@np.asarray(p['coef'])+p['intercept']

def trees(a,bundle,names):
    import lightgbm as lgb
    import xgboost as xgb
    count=len(a['history']);ids=a['area_ids']
    result={f:np.full((count,48,3),np.nan,np.float32) for f in ['tree_control','lightgbm','xgboost','ridge','original_trees']}
    for ti,target in enumerate(TARGETS):
        regular=features(a,names,ti,False);improved=features(a,names,ti,True)
        for ai,area in enumerate(AREAS):
            mask=ids==ai;rows=np.repeat(mask,48)
            if not mask.any():continue
            for family in ['tree_control','lightgbm','xgboost','ridge','original_trees']:
                x=improved if family in ['lightgbm','xgboost'] else regular
                if family=='ridge' or (family=='original_trees' and target=='load'):
                    folder='original' if family=='original_trees' else 'ridge'
                    prediction=ridge(bundle/folder/f'ridge_{area}_{target}.json',x[rows])
                elif family=='xgboost':
                    model=xgb.XGBRegressor(n_jobs=4);model.load_model(bundle/'trees'/f'{family}_{area}_{target}.ubj')
                    prediction=model.predict(x[rows])
                else:
                    path=bundle/'original'/f'lightgbm_{area}_{target}.txt' if family=='original_trees' else bundle/'trees'/f'{family}_{area}_{target}.txt'
                    model=lgb.Booster(model_file=str(path))
                    prediction=model.predict(x[rows],num_threads=4)
                result[family][mask,:,ti]=np.maximum(0,prediction.reshape(-1,48))
    return result

def columns(names):
    return [i for i,n in enumerate(names['future']) if n.startswith('wx_') or n in ['hour_sin','hour_cos','weekend','holiday','year_sin','year_cos']]

def neural(a,bundle,names):
    import torch
    from chronos import Chronos2Pipeline
    from .weather_mlp import WeatherMLP
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    result={};ids=a['area_ids'];out=np.zeros((len(ids),48,3),np.float32)
    for ti,target in enumerate(TARGETS[:2]):
        p=torch.load(bundle/'original'/f'mlp_{target}.pt',map_location='cpu',weights_only=True)
        model=WeatherMLP(p['features'],p['width'],ti);model.load_state_dict(p['state_dict']);model.eval()
        mean=np.asarray(p['means'])[ids,None];scale=np.asarray(p['scales'])[ids,None]
        history=torch.tensor((a['history'][:,:,ti]-mean)/scale,dtype=torch.float32)
        future=torch.tensor((a['future']-p['scaler_mean'])/p['scaler_scale'],dtype=torch.float32)
        with torch.inference_mode():
            out[:,:,ti]=np.maximum(0,model(history,future,torch.tensor(np.eye(2)[ids],dtype=torch.float32)).numpy()*scale+mean)
    result['original_neural']=out
    selected=columns(names);keys=[names['future'][j] for j in selected]
    examples=[dict(target=a['history'][i].T,
        past_covariates={k:a['past_full512'][i,-168:,j] for k,j in zip(keys,selected)},
        future_covariates={k:a['future'][i,:,j] for k,j in zip(keys,selected)}) for i in range(len(ids))]
    revision=json.loads((bundle/'chronos_model.json').read_text())
    for family in ['chronos_zero','chronos']:
        args={'revision':revision['revision']} if family=='chronos_zero' else {}
        source=revision['model'] if family=='chronos_zero' else bundle/'chronos'
        pipeline=Chronos2Pipeline.from_pretrained(source,device_map='cpu',dtype=torch.float32,**args)
        pipeline.model.eval()
        quantiles=pipeline.predict_quantiles(examples,prediction_length=48,quantile_levels=[.5],context_length=168,batch_size=176,cross_learning=False)[0]
        result[family]=np.maximum(0,np.stack([p.numpy()[:,:,0].T for p in quantiles]))
        del pipeline
    return result

def ttm(a,bundle,names):
    import torch
    from tsfm_public import TinyTimeMixerForPrediction
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    cols=columns(names);ids=a['area_ids'];result={}
    raw=np.concatenate([a['history512'],a['past_full512'][:,:,cols]],axis=-1)
    observed=np.isfinite(raw);observed[:,:,:3]=a['observed512'].astype(bool)
    for family in ['ttm','ttm_scaled']:
        past=raw.copy();future=np.concatenate([np.zeros((len(ids),48,3),np.float32),a['future'][:,:,cols]],axis=-1)
        means=np.zeros((len(ids),1,3),np.float32);scales=np.ones_like(means)
        if family=='ttm_scaled':
            scalers=json.loads((bundle/family/'input_scalers.json').read_text())
            for ai,area in enumerate(AREAS):
                mask=ids==ai;p=scalers[area];mean=np.array(p['mean'],np.float32);scale=np.array(p['scale'],np.float32)
                past[mask]=(past[mask]-mean)/scale;future[mask,:,3:]=(future[mask,:,3:]-mean[3:])/scale[3:]
                means[mask]=mean[:3];scales[mask]=scale[:3]
        model=TinyTimeMixerForPrediction.from_pretrained(bundle/family);model.eval()
        with torch.inference_mode():
            prediction=model(past_values=torch.from_numpy(np.nan_to_num(past)),past_observed_mask=torch.from_numpy(observed),
                future_values=torch.from_numpy(future),return_loss=False).prediction_outputs.numpy()
        result[family]=np.maximum(0,prediction*scales+means)
        del model
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('family',choices=['trees','neural','ttm']);p.add_argument('input');p.add_argument('output')
    p.add_argument('--explain',action='store_true')
    args=p.parse_args();begin=time.perf_counter();bundle=ensure()
    a=dict(np.load(args.input));names=json.loads((bundle/'feature_names.json').read_text())
    predictions=globals()[args.family](a,bundle,names)
    if args.explain and args.family=='trees':
        from .explanations import explain
        try:explanation=explain(a,bundle,names)
        except Exception as error:
            explanation={'status':'unavailable'}
            print(f'Explanation unavailable: {error}',file=sys.stderr,flush=True)
        Path(args.output).with_suffix('.explanations.json').write_text(json.dumps(explanation,allow_nan=False))
    if not all(np.isfinite(v).all() for v in predictions.values()):raise ValueError('Non-finite forecast')
    np.savez_compressed(args.output,**predictions)
    info={'worker':args.family,'seconds':time.perf_counter()-begin,
          'peakRssBytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)}
    Path(args.output).with_suffix('.json').write_text(json.dumps(info))
    print(json.dumps(info),flush=True)

if __name__=='__main__':main()
