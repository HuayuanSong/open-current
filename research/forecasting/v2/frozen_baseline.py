"""Score the v1 saved models on v2 cases without refitting or reading errors."""
from common import *
import argparse
import json
import numpy as np
import pandas as pd

def main(neural):
    a,m,names=load_data()
    if neural:
        import torch
        from neural import WeatherMLP
        out=np.zeros_like(a['y'])
        selections=json.loads((V1/'selection.json').read_text())
        for ti,target in enumerate(TARGETS):
            width=next(r['setting'] for r in selections if r['family']=='weather_mlp' and r['target']==target)
            artifact=torch.load(OLD/'models'/f'mlp_{target}_{width}.pt',map_location='cpu',weights_only=True)
            model=WeatherMLP(artifact['features'],width,ti)
            model.load_state_dict(artifact['state_dict']);model.eval()
            ai=m.area_id.to_numpy()
            mean=np.asarray(artifact['means'])[ai,None];scale=np.asarray(artifact['scales'])[ai,None]
            history=torch.tensor((a['history'][:,:,ti]-mean)/scale,dtype=torch.float32)
            future=torch.tensor((a['future']-artifact['scaler_mean'])/artifact['scaler_scale'],dtype=torch.float32)
            areas=torch.tensor(np.eye(2)[ai],dtype=torch.float32)
            with torch.inference_mode():
                out[:,:,ti]=np.maximum(0,model(history,future,areas).numpy()*scale+mean)
        np.savez_compressed(CACHE/'frozen_neural.npz',prediction=out)
    else:
        import lightgbm as lgb
        import joblib
        from build_data import tabular_features
        nn=np.load(CACHE/'frozen_neural.npz')['prediction']
        selected=pd.read_csv(V1/'selected_models.csv')
        out=np.full_like(a['y'],np.nan)
        for ti,target in enumerate(TARGETS):
            x,_=tabular_features(a['history'],a['future'],ti)
            for area in AREAS:
                mask=m.area.eq(area).to_numpy()
                family=selected.loc[selected.area.eq(area)&selected.target.eq(target),'model'].iloc[0]
                if family=='ridge':
                    model=joblib.load(OLD/'models'/f'ridge_{area}_{target}.joblib')
                else:
                    model=lgb.Booster(model_file=str(OLD/'models'/f'lightgbm_{area}_{target}.txt'))
                pred=np.maximum(0,model.predict(x[np.repeat(mask,H)]).reshape(-1,H))
                if family=='blend':
                    pred=.5*pred+.5*nn[mask,:,ti]
                out[mask,:,ti]=pred
        np.savez_compressed(CACHE/'frozen_final.npz',prediction=out)
        dev=(m.origin<pd.Timestamp('2026-10-01',tz='UTC')).to_numpy()
        np.savez_compressed(CACHE/'frozen_dev.npz',prediction=out[dev])
        print('Frozen v1 models scored without refitting.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--neural',action='store_true');args=p.parse_args();main(args.neural)
