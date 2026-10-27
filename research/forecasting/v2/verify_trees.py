"""Check persisted inference independently of truth values."""
from common import *
import argparse
import json
import time
import numpy as np
import pandas as pd
from trees import features

def main(family):
    a,m,names=load_data()
    expected=np.load(CACHE/f'{family}_final.npz')['prediction']
    maximum=0.;total_bytes=0;seconds=[]
    for ti,target in enumerate(TARGETS):
        x=features(a,names,ti,improved=family not in ['tree_control','ridge'])
        for area in AREAS:
            mask=m.area.eq(area).to_numpy() & (m.origin>=pd.Timestamp('2026-10-01',tz='UTC')).to_numpy()
            path=MODELS/f'{family}_{area}_{target}'
            if family=='xgboost':
                import xgboost as xgb
                path=path.with_suffix('.ubj');model=xgb.XGBRegressor(n_jobs=4);model.load_model(path)
            elif family=='ridge':
                import joblib
                path=path.with_suffix('.joblib');model=joblib.load(path)
            else:
                import lightgbm as lgb
                path=path.with_suffix('.txt');model=lgb.Booster(model_file=str(path),params={'num_threads':4})
            xx=x[np.repeat(mask,H)]
            p=np.maximum(0,model.predict(xx).reshape(-1,H))
            ref=expected[mask,:,ti]
            np.testing.assert_allclose(p,ref,rtol=1e-5,atol=.005)
            maximum=max(maximum,float(np.abs(p-ref).max()))
            total_bytes+=path.stat().st_size
            timings=[]
            for _ in range(11):
                start=time.perf_counter();model.predict(xx[:H]);timings.append(time.perf_counter()-start)
            seconds.append(float(np.median(timings[1:])))
    info=dict(family=family,maximum_reload_difference_mw=maximum,model_bytes=total_bytes,
              warm_seconds_all_six_series=sum(seconds))
    (HERE/f'{family}_verification.json').write_text(json.dumps(info,indent=2));print(info)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('family');main(p.parse_args().family)
