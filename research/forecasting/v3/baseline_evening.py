"""Score the actual deployed ensemble on the matched October evening inputs."""
from research.forecasting.v3.common import *
import json
import numpy as np
import pandas as pd
from forecast.score import run

def main():
    a=dict(np.load(CACHE/'data-18.npz'));m=pd.read_csv(CACHE/'samples-18.csv')
    keep=pd.to_datetime(m.origin,utc=True)>=pd.Timestamp('2026-10-01',tz='UTC')
    data={k:v[keep] for k,v in a.items() if k in ['history','future','history512','observed512','past_full512']}
    data['area_ids']=m.loc[keep,'area_id'].to_numpy()
    path=CACHE/'evening-baseline-input.npz';np.savez_compressed(path,**data)
    prediction,families,timings=run(path,CACHE/'baseline-evening-workers')
    np.savez_compressed(CACHE/'baseline-evening.npz',prediction=prediction,actual=a['y'][keep],**families)
    m.loc[keep].to_csv(HERE/'baseline-evening-origins.csv',index=False)
    write_json(HERE/'baseline-evening-runtime.json',timings)
    print('Saved real deployed ensemble forecasts for',int(keep.sum()),'area/origins',flush=True)

if __name__=='__main__':main()
