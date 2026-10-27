"""Reproduce stored predictions from exported artifacts through scoring APIs."""
from research.forecasting.v3.common import *
from research.forecasting.v3.train_weather import load,STUDIES
from research.forecasting.v3.adapt_experiment import data
from research.forecasting.v3.adaptation import conditions,error_features
from research.forecasting.v3.predict import predict_weather,predict_adaptation
import numpy as np
import pandas as pd
import time

def main():
    checks=[]
    for study in STUDIES:
        a,m,names=load(study);stored=dict(np.load(CACHE/f'{study}-evaluate.npz'))
        expected=pd.read_csv(HERE/f'{study}-evaluate-origins.csv')
        lookup={(r.origin.isoformat(),r.area):i for i,r in enumerate(m.itertuples())}
        ids=np.array([lookup[(pd.Timestamp(r.origin).isoformat(),r.area)] for r in expected.itertuples()])
        inputs={k:v[ids] for k,v in a.items() if k!='y'};area_ids=m.loc[ids,'area_id'].to_numpy()
        for variant,reference in stored.items():
            if variant=='actual':continue
            begin=time.perf_counter();out=predict_weather(inputs,area_ids,study,variant);elapsed=time.perf_counter()-begin
            good=np.isfinite(reference);np.testing.assert_allclose(out[good],reference[good],atol=.002,rtol=2e-6)
            checks.append(dict(study=study,variant=variant,max_delta_mw=float(np.abs(out[good]-reference[good]).max()),seconds=elapsed))
    a,m,names,p=data();c=conditions(a['future'],names['future']);area_ids=m.area.map({'DK1':0,'DK2':1}).to_numpy()
    levels=np.maximum(a['history'].mean(1),1)
    start=time.perf_counter();uncorrected=predict_adaptation(p,c,area_ids,levels)
    f=np.zeros((*a['y'].shape,4));g=np.zeros_like(f)
    for area in AREAS:
        mask=m.area.eq(area).to_numpy()
        for ti in range(3):
            f[mask,:,ti]=error_features(m.loc[mask,'origin'],uncorrected['fixed'][mask,:,ti],a['y'][mask,:,ti],levels[mask,ti])
            g[mask,:,ti]=error_features(m.loc[mask,'origin'],uncorrected['gate'][mask,:,ti],a['y'][mask,:,ti],levels[mask,ti])
    out=predict_adaptation(p,c,area_ids,levels,f,g)
    test=m.origin>=pd.Timestamp('2026-10-01',tz='UTC');expected=np.load(CACHE/'adapt-evaluation.npz')
    for key,reference in [('fixed','fixed'),('gate','selected_gate'),('corrected','selected_correction'),('combined','combined')]:
        np.testing.assert_allclose(out[key][test],expected[reference],atol=.002,rtol=2e-6)
        checks.append(dict(study='adapt',variant=key,max_delta_mw=float(np.abs(out[key][test]-expected[reference]).max())))
    checks.append(dict(study='adapt',operation='all historical replay + saved scoring',seconds=time.perf_counter()-start))
    write_json(HERE/'serving-verification.json',checks);print(checks,flush=True)

if __name__=='__main__':main()
