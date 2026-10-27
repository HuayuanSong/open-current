"""Export only demand references that passed the predeclared SHAP audit."""
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from forecast.features import tabular_features,AREAS
CACHE=ROOT/'.cache/forecasting/v2';HERE=Path(__file__).parent

def main():
    audit=json.loads((HERE/'shap-audit.json').read_text())
    assert all(r['publish'] for r in audit['series'] if r['target']=='load')
    a=dict(np.load(CACHE/'dataset.npz'));m=pd.read_csv(CACHE/'samples.csv')
    names=json.loads((CACHE/'feature_names.json').read_text())['future']
    x,extra=tabular_features(a['history'],a['future'],2)
    result={'version':'demand-shap-v1-2026-10-27','modelVersion':'ensemble-v2-2026-10-27',
        'shapVersion':audit['shapVersion'],'auditSha256':hashlib.sha256((HERE/'shap-audit.json').read_bytes()).hexdigest(),
        'featureNames':names+extra,'modelHashes':{},'areas':{}}
    for area in AREAS:
        ids=np.flatnonzero(m.area.eq(area)&m.split.eq('calibration'));result['areas'][area]={}
        result['modelHashes'][area]=hashlib.sha256((CACHE/'models'/f'tree_control_{area}_load.txt').read_bytes()).hexdigest()
        for band,(lo,hi) in enumerate([(0,6),(6,24),(24,48)]):
            pool=np.array([i*48+h for i in ids for h in range(lo,hi)]);result['areas'][area][str(band)]={}
            for size in [64,96]:
                chosen=np.random.default_rng(42).choice(pool,size,replace=False)
                result['areas'][area][str(band)][str(size)]=x[chosen].tolist()
    with gzip.open(ROOT/'forecast/explanation-backgrounds.json.gz','wt') as f:json.dump(result,f,separators=(',',':'),allow_nan=False)
    print('Exported validated demand-only explanation references.')

if __name__=='__main__':main()
