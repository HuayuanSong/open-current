"""Audited demand-only TreeSHAP, with a second-reference check at issue time."""
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from .features import tabular_features,AREAS

GROUPS=['Recent electricity','Wind outlook','Sunlight/cloud outlook','Temperature outlook','Calendar/horizon']
LEADS=[0,5,11,23,47]
BACKGROUND=Path(__file__).with_name('explanation-backgrounds.json.gz')

def grouping(names):
    result=[]
    for n in names:
        if n.startswith('wx_'):
            if 'wind' in n or 'power_curve' in n:result.append(1)
            elif 'temperature_2m' in n or 'heating' in n or 'cooling' in n:result.append(3)
            elif 'terrestrial' in n:result.append(4)
            else:result.append(2)
        elif n.startswith(('lag_','mean_','std_','max_')) or n in ['day_baseline','week_baseline','week_hour_mean']:result.append(0)
        else:result.append(4)
    return np.array(result)

def stable(reference,other,scale):
    material=abs(reference)>.05*scale
    signs=not material.any() or np.mean(np.sign(reference[material])==np.sign(other[material]))>=.9
    overlap=len(set(np.argsort(abs(reference))[-2:])&set(np.argsort(abs(other))[-2:]))/2
    change=float(np.abs(reference-other).sum()/max(np.abs(reference).sum(),.1*scale))
    return bool(signs and overlap>=.75 and change<=.30),change

def explain(a,bundle,names):
    import lightgbm as lgb
    import shap
    with gzip.open(BACKGROUND,'rt') as f:config=json.load(f)
    x,extras=tabular_features(a['history'],a['future'],2)
    if names['future']+extras!=config['featureNames']:raise ValueError('Explanation feature schema mismatch')
    group=grouping(config['featureNames']);out={}
    for ai,area in enumerate(AREAS):
        ids=np.flatnonzero(a['area_ids']==ai)
        if len(ids)!=1:raise ValueError('Live explanations require one forecast per region')
        path=bundle/'trees'/f'tree_control_{area}_load.txt'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=config['modelHashes'][area]:raise ValueError('Explanation model mismatch')
        model=lgb.Booster(model_file=str(path));rows=x[ids[0]*48+np.array(LEADS)]
        prediction=model.predict(rows,num_threads=4);scale=max(1.,float(np.median(abs(prediction))));items=[]
        for i,lead in enumerate(LEADS):
            band=str(0 if lead<6 else 1 if lead<24 else 2);values=[];baselines=[]
            for size in ['64','96']:
                background=np.asarray(config['areas'][area][band][size])
                if (model.predict(background,num_threads=4)<0).any():raise ValueError('Clipped reference predictions cannot be explained')
                explainer=shap.TreeExplainer(model,data=background,feature_perturbation='interventional',model_output='raw')
                phi=explainer.shap_values(rows[i:i+1],check_additivity=True)[0]
                values.append(np.array([phi[group==g].sum() for g in range(5)]));baselines.append(float(explainer.expected_value))
            ok,change=stable(values[0],values[1],scale)
            reconstruction=abs(baselines[0]+values[0].sum()-prediction[i])
            ok=bool(ok and reconstruction<=.1 and prediction[i]>=0)
            item={'lead':lead+1,'status':'ready' if ok else 'sensitive','referenceChange':change}
            if ok:item.update(baselineMW=baselines[0],predictionMW=float(prediction[i]),
                contributions=[{'name':name,'mw':float(v)} for name,v in zip(GROUPS,values[0])])
            items.append(item)
        out[area]={'load':items}
    return {'version':config['version'],'modelVersion':config['modelVersion'],'component':'tree_control',
        'reference':'August–September 2026 inputs from the same region and forecast lead band',
        'areas':out,'auditUrl':'https://github.com/HuayuanSong/open-current/tree/main/research/forecasting/uncertainty'}
