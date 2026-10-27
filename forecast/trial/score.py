"""CPU-only challengers and a fixed, target-specific primary recipe."""
import json
import numpy as np
import lightgbm as lgb
from forecast.features import AREAS,TARGETS
from .features import matrix

VARIANTS=['evening_control','newer_weather','fleet_weather','regional_weather']

def filename(variant,area,target):
    if variant=='evening_control':stem='evening_control'
    elif variant=='newer_weather':stem='evening_fresh'
    elif variant=='fleet_weather':stem='evening_fresh_fleet' if target=='wind' else 'evening_fresh'
    elif variant=='regional_weather':stem='regional_multi_weather'
    else:raise ValueError('Unknown trial candidate')
    return f'{stem}_{area}_{target}.txt'

def predict(a,bundle,protocol,available=None):
    if 'y' in a:raise ValueError('Future labels are not scoring inputs')
    names=json.loads((bundle/'feature-names.json').read_text());ids=a['area_ids']
    out={};available=set(VARIANTS if available is None else available)
    for variant in VARIANTS:
        if variant not in available:continue
        result=np.full((len(ids),48,3),np.nan)
        for ti,target in enumerate(TARGETS):
            x=matrix(a,names,ti,variant)
            for ai,area in enumerate(AREAS):
                mask=ids==ai
                if not mask.any():continue
                model=lgb.Booster(model_file=str(bundle/'models'/filename(variant,area,target)))
                if x.shape[1]!=model.num_feature():raise ValueError('Trial feature schema mismatch')
                result[mask,:,ti]=np.maximum(0,model.predict(x[np.repeat(mask,48)],num_threads=4).reshape(-1,48))
        if not np.isfinite(result).all():raise ValueError('Invalid challenger prediction')
        out[variant]=result
    primary=np.full((len(ids),48,3),np.nan)
    for ti,target in enumerate(TARGETS):
        source=protocol['primaryRecipe'][target]
        if source in out:primary[:,:,ti]=out[source][:,:,ti]
    out['primary']=primary
    return out
