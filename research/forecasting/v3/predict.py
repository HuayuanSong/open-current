"""CPU scoring of frozen research challengers without loading evaluation labels.

This is a local handoff, not a change to production. Weather inputs must already
be assembled with the source-vintage rules in prepare.py/weather.py. Correction
inputs must be derived from matured errors using adaptation.error_features.
"""
from research.forecasting.v3.common import CACHE,HERE,AREAS,TARGETS,digest
from research.forecasting.v3.train_weather import matrix
from research.forecasting.v3.adaptation import gate_predict,expand_weights,correction
import json
import joblib
import lightgbm as lgb
import numpy as np

def verify_artifact(path):
    manifest=json.loads((HERE/'artifacts.json').read_text())
    if digest(HERE/'feature-names.json')!=manifest['feature_names_sha256']:
        raise ValueError('Feature schema checksum mismatch')
    entry=manifest['files'][str(path.relative_to(CACHE))]
    if digest(path)!=entry['sha256']:raise ValueError(f'Artifact checksum mismatch: {path.name}')

def predict_weather(inputs,area_ids,study,variant):
    # Deliberately accept no labels; this also prevents accidental later feature
    # extensions from reaching into them during serving.
    if 'y' in inputs:raise ValueError('Scoring inputs must not contain future labels')
    names=json.loads((HERE/'feature-names.json').read_text())
    out=np.full((len(area_ids),48,3),np.nan)
    targets=[0] if study=='fleet' else range(3)
    for ti in targets:
        effective='fresh' if variant=='fresh_fleet' and ti!=0 else variant
        x=matrix(inputs,names,ti,study,effective)
        for ai,area in enumerate(AREAS):
            mask=area_ids==ai
            if not mask.any():continue
            path=CACHE/'weather-models'/f'{study}_{effective}_{area}_{TARGETS[ti]}.txt'
            verify_artifact(path);model=lgb.Booster(model_file=str(path))
            if x.shape[1]!=model.num_feature():raise ValueError('Feature schema mismatch')
            out[mask,:,ti]=np.maximum(0,model.predict(x[np.repeat(mask,48)],num_threads=4).reshape(-1,48))
    if not np.isfinite(out[:,:,list(targets)]).all():raise ValueError('Incomplete challenger forecast')
    return out

def predict_adaptation(families,conditions,area_ids,levels,error_inputs=None,gate_error_inputs=None):
    """families: N×48×3×9; error_inputs: N×48×3×4, already origin-safe.

    With no matured-error archive, return uncorrected forecasts explicitly.
    Gate and fixed forecasts need separate error archives because they are
    different predictors. Never apply another predictor's residuals silently.
    """
    out={k:np.full(families.shape[:3],np.nan) for k in ['fixed','gate','corrected','combined']}
    for ai,area in enumerate(AREAS):
        mask=area_ids==ai
        if not mask.any():continue
        for ti,target in enumerate(TARGETS):
            prefix=CACHE/'adapt-models'/f'{area}_{target}'
            path=prefix.with_name(prefix.name+'_gate.json');verify_artifact(path);saved=json.loads(path.read_text())
            weights=np.asarray(saved['weights']);p=families[mask,:,ti]
            base=(expand_weights(weights,len(p))*p).sum(-1)
            gate=gate_predict(saved['gate'],p,conditions[mask],weights)
            out['fixed'][mask,:,ti]=base;out['gate'][mask,:,ti]=gate
            for name,original,inputs,suffix in [('corrected',base,error_inputs,'correction'),('combined',gate,gate_error_inputs,'combined')]:
                if inputs is None:out[name][mask,:,ti]=original;continue
                path=prefix.with_name(prefix.name+f'_{suffix}.joblib');verify_artifact(path)
                # Only locally generated, hash-verified artifacts are deserialized.
                model=joblib.load(path)
                out[name][mask,:,ti]=correction(original,inputs[mask,:,ti],levels[mask,ti],saved['selection']['correction'],model)
    if not all(np.isfinite(v).all() for v in out.values()):raise ValueError('Incomplete adaptation forecast')
    return out
