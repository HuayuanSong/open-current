"""Temporal and artifact checks; no held-out error inspection."""
import hashlib
import json
import numpy as np
import pandas as pd
from common import *
from prepare import causal_fill
from trees import features

def test_original_artifacts_unchanged():
    manifest=json.loads((HERE/'original_artifacts.json').read_text())
    for name,digest in manifest.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name

def test_temporal_folds_and_holdout():
    _,m,_=load_data()
    for _,start,end in FOLDS+[('final','2026-10-01','2026-10-27')]:
        tr,va=masks(m,start,end)
        assert not np.any(tr&va)
        assert m.last_label[tr].max()<m.origin[va].min()
        assert m.last_label[va].max()<pd.Timestamp(end,tz='UTC')

def test_history_fill_cannot_see_later_observations():
    values=pd.DataFrame({'x':np.arange(700,dtype=float)})
    values.loc[[100,300,400,401,402,600],'x']=np.nan
    first=causal_fill(values)
    changed=values.copy();changed.loc[450:,'x']=1e9
    np.testing.assert_array_equal(first.loc[:449],causal_fill(changed).loc[:449])
    assert first.loc[300,'x']==132
    assert first.loc[100,'x']==76

def test_extended_history_preserves_known_recent_observations():
    a,m,_=load_data()
    np.testing.assert_array_equal(a['history512'][:,-168:],a['history'])
    assert a['observed512'][:,-168:].all()
    assert np.isfinite(a['history512']).all()

def test_tree_features_do_not_use_future_labels():
    a,m,names=load_data()
    a={k:v[-4:].copy() for k,v in a.items()}
    original=[features(a,names,ti) for ti in range(3)]
    a['y'][:]=1e20
    for ti in range(3):
        np.testing.assert_array_equal(features(a,names,ti),original[ti])

def test_frozen_models_match_original_predictions_on_overlapping_origins():
    a,m,_=load_data()
    old_meta=pd.read_csv(OLD/'samples.csv')
    old_meta['origin']=pd.to_datetime(old_meta.origin,utc=True)
    joined=m.reset_index(names='new').merge(old_meta.reset_index(names='old'),on=['origin','area'])
    old_pred=dict(np.load(OLD/'trained_predictions.npz'))
    current=np.load(CACHE/'frozen_final.npz')['prediction']
    selected=pd.read_csv(V1/'selected_models.csv')
    for row in selected.itertuples():
        ids=joined[joined.area.eq(row.area)]
        ti=TARGETS.index(row.target)
        np.testing.assert_allclose(current[ids.new,:,ti],old_pred[row.model][ids.old,:,ti],atol=.01,rtol=1e-5)

def test_same_development_cases_for_tree_families():
    a,m,_=load_data()
    keep=(m.origin<pd.Timestamp('2026-10-01',tz='UTC')).to_numpy()
    m=m.loc[keep].reset_index(drop=True)
    valid=np.logical_or.reduce([masks(m,start,end)[1] for _,start,end in FOLDS])
    for family in ['tree_control','ridge','lightgbm','xgboost']:
        pred=np.load(CACHE/f'{family}_dev.npz')['prediction']
        assert pred.shape==a['y'][keep].shape
        np.testing.assert_array_equal(np.isfinite(pred).all((1,2)),valid)
