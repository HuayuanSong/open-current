"""Run in the separate TTM environment after preprocessing changes."""
import numpy as np
import pandas as pd
from ttm_experiment import tensors

def fixture():
    rng=np.random.default_rng(42)
    a={'history512':rng.uniform(1,1000,(6,512,3)).astype(np.float32),
       'observed512':np.ones((6,512,3),np.float32),
       'past_full512':rng.uniform(1,500,(6,512,2)).astype(np.float32),
       'future':rng.uniform(1,500,(6,48,2)).astype(np.float32),
       'y':rng.uniform(1,1000,(6,48,3)).astype(np.float32)}
    names={'future':['wx_0_shortwave_radiation','wx_0_wind_speed_100m']}
    meta=pd.DataFrame({'area_id':[0,1,0,1,0,1]})
    tr=np.array([True,True,True,True,False,False])
    return a,meta,names,tr

def test_standardizer_does_not_use_validation_or_holdout():
    a,m,n,tr=fixture();first=tensors(a,n,m,tr)
    for k in ['history512','past_full512','future','y']:
        a[k][~tr]=1e9
    second=tensors(a,n,m,tr)
    assert first[6]==second[6]

def test_targets_roundtrip_and_future_target_inputs_are_empty():
    a,m,n,tr=fixture();data=tensors(a,n,m,tr)
    restored=data[3].numpy()*data[5]+data[4]
    np.testing.assert_allclose(restored,a['y'],rtol=1e-5,atol=.001)
    assert (data[2][:,:,:3]==0).all()

def test_original_observation_mask_survives_scaling():
    a,m,n,tr=fixture()
    a['observed512'][0,20,1]=0
    a['past_full512'][1,10,0]=np.nan
    data=tensors(a,n,m,tr)
    assert not data[1][0,20,1]
    assert not data[1][1,10,3]
    assert np.isfinite(data[0].numpy()).all()
