"""Leakage, physical aggregation and saved-model invariants."""
import json
import numpy as np
import pandas as pd
import pytest
from research.forecasting.v3.adaptation import (
    error_features,fit_gate,gate_predict,correction,fixed_weights)
from research.forecasting.v3.fleet import active,bidding_area
from research.forecasting.v3.prepare import fleet_features,regional_tail
from research.forecasting.v3.weather import latest_available_run

def test_future_labels_cannot_change_current_error_inputs():
    origins=pd.date_range('2026-08-01T12:00Z',periods=8,freq='D').as_unit('ns')
    p=np.full((8,48),100.);y=p+np.arange(8)[:,None]
    before=error_features(origins,p,y,np.ones(8)*100)
    times=origins.asi8[:,None]+np.arange(1,49)[None,:]*3600*10**9
    changed=y.copy();changed[times>=(origins[5]-pd.Timedelta(minutes=15)).value]=1e9
    after=error_features(origins,p,changed,np.ones(8)*100)
    np.testing.assert_array_equal(before[:6],after[:6])
    assert not np.array_equal(before[7],after[7])

def test_current_issue_never_corrects_itself():
    origins=pd.DatetimeIndex(['2026-08-01T12:00Z'])
    f=error_features(origins,np.ones((1,48)),np.ones((1,48))*1e9,[100])
    np.testing.assert_array_equal(f,0)

def test_overlapping_errors_keep_newest_issue_per_band():
    origins=pd.DatetimeIndex(['2026-08-01T10:00Z','2026-08-01T11:00Z','2026-08-01T18:00Z'])
    p=np.zeros((3,48));y=np.zeros((3,48));y[0]=100;y[1]=10
    f=error_features(origins,p,y,[1,1,1])
    # First-band latest residual is from the 11 UTC issue, not the older one.
    assert f[2,0,2]==10
    # 11 UTC target is only in first issue; 12–16 UTC are deduplicated and
    # 17 UTC is only in second: seven errors total, not twelve.
    expected=np.sqrt((100**2+6*10**2)/7)
    assert f[2,0,3]==pytest.approx(expected)

def test_gate_convexity_and_json_reload():
    rng=np.random.default_rng(4);p=rng.uniform(0,100,(12,48,3));y=p[:,:,0]
    c=rng.normal(size=(12,48,5));w=np.full((3,3),1/3)
    model=fit_gate(p,y,c,w,1)
    out,weights=gate_predict(model,p,c,w,True)
    np.testing.assert_allclose(weights.sum(-1),1,atol=1e-12)
    assert (weights>=0).all()
    assert (out>=p.min(-1)-1e-8).all() and (out<=p.max(-1)+1e-8).all()
    np.testing.assert_array_equal(out,gate_predict(json.loads(json.dumps(model)),p,c,w))

def test_fixed_blend_is_convex():
    rng=np.random.default_rng(2);p=rng.uniform(0,100,(5,48,3));y=p.mean(-1)
    w=fixed_weights(p,y)
    np.testing.assert_allclose(w.sum(-1),1);assert (w>=0).all()

def test_correction_is_clipped_tapered_and_nonnegative():
    p=np.ones((1,48))*2;features=np.ones((1,48,4))*-1e6
    out=correction(p,features,np.array([100]),'ewma2')
    assert (out>=0).all();assert out[0,0]==0;assert out[0,-1]>0
    positive=correction(p,-features,np.array([100]),'ewma2')
    assert positive[0,0]==22;assert positive[0,12]==12

def test_commissioning_and_retirement_are_origin_specific():
    f=pd.DataFrame({'connected':pd.to_datetime(['2025-01-01','2026-01-01'],utc=True),
                    'disconnected':pd.to_datetime(['2026-01-01',None],utc=True)})
    assert active(f,pd.Timestamp('2025-06-01',tz='UTC')).tolist()==[True,False]
    assert active(f,pd.Timestamp('2026-01-01',tz='UTC')).tolist()==[False,True]

def test_offshore_without_municipality_and_bornholm_are_not_misclassified():
    assert bidding_area(0,'sea',7.67,56.3)=='DK1'
    assert bidding_area(400,'land',14.9,55.1)=='DK2'
    assert bidding_area(707,'sea',11.2,56.6)=='DK1'
    assert bidding_area(390,'sea',12.9,55.0)=='DK2'
    assert bidding_area(0,'sea',np.nan,np.nan) is None
    with pytest.raises(ValueError,match='Unmapped'):
        bidding_area(0,'sea',12.9,55.0)

def test_capacity_is_used_without_weighting_solar_or_demand():
    times=pd.date_range('2026-08-01T13:00Z',periods=48,freq='h')
    points=[{'hourly':{'time':[t.isoformat() for t in times],'wind_speed_100m':[v]*48,'wind_direction_100m':[0]*48}}
            for v in [4,8,12]]
    locations=[dict(id='a',area='DK1',kind='land'),dict(id='b',area='DK1',kind='land'),dict(id='c',area='DK1',kind='sea')]
    fleet=pd.DataFrame({'cluster':['a','b','c'],'mw':[100,300,100], 'area':['DK1']*3,
        'connected':pd.to_datetime(['2020-01-01']*3,utc=True),'disconnected':pd.to_datetime([None]*3,utc=True)})
    values,names=fleet_features(points,fleet,locations,'DK1',times[0],times)
    assert values[0,names.index('fleet_land_wind')]==7
    assert values[0,names.index('fleet_all_mw')]==500
    assert not any('solar' in n or 'load' in n for n in names)
    points[0]['hourly']['wind_speed_100m'][0]=None
    missing,_=fleet_features(points,fleet,locations,'DK1',times[0],times)
    assert np.isnan(missing[0,names.index('fleet_land_power_mw')])

def test_weather_publication_and_replication_buffer():
    run=pd.Timestamp('2026-10-27T06:00Z');available=pd.Timestamp('2026-10-27T12:18:36Z')
    metadata=dict(last_run_initialisation_time=int(run.timestamp()),last_run_availability_time=int(available.timestamp()))
    assert latest_available_run(metadata,pd.Timestamp('2026-10-27T12:15Z')) is None
    assert latest_available_run(metadata,pd.Timestamp('2026-10-27T12:25Z')) is None
    assert latest_available_run(metadata,pd.Timestamp('2026-10-27T12:29Z'))==run

def test_regional_short_horizon_does_not_hide_interior_gaps():
    times=pd.date_range('2026-08-01T19:00Z',periods=48,freq='h')
    regional=np.ones((48,4));regional[-1]=np.nan;ecmwf=np.full((48,4),2.)
    out,native=regional_tail(regional,ecmwf,'2026-08-01T06:00Z',times)
    assert np.isfinite(out).all();np.testing.assert_array_equal(out[-1],2)
    assert native.sum()==47
    regional[15]=np.nan
    out,_=regional_tail(regional,ecmwf,'2026-08-01T06:00Z',times)
    assert np.isnan(out[15]).all() and np.isnan(out[-1]).all()

def test_prepared_capacity_matches_the_corrected_fleet_when_cache_exists():
    from research.forecasting.v3.common import CACHE,HERE
    if not (CACHE/'data-12.npz').exists():pytest.skip('Local research cache unavailable')
    fleet=pd.read_parquet(CACHE/'fleet.parquet')
    data=np.load(CACHE/'data-12.npz');meta=pd.read_csv(CACHE/'samples-12.csv')
    names=json.loads((HERE/'feature-names.json').read_text())['fleet']
    col=names.index('fleet_all_mw')
    for i,row in enumerate(meta.itertuples()):
        expected=fleet.loc[active(fleet,pd.Timestamp(row.origin))&fleet.area.eq(row.area),'mw'].sum()
        np.testing.assert_allclose(data['fleet'][i,:,col],expected,atol=.001,rtol=0)
