"""Matched noon/evening inputs with exact forecast vintages and fleet capacity."""
from research.forecasting.v3.common import *
import gzip
import json
import numpy as np
import pandas as pd
from forecast.features import weather_features, calendar, TARGETS, CONFIG
from research.forecasting.v3.weather import read
from research.forecasting.v3.fleet import active

def regional_tail(regional,ecmwf,run,times):
    """The DMI archive ends at run+59h; use same-issue ECMWF for hour60.

    Only the known terminal hour is bridged. Interior gaps remain missing and
    invalidate a matched case. Keep a source flag for fitting and reporting.
    """
    native=np.isfinite(regional).all(1)
    out=regional.copy()
    if native[:-1].all() and not native[-1] and times[-1]-pd.Timestamp(run)==pd.Timedelta(hours=60):
        out[-1]=ecmwf[-1]
    return out,native.astype(float)

def fleet_features(points,fleet,locations,area,origin,times):
    live=fleet.loc[active(fleet,origin)&fleet.area.eq(area)]
    capacity=live.groupby('cluster').mw.sum()
    features={};arrays=[];weights=[];kinds=[]
    for point,location in zip(points,locations,strict=True):
        if location['area']!=area:continue
        f=pd.DataFrame(point['hourly']);f.index=pd.to_datetime(f.pop('time'),utc=True);f=f.reindex(times)
        arrays.append(f);weights.append(float(capacity.get(location['id'],0)));kinds.append(location['kind'])
    speed=np.stack([f.wind_speed_100m.to_numpy(float) for f in arrays],axis=1)
    angle=np.radians(np.stack([f.wind_direction_100m.to_numpy(float) for f in arrays],axis=1))
    weights=np.asarray(weights)
    power=np.where((speed>=3)&(speed<25),np.clip((speed**3-27)/(12**3-27),0,1),0)
    # Missing wind must not silently become zero generation through np.where.
    power[~np.isfinite(speed)]=np.nan
    for kind in ['land','sea','all']:
        w=weights*np.array([kind=='all' or k==kind for k in kinds])
        total=w.sum()
        if total<=0:raise ValueError(f'No usable active {area}/{kind} fleet')
        w=w/total;mean=speed@w
        features[f'fleet_{kind}_mw']=np.full(len(times),total)
        features[f'fleet_{kind}_wind']=mean
        features[f'fleet_{kind}_spread']=np.sqrt(((speed-mean[:,None])**2)@w)
        features[f'fleet_{kind}_power_mw']=(power@w)*total
        features[f'fleet_{kind}_sin']=np.sin(angle)@w
        features[f'fleet_{kind}_cos']=np.cos(angle)@w
    return np.stack(list(features.values()),axis=-1),list(features)

def main():
    a,m,names=load_v2();fleet=pd.read_parquet(CACHE/'fleet.parquet')
    locations=json.loads((HERE/'weather-points.json').read_text())
    hourly=pd.read_parquet(V2/'electricity_hourly.parquet')
    observations={area:hourly[hourly.area.eq(area)].set_index('time')[TARGETS] for area in AREAS}
    datasets={};exclusions=[];manifest=[]
    for hour in [12,18]:
        collected={k:[] for k in ['history','future','history512','observed512','past_full512','y','daylight','fleet','fresh','regional','regional_native']}
        metadata=[]
        for i,row in enumerate(m.itertuples()):
            day=str(row.origin.date());origin=row.origin+pd.Timedelta(hours=hour-12)
            times=pd.date_range(origin+pd.Timedelta(hours=1),periods=48,freq='h')
            obs=observations[row.area]
            y=obs.reindex(times).to_numpy();h=obs.reindex(pd.date_range(end=origin,periods=168,freq='h')).to_numpy()
            if not np.isfinite(y).all() or not np.isfinite(h).all():
                exclusions.append(dict(origin=origin.isoformat(),area=row.area,reason='electricity_gap'));continue
            raw=read(V2/'weather'/f'{day}.json.gz')['data']
            ai=AREAS.index(row.area)
            wx,_,daylight=weather_features(raw[ai*3:ai*3+3],times)
            future=pd.concat([wx,calendar(times)],axis=1).to_numpy()
            fleet_raw=read(CACHE/'weather/fleet'/f'{day}.json.gz')
            fresh_raw=read(CACHE/'weather/fresh'/f'{day}.json.gz')
            assert fleet_raw['points']==locations,'Cached fleet geography changed'
            assert fresh_raw['points']==CONFIG['locations'],'Cached original geography changed'
            assert pd.Timestamp(fleet_raw['run'])==origin.floor('D')
            assert pd.Timestamp(fresh_raw['run'])==origin.floor('D')+pd.Timedelta(hours=6)
            if not isinstance(fleet_raw['data'],list) or not isinstance(fresh_raw['data'],list):
                exclusions.append(dict(origin=origin.isoformat(),area=row.area,reason='weather_unavailable'));continue
            ff,fn=fleet_features(fleet_raw['data'],fleet,locations,row.area,origin,times)
            # Noon origin deliberately retains 00 run: 06 is not conservatively available yet.
            new=future.copy()
            if hour==18:
                assert pd.Timestamp(fresh_raw['run'])+pd.Timedelta(hours=12,minutes=15)<=origin+pd.Timedelta(minutes=15)
                wx,_,_=weather_features(fresh_raw['data'][ai*3:ai*3+3],times)
                new=pd.concat([wx,calendar(times)],axis=1).to_numpy()
            regional=np.full_like(future,np.nan)
            native=np.zeros(48)
            regional_path=CACHE/'weather/regional'/f'{day}.json.gz'
            if hour==18 and regional_path.exists():
                rp=read(regional_path)
                if isinstance(rp['data'],list):
                    wx,_,_=weather_features(rp['data'][ai*3:ai*3+3],times)
                    regional=pd.concat([wx,calendar(times)],axis=1).to_numpy()
                    regional,native=regional_tail(regional,new,rp['run'],times)
            if not all(np.isfinite(x).all() for x in [future,new,ff]):
                exclusions.append(dict(origin=origin.isoformat(),area=row.area,reason='weather_gap'));continue
            history512=a['history512'][i];observed512=a['observed512'][i];past=a['past_full512'][i]
            if hour==18:
                history512=np.concatenate([history512[6:],h[-6:]],axis=0)
                observed512=np.concatenate([observed512[6:],np.ones((6,3))],axis=0)
                past=np.concatenate([past[6:],a['future'][i,:6]],axis=0)
            values=dict(history=h,future=future,history512=history512,observed512=observed512,past_full512=past,
                        y=y,daylight=daylight,fleet=ff,fresh=new,regional=regional,regional_native=native)
            for key,value in values.items():collected[key].append(value)
            metadata.append(dict(origin=origin.isoformat(),area=row.area,area_id=ai,last_label=times[-1].isoformat(),
                                 original_index=i,issue_time=(origin+pd.Timedelta(minutes=15)).isoformat(),
                                 control_run=row.origin.floor('D').isoformat(),
                                 fresh_run=(origin.floor('D')+pd.Timedelta(hours=6 if hour==18 else 0)).isoformat()))
        arrays={k:np.asarray(v,dtype=np.float32) for k,v in collected.items()}
        np.savez_compressed(CACHE/f'data-{hour}.npz',**arrays)
        pd.DataFrame(metadata).to_csv(CACHE/f'samples-{hour}.csv',index=False)
        datasets[str(hour)]={'origins_per_area':pd.DataFrame(metadata).groupby('area').size().to_dict(),
                            'regional_complete_rows':int(np.isfinite(arrays['regional']).all((1,2)).sum()),
                            'regional_native_hours_per_complete_row':sorted(set(arrays['regional_native'][np.isfinite(arrays['regional']).all((1,2))].sum(1).astype(int).tolist()))}
        print(hour,datasets[str(hour)],flush=True)
    pd.DataFrame(exclusions).to_csv(HERE/'weather-exclusions.csv',index=False)
    write_json(HERE/'feature-names.json',{**names,'fleet':fn})
    write_json(HERE/'data-audit.json',dict(datasets=datasets,excluded=len(exclusions),
        original_dataset_sha256=digest(V2/'dataset.npz'),protocol_sha256=digest(HERE/'PROTOCOL.md'),
        fleet_sha256=digest(CACHE/'fleet.parquet'),weather_points_sha256=digest(HERE/'weather-points.json'),
        regional_addendum_sha256=digest(HERE/'REGIONAL_ADDENDUM.md')))

if __name__=='__main__':main()
