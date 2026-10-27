"""Small serving-only counterpart of the frozen v3 weather transformations."""
import numpy as np
import pandas as pd
from forecast.features import features,weather_features,calendar,AREAS

def weather_array(points,times):
    if not isinstance(points,list) or len(points)!=6:raise ValueError('Expected six weather locations')
    result=[]
    for i in range(2):
        wx,_,_=weather_features(points[i*3:i*3+3],times)
        result.append(pd.concat([wx,calendar(times)],axis=1).to_numpy())
    return np.asarray(result,dtype=np.float32)

def fleet_array(points,locations,capacity,times):
    if len(points)!=len(locations):raise ValueError('Fleet location count mismatch')
    result=[]
    for area in AREAS:
        frames=[];weights=[];kinds=[]
        for point,location in zip(points,locations,strict=True):
            if location['area']!=area:continue
            frame=pd.DataFrame(point['hourly']);frame.index=pd.to_datetime(frame.pop('time'),utc=True)
            frames.append(frame.reindex(times));weights.append(capacity[location['id']]);kinds.append(location['kind'])
        speed=np.stack([f.wind_speed_100m.to_numpy(float) for f in frames],axis=1)
        angle=np.radians(np.stack([f.wind_direction_100m.to_numpy(float) for f in frames],axis=1))
        weights=np.asarray(weights);power=np.where((speed>=3)&(speed<25),np.clip((speed**3-27)/(12**3-27),0,1),0)
        power[~np.isfinite(speed)]=np.nan;columns=[]
        for kind in ['land','sea','all']:
            w=weights*np.array([kind=='all' or k==kind for k in kinds]);total=w.sum()
            if total<=0:raise ValueError('Empty active fleet cluster group')
            w=w/total;mean=speed@w
            columns += [np.full(len(times),total),mean,np.sqrt(((speed-mean[:,None])**2)@w),
                        (power@w)*total,np.sin(angle)@w,np.cos(angle)@w]
        result.append(np.stack(columns,-1))
    out=np.asarray(result,dtype=np.float32)
    if not np.isfinite(out).all():raise ValueError('Incomplete fleet weather')
    return out

def regional_tail(regional,ecmwf,run,times):
    out=regional.copy();native=np.isfinite(out).all(-1)
    for i in range(len(out)):
        if native[i,:-1].all() and not native[i,-1] and times[-1]-run==pd.Timedelta(hours=60):
            out[i,-1]=ecmwf[i,-1]
    if not np.isfinite(out).all():raise ValueError('DMI has an interior gap or incomplete horizon')
    return out,native.astype(np.float32)

def matrix(a,names,ti,variant):
    weather='future' if variant=='evening_control' else 'fresh'
    x=features({**a,'future':a[weather]},names,ti,True)
    if ti==0 and variant=='fleet_weather':x=np.c_[x,a['fleet'].reshape(len(x),-1)]
    if variant=='regional_weather':
        cols=[i for i,n in enumerate(names['future']) if n.startswith('wx_')]
        x=np.c_[x,a['regional'][:,:,cols].reshape(len(x),-1),a['regional_native'].ravel()]
    if not np.isfinite(x).all():raise ValueError('Non-finite trial feature')
    return x
