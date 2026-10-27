"""Causal error features and regularized, condition-dependent convex blending."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax

BANDS=[(0,6),(6,24),(24,48)]

def fixed_weights(prediction, actual):
    weights=[]
    for lo,hi in BANDS:
        x=prediction[:,lo:hi].reshape(-1,prediction.shape[-1]);y=actual[:,lo:hi].ravel()
        equal=np.full(x.shape[1],1/x.shape[1]);scale=max(float(np.mean(y)),1)
        fit=minimize(lambda w:np.abs(x@w-y).mean()/scale+.01*np.square(w-equal).sum(),equal,
            method='SLSQP',bounds=[(0,1)]*len(equal),constraints={'type':'eq','fun':lambda w:w.sum()-1},
            options={'maxiter':500,'ftol':1e-9})
        if not fit.success: raise RuntimeError(fit.message)
        w=np.maximum(fit.x,0);weights.append(w/w.sum())
    return np.asarray(weights)

def expand_weights(weights,count):
    result=np.empty((count,48,weights.shape[-1]))
    for (lo,hi),w in zip(BANDS,weights,strict=True):result[:,lo:hi]=w
    return result

def conditions(future,names):
    def average(suffix):
        columns=[i for i,n in enumerate(names) if n.startswith('wx_') and n.endswith(suffix)]
        if not columns:raise ValueError(f'Missing condition: {suffix}')
        return future[:,:,columns].mean(-1)
    return np.stack([average('wind_speed_100m'),average('shortwave_radiation'),
        average('temperature_2m'),average('cloud_cover'),
        np.broadcast_to(np.arange(1,49),(len(future),48))],-1)

def gate_predict(model, prediction, condition, weights, return_weights=False):
    x=np.clip((condition-np.asarray(model['mean']))/np.asarray(model['scale']),-3,3)
    x=np.concatenate([np.ones((*x.shape[:-1],1)),x],-1)
    prior=expand_weights(np.asarray(weights),len(x))
    w=softmax(np.log(np.maximum(prior,1e-5))+x@np.asarray(model['coef']),axis=-1)
    out=(w*prediction).sum(-1)
    return (out,w) if return_weights else out

def fit_gate(prediction,actual,condition,weights,penalty):
    mean=condition.mean((0,1));scale=condition.std((0,1));scale=np.maximum(scale,1e-6)
    x=np.clip((condition-mean)/scale,-3,3).reshape(-1,condition.shape[-1])
    x=np.c_[np.ones(len(x)),x];p=prediction.reshape(-1,prediction.shape[-1]);y=actual.ravel()
    prior=expand_weights(weights,len(actual)).reshape(p.shape)
    logprior=np.log(np.maximum(prior,1e-5));target_scale=max(float(y.mean()),1)
    def objective(flat):
        b=flat.reshape(x.shape[1],p.shape[1]);w=softmax(logprior+x@b,axis=1)
        out=(w*p).sum(1);error=(out-y)/target_scale;smooth=np.sqrt(error**2+.01**2)
        loss=smooth.mean()+penalty*np.square(b).mean()
        derivative=error/smooth/target_scale/len(y)
        grad=x.T@(derivative[:,None]*w*(p-out[:,None]))+2*penalty*b/b.size
        return loss,grad.ravel()
    fit=minimize(objective,np.zeros(x.shape[1]*p.shape[1]),jac=True,method='L-BFGS-B',
                 options={'maxiter':400,'ftol':1e-10})
    if not fit.success:raise RuntimeError(f'Gate optimization failed: {fit.message}')
    return dict(mean=mean.tolist(),scale=scale.tolist(),coef=fit.x.reshape(x.shape[1],p.shape[1]).tolist(),
                penalty=penalty,iterations=int(fit.nit))

def error_features(origins, prediction, actual, levels):
    """All historical forecasts, but only matured labels enter each origin.

    Within each lead band, keep the newest prior issue per target timestamp.
    Missing labels are ignored. Current-origin and future outcomes cannot enter.
    This replays daily research issues, not an archived half-hourly live ledger.
    """
    # Pandas 3 can preserve microsecond resolution; Timestamp.value is ns.
    origins=pd.DatetimeIndex(origins).as_unit('ns');n=len(origins)
    times=origins.asi8[:,None]+np.arange(1,49)[None,:]*3600*10**9
    issued=np.broadcast_to(origins.asi8[:,None]+15*60*10**9,(n,48))
    errors=actual-prediction
    result=np.zeros((n,48,4),float)
    for i,origin in enumerate(origins):
        cutoff=(origin-pd.Timedelta(minutes=15)).value
        for lo,hi in BANDS:
            t=times[:,lo:hi].ravel();v=errors[:,lo:hi].ravel();issue=issued[:,lo:hi].ravel()
            valid=(t<=cutoff)&(issue<origin.value)&(t>=cutoff-28*86400*10**9)&np.isfinite(v)
            history=pd.DataFrame({'time':t[valid],'issue':issue[valid],'error':v[valid]})
            history=history.sort_values(['time','issue']).drop_duplicates('time',keep='last')
            if history.empty:continue
            age=(cutoff-history.time.to_numpy())/(86400*10**9);e=history.error.to_numpy()
            summary=[float(np.average(e,weights=2**(-age/d))) for d in [2,7]]
            summary += [float(e[-1]),float(np.sqrt(np.mean(e**2)))]
            result[i,lo:hi]=np.asarray(summary)/max(float(levels[i]),1)
    return result

def correction(prediction, features, levels, option, model=None):
    if option=='none':return prediction.copy()
    if option.startswith('ewma'):
        raw=features[:,:,0 if option=='ewma2' else 1]
    else:
        raw=model.predict(features.reshape(-1,features.shape[-1])).reshape(prediction.shape)
    taper=2**(-np.arange(48)/12)
    adjustment=np.clip(raw,-.2,.2)*levels[:,None]*taper
    return np.maximum(0,prediction+adjustment)
