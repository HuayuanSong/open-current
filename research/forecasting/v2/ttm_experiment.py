"""Fine-tune a pretrained TTM backbone with an origin-specific weather head."""
from common import *
import argparse
import gc
import json
import resource
import time
import numpy as np
import pandas as pd
import torch
from tsfm_public import TinyTimeMixerForPrediction
torch.set_num_threads(4)
torch.set_num_interop_threads(1)

def covariates(names):
    return [i for i,n in enumerate(names['future']) if n.startswith('wx_') or
            n in ['hour_sin','hour_cos','weekend','holiday','year_sin','year_cos']]

def tensors(a,names,meta=None,train=None):
    cols=covariates(names)
    past=np.concatenate([a['history512'],a['past_full512'][:,:,cols]],axis=-1)
    mask=np.isfinite(past);mask[:,:,:3]=a['observed512'].astype(bool)
    # Ground truth is never supplied in the future input tensor, even during
    # training. Only exogenous forecast channels contain information.
    future=np.concatenate([np.zeros_like(a['y']),a['future'][:,:,cols]],axis=-1)
    truth=a['y'].copy()
    means=np.zeros((len(past),1,3),np.float32);scales=np.ones_like(means)
    scalers={}
    if train is not None:
        for ai,area in enumerate(AREAS):
            am=meta.area_id.eq(ai).to_numpy()
            fitted=np.where(mask[train & am],past[train & am],np.nan)
            mean=np.nanmean(fitted.astype(np.float64),axis=(0,1)).astype(np.float32)
            scale=np.nanstd(fitted.astype(np.float64),axis=(0,1)).astype(np.float32)
            scale=np.maximum(scale,1e-5)
            # Balance the three target losses using training labels only.
            mean[:3]=a['y'][train & am].mean((0,1))
            scale[:3]=np.maximum(a['y'][train & am].std((0,1)),1)
            past[am]=(past[am]-mean)/scale
            future[am,:,3:]=(future[am,:,3:]-mean[3:])/scale[3:]
            truth[am]=(truth[am]-mean[:3])/scale[:3]
            means[am]=mean[:3];scales[am]=scale[:3]
            scalers[area]={'mean':mean.tolist(),'scale':scale.tolist()}
    return (torch.from_numpy(np.nan_to_num(past)),torch.from_numpy(mask),
            torch.from_numpy(future),torch.from_numpy(truth),means,scales,scalers)

def make_model(channels):
    spec=json.loads((HERE/'ttm_model.json').read_text())
    torch.manual_seed(42)
    model=TinyTimeMixerForPrediction.from_pretrained(spec['model'],revision=spec['revision'],
        num_input_channels=channels,prediction_channel_indices=[0,1,2],
        exogenous_channel_indices=list(range(3,channels)),prediction_filter_length=48,
        enable_forecast_channel_mixing=True,decoder_mode='common_channel',
        fcm_context_length=2,fcm_use_mixer=True,loss='mae')
    for parameter in model.backbone.parameters():
        parameter.requires_grad=False
    return model

def forward(model,data,ids,use_cache=True):
    past,mask,future=data[:3]
    cached=getattr(model,'_frozen_backbone_cache',None)
    if use_cache and cached is not None:
        decoded,_=model.decoder(hidden_state=cached['last_hidden_state'][ids],
            patch_input=cached['patch_input'][ids],output_hidden_states=False)
        predicted=model.head(decoded,past_values=past[ids],future_values=future[ids])
        return model.backbone.scaler.inverse(data=predicted,
            loc=cached['loc'][ids,:,:3],scale=cached['scale'][ids,:,:3])
    return model(past_values=past[ids],past_observed_mask=mask[ids],
                 future_values=future[ids],return_loss=False).prediction_outputs

def cache_backbone(model,data):
    model.eval();saved={key:[] for key in ['last_hidden_state','patch_input','loc','scale']}
    # no_grad (not inference_mode): decoder gradients need to save these tensors.
    with torch.no_grad():
        for offset in range(0,len(data[0]),16):
            output=model.backbone(data[0][offset:offset+16],past_observed_mask=data[1][offset:offset+16])
            for key in saved:saved[key].append(getattr(output,key).detach())
    model._frozen_backbone_cache={k:torch.cat(v) for k,v in saved.items()}
    ids=np.arange(2)
    with torch.no_grad():
        torch.testing.assert_close(forward(model,data,ids),forward(model,data,ids,False),atol=1e-5,rtol=1e-5)

def predict(model,data,indices):
    model.eval();out=np.full(tuple(data[3].shape),np.nan,np.float32)
    with torch.inference_mode():
        for offset in range(0,len(indices),8):
            ids=indices[offset:offset+8]
            out[ids]=np.maximum(forward(model,data,ids,False).numpy()*data[5][ids]+data[4][ids],0)
    return out

def score(pred,y,m):
    vals=[]
    for area in AREAS:
        for ti in range(3):
            mask=m.area.eq(area).to_numpy() & np.isfinite(pred).all((1,2))
            vals.append(float(np.abs(pred[mask,:,ti]-y[mask,:,ti]).sum()/y[mask,:,ti].sum()))
    return float(np.mean(vals))

def main(final=False,pilot=False,scaled=False):
    family='ttm_scaled' if scaled else 'ttm'
    a,m,names=load_data()
    if not final:
        keep=(m.origin<pd.Timestamp('2026-10-01',tz='UTC')).to_numpy()
        a={k:v[keep] for k,v in a.items()};m=m.loc[keep].reset_index(drop=True)
    else:
        a['y'][m.origin>=pd.Timestamp('2026-10-01',tz='UTC')]=np.nan
    data=tensors(a,names)
    if final:
        selection=json.loads((HERE/f'{family}_selection.json').read_text())
        rates=[selection['learning_rate']];epochs=[selection['epochs']]
        folds=[('final','2026-10-01','2026-10-27')]
    else:
        rates=[1e-4,3e-4];epochs=[10,20,40];folds=FOLDS
    if pilot:
        rates=[1e-4];epochs=[1];folds=FOLDS[:1]
    candidates={f'{lr}_{ep}':np.full_like(a['y'],np.nan) for lr in rates for ep in epochs}
    runtime=[];curves=[]
    for fold,start,end in folds:
        tr,va=masks(m,start,end);trainids=np.flatnonzero(tr);validids=np.flatnonzero(va)
        if scaled:
            data=tensors(a,names,m,tr)
        scales=np.zeros((len(m),1,3),np.float32)
        for ai in range(2):
            area=m.area_id.eq(ai).to_numpy()
            scales[area]=np.maximum(a['y'][tr & area].std((0,1)),1)
        scales=torch.from_numpy(scales)
        if scaled:
            scales=torch.ones_like(scales)
        if pilot:
            trainids=trainids[:16];validids=validids[:2]
        for lr in rates:
            model=make_model(data[0].shape[-1])
            cache_begin=time.perf_counter()
            if scaled:
                cache_backbone(model,data)
            cache_seconds=time.perf_counter()-cache_begin
            optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=.01)
            rng=np.random.default_rng(42)
            begin=time.perf_counter()
            for epoch in range(1,max(epochs)+1):
                model.train();model.backbone.eval()
                shuffled=rng.permutation(trainids);losses=[]
                for offset in range(0,len(shuffled),16):
                    ids=shuffled[offset:offset+16]
                    optimizer.zero_grad(set_to_none=True)
                    predicted=forward(model,data,ids)
                    loss=((predicted-data[3][ids]).abs()/scales[ids]).mean()
                    if not torch.isfinite(loss):
                        raise ValueError('Non-finite TTM training loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1)
                    optimizer.step();losses.append(float(loss.detach()))
                curves.append(dict(fold=fold,learning_rate=lr,epoch=epoch,train_loss=float(np.mean(losses))))
                if epoch in epochs:
                    elapsed=time.perf_counter()-begin
                    pred=predict(model,data,validids)
                    candidates[f'{lr}_{epoch}'][validids]=pred[validids]
                    path=MODELS/f'{family}_{fold}_{lr}_{epoch}'
                    model.save_pretrained(path)
                    (path/'input_scalers.json').write_text(json.dumps(data[6],indent=2))
                    durations=[]
                    for _ in range(4):
                        tick=time.perf_counter();predict(model,data,validids[:2]);durations.append(time.perf_counter()-tick)
                    info=dict(fold=fold,learning_rate=lr,epochs=epoch,training_seconds=elapsed,
                              backbone_cache_seconds=cache_seconds,
                              warm_inference_seconds_two_areas=float(np.median(durations[1:])),
                              parameters=sum(p.numel() for p in model.parameters()),
                              trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                              peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                    runtime.append(info);print(json.dumps(info),flush=True)
                    reloaded=TinyTimeMixerForPrediction.from_pretrained(path)
                    restored=predict(reloaded,data,validids[:2])
                    np.testing.assert_allclose(restored[validids[:2]],pred[validids[:2]],atol=.01,rtol=1e-5)
                    # Perturbing unknown future targets cannot alter forecasts.
                    changed=list(data);changed[2]=data[2].clone();changed[2][:,:,:3]=1e7
                    check=predict(reloaded,changed,validids[:2])
                    np.testing.assert_array_equal(check[validids[:2]],restored[validids[:2]])
                    del reloaded,changed;gc.collect()
                elif epoch%5==0:
                    print(f'TTM {fold} lr={lr} epoch={epoch} train_loss={np.mean(losses):.5f}',flush=True)
            del model,optimizer;gc.collect()
        np.savez_compressed(CACHE/f'{family}_{"pilot" if pilot else "final_candidates" if final else "candidates"}.npz',**candidates)
    stage='pilot' if pilot else 'final' if final else 'dev'
    (HERE/f'{family}_{stage}_runtime.json').write_text(json.dumps(runtime,indent=2))
    (HERE/f'{family}_{stage}_curves.json').write_text(json.dumps(curves,indent=2))
    if pilot:
        return
    if not final:
        scores={k:score(v,a['y'],m) for k,v in candidates.items()}
        chosen=min(scores,key=scores.get)
        lr,ep=chosen.split('_')
        (HERE/f'{family}_selection.json').write_text(json.dumps(dict(learning_rate=float(lr),epochs=int(ep),scores=scores,
            covariates=[names['future'][i] for i in covariates(names)],frozen_backbone=True,scaled=scaled),indent=2))
        print('TTM selection',chosen,scores,flush=True)
    else:
        chosen=f'{rates[0]}_{epochs[0]}'
    np.savez_compressed(CACHE/f'{family}_{stage}.npz',prediction=candidates[chosen])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');p.add_argument('--pilot',action='store_true');p.add_argument('--scaled',action='store_true')
    args=p.parse_args();main(args.final,args.pilot,args.scaled)
