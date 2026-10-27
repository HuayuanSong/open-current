"""Per-location weather and fixed-origin Chronos-2 LoRA training."""
from common import *
import argparse
import gc
import json
import resource
import time
import numpy as np
import pandas as pd
import torch
import peft  # Required: never allow the library's full-fine-tuning fallback.
from chronos import Chronos2Pipeline
from chronos.chronos2.dataset import Chronos2Dataset, DatasetMode
from transformers import TrainerCallback

torch.set_num_threads(4)
torch.set_num_interop_threads(1)

class FlushLogs(TrainerCallback):
    def on_log(self, args, state, control, **kwargs):
        import sys
        sys.stdout.flush()

def covariates(names):
    return [i for i,n in enumerate(names['future']) if n.startswith('wx_') or
            n in ['hour_sin','hour_cos','weekend','holiday','year_sin','year_cos']]

def example(a,names,index,training=False):
    cols=covariates(names)
    target=a['history'][index].T
    past=a['past_full512'][index,-168:][:,cols]
    future=a['future'][index][:,cols]
    if training:
        target=np.concatenate([target,a['y'][index].T],axis=1)
        past=np.concatenate([past,future],axis=0)
    keys=[names['future'][j] for j in cols]
    return dict(target=target,
                past_covariates={n:past[:,j] for j,n in enumerate(keys)},
                future_covariates={n:None if training else future[:,j] for j,n in enumerate(keys)})

def check_training_window(a,names,index):
    item=example(a,names,index,True)
    ds=Chronos2Dataset([item],context_length=168,prediction_length=48,
                      batch_size=128,output_patch_size=16,min_past=168,mode=DatasetMode.TRAIN)
    for _ in range(5):
        context,labels,future,n=ds._construct_slice(0)
        np.testing.assert_array_equal(context[:3],a['history'][index].T)
        np.testing.assert_array_equal(labels[:3],a['y'][index].T)
        assert torch.isnan(labels[3:]).all() and torch.isnan(future[:3]).all()
        # Preprocessor orders covariates lexicographically.
        expected=np.stack([item['past_covariates'][k][-48:] for k in sorted(item['past_covariates'])])
        np.testing.assert_array_equal(future[3:],expected)

def predict(pipeline,a,names,indices):
    pipeline.model.eval()
    out=np.full_like(a['y'],np.nan)
    for offset in range(0,len(indices),4):
        ids=indices[offset:offset+4]
        values=pipeline.predict_quantiles([example(a,names,int(i)) for i in ids],
            prediction_length=48,quantile_levels=[.5],context_length=168,
            batch_size=176,cross_learning=False)[0]
        for i,v in zip(ids,values,strict=True):
            out[i]=np.maximum(0,v.numpy()[:,:,0].T)
    return out

def score(pred,y,m):
    vals=[]
    for area in AREAS:
        for ti in range(3):
            mask=m.area.eq(area).to_numpy() & np.isfinite(pred).all((1,2))
            vals.append(float(np.abs(pred[mask,:,ti]-y[mask,:,ti]).sum()/y[mask,:,ti].sum()))
    return float(np.mean(vals))

def main(final=False,pilot=False):
    a,m,names=load_data()
    if not final:
        keep=(m.origin<pd.Timestamp('2026-10-01',tz='UTC')).to_numpy()
        a={k:v[keep] for k,v in a.items()};m=m.loc[keep].reset_index(drop=True)
    else:
        a['y'][m.origin>=pd.Timestamp('2026-10-01',tz='UTC')]=np.nan
    rev=json.loads((V1/'chronos_model.json').read_text())
    torch.manual_seed(42);np.random.seed(42)
    start=time.perf_counter()
    base=Chronos2Pipeline.from_pretrained(rev['model'],revision=rev['revision'],device_map='cpu',dtype=torch.float32)
    print('Loaded Chronos',time.perf_counter()-start,flush=True)
    runtime=[]
    if final:
        chosen=json.loads((HERE/'chronos_selection.json').read_text())['steps']
        settings=[chosen];folds=[('final','2026-10-01','2026-10-27')]
    else:
        settings=[100,300];folds=FOLDS
    if pilot:
        settings=[2];folds=FOLDS[:1]
    candidates={steps:np.full_like(a['y'],np.nan) for steps in settings}
    zero=np.full_like(a['y'],np.nan)
    for fold,start,end in folds:
        tr,va=masks(m,start,end);trainids=np.flatnonzero(tr);validids=np.flatnonzero(va)
        check_training_window(a,names,int(trainids[-1]))
        if not pilot:
            z=predict(base,a,names,validids);zero[va]=z[va]
        inputs=[example(a,names,int(i),True) for i in trainids]
        for steps in settings:
            directory=MODELS/f'chronos_{fold}_{steps}'
            begin=time.perf_counter()
            fitted=base.fit(inputs,prediction_length=48,finetune_mode='lora',
                context_length=168,min_past=168,learning_rate=1e-5,num_steps=steps,
                batch_size=88,output_dir=directory,optim='adamw_torch',
                seed=42,logging_steps=25,disable_tqdm=True,dataloader_pin_memory=False,
                callbacks=[FlushLogs()])
            elapsed=time.perf_counter()-begin
            assert isinstance(fitted.model,peft.PeftModel)
            fitted.model.peft_config['default'].revision=rev['revision']
            fitted.save_pretrained(directory/'finetuned-ckpt')
            if pilot:
                validids=validids[:2]
            result=predict(fitted,a,names,validids)
            candidates[steps][validids]=result[validids]
            durations=[]
            for _ in range(4):
                begin=time.perf_counter();predict(fitted,a,names,validids[:2]);durations.append(time.perf_counter()-begin)
            info=dict(fold=fold,steps=steps,training_seconds=elapsed,
                      warm_inference_seconds_two_areas=float(np.median(durations[1:])),
                      trainable_parameters=sum(p.numel() for p in fitted.model.parameters() if p.requires_grad),
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            runtime.append(info)
            print(json.dumps(info),flush=True)
            # Verify local checkpoint before treating it as an inference artifact.
            reloaded=Chronos2Pipeline.from_pretrained(directory/'finetuned-ckpt',device_map='cpu',dtype=torch.float32)
            check=predict(reloaded,a,names,validids[:2])
            np.testing.assert_allclose(check[validids[:2]],result[validids[:2]],rtol=1e-5,atol=.01)
            del fitted,reloaded;gc.collect()
        np.savez_compressed(CACHE/f'chronos_{"pilot" if pilot else "final" if final else "candidates"}.npz',
                            zero=zero,**{str(k):v for k,v in candidates.items()})
    stage='pilot' if pilot else 'final' if final else 'dev'
    (HERE/f'chronos_{stage}_runtime.json').write_text(json.dumps(runtime,indent=2))
    if pilot:
        return
    if not final:
        scores={k:score(v,a['y'],m) for k,v in candidates.items()}
        chosen=min(scores,key=scores.get)
        (HERE/'chronos_selection.json').write_text(json.dumps(dict(steps=chosen,scores=scores,covariates=[names['future'][i] for i in covariates(names)],model=rev),indent=2))
        print('Chronos selection',chosen,scores,flush=True)
    np.savez_compressed(CACHE/f'chronos_{stage}.npz',prediction=candidates[chosen])
    np.savez_compressed(CACHE/f'chronos_zero_{stage}.npz',prediction=zero)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');p.add_argument('--pilot',action='store_true')
    args=p.parse_args();main(args.final,args.pilot)
