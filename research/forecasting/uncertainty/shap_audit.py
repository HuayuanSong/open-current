"""Reference stability and coherent-feature checks for the tree component."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='4'
import itertools
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from forecast.features import tabular_features,AREAS,TARGETS
CACHE=ROOT/'.cache/forecasting/v2';HERE=Path(__file__).parent
GROUPS=['Recent electricity','Wind outlook','Sunlight/cloud outlook','Temperature outlook','Calendar/horizon']
REFERENCES=[(32,13),(64,13),(64,42),(64,79),(96,42)]

def grouping(names):
    result=[]
    for n in names:
        if n.startswith('wx_'):
            if 'wind' in n or 'power_curve' in n:result.append(1)
            elif 'temperature_2m' in n or 'heating' in n or 'cooling' in n:result.append(3)
            elif 'terrestrial' in n:result.append(4)
            else:result.append(2)
        elif n.startswith(('lag_','mean_','std_','max_')) or n in ['day_baseline','week_baseline','week_hour_mean']:result.append(0)
        else:result.append(4)
    return np.array(result)

def coherent_shap(model,x,background,names,group):
    """Exact five-group Shapley values; preserve derived-feature identities."""
    count=len(x);b=len(background);coalitions=[]
    for mask in range(32):
        selected=np.array([bool(mask&(1<<g)) for g in range(5)])[group]
        rows=np.where(selected[None,None,:],x[:,None,:],background[None,:,:]).copy()
        # Wind and history transformations move with their raw inputs. PV
        # temperature is the cross-group feature that must be regenerated.
        if bool(mask&(1<<2))!=bool(mask&(1<<3)):
            for point in range(3):
                pv=names.index(f'wx_{point}_pv_temperature');rad=names.index(f'wx_{point}_shortwave_radiation');temp=names.index(f'wx_{point}_temperature_2m')
                rows[:,:,pv]=rows[:,:,rad]*(1-.004*(rows[:,:,temp]-25))
        coalitions.append(model.predict(rows.reshape(-1,len(names)),num_threads=4).reshape(count,b).mean(1))
    values=np.asarray(coalitions);phi=np.zeros((count,5))
    for g in range(5):
        for mask in range(32):
            if mask&(1<<g):continue
            k=mask.bit_count();weight=math.factorial(k)*math.factorial(4-k)/math.factorial(5)
            phi[:,g]+=weight*(values[mask|(1<<g)]-values[mask])
    return phi,values[0],values[-1]

def compare(reference,other,scale):
    material=np.abs(reference)>.05*scale[:,None]
    sign=float((np.sign(reference[material])==np.sign(other[material])).mean()) if material.any() else 1.
    overlaps=[len(set(np.argsort(abs(a))[-2:])&set(np.argsort(abs(b))[-2:]))/2 for a,b in zip(reference,other)]
    change=np.abs(reference-other).sum(1)/np.maximum(np.abs(reference).sum(1),.1*scale)
    return dict(signAgreement=sign,topTwoOverlap=float(np.median(overlaps)),medianChange=float(np.median(change)),p90Change=float(np.quantile(change,.9)))

def main():
    began=time.monotonic();a=dict(np.load(CACHE/'dataset.npz'));m=pd.read_csv(CACHE/'samples.csv');names=json.loads((CACHE/'feature_names.json').read_text())['future']
    selected_leads=[0,5,11,23,47];output=[];details={}
    for ti,target in enumerate(TARGETS):
        x,extra=tabular_features(a['history'],a['future'],ti);all_names=names+extra;group=grouping(all_names)
        for area in AREAS:
            current=np.flatnonzero(m.area.eq(area)&m.split.eq('test'))
            samples=np.array([i*48+h for i in current for h in selected_leads]);xs=x[samples]
            pre=np.flatnonzero(m.area.eq(area)&m.split.eq('calibration'))
            historical=np.array([i*48+h for i in pre for h in range(48)])
            model=lgb.Booster(model_file=str(CACHE/'models'/f'tree_control_{area}_{target}.txt'))
            prediction=model.predict(xs,num_threads=4);raw_negative=int((prediction<0).sum())
            methods=[np.zeros((len(samples),5)) for _ in REFERENCES];bases=[np.zeros(len(samples)) for _ in REFERENCES]
            coherent=np.zeros_like(methods[0]);coherent_base=np.zeros(len(samples));max_error=0.;reference_rows=[];insufficient=0
            # Match reference lead band; for solar match daylight as well.
            day=a['daylight'].reshape(-1).astype(bool);leads=samples%48
            keys=np.where(leads<6,0,np.where(leads<24,1,2))*2+(day[samples].astype(int) if target=='solar' else 0)
            hlead=historical%48;hkeys=np.where(hlead<6,0,np.where(hlead<24,1,2))*2+(day[historical].astype(int) if target=='solar' else 0)
            for key in np.unique(keys):
                subset=np.flatnonzero(keys==key);pool=historical[hkeys==key]
                if len(pool)<32:
                    insufficient+=len(subset);continue
                for j,(size,seed) in enumerate(REFERENCES):
                    ids=np.random.default_rng(seed).choice(pool,min(size,len(pool)),replace=False);background=x[ids]
                    explainer=shap.TreeExplainer(model,data=background,feature_perturbation='interventional',model_output='raw')
                    values=explainer.shap_values(xs[subset],check_additivity=True)
                    methods[j][subset]=np.stack([values[:,group==g].sum(1) for g in range(5)],axis=1)
                    bases[j][subset]=float(explainer.expected_value)
                    max_error=max(max_error,float(np.max(abs(values.sum(1)+explainer.expected_value-prediction[subset]))))
                    if (size,seed)==(64,42):
                        phi,base,exact=coherent_shap(model,xs[subset],background,all_names,group)
                        coherent[subset]=phi;coherent_base[subset]=base
                        max_error=max(max_error,float(np.max(abs(phi.sum(1)+base-prediction[subset]))))
                        reference_rows.extend(ids.tolist())
            reference=methods[2];scale=np.full(len(samples),max(1.,float(np.median(abs(prediction)))))
            informative=np.abs(reference).sum(1)>.1*scale
            comparisons=[compare(reference[informative],value[informative],scale[informative]) for j,value in enumerate(methods) if j!=2]
            structure=compare(reference[informative],coherent[informative],scale[informative])
            stable=all(c['signAgreement']>=.9 and c['topTwoOverlap']>=.75 and c['medianChange']<=.15 and c['p90Change']<=.30 for c in comparisons)
            structure_ok=structure['topTwoOverlap']>=.75 and structure['signAgreement']>=.9
            row=dict(area=area,target=target,examples=len(samples),informative=int(informative.sum()),rawNegative=raw_negative,
                     maxReconstructionErrorMW=max_error,referenceStable=stable,coherentAgreement=structure_ok,
                     insufficientReferences=insufficient,publish=bool(stable and structure_ok and max_error<=.1 and not insufficient),referenceComparisons=comparisons,coherentComparison=structure)
            output.append(row);details[area+'_'+target]={'rows':samples.tolist(),'prediction':prediction.tolist(),'baseline':bases[2].tolist(),
                'groups':reference.tolist(),'coherentGroups':coherent.tolist(),'coherentBaseline':coherent_base.tolist(),'referenceRows':reference_rows}
            print(json.dumps(row),flush=True)
    report={'shapVersion':shap.__version__,'component':'tree_control','groups':GROUPS,'references':REFERENCES,
            'seconds':time.monotonic()-began,'series':output,'publish':all(r['publish'] for r in output)}
    (HERE/'shap-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    (ROOT/'.cache/forecasting/uncertainty/shap-details.json').write_text(json.dumps(details))
    print('Audit complete',report['seconds'],'seconds; publish:',report['publish'],flush=True)

if __name__=='__main__':main()
