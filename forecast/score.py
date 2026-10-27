"""Run fixed model families and apply the exact development-selected weights."""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from .artifacts import ensure,ROOT

def combine(predictions,area_ids,selection):
    original=predictions['original_trees'].copy()
    for ai,ti in [(0,0),(1,1)]:
        mask=area_ids==ai
        original[mask,:,ti]=.5*original[mask,:,ti]+.5*predictions['original_neural'][mask,:,ti]
    predictions={**predictions,'frozen':original}
    out=np.full_like(original,np.nan)
    for rule in selection['selection']:
        ai=['DK1','DK2'].index(rule['area']);ti=['wind','solar','load'].index(rule['target']);mask=area_ids==ai
        if rule['selected']!='blend':
            out[mask,:,ti]=predictions[rule['selected']][mask,:,ti];continue
        for (lo,hi),weights in zip(selection['bands'],rule['weights'],strict=True):
            out[mask,lo:hi,ti]=sum(w*predictions[f][mask,lo:hi,ti] for f,w in zip(selection['families'],weights,strict=True))
    if not np.isfinite(out).all():raise ValueError('Incomplete ensemble; no weight substitution is allowed')
    return out,predictions

def run(input_path,work):
    bundle=ensure();work.mkdir(parents=True,exist_ok=True)
    python=os.environ.get('FORECAST_MAIN_PYTHON',sys.executable)
    ttm_python=os.environ.get('FORECAST_TTM_PYTHON',str(ROOT/'.cache/forecasting/v2/ttm-venv/bin/python'))
    predictions={};timings=[]
    for family,interpreter in [('trees',python),('neural',python),('ttm',ttm_python)]:
        output=work/f'{family}.npz'
        subprocess.run([interpreter,'-m','forecast.worker',family,str(input_path),str(output)],cwd=ROOT,check=True,timeout=360)
        predictions.update(dict(np.load(output)));timings.append(json.loads(output.with_suffix('.json').read_text()))
    ids=np.load(input_path)['area_ids'];selection=json.loads((bundle/'selection.json').read_text())
    out,all_predictions=combine(predictions,ids,selection)
    return out,all_predictions,timings

def main():
    bundle=ensure();out,predictions,timings=run(bundle/'fixture.npz',ROOT/'.cache/forecast-serving/verify')
    expected=dict(np.load(bundle/'expected.npz'));deltas={}
    for name,reference in expected.items():
        np.testing.assert_allclose(predictions[name],reference,rtol=2e-5,atol=.02,err_msg=name)
        deltas[name]=float(np.abs(predictions[name]-reference).max())
    print(json.dumps({'historicalParity':'passed','maxDifferenceMW':deltas,'timings':timings},indent=2))

if __name__=='__main__':main()
