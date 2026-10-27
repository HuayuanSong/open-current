"""Exercise real SHAP scoring on the deployment runner, even for reused origins."""
import json
import subprocess
import sys
import numpy as np
from .artifacts import ensure,ROOT

def main():
    bundle=ensure();a=dict(np.load(bundle/'fixture.npz'))
    ids=[int(np.flatnonzero(a['area_ids']==ai)[0]) for ai in [0,1]]
    work=ROOT/'.cache/forecast-serving/verify';work.mkdir(parents=True,exist_ok=True)
    inputs=work/'explanation-inputs.npz';output=work/'explanation-check.npz'
    np.savez_compressed(inputs,**{k:v[ids] for k,v in a.items()})
    subprocess.run([sys.executable,'-m','forecast.worker','trees',str(inputs),str(output),'--explain'],check=True,cwd=ROOT,timeout=180)
    result=json.loads(output.with_suffix('.explanations.json').read_text())
    assert result['component']=='tree_control'
    counts={}
    for area in ['DK1','DK2']:
        items=result['areas'][area]['load'];assert [i['lead'] for i in items]==[1,6,12,24,48]
        ready=[i for i in items if i['status']=='ready'];assert ready,area
        for item in ready:
            values=[c['mw'] for c in item['contributions']]
            assert np.isfinite(values).all()
            assert abs(item['baselineMW']+sum(values)-item['predictionMW'])<=.1
        counts[area]=len(ready)
    print(json.dumps({'explanationVerification':'passed','stableHours':counts}))

if __name__=='__main__':main()
