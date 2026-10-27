"""Build the immutable release bundle from the completed local experiment."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / '.cache/forecasting'
RESEARCH = ROOT / 'research/forecasting'
VERSION = 'ensemble-v2-2026-10-27'

def main():
    bundle = CACHE / 'release' / VERSION
    bundle.mkdir(parents=True, exist_ok=True)
    def copy(source, target):
        dest = bundle / target
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    for family in ['tree_control', 'lightgbm', 'xgboost']:
        ext = 'ubj' if family == 'xgboost' else 'txt'
        for path in (CACHE/'v2/models').glob(f'{family}_*.{ext}'):
            copy(path, 'trees/'+path.name)
    for area,target in [('DK1','wind'),('DK1','solar'),('DK2','wind'),('DK2','solar')]:
        copy(CACHE/'models'/f'lightgbm_{area}_{target}.txt', f'original/lightgbm_{area}_{target}.txt')
    for target in ['wind','solar']:
        copy(CACHE/'models'/f'mlp_{target}_128.pt', f'original/mlp_{target}.pt')
    # Portable ridge parameters avoid sklearn pickle/version dependencies in serving.
    for group,folder,targets in [('ridge',CACHE/'v2/models',['wind','solar','load']),
                                  ('original',CACHE/'models',['load'])]:
        for area in ['DK1','DK2']:
            for target in targets:
                pipeline=joblib.load(folder/f'ridge_{area}_{target}.joblib')
                scaler,model=pipeline.steps[0][1],pipeline.steps[1][1]
                out=bundle/group/f'ridge_{area}_{target}.json';out.parent.mkdir(exist_ok=True)
                out.write_text(json.dumps(dict(mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),
                    coef=model.coef_.tolist(),intercept=float(model.intercept_))))
    for name,source in [('chronos',CACHE/'v2/models/chronos_final_300/finetuned-ckpt'),
                         ('ttm',CACHE/'v2/models/ttm_final_0.0003_40'),
                         ('ttm_scaled',CACHE/'v2/models/ttm_scaled_final_0.0003_40')]:
        for p in source.iterdir():
            if p.is_file(): copy(p, name+'/'+p.name)
    for source,name in [(RESEARCH/'v2/final_selection.json','selection.json'),
                        (CACHE/'v2/feature_names.json','feature_names.json'),
                        (RESEARCH/'chronos_model.json','chronos_model.json')]:
        copy(source,name)
    for path in (ROOT/'forecast/licenses').iterdir():
        copy(path,'licenses/'+path.name)
    data=dict(np.load(CACHE/'v2/dataset.npz'))
    meta=pd.read_csv(CACHE/'v2/samples.csv')
    ids=np.flatnonzero(meta.split.eq('test').to_numpy())[[0,1,-2,-1]]
    fixture={k:data[k][ids] for k in ['history','future','history512','observed512','past_full512']}
    fixture['area_ids']=meta.area_id.to_numpy()[ids]
    np.savez_compressed(bundle/'fixture.npz',**fixture)
    selection=json.loads((bundle/'selection.json').read_text())
    predictions={name:np.load(CACHE/'v2'/f'{name}_final.npz')['prediction'][ids]
                 for name in selection['families']}
    np.savez_compressed(bundle/'expected.npz',**predictions)
    (bundle/'NOTICE.txt').write_text('Open Current ensemble v2. Project code: MIT. Chronos-2 and TinyTimeMixer base models and adaptations: Apache-2.0.\nBase model revisions and attribution: https://huggingface.co/amazon/chronos-2 and https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2.\nBenchmark and data methodology: https://github.com/HuayuanSong/open-current/tree/main/research/forecasting/v2\n')
    files={str(p.relative_to(bundle)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in bundle.rglob('*') if p.is_file() and p.name!='manifest.json'}
    (bundle/'manifest.json').write_text(json.dumps(dict(version=VERSION,files=files),indent=2))
    archive=bundle.parent/f'{VERSION}.tar.gz'
    with tarfile.open(archive,'w:gz') as output:
        output.add(bundle,arcname=VERSION)
    release=dict(version=VERSION,url=f'https://github.com/HuayuanSong/open-current/releases/download/{VERSION}/{archive.name}',
                 sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),bytes=archive.stat().st_size)
    (ROOT/'forecast/model-release.json').write_text(json.dumps(release,indent=2)+'\n')
    print(json.dumps(release,indent=2))

if __name__=='__main__': main()
