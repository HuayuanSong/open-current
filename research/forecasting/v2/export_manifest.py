"""Describe and fingerprint trained artifacts without publishing anything."""
from common import *
import hashlib
import json

def digest(path):
    return {'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def main():
    selection=json.loads((HERE/'final_selection.json').read_text())
    families={}
    for family in ['tree_control','ridge','lightgbm','xgboost']:
        extension='ubj' if family=='xgboost' else 'joblib' if family=='ridge' else 'txt'
        paths=[MODELS/f'{family}_{a}_{t}.{extension}' for a in AREAS for t in TARGETS]
        families[family]={'environment':'requirements.lock.txt','files':[digest(p) for p in paths]}
    steps=json.loads((HERE/'chronos_selection.json').read_text())['steps']
    folder=MODELS/f'chronos_final_{steps}'/'finetuned-ckpt'
    families['chronos']={'environment':'requirements.lock.txt',
        'base_model':json.loads((V1/'chronos_model.json').read_text()),
        'files':[digest(p) for p in folder.iterdir() if p.is_file()]}
    for family in ['ttm','ttm_scaled']:
        config=json.loads((HERE/f'{family}_selection.json').read_text())
        folder=MODELS/f'{family}_final_{config["learning_rate"]}_{config["epochs"]}'
        families[family]={'environment':'requirements-ttm.lock.txt','files':[digest(p) for p in folder.iterdir() if p.is_file()]}
    families['chronos_zero']={'base_model':json.loads((V1/'chronos_model.json').read_text())}
    families['frozen']={'selection':str((V1/'selected_models.csv').relative_to(ROOT)),
        'weights_directory':str((OLD/'models').relative_to(ROOT)),
        'integrity_manifest':str((HERE/'original_artifacts.json').relative_to(ROOT))}
    manifest={'scope':'Local research artifacts; no deployment or live scoring endpoint',
        'forecast_horizon_hours':48,'issue_hour_utc':12,'assumed_publication_buffer_minutes':15,
        'data_config':json.loads((V1/'config.json').read_text()),
        'feature_order':json.loads((CACHE/'feature_names.json').read_text()),
        'selection':selection,'families':families,
        'source_files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')},
        'protocol':digest(HERE/'PROTOCOL.md'),'data_manifest':digest(CACHE/'manifest.json')}
    (HERE/'artifacts.json').write_text(json.dumps(manifest,indent=2))
    print('Wrote local artifact manifest with file hashes and pinned model revisions.')

if __name__=='__main__':
    main()
