"""Local experiment paths, shared with neither serving nor older experiments."""
from pathlib import Path
import os
import sys
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[name] = '4'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CACHE = ROOT / '.cache/forecasting/v3'
V2 = ROOT / '.cache/forecasting/v2'
V2_CODE = HERE.parent / 'v2'
CACHE.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
AREAS = ['DK1', 'DK2']
TARGETS = ['wind', 'solar', 'load']
BANDS = [(0, 6), (6, 24), (24, 48)]
FAMILIES = ['frozen','tree_control','ridge','lightgbm','xgboost','chronos_zero','chronos','ttm','ttm_scaled']

def digest(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    import json
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def load_v2():
    import json
    import numpy as np
    import pandas as pd
    a = dict(np.load(V2 / 'dataset.npz'))
    m = pd.read_csv(V2 / 'samples.csv')
    for key in ['origin','last_label','issue_time']:
        m[key] = pd.to_datetime(m[key], utc=True)
    names = json.loads((V2 / 'feature_names.json').read_text())
    return a, m, names
