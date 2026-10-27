"""Shared experiment paths and time boundaries. No ML runtime imports."""
from pathlib import Path
import sys
import os
for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[variable] = '4'
HERE = Path(__file__).resolve().parent
V1 = HERE.parent
ROOT = V1.parents[1]
OLD = ROOT / '.cache/forecasting'
CACHE = OLD / 'v2'
MODELS = CACHE / 'models'
MODELS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(V1))
os.environ['HF_HOME'] = str(OLD / 'huggingface')
H = 48
TARGETS = ['wind', 'solar', 'load']
AREAS = ['DK1', 'DK2']
FOLDS = [('june_july', '2026-06-01', '2026-08-01'),
         ('august_september', '2026-08-01', '2026-10-01')]

def load_data():
    import numpy as np
    import pandas as pd
    import json
    a = dict(np.load(CACHE / 'dataset.npz'))
    m = pd.read_csv(CACHE / 'samples.csv')
    for key in ['origin', 'last_label']:
        m[key] = pd.to_datetime(m[key], utc=True)
    names = json.loads((CACHE / 'feature_names.json').read_text())
    return a, m, names

def masks(meta, start, end):
    import pandas as pd
    start, end = pd.Timestamp(start, tz='UTC'), pd.Timestamp(end, tz='UTC')
    return ((meta.last_label < start).to_numpy(),
            ((meta.origin >= start) & (meta.last_label < end)).to_numpy())
