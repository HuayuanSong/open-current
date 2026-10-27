"""Check today's exact weather inputs without issuing or backfilling forecasts."""
import json
import numpy as np
import pandas as pd
from forecast import data
from .artifacts import ROOT,ensure
from .sources import inputs

def main():
    p=json.loads((ROOT/'forecast/trial/protocol.json').read_text());at=data.now()
    # Probe only a run that is already initialized. This diagnostic is explicitly
    # not an issued forecast and never enters the prospective ledger.
    day=at.floor('D') if at.hour>=14 else at.floor('D')-pd.Timedelta(days=1)
    origin=day+pd.Timedelta(hours=18)
    meta=dict(origin=origin.isoformat(),weatherRun=day.isoformat(),times=[t.isoformat() for t in pd.date_range(origin+pd.Timedelta(hours=1),periods=48,freq='h')])
    a,sources,available,errors=inputs({},meta,ensure(),p)
    print(json.dumps(dict(diagnosticOnly=True,checkedAt=data.now().isoformat(),origin=origin.isoformat(),
        available=available,errors=errors,shapes={k:list(v.shape) for k,v in a.items()},
        regionalNativeHours=a.get('regional_native',np.zeros((2,48))).sum(1).tolist(),
        sources={k:{f:r[f] for f in ['model','run','retrievedAt','sha256']} for k,r in sources.items()}),indent=2))

if __name__=='__main__':main()
