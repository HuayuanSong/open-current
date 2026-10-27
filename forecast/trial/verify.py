"""Check challenger inference against saved research forecasts on any CPU."""
import json
import time
import numpy as np
from .artifacts import ensure,ROOT
from .score import predict

def main():
    start=time.perf_counter();bundle=ensure();protocol=json.loads((ROOT/'forecast/trial/protocol.json').read_text())
    fixture=dict(np.load(bundle/'fixture.npz'));expected=dict(np.load(bundle/'expected.npz'))
    out=predict(fixture,bundle,protocol);deltas={}
    for name,reference in expected.items():
        np.testing.assert_allclose(out[name],reference,rtol=2e-5,atol=.002,err_msg=name)
        deltas[name]=float(np.abs(out[name]-reference).max())
    for ti,target in enumerate(['wind','solar','load']):
        np.testing.assert_array_equal(out['primary'][:,:,ti],out[protocol['primaryRecipe'][target]][:,:,ti])
    print(json.dumps(dict(status='passed',maxDifferenceMW=deltas,seconds=time.perf_counter()-start)),flush=True)

if __name__=='__main__':main()
