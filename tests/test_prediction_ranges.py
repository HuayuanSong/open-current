"""Leakage, archived ranges and attribution reconstruction invariants."""
import copy
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from forecast import intervals
from forecast.explanations import stable
from forecast.publish import performance

class PredictionRangesTests(unittest.TestCase):
    def history(self):
        return [{'area':area,'origin':f'2026-10-{day:02}T12:00:00+00:00',
                 'lastLabel':f'2026-10-{day+2:02}T12:00:00+00:00',
                 'prediction':np.full((48,3),100.).tolist(),'actual':np.full((48,3),110.).tolist(),
                 'daylight':([True]*12+[False]*12)*2,'source':'issued'}
                for area in ['DK1','DK2'] for day in range(1,21)]

    def test_incomplete_future_outcomes_never_enter_calibration(self):
        at=pd.Timestamp('2026-10-15T12:10Z')
        rows=intervals.mature_history(self.history(),'DK1',at,14)
        self.assertTrue(all(pd.Timestamp(r['lastLabel'])<at-pd.Timedelta(minutes=15) for r in rows))
        self.assertNotIn('2026-10-13T12:00:00+00:00',[r['origin'] for r in rows])

    def test_too_little_calibration_fails_closed(self):
        with self.assertRaises(ValueError):intervals.mature_history(self.history()[:3],'DK1',pd.Timestamp('2026-10-27T12:00Z'),14)

    def test_solar_nights_are_absent_and_new_truth_cannot_change_saved_ranges(self):
        state={'rangeHistory':self.history()};p=np.full((2,48,3),100.)
        day=np.tile(np.array(([True]*12+[False]*12)*2),(2,1));at='2026-10-27T12:30:00+00:00'
        band=intervals.build(p,day,at,state);saved=json.dumps(band)
        for area in ['DK1','DK2']:
            for target in ['wind','solar','load']:
                b=band['areas'][area][target]
                for i,(lo,hi) in enumerate(zip(b['lower'],b['upper'])):
                    if target=='solar' and not day[0,i]:self.assertIsNone(lo);self.assertIsNone(hi)
                    else:self.assertLessEqual(lo,100);self.assertGreaterEqual(hi,100);self.assertGreaterEqual(lo,0)
        state['rangeHistory'][0]['actual']=np.full((48,3),1e8).tolist()
        self.assertEqual(json.dumps(band),saved)

    def test_live_coverage_counts_only_saved_ranges_of_current_version(self):
        version=intervals.validation()['version'];stamp='2026-10-27T11:00:00+00:00';issued='2026-10-27T10:30Z'
        record={'id':'new','modelVersion':'v','issuedAt':issued,'times':[stamp],
                'areas':{a:{t:[10] for t in ['wind','solar','load']} for a in ['DK1','DK2']},
                'ranges':{'version':version,'issuedAt':issued,'areas':{'DK1':{'wind':{'lower':[0],'upper':[20]},'solar':{'lower':[None],'upper':[None]}}}}}
        legacy=copy.deepcopy(record);legacy['id']='old';legacy.pop('ranges')
        late=copy.deepcopy(record);late['id']='retrofit';late['ranges']['issuedAt']='2026-10-27T12:00Z'
        state={'forecasts':[record,legacy,late],'hourly':[dict(area='DK1',time=stamp,wind=0,solar=0,load=100)]}
        result=performance(state,pd.Timestamp('2026-10-27T12:30Z'),'v')['DK1']
        self.assertEqual(result['wind']['rangeHours'],1);self.assertEqual(result['wind']['rangeCoverage'],1)
        self.assertEqual(result['solar']['rangeHours'],0)

    def test_reference_instability_is_not_published(self):
        reference=np.array([200.,-100.,2.,4.,8.]);other=np.array([-150.,50.,2.,4.,8.])
        self.assertFalse(stable(reference,other,2000)[0])
        self.assertTrue(stable(reference,reference.copy(),2000)[0])

    def test_published_range_validation_has_explicit_daylight_scope(self):
        validation=intervals.validation()
        for area in ['DK1','DK2']:
            self.assertEqual(validation['series'][area]['solar']['scope'],'daylight')
            self.assertTrue(all(r['enabled'] for r in validation['series'][area].values()))

if __name__=='__main__':unittest.main()
