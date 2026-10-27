"""Prospective timing, immutable archives, matched scoring and model handoff."""
import copy
import json
import unittest
import tempfile
from unittest.mock import patch,Mock
from pathlib import Path
import numpy as np
import pandas as pd
from forecast.features import AREAS,TARGETS
from forecast.trial.ledger import initialize,eligible_origin,append,freeze_observations,review_after
from forecast.trial.evaluate import metrics
from forecast.trial.features import regional_tail,fleet_array
from forecast.data import restore

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=json.loads((ROOT/'forecast/trial/protocol.json').read_text())

def record(day='2026-10-27',candidate=12):
    origin=pd.Timestamp(day,tz='UTC')+pd.Timedelta(hours=18)
    values=lambda v:{a:{t:[float(v)]*48 for t in TARGETS} for a in AREAS}
    return dict(id=PROTOCOL['version']+'/'+origin.isoformat(),trialVersion=PROTOCOL['version'],
        baselineVersion=PROTOCOL['baselineVersion'],origin=origin.isoformat(),issuedAt=(origin+pd.Timedelta(minutes=37)).isoformat(),
        times=[t.isoformat() for t in pd.date_range(origin+pd.Timedelta(hours=1),periods=48,freq='h')],
        predictions={'baseline':values(20),'primary':values(candidate)},daylight={a:[True]*48 for a in AREAS})

class TrialTests(unittest.TestCase):
    def setUp(self):self.trial=initialize({},copy.deepcopy(PROTOCOL))

    def test_fixed_daily_issue_window_and_timezone(self):
        self.assertIsNone(eligible_origin(pd.Timestamp('2026-10-27T18:07Z'),PROTOCOL))
        self.assertIsNone(eligible_origin(pd.Timestamp('2026-10-27T19:07Z'),PROTOCOL))
        self.assertEqual(eligible_origin(pd.Timestamp('2026-10-27T19:37+01:00'),PROTOCOL),pd.Timestamp('2026-10-27T18:00Z'))
        self.assertIsNone(eligible_origin(pd.Timestamp('2026-12-08T18:37Z'),PROTOCOL))

    def test_no_backdated_or_late_forecast(self):
        r=record()
        with self.assertRaisesRegex(ValueError,'prospective'):append(self.trial,r,pd.Timestamp('2026-10-28T18:37Z'))
        with self.assertRaisesRegex(ValueError,'prospective'):append(self.trial,r,pd.Timestamp('2026-10-27T19:01Z'))
        r['issuedAt']='2026-10-27T18:36Z'
        with self.assertRaisesRegex(ValueError,'timestamp'):append(self.trial,r,pd.Timestamp('2026-10-27T18:37Z'))

    def test_duplicate_never_replaces_values(self):
        r=record();self.assertTrue(append(self.trial,r,pd.Timestamp(r['issuedAt'])))
        again=record(candidate=999);again['issuedAt']='2026-10-27T18:45:00+00:00'
        self.assertFalse(append(self.trial,again,pd.Timestamp(again['issuedAt'])))
        self.assertEqual(self.trial['records'][0]['predictions']['primary']['DK1']['wind'][0],12)

    def test_protocol_and_records_cannot_change(self):
        state={};t=initialize(state,PROTOCOL);r=record();append(t,r,pd.Timestamp(r['issuedAt']))
        changed={**PROTOCOL,'minimumCompleteOrigins':2}
        with self.assertRaisesRegex(ValueError,'protocol'):initialize(state,changed)
        t['records'][0]['predictions']['primary']['DK1']['wind'][0]=999
        with self.assertRaisesRegex(ValueError,'modified'):initialize(state,PROTOCOL)

    def test_invalid_horizon_or_baseline_rejected(self):
        r=record();r['times'][0]=r['origin']
        with self.assertRaisesRegex(ValueError,'targets'):append(self.trial,r,pd.Timestamp(r['issuedAt']))
        r=record();r['predictions']['baseline']['DK1']['wind']=None
        with self.assertRaisesRegex(ValueError,'Baseline'):append(self.trial,r,pd.Timestamp(r['issuedAt']))

    def test_observation_age_revision_zero_and_timeout(self):
        r=record();append(self.trial,r,pd.Timestamp(r['issuedAt']));stamp=pd.Timestamp(r['times'][0])
        row=dict(time=stamp.isoformat(),area='DK1',wind=0,solar=0,load=100)
        freeze_observations(self.trial,[row],stamp+pd.Timedelta(hours=23))
        self.assertFalse(self.trial['observations'])
        freeze_observations(self.trial,[row],stamp+pd.Timedelta(hours=24))
        freeze_observations(self.trial,[{**row,'wind':999}],stamp+pd.Timedelta(hours=25))
        self.assertEqual(self.trial['observations'][stamp.isoformat()]['DK1']['wind']['value'],0)
        self.assertEqual(self.trial['observations'][stamp.isoformat()]['DK1']['solar']['value'],0)
        late={**row,'time':r['times'][1]}
        freeze_observations(self.trial,[late],pd.Timestamp(late['time'])+pd.Timedelta(hours=73))
        self.assertNotIn(late['time'],self.trial['observations'])

    def fill(self,r,omit_last=False):
        for t in r['times'][:-1] if omit_last else r['times']:
            hourly=[dict(time=t,area=a,wind=10,solar=10,load=10) for a in AREAS]
            freeze_observations(self.trial,hourly,pd.Timestamp(t)+pd.Timedelta(hours=24))

    def test_complete_paired_trajectory_and_known_improvement(self):
        r=record();append(self.trial,r,pd.Timestamp(r['issuedAt']));self.fill(r,True)
        at=pd.Timestamp(r['times'][-1])+pd.Timedelta(hours=25)
        stats=metrics(self.trial,at)['metrics']['primary']['DK1']['wind']
        self.assertEqual(stats['completeOrigins'],0);self.assertIsNone(stats['maeReductionPct'])
        self.fill(r);stats=metrics(self.trial,at)['metrics']['primary']['DK1']['wind']
        self.assertEqual(stats['completeOrigins'],1);self.assertAlmostEqual(stats['baselineMAEMW'],10)
        self.assertAlmostEqual(stats['candidateMAEMW'],2);self.assertAlmostEqual(stats['maeReductionPct'],80)
        self.assertEqual(stats['decision'],'collecting')

    def test_future_captures_are_not_scored_early(self):
        r=record();append(self.trial,r,pd.Timestamp(r['issuedAt']));self.fill(r)
        stats=metrics(self.trial,pd.Timestamp(r['times'][-1]))['metrics']['primary']['DK1']['wind']
        self.assertEqual(stats['completeOrigins'],0)

    def test_missing_candidate_has_no_baseline_substitution(self):
        r=record();r['predictions']['primary']['DK1']['solar']=None
        append(self.trial,r,pd.Timestamp(r['issuedAt']));self.fill(r)
        out=metrics(self.trial,pd.Timestamp(r['times'][-1])+pd.Timedelta(hours=25))
        self.assertEqual(out['metrics']['primary']['DK1']['solar']['issuedOrigins'],0)
        self.assertEqual(out['metrics']['primary']['DK1']['wind']['completeOrigins'],1)

    def test_end_requires_enough_evidence_and_never_promotes(self):
        out=metrics(self.trial,review_after(PROTOCOL)+pd.Timedelta(hours=1))
        self.assertEqual(out['scheduledOrigins'],42);self.assertFalse(out['automaticPromotion'])
        self.assertEqual(out['metrics']['primary']['DK1']['wind']['decision'],'insufficient_data')

    def test_regional_tail_only_bridges_the_prespecified_final_hour(self):
        times=pd.date_range('2026-10-27T19:00Z',periods=48,freq='h');run=pd.Timestamp('2026-10-27T06:00Z')
        regional=np.ones((2,48,5));regional[:,-1]=np.nan;ecmwf=np.full_like(regional,2)
        result,native=regional_tail(regional,ecmwf,run,times)
        np.testing.assert_array_equal(result[:,-1],2);np.testing.assert_array_equal(native.sum(1),47)
        regional[0,5]=np.nan
        with self.assertRaisesRegex(ValueError,'interior'):regional_tail(regional,ecmwf,run,times)

    def test_missing_published_archive_blocks_restore(self):
        response=Mock(status_code=404)
        with patch('forecast.data.requests.get',return_value=response),patch('forecast.data.read_state',return_value={}):
            with self.assertRaisesRegex(ValueError,'refusing'):restore('https://example.test',require_existing=True)

    def test_issue_writes_audit_evidence_without_changing_public_forecasts(self):
        from forecast.trial.run import issue
        at=pd.Timestamp('2026-10-27T18:37Z');r=record();names=['wx_0_terrestrial_radiation']
        arrays={'future':np.ones((2,48,1))*20,'area_ids':np.array([0,1])}
        meta=dict(origin=r['origin'],weatherRun='2026-10-27T00:00Z',times=r['times'],featureNames=names)
        state={'weather':{},'forecasts':[{'id':'public','values':[123]}]};original=copy.deepcopy(state['forecasts'])
        values=np.full((2,48,3),10.)
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);(folder/'feature-names.json').write_text(json.dumps({'future':names}))
            with patch('forecast.trial.run.WORK',folder/'evidence'),patch('forecast.trial.run.ensure',return_value=folder),\
                 patch('forecast.trial.run.data.inputs',return_value=(arrays,meta)),patch('forecast.trial.run.data.now',return_value=at),\
                 patch('forecast.trial.run.weather_inputs',return_value=(arrays,{},['newer_weather'],{})),\
                 patch('forecast.trial.run.baseline_score',return_value=(values,{},[])),\
                 patch('forecast.trial.run.predict',return_value={'primary':values.copy()}):
                self.assertTrue(issue(state,self.trial,at))
            self.assertTrue((folder/'evidence/record.json').exists())
            self.assertTrue((folder/'evidence/inputs.npz').exists())
        self.assertEqual(state['forecasts'],original)
        self.assertEqual(len(self.trial['records'][0]['daylight']['DK1']),48)
        self.assertTrue(all(self.trial['records'][0]['daylight']['DK1']))

    def test_background_tick_initializes_summary_without_issuing(self):
        from forecast.trial.run import main
        state=dict(updatedAt='2026-10-27T12:00Z',hourly=[dict(area='DK1',time='2026-10-27T12:00Z',wind=10,solar=10,load=10)],forecasts=[{'id':'public'}])
        written=[]
        with tempfile.TemporaryDirectory() as temp:
            summary=Path(temp)/'trial.json'
            with patch('forecast.trial.run.SUMMARY',summary),patch('forecast.trial.run.data.read_state',return_value=state),\
                 patch('forecast.trial.run.data.now',return_value=pd.Timestamp('2026-10-27T15:00Z')),\
                 patch('forecast.trial.run.data.write_state',side_effect=lambda s:written.append(copy.deepcopy(s))),\
                 patch('sys.argv',['trial']),patch.dict('os.environ',{},clear=True):
                main()
            result=json.loads(summary.read_text())
        self.assertEqual(result['status'],'waiting_for_first_issue');self.assertEqual(result['issuedOrigins'],0)
        self.assertEqual(written[0]['forecasts'],[{'id':'public'}])

if __name__=='__main__':unittest.main()
