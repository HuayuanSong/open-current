"""Causal missing-data handling and honest post-issue evaluation."""
import unittest
import numpy as np
import pandas as pd
from forecast.data import fill_history
from forecast.publish import performance,payload

class ForecastServingTests(unittest.TestCase):
    def history(self):
        index=pd.date_range('2026-09-01',periods=800,freq='h',tz='UTC')
        frame=pd.DataFrame({k:np.arange(800,dtype=float)+100 for k in ['wind','solar','load']},index=index)
        return frame,index[-1]

    def test_limited_older_gaps_use_past_only(self):
        frame,origin=self.history();frame.iloc[-20]=np.nan
        values,mask,count=fill_history(frame,origin)
        self.assertEqual(count,1);self.assertFalse(mask[-20].any())
        np.testing.assert_array_equal(values[-20],frame.iloc[-188].to_numpy())

    def test_latest_missing_hour_is_never_filled(self):
        frame,origin=self.history();frame.iloc[-1]=np.nan
        with self.assertRaisesRegex(ValueError,'recent'):fill_history(frame,origin)

    def test_long_gap_stops_update(self):
        frame,origin=self.history();frame.iloc[-20:-16]=np.nan
        with self.assertRaisesRegex(ValueError,'Too many'):fill_history(frame,origin)

    def test_mature_scores_preserve_zero_and_exclude_unknown_observations(self):
        at=pd.Timestamp('2026-10-27T12:30Z');times=['2026-10-27T11:00:00+00:00','2026-10-27T13:00:00+00:00']
        values={k:[10,999] for k in ['wind','solar','load']}
        record=dict(id='test',modelVersion='v1',issuedAt='2026-10-27T10:30Z',origin='2026-10-27T10:00Z',times=times,areas={'DK1':values,'DK2':values})
        state={'forecasts':[record], 'hourly':[dict(area='DK1',time=times[0],wind=0,solar=None,load=100)]}
        stats=performance(state,at,'v1')
        self.assertEqual(stats['DK1']['wind']['forecastHours'],1)
        self.assertEqual(stats['DK1']['wind']['maeMW'],10)
        self.assertIsNone(stats['DK1']['wind']['wapePct'])
        self.assertEqual(stats['DK1']['solar']['forecastHours'],0)
        self.assertEqual(performance(state,at,'v2')['DK1']['load']['forecastHours'],0)

    def test_failure_preserves_old_forecast_and_issue_time(self):
        at=pd.Timestamp('2026-10-27T12:30Z')
        record=dict(id='test',modelVersion='v1',issuedAt='2026-10-27T10:30Z',origin='2026-10-27T10:00Z',times=[],areas={})
        state={'forecasts':[record],'hourly':[]}
        result=payload(state,at,'v1','weather unavailable')
        self.assertEqual(result['status'],'delayed');self.assertEqual(result['latest']['issuedAt'],record['issuedAt'])

    def test_late_issue_never_scores_already_elapsed_target_hours(self):
        stamp='2026-10-27T11:00:00+00:00'
        record=dict(id='late',modelVersion='v1',issuedAt='2026-10-27T11:30Z',
                    times=[stamp],areas={a:{t:[10] for t in ['wind','solar','load']} for a in ['DK1','DK2']})
        state={'forecasts':[record],'hourly':[dict(area='DK1',time=stamp,wind=10,solar=10,load=10)]}
        result=performance(state,pd.Timestamp('2026-10-27T12:30Z'),'v1')
        self.assertEqual(result['DK1']['wind']['forecastHours'],0)

if __name__=='__main__':unittest.main()
