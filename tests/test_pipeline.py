import copy
import json
from pathlib import Path
import sys
import unittest
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from normalize import normalize_power, normalize_flows, timestamp, merge
import normalize


class NormalizationTests(unittest.TestCase):
    def power(self, values, time='2026-10-25T02:15:00+02:00'):
        return {'unit': 'MW', 'series': [{'id': key} for key in values],
                'data': [{'timestamp': time, 'values': values}]}

    def test_storage_and_auxiliary_series_do_not_inflate_shares(self):
        raw = self.power({'wind_onshore': 400, 'nuclear': 400, 'fossil_gas': 200,
                          'hydro_pumped_storage': 500, 'load': 1600,
                          'renewable_share_of_generation': 50})
        row = normalize_power(raw, 10**15)[0]
        self.assertEqual(row['total'], 1000)
        self.assertEqual(row['renewable'], 40)
        self.assertEqual(row['lowCarbon'], 80)

    def test_missing_generation_is_not_zero(self):
        row = normalize_power(self.power({'wind_onshore': 400, 'solar': None}), 10**15)[0]
        self.assertIsNone(row['total'])
        self.assertIsNone(row['renewable'])
        self.assertIn('solar', row['missing'])

    def test_zero_is_valid_and_mixed_waste_is_not_assumed_renewable(self):
        row = normalize_power(self.power({'solar': 0, 'wind_onshore': 50, 'waste': 50}), 10**15)[0]
        self.assertEqual(row['total'], 100)
        self.assertEqual(row['renewable'], 50)

    def test_incomplete_group_is_not_partially_summed(self):
        row = normalize_power(self.power({'wind_onshore': 50, 'wind_offshore': None}), 10**15)[0]
        self.assertIsNone(row['g']['wind'])

    def test_unknown_series_fails_instead_of_changing_the_denominator(self):
        with self.assertRaises(ValueError):
            normalize_power(self.power({'wind_onshore': 400, 'new_generator_type': 300}), 10**15)

    def test_offset_preserves_two_fall_back_hours(self):
        first = timestamp('2026-10-25T02:15:00+02:00')
        second = timestamp('2026-10-25T02:15:00+01:00')
        self.assertEqual(second-first, 3600000)
        with self.assertRaises(ValueError):
            timestamp('2026-10-25T02:15:00')

    def test_flows_convert_gw_to_mw_and_keep_sign(self):
        raw = {'unit': 'GW', 'series': [{'id': k, 'name': k.title()} for k in ['germany','sweden','sum']],
               'data': [{'timestamp': '2026-10-25T03:00:00Z', 'values': {'germany': -1.2, 'sweden': .5, 'sum': -.7}}]}
        row = normalize_flows(raw, 10**15)[0]
        self.assertEqual(row['net'], -700)
        self.assertEqual(row['borders'][0]['mw'], -1200)
        self.assertEqual(row['borders'][0]['code'], 'de')
        raw['data'][0]['values']['sweden'] = None
        self.assertIsNone(normalize_flows(raw, 10**15)[0]['net'])

    def test_future_actual_observations_are_not_selected(self):
        raw = self.power({'solar': 100})
        self.assertEqual(normalize_power(raw, timestamp('2020-01-01T00:00:00Z')), [])

    def test_revision_replaces_timestamp_and_outage_retains_history(self):
        old = [{'t': 1, 'v': 1}, {'t': 2, 'v': 2}, {'t': 3, 'v': 3}]
        self.assertEqual(merge(old, [], 2), old[1:])
        self.assertEqual(merge(old, [{'t': 2, 'v': 20}], 2), [{'t': 2, 'v': 20}, old[2]])

    def test_danish_area_balance_avoids_double_counting_bornholm(self):
        record = {'PriceArea': 'DK2', 'Minutes5UTC': '2026-10-25T10:00:00',
                  'ProductionLt100MW': 100, 'ProductionGe100MW': 100,
                  'OffshoreWindPower': 100, 'OnshoreWindPower': 100, 'SolarPower': 100,
                  'ExchangeGreatBelt': 50, 'ExchangeGermany': -100, 'ExchangeSweden': 25,
                  'BornholmSE4': 10, 'ExchangeNorway': None, 'ExchangeNetherlands': None,
                  'ExchangeGreatBritain': None}
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base/'cache/energinet').mkdir(parents=True)
            (base/'data').mkdir()
            (base/'cache/energinet/ElectricityProdex5MinRealtime.json').write_text(json.dumps({'records':[record]}))
            with patch.object(normalize,'CACHE',base/'cache'), patch.object(normalize,'DATA',base/'data'):
                normalize.normalize_danish(10**15,0)
            output = json.loads((base/'data/denmark.json').read_text())['areas']['DK2']['power'][0]
            self.assertEqual(output['load'],475)  # 500 generation + 50 internal - 75 external.
            self.assertNotIn('no',output['borders'])


class PublishedDataTests(unittest.TestCase):
    def test_data_invariants(self):
        index = json.loads((ROOT / 'dist/data/index.json').read_text())
        self.assertGreater(index['coverage'], 0)
        for path in (ROOT / 'dist/data/countries').glob('*.json'):
            country = json.loads(path.read_text())
            for key in ['power', 'flows']:
                times = [r['t'] for r in country[key]]
                self.assertEqual(times, sorted(set(times)), (path.name, key))
            for row in country['power']:
                for key in ['renewable', 'lowCarbon']:
                    if row[key] is not None:
                        self.assertGreaterEqual(row[key], 0, path.name)
                        self.assertLessEqual(row[key], 100.01, path.name)
                if row['total'] is not None:
                    self.assertAlmostEqual(row['total'], sum(v for k,v in row['g'].items() if k!='storage'), places=2)
            for row in country['flows']:
                if row['net'] is not None:
                    self.assertAlmostEqual(row['net'], sum(b['mw'] for b in row['borders']), places=2)


if __name__ == '__main__':
    unittest.main()

