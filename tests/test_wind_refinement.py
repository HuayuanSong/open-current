"""Causality, provenance, target isolation and uncertainty failure behavior."""

import copy
from contextlib import ExitStack
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from forecast import data, weather_ensemble as ensemble, wind_ranges, wind_refinement as trial, wind_short_term as short


class WindRefinementTests(unittest.TestCase):
    def setUp(self):
        self.at = pd.Timestamp("2026-11-17T09:30Z")
        self.times = pd.date_range("2026-11-17T10:00Z", periods=48, freq="h")
        self.raw = [dict(PriceArea=area, Minutes5UTC=t.tz_localize(None).isoformat(),
            OnshoreWindPower=100+i, OffshoreWindPower=300) for i, t in enumerate(
                pd.date_range(self.at-pd.Timedelta(hours=4), self.at, freq="5min")) for area in data.AREAS]
        self.names = json.loads((data.ROOT / "research/forecasting/weather_retrain/model-manifest.json").read_text())["featureNames"]["future"]
        self.arrays = dict(history=np.full((2, 168, 3), 100.), observed512=np.full((2, 512, 3), 100.),
                           ecmwf_future=np.ones((2, 48, len(self.names))))
        self.protocol = json.loads((short.HERE / "protocol.json").read_text())

    def test_only_completed_buffered_measurements_influence_features(self):
        before = short.measurements(self.raw, self.at)
        altered = copy.deepcopy(self.raw)
        for row in altered:
            if pd.Timestamp(row["Minutes5UTC"], tz="UTC") > self.at-pd.Timedelta(minutes=15):
                row["OnshoreWindPower"] = 999999
        self.assertEqual(before, short.measurements(altered, self.at))
        self.assertEqual(before["DK1"]["observationEnd"], "2026-11-17T09:20:00+00:00")
        missing = [r for r in self.raw if r["Minutes5UTC"] != "2026-11-17T08:00:00"]
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            short.measurements(missing, self.at)

    def test_ramp_features_use_only_six_weather_hours_and_observed_history(self):
        recent = short.measurements(self.raw, self.at)
        before, columns = short.features(self.arrays["history"], self.arrays["ecmwf_future"], self.names, recent, self.times)
        changed = self.arrays["ecmwf_future"].copy(); changed[:, 6:] = 99999
        after, _ = short.features(self.arrays["history"], changed, self.names, recent, self.times)
        np.testing.assert_array_equal(before, after)
        self.assertEqual(columns, json.loads((short.HERE / "selection.json").read_text())["featureNames"])

    def points(self):
        return [dict(latitude=p["latitude"], longitude=p["longitude"], utc_offset_seconds=0,
            hourly_units={k:"m/s" for k in ensemble.VARIABLES}, hourly=dict(
                time=[t.tz_localize(None).isoformat() for t in self.times],
                wind_speed_100m=[8]*48, wind_speed_100m_spread=[2]*48)) for p in ensemble.CONFIG["locations"]]

    def test_ensemble_spread_is_not_confused_with_spatial_spread(self):
        values = ensemble.values(self.points(), self.times)
        self.assertEqual(values["DK1"]["meanLocalSpreadMS"], [2]*48)
        for mutation in ("units", "null", "location"):
            points = self.points()
            if mutation == "units": points[0]["hourly_units"]["wind_speed_100m"] = "km/h"
            if mutation == "null": points[0]["hourly"]["wind_speed_100m_spread"][0] = None
            if mutation == "location": points[0]["latitude"] = 0
            with self.assertRaises(ValueError): ensemble.values(points, self.times)

    def test_reusing_an_ensemble_snapshot_preserves_its_original_retrieval_time(self):
        record = dict(retrievedAt=(self.at-pd.Timedelta(minutes=20)).isoformat(),
                      parameters=ensemble.parameters(), data=self.points())
        with patch.object(ensemble.requests, "get") as get:
            result = ensemble.snapshot({"weatherEnsemble": record}, self.times, self.at)
        get.assert_not_called(); self.assertIs(result, record)

    def test_ranges_need_finite_input_and_do_not_claim_spread_is_a_power_interval(self):
        prediction = np.full(48, 1000.)
        scales = wind_ranges.scale("ensemble", prediction, "DK1", weather=dict(meanWindMS=[10]*48, meanLocalSpreadMS=[2]*48))
        np.testing.assert_array_equal(scales, 600)
        rule = wind_ranges.fit(np.full((12,48), 1100.), np.full((12,48),1000.), np.full((12,48),600.), "ensemble")
        lo, hi = wind_ranges.bounds(prediction, scales, rule)
        np.testing.assert_allclose(lo, 900); np.testing.assert_allclose(hi, 1100)
        with self.assertRaises(ValueError): wind_ranges.scale("ensemble", prediction, "DK1")
        bad = np.ones((12,48)); bad[0,0] = np.nan
        with self.assertRaises(ValueError): wind_ranges.fit(bad, bad, bad, "power")

    def issue(self, *, ensemble_failure=False, missing_history=False):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        folder = Path(temporary.name)
        arrays = copy.deepcopy(self.arrays)
        if missing_history: arrays["observed512"][0,-1,0] = np.nan
        np.savez_compressed(folder / "inputs.npz", **arrays)
        with gzip.open(folder / "sources.json.gz", "wt") as f: json.dump(dict(electricity=self.raw), f)
        base = dict(id="public", modelVersion="public-wind-v1", diagnosticOnly=False,
            issuedAt=(self.at-pd.Timedelta(minutes=2)).isoformat(), origin="2026-11-17T09:00:00+00:00",
            times=[t.isoformat() for t in self.times], areas={a:dict(wind=[100.]*48, solar=[50.]*48) for a in data.AREAS},
            recentWind={a:dict(selectedAt=self.at.isoformat()) for a in data.AREAS},
            inputHash=trial.digest(folder / "inputs.npz"), sourceHash=trial.digest(folder / "sources.json.gz"))
        base["sha256"] = trial.canonical_hash(base)
        (folder / "record.json").write_text(json.dumps(base))
        state = dict(hourly=[], windForecasts=[base], forecasts=[{"unchanged": True}])
        before = copy.deepcopy(state)
        snapshot = dict(model="ensemble", data=self.points(), retrievedAt=self.at.isoformat())
        with ExitStack() as stack:
            stack.enter_context(patch.object(trial, "WIND_EVIDENCE", folder))
            stack.enter_context(patch.object(data, "now", return_value=self.at))
            stack.enter_context(patch.object(ensemble, "snapshot", side_effect=OSError("offline") if ensemble_failure else None, return_value=snapshot))
            status = trial.issue(state, folder / "result", self.protocol, "frozen")
            self.assertEqual(status, "issued")
            self.assertEqual(trial.issue(state, folder / "result", self.protocol, "frozen"), "already-recorded")
        self.assertEqual(before, {k: state[k] for k in before})
        return state

    def test_live_challenger_preserves_public_forecasts_and_only_changes_six_wind_hours(self):
        state = self.issue()
        record = state["windResearch"]["records"][0]
        self.assertIsNone(record["shortTermError"])
        self.assertEqual(record["ranges"], {})
        for area in data.AREAS: self.assertEqual(record["challenger"][area][6:], record["baseline"][area][6:])
        self.assertEqual(record["sha256"], trial.canonical_hash({k:v for k,v in record.items() if k!='sha256'}))

    def test_weather_failure_retains_ramp_prediction_and_missing_inputs_use_visible_fallback(self):
        record = self.issue(ensemble_failure=True)["windResearch"]["records"][0]
        self.assertIsNone(record["weather"]); self.assertEqual(record["weatherError"], "offline")
        record = self.issue(missing_history=True)["windResearch"]["records"][0]
        self.assertEqual(record["baseline"], record["challenger"])
        self.assertIn("Missing", record["shortTermError"])

    def test_evaluation_waits_for_complete_outcomes_and_deduplicates_refreshes(self):
        state = self.issue(); record = state["windResearch"]["records"][0]
        state["hourly"] = [dict(area=a, time=t, wind=100.) for a in data.AREAS for t in record["times"]]
        self.assertEqual(trial.completed([record], state, self.times[5], horizon=6), [])
        matched = trial.completed([record, {**record, "id":"duplicate"}], state, self.times[5]+pd.Timedelta(minutes=15), horizon=6)
        self.assertEqual(len(matched), 1)

    def test_range_calibration_cannot_use_future_outcomes_or_other_model_versions(self):
        previous, actual = [], {}
        for i, origin in enumerate(pd.date_range("2026-11-10T00:00Z", "2026-11-17T06:00Z", freq="6h")):
            times = pd.date_range(origin+pd.Timedelta(hours=1), periods=48, freq="h")
            previous.append(dict(id=str(i), version=self.protocol["version"], baselineVersion="current",
                origin=origin.isoformat(), issuedAt=(origin+pd.Timedelta(minutes=30)).isoformat(),
                times=[t.isoformat() for t in times], baseline={a:[100.]*48 for a in data.AREAS},
                weather={a:dict(meanWindMS=[10.]*48, meanLocalSpreadMS=[2.]*48) for a in data.AREAS}))
            for area in data.AREAS:
                for t in times: actual[(area,t.isoformat())] = dict(area=area,time=t.isoformat(),wind=200.)
        current = dict(issuedAt=self.at.isoformat(), version=self.protocol["version"], baselineVersion="current",
                       baseline={a:[100.]*48 for a in data.AREAS}, weather=previous[-1]["weather"])
        state = dict(hourly=list(actual.values()))
        before = trial.calibrate(current, previous, state, self.protocol)
        self.assertEqual(set(before), set(trial.METHODS))
        for row in state["hourly"]:
            if pd.Timestamp(row["time"]) > self.at-pd.Timedelta(minutes=15): row["wind"] = 99999
        after = trial.calibrate(current, previous, state, self.protocol)
        self.assertEqual(before, after)
        for value in after.values():
            self.assertLessEqual(pd.Timestamp(value["lastCalibrationTarget"])+pd.Timedelta(minutes=15), self.at)
        alien = [{**r, "baselineVersion":"old"} for r in previous]
        self.assertEqual(trial.calibrate(current, alien, state, self.protocol), {})
        current["weather"] = None
        missing = trial.calibrate(current, previous, state, self.protocol)
        self.assertNotIn("ensemble", missing)
        self.assertIn("absolute", missing)


if __name__ == "__main__":
    unittest.main()
