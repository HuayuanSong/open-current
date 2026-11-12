"""Weather availability, complete-horizon fallback and unchanged trial inputs."""

import copy
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import requests

from forecast import data, weather_runs
from forecast.features import CONFIG


def points(run, wind=6):
    times = pd.date_range(run, periods=120, freq="h")
    return [
        {"hourly": {"time": [t.isoformat() for t in times], **{
            name: [float(wind if name == "wind_speed_100m" else 1)] * len(times)
            for name in CONFIG["variables"]
        }}} for _ in CONFIG["locations"]
    ]


def entry(run, available, retrieved, wind=8):
    return {"model": "ecmwf_ifs", "run": run.isoformat(),
            "availableAt": available.isoformat(), "retrievedAt": retrieved.isoformat(),
            "data": points(run, wind)}


class WeatherRunsTests(unittest.TestCase):
    def setUp(self):
        self.run = pd.Timestamp("2026-11-12T00:00Z")
        self.at = pd.Timestamp("2026-11-12T10:30Z")
        self.available = self.run + pd.Timedelta(hours=6, minutes=30)
        self.metadata = dict(last_run_initialisation_time=int(self.run.timestamp()),
                             last_run_availability_time=int(self.available.timestamp()))

    def test_waits_for_actual_availability_and_replication(self):
        for at in [self.available - pd.Timedelta(seconds=1), self.available + pd.Timedelta(minutes=9)]:
            self.assertIsNone(weather_runs.available_run(self.metadata, at))
        self.assertEqual(weather_runs.available_run(self.metadata, self.available + pd.Timedelta(minutes=10)),
                         (self.run, self.available))
        self.assertIsNone(weather_runs.available_run(self.metadata, self.run + pd.Timedelta(hours=49)))

    def test_invalid_or_future_metadata_cannot_authorize_a_run(self):
        invalid = {**self.metadata, "last_run_availability_time": int(self.run.timestamp()) - 1}
        with self.assertRaises(ValueError):
            weather_runs.available_run(invalid, self.at)
        invalid = {**self.metadata, "last_run_initialisation_time": float("nan")}
        with self.assertRaises(ValueError):
            weather_runs.available_run(invalid, self.at)
        self.assertIsNone(weather_runs.available_run(self.metadata, self.run - pd.Timedelta(hours=1)))

    @patch.object(weather_runs.time, "sleep")
    def test_fetches_exact_run_once_without_changing_daily_history(self, sleep):
        state = {"weather": {"untouched": {"data": [1]}}, "forecasts": [{"id": "issued"}], "trial": {"frozen": True}}
        original = copy.deepcopy(state)
        request = Mock(side_effect=[self.metadata, points(self.run), self.metadata])
        weather_runs.refresh(state, self.at, request)
        weather_runs.refresh(state, self.at, request)
        self.assertEqual(request.call_count, 3)
        self.assertEqual(request.call_args_list[1].args[1]["run"], "2026-11-12T00:00")
        for key in original:
            self.assertEqual(state[key], original[key])
        self.assertEqual(list(state["weatherRuns"]), [self.run.isoformat()])

    @patch.object(weather_runs.time, "sleep")
    def test_incomplete_download_does_not_poison_cache_and_can_retry(self, sleep):
        incomplete = points(self.run)
        incomplete[0]["hourly"]["wind_speed_100m"][20] = None
        state = {"weatherRuns": {}}
        request = Mock(side_effect=[self.metadata, incomplete, self.metadata, points(self.run)])
        status = weather_runs.refresh(state, self.at, request)
        self.assertIn("Incomplete", status["error"])
        self.assertEqual(state["weatherRuns"], {})
        self.assertIsNone(weather_runs.refresh(state, self.at, request)["error"])
        self.assertIn(self.run.isoformat(), state["weatherRuns"])

    @patch.object(weather_runs.time, "sleep")
    def test_metadata_failure_preserves_last_good_run(self, sleep):
        cached = entry(self.run, self.available, self.available + pd.Timedelta(minutes=11))
        state = {"weatherRuns": {self.run.isoformat(): cached}}
        request = Mock(side_effect=requests.HTTPError("503"))
        status = weather_runs.refresh(state, self.at, request)
        self.assertEqual(status["error"], "503")
        self.assertEqual(list(weather_runs.candidates(state, self.at)), [(self.run, cached)])

    def test_future_retrieval_or_missing_availability_is_ineligible(self):
        cached = entry(self.run, self.available, self.at + pd.Timedelta(seconds=1))
        state = {"weatherRuns": {self.run.isoformat(): cached}}
        self.assertEqual(list(weather_runs.candidates(state, self.at)), [])
        cached.pop("availableAt")
        self.assertEqual(list(weather_runs.candidates(state, self.at)), [])

    def state(self):
        previous = self.run - pd.Timedelta(days=1)
        hours = pd.date_range(end=self.at.floor("h"), periods=600, freq="h")
        return {
            "hourly": [dict(time=t.isoformat(), area=area, wind=100., solar=20., load=1000.)
                       for t in hours for area in ["DK1", "DK2"]],
            "weather": {str(previous.date()): {"retrievedAt": previous.isoformat(), "data": points(previous)}},
            "weatherRuns": {self.run.isoformat(): entry(self.run, self.available, self.available + pd.Timedelta(minutes=11))},
        }

    def test_latest_changes_future_only_and_default_trial_ignores_it(self):
        state = self.state()
        old, old_meta = data.inputs(state, self.at)
        changed, meta = data.inputs(state, self.at, latest_weather=True)
        self.assertEqual(meta["weatherRun"], self.run.isoformat())
        self.assertEqual(meta["weatherSelection"], "recorded-availability")
        self.assertNotEqual(meta["weatherRun"], old_meta["weatherRun"])
        self.assertFalse(np.array_equal(old["future"], changed["future"]))
        for key in old:
            if key != "future":
                np.testing.assert_array_equal(old[key], changed[key])
        without_cache = {**state, "weatherRuns": {}}
        unchanged, unchanged_meta = data.inputs(without_cache, self.at)
        self.assertEqual(unchanged_meta, old_meta)
        for key in old:
            np.testing.assert_array_equal(unchanged[key], old[key])

    def test_incomplete_new_run_falls_back_for_both_areas_together(self):
        state = self.state()
        state["weatherRuns"][self.run.isoformat()]["data"][5]["hourly"]["wind_speed_100m"][20] = None
        old, old_meta = data.inputs(state, self.at)
        fallback, meta = data.inputs(state, self.at, latest_weather=True)
        self.assertEqual(meta["weatherRun"], old_meta["weatherRun"])
        self.assertEqual(meta["weatherSelection"], "daily-fallback")
        for key in old:
            np.testing.assert_array_equal(old[key], fallback[key])


if __name__ == "__main__":
    unittest.main()
