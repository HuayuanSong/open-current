"""Publication-time safeguards and exact wind-only refit compatibility."""

import copy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import requests

from forecast import weather_sources as sources


class FreshWeatherTests(unittest.TestCase):
    def setUp(self):
        self.at = pd.Timestamp("2026-11-16T12:30Z")
        self.times = pd.date_range("2026-11-16T13:00Z", periods=48, freq="h")
        hourly = {"time": self.times.strftime("%Y-%m-%dT%H:%M").tolist()}
        hourly.update({name: [5.] * 48 for name in sources.CONFIG["variables"]})
        self.points = [dict(hourly=copy.deepcopy(hourly)) for _ in sources.CONFIG["locations"]]
        self.entry = dict(model=sources.MODELS["dmi"], run="2026-11-16T09:00:00+00:00",
                          availableAt="2026-11-16T12:06:00+00:00",
                          retrievedAt="2026-11-16T12:20:00+00:00", data=self.points)
        self.metadata = dict(last_run_initialisation_time=int(pd.Timestamp(self.entry["run"]).timestamp()),
                             last_run_availability_time=int(pd.Timestamp(self.entry["availableAt"]).timestamp()))

    def test_fresh_dmi_requires_publication_margin_and_actual_retrieval(self):
        self.assertTrue(sources.eligible(self.entry, "dmi", self.at))
        for change in [dict(availableAt="2026-11-16T12:15Z"), dict(retrievedAt="2026-11-16T12:31Z"),
                       dict(run="2026-11-16T13:00Z"), dict(availableAt="2026-11-16T12:06"),
                       dict(model="ecmwf_ifs"), dict(run="2026-11-13T09:00Z")]:
            with self.subTest(change=change):
                self.assertFalse(sources.eligible({**self.entry, **change}, "dmi", self.at))

    def test_latest_incomplete_run_falls_back_to_complete_recorded_run(self):
        missing = copy.deepcopy(self.entry)
        missing.update(run="2026-11-16T12:00:00+00:00", availableAt="2026-11-16T12:01Z")
        missing["data"][0]["hourly"]["wind_speed_100m"][20] = None
        state = dict(availableWeather={"dmi": {"old": self.entry, "new": missing}})
        self.assertIs(sources.select(state, "dmi", self.at, self.times), self.entry)
        state["availableWeather"]["dmi"].pop("old")
        with self.assertRaisesRegex(ValueError, "No complete"):
            sources.select(state, "dmi", self.at, self.times)

    def test_download_is_cached_once_and_keeps_precise_source_identity(self):
        state = dict(trial={"original": True}, windUpdate={"original": True})
        with patch.object(sources.time, "sleep"), patch.object(sources.data, "now", return_value=self.at), patch.object(
            sources.data, "request", side_effect=[self.metadata, self.points, self.metadata]
        ) as get:
            sources.refresh(state, "dmi", self.at, self.times, get)
            sources.refresh(state, "dmi", self.at, self.times, get)
        self.assertEqual(get.call_count, 3)
        self.assertEqual(get.call_args_list[1].args[1]["models"], sources.MODELS["dmi"])
        self.assertEqual(get.call_args_list[1].args[1]["run"], "2026-11-16T09:00")
        selected = sources.select(state, "dmi", self.at, self.times)
        self.assertEqual(selected["retrievedAt"], self.at.isoformat())
        self.assertEqual(state["trial"], {"original": True})
        self.assertEqual(state["windUpdate"], {"original": True})

    def test_unpublished_run_is_not_downloaded_and_failure_preserves_cached_data(self):
        state = dict(availableWeather={"dmi": {self.entry["run"]: self.entry}})
        metadata = {**self.metadata, "last_run_availability_time": int((self.at - pd.Timedelta(minutes=5)).timestamp())}
        with patch.object(sources.time, "sleep"), patch.object(sources.data, "now", return_value=self.at), patch.object(
            sources.data, "request", side_effect=[metadata, requests.Timeout("slow provider")]
        ) as get:
            sources.refresh(state, "dmi", self.at, self.times, get)
            status = sources.refresh(state, "dmi", self.at, self.times, get)
        self.assertEqual(get.call_count, 2)
        self.assertIn("slow provider", status["error"])
        self.assertEqual(sources.select(state, "dmi", self.at, self.times), self.entry)

    def test_future_inputs_never_rewrite_historical_features(self):
        frames = []
        for ai, _ in enumerate(sources.AREAS):
            wx = sources.weather_features(self.points[ai * 3:ai * 3 + 3], self.times)[0]
            frames.append(pd.concat([wx, sources.calendar(self.times)], axis=1))
        baseline = dict(future=np.zeros((2, 48, len(frames[0].columns))), history=np.ones((2, 168, 3)))
        before = copy.deepcopy(baseline)
        result = sources.future_inputs(baseline, self.points, self.times, list(frames[0].columns))
        self.assertIs(result["history"], baseline["history"])
        np.testing.assert_array_equal(baseline["future"], before["future"])
        self.assertFalse(np.array_equal(result["future"], baseline["future"]))
        with self.assertRaisesRegex(ValueError, "schema"):
            sources.future_inputs(baseline, self.points, self.times, ["wrong"])


if __name__ == "__main__":
    unittest.main()
