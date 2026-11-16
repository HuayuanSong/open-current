"""Wind updates preserve old forecasts, target boundaries and failure behavior."""

import copy
from contextlib import ExitStack, redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from forecast import fresh_wind as wind, publish


class FreshWindTests(unittest.TestCase):
    def setUp(self):
        self.at = pd.Timestamp("2026-11-17T09:30Z")
        self.times = pd.date_range("2026-11-17T10:00Z", periods=48, freq="h")
        self.meta = dict(origin="2026-11-17T09:00:00+00:00", weatherRun="2026-11-16T00:00:00+00:00",
                         featureNames=["test"], times=[t.isoformat() for t in self.times])
        self.arrays = dict(area_ids=np.array([0, 1]), future=np.ones((2, 48, 1)), history=np.ones((2, 168, 3)))
        self.base = np.full((2, 48, 3), 100.)
        self.ecmwf = self.base.copy(); self.ecmwf[:, :, 0] = 200
        self.dmi = self.base.copy(); self.dmi[:, :, 0] = 400
        self.selection = dict(version="test-wind", dmiWeights={"DK1": .25, "DK2": 0},
            correctionRules={a: dict(strength=1, halfLifeHours=1, trendWeight=0) for a in wind.data.AREAS})
        self.raw = [dict(PriceArea=a, Minutes5UTC=t.tz_localize(None).isoformat(),
                        OnshoreWindPower=70, OffshoreWindPower=50)
                    for t in pd.date_range(self.at - pd.Timedelta(hours=1), self.at, freq="5min")
                    for a in wind.data.AREAS]
        self.state = dict(hourly=[], forecasts=[{"original": True}], trial={"original": True}, windUpdate={"original": True})

    def execute_issue(self, state=None, *, late=False):
        state = state if state is not None else self.state
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            stack.enter_context(patch.object(wind.data, "now", return_value=self.at))
            stack.enter_context(patch.object(wind.data, "inputs", return_value=(self.arrays, copy.deepcopy(self.meta))))
            stack.enter_context(patch.object(wind.weather_sources, "refresh"))
            stack.enter_context(patch.object(wind.weather_sources, "select", side_effect=lambda _, s, *args: dict(
                model=wind.weather_sources.MODELS[s], run="2026-11-17T06:00:00+00:00", data=[],
                availableAt="2026-11-17T09:00:00+00:00", retrievedAt="2026-11-17T09:20:00+00:00")))
            stack.enter_context(patch.object(wind.weather_sources, "future_inputs", return_value=self.arrays))
            stack.enter_context(patch.object(wind, "measurements", return_value=(self.raw, self.at.isoformat())))
            stack.enter_context(patch.object(wind, "ensure_models"))
            stack.enter_context(patch.object(wind.compare, "verify_models", return_value={"featureNames": []}))
            (Path(folder) / "selection.json").write_text('{}')
            stack.enter_context(patch.object(wind.compare, "ensure", return_value=Path(folder)))
            stack.enter_context(patch.object(wind, "model_members", return_value=({}, {})))
            stack.enter_context(patch.object(wind, "combine", return_value=(self.base, {})))
            predictions = stack.enter_context(patch.object(wind, "predict_wind", side_effect=[self.dmi, self.ecmwf]))
            # Source names are sorted: dmi, then ecmwf.
            if late:
                count = 0
                def score(*args):
                    nonlocal count
                    count += 1
                    if count == 2:
                        wind.data.now.return_value = self.at + pd.Timedelta(hours=1)
                    return self.dmi if count == 1 else self.ecmwf
                predictions.side_effect = score
            status = wind.issue(state, self.selection, "frozen", Path(folder) / "evidence")
            return status

    def test_weighting_and_correction_touch_only_wind_and_first_six_hours(self):
        mixed = wind.weighted_wind(dict(baseline=self.base, ecmwf=self.ecmwf, dmi=self.dmi), self.selection["dmiWeights"])
        np.testing.assert_array_equal(mixed[0, :, 0], 250)
        np.testing.assert_array_equal(mixed[1, :, 0], 200)
        np.testing.assert_array_equal(mixed[:, :, 1:], self.base[:, :, 1:])
        before = copy.deepcopy(self.state)
        self.assertEqual(self.execute_issue(), "issued")
        record = self.state["windForecasts"][0]
        for key in before:
            self.assertEqual(self.state[key], before[key])
        for ai, area in enumerate(wind.data.AREAS):
            self.assertEqual(record["areas"][area]["solar"], [100.] * 48)
            self.assertEqual(record["areas"][area]["load"], [100.] * 48)
            self.assertEqual(record["areas"][area]["wind"][6:], mixed[ai, 6:, 0].tolist())
        self.assertEqual(record["ranges"]["areas"], {})
        self.assertEqual(record["weatherSources"]["dmi"]["model"], wind.weather_sources.MODELS["dmi"])
        self.assertEqual(record["sha256"], wind.canonical_hash({k: v for k, v in record.items() if k != "sha256"}))

    def test_duplicate_issue_preserves_original_values(self):
        self.execute_issue()
        before = copy.deepcopy(self.state["windForecasts"])
        self.assertEqual(self.execute_issue(), "already-recorded")
        self.assertEqual(self.state["windForecasts"], before)

    def test_late_issue_is_never_saved(self):
        with self.assertRaisesRegex(ValueError, "missed the first"):
            self.execute_issue(late=True)
        self.assertEqual(self.state["windForecasts"], [])

    def test_tampered_issued_record_is_rejected(self):
        self.execute_issue()
        self.state["windForecasts"][0]["areas"]["DK1"]["wind"][0] += 1
        with self.assertRaisesRegex(ValueError, "modified"):
            self.execute_issue()

    def test_zero_weight_does_not_require_unused_source_or_accept_invalid_weight(self):
        result = wind.weighted_wind(dict(baseline=self.base, ecmwf=self.ecmwf), {"DK1": 0, "DK2": 0})
        np.testing.assert_array_equal(result, self.ecmwf)
        for value in (float("nan"), -1, 2):
            with self.assertRaisesRegex(ValueError, "weight"):
                wind.weighted_wind(dict(baseline=self.base), {"DK1": value, "DK2": 0})

    def test_probe_cannot_change_published_files(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            stack.enter_context(patch("sys.argv", ["fresh-wind", "--probe"]))
            stack.enter_context(patch.object(wind, "CACHE", Path(folder)))
            stack.enter_context(patch.object(wind, "configuration", return_value=(self.selection, "frozen")))
            stack.enter_context(patch.object(wind.data, "read_state", return_value=copy.deepcopy(self.state)))
            stack.enter_context(patch.object(wind, "issue", return_value="waiting-for-source-hour"))
            write = stack.enter_context(patch.object(wind.data, "write_state"))
            public = stack.enter_context(patch.object(wind.publish, "write_public"))
            stack.enter_context(redirect_stdout(io.StringIO()))
            wind.main()
            write.assert_not_called(); public.assert_not_called()
            self.assertTrue((Path(folder) / "probe/status.json").exists())

    def test_failed_refresh_preserves_issued_wind_and_every_existing_trial(self):
        self.execute_issue()
        before = copy.deepcopy(self.state)
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            root = Path(folder)
            public_path = root / "forecasts.json"
            public_path.write_text(json.dumps(dict(modelVersion="daily", message=None)))
            stack.enter_context(patch("sys.argv", ["fresh-wind"]))
            stack.enter_context(patch.object(wind, "CACHE", root))
            stack.enter_context(patch.object(wind, "configuration", return_value=(self.selection, "frozen")))
            stack.enter_context(patch.object(wind.data, "now", return_value=self.at))
            stack.enter_context(patch.object(wind.data, "PUBLIC_PATH", public_path))
            stack.enter_context(patch.object(wind.data, "read_state", return_value=self.state))
            stack.enter_context(patch.object(wind, "issue", side_effect=ValueError("DMI unavailable")))
            write = stack.enter_context(patch.object(wind.data, "write_state"))
            public = stack.enter_context(patch.object(wind.publish, "write_public"))
            stack.enter_context(redirect_stdout(io.StringIO()))
            with self.assertRaises(SystemExit):
                wind.main()
        saved = write.call_args.args[0]
        for key in ("windForecasts", "forecasts", "trial", "windUpdate"):
            self.assertEqual(saved[key], before[key])
        self.assertEqual(saved["windServing"]["lastAttempt"]["error"], "DMI unavailable")
        self.assertEqual(public.call_args.args[2:], ("daily", None))

    def test_published_wind_errors_and_comparisons_never_use_old_model_records(self):
        self.execute_issue()
        issued = self.state["windForecasts"][0]
        state = {**self.state, "forecasts": [{**issued, "modelVersion": "daily"}],
                 "windServing": dict(version="test-wind", lastAttempt={"error": None})}
        stamp = issued["times"][0]
        state["hourly"] = [dict(time=stamp, area=a, wind=0, solar=0, load=0) for a in wind.data.AREAS]
        payload = publish.payload(state, self.at + pd.Timedelta(hours=2), "daily")
        self.assertEqual(payload["latest"]["modelVersion"], "daily")
        fresh = payload["windOutlook"]
        self.assertEqual(fresh["latest"]["modelVersion"], "test-wind")
        self.assertIsNone(fresh["comparison"])
        self.assertIsNone(fresh["rangeValidation"])
        self.assertEqual(fresh["performance"]["DK1"]["wind"]["forecastHours"], 1)
        self.assertEqual(fresh["performance"]["DK1"]["wind"]["rangeHours"], 0)
        self.assertNotIn("solar", fresh["performance"]["DK1"])
        state["windServing"]["lastAttempt"]["error"] = "provider failed"
        failed = publish.payload(state, self.at + pd.Timedelta(hours=2), "daily")
        self.assertEqual(failed["windOutlook"]["status"], "delayed")
        self.assertEqual(failed["windOutlook"]["latest"], issued)
        self.assertEqual(failed["status"], "ready")

    def test_worker_reuse_requires_matching_inputs_and_newer_outputs(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(wind.data, "ROOT", Path(folder)):
            work = Path(folder) / ".cache/forecast-serving/run"
            work.mkdir(parents=True)
            path = work / "inputs.npz"
            np.savez_compressed(path, **self.arrays)
            os.utime(path, (100, 100))
            for family in ("trees", "neural", "ttm"):
                output = work / f"{family}.npz"
                np.savez_compressed(output, **{family: self.base})
                os.utime(output, (200, 200))
            with patch.object(wind, "baseline_score", return_value=(self.base, {"replacement": self.base}, [])) as score:
                members, runtime = wind.model_members(self.arrays, Path(folder) / "scratch")
                score.assert_not_called()
                self.assertTrue(runtime["reusedWorkers"])
                os.utime(work / "trees.npz", (50, 50))
                _, runtime = wind.model_members(self.arrays, Path(folder) / "scratch")
                self.assertFalse(runtime["reusedWorkers"])
                score.assert_called_once()


if __name__ == "__main__":
    unittest.main()
