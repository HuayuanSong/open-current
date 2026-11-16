"""Causality, missing-data, immutable evidence and isolation of wind candidates."""

import copy
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from forecast import wind_update as candidate
from forecast.wind_nowcast import adjust, recent_wind


class WindUpdateTests(unittest.TestCase):
    def setUp(self):
        self.at = pd.Timestamp("2026-11-17T10:30Z")
        self.origin = self.at.floor("h")
        self.times = [t.isoformat() for t in pd.date_range(self.origin + pd.Timedelta(hours=1), periods=48, freq="h")]
        self.raw = [dict(PriceArea=area, Minutes5UTC=t.isoformat(), OnshoreWindPower=100.0 + i, OffshoreWindPower=50.0)
                    for area in ["DK1", "DK2"]
                    for i, t in enumerate(pd.date_range("2026-11-17T09:30Z", periods=20, freq="5min"))]
        self.rules = {area: dict(strength=1, halfLifeHours=2, trendWeight=1) for area in ["DK1", "DK2"]}
        self.protocol, self.selection, self.fingerprint = candidate.configuration()

    def test_cutoff_uses_completed_intervals_and_future_values_cannot_change_features(self):
        before = recent_wind(self.raw, self.at)
        self.assertEqual(before["DK1"]["observationEnd"], "2026-11-17T10:20:00+00:00")
        changed = copy.deepcopy(self.raw)
        for row in changed:
            if pd.Timestamp(row["Minutes5UTC"]) >= pd.Timestamp("2026-11-17T10:20Z"):
                row["OnshoreWindPower"] = 1e8
        self.assertEqual(before, recent_wind(changed, self.at))
        self.assertAlmostEqual(before["DK1"]["levelMW"], 158)

    def test_missing_negative_duplicate_and_stale_sources_fail_closed(self):
        last = "2026-11-17T10:15:00+00:00"
        missing = [r for r in self.raw if not (r["PriceArea"] == "DK2" and r["Minutes5UTC"] == last)]
        # Removing an interior interval must not be interpolated.
        missing = [r for r in missing if r["Minutes5UTC"] != "2026-11-17T10:05:00+00:00"]
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            recent_wind(missing, self.at)
        with self.assertRaisesRegex(ValueError, "stale"):
            recent_wind(self.raw[:2], self.at)
        bad = copy.deepcopy(self.raw)
        for row in bad:
            if row["Minutes5UTC"] == "2026-11-17T10:05:00+00:00":
                row["OffshoreWindPower"] = None
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            recent_wind(bad, self.at)
        conflicting = copy.deepcopy(next(r for r in self.raw if r["Minutes5UTC"] == last))
        conflicting["OffshoreWindPower"] += 1
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            recent_wind(self.raw + [conflicting], self.at)

    def test_zero_generation_is_a_valid_observation(self):
        raw = [{**r, "OnshoreWindPower": 0, "OffshoreWindPower": 0} for r in self.raw]
        recent = recent_wind(raw, self.at)
        self.assertEqual(recent["DK1"]["levelMW"], 0)
        output = adjust(np.ones((2, 48, 3)), self.times, recent, self.rules)
        self.assertTrue(np.isfinite(output).all())
        self.assertTrue((output >= 0).all())

    def test_correction_only_changes_six_wind_hours_and_never_mutates_inputs(self):
        base = np.arange(288).reshape(2, 48, 3).astype(float) + 500
        saved = base.copy()
        output = adjust(base, self.times, recent_wind(self.raw, self.at), self.rules)
        np.testing.assert_array_equal(base, saved)
        np.testing.assert_array_equal(output[:, :, 1:], base[:, :, 1:])
        np.testing.assert_array_equal(output[:, 6:, 0], base[:, 6:, 0])
        self.assertFalse(np.array_equal(output[:, :6, 0], base[:, :6, 0]))
        none = {area: {**rule, "strength": 0} for area, rule in self.rules.items()}
        np.testing.assert_array_equal(adjust(base, self.times, recent_wind(self.raw, self.at), none), base)

    def test_modified_evidence_or_candidate_cannot_silently_reset_history(self):
        state = {}
        trial = candidate.initialize(state, self.protocol, self.fingerprint)
        record = dict(origin=self.origin.isoformat(), times=self.times)
        record["sha256"] = candidate.canonical_hash(record)
        trial["records"].append(record)
        candidate.initialize(state, self.protocol, self.fingerprint)
        with self.assertRaisesRegex(ValueError, "changed"):
            candidate.initialize(state, self.protocol, "different")
        record["origin"] = "2026-11-17T09:00:00+00:00"
        with self.assertRaisesRegex(ValueError, "modified"):
            candidate.initialize(state, self.protocol, self.fingerprint)

    def test_first_mature_observation_is_retained_and_targets_freeze_independently(self):
        trial = dict(records=[dict(times=self.times)], observations={})
        stamp = self.times[0]
        rows = [dict(time=stamp, area="DK1", wind=100, solar=None)]
        candidate.freeze_observations(trial, rows, pd.Timestamp(stamp) + pd.Timedelta(hours=23))
        self.assertEqual(trial["observations"], {})
        candidate.freeze_observations(trial, rows, pd.Timestamp(stamp) + pd.Timedelta(hours=24))
        rows[0]["wind"] = 200
        candidate.freeze_observations(trial, rows, pd.Timestamp(stamp) + pd.Timedelta(hours=30))
        self.assertEqual(trial["observations"][stamp]["DK1"]["value"], 100)

    def exercise_issue(self, *, duplicate=False, late=False):
        meta = dict(origin=self.origin.isoformat(), times=self.times, weatherRun="2026-11-16T00:00:00+00:00")
        fresh_meta = {**meta, "weatherRun": "2026-11-17T00:00:00+00:00", "weatherSelection": "recorded-availability",
                      "weatherAvailableAt": "2026-11-17T06:30:00+00:00", "weatherRetrievedAt": self.at.isoformat()}
        base = dict(history=np.ones((2, 168, 3)), future=np.ones((2, 48, 1)))
        fresh = {**base, "future": np.zeros((2, 48, 1))}
        state = dict(forecasts=[dict(id="untouched")], trial={"untouched": True}, weatherRuns={})
        original = copy.deepcopy(state)
        trial = candidate.initialize(state, self.protocol, self.fingerprint)
        if duplicate:
            trial["records"].append(dict(origin=meta["origin"]))
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            times = [self.at] * 4 + [self.at + pd.Timedelta(hours=1) if late else self.at]
            stack.enter_context(patch.object(candidate.data, "now", side_effect=times))
            stack.enter_context(patch.object(candidate.data, "inputs", side_effect=[(base, meta), (fresh, fresh_meta)]))
            stack.enter_context(patch.object(candidate, "ensure_models"))
            request = stack.enter_context(patch.object(candidate.weather_runs, "refresh", return_value={}))
            stack.enter_context(patch.object(candidate, "measurements", return_value=(self.raw, self.at.isoformat())))
            stack.enter_context(patch.object(candidate, "paired_predictions", return_value=(np.full((2, 48, 3), 100.), np.full((2, 48, 3), 200.), {})))
            stack.enter_context(patch.object(candidate.intervals, "build", return_value={"areas": {}}))
            if late:
                with self.assertRaisesRegex(ValueError, "missed the first target"):
                    candidate.issue(state, trial, self.protocol, self.selection, Path(directory))
                self.assertEqual(trial["records"], [])
            else:
                result = candidate.issue(state, trial, self.protocol, self.selection, Path(directory))
                self.assertEqual(result, "already-recorded" if duplicate else "issued")
                if duplicate:
                    request.assert_not_called()
                else:
                    self.assertEqual(len(trial["records"]), 1)
                    self.assertTrue((Path(directory) / "record.json").exists())
                    candidate.initialize(state, self.protocol, self.fingerprint)
        self.assertEqual(state["forecasts"], original["forecasts"])
        self.assertEqual(state["trial"], original["trial"])

    def test_real_issue_records_provenance_without_changing_production(self):
        self.exercise_issue()

    def test_duplicate_issue_does_not_spend_weather_quota_or_replace_values(self):
        self.exercise_issue(duplicate=True)

    def test_late_issue_is_not_backfilled(self):
        self.exercise_issue(late=True)

    def test_probe_cannot_write_production_files(self):
        state = dict(hourly=[], forecasts=[{"original": True}], trial={"original": True})
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            stack.enter_context(patch("sys.argv", ["wind-update", "--probe"]))
            stack.enter_context(patch.object(candidate, "CACHE", Path(directory)))
            stack.enter_context(patch.object(candidate.data, "read_state", return_value=state))
            stack.enter_context(patch.object(candidate, "issue", return_value="already-recorded"))
            write = stack.enter_context(patch.object(candidate.data, "write_state"))
            stack.enter_context(redirect_stdout(io.StringIO()))
            candidate.main()
            write.assert_not_called()
            self.assertTrue((Path(directory) / "probe/status.json").exists())

    def test_failed_candidate_preserves_all_existing_production_and_trial_records(self):
        state = dict(hourly=[], forecasts=[{"original": True}], trial={"original": True})
        before = copy.deepcopy(state)
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            stack.enter_context(patch("sys.argv", ["wind-update"]))
            stack.enter_context(patch.object(candidate, "CACHE", Path(directory) / "cache"))
            stack.enter_context(patch.object(candidate, "PUBLIC", Path(directory) / "wind-update.json"))
            stack.enter_context(patch.object(candidate.data, "read_state", return_value=state))
            stack.enter_context(patch.object(candidate, "issue", side_effect=ValueError("weather unavailable")))
            write = stack.enter_context(patch.object(candidate.data, "write_state"))
            stack.enter_context(redirect_stdout(io.StringIO()))
            with self.assertRaises(SystemExit):
                candidate.main()
            written = write.call_args.args[0]
            self.assertEqual(written["forecasts"], before["forecasts"])
            self.assertEqual(written["trial"], before["trial"])
            self.assertEqual(written["windUpdate"]["lastAttempt"]["status"], "unavailable")

    def test_partial_and_diagnostic_outcomes_cannot_enter_report(self):
        trial = dict(records=[dict(origin=self.origin.isoformat(), times=self.times, diagnosticOnly=False)],
                     observations={}, lastAttempt=None)
        trial["observations"][self.times[0]] = {area: dict(value=100) for area in ["DK1", "DK2"]}
        self.assertEqual(candidate.report(trial, self.protocol, self.at)["completeOrigins"], 0)
        trial["observations"] = {t: {a: dict(value=100) for a in ["DK1", "DK2"]} for t in self.times}
        trial["records"][0]["diagnosticOnly"] = True
        self.assertEqual(candidate.report(trial, self.protocol, self.at)["completeOrigins"], 0)


if __name__ == "__main__":
    unittest.main()
