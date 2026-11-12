"""Chronological fitting and independent prospective evidence safeguards."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests

from forecast.features import AREAS, TARGETS
from forecast.trial.ledger import canonical_hash
from forecast.weather_runs import parameters
from research.forecasting.weather_retrain import compare, prepare, review
from research.forecasting.weather_retrain.prepare import build_history, matching_request
from research.forecasting.weather_retrain.prospective import eligible_origin, observation_snapshot
from research.forecasting.weather_retrain.train import masks
from tests.test_weather_runs import points


class WeatherRetrainTests(unittest.TestCase):
    def test_temporal_split_purges_overlapping_labels(self):
        cutoff = pd.Timestamp("2026-08-01T00:00Z")
        meta = pd.DataFrame({"origin": pd.to_datetime([
            "2026-07-29T21:00Z", "2026-07-30T00:00Z", "2026-07-31T21:00Z", "2026-08-01T03:00Z"], utc=True)})
        meta["last_label"] = meta.origin + pd.Timedelta(hours=48)
        train, valid = masks(meta, cutoff, pd.Timestamp("2026-10-01T00:00Z"))
        np.testing.assert_array_equal(train, [True, False, False, False])
        np.testing.assert_array_equal(valid, [False, False, False, True])

    def test_future_observations_cannot_change_history_or_gap_filling(self):
        origin = pd.Timestamp("2026-09-01T09:00Z")
        times = pd.date_range(end=origin + pd.Timedelta(hours=48), periods=800, freq="h")
        frame = pd.DataFrame(np.ones((800, 3)), index=times, columns=TARGETS)
        frame.loc[origin - pd.Timedelta(hours=10)] = np.nan
        past = pd.DataFrame({"wx_0_temperature_2m": np.ones(800)}, index=times)
        before, gaps = build_history(frame, past, origin)
        frame.loc[frame.index > origin] = 100000
        after, other_gaps = build_history(frame, past, origin)
        self.assertEqual(gaps, 1)
        self.assertEqual(gaps, other_gaps)
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])

    def test_cache_provenance_checks_run_and_geography(self):
        run = pd.Timestamp("2026-09-01T06:00Z")
        params = parameters(run)
        record = {"url": "https://single-runs-api.open-meteo.com/v1/forecast?" + urlencode(params)}
        self.assertTrue(matching_request(record, run))
        changed = {**params, "latitude": "1,2,3,4,5,6"}
        self.assertFalse(matching_request({"url": "https://example.org/?" + urlencode(changed)}, run))
        self.assertFalse(matching_request(record, run + pd.Timedelta(hours=6)))

    def test_transport_timeout_retries_without_caching_a_partial_download(self):
        run = pd.Timestamp("2026-09-01T12:00Z")
        times = pd.date_range(run + pd.Timedelta(hours=15), periods=48, freq="h")
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with patch.object(prepare, "CACHE", folder), patch.object(prepare.time, "sleep"), \
                 patch.object(prepare.data, "request", side_effect=[requests.Timeout("timeout"), points(run)]) as request:
                result, path = prepare.weather(run, times)
                self.assertEqual(request.call_count, 2)
                self.assertTrue(path.exists())
                cached, _ = prepare.weather(run, times)
                self.assertEqual(request.call_count, 2)
                self.assertEqual(cached, result)

    def test_cached_only_preparation_never_spends_more_api_quota(self):
        run = pd.Timestamp("2026-09-01T12:00Z")
        times = pd.date_range(run + pd.Timedelta(hours=15), periods=48, freq="h")
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with patch.object(prepare, "CACHE", folder), patch.object(prepare, "ROOT", folder), \
                 patch.object(prepare.data, "request") as request:
                with self.assertRaisesRegex(ValueError, "not cached"):
                    prepare.weather(run, times, cached_only=True)
                request.assert_not_called()

    def test_component_refit_preserves_other_members_and_blend_weights(self):
        shape = (2, 48, 3)
        families = ["frozen", "tree_control", "ridge", "lightgbm", "xgboost", "chronos_zero", "chronos", "ttm", "ttm_scaled"]
        members = {name: np.full(shape, 10., np.float32) for name in families if name != "frozen"}
        members.update(original_trees=np.full(shape, 10., np.float32), original_neural=np.full(shape, 10., np.float32))
        weights = [.2, .2, .1, .2, .1, .05, .05, .05, .05]
        selection = dict(families=families, bands=[[0, 48]], selection=[
            dict(area=area, target=target, selected="blend", weights=[weights]) for area in AREAS for target in TARGETS])
        inputs = dict(area_ids=np.array([0, 1]), future=np.zeros((2, 48, 1)))
        latest = {**inputs, "future": np.ones((2, 48, 1))}
        def refitted(arrays, names, folder):
            value = 20. if arrays is latest else 10.
            return {name: np.full(shape, value, np.float32) for name in ("tree_control", "ridge", "lightgbm", "xgboost")}
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "selection.json").write_text(json.dumps(selection))
            with patch.object(compare, "verify_models", return_value={"featureNames": {}}), \
                 patch.object(compare, "baseline_score", return_value=(np.full(shape, 10.), members, [])), \
                 patch.object(compare, "ensure", return_value=folder), patch.object(compare, "predict", side_effect=refitted):
                outputs, _ = compare.score_pair(inputs, latest, folder / "work")
        np.testing.assert_allclose(outputs["daily"], 10)
        np.testing.assert_allclose(outputs["latest"], 16)
        for values in members.values():
            np.testing.assert_array_equal(values, np.full(shape, 10.))

    def test_issue_window_rejects_early_late_and_out_of_period(self):
        for stamp in ("2026-11-12T03:30Z", "2026-11-13T03:14Z", "2026-11-13T03:59Z", "2026-11-13T04:30Z", "2026-12-11T03:30Z"):
            self.assertIsNone(eligible_origin(pd.Timestamp(stamp)))
        self.assertEqual(eligible_origin(pd.Timestamp("2026-11-13T04:30+01:00")), pd.Timestamp("2026-11-13T03:00Z"))

    def test_observations_require_the_frozen_maturity_window(self):
        at = pd.Timestamp("2026-11-17T09:00Z")
        rows = [dict(time=(at - pd.Timedelta(hours=age)).isoformat(), area="DK1", wind=age, solar=1., load=100.)
                for age in (23, 24, 72, 73)]
        snapshots = observation_snapshot(rows, at)
        self.assertEqual(len(snapshots), 6)
        self.assertEqual({r["value"] for r in snapshots if r["target"] == "wind"}, {24, 72})

    def test_later_revisions_do_not_replace_first_eligible_observation(self):
        valid = pd.Timestamp("2026-11-14T04:00Z")
        snapshots = [dict(time=valid.isoformat(), area="DK1", target="wind", value=value,
                          capturedAt=(valid + pd.Timedelta(hours=age)).isoformat())
                     for age, value in [(30, 300.), (24, 200.), (23, 100.), (73, 400.)]]
        frozen = review.first_observations(snapshots, valid + pd.Timedelta(hours=80))
        self.assertEqual(frozen[(valid.isoformat(), "DK1", "wind")]["value"], 200.)

    def record(self):
        origin = pd.Timestamp("2026-11-13T03:00Z")
        record = dict(origin=origin.isoformat(), issuedAt="2026-11-13T03:30:00+00:00", diagnosticOnly=False,
            times=[t.isoformat() for t in pd.date_range(origin + pd.Timedelta(hours=1), periods=48, freq="h")],
            protocolHash="fixed", modelManifestHash="fixed", inferenceLockHash="fixed",
            predictions={name: {area: {target: [1.] * 48 for target in TARGETS} for area in AREAS}
                         for name in ("baseline", "daily", "latest")})
        record["sha256"] = canonical_hash(record)
        return record

    def test_modified_or_diagnostic_records_are_rejected(self):
        with patch.object(review, "digest", return_value="fixed"):
            record = self.record()
            review.validate_record(record)
            changed = copy.deepcopy(record)
            changed["predictions"]["latest"]["DK1"]["wind"][0] = 2.
            with self.assertRaisesRegex(ValueError, "modified"):
                review.validate_record(changed)
            record["diagnosticOnly"] = True
            with self.assertRaisesRegex(ValueError, "diagnostic"):
                review.validate_record(record)

    def test_late_record_rejected_even_with_valid_checksum(self):
        record = self.record()
        record["issuedAt"] = "2026-11-13T04:00:00+00:00"
        record["sha256"] = canonical_hash({k: v for k, v in record.items() if k != "sha256"})
        with patch.object(review, "digest", return_value="fixed"):
            with self.assertRaisesRegex(ValueError, "backfilled"):
                review.validate_record(record)


if __name__ == "__main__":
    unittest.main()
