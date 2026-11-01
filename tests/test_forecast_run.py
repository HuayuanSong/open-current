"""Forecast publication preserves issued records through success and failure."""

import copy
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from forecast import publish as publisher, run


class ForecastRunTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.at = pd.Timestamp("2026-11-01T12:30:00+00:00")
        self.origin = self.at.floor("h")
        self.names = ["wx_0_terrestrial_radiation"]
        (self.root / "feature_names.json").write_text(
            json.dumps({"future": self.names})
        )
        self.prediction = np.arange(288, dtype=float).reshape(2, 48, 3) + 100
        self.control = self.prediction + 10
        daylight = np.tile([100.0] * 12 + [0.0] * 12, (2, 2))
        self.arrays = {"future": daylight[:, :, None]}
        self.metadata = {
            "origin": self.origin.isoformat(),
            "weatherRun": "2026-11-01T00:00:00+00:00",
            "featureNames": self.names,
            "times": [
                stamp.isoformat()
                for stamp in pd.date_range(
                    self.origin + pd.Timedelta(hours=1), periods=48, freq="h"
                )
            ],
        }
        self.original = {
            "id": "previous-issue",
            "modelVersion": run.CONFIG["version"],
            "issuedAt": "2026-10-31T12:30:00+00:00",
            "origin": "2026-10-31T12:00:00+00:00",
            "times": [],
            "areas": {},
        }
        self.trial = {"records": [{"sha256": "unchanged-existing-evidence"}]}
        self.state = {
            "schema": 1,
            "hourly": [],
            "weather": {},
            "forecasts": [copy.deepcopy(self.original)],
            "trial": copy.deepcopy(self.trial),
        }

    def execute(self, args=(), failure=None, prediction=None, issued_at=None):
        work = self.root / ".cache/forecast-serving/run"
        work.mkdir(parents=True, exist_ok=True)
        (work / "trees.explanations.json").write_text('{"fixture":true}')
        ranges = {
            "areas": {
                area: {
                    target: {"lower": [0.0] * 48, "upper": [1000.0] * 48}
                    for target in run.data.TARGETS
                }
                for area in run.data.AREAS
            }
        }
        mocks = {
            "electricity": Mock(),
            "weather": Mock(),
            "inputs": Mock(return_value=(self.arrays, copy.deepcopy(self.metadata))),
            "score": Mock(
                return_value=(
                    self.prediction if prediction is None else prediction,
                    {"tree_control": self.control},
                    [],
                )
            ),
            "restore": Mock(return_value=self.state),
        }
        if failure:
            mocks[failure].side_effect = RuntimeError(f"fixture {failure} failed")
        written = []
        public = []

        # Exercise actual payload construction while keeping every write temporary.
        def capture_public(state, at, version, error):
            value = publisher.payload(state, at, version, error)
            json.dumps(value, allow_nan=False)
            public.append(copy.deepcopy(value))
            return value

        with ExitStack() as stack:
            stack.enter_context(patch.object(run.sys, "argv", ["forecast", *args]))
            stack.enter_context(patch.object(run, "ROOT", self.root))
            stack.enter_context(patch.object(run, "ensure", return_value=self.root))
            stack.enter_context(
                patch.object(run.data, "read_state", return_value=self.state)
            )
            stack.enter_context(
                patch.object(
                    run.data,
                    "now",
                    side_effect=[self.at, issued_at or self.at, issued_at or self.at],
                )
            )
            stack.enter_context(
                patch.object(
                    run.data,
                    "write_state",
                    side_effect=lambda value: written.append(copy.deepcopy(value)),
                )
            )
            stack.enter_context(
                patch.object(run, "write_public", side_effect=capture_public)
            )
            stack.enter_context(patch.object(run.intervals, "refresh_history"))
            stack.enter_context(
                patch.object(run.intervals, "build", return_value=ranges)
            )
            for name, mock in mocks.items():
                stack.enter_context(
                    patch.object(run if name == "score" else run.data, name, mock)
                )
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            error = None
            try:
                run.main()
            except (SystemExit, RuntimeError) as exc:
                error = exc
        return written, public, mocks, error

    def test_success_preserves_archive_and_serializes_all_six_series(self):
        written, public, mocks, error = self.execute()
        self.assertIsNone(error)
        self.assertEqual(written[0]["forecasts"][0], self.original)
        self.assertEqual(written[0]["trial"], self.trial)
        latest = public[0]["latest"]
        self.assertEqual(public[0]["status"], "ready")
        self.assertEqual(latest["issuedAt"], self.at.isoformat())
        self.assertEqual(latest["times"], self.metadata["times"])
        for area_index, area in enumerate(run.data.AREAS):
            for target_index, target in enumerate(run.data.TARGETS):
                self.assertEqual(
                    latest["areas"][area][target],
                    self.prediction[area_index, :, target_index].tolist(),
                )
                self.assertEqual(
                    latest["treeControl"][area][target],
                    self.control[area_index, :, target_index].tolist(),
                )
        mocks["electricity"].assert_called_once()
        mocks["weather"].assert_called_once()

    def test_duplicate_issue_is_not_rescored_or_replaced(self):
        self.execute()
        issued = copy.deepcopy(self.state["forecasts"])
        written, _, mocks, error = self.execute(args=("--offline",))
        self.assertIsNone(error)
        self.assertEqual(written[0]["forecasts"], issued)
        mocks["score"].assert_not_called()
        mocks["electricity"].assert_not_called()
        mocks["weather"].assert_not_called()

    def test_source_and_model_failures_preserve_previous_issue(self):
        for failure in ("electricity", "weather", "inputs", "score"):
            with self.subTest(failure=failure):
                written, public, _, error = self.execute(
                    args=("--strict",), failure=failure
                )
                self.assertIsInstance(error, SystemExit)
                self.assertEqual(error.code, 1)
                self.assertEqual(written[0]["forecasts"], [self.original])
                self.assertEqual(public[0]["status"], "delayed")
                self.assertEqual(public[0]["latest"], self.original)

    def test_invalid_predictions_do_not_replace_previous_issue(self):
        for value in (-1.0, 1e9):
            with self.subTest(value=value):
                written, public, _, error = self.execute(
                    prediction=np.full((2, 48, 3), value)
                )
                self.assertIsNone(error)
                self.assertEqual(written[0]["forecasts"], [self.original])
                self.assertEqual(public[0]["status"], "delayed")

    def test_failed_restore_never_writes_or_issues(self):
        written, public, mocks, error = self.execute(
            args=("--restore",), failure="restore"
        )
        self.assertIsInstance(error, RuntimeError)
        self.assertEqual(written, [])
        self.assertEqual(public, [])
        mocks["score"].assert_not_called()

    def test_retention_preserves_the_exact_35_day_boundary(self):
        cutoff = self.at - pd.Timedelta(days=35)
        for identifier, issued in [
            ("boundary", cutoff),
            ("expired", cutoff - pd.Timedelta(seconds=1)),
        ]:
            record = copy.deepcopy(self.original)
            record.update(id=identifier, issuedAt=issued.isoformat())
            self.state["forecasts"].append(record)
        written, _, _, error = self.execute()
        self.assertIsNone(error)
        ids = [record["id"] for record in written[0]["forecasts"]]
        self.assertIn("boundary", ids)
        self.assertNotIn("expired", ids)
        self.assertIn(self.original["id"], ids)
        self.assertEqual(written[0]["trial"], self.trial)

    def test_late_issue_never_adds_ranges_to_elapsed_hours(self):
        _, public, _, error = self.execute(
            issued_at=self.origin + pd.Timedelta(hours=2)
        )
        self.assertIsNone(error)
        for targets in public[0]["latest"]["ranges"]["areas"].values():
            for bounds in targets.values():
                self.assertEqual(bounds["lower"][:2], [None, None])
                self.assertEqual(bounds["upper"][:2], [None, None])
                self.assertEqual(bounds["lower"][2], 0.0)


if __name__ == "__main__":
    unittest.main()
