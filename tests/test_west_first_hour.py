import copy
import gzip
import hashlib
import json
import os
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from forecast import west_first_hour as trial
from forecast.trial.ledger import canonical_hash


class WestFirstHourTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name)
        self.source_file = self.folder / "sources.json.gz"
        self.target = (trial.START + trial.HOUR).isoformat()

        def stamp(minutes):
            return (trial.START + pd.Timedelta(minutes=minutes)).isoformat()

        self.at = pd.Timestamp(stamp(34))
        rows = [dict(PriceArea="DK1", Minutes5UTC=stamp(t), OnshoreWindPower=mw, OffshoreWindPower=0)
                for t, mw in ((0, 100), (5, 120), (10, 140), (15, 160), (20, 9000))]
        self.source = dict(electricity=rows, electricityRetrievedAt=stamp(28))
        raw = gzip.compress(json.dumps(self.source).encode())
        self.source_file.write_bytes(raw)
        self.base = dict(id="wind-1", modelVersion=trial.BASELINE, githubRunId="123", origin=stamp(0),
                         issuedAt=stamp(33), times=[self.target], areas={"DK1": {"wind": [150]}},
                         sourceHash=hashlib.sha256(raw).hexdigest(), recentWind={"DK1": dict(
                             selectedAt=stamp(32), observationEnd=stamp(20), bufferMinutes=10, levelMW=140)})
        self.base["sha256"] = canonical_hash(self.base)
        self.state = {"windForecasts": [self.base], "hourly": []}
        self.ledger = {"records": {}, "observations": {}}

    def issue(self, at=None):
        return trial.record_issue(self.state, self.ledger, self.source_file, "123", lambda: at or self.at)

    def test_buffered_partial_hour_matches_hand_calculation(self):
        recent = self.source["electricity"]
        recent += [dict(recent[0]), dict(recent[0], PriceArea="DK2", OnshoreWindPower=12345)]
        pred, known = trial.alternatives(self.base, self.source)
        self.assertEqual(list(known.values()), [100, 120, 140, 160])
        self.assertEqual(pred["hold"], 140)
        self.assertAlmostEqual(pred["partial"], (520 + 8 * 140) / 12)
        recent[0]["OffshoreWindPower"] = recent[-2]["OffshoreWindPower"] = None
        self.assertEqual(trial.alternatives(self.base, self.source)[0]["partial"], 140)
        recent[0]["OffshoreWindPower"] = recent[-2]["OffshoreWindPower"] = 0
        recent[-2]["OnshoreWindPower"] = 101
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            trial.alternatives(self.base, self.source)

    def test_rejects_bad_input_cutoffs_and_unreproducible_anchor(self):
        for key, value in (("selectedAt", self.target), ("observationEnd", self.at.isoformat()),
                           ("bufferMinutes", 5), ("levelMW", 200)):
            with self.subTest(key=key):
                base = copy.deepcopy(self.base)
                base["recentWind"]["DK1"][key] = value
                with self.assertRaises(ValueError):
                    trial.alternatives(base, self.source)
        with self.assertRaises(ValueError):
            trial.alternatives(self.base, {**self.source, "electricityRetrievedAt": self.target})

    def test_only_first_live_west_issue_can_enroll(self):
        original = copy.deepcopy(self.state)
        self.assertEqual(self.issue(), "issued")
        saved = copy.deepcopy(self.ledger["records"][self.target])
        self.assertEqual(saved["publicHash"], self.base["sha256"])
        self.assertEqual(saved["predictionsMW"]["public"], 150)
        self.assertEqual(saved["issuedAt"], self.at.isoformat())
        self.assertEqual(self.issue(), "already-recorded")
        self.assertEqual(self.ledger["records"], {self.target: saved})
        self.assertEqual(self.state, original)
        self.ledger["records"].clear()
        self.state["windForecasts"].insert(0, {**self.base, "issuedAt": (self.at - 3 * pd.Timedelta(minutes=1)).isoformat(),
                                             "githubRunId": "111"})
        self.assertEqual(self.issue(), "first-public-issue-unavailable-this-run")
        self.state["windForecasts"] = [{**self.base, "areas": {"DK2": {"wind": [150]}}}]
        self.assertEqual(self.issue(), "first-public-issue-unavailable-this-run")
        self.assertFalse(self.ledger["records"])

    def test_late_or_corrupt_evidence_cannot_enroll(self):
        clock = iter((self.at, pd.Timestamp(self.target))).__next__
        self.assertEqual(trial.record_issue(self.state, self.ledger, self.source_file, "123", clock), "missed-target")
        for at in (trial.START - trial.HOUR, trial.STOP):
            self.assertEqual(self.issue(at), "outside-issue-window")
        self.source_file.write_bytes(b"wrong snapshot")
        with self.assertRaisesRegex(ValueError, "source checksum"):
            self.issue()
        self.base["areas"]["DK1"]["wind"][0] = 42
        with self.assertRaisesRegex(ValueError, "forecast checksum"):
            self.issue()
        self.assertFalse(self.ledger["records"])

    def test_outcome_must_be_mature_and_is_frozen_once(self):
        self.issue()
        end = pd.Timestamp(self.target)
        row = dict(time=self.target, area="DK1", wind=0)
        checked = end + pd.Timedelta(minutes=31)
        for changes in ({}, {"retrievedAt": (end + pd.Timedelta(minutes=14)).isoformat()},
                        {"retrievedAt": (checked + trial.HOUR).isoformat()},
                        {"area": "DK2", "retrievedAt": checked.isoformat()}, {"wind": None, "retrievedAt": checked.isoformat()}):
            trial.freeze_outcomes(self.ledger, [{**row, **changes}], checked)
        self.assertFalse(self.ledger["observations"])
        row["retrievedAt"] = (end + pd.Timedelta(minutes=15)).isoformat()
        trial.freeze_outcomes(self.ledger, [row], checked)
        trial.freeze_outcomes(self.ledger, [{**row, "wind": 999}], checked)
        self.assertEqual(self.ledger["observations"][self.target], {"mw": 0, "retrievedAt": row["retrievedAt"]})
        report = trial.evaluate(self.ledger, checked)
        self.assertEqual((report["pairedHours"], report["missedIssueHours"]), (1, 0))
        self.assertEqual(report["maeMW"]["public"], 150)
        self.assertEqual(report["daily"][0]["day"], 1)
        last = trial.STOP.isoformat()
        self.ledger["records"][last] = self.ledger["records"][self.target]
        trial.freeze_outcomes(self.ledger, [{**row, "time": last, "retrievedAt": (trial.REVIEW_AT + trial.SLOT).isoformat()}],
                              trial.REVIEW_AT + trial.HOUR)
        self.assertNotIn(last, self.ledger["observations"])
        trial.freeze_outcomes(self.ledger, [{**row, "time": last, "retrievedAt": trial.REVIEW_AT.isoformat()}], trial.REVIEW_AT)
        self.assertEqual(trial.evaluate(self.ledger, trial.REVIEW_AT)["daily"][-1]["day"], 14)

    def test_workflow_round_trip_and_final_review(self):
        path = self.folder / "state.json.gz"
        review = self.folder / "review.json"
        summary, output = self.folder / "summary.md", self.folder / "outputs"
        read, write = trial.data.read_state, trial.data.write_state
        write({"schema": 1, **self.state}, path)
        env = {"GITHUB_RUN_ID": "123", "PUBLIC_WIND_ISSUED": "true",
               "GITHUB_STEP_SUMMARY": str(summary), "GITHUB_OUTPUT": str(output)}
        with (patch.dict(os.environ, env), patch.object(trial, "SOURCE", self.source_file),
              patch("sys.stdout", new_callable=StringIO),
              patch.object(trial, "REVIEW_FILE", review), patch.object(trial.data, "now") as clock,
              patch.object(trial.data, "read_state", side_effect=lambda: read(path)),
              patch.object(trial.data, "write_state", side_effect=lambda s: write(s, path))):
            clock.return_value = self.at
            trial.main()
            state = read(path)
            self.assertEqual(state["windForecasts"], self.state["windForecasts"])
            state["hourly"] = [dict(time=self.target, area="DK1", wind=140,
                                   retrievedAt=(pd.Timestamp(self.target) + trial.HOUR).isoformat())]
            write(state, path)
            os.environ["PUBLIC_WIND_ISSUED"] = "false"
            clock.return_value = pd.Timestamp(self.target) + trial.HOUR
            trial.main()
            clock.return_value = trial.REVIEW_AT
            trial.main()
            final = json.loads(review.read_text())
            self.assertEqual(final["status"], "reviewed")
            self.assertEqual(final["scheduledHours"], 336)
            self.assertEqual(final["missedIssueHours"], 335)
            self.assertEqual(final["lowestObservedMAE"], ["hold"])
            self.assertEqual(final["reviewRunId"], "123")
            self.assertEqual(read(path)[trial.KEY]["report"], final)
            snapshot = path.read_bytes()
            clock.return_value += 10 * trial.HOUR
            trial.main()
            self.assertEqual(path.read_bytes(), snapshot)
            self.assertEqual(output.read_text(), "reviewed=true\n")
            self.assertIn("Final review run", summary.read_text())
