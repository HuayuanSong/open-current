import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from forecast import data, wind_model_review as review


def issue(origin, model, minute, wind, areas=("DK1", "DK2")):
    return dict(modelVersion=model, origin=origin.isoformat(), issuedAt=(origin + pd.Timedelta(minutes=minute)).isoformat(),
                times=[t.isoformat() for t in pd.date_range(origin + review.HOUR, periods=48, freq="h")],
                areas={area: {"wind": [wind] * 48} for area in areas})


class PublicWindReviewTests(unittest.TestCase):
    def test_calendar_before_review_needs_no_archive_or_model_scoring(self):
        with TemporaryDirectory() as folder:
            archive, output = Path(folder) / "archive.gz", Path(folder) / "report.json"
            for at, status in [(review.START - review.HOUR, "scheduled"), (review.START, "collecting"),
                               (review.STOP, "awaiting-outcomes"), (review.REVIEW_AT - pd.Timedelta(seconds=1), "awaiting-outcomes")]:
                report, finalized = review.publish_review(at, archive, output)
                self.assertEqual(report["status"], status)
                self.assertEqual(report["plannedOrigins"], 408)
                self.assertEqual(report["criteria"]["minimumCompletePairs"], 327)
                self.assertEqual(json.loads(output.read_text()), report)
                self.assertFalse(finalized)
                self.assertFalse(archive.exists())
            with self.assertRaises(FileNotFoundError):
                review.publish_review(review.REVIEW_AT, archive, output)
            self.assertEqual(json.loads(output.read_text())["status"], "awaiting-review")
            self.assertFalse(archive.exists())

    def test_only_on_time_public_issues_and_complete_regional_values_qualify(self):
        base = issue(review.START, review.CURRENT, 20, 100)
        partial = dict(base, areas={"DK1": {"wind": [None] * 48}, "DK2": {"wind": [0] * 48}})
        records = [dict(base, diagnosticOnly=True), dict(base, modelVersion="different"), dict(base, times=base["times"][1:]),
                   issue(review.START, review.CURRENT, 60, 0), issue(review.START, review.CURRENT, -1, 0),
                   issue(review.START - review.HOUR, review.CURRENT, 59, 0), issue(review.STOP, review.CURRENT, 0, 0),
                   partial, issue(review.STOP - review.HOUR, review.CURRENT, 59, 0)]
        valid = list(review.on_time_issues(records, review.CURRENT))
        self.assertEqual(len(valid), 2)
        self.assertEqual(list(valid[0]["areas"]), ["DK2"])
        self.assertEqual(valid[1]["times"][-1], (review.STOP + 47 * review.HOUR).isoformat())

    def test_final_uses_the_common_issued_pair_and_freezes_even_after_data_revisions(self):
        old = issue(review.START, review.PREVIOUS, 10, 115)
        current = issue(review.START, review.CURRENT, 30, 103, ["DK2"])
        next_old = issue(review.START + review.HOUR, review.PREVIOUS, 10, 200)
        next_new = issue(review.START + review.HOUR, review.CURRENT, 20, 200)
        # The next pair shares 47 outcomes. Its last reading arrived after this review.
        rows = [dict(area=a, time=t, wind=100, retrievedAt=(pd.Timestamp(t) + review.HOUR).isoformat())
                for a in ("DK1", "DK2") for t in old["times"]]
        rows += [dict(area=a, time=next_old["times"][-1], wind=100,
                      retrievedAt=(review.REVIEW_AT + review.HOUR).isoformat()) for a in ("DK1", "DK2")]
        state = dict(schema=1, hourly=rows, forecasts=[old, issue(review.START, review.PREVIOUS, 40, 190), next_old],
                     windForecasts=[current, issue(review.START, review.CURRENT, 45, 108), next_new])
        with TemporaryDirectory() as folder:
            archive, output = Path(folder) / "archive.gz", Path(folder) / "report.json"
            data.write_state(state, archive)
            report, finalized = review.publish_review(review.REVIEW_AT, archive, output, "123")
            self.assertTrue(finalized)
            self.assertEqual(report["runId"], "123")
            east, west = report["areas"]["DK2"], report["areas"]["DK1"]
            self.assertEqual((east["previousMAE"], east["currentMAE"]), (15, 3))
            self.assertEqual((west["previousMAE"], west["currentMAE"]), (90, 8))
            self.assertEqual([b["hours"] for b in east["byLead"]], [6, 18, 24])
            self.assertEqual((east["origins"], east["previousOrigins"], east["currentOrigins"]), (1, 2, 2))
            self.assertEqual(east["decision"], "insufficient-evidence")
            saved = data.read_state(archive)
            saved["hourly"] = []
            data.write_state(saved, archive)
            before = archive.read_bytes()
            output.unlink()
            restored, finalized = review.publish_review(review.REVIEW_AT + review.HOUR, archive, output, "456")
            self.assertFalse(finalized)
            self.assertEqual(restored, report)
            self.assertEqual(json.loads(output.read_text()), report)
            self.assertEqual(archive.read_bytes(), before)

    def test_overall_lead_reliability_and_sample_rules_are_independent(self):
        band = {"currentMAE": 105.0, "previousMAE": 100.0}
        scores = dict(origins=327, hours=327 * 48, previousMAE=100.0, currentMAE=95.0, byLead=[band] * 3,
                      previousOrigins=408, currentOrigins=388)
        def decide(**changes):
            return review.regional_decision({**scores, **changes})["decision"]

        self.assertEqual(decide(), "eligible-to-retire-previous")
        self.assertEqual(decide(origins=326), "insufficient-evidence")
        self.assertEqual(decide(currentOrigins=387), "keep-previous-available")
        self.assertEqual(decide(currentMAE=95.01), "keep-previous-available")
        self.assertEqual(decide(byLead=[band, band, {**band, "currentMAE": 105.01}]), "keep-previous-available")
        self.assertEqual(decide(previousMAE=0, currentMAE=0), "keep-previous-available")
