from datetime import datetime, timezone
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from health import build_health


class HealthTests(unittest.TestCase):
    def test_code_deployment_does_not_make_old_observations_or_forecast_fresh(self):
        old = "2026-10-30T10:00:00+00:00"
        result = build_health(
            {"generatedAt": old, "countries": {"dk": {"power": {"t": 123}, "flows": None}}},
            {"checkedAt": "2026-11-02T10:00:00+00:00", "latest": {"issuedAt": old}},
            {"areas": {"DK1": {"power": [{"t": 200}, {"t": 100}], "co2": []}}},
            datetime(2026, 11, 2, tzinfo=timezone.utc),
        )
        self.assertEqual(result["snapshotAt"], old)
        self.assertEqual(result["forecast"]["issuedAt"], old)
        self.assertEqual(result["sources"]["energinet"]["DK1"], {"power": 200, "co2": None})
        self.assertIsNone(result["sources"]["energyCharts"]["dk"]["flows"])

    def test_missing_forecast_is_not_given_a_new_issue_time(self):
        result = build_health({"generatedAt": "2026-11-02T10:00:00Z"}, {}, {})
        self.assertIsNone(result["forecast"]["issuedAt"])
        self.assertIsNone(result["forecast"]["horizonEnd"])
