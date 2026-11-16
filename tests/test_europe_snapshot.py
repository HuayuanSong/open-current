"""Slow collection cannot roll back a newer snapshot or modify forecast state."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import europe_snapshot as snapshot
import fetch


class EuropeSnapshotTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.bundle, self.cache = self.root / "bundle", self.root / "cache"
        self.bundle.mkdir(); self.cache.mkdir()
        self.index = self.root / "index.json"
        self.index.write_text(json.dumps(dict(europeCollection=None)))
        self.at = datetime(2026, 11, 16, 14, tzinfo=timezone.utc)
        self.name = "public_power/dk.json"
        self.payload = dict(data=[{"example": 1}], license="CC BY 4.0", _retrieved_at="2026-11-16T13:00:00+00:00")
        self.manifest = dict(schema=1, collectedAt="2026-11-16T13:01:00+00:00", failures=[], files={})
        self.write()

    def write(self):
        path = self.bundle / self.name
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_text(json.dumps(self.payload))
        self.manifest["files"] = {self.name: hashlib.sha256(path.read_bytes()).hexdigest()}
        (self.bundle / "manifest.json").write_text(json.dumps(self.manifest))

    def install(self):
        return snapshot.install(self.bundle, self.cache, self.index, self.at)

    def test_new_raw_feeds_only_and_original_source_times_are_preserved(self):
        forecast = self.root / "forecast-state.json.gz"
        forecast.write_bytes(b"previous issued forecasts")
        self.assertTrue(self.install())
        self.assertEqual(json.loads((self.cache / self.name).read_text()), self.payload)
        self.assertEqual(forecast.read_bytes(), b"previous issued forecasts")

    def test_equal_and_older_collections_cannot_replace_published_data(self):
        for previous in ("2026-11-16T13:01:00+00:00", "2026-11-16T13:30:00+00:00"):
            self.index.write_text(json.dumps(dict(europeCollection={"collectedAt": previous})))
            self.assertFalse(self.install())
            self.assertFalse((self.cache / self.name).exists())

    def test_future_unlicensed_and_corrupt_bundles_fail_before_writing(self):
        variants = [dict(_retrieved_at="2026-11-16T15:00:00Z"), dict(license="unknown"), dict(data=None)]
        for fields in variants:
            with self.subTest(fields=fields):
                original = self.payload.copy()
                self.payload.update(fields); self.write()
                with self.assertRaises(ValueError):
                    self.install()
                self.assertFalse((self.cache / self.name).exists())
                self.payload = original
        self.write()
        (self.bundle / self.name).write_text("tampered")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.install()

    def test_unknown_paths_and_forecast_archives_are_rejected(self):
        for name in ("../escape.json", "forecast-state.json.gz", "scripts/fetch.py"):
            self.manifest["files"] = {name: "bad"}
            (self.bundle / "manifest.json").write_text(json.dumps(self.manifest))
            with self.assertRaisesRegex(ValueError, "Unexpected"):
                self.install()

    def test_fast_mode_never_calls_european_queues(self):
        with patch("sys.argv", ["fetch", "--source", "denmark"]), patch.object(fetch, "CACHE", self.cache), \
             patch.object(fetch, "fetch_europe") as europe, patch.object(fetch, "energinet", return_value=[]) as dk:
            fetch.main()
        europe.assert_not_called(); dk.assert_called_once()

    def test_collector_never_calls_danish_sources(self):
        with patch("sys.argv", ["fetch", "--source", "europe"]), patch.object(fetch, "CACHE", self.cache), \
             patch.object(fetch, "fetch_europe", return_value=[]) as europe, patch.object(fetch, "energinet") as dk:
            fetch.main()
        europe.assert_called_once(); dk.assert_not_called()
        self.assertTrue((self.cache / "europe-fetch-status.json").exists())

    def test_scheduled_publish_starts_after_source_hour_buffer(self):
        import re
        workflow = (Path(__file__).resolve().parents[1] / ".github/workflows/pages.yml").read_text()
        minutes = re.search(r'cron: "([0-9,]+) \* \* \* \*"', workflow).group(1)
        for minute in map(int, minutes.split(",")):
            at = self.at.replace(minute=minute)
            from datetime import timedelta
            latest_eligible_hour = (at - timedelta(minutes=15)).replace(minute=0)
            self.assertEqual(latest_eligible_hour, at.replace(minute=0))


if __name__ == "__main__":
    unittest.main()
