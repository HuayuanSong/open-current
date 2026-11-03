"""One complete Energinet collection serves observations and forecast inputs."""

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pandas as pd

from forecast import data

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fetch


class SharedElectricityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / ".cache/energinet/ElectricityProdex5MinRealtime.json"
        self.path.parent.mkdir(parents=True)
        self.at = pd.Timestamp("2026-11-03T13:30Z")
        self.retrieved = "2026-11-03T13:29:00+00:00"
        records = []
        for stamp in pd.date_range("2026-10-30T23:00Z", self.at, freq="5min"):
            for area in data.AREAS:
                row = dict(
                    Minutes5UTC=stamp.tz_localize(None).isoformat(), PriceArea=area,
                    OffshoreWindPower=100, OnshoreWindPower=200, SolarPower=50,
                    ProductionLt100MW=60, ProductionGe100MW=400,
                    ExchangeGermany=10, ExchangeSweden=20, ExchangeGreatBelt=30,
                    ExchangeNorway=40, ExchangeNetherlands=50, ExchangeGreatBritain=60,
                )
                if area == "DK2" and stamp == pd.Timestamp("2026-11-03T12:05Z"):
                    row["SolarPower"] = None
                records.append(row)
        self.payload = dict(
            dataset="ElectricityProdex5MinRealtime", records=records,
            total=len(records), _start="2026-10-31", _retrieved_at=self.retrieved,
        )
        self.previous = dict(time="2026-10-30T12:00:00+00:00", area="DK1",
                             wind=123, solar=0, load=456)

    def write(self, payload):
        self.path.write_text(json.dumps(payload))

    def test_shared_response_matches_direct_fetch_without_another_request(self):
        self.write(self.payload)
        shared = {"hourly": [self.previous.copy()]}
        direct = copy.deepcopy(shared)
        with patch.object(data, "ROOT", self.root), patch.object(data, "request") as get:
            shared_frame = data.electricity(shared, self.at)
        get.assert_not_called()
        self.path.unlink()
        with patch.object(data, "ROOT", self.root), patch.object(
            data, "request", return_value=self.payload
        ) as get:
            direct_frame = data.electricity(direct, self.at)
        get.assert_called_once()
        pd.testing.assert_frame_equal(shared_frame, direct_frame)
        self.assertEqual(shared["hourly"], direct["hourly"])
        self.assertEqual(shared["electricityRetrievedAt"], self.retrieved)
        self.assertIn(self.previous, shared["hourly"])
        self.assertLessEqual(shared_frame.time.max(), self.at.floor("h"))
        missing = shared_frame.loc[
            shared_frame.area.eq("DK2") & shared_frame.time.eq(self.at.floor("h")),
            "solar",
        ]
        self.assertTrue(missing.isna().all())

    def test_stale_partial_future_and_wrong_dataset_cache_cannot_be_reused(self):
        variants = [
            {"_retrieved_at": "2026-11-03T13:19:00Z"},
            {"_retrieved_at": "2026-11-03T13:31:00Z"},
            {"_retrieved_at": "2026-11-03T13:29:00"},
            {"_start": "2026-11-02"},
            {"total": self.payload["total"] + 1},
            {"records": []},
            {"dataset": "CO2Emis"},
            {"_retrieved_at": "invalid"},
        ]
        for changes in variants:
            with self.subTest(changes=changes), patch.object(data, "ROOT", self.root):
                self.write({**self.payload, **changes})
                self.assertIsNone(data.collected_electricity(
                    datetime(2026, 10, 31).date(), self.at
                ))

    def test_short_collection_cannot_replace_the_35_day_bootstrap_request(self):
        self.write(self.payload)
        with patch.object(data, "ROOT", self.root), patch.object(
            data, "request", return_value=self.payload
        ) as get:
            data.electricity({"hourly": []}, self.at)
        self.assertEqual(get.call_args.args[1]["start"], "2026-09-29")

    def test_collector_records_complete_three_day_window_for_forecast_reuse(self):
        class FixedDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                return datetime(2026, 11, 3, 13, 29, tzinfo=timezone.utc).astimezone(tz)

        co2 = dict(dataset="CO2Emis", total=1, records=[{"CO2Emission": 100}])
        with patch.object(fetch, "CACHE", self.root / ".cache"), patch.object(
            fetch, "datetime", FixedDateTime
        ), patch.object(fetch, "fetch_json", side_effect=[self.payload, co2]) as get:
            self.assertEqual(fetch.energinet(1), [])
        power_query, co2_query = [parse_qs(urlparse(call.args[0]).query) for call in get.call_args_list]
        self.assertEqual(power_query, {"start": ["2026-10-31"], "end": ["now"], "limit": ["5000"]})
        self.assertEqual(co2_query["startOffset"], ["-P1D"])
        with patch.object(data, "ROOT", self.root):
            self.assertIsNotNone(data.collected_electricity(datetime(2026, 10, 31).date(), self.at))

    def test_truncated_collection_does_not_replace_last_complete_response(self):
        self.write(self.payload)
        with patch.object(fetch, "CACHE", self.root / ".cache"), patch.object(
            fetch, "fetch_json", return_value={**self.payload, "total": self.payload["total"] + 1}
        ):
            failures = fetch.energinet(1)
        self.assertEqual(len(failures), 2)
        self.assertEqual(json.loads(self.path.read_text()), self.payload)


if __name__ == "__main__":
    unittest.main()
