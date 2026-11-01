"""Source retry policies preserve provider delays without live requests."""

from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from email.message import Message
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

import pandas as pd
import requests

from forecast import data

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fetch


def response(status, retry_after=None):
    result = requests.Response()
    result.status_code = status
    result.url = "https://source.example.test/"
    result._content = b'{"ok":true}'
    if retry_after is not None:
        result.headers["Retry-After"] = retry_after
    return result


class ForecastRetryTests(unittest.TestCase):
    def test_numeric_and_http_date_delays_are_respected(self):
        for header in ["10", "Sun, 01 Nov 2026 12:00:10 GMT"]:
            with self.subTest(header=header), patch.object(
                data.requests, "get", side_effect=[response(429, header), response(200)]
            ) as get, patch.object(data.time, "sleep") as sleep, patch.object(
                data, "now", return_value=pd.Timestamp("2026-11-01T12:00Z")
            ):
                self.assertEqual(
                    data.request("https://source.example.test"), {"ok": True}
                )
                self.assertEqual(get.call_count, 2)
                sleep.assert_called_once_with(10)

    def test_delays_above_the_job_budget_fail_without_an_early_retry(self):
        for header in ["120", "Sun, 01 Nov 2026 12:02:00 GMT"]:
            with self.subTest(header=header), patch.object(
                data.requests, "get", side_effect=[response(429, header), response(200)]
            ) as get, patch.object(data.time, "sleep") as sleep, patch.object(
                data, "now", return_value=pd.Timestamp("2026-11-01T12:00Z")
            ):
                with self.assertRaises(requests.HTTPError):
                    data.request("https://source.example.test")
                self.assertEqual(get.call_count, 1)
                sleep.assert_not_called()

    def test_malformed_retry_delays_use_the_normal_finite_fallback(self):
        for header in ["bad date", "", "NaN", "inf", "-1"]:
            with self.subTest(header=header), patch.object(
                data.requests, "get", side_effect=[response(503, header), response(200)]
            ), patch.object(data.time, "sleep") as sleep:
                self.assertEqual(
                    data.request("https://source.example.test"), {"ok": True}
                )
                sleep.assert_called_once_with(5)

    def test_nonretryable_errors_and_exhaustion_remain_bounded(self):
        for status, expected_calls in [(404, 1), (503, 3)]:
            with self.subTest(status=status), patch.object(
                data.requests, "get", return_value=response(status)
            ) as get, patch.object(data.time, "sleep") as sleep:
                with self.assertRaises(requests.HTTPError):
                    data.request("https://source.example.test")
                self.assertEqual(get.call_count, expected_calls)
                self.assertEqual(sleep.call_count, expected_calls - 1)


class CollectionRetryTests(unittest.TestCase):
    def retry_error(self, header):
        headers = Message()
        headers["Retry-After"] = header
        return HTTPError(
            "https://source.example.test", 429, "wait", headers, io.BytesIO()
        )

    def collect_with_clock(
        self,
        failures_before_success,
        interval=31,
        terminal_retry_after="1",
        keys=("dk", "de"),
    ):
        elapsed = 0
        starts = []
        payload = json.dumps({"license": "CC BY 4.0", "data": []}).encode()

        class ClockDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                current = datetime(2026, 11, 1, 12, tzinfo=timezone.utc) + timedelta(
                    seconds=elapsed
                )
                return current.astimezone(tz) if tz else current.replace(tzinfo=None)

        def sleep(seconds):
            nonlocal elapsed
            elapsed += seconds

        def open_response(_request, **_kwargs):
            nonlocal elapsed
            starts.append(elapsed)
            elapsed += 5  # Response handling time also counts toward spacing.
            if len(starts) <= failures_before_success:
                header = (
                    terminal_retry_after
                    if len(starts) == failures_before_success
                    else "1"
                )
                raise self.retry_error(header)
            return io.BytesIO(payload)

        with tempfile.TemporaryDirectory() as directory, patch.object(
            fetch, "CACHE", Path(directory)
        ), patch.object(
            fetch.urllib.request, "urlopen", side_effect=open_response
        ), patch.object(
            fetch.time, "monotonic", side_effect=lambda: elapsed
        ), patch.object(
            fetch, "datetime", ClockDateTime
        ), patch.object(
            fetch.time, "sleep", side_effect=sleep
        ), redirect_stdout(
            io.StringIO()
        ):
            failures = fetch.queue("public_power", keys, "country", 1, interval)
        return starts, failures, elapsed

    def test_successful_requests_are_paced_start_to_start(self):
        starts, failures, _ = self.collect_with_clock(0)
        self.assertEqual(starts, [0, 31])
        self.assertEqual(failures, [])

    def test_next_country_waits_from_last_attempt_after_successful_retry(self):
        starts, failures, _ = self.collect_with_clock(1)
        self.assertEqual(starts, [0, 31, 62])
        self.assertEqual(failures, [])

    def test_next_country_waits_from_last_attempt_after_retry_exhaustion(self):
        starts, failures, _ = self.collect_with_clock(4)
        self.assertEqual(starts, [0, 31, 62, 93, 124])
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["key"], "dk")

    def test_terminal_backoff_is_preserved_when_collecting_the_next_country(self):
        for header in ["120", "Sun, 01 Nov 2026 12:03:38 GMT"]:
            with self.subTest(header=header):
                starts, failures, _ = self.collect_with_clock(
                    4, terminal_retry_after=header
                )
                self.assertEqual(starts[:4], [0, 31, 62, 93])
                self.assertGreaterEqual(starts[4] - (starts[3] + 5), 120)
                self.assertEqual(len(failures), 1)

    def test_terminal_backoff_does_not_sleep_when_no_more_keys_remain(self):
        starts, failures, elapsed = self.collect_with_clock(
            4, terminal_retry_after="120", keys=("dk",)
        )
        self.assertEqual(starts, [0, 31, 62, 93])
        self.assertEqual(elapsed, 98)
        self.assertEqual(len(failures), 1)

    def test_malformed_terminal_backoff_uses_normal_endpoint_spacing(self):
        for header in ["bad date", "inf", "-1"]:
            with self.subTest(header=header):
                starts, failures, _ = self.collect_with_clock(
                    4, terminal_retry_after=header
                )
                self.assertEqual(starts, [0, 31, 62, 93, 124])
                self.assertEqual(len(failures), 1)

    def test_energy_charts_retries_keep_the_configured_endpoint_spacing(self):
        starts, failures, _ = self.collect_with_clock(1, interval=45)
        self.assertEqual(starts, [0, 45, 90])
        self.assertEqual(failures, [])

    def test_the_faster_feed_policy_does_not_inherit_energy_charts_spacing(self):
        with patch.object(
            fetch.urllib.request,
            "urlopen",
            side_effect=[self.retry_error("1"), io.BytesIO(b'{"records":[]}')],
        ), patch.object(fetch.time, "sleep") as sleep:
            self.assertEqual(
                fetch.fetch_json("https://source.example.test"), {"records": []}
            )
            sleep.assert_called_once_with(2)

    def test_nonfinite_retry_header_cannot_produce_an_unbounded_sleep(self):
        with patch.object(
            fetch.urllib.request,
            "urlopen",
            side_effect=[self.retry_error("inf"), io.BytesIO(b"{}")],
        ), patch.object(fetch.time, "sleep") as sleep:
            self.assertEqual(fetch.fetch_json("https://source.example.test"), {})
            self.assertTrue(math.isfinite(sleep.call_args.args[0]))


if __name__ == "__main__":
    unittest.main()
