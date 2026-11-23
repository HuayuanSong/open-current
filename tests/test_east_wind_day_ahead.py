import copy
from datetime import date
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import pandas as pd
import requests

from forecast import data, east_wind_day_ahead as bench
from forecast.trial.ledger import canonical_hash


def wind_record(day, mw=100, minute=-30, area="DK2", diagnostic=False):
    issued = bench.release(day) + pd.Timedelta(minutes=minute)
    origin = issued.floor("h")
    targets = pd.date_range(origin + bench.HOUR, periods=48, freq="h")
    r = dict(id=f"{area}-{minute}", modelVersion="public-version", origin=origin.isoformat(),
             issuedAt=issued.isoformat(), diagnosticOnly=diagnostic, times=[t.isoformat() for t in targets],
             areas={area: {"wind": [mw] * 48}})
    r["sha256"] = canonical_hash(r)
    return r


def forecasts(day):
    rows = [dict(PriceArea="DK2", HourUTC=(end - bench.HOUR).strftime("%Y-%m-%dT%H:%M:%S"),
                 ForecastType=kind, ForecastDayAhead=value)
            for end in bench.hour_ends(day) for kind, value in zip(bench.WIND_TYPES, (0, 40))]
    return dict(dataset="Forecasts_Hour", total=len(rows), records=rows[::-1])


class EastDayAheadTests(unittest.TestCase):
    day = bench.FIRST_DAY

    def test_release_request_and_hours_follow_danish_calendar_across_both_clock_changes(self):
        for day, count, release_hour in [(self.day, 24, 17), (date(2027, 3, 28), 23, 17), (date(2027, 10, 31), 25, 16)]:
            with self.subTest(day=day):
                ends = bench.hour_ends(day)
                self.assertEqual(len(ends), count)
                self.assertEqual(bench.release(day).hour, release_hour)
                self.assertEqual(bench.release(day).tz_convert(bench.ZONE).hour, 18)
                self.assertEqual((ends[0] - bench.HOUR).tz_convert(bench.ZONE).date(), day)
                self.assertEqual(ends[-1].tz_convert(bench.ZONE).date(), day + bench.DAY)
                source = forecasts(day)
                with patch.object(requests, "get") as get:
                    get.return_value.json.return_value = source
                    parts = bench.operator_values(bench.fetch_energinet(day), ends)
                    params = get.call_args.kwargs["params"]
                    self.assertEqual((params["start"], params["end"]), (str(day), str(day + bench.DAY)))
                    self.assertEqual(json.loads(params["filter"]), {"PriceArea": ["DK2"], "ForecastType": list(bench.WIND_TYPES)})
                    self.assertNotIn("ForecastCurrent", params["columns"])
                self.assertEqual(list(parts), list(ends))
                self.assertTrue(all(v == {"Onshore Wind": 0, "Offshore Wind": 40} for v in parts.values()))

    def test_missing_duplicate_wrong_zone_or_wrong_hour_never_becomes_zero(self):
        for change in ({"PriceArea": "DK1"}, {"ForecastDayAhead": None}, {"ForecastDayAhead": True},
                       {"ForecastDayAhead": float("nan")}, {"ForecastCurrent": 999, "ForecastDayAhead": None},
                       {"HourUTC": None}, {"HourUTC": "bad"}, {"HourUTC": "2026-11-27T00:00:00"},
                       {"ForecastType": "Onshore Wind"}):
            payload = forecasts(self.day)
            payload["records"][0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                bench.operator_values(payload, bench.hour_ends(self.day))
        bad = forecasts(self.day)
        bad["records"].pop()
        with self.assertRaisesRegex(ValueError, "complete"):
            bench.operator_values(bad, bench.hour_ends(self.day))

    def test_only_latest_eligible_issued_public_forecast_enters_the_blend(self):
        cutoff = bench.release(self.day)
        records = [wind_record(self.day, mw=55, minute=-60), wind_record(self.day),
                   wind_record(self.day, mw=9, minute=-2, area="DK1"),
                   wind_record(self.day, mw=999, minute=-1, diagnostic=True), wind_record(self.day, mw=5, minute=1)]
        days = {}
        fetch = Mock(return_value=forecasts(self.day))
        self.assertFalse(bench.collect(days, records, cutoff - pd.Timedelta(seconds=1), lambda: cutoff, fetch))
        fetch.assert_not_called()
        self.assertTrue(bench.collect(days, records, cutoff, lambda: cutoff, fetch))
        issue = days[str(self.day)]["issue"]
        self.assertEqual(issue["openCurrent"]["id"], records[1]["id"])
        self.assertEqual(issue["openCurrent"]["sha256"], records[1]["sha256"])
        self.assertEqual(len(issue["hours"]), 24)
        self.assertEqual(next(iter(issue["hours"].values())),
                         dict(openCurrent=100, energinet=40, halfAndHalf=70, onshoreMW=0, offshoreMW=40))
        self.assertEqual(issue["sha256"], canonical_hash({k: v for k, v in issue.items() if k != "sha256"}))
        self.assertFalse(bench.collect(days, records, cutoff + bench.HOUR, lambda: cutoff, fetch))
        fetch.assert_called_once_with(self.day)
        records[1]["areas"]["DK2"]["wind"][0] = 333
        with self.assertRaisesRegex(ValueError, "checksum"):
            bench.public_values(records, cutoff, bench.hour_ends(self.day))

    def test_incomplete_pair_or_late_capture_is_never_backfilled(self):
        cutoff, ends = bench.release(self.day), bench.hour_ends(self.day)
        days, records = {}, [wind_record(self.day)]
        fetch = Mock(return_value=forecasts(self.day))
        fetch.return_value["records"][0]["ForecastDayAhead"] = None
        bench.collect(days, records, cutoff, lambda: cutoff, fetch)
        self.assertNotIn("issue", days[str(self.day)])
        self.assertEqual(bench.report(days, cutoff)["days"][0]["status"], "awaiting-capture")
        fetch.return_value = forecasts(self.day)
        start = ends[0] - bench.HOUR
        bench.collect(days, records, start - pd.Timedelta(seconds=1), lambda: start, fetch)
        self.assertNotIn("issue", days[str(self.day)])
        self.assertFalse(bench.collect(days, records, start, lambda: start, fetch))
        report = bench.report(days, start)
        self.assertEqual((report["missedDays"], report["issuedDays"], fetch.call_count), (1, 0, 2))
        self.assertIn("missed", report["days"][0]["reason"])
        gap = bench.report({}, start + 2 * bench.DAY)
        self.assertEqual(gap["missedDays"], 3)

    def test_no_usable_public_forecast_at_cutoff_skips_the_source_request(self):
        cutoff = bench.release(self.day)
        short = wind_record(self.day, minute=-5)
        short["times"] = short["times"][:10]
        short["areas"]["DK2"]["wind"] = [100] * 10
        del short["sha256"]
        short["sha256"] = canonical_hash(short)
        for records in ([wind_record(self.day, minute=1)], [wind_record(self.day), short]):
            with self.subTest(records=len(records)):
                days, fetch = {}, Mock()
                bench.collect(days, records, cutoff, lambda: cutoff, fetch)
                fetch.assert_not_called()
                self.assertNotIn("issue", days[str(self.day)])

    def test_retry_after_then_recover_before_midnight(self):
        cutoff = bench.release(self.day)
        response = requests.Response()
        response.status_code = 429
        response.headers["Retry-After"] = "3600"
        fetch = Mock(side_effect=[requests.HTTPError(response=response), forecasts(self.day)])
        days, records = {}, [wind_record(self.day)]
        bench.collect(days, records, cutoff, lambda: cutoff, fetch)
        self.assertFalse(bench.collect(days, records, cutoff + bench.HOUR / 2, lambda: cutoff, fetch))
        self.assertIn("429", days[str(self.day)]["reason"])
        self.assertTrue(bench.collect(days, records, cutoff + bench.HOUR, lambda: cutoff + bench.HOUR, fetch))
        self.assertIn("issue", days[str(self.day)])
        self.assertNotIn("reason", days[str(self.day)])
        self.assertEqual(fetch.call_count, 2)

    def test_rate_limit_can_span_two_evenings(self):
        cutoff = bench.release(self.day)
        response = requests.Response()
        response.status_code = 429
        response.headers["Retry-After"] = str(25 * 3600)
        days, fetch = {}, Mock(side_effect=requests.HTTPError(response=response))
        records = [wind_record(self.day), wind_record(self.day + bench.DAY)]
        bench.collect(days, records, cutoff, lambda: cutoff, fetch)
        tomorrow = bench.release(self.day + bench.DAY)
        self.assertFalse(bench.collect(days, records, tomorrow, lambda: tomorrow, fetch))
        fetch.assert_called_once()

    def test_frozen_issues_and_shared_mature_outcomes_survive_restore_and_source_revisions(self):
        cutoff, ends = bench.release(self.day), bench.hour_ends(self.day)
        state = dict(schema=1, hourly=[], windForecasts=[wind_record(self.day)], forecasts=[], unrelated={"untouched": 1})
        fetch = Mock(return_value=forecasts(self.day))
        with TemporaryDirectory() as temp:
            path, out = Path(temp) / "state.gz", Path(temp) / "report.json"
            with self.assertRaises(FileNotFoundError):
                bench.refresh(path, out, lambda: cutoff, fetch)
            data.write_state(state, path)
            scheduled = bench.refresh(path, out, lambda: cutoff - bench.HOUR, fetch)
            self.assertEqual((scheduled["days"], scheduled["nextCutoff"]), ([], cutoff.isoformat()))
            first = bench.refresh(path, out, lambda: cutoff, fetch)
            self.assertEqual((first["issuedDays"], first["pairedHours"], first["maeMW"]["openCurrent"]), (1, 0, None))
            state = data.read_state(path)
            frozen = copy.deepcopy(state[bench.KEY][str(self.day)]["issue"])
            def actual(end, mw, minutes=15, area="DK2"):
                return dict(time=end.isoformat(), area=area, wind=mw, retrievedAt=(end + pd.Timedelta(minutes=minutes)).isoformat())
            state["hourly"] = [actual(ends[0], 70, 14), actual(ends[0], 100, area="DK1"),
                               actual(ends[0], 60), actual(ends[1], None), actual(ends[2], 80)]
            state["windForecasts"] = []
            data.write_state(state, path)
            partial = bench.refresh(path, out, lambda: ends[1] + bench.HOUR / 4, fetch)
            self.assertEqual(partial["pairedHours"], 1)
            self.assertEqual(partial["maeMW"], dict(openCurrent=40, energinet=20, halfAndHalf=10))
            state = data.read_state(path)
            state["hourly"] = [actual(ends[0], 999), *[actual(end, 80) for end in ends[1:]]]
            data.write_state(state, path)
            full = bench.refresh(path, out, lambda: ends[-1] + bench.HOUR / 2, fetch)
            # No run captured the following day while this day's last observation matured.
            self.assertEqual((full["pairedHours"], full["issuedDays"], full["completeDays"], full["missedDays"]), (24, 1, 1, 1))
            self.assertAlmostEqual(full["maeMW"]["openCurrent"], (40 + 23 * 20) / 24)
            self.assertAlmostEqual(full["maeMW"]["energinet"], (20 + 23 * 40) / 24)
            self.assertEqual(full["maeMW"]["halfAndHalf"], 10)
            self.assertEqual(json.loads(out.read_text()), full)
            saved = data.read_state(path)
            self.assertEqual(saved[bench.KEY][str(self.day)]["issue"], frozen)
            self.assertEqual(saved[bench.KEY][str(self.day)]["observations"][ends[0].isoformat()]["mw"], 60)
            self.assertEqual(saved["unrelated"], {"untouched": 1})
            fetch.assert_called_once()
            frozen_hour = saved[bench.KEY][str(self.day)]["issue"]["hours"][ends[0].isoformat()]
            frozen_hour["halfAndHalf"] = 0
            with self.assertRaisesRegex(ValueError, "checksum"):
                bench.report(saved[bench.KEY], ends[-1] + bench.HOUR)
