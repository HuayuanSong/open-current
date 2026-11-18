import copy
import json
from pathlib import Path
import unittest

import pandas as pd

from forecast import publish, wind_update
from forecast.trial import ledger
from forecast.trial.evaluate import issue_coverage
from forecast.weather_metadata import UNITS, validate_locations
from forecast.features import CONFIG


class ForecastReviewTests(unittest.TestCase):
    def test_only_common_outcomes_and_earliest_current_issue_enter_each_region(self):
        origin = pd.Timestamp("2026-11-17T09:00Z")
        times = [t.isoformat() for t in pd.date_range(origin + pd.Timedelta(hours=1), periods=48, freq="h")]
        def record(minute, value, areas):
            return dict(origin=origin.isoformat(), issuedAt=(origin+pd.Timedelta(minutes=minute)).isoformat(), times=times,
                        areas={area: dict(wind=[value]*48) for area in areas})
        old = [record(10, 90, ["DK1","DK2"]), record(40, 200, ["DK1","DK2"])]
        new = [record(30, 97, ["DK2"]), record(45, 100, ["DK1","DK2"])]
        actual = [dict(area=a,time=t,wind=100) for a in ["DK1","DK2"] for t in times]
        out = publish.paired_wind(old, new, actual, origin+pd.Timedelta(hours=7))
        self.assertEqual((out["DK2"]["previousMAE"],out["DK2"]["currentMAE"]),(10,3))
        self.assertEqual((out["DK1"]["previousMAE"],out["DK1"]["currentMAE"]),(100,0))
        self.assertEqual([b["hours"] for b in out["DK2"]["byLead"]],[6,0,0])
        self.assertIs(out["DK2"]["comparison"]["previous"],old[0])
        self.assertIs(out["DK2"]["comparison"]["current"],new[0])
        self.assertEqual(out["DK2"]["distinctHours"],6)

    def test_both_experiments_require_actual_retrieval_in_the_freezing_window(self):
        stamp = pd.Timestamp("2026-11-16T00:00Z")
        at = stamp + pd.Timedelta(hours=24, minutes=7)
        row = dict(time=stamp.isoformat(), area="DK2", wind=0)
        for freeze in [ledger.freeze_observations, wind_update.freeze_observations]:
            trial = dict(records=[dict(times=[row["time"]])], observations={},
                         protocol=dict(observationMinAgeHours=24,observationMaxAgeHours=72))
            for retrieved in [None,stamp+pd.Timedelta(hours=23,minutes=50),at+pd.Timedelta(minutes=1)]:
                freeze(trial,[dict(row,retrievedAt=retrieved.isoformat() if retrieved else None)],at)
                self.assertFalse(trial["observations"])
            freeze(trial,[dict(row,retrievedAt=at.isoformat())],at+pd.Timedelta(minutes=10))
            observation=trial["observations"][row["time"]]["DK2"]
            self.assertEqual(observation.get("wind",observation)["capturedAt"],at.isoformat())

    def test_trial_coverage_accounts_for_a_still_open_issue_window(self):
        protocol=json.loads((Path(__file__).parents[1]/"forecast/trial/protocol.json").read_text())
        records=[dict(origin=d.isoformat()) for d in pd.date_range("2026-10-27T18:00Z",periods=8)]
        result=issue_coverage(dict(records=records,protocol=protocol),pd.Timestamp("2026-11-18T18:30Z"))
        self.assertEqual((result["requiredIssues"],result["maximumPossibleIssues"]),(34,28))
        self.assertFalse(result["coverageReachable"])

    def test_weather_rejects_swapped_coordinates_units_and_timezone(self):
        points=[dict(latitude=p["latitude"],longitude=p["longitude"],hourly_units=UNITS.copy(),utc_offset_seconds=0)
                for p in CONFIG["locations"]]
        validate_locations(points)
        invalid=copy.deepcopy(points);invalid[0]["hourly_units"]["wind_speed_100m"]="km/h"
        for case in [points[3:]+points[:3],invalid,[dict(points[0],utc_offset_seconds=3600),*points[1:]]]:
            with self.assertRaises(ValueError):
                validate_locations(case)
