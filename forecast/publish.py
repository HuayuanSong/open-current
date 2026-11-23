"""Publish a compact panel payload and score only forecasts actually issued."""

from collections import defaultdict
import json

import pandas as pd

from .data import PUBLIC_PATH
from .features import AREAS, H, TARGETS
from .intervals import validation
from .trial.evaluate import issue_coverage


def performance(state, at, version, compatible=()):
    actual = {
        (reading["area"], reading["time"]): reading for reading in state["hourly"]
    }
    totals = {
        area: {
            target: dict(
                error=0.0,
                actual=0.0,
                count=0,
                origins=set(),
                range_count=0,
                covered=0,
                width=0.0,
                range_origins=set(),
            )
            for target in TARGETS
        }
        for area in AREAS
    }
    range_version = (validation() or {}).get("version")
    earliest_issue = at - pd.Timedelta(days=7)
    latest_observation = at - pd.Timedelta(minutes=15)
    for record in state["forecasts"]:
        if record["modelVersion"] not in (version, *compatible):
            continue
        issued_at = pd.Timestamp(record["issuedAt"])
        if issued_at < earliest_issue:
            continue
        eligible_hours = []
        for index, stamp in enumerate(record["times"]):
            valid_time = pd.Timestamp(stamp)
            if valid_time <= issued_at or valid_time > latest_observation:
                continue
            eligible_hours.append((index, stamp))
        for area in record["areas"]:
            for index, stamp in eligible_hours:
                observed = actual.get((area, stamp))
                if not observed:
                    continue
                for target in TARGETS:
                    value = observed.get(target)
                    if value is None:
                        continue
                    stat = totals[area][target]
                    stat["error"] += abs(record["areas"][area][target][index] - value)
                    stat["actual"] += value
                    stat["count"] += 1
                    stat["origins"].add(record["id"])
                    ranges = record.get("ranges", {})
                    band = ranges.get("areas", {}).get(area, {}).get(target)
                    if (
                        band
                        and ranges.get("version") == range_version
                        and ranges.get("issuedAt") == record["issuedAt"]
                    ):
                        lower = band["lower"][index]
                        upper = band["upper"][index]
                        if lower is not None and upper is not None:
                            stat["range_count"] += 1
                            stat["covered"] += int(lower <= value <= upper)
                            stat["width"] += upper - lower
                            stat["range_origins"].add(record["id"])
    return {
        area: {
            target: {
                "maeMW": stats["error"] / stats["count"] if stats["count"] else None,
                "wapePct": (
                    100 * stats["error"] / stats["actual"]
                    if stats["actual"] > 0
                    else None
                ),
                "forecastHours": stats["count"],
                "origins": len(stats["origins"]),
                "rangeCoverage": (
                    stats["covered"] / stats["range_count"]
                    if stats["range_count"]
                    else None
                ),
                "rangeHours": stats["range_count"],
                "rangeOrigins": len(stats["range_origins"]),
                "meanRangeWidthMW": (
                    stats["width"] / stats["range_count"]
                    if stats["range_count"]
                    else None
                ),
            }
            for target, stats in targets.items()
        }
        for area, targets in totals.items()
    }


def base_payload(state, at, version, error=None, compatible=(), *, both_regions=False):
    records = [r for r in state["forecasts"] if r["modelVersion"] in (version, *compatible)
               and (not both_regions or all(area in r["areas"] for area in AREAS))]
    latest = (
        max(records, key=lambda r: (r["origin"], r["issuedAt"])) if records else None
    )
    if not latest:
        status = "unavailable"
    elif error:
        status = "delayed"
    else:
        status = "ready"
    result = {
        "schema": 1,
        "checkedAt": at.isoformat(),
        "modelVersion": version,
        "experimental": True,
        "status": status,
        "message": error,
        "latest": latest,
        "comparison": None,
        "observations": {},
        "performance": performance(state, at, version, compatible),
        "rangeValidation": validation(),
        "methodologyUrl": "https://github.com/HuayuanSong/open-current/tree/main/research/forecasting/v2",
    }
    observation_cutoff = at - pd.Timedelta(hours=72)
    for area in AREAS:
        result["observations"][area] = [
            {k: v for k, v in r.items() if k != "retrievedAt"}
            for r in state["hourly"]
            if r["area"] == area and pd.Timestamp(r["time"]) >= observation_cutoff
        ]
    if latest:
        latest_origin = pd.Timestamp(latest["origin"])
        prior = [r for r in records if pd.Timestamp(r["origin"]) < latest_origin]
        if prior:
            # Prefer a forecast issued about 24 hours earlier; never reconstruct it.
            target = latest_origin - pd.Timedelta(hours=24)
            result["comparison"] = min(
                prior, key=lambda r: abs(pd.Timestamp(r["origin"]) - target)
            )
    return result


def paired_wind(previous, current, hourly, at, *, issued_since=None, require_full_horizon=False):
    """Compare one prospectively issued pair per region and reference hour."""
    cutoff = at - pd.Timedelta(minutes=15)
    start = at - pd.Timedelta(days=7) if issued_since is None else issued_since
    actual = {(r["area"], r["time"]): r.get("wind") for r in hourly}
    result = {}
    for area in AREAS:
        groups = []
        for records in (previous, current):
            group = defaultdict(list)
            for r in records:
                if area in r["areas"] and start <= pd.Timestamp(r["issuedAt"]) <= at:
                    group[r["origin"]].append(r)
            groups.append(group)
        old, new = groups
        sums = [[0, 0., 0.] for _ in range(3)]
        hours, comparisons = set(), []
        origins = 0
        for origin in sorted(old.keys() & new.keys()):
            challenger = min(new[origin], key=lambda r: r["issuedAt"])
            issued = pd.Timestamp(challenger["issuedAt"])
            prior = [r for r in old[origin] if pd.Timestamp(r["issuedAt"]) <= issued]
            if not prior:
                continue
            baseline = max(prior, key=lambda r: r["issuedAt"])
            if baseline["times"] != challenger["times"]:
                continue
            outcomes = []
            for i, stamp in enumerate(challenger["times"]):
                valid = pd.Timestamp(stamp)
                value = actual.get((area, stamp))
                if value is not None and issued < valid <= cutoff:
                    outcomes.append((i, stamp, value))
            if not outcomes or (require_full_horizon and len(outcomes) != H):
                continue
            for i, stamp, value in outcomes:
                band = sums[0 if i < 6 else 1 if i < 24 else 2]
                band[0] += 1
                band[1] += abs(baseline["areas"][area]["wind"][i] - value)
                band[2] += abs(challenger["areas"][area]["wind"][i] - value)
                hours.add(stamp)
            origins += 1
            if pd.Timestamp(origin) >= at - pd.Timedelta(hours=71):
                comparisons.append((origin, baseline, challenger))
        def metric(total):
            n, a, b = total
            return dict(hours=n, previousMAE=a/n if n else None, currentMAE=b/n if n else None)
        summary = dict(**metric([sum(v) for v in zip(*sums)]), origins=origins, distinctHours=len(hours),
                       previousOrigins=len(old), currentOrigins=len(new), comparison=None,
                       byLead=[dict(start=lo, end=hi, **metric(values)) for lo, hi, values in
                               zip((1, 7, 25), (6, 24, 48), sums)])
        if comparisons:
            target = cutoff - pd.Timedelta(hours=24)
            _, previous_record, current_record = min(comparisons, key=lambda p: abs(pd.Timestamp(p[0])-target))
            summary["comparison"] = dict(previous=previous_record, current=current_record, issuedAfter=current_record["issuedAt"])
        result[area] = summary
    return result


def payload(state, at, version, error=None):
    result = base_payload(state, at, version, error)
    trial = state.get("trial") or {}
    if "protocol" in trial:
        result["weatherTrialCoverage"] = issue_coverage(trial, at)
    serving = state.get("windServing")
    if serving:
        wind_state = {**state, "forecasts": [r for r in state.get("windForecasts", []) if not r.get("diagnosticOnly")]}
        compatible = serving.get("compatibleVersions", [])
        versions = (serving["version"], *compatible)
        records = [r for r in wind_state["forecasts"] if r["modelVersion"] in versions]
        wind = base_payload(wind_state, at, serving["version"], serving["lastAttempt"].get("error"), compatible, both_regions=True)
        wind["lastAttempt"] = serving["lastAttempt"]
        wind["latestByArea"] = {}
        wind["waitingForWeather"] = []
        newest = max(records, key=lambda r: r["issuedAt"], default=None)
        if newest and newest.get("missingWeather"):
            wind["waitingForWeather"] = [area for area in AREAS if area not in newest["areas"]]
        for area in AREAS:
            regional = max((r for r in records if area in r["areas"]),
                           key=lambda r: (r["origin"], r["issuedAt"]), default=None)
            if regional and regional is not wind["latest"]:
                wind["latestByArea"][area] = regional
        result["windComparison"] = paired_wind(
            [r for r in state["forecasts"] if r["modelVersion"] == version], records, state["hourly"], at)
        wind["rangeValidation"] = None
        wind["methodologyUrl"] = "https://github.com/HuayuanSong/open-current/tree/main/research/forecasting/fresh_wind"
        wind["performance"] = {area: {"wind": targets["wind"]} for area, targets in wind["performance"].items()}
        result["windOutlook"] = wind
    return result


def write_public(state, at, version, error=None):
    public = payload(state, at, version, error)
    temporary = PUBLIC_PATH.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(public, separators=(",", ":"), allow_nan=False) + "\n"
    )
    temporary.replace(PUBLIC_PATH)
    return public
