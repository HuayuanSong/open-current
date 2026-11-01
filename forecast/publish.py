"""Publish a compact panel payload and score only forecasts actually issued."""

import json

import pandas as pd

from .data import PUBLIC_PATH
from .features import AREAS, TARGETS
from .intervals import validation


def performance(state, at, version):
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
        if record["modelVersion"] != version:
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
        for area in AREAS:
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


def payload(state, at, version, error=None):
    records = [r for r in state["forecasts"] if r["modelVersion"] == version]
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
        "performance": performance(state, at, version),
        "rangeValidation": validation(),
        "methodologyUrl": "https://github.com/HuayuanSong/open-current/tree/main/research/forecasting/v2",
    }
    observation_cutoff = at - pd.Timedelta(hours=72)
    for area in AREAS:
        result["observations"][area] = [
            r
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


def write_public(state, at, version, error=None):
    public = payload(state, at, version, error)
    temporary = PUBLIC_PATH.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(public, separators=(",", ":"), allow_nan=False) + "\n"
    )
    temporary.replace(PUBLIC_PATH)
    return public
