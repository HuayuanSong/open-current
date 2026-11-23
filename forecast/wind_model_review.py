"""The 13 December decision on displaying the two issued public wind models."""

import json
import math
import os

import pandas as pd

from . import data
from .features import AREAS, H
from .observation_time import valid_wind
from .publish import paired_wind

START = pd.Timestamp("2026-11-24T00:00Z")
STOP = pd.Timestamp("2026-12-11T00:00Z")
REVIEW_AT = pd.Timestamp("2026-12-13T00:00Z")
HOUR = pd.Timedelta(hours=1)
SLOTS = (STOP - START) // HOUR
PREVIOUS = "ensemble-v2-2026-10-27"
CURRENT = "fresh-wind-2026-11-18"
MIN_GAIN = 5
MAX_LEAD_REGRESSION = 5
MAX_ISSUANCE_GAP = 5
MIN_COMPLETE_PAIRS = math.ceil(0.8 * SLOTS)
ARCHIVE_KEY = "publicWindModelReview20261213"


def on_time_issues(records, version):
    for record in records:
        if record["modelVersion"] != version or record.get("diagnosticOnly"):
            continue
        origin = pd.Timestamp(record["origin"])
        issued = pd.Timestamp(record["issuedAt"])
        if not (START <= origin < STOP and origin == origin.floor("h") and origin <= issued < origin + HOUR):
            continue
        targets = pd.date_range(origin + HOUR, periods=H, freq="h")
        if len(record["times"]) != H or any(pd.Timestamp(t) != expected for t, expected in zip(record["times"], targets)):
            continue
        areas = {
            area: values for area, values in record["areas"].items()
            if area in AREAS and isinstance(values.get("wind"), list)
            and len(values["wind"]) == H and all(map(valid_wind, values["wind"]))
        }
        if areas:
            yield dict(origin=origin.tz_convert("UTC").isoformat(), issuedAt=issued.tz_convert("UTC").isoformat(),
                       times=[t.isoformat() for t in targets.tz_convert("UTC")], areas=areas)


def eligible_observations(rows, at):
    for row in rows:
        stamp = pd.Timestamp(row["time"])
        if not START < stamp <= STOP + (H - 1) * HOUR or not valid_wind(row.get("wind")):
            continue
        captured = pd.to_datetime(row.get("retrievedAt"), utc=True, errors="coerce")
        if pd.notna(captured) and stamp + pd.Timedelta(minutes=15) <= captured <= at:
            yield dict(area=row["area"], time=stamp.tz_convert("UTC").isoformat(), wind=row["wind"])


def regional_decision(scores):
    summary = {key: value for key, value in scores.items() if key != "comparison"}
    old, new = scores["previousMAE"], scores["currentMAE"]
    gain = 100 * (1 - new / old) if old else None
    gap = 100 * (scores["previousOrigins"] - scores["currentOrigins"]) / SLOTS
    checks = dict(
        enoughCompletePairs=scores["origins"] >= MIN_COMPLETE_PAIRS,
        overallError=(old > 0 and new <= old * (1 - MIN_GAIN / 100)) if old is not None else None,
        leadErrors=all(b["currentMAE"] <= b["previousMAE"] * (1 + MAX_LEAD_REGRESSION / 100)
                       for b in scores["byLead"]) if scores["hours"] else None,
        issuance=gap <= MAX_ISSUANCE_GAP,
    )
    decision = "insufficient-evidence"
    if checks["enoughCompletePairs"]:
        decision = "eligible-to-retire-previous" if all(checks.values()) else "keep-previous-available"
    return dict(**summary, improvementPct=gain, issuanceGapPercentagePoints=gap, checks=checks, decision=decision)


def freeze_review(protocol, at, state_path, run_id):
    if not state_path.is_file():
        raise FileNotFoundError("Issued forecast archive is unavailable for the wind model review")
    state = data.read_state(state_path)
    if ARCHIVE_KEY in state:
        return state[ARCHIVE_KEY], False
    scores = paired_wind(
        list(on_time_issues(state["forecasts"], PREVIOUS)),
        list(on_time_issues(state.get("windForecasts", []), CURRENT)),
        eligible_observations(state["hourly"], at), at, issued_since=START, require_full_horizon=True,
    )
    report = {**protocol, "status": "reviewed", "reviewedAt": at.isoformat(), "runId": run_id,
              "areas": {area: regional_decision(values) for area, values in scores.items()}}
    state[ARCHIVE_KEY] = report
    state["updatedAt"] = at.isoformat()
    data.write_state(state, state_path)
    return report, True


def publish_review(at, state_path, output_path, run_id=None):
    """Publish a status while collecting, then reuse the decision from the first successful review."""
    report = dict(
        issueStart=START.isoformat(), issueEndExclusive=STOP.isoformat(), reviewAt=REVIEW_AT.isoformat(),
        previousModel=PREVIOUS, currentModel=CURRENT, plannedOrigins=SLOTS,
        criteria=dict(minimumMAEReductionPct=MIN_GAIN, maximumLeadRegressionPct=MAX_LEAD_REGRESSION,
                      maximumIssuanceGapPercentagePoints=MAX_ISSUANCE_GAP, minimumCompletePairs=MIN_COMPLETE_PAIRS),
    )
    finalized = False
    try:
        if at < START:
            report["status"] = "scheduled"
        elif at < STOP:
            report["status"] = "collecting"
        elif at < REVIEW_AT:
            report["status"] = "awaiting-outcomes"
        else:
            report["status"] = "awaiting-review"
            report, finalized = freeze_review(report, at, state_path, run_id)
    finally:
        temporary = output_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(output_path)
    return report, finalized


if __name__ == "__main__":
    report, finalized = publish_review(data.now(), data.STATE_PATH, data.PUBLIC_PATH.with_name("wind-model-review.json"),
                                       os.environ.get("GITHUB_RUN_ID"))
    print(f"Public wind comparison review · {report['status']} · {report['reviewAt']}")
    if finalized and os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            output.write("finalized=true\n")
