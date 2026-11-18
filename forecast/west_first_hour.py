"""Two prospective DK1 first-hour forecasts scored beside the public wind model."""

import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean

import pandas as pd

from . import data
from .observation_time import capture_time
from .trial.ledger import canonical_hash


START = pd.Timestamp("2026-11-18T14:00Z")
STOP = START + pd.Timedelta(days=14)
REVIEW_AT = STOP + pd.Timedelta(hours=1)
HOUR = pd.Timedelta(hours=1)
SLOT = pd.Timedelta(minutes=5)
BUFFER = pd.Timedelta(minutes=10)
BASELINE = "fresh-wind-2026-11-18"
KEY = "westFirstHourTrial"
METHODS = {"public": "Public model", "hold": "15-minute hold", "partial": "Observed intervals + hold"}
SOURCE = data.ROOT / ".cache/fresh-wind/live/evidence/sources.json.gz"
REVIEW_FILE = data.ROOT / ".cache/west-first-hour/review.json"


def valid_power(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def alternatives(base, sources):
    """Use the same DK1 snapshot, time cutoff and hold level as the archived issue."""
    recent = base["recentWind"]["DK1"]
    origin = pd.Timestamp(base["origin"])
    end = pd.Timestamp(recent["observationEnd"])
    selected = pd.Timestamp(recent["selectedAt"])
    retrieved = pd.Timestamp(sources["electricityRetrievedAt"])
    if not (end <= retrieved <= selected <= pd.Timestamp(base["issuedAt"])
            and end + BUFFER <= selected and recent["bufferMinutes"] == 10):
        raise ValueError("Wind observation provenance differs from the trial protocol")

    first = min(origin, end - 3 * SLOT)
    readings = {}
    for row in sources["electricity"]:
        if row["PriceArea"] != "DK1":
            continue
        stamp = pd.to_datetime(row["Minutes5UTC"], utc=True)
        if pd.isna(stamp) or not first <= stamp <= end - SLOT or stamp != stamp.floor("5min"):
            continue
        parts = (row.get("OnshoreWindPower"), row.get("OffshoreWindPower"))
        if not all(valid_power(v) for v in parts):
            continue
        value = sum(parts)
        if stamp in readings and readings[stamp] != value:
            raise ValueError("Conflicting five-minute DK1 observations")
        readings[stamp] = value

    hold = recent["levelMW"]
    anchors = [readings.get(end - i * SLOT) for i in (1, 2, 3)]
    if not valid_power(hold) or None in anchors or not math.isclose(hold, mean(anchors), abs_tol=1e-6):
        raise ValueError("Archived observations do not reproduce the public issue's hold level")
    known = {t.isoformat(): mw for t, mw in sorted(readings.items()) if origin <= t < origin + HOUR}
    partial = (sum(known.values()) + (12 - len(known)) * hold) / 12
    return {"hold": hold, "partial": partial}, known


def record_issue(state, trial, source_file, run_id, clock):
    now = clock()
    origin = now.floor("h")
    target = (origin + HOUR).isoformat()
    if not START <= origin < STOP:
        return "outside-issue-window"
    if target in trial["records"]:
        return "already-recorded"
    candidates = (r for r in state.get("windForecasts", []) if not r.get("diagnosticOnly")
                  and "DK1" in r["areas"] and pd.Timestamp(r["origin"]) == origin)
    base = min(candidates, key=lambda r: pd.Timestamp(r["issuedAt"]), default=None)
    if not base or base["githubRunId"] != run_id:
        return "first-public-issue-unavailable-this-run"
    if base["modelVersion"] != BASELINE:
        return "different-public-model"
    public_at = pd.Timestamp(base["issuedAt"])
    if not origin <= public_at <= now or pd.Timestamp(base["times"][0]) != origin + HOUR:
        raise ValueError("Invalid public first-hour issue")
    if base["sha256"] != canonical_hash({k: v for k, v in base.items() if k != "sha256"}):
        raise ValueError("Public forecast checksum mismatch")
    raw = source_file.read_bytes()
    if hashlib.sha256(raw).hexdigest() != base["sourceHash"]:
        raise ValueError("Public wind source checksum mismatch")
    sources = json.loads(gzip.decompress(raw))
    predictions, known = alternatives(base, sources)
    predictions["public"] = base["areas"]["DK1"]["wind"][0]
    if not all(valid_power(v) for v in predictions.values()):
        raise ValueError("Invalid first-hour forecast")
    issued = clock()
    if not public_at <= issued < origin + HOUR:
        return "missed-target"
    recent = base["recentWind"]["DK1"]
    record = dict(targetEnd=target, issuedAt=issued.isoformat(), publicIssuedAt=base["issuedAt"],
                  publicId=base["id"], publicHash=base["sha256"], sourceHash=base["sourceHash"],
                  retrievedAt=sources["electricityRetrievedAt"], selectedAt=recent["selectedAt"],
                  observationEnd=recent["observationEnd"], observedFiveMinuteMW=known,
                  predictionsMW=predictions, githubRunId=run_id)
    record["sha256"] = canonical_hash(record)
    trial["records"][target] = record
    return "issued"


def freeze_outcomes(trial, hourly, at):
    cutoff = min(at, REVIEW_AT)
    for row in hourly:
        target = row["time"]
        if (row["area"] != "DK1" or target not in trial["records"]
                or target in trial["observations"] or not valid_power(row.get("wind"))):
            continue
        captured = capture_time(row, cutoff, minimum=0.25, maximum=72)
        if captured:
            trial["observations"][target] = {"mw": row["wind"], "retrievedAt": captured}


def evaluate(trial, at):
    records = trial["records"]
    observations = trial["observations"]
    errors, days = [], {}
    for target, actual in sorted(observations.items()):
        pred = records[target]["predictionsMW"]
        row = {m: abs(pred[m] - actual["mw"]) for m in METHODS}
        errors.append(row)
        # Fourteen full 24-hour blocks from enrollment, including the last target in each.
        day = int((pd.Timestamp(target) - HOUR - START) // pd.Timedelta(days=1)) + 1
        days.setdefault(day, []).append(row)

    def scores(rows):
        return {m: mean(r[m] for r in rows) for m in METHODS} if rows else {}

    mae = scores(errors)
    lowest = min(mae.values(), default=None)
    closed = max(0, int((min(at, STOP) - START) // HOUR))
    return dict(status="reviewed" if at >= REVIEW_AT else "collecting" if at < STOP else "awaiting-outcomes",
                checkedAt=at.isoformat(), start=START.isoformat(), stop=STOP.isoformat(), reviewAt=REVIEW_AT.isoformat(),
                scheduledHours=int((STOP - START) // HOUR), closedHours=closed, issuedHours=len(records),
                missedIssueHours=closed - sum(pd.Timestamp(t) <= at for t in records),
                pairedHours=len(errors), unscoredIssuedHours=len(records) - len(errors),
                maeMW=mae, lowestObservedMAE=[m for m in mae if mae[m] == lowest],
                daily=[dict(day=day, startingAt=(START + pd.Timedelta(days=day-1)).isoformat(),
                            pairedHours=len(rows), maeMW=scores(rows)) for day, rows in sorted(days.items())])


def write_summary(report, attempt):
    lines = ["### West Denmark first-hour wind trial", "",
             f"{report['status'].capitalize()} · paired hours **{report['pairedHours']}** · "
             f"issued hours **{report['issuedHours']}** · missed hours **{report['missedIssueHours']}**.",
             f"Refresh result · {attempt.replace('-', ' ')}. Review cutoff {REVIEW_AT:%d %b %Y, %H:%M} UTC.", ""]
    if report["maeMW"]:
        lines += ["| Forecast | Mean absolute error |", "| --- | ---: |"]
        lines += [f"| {label} | {report['maeMW'][m]:.2f} MW |" for m, label in METHODS.items()]
    if report.get("reviewRunId"):
        if report["maeMW"]:
            leaders = ", ".join(METHODS[m] for m in report["lowestObservedMAE"])
            lines += ["", f"Lowest observed error · {leaders}. Scored daily blocks · {len(report['daily'])} of 14.",
                      "", "| Block starting (UTC) | Paired hours | Public MAE (MW) | Hold MAE (MW) | Partial MAE (MW) |",
                      "| --- | ---: | ---: | ---: | ---: |"]
            for day in report["daily"]:
                values = " | ".join(f"{day['maeMW'][m]:.2f}" for m in METHODS)
                lines.append(f"| {pd.Timestamp(day['startingAt']):%d %b %H:%M} | {day['pairedHours']} | {values} |")
        lines += ["", f"[Final review run](https://github.com/HuayuanSong/open-current/actions/runs/{report['reviewRunId']})"]
    print("\n".join(lines), flush=True)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as out:
            out.write("\n".join(lines) + "\n")


def main():
    state = data.read_state()
    trial = state.get(KEY)
    if trial and trial.get("report", {}).get("status") == "reviewed":
        write_summary(trial["report"], "complete")
        return
    fingerprint = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if trial is None:
        trial = state[KEY] = dict(protocolVersion="dk1-first-hour-2026-11-18", baselineVersion=BASELINE,
                                  implementationHash=fingerprint, records={}, observations={})
    if trial["implementationHash"] != fingerprint:
        raise ValueError("First-hour trial definition changed during enrollment")
    for target, record in trial["records"].items():
        if target != record["targetEnd"] or record["sha256"] != canonical_hash({k: v for k, v in record.items() if k != "sha256"}):
            raise ValueError("Issued first-hour trial evidence changed")
    run_id = os.environ.get("GITHUB_RUN_ID")
    attempt, error = "no-new-public-wind-issue", None
    if os.environ.get("PUBLIC_WIND_ISSUED") == "true" and run_id:
        try:
            attempt = record_issue(state, trial, SOURCE, run_id, data.now)
        except (KeyError, ValueError, OSError, TypeError) as exc:
            attempt, error = "invalid-issue-evidence", exc
    at = data.now()
    freeze_outcomes(trial, state["hourly"], at)
    report = trial["report"] = evaluate(trial, at)
    trial["lastAttempt"] = {"at": at.isoformat(), "status": attempt, "error": str(error) if error else None}
    if report["status"] == "reviewed":
        report["reviewRunId"] = run_id
        REVIEW_FILE.parent.mkdir(parents=True, exist_ok=True)
        REVIEW_FILE.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    state["updatedAt"] = at.isoformat()
    data.write_state(state)
    write_summary(report, attempt)
    if report["status"] == "reviewed" and os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as out:
            out.write("reviewed=true\n")
    if error:
        raise error


if __name__ == "__main__":
    main()
