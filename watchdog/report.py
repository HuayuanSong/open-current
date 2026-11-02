"""Summarize observed freshness over the most recent 48 hours."""

import argparse
from datetime import datetime, timedelta, timezone
import json
import urllib.request


def summarize(records, now=None):
    now = now or datetime.now(timezone.utc)
    parse = lambda value: datetime.fromisoformat(value.replace("Z", "+00:00"))
    ordered = sorted(records, key=lambda row: row["checkedAt"])
    window = [row for row in ordered if now - timedelta(hours=48) <= parse(row["checkedAt"]) <= now]
    first = parse(ordered[0]["checkedAt"]) if ordered else now
    last = parse(window[-1]["checkedAt"]) if window else None
    checks = [parse(row["checkedAt"]) for row in window]
    gaps = [(b - a).total_seconds() / 60 for a, b in zip(checks, checks[1:])]
    if last:
        gaps.append((now - last).total_seconds() / 60)
    ages = [row["health"]["snapshotAgeMinutes"] for row in window if row.get("health")]
    issues = [row for row in window if row.get("error") or row.get("reason") or not row.get("configured")]
    return {
        "checkedAt": now.isoformat(),
        "monitoringStartedAt": first.isoformat() if ordered else None,
        "hoursSinceFirstRecordedCheck": round((now - first).total_seconds() / 3600, 2),
        "full48HoursElapsed": bool(ordered) and now - first >= timedelta(hours=48),
        "checksInLast48Hours": len(window),
        "latestCheckAt": last.isoformat() if last else None,
        "maximumObservedCheckGapMinutes": round(max(gaps), 1) if gaps else None,
        "maximumObservedSnapshotAgeMinutes": max(ages) if ages else None,
        "recoveryDispatches": sum(row.get("action") == "recovery_dispatched" for row in window),
        "checksNeedingAttention": len(issues),
        "note": "Observed at polling times; elapsed time alone does not establish reliability. Inspect gaps and source/forecast failures in /history.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="Watchdog workers.dev URL")
    args = parser.parse_args()
    request = urllib.request.Request(args.url.rstrip("/") + "/history", headers={"Cache-Control": "no-cache"})
    with urllib.request.urlopen(request, timeout=20) as response:
        records = json.load(response)
    print(json.dumps(summarize(records), indent=2))


if __name__ == "__main__":
    main()
