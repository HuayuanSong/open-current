"""Publish small freshness metadata without changing any source timestamps."""

from datetime import datetime, timezone
import json
from pathlib import Path

from fetch import write_json

ROOT = Path(__file__).resolve().parents[1]


def latest_time(rows):
    return max((row["t"] for row in rows if row.get("t") is not None), default=None)


def build_health(index, forecasts, danish, at=None):
    latest = forecasts.get("latest") or {}
    return {
        "schema": 1,
        "preparedAt": (at or datetime.now(timezone.utc)).isoformat(),
        "snapshotAt": index["generatedAt"],
        "forecast": {
            "checkedAt": forecasts.get("checkedAt"),
            "issuedAt": latest.get("issuedAt"),
            "origin": latest.get("origin"),
            "horizonEnd": (latest.get("times") or [None])[-1],
            "status": forecasts.get("status"),
        },
        "sources": {
            "energinet": {
                area: {feed: latest_time(series.get(feed, [])) for feed in ("power", "co2")}
                for area, series in danish.get("areas", {}).items()
            },
            "energyCharts": {
                code: {feed: (country.get(feed) or {}).get("t") for feed in ("power", "flows")}
                for code, country in index.get("countries", {}).items()
            },
        },
        "normalizationIssueCount": len(index.get("normalizationIssues", [])),
    }


def main():
    data = ROOT / "dist/data"
    payloads = [json.loads((data / name).read_text()) for name in (
        "index.json", "forecasts.json", "denmark.json"
    )]
    write_json(data / "health.json", build_health(*payloads))
    print("Published freshness metadata using original observation and issue times.")


if __name__ == "__main__":
    main()
