"""Prospectively compare DK2 day-ahead wind forecasts at the Danish evening cutoff."""

from datetime import date, timedelta
import json
import os

import pandas as pd
import requests

from . import data
from .observation_time import capture_time, valid_wind
from .trial.ledger import canonical_hash

FIRST_DAY = date(2026, 11, 24)
ZONE = "Europe/Copenhagen"
DAY = timedelta(days=1)
HOUR = pd.Timedelta(hours=1)
KEY = "eastWindDayAhead"
MODELS = ("openCurrent", "energinet", "halfAndHalf")
WIND_TYPES = ("Onshore Wind", "Offshore Wind")
SOURCE = "https://api.energidataservice.dk/dataset/Forecasts_Hour"
REPORT_PATH = data.PUBLIC_PATH.with_name("east-wind-day-ahead.json")


def release(day):
    return pd.Timestamp(f"{day - DAY} 18:00", tz=ZONE).tz_convert("UTC")


def hour_ends(day):
    start = pd.Timestamp(day, tz=ZONE)
    stop = pd.Timestamp(day + DAY, tz=ZONE)
    return pd.date_range(start, stop, freq="h", inclusive="right").tz_convert("UTC")


def fetch_energinet(day):
    params = dict(start=str(day), end=str(day + DAY), limit=60,
                  columns="HourUTC,PriceArea,ForecastType,ForecastDayAhead",
                  filter=json.dumps({"PriceArea": ["DK2"], "ForecastType": WIND_TYPES}))
    response = requests.get(SOURCE, params=params, timeout=(3, 8))
    response.raise_for_status()
    return response.json()


def operator_values(payload, ends):
    rows = payload.get("records") if isinstance(payload, dict) else None
    if (not isinstance(rows, list) or payload.get("dataset") != "Forecasts_Hour"
            or payload.get("total") != len(rows) or len(rows) != len(ends) * len(WIND_TYPES)):
        raise ValueError("Waiting for a complete Energinet day-ahead release")
    by_hour = {end: {} for end in ends}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Invalid Energinet forecast row")
        # HourUTC labels the start. Open Current labels the end of the same hour.
        start = pd.to_datetime(row.get("HourUTC"), utc=True, errors="coerce")
        if not isinstance(start, pd.Timestamp) or pd.isna(start):
            raise ValueError("Invalid Energinet delivery hour")
        end = start + HOUR
        kind, value = row.get("ForecastType"), row.get("ForecastDayAhead")
        if (row.get("PriceArea") != "DK2" or end not in by_hour or kind not in WIND_TYPES
                or kind in by_hour[end] or not valid_wind(value)):
            raise ValueError("Incomplete or inconsistent Energinet wind components")
        by_hour[end][kind] = value
    return by_hour


def public_values(records, cutoff, ends):
    candidates = (r for r in records if not r.get("diagnosticOnly") and "DK2" in r["areas"]
                  and pd.Timestamp(r["origin"]) <= pd.Timestamp(r["issuedAt"]) <= cutoff)
    latest = max(candidates, key=lambda r: (pd.Timestamp(r["origin"]), pd.Timestamp(r["issuedAt"])), default=None)
    if latest is None:
        raise ValueError("No issued East Denmark wind forecast at the evening cutoff")
    if latest.get("sha256") != canonical_hash({k: v for k, v in latest.items() if k != "sha256"}):
        raise ValueError("Open Current issue checksum differs from the archive")
    times = pd.to_datetime(latest["times"], utc=True)
    values = latest["areas"]["DK2"]["wind"]
    if len(times) != len(values) or not times.is_unique:
        raise ValueError("Invalid hours in the Open Current issue")
    wind = dict(zip(times, values))
    if not all(valid_wind(wind.get(end)) for end in ends):
        raise ValueError("The Open Current forecast at the cutoff does not cover the delivery day")
    source = {k: latest[k] for k in ("id", "sha256", "issuedAt", "origin", "modelVersion")}
    if latest.get("githubRunId"):
        source["githubRunId"] = latest["githubRunId"]
    return source, wind


def collect(days, records, started, clock, fetch):
    local = started.tz_convert(ZONE)
    day = local.date() + DAY
    if day < FIRST_DAY or started < release(day):
        return False
    if "issue" in days.get(str(day), {}):
        return False
    # A long provider Retry-After can overlap a later evening.
    if any(started < pd.Timestamp(slot["retryAt"]) for slot in days.values() if "retryAt" in slot):
        return False
    retry_seconds = 15 * 60
    try:
        ends = hour_ends(day)
        source, public = public_values(records, release(day), ends)
        operator = operator_values(fetch(day), ends)
        hours = {}
        for end, components in operator.items():
            total = sum(components.values())
            hours[end.isoformat()] = dict(openCurrent=public[end], energinet=total,
                                          halfAndHalf=(public[end] + total) / 2,
                                          onshoreMW=components["Onshore Wind"], offshoreMW=components["Offshore Wind"])
        captured = clock()
        if not started <= captured < ends[0] - HOUR:
            raise ValueError("Energinet capture missed the start of the delivery day")
        issue = dict(cutoff=release(day).isoformat(), capturedAt=captured.isoformat(), openCurrent=source, hours=hours)
        issue["sha256"] = canonical_hash(issue)
        days[str(day)] = dict(issue=issue, observations={})
        return True
    except (ValueError, requests.RequestException) as exc:
        reason = str(exc)
        if isinstance(exc, requests.RequestException):
            reason = "Energinet request failed"
            if exc.response is not None:
                reason += f" (HTTP {exc.response.status_code})"
                retry_seconds = max(retry_seconds, data.retry_delay(exc.response.headers.get("Retry-After")))
        checked = clock()
        days[str(day)] = dict(lastAttemptAt=checked.isoformat(), reason=reason,
                              retryAt=(checked + pd.Timedelta(seconds=retry_seconds)).isoformat())
        return True


def freeze_observations(days, hourly, at):
    changed = False
    for row in hourly:
        if row["area"] != "DK2" or not valid_wind(row.get("wind")):
            continue
        end = pd.Timestamp(row["time"])
        day = (end - HOUR).tz_convert(ZONE).date()
        slot = days.get(str(day))
        if not slot or "issue" not in slot:
            continue
        stamp = end.tz_convert("UTC").isoformat()
        if stamp in slot["observations"] or stamp not in slot["issue"]["hours"]:
            continue
        captured = capture_time(row, at, minimum=0.25)
        if captured:
            slot["observations"][stamp] = dict(mw=row["wind"], retrievedAt=captured)
            changed = True
    return changed


def mean_errors(totals, count):
    return {model: total / count if count else None for model, total in totals.items()}


def report(days, at):
    local = at.tz_convert(ZONE)
    last_day = local.date() + DAY if local.hour >= 18 else local.date()
    daily = []
    totals = dict.fromkeys(MODELS, 0.0)
    paired = issued = complete = missed = 0
    for offset in range((last_day - FIRST_DAY).days + 1):
        day = FIRST_DAY + offset * DAY
        slot = days.get(str(day), {})
        issue = slot.get("issue")
        if not issue:
            closed = day <= local.date()
            missed += int(closed)
            daily.append(dict(date=str(day), status="missed" if closed else "awaiting-capture",
                              reason=slot.get("reason"), lastAttemptAt=slot.get("lastAttemptAt")))
            continue
        if issue["sha256"] != canonical_hash({k: v for k, v in issue.items() if k != "sha256"}):
            raise ValueError(f"Saved day-ahead forecast for {day} failed its checksum")
        issued += 1
        observations = slot["observations"]
        count = len(observations)
        sums = {model: sum(abs(issue["hours"][end][model] - actual["mw"]) for end, actual in observations.items())
                for model in MODELS}
        for model in MODELS:
            totals[model] += sums[model]
        paired += count
        full = count == len(issue["hours"])
        complete += int(full)
        daily.append(dict(date=str(day), status="complete" if full else "scoring" if count else "awaiting-observations",
                          hours=len(issue["hours"]), pairedHours=count, maeMW=mean_errors(sums, count),
                          capturedAt=issue["capturedAt"], cutoff=issue["cutoff"], openCurrent=issue["openCurrent"],
                          sha256=issue["sha256"]))
    return dict(schema=1, area="DK2", status="collecting" if daily else "scheduled",
                timezone=ZONE, checkedAt=at.isoformat(), firstDeliveryDay=str(FIRST_DAY),
                nextCutoff=release(max(FIRST_DAY, last_day + DAY)).isoformat(),
                issuedDays=issued, completeDays=complete, missedDays=missed, pairedHours=paired,
                maeMW=mean_errors(totals, paired), days=daily,
                source="https://www.energidataservice.dk/tso-electricity/Forecasts_Hour",
                evidence=dict(file="forecast-state.json.gz", key=KEY))


def refresh(state_path, report_path, clock=data.now, fetch=fetch_energinet):
    if not state_path.is_file():
        raise FileNotFoundError("Day-ahead comparison requires the restored forecast archive")
    state = data.read_state(state_path)
    initialized = KEY not in state
    days = state.setdefault(KEY, {})
    changed = collect(days, state.get("windForecasts", []), clock(), clock, fetch)
    at = clock()
    observed = freeze_observations(days, state["hourly"], at)
    result = report(days, at)
    if initialized or changed or observed:
        state["updatedAt"] = at.isoformat()
        data.write_state(state, state_path)
    temporary = report_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(report_path)
    return result


if __name__ == "__main__":
    result = refresh(data.STATE_PATH, REPORT_PATH)
    summary = (f"East Denmark day-ahead wind benchmark · {result['issuedDays']} days captured · "
               f"{result['pairedHours']} hours scored · {result['missedDays']} missed days")
    if result["pairedHours"]:
        mae = result["maeMW"]
        summary += (f"\n\nMean absolute error · Open Current {mae['openCurrent']:.1f} MW · "
                    f"Energinet {mae['energinet']:.1f} MW · 50/50 average {mae['halfAndHalf']:.1f} MW")
    print(summary)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as output:
            print(summary, file=output)
