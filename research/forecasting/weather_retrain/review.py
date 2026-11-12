"""Review immutable evidence; no training, retroactive issues or promotion."""

import argparse
import io
import json
import subprocess
import zipfile

import numpy as np
import pandas as pd

from forecast.features import AREAS, TARGETS
from forecast.trial.ledger import canonical_hash
from .common import CACHE, HERE, PROTOCOL, boundary, digest, write_json
from .prospective import eligible_origin


def download():
    pages = subprocess.check_output(["gh", "api", "--paginate", "--slurp",
        "repos/HuayuanSong/open-current/actions/artifacts?per_page=100"], text=True)
    for page in json.loads(pages):
        for artifact in page["artifacts"]:
            if not artifact["name"].startswith("weather-refit-evidence-") or artifact["expired"]:
                continue
            path = CACHE / "archive" / str(artifact["id"])
            if (path / ".complete").exists():
                continue
            content = subprocess.check_output(["gh", "api", artifact["archive_download_url"]])
            path.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                for name in archive.namelist():
                    destination = (path / name).resolve()
                    if not destination.is_relative_to(path.resolve()):
                        raise ValueError("Unsafe evidence archive path")
                archive.extractall(path)
            (path / ".complete").touch()


def validate_record(record):
    if record.get("diagnosticOnly"):
        raise ValueError("A diagnostic must never enter the prospective trial")
    expected = canonical_hash({k: v for k, v in record.items() if k != "sha256"})
    if record["sha256"] != expected:
        raise ValueError("Issued evidence was modified")
    for key, path in [("protocolHash", HERE / "protocol.json"), ("modelManifestHash", HERE / "model-manifest.json"),
                      ("inferenceLockHash", HERE / "inference-lock.json")]:
        if record[key] != digest(path):
            raise ValueError(f"Prospective fingerprint mismatch: {key}")
    origin = eligible_origin(record["issuedAt"])
    if origin is None or origin != pd.Timestamp(record["origin"]):
        raise ValueError("A late or backfilled forecast cannot be evaluated")
    times = pd.date_range(origin + pd.Timedelta(hours=1), periods=48, freq="h")
    if list(pd.to_datetime(record["times"], utc=True)) != list(times):
        raise ValueError("Unexpected prospective horizon")
    for variant in ("baseline", "daily", "latest"):
        for area in AREAS:
            for target in TARGETS:
                values = record["predictions"][variant][area][target]
                if len(values) != 48 or not np.isfinite(values).all() or min(values) < 0:
                    raise ValueError("Incomplete prospective forecast")


def first_observations(snapshots, at):
    frozen = {}
    for item in sorted(snapshots, key=lambda r: r["capturedAt"]):
        captured, valid = pd.Timestamp(item["capturedAt"]), pd.Timestamp(item["time"])
        age = (captured - valid).total_seconds() / 3600
        if captured > at or not 24 <= age <= 72:
            continue
        if item["area"] not in AREAS or item["target"] not in TARGETS:
            raise ValueError("Unexpected observation series")
        value = item["value"]
        if not np.isfinite(value) or value < 0 or (item["target"] == "load" and value == 0):
            continue
        key = (valid.isoformat(), item["area"], item["target"])
        frozen.setdefault(key, item)
    return frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.download:
        download()
    records, snapshots, attempted = [], [], []
    for path in sorted((CACHE / "archive").rglob("attempt.json")):
        attempt = json.loads(path.read_text())
        if attempt.get("diagnosticOnly"):
            continue
        if attempt["protocolHash"] != digest(HERE / "protocol.json"):
            raise ValueError("Evidence belongs to another protocol")
        attempted.append(attempt)
        observations = path.with_name("observations.json")
        if observations.exists():
            if attempt.get("observationsHash") != digest(observations):
                raise ValueError("Observation snapshot changed")
            snapshots.extend(json.loads(observations.read_text()))
        source = path.with_name("record.json")
        if source.exists():
            record = json.loads(source.read_text())
            validate_record(record)
            if digest(path.with_name("inputs.npz")) != record["inputHash"] or digest(path.with_name("weather-source.json.gz")) != record["sourceHash"]:
                raise ValueError("Issued input evidence changed")
            records.append(record)
    # Scheduled retries are duplicates, not extra independent origins. The
    # first valid issue wins irrespective of later prediction errors.
    unique = {}
    for record in sorted(records, key=lambda r: r["issuedAt"]):
        unique.setdefault(record["origin"], record)
    at = pd.Timestamp.now(tz="UTC")
    observed = first_observations(snapshots, at)
    rows, complete = [], []
    for record in unique.values():
        keys = [(pd.Timestamp(t).isoformat(), area, target) for t in record["times"] for area in AREAS for target in TARGETS]
        if not all(key in observed for key in keys):
            continue
        complete.append(record["origin"])
        for variant in ("baseline", "daily", "latest"):
            for area in AREAS:
                for target in TARGETS:
                    for lead, t in enumerate(record["times"]):
                        rows.append(dict(origin=record["origin"], area=area, target=target, variant=variant, lead=lead + 1,
                            prediction=record["predictions"][variant][area][target][lead],
                            actual=observed[(pd.Timestamp(t).isoformat(), area, target)]["value"]))
    planned = [day + pd.Timedelta(hours=hour) for day in pd.date_range(boundary("prospectiveStart"),
               boundary("prospectiveEndExclusive") - pd.Timedelta(days=1)) for hour in PROTOCOL["prospectiveHours"]]
    review_after = planned[-1] + pd.Timedelta(hours=48 + 72)
    report = dict(checkedAt=at.isoformat(), reviewAfter=review_after.isoformat(), automaticPromotion=False,
        plannedOrigins=len(planned), issuedOrigins=len(unique), completeOrigins=len(complete),
        distinctDays=len({t[:10] for t in complete}), issueCoverage=len(unique) / len(planned),
        attempts=len(attempted), status="collecting" if at < review_after else "insufficient-data", metrics=[])
    if rows:
        frame = pd.DataFrame(rows)
        frame["absoluteError"] = np.abs(frame.prediction - frame.actual)
        metrics = frame.groupby(["area", "target", "variant"]).absoluteError.mean().unstack("variant")
        frame["band"] = pd.cut(frame.lead, [0, 6, 24, 48], labels=["1-6", "7-24", "25-48"])
        bands = frame.groupby(["area", "target", "variant", "band"], observed=True).absoluteError.mean().unstack("variant")
        reduction = 100 * (1 - metrics.latest / metrics.baseline)
        regression = 100 * (bands.latest / bands.baseline - 1)
        report["metrics"] = metrics.reset_index().to_dict("records")
        enough = (len(complete) >= PROTOCOL["minimumCompleteOrigins"] and report["distinctDays"] >= PROTOCOL["minimumDistinctDays"]
                  and report["issueCoverage"] >= PROTOCOL["minimumIssueCoverage"])
        passes = (enough and np.isfinite(reduction).all() and np.isfinite(regression).all()
                  and reduction.loc[("DK1", "wind")] >= PROTOCOL["minimumDK1WindImprovementPct"]
                  and reduction.min() >= -PROTOCOL["maximumSeriesRegressionPct"]
                  and regression.max() <= PROTOCOL["maximumLeadBandRegressionPct"])
        if at >= review_after and enough:
            report["status"] = "eligible-for-further-review" if passes else "retain-baseline"
        frame.to_csv(CACHE / "prospective-predictions.csv", index=False)
        bands.reset_index().to_csv(CACHE / "prospective-lead-metrics.csv", index=False)
    write_json(CACHE / "prospective-review.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
