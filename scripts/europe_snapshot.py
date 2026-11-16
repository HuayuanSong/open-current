"""Transfer European raw feeds to the sole publisher without sharing forecast state.

The slow collector only uploads an artifact. The publisher restores the live
site first, accepts a newer, complete/checksummed bundle, then normalizes it
alongside fresh Danish readings. Failed collection retains published history.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from config import COUNTRIES, PRICE_ZONES
from fetch import CACHE, ROOT, write_json

ALLOWED = {f"{endpoint}/{key}.json" for endpoint, keys in (
    ("public_power", COUNTRIES), ("cbpf", COUNTRIES), ("price", PRICE_ZONES)
) for key in keys}


def timestamp(value):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("Collection time must include its timezone")
    return parsed


def package(destination, cache=CACHE):
    status = json.loads((cache / "europe-fetch-status.json").read_text())
    files = {}
    for name in sorted(ALLOWED):
        path = cache / name
        if path.exists():
            raw = path.read_bytes()
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            files[name] = hashlib.sha256(raw).hexdigest()
    if not files:
        raise ValueError("No European feeds collected")
    manifest = dict(schema=1, collectedAt=status["finished_at"], files=files,
                    failures=status["failures"], githubRunId=os.environ.get("GITHUB_RUN_ID"))
    write_json(destination / "manifest.json", manifest)


def install(bundle, cache=CACHE, index_path=ROOT / "dist/data/index.json", at=None):
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest.get("schema") != 1 or not manifest.get("files"):
        raise ValueError("Invalid European collection manifest")
    collected = timestamp(manifest["collectedAt"])
    if collected > (at or datetime.now(timezone.utc)):
        raise ValueError("European collection is dated in the future")
    previous = json.loads(index_path.read_text()).get("europeCollection") or {}
    if previous.get("collectedAt") and timestamp(previous["collectedAt"]) >= collected:
        return False
    # Validate everything before writing any inputs. Only allow known raw feeds;
    # no artifact is allowed to supply executable code or a forecast archive.
    payloads = {}
    for name, checksum in manifest["files"].items():
        if name not in ALLOWED:
            raise ValueError("Unexpected European collection file")
        raw = (bundle / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != checksum:
            raise ValueError("European collection checksum mismatch")
        payload = json.loads(raw)
        if not isinstance(payload.get("data"), list) or "CC BY 4.0" not in payload.get("license", ""):
            raise ValueError("Invalid or unlicensed European response")
        if timestamp(payload["_retrieved_at"]) > collected:
            raise ValueError("Response retrieved after collection finished")
        payloads[name] = payload
    for name, payload in payloads.items():
        write_json(cache / name, payload)
    write_json(cache / "europe-collection.json", {k: v for k, v in manifest.items() if k != "files"})
    return True


def download(destination):
    repo = os.environ["GITHUB_REPOSITORY"]
    result = subprocess.run([
        "gh", "run", "list", "--repo", repo, "--workflow", "collect-europe.yml",
        "--branch", "main", "--status", "success", "--limit", "1", "--json", "databaseId",
    ], check=True, capture_output=True, text=True, timeout=30)
    runs = json.loads(result.stdout)
    if not runs:
        print("No completed European collection yet; retaining published readings.")
        return False
    subprocess.run([
        "gh", "run", "download", str(runs[0]["databaseId"]), "--repo", repo,
        "--name", "europe-feeds", "--dir", str(destination),
    ], check=True, capture_output=True, timeout=45)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("package", "import"))
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args()
    if args.operation == "package":
        package(args.bundle or CACHE / "europe-snapshot")
    elif args.bundle:
        print("Imported newer European collection:", install(args.bundle))
    else:
        with tempfile.TemporaryDirectory() as folder:
            bundle = Path(folder)
            if download(bundle):
                print("Imported newer European collection:", install(bundle))


if __name__ == "__main__":
    main()
