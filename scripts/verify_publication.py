"""Verify that Pages serves the exact data files uploaded by this workflow run."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from http.client import HTTPException
import json
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen


FILES = ("data/health.json", "data/index.json", "data/forecasts.json", "data/denmark.json")


def create_manifest(site):
    files = {}
    for name in FILES:
        content = (site / name).read_bytes()
        files[name] = {"sha256": hashlib.sha256(content).hexdigest(), "size": len(content)}
    health = json.loads((site / "data/health.json").read_bytes())
    return {
        "schema": 1,
        "snapshotAt": health["snapshotAt"],
        "forecastIssuedAt": health["forecast"].get("issuedAt"),
        "files": files,
    }


def load_manifest(path):
    manifest = json.loads(path.read_text())
    if not isinstance(manifest, dict) or manifest.get("schema") != 1:
        raise ValueError("Invalid publication manifest schema")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(FILES):
        raise ValueError("Publication manifest must include all four data files")
    for details in files.values():
        if (
            not isinstance(details, dict)
            or not isinstance(details.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", details["sha256"])
            or type(details.get("size")) is not int
            or details["size"] < 1
        ):
            raise ValueError("Invalid file checksum or size in publication manifest")
    return manifest


def check_file(base_url, name, expected):
    # Check the canonical URLs visitors use, not a unique URL that could hide a
    # stale cache entry. No GitHub credentials are sent to the public website.
    request = Request(base_url.rstrip("/") + "/" + name, headers={
        "User-Agent": "OpenCurrent-Publication-Check/1.0",
        "Cache-Control": "no-cache",
    })
    try:
        with urlopen(request, timeout=10) as response:
            # A larger body already cannot match; bound memory and transfer size.
            content = response.read(expected["size"] + 1)
        if len(content) != expected["size"]:
            return f"{name}: public size differs from uploaded file"
        if hashlib.sha256(content).hexdigest() != expected["sha256"]:
            return f"{name}: public checksum differs from uploaded file"
    except (OSError, HTTPException) as error:
        return f"{name}: {error}"
    return None


def check_once(manifest, base_url):
    with ThreadPoolExecutor(max_workers=len(FILES)) as pool:
        results = list(pool.map(
            lambda name: check_file(base_url, name, manifest["files"][name]), FILES,
        ))
    return [error for error in results if error]


def verify(manifest, base_url, attempts=7, interval=30):
    if attempts < 1 or interval < 0:
        raise ValueError("Attempts must be positive and interval must be nonnegative")
    print(
        f"Expecting snapshot {manifest.get('snapshotAt')} and forecast "
        f"{manifest.get('forecastIssuedAt')}", flush=True,
    )
    for attempt in range(1, attempts + 1):
        errors = check_once(manifest, base_url)
        if not errors:
            print("Verified: all four public data files match the uploaded snapshot.", flush=True)
            return True
        print(f"Publication check {attempt}/{attempts}: " + "; ".join(errors), flush=True)
        if attempt < attempts:
            time.sleep(interval)
    print("Publication remains unverified; the public site may still serve older data.", flush=True)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser("manifest", help="Record the files about to be uploaded")
    manifest.add_argument("--site-dir", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("check", help="Verify public files against the build manifest")
    check.add_argument("--manifest", type=Path, required=True)
    check.add_argument("--url", required=True)
    check.add_argument("--attempts", type=int, default=7)
    check.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()
    if args.command == "manifest":
        args.output.write_text(json.dumps(create_manifest(args.site_dir), indent=2) + "\n")
    else:
        raise SystemExit(0 if verify(
            load_manifest(args.manifest), args.url, args.attempts, args.interval,
        ) else 1)


if __name__ == "__main__":
    main()
