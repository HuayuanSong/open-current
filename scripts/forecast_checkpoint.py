"""Recover the forecast archive from the last successful main Pages workflow."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
from urllib.request import Request, urlopen

FILES = ("forecast-state.json.gz", "forecasts.json")


def github(*args):
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True, timeout=90).stdout


def download(repo, destination):
    runs = json.loads(github("run", "list", "--repo", repo, "--workflow", "pages.yml",
                             "--branch", "main", "--status", "success", "--limit", "1", "--json", "databaseId"))
    if not runs:
        raise ValueError("No successful main publication to restore")
    run = str(runs[0]["databaseId"])
    artifacts = json.loads(github("api", f"repos/{repo}/actions/runs/{run}/artifacts?per_page=100"))["artifacts"]
    available = {a["name"] for a in artifacts if not a["expired"]}
    destination.mkdir(parents=True, exist_ok=True)
    if "forecast-checkpoint" in available:
        github("run", "download", run, "--repo", repo, "--name", "forecast-checkpoint", "--dir", str(destination))
    elif "github-pages" in available:
        # Also bootstraps the first release from the previous, immutable Pages artifact.
        with tempfile.TemporaryDirectory() as tmp:
            github("run", "download", run, "--repo", repo, "--name", "github-pages", "--dir", tmp)
            with tarfile.open(Path(tmp) / "artifact.tar") as archive:
                members = {m.name.removeprefix("./"): m for m in archive.getmembers() if m.isfile()}
                for name in FILES:
                    member = members[f"data/{name}"]
                    if not 0 < member.size <= 64 * 1024 * 1024:
                        raise ValueError(f"Invalid checkpoint size for {name}")
                    (destination / name).write_bytes(archive.extractfile(member).read())
    else:
        raise ValueError(f"Recovery artifacts expired for the last successful Pages run {run}")
    print(f"Loaded independent forecast checkpoint from successful Pages run {run}", flush=True)


def verify_live(base, checkpoint):
    for name in FILES:
        raw = (checkpoint / name).read_bytes()
        expected = sha256(raw).digest()
        for suffix in ("", f"?checkpoint={expected.hex()}"):
            req = Request(f"{base.rstrip('/')}/data/{name}{suffix}", headers={"Cache-Control": "no-cache"})
            with urlopen(req, timeout=40) as response:
                actual = response.read(len(raw) + 1)
            if len(actual) == len(raw) and sha256(actual).digest() == expected:
                break
        else:
            raise ValueError(f"Published {name} differs from the independent checkpoint; refusing archive rollback")


def restore(repo, base, target):
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = Path(tmp)
        download(repo, checkpoint)
        verify_live(base, checkpoint)
        # Both files must match before writing either. The build workspace is disposable.
        for name in FILES:
            temporary = target / (name + ".tmp")
            temporary.write_bytes((checkpoint / name).read_bytes())
            temporary.replace(target / name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--target", default="dist/data", type=Path)
    args = parser.parse_args()
    restore(args.repo, args.url, args.target)
