"""Paths and immutable experiment metadata; no production state writes."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CACHE = ROOT / ".cache/weather-retrain"
PROTOCOL = json.loads((HERE / "protocol.json").read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def boundary(name):
    import pandas as pd
    return pd.Timestamp(PROTOCOL[name], tz="UTC")
