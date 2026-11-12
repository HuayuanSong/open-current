"""Package the research candidate without changing production model pointers."""

import json
import tarfile

import numpy as np

from .common import CACHE, HERE, ROOT, PROTOCOL, digest, write_json
from .compare import verify_models
from .models import predict


def main():
    manifest = verify_models()
    sources = [
        "forecast/data.py", "forecast/features.py", "forecast/weather_runs.py", "forecast/score.py",
        "forecast/worker.py", "forecast/weather_mlp.py", "forecast/artifacts.py", "forecast/config.json",
        "forecast/model-release.json", "forecast/requirements-main.txt", "forecast/requirements-ttm.txt",
        "forecast/trial/ledger.py",
    ]
    sources += [str((HERE / name).relative_to(ROOT)) for name in
                ("common.py", "models.py", "compare.py", "prospective.py", "verify.py", "protocol.json", "PROTOCOL.md")]
    lock = {name: digest(ROOT / name) for name in sources}
    lock_path = HERE / "inference-lock.json"
    if lock_path.exists() and json.loads(lock_path.read_text()) != lock:
        raise ValueError("Previously frozen prospective inference changed; do not rewrite its lock")
    write_json(lock_path, lock)
    name = PROTOCOL["version"].replace("retrain", "refit")
    path = CACHE / f"{name}.tar.gz"
    release_path = HERE / "model-release.json"
    if release_path.exists():
        release = json.loads(release_path.read_text())
        if (release["modelManifestHash"] != digest(HERE / "model-manifest.json")
                or release["inferenceLockHash"] != digest(lock_path) or not path.exists()
                or release["sha256"] != digest(path)):
            raise ValueError("An existing candidate release differs; do not overwrite it")
        print(json.dumps(release, indent=2))
        return
    arrays = dict(np.load(CACHE / "dataset.npz"))
    fixture = {key: arrays[key][-2:] for key in ("history", "history512", "observed512", "past_full512", "future", "area_ids")}
    np.testing.assert_array_equal(fixture["area_ids"], [0, 1])
    fresh = arrays["fresh"][-2:]
    np.savez_compressed(CACHE / "models/fixture.npz", **fixture, latest_future=fresh)
    expected = {}
    for policy in ("daily", "latest"):
        inputs = fixture if policy == "daily" else {**fixture, "future": fresh}
        expected.update({f"{policy}__{family}": values for family, values in
                         predict(inputs, manifest["featureNames"], CACHE / "models" / policy).items()})
    np.savez_compressed(CACHE / "models/expected.npz", **expected)
    fixtures = {name: digest(CACHE / "models" / name) for name in ("fixture.npz", "expected.npz")}
    with tarfile.open(path, "w:gz") as archive:
        for relative in ["manifest.json", *manifest["models"], *fixtures]:
            archive.add(CACHE / "models" / relative, arcname=relative)
    release = dict(version=name, url=f"https://github.com/HuayuanSong/open-current/releases/download/{name}/{path.name}",
                   sha256=digest(path), bytes=path.stat().st_size,
                   fixtures=fixtures,
                   modelManifestHash=digest(HERE / "model-manifest.json"), inferenceLockHash=digest(lock_path))
    write_json(release_path, release)
    print(json.dumps(release, indent=2))


if __name__ == "__main__":
    main()
