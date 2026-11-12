"""Fit fixed settings, purge overlapping labels, and verify portable exports."""

import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "4"

import importlib.metadata
import json
import time

import numpy as np
import pandas as pd

from forecast.features import AREAS, TARGETS, features
from .common import CACHE, HERE, PROTOCOL, boundary, digest, write_json
from .models import create, model_path, predict_one, save, setting_for


def masks(meta, cutoff, end=None):
    train = (meta.last_label < cutoff).to_numpy(copy=True)
    valid = (meta.origin >= cutoff).to_numpy(copy=True)
    if end is not None:
        valid &= (meta.last_label < end).to_numpy()
    if train.any() and valid.any():
        assert meta.loc[train, "last_label"].max() < meta.loc[valid, "origin"].min()
    return train, valid


def main():
    arrays = dict(np.load(CACHE / "dataset.npz"))
    meta = pd.read_csv(CACHE / "samples.csv")
    for key in ("origin", "last_label"):
        meta[key] = pd.to_datetime(meta[key], utc=True)
    names = json.loads((CACHE / "feature-names.json").read_text())
    audit = json.loads((HERE / "data-audit.json").read_text())
    assert audit["datasetHash"] == digest(CACHE / "dataset.npz")
    assert audit["protocolHash"] == digest(HERE / "protocol.json")
    assert meta.last_label.max() < boundary("finalFitLabelsBefore")
    fingerprint = dict(datasetHash=audit["datasetHash"], protocolHash=digest(HERE / "protocol.json"),
        dataAuditHash=digest(HERE / "data-audit.json"), acquisitionNoteHash=digest(HERE / "ACQUISITION.md"),
        sourceHashes={name: digest(HERE / name) for name in ("models.py", "train.py")},
        featureSourceHash=digest(HERE.parents[2] / "forecast/features.py"),
        packages={name: importlib.metadata.version(name) for name in ("numpy", "pandas", "lightgbm", "xgboost", "scikit-learn")})
    completed = CACHE / "models/manifest.json"
    if completed.exists():
        frozen = json.loads(completed.read_text())
        if any(frozen.get(key) != value for key, value in fingerprint.items()):
            raise ValueError("A completed refit is already frozen; use a new experiment for changed data or code")
    timings, development = [], []
    for policy in ("daily", "latest"):
        inputs = {**arrays, "future": arrays["fresh" if policy == "latest" else "future"]}
        for ti, target in enumerate(TARGETS):
            matrices = {improved: features(inputs, names, ti, improved) for improved in (False, True)}
            if not all(np.isfinite(x).all() for x in matrices.values()):
                raise ValueError("Non-finite training features")
            for family in PROTOCOL["families"]:
                x = matrices[family in ("lightgbm", "xgboost")]
                for ai, area in enumerate(AREAS):
                    setting = setting_for(family, area, target)
                    for stage, cutoff in [("development", boundary("developmentFitLabelsBefore")),
                                          ("final", boundary("finalFitLabelsBefore"))]:
                        train, valid = masks(meta, cutoff, boundary("developmentEndExclusive") if stage == "development" else None)
                        train &= arrays["area_ids"] == ai
                        valid &= arrays["area_ids"] == ai
                        if train.sum() < 50:
                            raise ValueError("Too few training origins")
                        folder = CACHE / ("models" if stage == "final" else "development-models") / policy
                        path = model_path(folder, family, area, target)
                        stamp = path.with_suffix(path.suffix + ".fit.json")
                        expected = {**fingerprint, "stage": stage, "policy": policy, "family": family,
                                    "area": area, "target": target, "setting": setting}
                        reused = False
                        if path.exists() and stamp.exists():
                            saved = json.loads(stamp.read_text())
                            reused = saved["fingerprint"] == expected and saved["modelHash"] == digest(path)
                        if reused:
                            seconds = saved["trainSeconds"]
                        else:
                            model = create(family, setting)
                            kwargs = {}
                            if family in ("lightgbm", "xgboost") and setting >= 2:
                                days = (cutoff - meta.origin).dt.total_seconds().to_numpy() / 86400
                                kwargs["sample_weight"] = np.repeat(np.power(.5, days[train] / 180), 48)
                            train_x = x[np.repeat(train, 48)]
                            train_y = arrays["y"][train, :, ti].ravel()
                            begin = time.perf_counter()
                            model.fit(train_x, train_y, **kwargs)
                            seconds = time.perf_counter() - begin
                            save(model, family, path)
                            fixture = train_x[-96:]
                            np.testing.assert_allclose(predict_one(path, family, fixture),
                                np.maximum(0, model.predict(fixture)), rtol=2e-5, atol=.02)
                            write_json(stamp, dict(fingerprint=expected, modelHash=digest(path), trainSeconds=seconds))
                        timings.append(dict(policy=policy, family=family, area=area, target=target, stage=stage,
                                            trainOrigins=int(train.sum()), seconds=seconds, reused=reused))
                        if stage == "development":
                            output = predict_one(path, family, x[np.repeat(valid, 48)]).reshape(-1, 48)
                            truth = arrays["y"][valid, :, ti]
                            error = np.abs(output - truth)
                            development.append(dict(policy=policy, family=family, area=area, target=target,
                                origins=int(valid.sum()), maeMW=float(error.mean()), wapePct=float(error.sum() / truth.sum() * 100)))
                        print(f"{stage} {policy} {family} {area}/{target}: {int(train.sum())} origins, {seconds:.1f}s" + (" (cached)" if reused else ""), flush=True)
    pd.DataFrame(development).to_csv(HERE / "development-metrics.csv", index=False)
    write_json(HERE / "training-runtime.json", timings)
    paths = [model_path(CACHE / "models" / policy, family, area, target)
             for policy in ("daily", "latest") for family in PROTOCOL["families"]
             for area in AREAS for target in TARGETS]
    manifest = {**fingerprint, "baselineVersion": PROTOCOL["baselineVersion"],
                "featureNames": names, "models": {str(p.relative_to(CACHE / "models")): digest(p) for p in paths}}
    write_json(CACHE / "models/manifest.json", manifest)
    write_json(HERE / "model-manifest.json", manifest)
    print(f"Frozen {len(paths)} exported models", flush=True)


if __name__ == "__main__":
    main()
