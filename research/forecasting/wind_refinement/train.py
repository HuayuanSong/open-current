"""Choose short-term models on August/September, then freeze and refit pre-November."""

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from forecast.wind_short_term import predict
from research.forecasting.weather_retrain.common import digest, write_json
from .prepare import CACHE, HERE


def fit(kind, setting, x, y):
    if kind == "ridge":
        scale = StandardScaler().fit(x)
        model = Ridge(alpha=setting).fit(scale.transform(x), y)
        return dict(mean=scale.mean_.tolist(), scale=scale.scale_.tolist(),
                    coef=model.coef_.tolist(), intercept=float(model.intercept_))
    model = lgb.LGBMRegressor(objective="regression_l1", num_leaves=setting,
        max_depth=4, n_estimators=250, learning_rate=.04, min_child_samples=80,
        reg_lambda=10, random_state=42, n_jobs=2, verbosity=-1, deterministic=True,
        force_col_wise=True).fit(x, y)
    return model.booster_


def main():
    protocol = json.loads((HERE / "protocol.json").read_text())
    with np.load(CACHE / "dataset.npz") as saved:
        x, y, level = (saved[k] for k in ("x", "y", "level"))
    samples = pd.read_csv(CACHE / "samples.csv")
    origins = pd.to_datetime(samples.origin, utc=True)
    last_label = origins + pd.Timedelta(hours=6)
    fit_mask = last_label < pd.Timestamp(protocol["developmentFitLabelsBefore"])
    select = (origins >= pd.Timestamp(protocol["selectionStart"])) & (last_label < pd.Timestamp(protocol["selectionEndExclusive"]))
    final = last_label < pd.Timestamp(protocol["finalFitLabelsBefore"])
    target = y - level[:, None]
    selected, scores = {}, []
    models = HERE / "models"
    models.mkdir(exist_ok=True)
    for area in ("DK1", "DK2"):
        region = samples.area.eq(area)
        train, valid, full = (region & mask for mask in (fit_mask, select, final))
        options = []
        for kind, settings in (("ridge", protocol["ridgeAlphas"]), ("lightgbm", protocol["lightgbmLeaves"])):
            for setting in settings:
                model = fit(kind, setting, x[train].reshape(-1, x.shape[-1]), target[train].ravel())
                pred = predict(model, x[valid].reshape(-1, x.shape[-1]), np.repeat(level[valid], 6)).reshape(-1, 6)
                error = float(np.abs(pred - y[valid]).mean())
                row = dict(area=area, family=kind, setting=setting, maeMW=error,
                           fittingOrigins=int(train.sum()), validationOrigins=int(valid.sum()))
                scores.append(row); options.append((error, kind, setting))
                print(json.dumps(row), flush=True)
        _, kind, setting = min(options)
        model = fit(kind, setting, x[full].reshape(-1, x.shape[-1]), target[full].ravel())
        if kind == "ridge":
            path = models / f"{area}.json"
            write_json(path, model)
            loaded = json.loads(path.read_text())
        else:
            path = models / f"{area}.txt"
            model.save_model(path)
            loaded = lgb.Booster(model_file=str(path))
        check = x[valid].reshape(-1, x.shape[-1])
        np.testing.assert_allclose(predict(loaded, check, 0), predict(model, check, 0), rtol=0, atol=1e-7)
        selected[area] = dict(family=kind, setting=setting, file=path.name, sha256=digest(path),
            fittingOrigins=int(full.sum()), maximumLabel=(last_label[full].max()).isoformat())
    result = dict(version=protocol["version"], protocolHash=digest(HERE / "protocol.json"),
                  featureNames=json.loads((CACHE / "feature-names.json").read_text()),
                  datasetHash=digest(CACHE / "dataset.npz"), models=selected, developmentScores=scores,
                  publicPromotion=False)
    write_json(HERE / "selection.json", result)


if __name__ == "__main__":
    main()
