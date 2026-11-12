"""Refitted CPU components with the production feature order and blend contract."""

import json

import numpy as np

from forecast.features import AREAS, TARGETS, features
from .common import PROTOCOL


def setting_for(family, area, target):
    return next(row["setting"] for row in PROTOCOL["settings"][family]
                if row["area"] == area and row["target"] == target)


def create(family, setting):
    if family == "ridge":
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), Ridge(alpha=[1, 100, 10000][setting]))
    large = setting % 2 == 1
    common = dict(n_estimators=800 if large else 400, learning_rate=.04, reg_lambda=10,
                  n_jobs=PROTOCOL["threads"], random_state=PROTOCOL["seed"])
    if family in ("tree_control", "lightgbm"):
        import lightgbm as lgb
        return lgb.LGBMRegressor(**common, num_leaves=31 if large else 15,
            min_child_samples=60 if large else 100, objective="regression_l1",
            verbosity=-1, deterministic=True, force_col_wise=True)
    import xgboost as xgb
    return xgb.XGBRegressor(**common, max_depth=5 if large else 4,
        min_child_weight=60 if large else 100, objective="reg:absoluteerror",
        tree_method="hist", device="cpu")


def model_path(folder, family, area, target):
    extension = {"ridge": "json", "xgboost": "ubj"}.get(family, "txt")
    return folder / f"{family}_{area}_{target}.{extension}"


def save(model, family, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if family == "ridge":
        scaler, ridge = model.steps[0][1], model.steps[1][1]
        path.write_text(json.dumps(dict(mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
            coef=ridge.coef_.tolist(), intercept=float(ridge.intercept_)), allow_nan=False))
    elif family == "xgboost":
        model.save_model(path)
    else:
        model.booster_.save_model(str(path))


def predict_one(path, family, x):
    if family == "ridge":
        from forecast.worker import ridge
        prediction = ridge(path, x)
    elif family == "xgboost":
        import xgboost as xgb
        model = xgb.XGBRegressor(n_jobs=PROTOCOL["threads"])
        model.load_model(path)
        prediction = model.predict(x)
    else:
        import lightgbm as lgb
        model = lgb.Booster(model_file=str(path))
        prediction = model.predict(x, num_threads=PROTOCOL["threads"])
    return np.maximum(0, prediction)


def predict(arrays, names, folder):
    count = len(arrays["area_ids"])
    outputs = {f: np.full((count, 48, 3), np.nan, np.float32) for f in PROTOCOL["families"]}
    for ti, target in enumerate(TARGETS):
        matrices = {improved: features(arrays, names, ti, improved) for improved in (False, True)}
        for family in PROTOCOL["families"]:
            x = matrices[family in ("lightgbm", "xgboost")]
            if not np.isfinite(x).all():
                raise ValueError("Incomplete refit model inputs")
            for ai, area in enumerate(AREAS):
                keep = arrays["area_ids"] == ai
                if keep.any():
                    output = predict_one(model_path(folder, family, area, target), family, x[np.repeat(keep, 48)])
                    outputs[family][keep, :, ti] = output.reshape(-1, 48)
    if not all(np.isfinite(x).all() for x in outputs.values()):
        raise ValueError("Incomplete refitted predictions")
    return outputs
