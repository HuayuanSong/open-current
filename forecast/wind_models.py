"""Use the existing refitted components for wind without rescoring other targets."""

import numpy as np

from .features import AREAS, features
from .score import combine
from research.forecasting.weather_retrain import models


def predict_wind(arrays, members, names, folder, selection):
    matrices = {improved: features(arrays, names, 0, improved) for improved in (False, True)}
    if not all(np.isfinite(x).all() for x in matrices.values()):
        raise ValueError("Incomplete wind model inputs")
    changed = dict(members)
    for family in ("tree_control", "ridge", "lightgbm", "xgboost"):
        values = members[family].copy()
        x = matrices[family in ("lightgbm", "xgboost")]
        for ai, area in enumerate(AREAS):
            keep = arrays["area_ids"] == ai
            if keep.any():
                predictions = models.predict_one(models.model_path(folder, family, area, "wind"),
                                                 family, x[np.repeat(keep, 48)])
                values[keep, :, 0] = predictions.reshape(-1, 48)
        changed[family] = values
    result = combine(changed, arrays["area_ids"], selection)[0]
    return result
