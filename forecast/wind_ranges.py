"""Research-only conditional wind ranges; no probability guarantee under drift."""

import numpy as np

BANDS = ((0, 6), (6, 24), (24, 48))


def scale(method, prediction, area, *, disagreement=None, weather=None):
    prediction = np.asarray(prediction, float)
    floor = 150 if area == "DK1" else 50
    if method in ("absolute", "signed"):
        return np.ones_like(prediction)
    power = np.maximum(floor, .15 * prediction)
    if method == "power":
        return power
    if method == "weather_disagreement":
        if disagreement is None:
            raise ValueError("Weather disagreement is required")
        return np.maximum(power, np.abs(disagreement))
    if method == "ensemble":
        if weather is None:
            raise ValueError("An issued ensemble snapshot is required")
        mean = np.asarray(weather["meanWindMS"], float)
        spread = np.asarray(weather["meanLocalSpreadMS"], float)
        if mean.shape != prediction.shape or spread.shape != prediction.shape:
            raise ValueError("Ensemble wind horizon differs from power forecast")
        # Cubic wind-to-power sensitivity is a rough scale feature only; its
        # relationship to power errors must be learned and checked on outcomes.
        return np.maximum(power, 3 * prediction * spread / np.maximum(mean, 3))
    raise ValueError("Unknown wind range method")


def fit(actual, prediction, scales, method):
    actual, prediction, scales = (np.asarray(x, float) for x in (actual, prediction, scales))
    if actual.shape != prediction.shape or actual.shape != scales.shape or actual.ndim != 2 or actual.shape[1] != 48:
        raise ValueError("Expected matched 48-hour calibration paths")
    if not np.isfinite([actual, prediction, scales]).all() or (scales <= 0).any():
        raise ValueError("Invalid wind calibration inputs")
    residual = (actual - prediction) / scales
    quantiles = []
    for start, end in BANDS:
        values = residual[:, start:end].ravel()
        if method == "signed":
            lower, upper = np.quantile(values, [.1, .9], method="higher")
        else:
            q = min(1, np.ceil((len(values) + 1) * .8) / len(values))
            upper = np.quantile(np.abs(values), q, method="higher")
            lower = -upper
        quantiles.append([float(lower), float(upper)])
    return dict(method=method, quantiles=quantiles, nominalCoverage=.8, calibrationOrigins=len(actual))


def bounds(prediction, scales, rule):
    prediction, scales = np.asarray(prediction, float), np.asarray(scales, float)
    if prediction.shape != (48,) or scales.shape != (48,) or not np.isfinite([prediction, scales]).all() or (scales <= 0).any():
        raise ValueError("Invalid wind range inputs")
    lower, upper = prediction.copy(), prediction.copy()
    for (start, end), (lo, hi) in zip(BANDS, rule["quantiles"], strict=True):
        if not np.isfinite([lo, hi]).all() or lo > hi:
            raise ValueError("Invalid range quantiles")
        lower[start:end] += scales[start:end] * lo
        upper[start:end] += scales[start:end] * hi
    return np.maximum(0, lower), np.maximum(0, upper)
