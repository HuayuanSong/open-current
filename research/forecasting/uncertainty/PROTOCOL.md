# Prediction ranges and explanation audit

Frozen before evaluating range coverage or SHAP stability, 2026-10-27.
Point models and blend weights stay unchanged.

## Prediction ranges

Target: 80% pointwise coverage for each region/target, not simultaneous coverage
of an entire 48-hour trajectory. Use archived out-of-fold June–September
predictions of the existing fixed blend. The blend itself was selected using
these development predictions, so no exchangeability or finite-sample conformal
guarantee is claimed. October has already been inspected for point accuracy;
it is a separate descriptive range audit, not a new untouched test set.

Choose among absolute, output-scaled and signed-error empirical ranges using
expanding pre-October validation. Initial calibration is June–July; evaluate
August and September sequentially, adding only labels that matured at least
15 minutes before each origin. Bucket by existing lead bands (1–6, 7–24,
25–48 hours); solar additionally separates daylight/night using issued
terrestrial radiation. Scaling uses only predicted output and a floor derived
from calibration-period observations. Select the smallest mean interval score
at alpha=.2, a proper score penalizing width and misses. Freeze the choice and
calibrate on all mature pre-October development residuals before opening the
October audit. Bounds are non-negative and include the deployed point forecast.

Report coverage, mean width, interval score, daylight solar coverage and
lead-band coverage. Preserve origin counts and use origin/week summaries;
overlapping forecast hours are not independent trials. A public experimental
80%-target range must achieve 70–90% overall coverage per series, at least
65% in each lead band and daylight solar subset, and improve interval score
over a pre-October climatological range. Failed series remain unavailable with
a reason; thresholds are engineering gates, not statistical proof.

Freeze each range with its issue time and calibration version. Never backfill
ranges onto already issued forecasts for live coverage scoring. Measure
prospective coverage separately. Initial validation covers daily 12 UTC
origins with complete inputs; other issue times and filled inputs remain an
explicit operational extrapolation. Longer seasonal evidence is needed.

## SHAP audit

Explain the refitted original-feature LightGBM component (`tree_control`), not
the complete neural/tree ensemble. Use all eligible October origins, fixed
leads 1, 6, 12, 24 and 48, and all six region/target combinations. References
come only from pre-October observations, matched by region and lead band.

Evaluate grouped attributions for recent electricity, wind, sunlight/cloud,
temperature and calendar/horizon. Test multiple random reference samples and
sample sizes. Compare engineered-feature TreeSHAP with a grouped raw-input
Shapley calculation that recomputes derived features. Document that these are
different attribution games: agreement is a robustness check, not proof that
one is the uniquely true explanation. Preserve the baseline and explain raw
tree output; assess output clipping explicitly.

Publication requires additive reconstruction within .1 MW; at least 90%
agreement on materially nonzero contribution signs; median top-two overlap
at least .75; median normalized attribution change <=.15 and 90th percentile
<=.30 across same-distribution reference samples. Exclude trivial near-baseline
predictions from rank/sign gates and report their count. Require qualitatively
consistent leading groups under coherent raw-input masking; do not publish
individual engineered-feature causal stories. Report all series, failures and
methods. Publish only if the audit supports a useful, stable explanation with
an explicit component label; otherwise retain explanations internally.

References: [time-series conformal prediction](https://arxiv.org/abs/2202.07282),
[dependent-feature Shapley values](https://arxiv.org/abs/1903.10464),
[TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html).
