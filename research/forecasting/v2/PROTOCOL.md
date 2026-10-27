# Forecasting improvement experiment, frozen 27 October 2026

The first experiment is preserved. Its June–September test has already been
inspected and is now development data. This experiment reserves October 1–24,
2026 forecast origins (48-hour outcomes through October 26) as a new, short
holdout. Do not inspect October model errors until settings and blend weights
are saved. This is an autumn check, not a full-year accuracy estimate.

## Data and eligibility

Reuse the original source snapshots, SCADA normalization, six weather points,
00 UTC ECMWF runs, 12:15 UTC assumed issue time and 48-hour horizon. Download
October into a separate cache. No observed future weather. Require complete
168-hour target histories and 48-hour labels, as before. Longer 512-hour
histories for TinyTimeMixer use actual older observations, with missing older
values filled from earlier weekly lags or earlier values, plus observed masks.
No later observations may fill a gap. Report eligibility and exclusions.

Keep per-location weather for the neural models, rather than the earlier
averaged weather representation. Historical weather comes from earlier issued
runs. All scalers and model fitting use the corresponding training partition.
Fine-tuning windows have fixed forecast origins: no random slice may move the
cutoff into the forecast weather or target horizon.

## Development and selection

Two expanding training folds: labels strictly before June 1, 2026 predict
June–July; labels strictly before August 1 predict August–September. Purge
48-hour label overlap at each boundary. Select tree configurations on pooled
out-of-fold MAE per area and target. Refit selected settings using labels
strictly before October 1. Use identical eligible evaluation cases.

1. LightGBM and XGBoost: four configurations each, matched absolute-error
   objectives and 400/800 boosting rounds; shallow/regularized and larger
   trees, uniform or 180-day half-life recency weights. Add causal recent
   weather/output conversion features and cross-target history statistics.
   Include updated ridge and an unchanged-feature LightGBM control to separate
   refitting from feature changes. Include original frozen selected models.
2. Chronos-2: pinned original weights; per-location weather zero-shot control;
   LoRA fine-tuning with fixed-origin windows. Compare 100 and 300 steps at
   learning rate 1e-5, batch around four windows. Select by development MAE.
   A training-only timing/shape pilot may adjust batch size for memory/runtime,
   with changes recorded. No cross-learning between evaluation origins.
3. TinyTimeMixer R2: pin a public Apache-2.0 checkpoint with 512-hour context,
   truncate its 96-hour output to 48 hours. Fine-tune with future weather and
   channel mixing. Compare learning rates 1e-4 and 3e-4 for 40 epochs, with
   checkpoints at 10, 20 and 40 epochs selected on development MAE. Freeze
   backbone initially; record any necessary compatibility changes.

Select the neural configuration globally by mean area/target WAPE across the
two folds. Choose a final model separately for each area/target from the
selected families. Also evaluate a convex blend fitted to out-of-fold forecasts
in lead bands 1–6, 7–24 and 25–48 hours, with shrinkage toward equal weights.
Select between a single model and blend using development data only.

## Reporting and checks

Report October MAE, WAPE, daylight solar MAE, error change against the original
frozen models and updated tree control, all family results, sample counts,
warm CPU inference, model sizes and training times. Paired calendar-week
bootstrap intervals are descriptive with very few October weeks; do not claim
statistical certainty or a guaranteed improvement. Selection is not revisited
after inspecting October. No prediction-interval guarantee is attempted here.

Verify causal window slicing, train/validation label separation, original
artifact integrity, no target leakage through future covariate tensors, and
checkpoint reload consistency. Save code, package versions, source revisions,
predictions and settings. Work remains local; no website deployment.
