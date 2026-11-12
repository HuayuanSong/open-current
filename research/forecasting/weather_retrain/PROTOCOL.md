# Refitting for fresher weather

Frozen on 12 November 2026, before preparing data or fitting this experiment.
The prior latest-input replay failed promotion. Its November outcomes, and all
earlier experiments, are development evidence rather than untouched tests.

Refit the ensemble's tree-control, ridge, LightGBM and XGBoost components for
two input policies: the operational daily 00 UTC policy, and the most recent
six-hourly ECMWF run under an assumed eight-hour historical publication lag.
Use the existing selected hyperparameters, feature definitions, six locations,
seed and four CPU threads. Do not retune after inspecting November errors.
The matching daily-policy refit separates retraining from input freshness.

Prepare two origins per eligible date, alternating 03/15 and 09/21 UTC by
calendar-day parity from 1 January 2025 through 29 October 2026. Eligibility
requires the production electricity-history checks and all 48 future labels
for both regions. Require complete weather horizons under both policies.
Keep historical weather and target inputs identical. Existing caches may be
reused only with matching request provenance and sufficient horizon. Record
exclusions, source hashes, feature order and exact model-run timestamps.

Development fitting uses labels strictly before 1 August 2026. Report August
and September errors only for origins whose last label is before 1 October.
Existing selected hyperparameters previously used these months: these are
diagnostics, not a fresh validation claim. Final fitting uses labels strictly
before 1 November. The November 1–9 comparison is also explicitly diagnostic.
It excludes target overlap with final fitting. Never include evaluation labels
in fitting, scaling, sample weights, stopping rules or model selection.

For the full-ensemble comparison replace exactly the four refitted families;
retain original blend weights and the original inputs/weights for all other
members. Only refitted members see latest weather. Report both matched refits
and the unchanged production ensemble. This is a staged component update, not
a refit of the foundation models. No point forecasts, ranges, explanations or
records in the production archive or existing prospective trial are modified.

The new prospective period is 13 November–10 December, at 03, 09, 15 and 21
UTC. Record predictions before outcomes and within minutes 15–59 of each origin.
Use provider availability plus ten minutes, never simulated publication times,
for live run selection. Retain weather provenance and immutable inputs and
model hashes. Missing issues cannot be backfilled as prospective forecasts.
Freeze outcomes 24–72 hours after each target hour; missing frozen observations
exclude a paired trajectory. Evaluate only after the complete trial matures.

Promotion requires at least 80 completed origins, 21 distinct origin days,
80% issue coverage, 5% lower DK1 wind MAE than the deployed ensemble, no overall
MAE increase for any series and no lead-band increase over 5% (1–6, 7–24,
25–48 hours). Report both refits; the latest-weather version is the predefined
primary challenger. These gates are necessary, not sufficient: prediction
ranges and explanations need separate validation. There is no automatic
promotion. The existing December trial remains frozen and independent.
