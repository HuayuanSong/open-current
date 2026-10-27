# Forecasting experiment: frozen before fitting

Research date: 27 October 2026. Local experiment for Open Current; no deployment.

## Question

Can freely accessible, issued weather forecasts improve the next 48 hourly wind,
solar and system-load estimates in DK1 and DK2, with inference practical on a CPU?
This is an operational-SCADA prediction task, not settlement-grade demand prediction.

## Data and time

- Energinet `ElectricityProdex5MinRealtime`, December 2024–September 2026. Hourly
  means require all twelve five-minute measurements. Wind is onshore plus offshore;
  load is generation plus net imports, including Great Belt and excluding duplicate
  Bornholm flow. DK2 has only German/Swedish external links. Do not fill unknown values.
- Open-Meteo Single Runs: explicitly selected ECMWF IFS, each day's 00 UTC run,
  January 2025–September 2026. Three representative coordinates per bidding area,
  including offshore locations. These are geographic samples, not capacity weights.
- One forecast origin daily, after the hour ending 12 UTC, assuming a 15-minute
  publication buffer. Inputs stop at that completed hour; predict hours ending
  13 UTC through 12 UTC two days later. This is **not** a backtest of 48 updates/day.
- Twelve hours after weather initialization is a conservative availability assumption;
  historical API publication times are not archived. Historical SCADA is retrieved
  today, not an immutable vintage. No claim of a fully audited point-in-time backtest.
- Radiation is the preceding-hour average, aligned with electricity's hour-ending
  timestamp. Weather speeds are m/s; electrical targets are MW, not summed MW or MWh.
- Training: 2025. Validation/model selection: January–March 2026. Uncertainty
  calibration: April–May 2026. Untouched test: June–September 2026.
- Purge origins whose 48-hour labels cross a split boundary. Fit transformations on
  training only. Use the same eligible origins for every model. No random train/test split.
- Remove the provider's documented erroneous March 2026 periods and invalid negative
  generation/nonpositive load. No test-derived outlier thresholds. Preserve unexplained
  source anomalies and report that limitation, including nighttime solar offsets.

## Candidates and selection budget

1. Persistence, repeated last complete day and last week baselines.
2. Ridge regression with calendar, recent production, and nonlinear weather features.
3. LightGBM, two compact parameter settings, weather and no-weather ablation;
   quantile models for 10th/90th percentiles and separate calibration-period correction.
4. A compact TiDE-inspired residual MLP: seven days of target history, known future
   weather/calendar, feature projection, encoder/decoder, temporal decoder and linear
   history skip. Two widths, validation early stopping. Explicitly a local adaptation,
   not a reproduction of the TiDE paper's benchmark or its published numerical results.
5. Chronos-2 pretrained, with known future covariates, as a current foundation-model
   comparator. No claim that zero-shot weights were trained locally. Record exact
   model revision and pretraining-contamination uncertainty; test is after paper release.

Select model settings and a fixed 50:50 tree/neural blend using validation MAE per
target/area; never select using the test. Keep the frozen fit for the test to avoid
confounding architecture with retraining schedules. For neural training use three
targets separately, pooling areas with a static indicator and training-only scaling.
If a dependency or resource limit prevents a candidate, record the actual failure
and evaluate the feasible candidates; do not silently substitute another algorithm.

## Evaluation and artifacts

MAE and RMSE (MW), baseline-relative MAE skill, horizon buckets 1–6/7–24/25–48,
daylight-only solar, hourly ramp error and large-load hours. P10/P90 coverage,
width and pinball loss. Forecast origins overlap; use block resampling of days
for uncertainty on model comparisons, not independent hourly confidence intervals.
Time training and warm inference with four CPU threads; record machine and model
sizes. Local Apple CPU timing is not a measurement on a GitHub-hosted Linux runner.

Save code, exact environment, local data manifest/checksums, predictions, model
artifacts, metrics, figures and a linked research/results report. Raw data and weights
stay in ignored `.cache/forecasting/`; reviewable reports live here.

## Limits fixed in advance

One annual training cycle, four test months, only two Danish bidding areas; no claim
of generalization to all Europe or a winter test. No turbine-level capacity/location
inventory, outages or curtailment labels. Free Open-Meteo use is non-commercial and
rate limited. The experiment assesses national/area-level aggregates, not individual
plants. Weather ensembles and regional DMI forecasts are follow-up candidates,
because their comparable archived histories are shorter or substantially larger.
