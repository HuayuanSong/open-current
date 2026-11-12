# Refitting the ensemble for fresher weather

This separate experiment retrains the tree-control, ridge, LightGBM and XGBoost
components using more recent ECMWF forecast vintages. It does not change the
public forecast, its ranges or explanations, or the existing prospective trial.
Read the [measured results](RESULTS.md) and [frozen protocol](PROTOCOL.md).

## What changes

Two matched refits use identical electricity histories, outcomes and settings:
one uses the original daily weather policy and one uses fresher weather. The
four refitted components replace their corresponding ensemble members with
the original blend weights retained. Foundation-model and original-model
members keep their existing weights **and daily-weather inputs**. This tests a
staged update rather than feeding a new input distribution to every old model.

Historical preparation uses two alternating origins per day, covering all four
six-hourly weather cycles. Each example uses an exact archived forecast run,
not observed future weather. The assumed eight-hour publication lag is a
historical approximation; live collection uses actual availability metadata
and a ten-minute replication margin. Missing histories or horizons are excluded.

The August–September component scores and November 1–9 full-blend comparison
are development diagnostics. Those periods have already been inspected, and
the inherited hyperparameters were previously selected using development
months. Final fitting excludes labels on or after 1 November. Historical
improvement cannot, on its own, justify public promotion.
The backfill encountered a provider rate limit before all planned runs were
downloaded. Fitting uses complete cached pairs only; see the
[pre-fitting acquisition note](ACQUISITION.md) and coverage/exclusion audit.
Overlapping 48-hour forecasts are dependent; their hour counts are not counts
of independent experiments. Four daily origins also do not validate every
possible half-hourly issue time of the public service.

## New prospective comparison

The independent [workflow](../../../.github/workflows/weather-retrain.yml)
records four origins per day from **13 November through 10 December 2026**.
There are two attempts per issue window; review keeps the **first valid issue**,
regardless of which prediction later performs better. Late issues and diagnostic
probes are rejected. Missing issues are never backfilled.
The workflow checks out the candidate's fixed release tag, so unrelated later
changes on `main` cannot silently change the models or inference code under test.
The Linux runner also checks each refitted family against a fixed local
prediction fixture before recording forecasts.

Each run stores its inputs, weather source, availability timestamps, model and
code hashes, predictions and observation snapshots in a new GitHub Actions
artifact retained for 90 days. No ledger or predictions are written to `dist/`.
The first eligible observation captured 24–72 hours after each target hour is
used permanently; later revisions cannot replace it. Artifacts are separate
from the old trial and can be downloaded for local review. Collection continues
after the last issue so its outcomes can mature.

Final review is available **after 15 December at 21:00 UTC**. At least 80 complete
origins, 21 distinct days and 80% issue coverage are required. The latest-weather
refit is the predefined primary candidate; the daily refit is a control. There
is no automatic promotion. Model ranges and explanations require separate
validation before any eventual public switch.
After that maturity date, the workflow also downloads the evidence and saves
the comparison report as a separate artifact. It does not deploy a winning model.

## Reproduce locally

Use the existing Python and TTM environments described in
[the production forecasting README](../../../forecast/README.md). The historical
preparation also requires the existing v2/v3 caches. Downloads and model fitting
are resumable; weights and raw observations stay in ignored cache directories.

```sh
.venv/bin/python -m research.forecasting.weather_retrain.prepare --cached-only \
  --state .cache/forecast-audit-20261112/forecast-state.json.gz
.venv/bin/python -m research.forecasting.weather_retrain.train
.venv/bin/python -m research.forecasting.weather_retrain.compare \
  --state .cache/forecast-audit-20261112/forecast-state.json.gz
.venv/bin/python -m research.forecasting.weather_retrain.package

# Explicit diagnostic: never counted as a prospective issue.
.venv/bin/python -m research.forecasting.weather_retrain.prospective --probe

# Requires GitHub CLI authentication to download workflow evidence.
.venv/bin/python -m research.forecasting.weather_retrain.review --download
```

Do not modify the protocol, model files or frozen inference sources during
collection. Checks reject changed hashes rather than silently mixing experiments.
To repeat training with different settings, use a new experiment version.
