# Deployed forecast ensemble

The fixed v2 ensemble predicts hourly wind, solar and derived system load for
DK1/DK2, 48 hours ahead. Model families and lead-dependent weights are identical
to the development-selected research ensemble. There is no training in the
scheduled scoring job. `model-release.json` pins a checksummed GitHub Release;
the Chronos base checkpoint is pinned separately to its Hugging Face revision.

## Serving and storage

`python -m forecast.run --restore --strict` restores the previous public state,
refreshes electricity/weather inputs, runs three isolated inference workers,
applies fixed weights and writes `dist/data/forecasts.json`. Torch and LightGBM
run in separate processes; the IBM model has its own dependency environment.
`setup.sh` installs CPU PyTorch wheels for both environments on Linux.

`dist/data/forecast-state.json.gz` persists 42 days of normalized observations,
weather runs and 35 days of forecasts on Pages. It contains public source data,
not credentials. This is durable across GitHub Actions cache eviction. A restore
failure prevents replacing it with an empty archive. Model/download/environment
caches improve runtime but are not the source of truth for issued forecasts.

Each forecast is identified by model version, source-hour origin and weather
run. Repeated runs preserve its original values and issue timestamp. The live
error table scores only archived, actually issued forecasts against observed
hours; missing values stay missing, genuine zero values stay zero. Overlapping
forecasts contribute separate errors and the UI reports both forecasted-hour
and origin counts. The seven-day live metrics are separate from the historical
benchmark. Experimental prediction ranges target 80% pointwise coverage, with
measured historical and prospective coverage reported separately. The toggle
is remembered in the browser. Solar ranges are daylight-only because overnight
coverage failed. See the [range and SHAP audit](../research/forecasting/uncertainty/README.md).

`interval-history.json.gz` seeds rolling calibration from archived out-of-sample
errors. `rangeHistory` in published state retains 64 days of daily calibration
records, gradually replaced by actually issued forecasts. Only fully mature
48-hour outcomes can enter calibration. Range version, cutoff and origin counts
are frozen in each issued record; bounds are never retrofitted onto old outlooks.

Demand-only TreeSHAP explanations passed the internal reference stability gate.
The frontend labels them as the LightGBM component and shows its baseline,
prediction and the different ensemble forecast. Five horizons are supported.
Live explanations must also agree with a second reference sample. Wind/solar
explanations failed the audit and are not published. Explanation references are
versioned with model hashes; attribution is not a causal claim.

## Operational differences from the benchmark

The research benchmark used one origin per day (12 UTC) and complete recent
histories. Serving attempts a refresh twice per hour using the most recent
jointly complete source hour, with a conservative 15-minute buffer. It retains
the same daily 00 UTC ECMWF model, made eligible after 12:15 UTC. Five days are
requested to cover 48 hours from origins throughout the day. Historical weather
features continue to use the preceding issued daily forecast, as in training.

Up to three missing older hours in the last 168 hours may be filled from prior
weekly/daily observations or forward-filled earlier values. No backward fill
or interpolation from later observations is used. Recent three hours must be
complete, gaps over two consecutive hours stop an update, and the UI reports
the number filled. TTM retains original observation masks in its 512-hour
history. These operating choices need prospective validation; the historical
benchmark numbers are not claimed as live accuracy.

A failed source or model update keeps the previous valid forecast, with a
delay notice. The ensemble is never silently reweighted around a failed model.
The frontend independently marks forecasts over two hours old as delayed.

## Validation

```sh
python -m forecast.score  # all families against archived golden predictions
python -m unittest discover -s tests -p 'test_forecast_serving.py' -v
node --test tests/forecast-model.test.mjs
```

Set `FORECAST_TTM_PYTHON` to the separate IBM environment and optionally
`FORECAST_MAIN_PYTHON` to the main scoring environment. `forecast.score` checks
portable ridge export, original neural models, tree models, both Chronos
variants and both TTM variants against the original experiment. Scoring code
does not read actual future electricity targets.

Weather: [Open-Meteo Single Runs API](https://open-meteo.com/en/docs/single-runs-api),
ECMWF IFS. Electricity: Energinet `ElectricityProdex5MinRealtime`, CC BY 4.0.
Source requests need no API keys. Free Open-Meteo usage is for non-commercial
use within its published limits. Pretrained models retain Apache-2.0 licences;
the release includes the Apache-2.0 licence text, notices and links. Model code and data retain their own
licences; the application is an independent experiment.
