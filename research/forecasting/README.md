# Open Current forecasting lab

Local research, training and evaluation of weather-informed wind, solar and system
load forecasts for DK1 and DK2. The selected v2 ensemble has a separate
[production scoring pipeline](../../forecast/README.md); experimental candidates
remain local until evaluated for release.

- [Findings and model rationale](RESEARCH.md)
- [Experimental protocol](PROTOCOL.md)
- [Measured results](RESULTS.md)
- [Exact data/feature/split configuration](config.json)
- [Deployed ensemble experiment](v2/RESULTS.md)
- [Weather inputs and adaptive forecasting experiments](v3/RESULTS.md)

## Reproduce

Run from the repository root. Use Python 3.12 and an isolated environment. The
recorded lock file is from macOS arm64; Linux hardware/wheels can yield different
runtime and small numerical differences. LightGBM on some macOS installations
requires the OpenMP runtime.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r research/forecasting/requirements.lock.txt
.venv/bin/python -m pytest research/forecasting/test_forecasting.py -q
.venv/bin/python research/forecasting/fetch_data.py all
.venv/bin/python research/forecasting/build_data.py
.venv/bin/python research/forecasting/train.py
.venv/bin/python research/forecasting/chronos_benchmark.py
.venv/bin/python research/forecasting/evaluate.py
.venv/bin/python research/forecasting/verify_artifacts.py
```

No API credentials are used. The weather backfill makes 638 daily-run requests,
each for six locations. Check Open-Meteo's current non-commercial usage terms and
rate limits before increasing the scope. Cached requests are skipped. Explicitly
unavailable weather runs are recorded as exclusions; other HTTP errors fail the
download instead of silently producing a smaller benchmark.

Chronos downloads the pinned public model in [chronos_model.json](chronos_model.json)
on first use. `cross_learning=False` prevents a later forecast origin from sharing
its target history with an earlier origin. Prediction checkpoints allow inference
to resume; remove only `.cache/forecasting/chronos_predictions.npz` if intentionally
changing its inputs or model configuration.

## Where the outputs live

Reviewable code, bibliography, configuration, summary tables and figures live in
this folder. Large data and weights live in ignored `.cache/forecasting/`:

- `electricity/`, `weather/`: compressed source responses with URLs/retrieval times.
- `manifest.json`: raw-file checksums and backfill configuration.
- `electricity_hourly.parquet`: normalized MW values on hour-ending timestamps.
- `dataset.npz`, `samples.csv`, `exclusions.csv`: model inputs and origin metadata.
- `models/`: trained ridge/tree/neural models and neural learning curves.
- `trained_predictions.npz`, `chronos_predictions.npz`: saved forecast arrays.
- `huggingface/`: pinned pretrained checkpoint.

The finalized-snapshot SCADA archive can change upstream. A new download is not
guaranteed to reproduce the original labels byte for byte; preserve the cache
and manifest when comparing later experiments. Do not tune on the test results
and then continue calling that same period untouched. New tuning needs a new
future evaluation period or a properly nested temporal evaluation.

The experiment requires complete history and targets, so its accuracy applies
to eligible windows. A production integration still needs missing-data fallback,
actual publication tracking, cold-start/runtime measurement on Linux, and testing
at every intended refresh time. It can load saved models and publish forecast JSON
without training during each refresh.

Verification for the completed experiment: 16 temporal/data-pipeline tests passed.
Reloading 18 saved statistical/tree models and three selected neural models
reproduced their stored predictions within 0.005 MW absolute tolerance plus
floating-point relative tolerance. Chronos P10/P50/P90 outputs had no crossings.
