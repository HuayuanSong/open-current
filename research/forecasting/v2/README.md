# Local forecasting improvement experiment

The selected ensemble now has a separate [production scoring pipeline](../../../forecast/README.md)
and a [public forecast panel](https://huayuansong.github.io/open-current/#forecasts).
The experiment and historical results below are preserved; serving uses pinned
weights from a GitHub Release and measures live accuracy separately.

Read [RESULTS.md](RESULTS.md) for measured outcomes, [PROTOCOL.md](PROTOCOL.md)
for the frozen experiment design and [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md)
for runtime adjustments. The original experiment is preserved in the parent folder.

Run commands from the `open-current` repository. Raw data and weights live in
`.cache/forecasting/v2/`, not in the published static application. The v2 data
cache links to original raw snapshots and stores October separately.
This requires the completed original experiment cache; on a clean machine,
follow the [parent experiment instructions](../README.md) first.

```sh
.venv/bin/python research/forecasting/v2/prepare.py
.venv/bin/python research/forecasting/v2/frozen_baseline.py --neural
.venv/bin/python research/forecasting/v2/frozen_baseline.py

# Repeat for tree_control, ridge, lightgbm and xgboost.
.venv/bin/python research/forecasting/v2/trees.py lightgbm
.venv/bin/python research/forecasting/v2/trees.py lightgbm --final

.venv/bin/python research/forecasting/v2/chronos_experiment.py
.venv/bin/python research/forecasting/v2/chronos_experiment.py --final

# Separate dependency environment required by the IBM package.
.cache/forecasting/v2/ttm-venv/bin/python research/forecasting/v2/ttm_experiment.py
.cache/forecasting/v2/ttm-venv/bin/python research/forecasting/v2/ttm_experiment.py --final
.cache/forecasting/v2/ttm-venv/bin/python research/forecasting/v2/ttm_experiment.py --scaled
.cache/forecasting/v2/ttm-venv/bin/python research/forecasting/v2/ttm_experiment.py --scaled --final

# Freeze selection BEFORE inspecting any October errors.
.venv/bin/python research/forecasting/v2/evaluate.py select
.venv/bin/python research/forecasting/v2/evaluate.py evaluate
.venv/bin/python research/forecasting/v2/export_manifest.py
.venv/bin/python -m pytest research/forecasting/v2/test_integrity.py research/forecasting/test_forecasting.py tests/test_pipeline.py -q
.cache/forecasting/v2/ttm-venv/bin/python -m pytest research/forecasting/v2/test_ttm_preprocessing.py -q
```

Package versions are recorded separately for the main and TTM environments.
Create environments with Python 3.12 and install the corresponding lock file.
The original `.venv` additionally needs `xgboost` and `peft`; TTM is isolated
because its declared PyTorch and scikit-learn bounds differ. Run LightGBM and
PyTorch in separate processes on macOS to avoid conflicting OpenMP runtimes.

Neural scripts accept `--pilot` for shape, save/reload, input leakage and timing
checks. The Chronos pilot has two steps; TTM has one minibatch. They are not
accuracy benchmarks. All Hugging Face downloads use pinned public revisions
and require no token. Download/rate-limit warnings do not indicate a key is
required. Cached inference works locally after downloading weights.

Development prediction files preserve all pre-October sample positions, with
NaN outside the two validation periods. Final files preserve all sample
positions, with forecasts in October only (the frozen original scores all
positions for integrity checks). Final fitting scripts mask October labels;
only the explicit evaluation stage computes their errors.

After this evaluation, October is no longer an untouched test set. Further
model selection needs a fresh period or a clearly labelled development analysis.

`artifacts.json` lists final model files and hashes, input feature order, pinned
Hugging Face revisions and the frozen blend definition. It is a local handoff
manifest, not a deployed forecast service. Model weights remain in the ignored
cache rather than being added to the website or repository.
