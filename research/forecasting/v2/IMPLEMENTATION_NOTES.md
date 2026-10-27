# Implementation notes recorded before final evaluation

- TinyTimeMixer uses an isolated environment because granite-tsfm 0.3.11
  requires older PyTorch/scikit-learn versions than the original experiment.
  The original environment and artifacts are retained.
- The Chronos pilot caught a training/evaluation-mode difference when reloading
  adapters. Inference now explicitly calls `eval()`; the corrected pilot passes
  reload checks. Adapter metadata pins the base model revision as well.
- The four-window Chronos pilot took about 2.7 seconds per update and peaked
  near 4.7 GB process memory. Use two windows (88 target/covariate series) per
  update for the full CPU experiment. Step budgets remain 100 and 300. This is
  a runtime-driven pilot adjustment, made before the development comparison.
- Chronos training windows contain exactly 168 history + 48 label hours and
  set `min_past=168`. The library's random cutoff therefore has exactly one
  possible value. Checks verify history, label and covariate slices explicitly.
- TinyTimeMixer keeps the pretrained channel-independent decoder and adds its
  supported forecast channel-mixing/weather component, initialized from
  scratch. The backbone is frozen; decoder and weather head are fine-tuned.
  Future target channels are zero in all forward calls; labels enter only the
  external scaled absolute-error loss. Perturbing future target channels is
  checked not to change predictions. Older target-history gaps use causal
  fills with original observation masks, not synthetic context padding.
- The complete-case October holdout has 13 origins per area, rather than all
  24 requested dates. All families use those same cases. Missing input hours
  can invalidate several overlapping seven-day histories.

## Development-stage addition, before October evaluation

The first TinyTimeMixer integration used raw units, with internal context
normalization and a scaled target loss. Inspection of IBM's official
[exogenous-variable tutorial](https://github.com/ibm-granite/granite-tsfm/blob/main/notebooks/tutorial/ttm_with_exog_tutorial.ipynb)
showed that their example also standardizes inputs before the model. Add a
second `ttm_scaled` candidate with area-specific training-only mean/std for
targets and weather, preserving original masks and fixed forecast origins.
Compare the same two learning rates and 10/20/40-epoch checkpoints on the same
development folds. Retain the original TTM results as an explicit ablation.
Both can enter development selection; neither is chosen using October errors.
This expands the original protocol during development and is not represented
as an originally preregistered choice.

For the scaled run, cache the frozen, evaluation-mode backbone representations
during training. The decoder and weather head still receive gradients. Check
the cached calculation against the library's normal forward pass before
training. Validation and inference timing use the full uncached model, so
reported inference does not benefit from cached test features.
