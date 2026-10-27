# Forecast inputs and adaptation

Four local experiments extend the deployed ensemble: capacity-weighted wind
weather, newer weather runs (plus a regional model), causal recent-error
correction, and condition-dependent ensemble weights.

See [RESULTS.md](RESULTS.md) for the measured results and [comparison chart](comparison.png).

Read [PROTOCOL.md](PROTOCOL.md) before interpreting results. June–October has
already been examined in earlier work; this is retrospective research with
chronological fitting, not a new untouched holdout. [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md)
records source repairs and implementation corrections. Nothing in this folder
changes the deployed point forecasts, their ranges, or explanations.

## Reproduce locally

Run from the repository root using Python 3.12 and the existing v2 data/model
cache. The v3 requirements add Excel parsing and coordinate conversion to the
v2 environment. All requests are keyless. Raw sources and fitted models stay
in `.cache/forecasting/v3/`; source files are hashed. No target values enter
the geographic clustering.

```sh
.venv/bin/python -m pip install -r research/forecasting/v3/requirements.txt
.venv/bin/python -m research.forecasting.v3.fleet
.venv/bin/python -m research.forecasting.v3.weather --pilot
.venv/bin/python -m research.forecasting.v3.weather
.venv/bin/python -m research.forecasting.v3.prepare

# Select before evaluating. Repeat for evening and regional studies.
.venv/bin/python -m research.forecasting.v3.train_weather fleet select
.venv/bin/python -m research.forecasting.v3.train_weather fleet evaluate
.venv/bin/python -m research.forecasting.v3.train_weather evening select
.venv/bin/python -m research.forecasting.v3.train_weather evening evaluate
.venv/bin/python -m research.forecasting.v3.train_weather regional select
.venv/bin/python -m research.forecasting.v3.train_weather regional evaluate

.venv/bin/python -m research.forecasting.v3.adapt_experiment select
.venv/bin/python -m research.forecasting.v3.adapt_experiment evaluate
.venv/bin/python -m research.forecasting.v3.baseline_evening
.venv/bin/python -m research.forecasting.v3.report
.venv/bin/python -m research.forecasting.v3.verify
.venv/bin/python -m pytest research/forecasting/v3/test_v3.py -q
```

Alternatively, `python -m research.forecasting.v3.run_experiment` performs the
whole sequence. `--skip-fetch` reuses the completed source cache. Independent
weather studies run in separate processes; all their settings are frozen
before any October evaluation. `predict.py` scores the saved, checksummed
challengers without receiving future labels. `verify.py` checks those outputs
against the original research predictions.

Weather backfill is cached and paced at one six-location request per 1.05
seconds, including response time. It requests only dates with eligible v2
electricity histories. The initial backfill needs roughly 1,065 requests;
provider billing/rate-limit units may count each location separately. A daily
quota stops the job with an actionable error. Resume later using the same
cache. Missing archived runs are recorded and comparisons exclude them
explicitly; no observed-weather replacement is used.

## What is compared

- `fleet`: 12 UTC origins, original weather versus six fleet-based weather
  points added to the original inputs, with date-specific installed-capacity
  features. Only wind changes.
- `evening`: 18 UTC origins, a refitted 00-run control versus 06-run weather,
  then 06-run weather plus fleet features. Labels and electricity histories
  are identical between candidates. Fleet features still use the 00 run;
  the original six locations use the newer 06 run. This separates the two
  changes without doubling the initial weather backfill.
- `regional`: 18 UTC origins on available DMI dates, with matching shorter
  training histories for ECMWF, DMI, and combined weather inputs. The DMI
  archive covers 47 of these 48 hours; the final hour uses the same-issue
  ECMWF forecast plus an explicit source indicator. Interior gaps are not
  filled. See the pre-training [horizon addendum](REGIONAL_ADDENDUM.md).
- `adapt`: 12 UTC origins, exact deployed blend weights for October; a
  development baseline fitted only before August for August–September.
  Correction and gating are evaluated separately and together. Daily research
  forecasts do not reproduce the more frequent live error stream.

Selection JSON files refuse different overwrites. All candidate results,
including losses, are retained. Booster save/reload checks run after fitting;
the other tests perturb unavailable labels, check forecast publication timing,
deduplicate overlapping errors, and verify nonnegative convex weights.
