# Prospective weather forecast trial

The trial records new candidate forecasts alongside the unchanged deployed
ensemble. It does not alter the dashboard's predictions, ranges or explanations.
Historical October results selected these candidates; only forecasts recorded
from **27 October through 7 December 2026** enter this prospective comparison.

The [frozen protocol](protocol.json) defines 42 daily forecast origins at
**18:00 UTC**. Issuance must finish between 18:15 and 18:59 UTC, before the first
target hour ends at 19:00 UTC. The existing half-hourly GitHub workflow normally
provides an opportunity at 18:37 UTC (19:37 Danish winter time). Scheduled runs
can be delayed or skipped: missed windows remain missing and are never backfilled.

## Fixed candidates

The primary candidate uses the same recipe in both DK1 and DK2:

| Target | Candidate |
| --- | --- |
| Wind | 06 UTC ECMWF forecasts plus capacity-weighted fleet weather from 00 UTC |
| Solar | Combined 06 UTC ECMWF/DMI inputs |
| Demand | 06 UTC ECMWF inputs |

All are the fitted LightGBM models from the [v3 experiment](../../research/forecasting/v3/RESULTS.md).
The baseline is the exact deployed v2 nine-family ensemble, recomputed on the
same captured electricity history and original 00 UTC weather snapshot. This
avoids comparing a new candidate against a public forecast issued with older
observations. A refitted evening tree control and the three full weather
variants are retained as secondary diagnostics. No additional winner is chosen
from them automatically.

DMI supplies 47 hours at this origin. Only its known terminal missing hour is
filled from the already-available ECMWF run, with a source indicator. Interior
gaps invalidate that candidate. Missing inputs leave the corresponding target
unavailable; the baseline is never silently substituted into a candidate's score.

Weights, feature definitions, six fleet points, and installed capacity from
the register snapshot available on 27 October are frozen. New fleet additions
or source revisions do not trigger a hidden refit during the trial. No tuning,
retraining, conditional ensemble weights or recent-error correction is added
to the primary recipe.

## How evidence is retained

- Each day's predictions, exact issue time, model version, source-run metadata,
  input hashes, workflow run ID and commit are saved once. Duplicate attempts
  preserve the original record; modified records fail integrity checks.
- The ledger lives under `trial` in the existing `forecast-state.json.gz` on
  GitHub Pages. It survives runner/cache eviction and is not subject to the
  production forecast ledger's 35-day trimming. A missing remote archive blocks
  deployment rather than replacing previously issued trial evidence.
- Raw source snapshots and feature arrays are retained as GitHub Actions
  artifacts for 90 days whenever a trial forecast is issued. These contain
  only public electricity/weather data. The static ledger contains all forecast
  values and the frozen observed values needed to reproduce the scores.
- For each target and valid time, store the **first complete observation captured
  24–72 hours later**. Later revisions cannot change that trial outcome. This is
  a consistent snapshot of an operational feed, not a claim of final settlement
  data. Missing outcomes stay missing.

## Evaluation and review

Metrics update on the existing refresh schedule at
[`data/forecast-trial.json`](https://huayuansong.github.io/open-current/data/forecast-trial.json).
The public forecast panel continues using v2; the trial has no dashboard panel.

Score only complete, paired 48-hour trajectories. Report MAE, WAPE, daylight
solar error, lead-band errors, complete/partial counts, missed issue windows and
weekly-block bootstrap intervals. Overlapping forecasts are dependent, and the
intervals are descriptive with few weeks; this does not establish year-round
performance or causality.

Final review opens **12 December 2026 at 18:00 UTC**, after the final forecast's
48-hour horizon and 72-hour observation window. No early stopping is allowed.
For each area/target, review eligibility requires at least 28 complete paired
origins spanning six calendar weeks and issuance on at least 80% of the 42
planned dates. A candidate is flagged for review if MAE falls by at least 3%
and no lead band regresses by more than 5%. Otherwise retain the baseline or
report insufficient evidence. Solar daylight performance and uncertainty
intervals are additional diagnostics to consider during that review.

**There is no automatic model promotion.** A later release needs an explicit
decision, new range calibration and a renewed explanation audit. The trial
summary continues updating automatically; no messages or notifications are sent.

## Local and CI commands

```sh
python -m forecast.trial.verify  # real saved-model parity fixture, no API calls
python -m forecast.trial.probe   # exact-run weather availability, never issues
python -m forecast.trial.run --strict  # uses already-restored production state
python -m unittest tests.test_forecast_trial -v
```

`package.py` is a one-time local export from the completed research cache.
Scheduled jobs download the checksummed release; they do not need the research
cache, Excel/coordinate libraries, or training infrastructure.

Operational references: [GitHub scheduling](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule),
[Open-Meteo single runs](https://open-meteo.com/en/docs/single-runs-api),
[model availability](https://open-meteo.com/en/docs/model-updates).
