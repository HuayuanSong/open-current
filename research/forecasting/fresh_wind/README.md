# Current-weather wind forecasts

Implemented 16 November 2026 as the public wind outlook. Solar and demand keep
the original ensemble, daily ECMWF inputs, ranges and explanations. The existing
prospective experiments keep their protocols and previously issued records.

## What changed

The refitted LightGBM, XGBoost, ridge and tree-control components now receive the
latest complete ECMWF run with recorded availability evidence and retrieval times. There
is no fixed noon waiting period for the public wind update. West Denmark also
uses DMI HARMONIE weather: 75% ECMWF and 25% DMI predictions from this refitted
branch. East Denmark uses ECMWF; substituting DMI increased its diagnostic error.
The other ensemble members retain their original inputs and fixed weights.

A bounded correction based on two consecutive 15-minute wind-production means
changes the first six hours only. It uses completed five-minute intervals with
a ten-minute publication margin and decays with lead time. These are measured
power readings, not weather-station wind speeds.

Both weather providers are checked during ordinary refreshes. Downloads specify
the exact initialization, all six locations, the trained units and variables.
Metadata-selected runs require a ten-minute replication margin. If geographical
metadata replicas lag, one plausible newer initialization is checked directly
against the exact-run archive. It qualifies only when the response starts at
that initialization and all locations, variables and 48 future hours are present.
Its first actual retrieval is recorded as the availability evidence; no earlier
publication time is inferred from the schedule. Missing required inputs preserve the last issued
wind forecast and display a delay notice. A cached, still-complete run can be
used when metadata retrieval fails; its original run time remains visible.
GitHub scheduling and upstream publication still limit freshness.

The initial live check reproduced this discrepancy: the runner's metadata still
reported ECMWF 00 UTC while the exact 06 UTC run was accessible elsewhere.
Version `fresh-wind-2026-11-16.1` adds direct complete-response evidence; it retains
the first version's issued record and does not change the selected model weights.

Sources: [DMI model documentation](https://open-meteo.com/en/docs/dmi-api),
[ECMWF documentation](https://open-meteo.com/en/docs/ecmwf-api), and
[individual forecast runs](https://open-meteo.com/en/docs/single-runs-api).
The non-commercial, unauthenticated endpoints are used. DMI updates every three
hours, ECMWF every six; those are initialization cycles, not immediate delivery.

## Evidence and limits

Source weights were chosen on 28 development origins from 1–7 November with all
48-hour outcomes before 10 November. The grid was DMI weights 0, .25, .5, .75, 1.
The regional choice was fixed before checking the 16 later origins from
10–13 November. The five-minute correction came from the earlier frozen
[wind experiment](../wind_update/README.md). No power model was refitted on
these evaluation outcomes.

| Wind mean absolute error, MW | West Denmark | East Denmark |
| --- | ---: | ---: |
| Previous public forecast | 722.66 | 115.75 |
| Current ECMWF refit + correction | 622.23 | 100.58 |
| DMI substitution + correction | 621.60 | 134.62 |
| Selected regional mixture + correction | **619.21** | **100.58** |

The selected update reduced overall error by **14.3% / 13.1%** versus the previous
forecast and improved all three lead bands (1–6, 7–24 and 25–48 hours). DMI's
additional contribution in the west was small: about 0.5% versus the ECMWF-only
update. Most of the gain came from refitted components and recent-power inputs.
Two additional 14 November stress origins also improved in both regions; errors
remained substantial. These checks do not promise accurate forecasts every hour.

This is a short retrospective diagnostic, with revised electricity observations
and assumed historical weather availability of eight hours for ECMWF and four
for DMI. November outcomes have already been inspected in this project, and the
overlapping 48-hour forecasts are dependent. These are not independent live or
year-round accuracy claims. Live error statistics start with the new version's
actually issued forecasts and exclude the original model and diagnostic probes.

New 80%-target ranges fitted on the earlier development outcomes covered only
**35.2% / 54.2%** of the later outcomes. They are **not published**. Wind displays
the point forecast with an explicit unavailable-range explanation; solar and
demand retain their range controls. The old model's bands are not transplanted
onto new predictions, and no range is added retrospectively.

Full evidence: [protocol](protocol.json), [results](results.json),
[selected policy](selection.json), [range check](range-check.json), and
[additional stress cases](stress-results.json).

## Operation and reproducibility

`python -m forecast.fresh_wind` runs after the original ensemble and existing
comparison collector. It reuses CPU worker outputs only if every input array
matches and the output files postdate those inputs. Raw electricity is shared
with collection; source responses are not requested separately for every model.
The update needs the already pinned weather-refit model bundle and no new
training job, API key, hosting service or GPU.

Records persist under `windForecasts` in `forecast-state.json.gz`. Each saves
original predictions, issue time, source timestamps, recent-power inputs,
model/source/input hashes and a version. New issues must precede the first
target hour's end. Duplicate source/observation combinations preserve the first
record. Inference changes require a new model version. Raw inputs and sources
are retained as GitHub Actions artifacts for 90 days; issued forecasts have
35-day retention. These operational records are separate from frozen trials.

`forecasts.json` retains the original payload and adds `windOutlook`, with its
own latest forecast, comparisons, errors and status. The browser selects this
payload for Wind and keeps the original payload for Solar and Demand. Downloads
use the displayed record. If the first updated wind forecast is unavailable,
the original remains visibly labelled as the previous model; source failures
also show a delay status. `health.json` reports the two
issue times separately. Only static JSON is served to visitors.

```sh
.venv/bin/python -m research.forecasting.fresh_wind.evaluate
.venv/bin/python -m research.forecasting.fresh_wind.check_ranges
.venv/bin/python -m research.forecasting.fresh_wind.stress
.venv/bin/python -m forecast.verify_wind
.venv/bin/python -m unittest tests.test_fresh_weather tests.test_fresh_wind -v
```

The replay uses the timestamped snapshots in `.cache/wind-update/`, prior
baseline inference caches, and checksummed native model weights. It downloads
exact archived DMI runs with pacing, saves sources and resumes per-origin
inference. For a current local diagnostic, `python -m forecast.fresh_wind --probe
--state path/to/state.json.gz` writes only to `.cache/fresh-wind/live/probe/`;
it cannot change published files or contribute live performance scores.
