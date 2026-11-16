# Short-term wind and uncertainty experiments

This work has four parts: independent European collection, a small 1–6-hour wind
challenger, conditional prediction ranges, and weather-ensemble inputs. The
published wind/solar/demand models are unchanged by these experiments.

## Faster publication

`collect-europe.yml` runs the slow, paced Energy-Charts queues independently and
uploads raw responses with SHA-256 checksums. `pages.yml` restores the live site,
imports the latest completed collection, fetches Danish readings, and performs
forecast inference. It does not wait for an active European collection.

Only `pages.yml` publishes the site. Collection artifacts cannot supply a forecast
archive or executable code. Older/future-dated, corrupt, unknown or unlicensed
inputs are rejected. Missing collections preserve published readings. Collection
provenance is retained as `europeCollection` in `index.json`; original observation
timestamps remain intact. GitHub scheduling itself remains best effort.

## Short-term model

The challenger predicts the change from the latest measured wind level over the
next six hours. Inputs include two completed 15-minute production means, three
hours of five-minute history, hourly lags, offshore share, and future ECMWF wind
speed, direction and generic power-curve features at the existing six locations.
The same ten-minute publication buffer is applied in replay and serving.
Incomplete histories produce an explicit fallback to the existing public model.

Ridge (two regularization strengths) and LightGBM (7/15 leaves, 250 trees) were
compared. Development training labels precede August 2026; selection uses
August/September outcomes. Final fitting uses labels before November, with the
last available training label on October 26. Both areas selected 15-leaf
LightGBM. Each final model uses **1,003 historical origins**; 67 attempted origins
were excluded for missing inputs. See `data-audit.json`, `selection.json` and
the frozen settings in `protocol.json`.

The November 10–13 diagnostic has 16 origins and compares with the newly
deployed current-weather wind forecast, including its existing correction:

| Region | Current 1–6h MAE | Challenger MAE | Reduction | Local checks |
| --- | ---: | ---: | ---: | --- |
| West / DK1 | 214.89 MW | 209.97 MW | 2.29% | Failed minimum gain and lead checks |
| East / DK2 | 59.89 MW | 44.24 MW | 26.13% | Passed local checks |

Errors on the small sharp-change subset fell from 571.50 to 545.21 MW in DK1
(8 hours) and from 129.44 to 94.28 MW in DK2 (11 hours). These subsets are too
small for a general ramp-performance claim. First-hour DK2 error increased
slightly (15.21 to 15.57 MW), within the predeclared 5% lead tolerance.
No fallback was needed in these 16 diagnostic origins.

Two later November 14 stress cases gave mixed results (`stress-results.json`).
DK1 improved in both (243.47 → 218.67 MW and 421.51 → 242.04 MW). DK2 improved
at 09 UTC (83.24 → 71.79 MW) but worsened substantially at 03 UTC
(100.19 → 142.27 MW). Settings were not changed after this check. This is a
concrete reason to retain the current public model while collecting live evidence.

**These are retrospective results on previously inspected November dates.**
Revised electricity observations and assumed historical weather release lags
are used. Overlapping forecast hours are dependent. A new model is not promoted
publicly on this evidence alone. The saved exports are parity-checked against
their training models; scoring is CPU-only, with no scheduled retraining.

## Conditional ranges

The retrospective experiment compares absolute, signed, power-scaled and
ECMWF/DMI-disagreement-scaled residual ranges with 7/14-day windows. Calibration
only uses complete 48-hour outcomes matured before each issue. Method selection
finishes before the November 10 evaluation begins. Evaluation checks lead-band
coverage, interval score (miss penalties plus width), and width inflation.

The selected DK1 disagreement ranges reached 73.0% overall coverage, but only
61.2% at 25–48 hours and failed the width check. DK2's selected absolute ranges
covered 57.2% overall and failed the coverage/score checks. **Neither is public.**
Full comparisons and failures remain in `results.json`.

## Weather ensembles and live evaluation

The [Ensemble Mean API](https://open-meteo.com/en/docs/ensemble-mean-api) supplies
ECMWF wind means and spreads without a source API key. Current values were
verified at all six locations. Exact `run=` requests for this ensemble-mean
model returned unavailable, while a historical date-range request returned a
stitched series. That series is not used as a historical issuance-time forecast.

`forecast.weather_ensemble` instead archives actual current responses, request
parameters, checksums and retrieval times. It does not invent an initialization
time. A complete snapshot is reused for at most an hour. The average of local
member spreads is an uncertainty feature; it is not the spread of a regional
average, because spatial member correlations are unavailable.

`forecast.wind_refinement` records both regions' short-term challengers beside
the unchanged public baseline, plus these internal 80%-target range candidates:
absolute, signed, power-scaled and ensemble-spread-scaled. Ranges begin only
after 12 complete calibration origins exist. The ensemble sensitivity proxy
requires calibration against electricity errors; weather spread alone is not a
power prediction range. All issued points/bounds remain immutable.

Only one first-issued forecast per six-hour origin bucket contributes to
calibration/evaluation. Every run checks matured outcomes; at least 14 distinct
evaluation days are required for the prospective evidence checks. Range checks
also require acceptable coverage in every lead band, better interval score,
and bounded width inflation. **There is no automatic promotion**, even if
those checks pass. The prospective analysis remains conditional on source
availability and cannot establish year-round coverage.

The published state stores these separately under `windResearch`; the normal
forecast panel and its accuracy statistics exclude them. Evidence artifacts
retain full ensemble responses for 90 days. A short, bounded optional fetch or
failed challenger never blocks public inference. No additional browser API
requests or paid infrastructure are introduced.

## Reproduce and inspect

Using the existing local source caches and forecasting environment:

```sh
.venv/bin/python -m research.forecasting.wind_refinement.prepare
.venv/bin/python -m research.forecasting.wind_refinement.train
.venv/bin/python -m research.forecasting.wind_refinement.evaluate
.venv/bin/python -m research.forecasting.wind_refinement.stress
.venv/bin/python -m forecast.wind_refinement --probe
```

Reproduction overwrites local research outputs; run in a separate checkout if
changing inputs or settings. Published challenger fingerprints reject changed
inference under the same version. Probes do not write public files. The live
step requires the current run's checksummed public wind evidence; it will not
retroactively generate a challenger from a stale record.
