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
last available training label on October 21. The 18 November revision requires
24 observed wind hours in both areas. Of 1,070 attempted origins, 977 met the
input criteria and 93 were excluded. The final selections were 15-leaf LightGBM
for West and ridge regression with alpha 100 for East. Earlier issued research
predictions retain their version and are evaluated separately.

| Region | Current 1–6h MAE | Challenger MAE | Reduction | Checks over 16 origins, 10–13 November |
| --- | ---: | ---: | ---: | --- |
| West / DK1 | 214.89 MW | 209.40 MW | 2.55% | Below the minimum gain and lead criteria |
| East / DK2 | 59.89 MW | 50.41 MW | 15.82% | Passed the aggregate criterion and missed the first-hour criterion |

First-hour East error rose from 15.21 to 28.28 MW. On two later November 14 stress
origins, West error fell from 243.47 to 193.32 MW and from 421.51 to 234.52 MW.
East error rose from 100.19 to 134.98 MW and from 83.24 to 107.82 MW.
The challenger continues as an internal prospective experiment.

**These are retrospective results on previously inspected November dates.**
Revised electricity observations and assumed historical weather release lags
are used. Overlapping forecast hours are dependent. A new model is not promoted
publicly on this evidence alone. The saved exports are parity-checked against
their training models; scoring is CPU-only, with no scheduled retraining.

## West Denmark first-hour trial

The first wind forecast for an hour is issued while that hour is underway. This
prospective comparison covers the 336 West Denmark hours beginning **18 November
2026 at 14:00 UTC** through **2 December at 14:00 UTC**, exclusive of the final
start time. Each hour contributes at most one case. Both alternatives use the
electricity snapshot archived with the first public DK1 wind issue for that hour.

| Estimate | Calculation |
| --- | --- |
| Public model | The first-hour wind value published by `fresh-wind-2026-11-18`. |
| 15-minute hold | The mean production in the three most recent eligible five-minute intervals. |
| Observed intervals + hold | Average the hour's twelve intervals, using available onshore plus offshore readings and filling the other intervals with the hold value. |

The public model's ten-minute observation buffer applies to both alternatives.
The trial verifies the snapshot checksum and the original hold level, then records
the predictions before the target hour closes. Missing hours remain visible in
the coverage count. The first complete DK1 hourly observation retrieved 15 minutes
to 72 hours after its end is frozen for all three estimates. A later revision
cannot change a recorded result.

The first refresh on or after **2 December at 15:00 UTC (16:00 Copenhagen)** saves
the review using outcomes retrieved by that deadline. It reports paired mean
absolute error in MW, missed and unscored hours, and errors for each daily block.
The result is descriptive and any production change requires a separate decision.
Every refresh includes a progress summary. The review run also provides the
`west-first-hour-review` JSON artifact for 90 days and the archived
`westFirstHourTrial` ledger retains the results. [Refresh runs](https://github.com/HuayuanSong/open-current/actions/workflows/pages.yml)

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
