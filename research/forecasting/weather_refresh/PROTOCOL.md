# Latest ECMWF weather: paired replay

Frozen before scoring on 12 November 2026. Compare the existing daily 00 UTC
weather selection with the most recent six-hourly ECMWF run. Model weights,
ensemble weights, electricity history and historical weather features are identical.
No fitting or per-target selection is allowed in this comparison.

Use 48 fixed origins: 03, 09, 15 and 21 UTC on 29 October–9 November inclusive,
with issuance 30 minutes later and a complete 48-hour outcome in each region.
These hours avoid assuming instantaneous availability at a model initialization
boundary. The latest-run replay assumes an eight-hour publication/replication
lag. Historical per-run availability timestamps are unavailable, so this is a
retrospective input sensitivity check, **not** a verified live-availability backtest.
Production selection must use actual availability metadata plus ten minutes and
verify the downloaded horizon. Missing replay runs are reported, never replaced
with a later run or observed weather.

The public state snapshot supplies revised historical electricity observations.
Both alternatives use the same history, restricted to each origin, and the same
outcomes. This does not reconstruct every observation vintage originally seen
by a live job. Only fully completed paired trajectories enter the headline scores.
Overlapping windows are dependent; the sample spans fewer than two weeks.

The gate requires at least 40 complete origins, at least 5% lower DK1 wind MAE,
no increase in overall MAE for any of the six area/target series, and no increase
over 5% in any 1–6, 7–24 or 25–48-hour lead band. Missing or non-finite inputs
cannot pass. Report MAE, WAPE, signed bias, daily results and sample counts.
Failure means retain the public baseline; do not tune on these outcomes or
silently choose a different weather policy per target. Passing is necessary but
not sufficient for promotion: prediction ranges and public validation claims also
need to remain valid for the changed input policy.

The existing prospective trial, its fixed baseline, model weights, issue records,
observation-freezing policy and inference lock remain unchanged.

Run from the repository root:

```sh
.venv/bin/python -m research.forecasting.weather_refresh.evaluate \
  --state .cache/forecast-audit-20261112/forecast-state.json.gz
```

Downloads and intermediate arrays are resumable under `.cache/weather-refresh/`.
The report records the protocol and state hashes. Failed batches stop execution;
completed batches are reused only when their input hashes match.
