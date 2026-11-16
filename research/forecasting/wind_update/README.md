# Fresher weather and short-term wind correction

Implemented 16 November 2026. **This is a separate comparison, not a replacement
for the public forecast.** It uses the latest availability-verified ECMWF run for
the refitted ensemble components, then tests a correction from recent five-minute
wind-production measurements. Solar and demand remain exactly the production
values. The correction changes only the first six wind hours.

## Diagnostic results

The correction strength, decay and trend weight were selected using 1–8 November
examples whose six-hour outcomes precede 9 November. The fixed recipe was then
compared on 16 origins from 10–13 November, covering outcomes through 15 November.
The current-day miss was not used for tuning. These are development diagnostics:
the project has already inspected November outcomes, archived observations can
be revised, weather availability assumes an eight-hour lag, and overlapping
48-hour trajectories are dependent. This is not independent prospective evidence.

| Wind MAE, MW | DK1 | DK2 |
| --- | ---: | ---: |
| Original ensemble, all 48 hours | 722.66 | 115.75 |
| Fresh-weather refit, wind only | 623.62 | 101.51 |
| Fresh weather plus correction | 622.23 | 100.58 |
| Original ensemble, first six hours | 233.90 | 70.48 |
| Combined candidate, first six hours | 216.46 | 59.89 |

The combined recipe reduced overall error by approximately **13.9% / 13.1%** and
first-six-hour error by **7.5% / 15.0%** in DK1 / DK2. Every reported lead band
improved in this short comparison. This does not establish year-round gains.

Candidate ranges were calibrated on 28 development origins whose full outcomes
precede the later evaluation period. Their 80%-target coverage was only
**36.1% / 54.2%**. This failed range check is a reason not to promote the candidate
or show its bands as validated public ranges. The current dashboard also shows
a warning when its own recent coverage falls below 75%, with at least 24 scored
forecast hours across three issued outlooks. The displayed target remains 80%.

Full results: [results.json](results.json), selected correction and range rules:
[selection.json](selection.json), evaluation contract: [protocol.json](protocol.json).

## Operational comparison

The earlier refit collector recorded one valid origin in its first ten attempts;
the other attempts were outside its narrow issue windows. This comparison runs
after the normal production refresh, including watchdog-triggered refreshes,
and records the first valid comparison for each available source hour. It does
not modify or backfill either existing trial. GitHub scheduling still has no
latency guarantee, and missed or late comparisons remain missing.

- Read the newest provider-available ECMWF run with a ten-minute replication
  margin. Require all six locations and the complete forecast horizon; unavailable
  fresh inputs leave the candidate unavailable instead of silently using old data.
- Reuse production model outputs only when their input arrays match exactly and
  all worker files were written after those inputs. Refitted components see the
  fresh weather; the remaining members retain their original daily inputs.
- Use two complete 15-minute wind means, keeping ten minutes after the last
  five-minute interval ends for publication. Missing components, gaps, conflicting
  readings and stale sources invalidate the correction. No future readings or
  backward filling enter features. The trend contribution is bounded and decays.
- Store all three forecasts (baseline, fresh wind, corrected wind), original
  ranges, source times, hashes and raw input artifacts before the first target
  hour ends. Duplicates preserve the first issue. Diagnostic probes cannot write
  to the production archive or contribute prospective scores.
- Freeze each area's first valid outcome captured 24–72 hours later. Report only
  complete paired trajectories, with separate point-error and range diagnostics.
  Inference fingerprints prevent a changed candidate from silently continuing
  the same experiment.

Records persist under `windUpdate` in the existing published forecast state;
the current status is [data/wind-update.json](https://huayuansong.github.io/open-current/data/wind-update.json).
Raw evidence is retained in GitHub Actions artifacts for 90 days. Candidate
failures do not block publication of ordinary observations or forecasts.

The fixed collection period is 16 November–14 December, with review after
20 December 00:00 UTC. Minimum evidence is 80 complete origins across 21 days;
point-error, lead-band, coverage and interval-score checks are separate. These
checks are necessary, not sufficient, and **there is no automatic promotion**.
If range calibration still fails, a new range candidate needs its own evaluation.

## Reproduce

The replay requires a current state snapshot at `.cache/wind-update/state.json.gz`
and the timestamped raw electricity download at `.cache/wind-update/electricity.json.gz`.
It reuses the earlier refit development predictions and checksummed model bundle.
Cached replay batches resume only when the input, protocol and model hashes match.

```sh
.venv/bin/python -m research.forecasting.wind_update.evaluate
.venv/bin/python -m unittest tests.test_wind_update -v
.venv/bin/python -m forecast.wind_update --probe --state .cache/wind-update/state.json.gz
```

The last command writes only to `.cache/wind-update/live/probe/`. It requires a
recent source hour whose first target has not yet ended. Scheduled production
uses `python -m forecast.wind_update` after restoring the current public archive.
