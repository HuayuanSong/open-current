# Code review — 1 November 2026

Reviewed from `4e2a818362c86515ad1060e1ea32185cb1746c30`. Changes were prepared and
validated locally; this review did not commit, publish, refresh live datasets or
train models.

## Scope and approach

Three focused reviews covered the data pipeline, browser application, and
forecast serving/evaluation. A separate code-simplifier pass followed the fixes.
The review considered correctness, runtime cost, duplication, naming, Python
conventions, repository boundaries and preservation of intended behavior.

Before editing, the review captured tracked-file checksums, published forecast
state, model inference outputs, feature arrays, prediction ranges and browser
screenshots. Regression cases reproduced the confirmed defects before fixes.

## Changes

| Area | Problem | Result |
| --- | --- | --- |
| Country selection | A pending or failed country load could leave the previous country's chart, prices and download active. Late responses could also populate a stale cache. | Clear dependent views, disable downloads until the selected country loads, validate the response country and ignore superseded requests before caching. |
| Refresh recovery | A new index could be accepted before the Denmark response succeeded, causing the next refresh to skip the failed snapshot. | Retrieve both before accepting the snapshot; retry a missing selected-country payload even when the index timestamp is unchanged. |
| Current prices | The displayed price could remain after its market interval ended. | Advance or expire prices with a local timer and recheck when the tab becomes visible. Historical selection stays fixed; no extra API polling. |
| Forecast interaction | An empty comparison could retain pointer/keyboard handlers from the previous outlook. | Detach handlers before the empty-chart return and reset its inspection position. |
| Forecast retries | HTTP-date or malformed `Retry-After` values could fail parsing; long provider delays were shortened to 30 seconds. | Parse numeric/date delays with finite fallback values. Fail the update when the requested wait exceeds the job's 30-second limit, preserving the previous forecast instead of retrying early. |
| Collector retries | A retry could break the configured spacing, including between the last retry and the next country. Exhausted retries could also lose the provider's requested backoff. | Apply endpoint pacing across attempts and countries, retain terminal `Retry-After` delays, and keep Energinet's faster policy separate. |
| Price normalization | Malformed price rows could interrupt unrelated valid generation updates; rejected feeds could replace retained price-interval metadata. | Validate each price response independently, report failures, and preserve both old rows and their original interval. |

The simplifier expanded dense Python and forecast JavaScript, clarified local
names, removed unused imports/state, shared forecast-series serialization and
range checks, and reused the UTC formatter. Numerical operations and their
accumulation order were preserved. A redundant country-chart redraw was removed.

Archive performance scoring now parses issue times and eligibility cutoffs once
per forecast. On a local fixture with 336 archived forecast records, median time
over three runs fell from **91.95 ms to 11.39 ms**, approximately **8.1× faster**,
with identical JSON output. This is a benchmark of that function, not an 8×
improvement to inference, collection or the full scheduled job.

`npm test` now includes both JavaScript test files. Non-scheduled workflow
validation now runs the main explorer journeys, forecast journeys and browser
regressions together through `tests/browser_suite.py`. Asset fingerprints were
rebuilt; no page styling or layout was redesigned.

## Repository structure

The existing boundaries fit this application:

- `scripts/` contains the standard-library electricity collector and normalizer.
- `forecast/` contains serving, model loading, persisted state and evaluation.
- `forecast/trial/` isolates the frozen prospective experiment.
- `research/` retains training and evaluation evidence outside serving.
- `dist/` contains authored static frontend files and generated published data.
- `tests/` exercises data semantics, serving, trial invariants and browser use.

No framework, directory migration or new runtime dependency was necessary.
The collector and forecast request helpers deliberately remain separate: they
use different HTTP libraries, pacing requirements and job budgets. Sharing them
would couple two otherwise independent environments. The small geographic
dataset does not justify a more complex spatial indexing algorithm.

The [main README](../README.md#architecture) now diagrams the source APIs, weather
inputs, model artifacts, inference workers, fixed ensemble, prediction archive,
GitHub Actions and static GitHub Pages frontend.

## Validation

- Python unit and regression suite: **62 passed**, including simulated-clock
  checks for normal pacing, retries, exhaustion, terminal numeric/date backoff,
  subsequent countries and no unnecessary wait after the final key.
- JavaScript model tests: **9 passed**.
- New fixture-backed browser regressions: **8 passed**.
- Existing main and forecast browser journeys passed at widths 390, 768, 1024
  and 1440, with no page errors or external API requests.
- **11 newly captured screenshots were byte-identical** to the corresponding
  pre-change screenshots.
- Real cached CPU inference: every model-family output and the ensemble matched
  the pre-change run **exactly**. The release parity check also passed its
  existing tolerances; the largest release-fixture difference was 0.001221 MW
  in adapted Chronos, already present before this review.
- The historical pipeline replay at `2026-10-27T12:29:32.640278+00:00` produced
  exactly identical feature arrays, metadata, public payload and prediction
  ranges. It selected the same `2026-10-27T12:00:00+00:00` origin.
- Normalizing 15 cached valid price feeds preserved all **10,944 rows** and
  interval values.
- Mermaid rendering and documentation-link checks passed. The asset build is
  idempotent and generated fingerprints match their files.

Model weights, selection weights, calibration artifacts, published datasets and
trial records were not regenerated. All **254 protected tracked files**, including
the **eight frozen inference files**, were kept byte-identical. Numerical
validation used cached artifacts without upstream requests. Local tests cannot
establish the availability or performance
of an upstream provider or a future GitHub-hosted run.

## Additional regression validation

After the initial review, the full **62-test Python suite** also passed in an
isolated **Linux x86-64 container with Python 3.12.14**, using the pinned NumPy,
pandas, holidays, requests and LightGBM versions. The repository was mounted
read-only and network access was disabled during tests. This checks the unit
suite on Linux; it does not reproduce the full GitHub-hosted workflow or rerun
the neural models on Linux.

An offline comparison against the original revision used a fixed random seed
(`20261101`) for **441 exact comparisons**: 80 archive-scoring cases, 80 public
payloads, 45 range fits, 27 range-bound calculations, 80 causal-history cases,
34 cached generation feeds, 33 cached flow feeds, 50 history merges and 12
forecast-serving scenarios. Results, including expected error outcomes, matched.

Browser comparisons at a fixed clock checked **114 settled UI states** against
the original JavaScript: 34 countries across three chart types, plus all six
forecast series with ranges enabled and disabled. Rendered content and SVG
geometry matched exactly, with no page errors or external requests.

Permanent tests now also cover the forecast publication orchestration: duplicate
issues, source/model failures, failed restoration, invalid predictions, the
35-day retention boundary and suppression of ranges for already elapsed hours.
Browser regressions additionally cover mismatched country responses, recovery
without a new index, zero/negative prices and gaps between price intervals.

No further production-code changes were needed in this additional pass. The
changes remain local; the separate trial finding below remains deferred.

## Deferred finding: trial observation provenance

**P2 — a retained reading can be assigned a later capture time after a failed
refresh.** This concerns the separate prospective trial, not the production
ensemble's point forecasts.

[`forecast/trial/run.py`](../forecast/trial/run.py) passes the current processing
time to [`freeze_observations`](../forecast/trial/ledger.py). The function uses
that time for observation-age eligibility and stores it as `capturedAt`.
If a source refresh fails, retained readings can therefore cross the trial's
24-hour minimum age without actually having been retrieved again.

A local reproduction used a target at 27 October 19:00 UTC, a retained source
retrieval at 28 October 18:50 UTC (**23 hours 50 minutes** later), and a trial
tick at 19:07 UTC. The ledger accepted the reading with the tick's timestamp,
claiming **24 hours 7 minutes** of age. Source calls, model issuance and state
writes were mocked; only temporary output was written. This demonstrates the
failure mode, not that an existing published observation was affected.

Repair needs reliable retrieval provenance for the observations being frozen,
plus a decision about how to flag any existing affected evidence. A global
last-refresh timestamp alone is insufficient for rows retained across partial
responses. Changing records silently would conflict with the experiment's
append-only audit rules, so this pass leaves its code, protocol, existing records
and schedule untouched. Review this issue before interpreting the prospective
trial's final results; do not retrospectively rewrite capture times.
