# Weather inputs and forecast adaptation — 27 October 2026

This implements the four requested experiments locally. The deployed v2 model
is the baseline. Earlier experiments have already examined June–October 2026;
these comparisons are **retrospective development evidence**, not a new untouched
holdout. Do not promote a winner automatically or reuse its old interval
calibration. Freeze this protocol before computing new candidate errors.

## 1. Fleet-aware weather

Download and hash the Danish Energy Agency September 2026 wind register. Use
technical fields only; never production columns. Reconstruct active capacity
using connection/disconnection dates. Exclude missing/invalid coordinates and
report missing capacity. Handle Bornholm's UTM33 coordinates separately.
Assign bidding area using municipality codes (west >=400, east <400), with
explicit treatment of offshore records. Cluster within area and land/sea:
DK1 two land + one sea clusters; DK2 one land + two sea clusters. Choose fixed
cluster representatives from turbines connected before January 2025, using
capacity-weighted geography only. Later assets join their nearest existing
cluster. This is a revised register, not an as-published historical vintage;
its retrospective corrections remain a stated limitation.

Fetch archived 00 UTC ECMWF forecasts at these six representative points.
Add capacity-weighted wind speed, direction, spread, generic power-curve MW,
and separate land/sea features to the existing LightGBM representation. Do not
use wind capacity to weight solar or demand weather. Keep their original inputs.

## 2. Newer model runs

Compare 00 and 06 UTC ECMWF inputs at **18:15 UTC issuance** (18 UTC electricity
origin), allowing a conservative 12h15 publication lag for both. This avoids
assuming the 06 run was available at the old 12:15 issue time: observed provider
metadata today reports availability at 12:18:36, before the extra 10min server
consistency buffer. Preserve exact run/valid times and retrieved timestamps.
Build matching 18 UTC histories and labels, and a refitted 00-run control.
Also test 06 UTC DMI HARMONIE at the six original points where its archive is
available (April 2026 onward), with matched training dates and control. Do not
substitute observed weather or stitch newer runs into an older issue.

For both weather studies use two predeclared LightGBM settings: 400 rounds,
15 leaves, min-child 100; and 800 rounds,31 leaves,min-child60. Both L1 loss,
learning rate .04, L2=10, seed42,4CPU threads. Expanding folds train labels
strictly before June1 / August1 / October1. June–July and August–September
select settings; October diagnoses frozen choices. Purge overlapping 48h
labels. Report identical eligible cases for each paired comparison, counts,
MAE,WAPE, lead-band errors, daylight solar error and CPU timings.

## 3. Online error correction

Use historical out-of-sample family predictions at 12 UTC. Compare no correction,
exponentially weighted recent error correction (2/7-day half-life), and ridge
residual prediction with alphas10/100. Correction inputs use only forecast
errors with valid times <=origin−15min and forecasts issued before that valid
time. Never use the current trajectory's future outcomes. Deduplicate overlapping
forecast errors by target time and lead band. Clip corrections to 20% of the
preceding week's mean output/load, and taper their effect with a 12h half-life.
June–July supplies initial correction training; August–September selects;
October evaluates the frozen choice with sequential matured-label updates.

## 4. Conditional ensemble weights

Compare fixed convex weights with a small softmax gate over the nine existing
families, using wind, radiation, temperature, cloud and lead-time conditions.
Learn per-area/target parameters from June–July out-of-sample predictions for
August–September selection; refit on June–September for October. Standardize
using training inputs only; penalize gate coefficients (0.1 / 1.0). Weights
remain nonnegative and sum to one, anchored to the same fixed baseline.
Use June–July-only fixed weights for the development comparison; the deployed
v2 weights used all June–September and therefore cannot be treated as unseen
development predictions. October comparison uses the exact deployed weights.
Evaluate correction, gate, and their combination, retaining failures.

## Reporting and prospective use

All data requests are public and keyless, cached and rate limited. Stop on a
daily rate limit; do not evade provider quotas. Record failures and exclusions.
Tests cover timestamps, commissioning, train/test label separation, future-label
perturbation, overlapping errors, convexity, artifact reload and model outputs.
Keep raw data/weights in ignored cache; commit code, manifests, selections and
reports. Export frozen artifacts suitable for a future shadow evaluation.
Recommend promotion only after new issued forecasts mature; no guaranteed gain
or year-round significance from a small, previously inspected October sample.

Sources: [turbine register](https://ens.dk/analyser-og-statistik/data-oversigt-over-energisektoren),
[single weather runs](https://open-meteo.com/en/docs/single-runs-api),
[availability metadata](https://open-meteo.com/en/docs/model-updates),
[ECMWF dissemination](https://confluence.ecmwf.int/pages/viewpage.action?pageId=540563877).
