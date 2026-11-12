# Component refit: diagnostic results

Completed locally on 12 November 2026. **The public forecast is unchanged.**
The latest-weather component refit improves overall MAE in all six series in
this short historical diagnostic. This is encouraging development evidence,
not an independent validation or authorization to promote the candidate.

## Fitting and coverage

The fixed data preparation retained **1,070 matched origins**, each with both
DK1 and DK2: 2,140 regional examples. It considered 1,334 origins from January
2025 through October 2026. Of 264 exclusions, 45 lacked cached weather after
the provider rate limit; the other 219 failed history, outcome or weather-horizon
checks. Only ten October origins remained available per region, so recent
training coverage is thin. See [coverage](coverage.csv), [exclusions](exclusions.csv),
[data audit](data-audit.json) and the [acquisition note](ACQUISITION.md).

The 96 fixed-setting development/final fits took **233.6 seconds of fitting time**
on the local Mac, excluding data acquisition, preparation and evaluation. There
are 48 exported final models: four families, three targets and two regions under
two weather policies. Export/reload checks passed. The bundle includes a fixed
prediction fixture for checking the Linux runner against the local exports.

Validation passed **99 Python tests and 13 JavaScript tests**. On GitHub's Linux
runner, all eight policy/family prediction arrays matched the Mac fixture
exactly (maximum difference **0 MW**). A live diagnostic fetched current
electricity observations and the 12 November 06 UTC ECMWF run, then scored both
refits alongside the baseline. This diagnostic is excluded from prospective
evaluation. See the [successful workflow run](https://github.com/HuayuanSong/open-current/actions/runs/43221644598).
The [normal deployment checks](https://github.com/HuayuanSong/open-current/actions/runs/43221387900)
also passed; the published model remains `ensemble-v2-2026-10-27`.

## November full-ensemble diagnostic

36 complete origins at 03, 09, 15 and 21 UTC on November 1–9; 48-hour outcomes
give 1,728 scored hours per region/target/variant. These overlapping forecasts
are dependent. The same electricity inputs and outcomes are used for all three
variants; original ensemble weights and non-refitted members remain fixed.

| Region | Target | Current MAE (MW) | Daily-policy refit | Latest-weather refit | Latest reduction vs current |
| --- | --- | ---: | ---: | ---: | ---: |
| DK1 | Wind | 224.83 | 231.44 | 220.11 | 2.10% |
| DK2 | Wind | 65.25 | 66.68 | 60.60 | 7.13% |
| DK1 | Solar | 88.01 | 84.31 | 83.45 | 5.18% |
| DK2 | Solar | 24.50 | 23.71 | 22.70 | 7.35% |
| DK1 | Demand | 164.80 | 160.78 | 161.02 | 2.30% |
| DK2 | Demand | 66.44 | 67.21 | 66.23 | 0.31% |

Against the matched daily-policy refit, fresher weather reduced wind MAE by
**4.90% / 9.12%** in DK1 / DK2. Daily-policy refitting alone increased wind
error in this diagnostic, so the weather-policy control matters. For DK1 demand,
the latest-weather refit was slightly worse than the daily refit (0.15%).

The worst latest-refit lead-band increase versus the deployed model was **0.28%**
for DK2 demand at 25–48 hours. DK1 wind at 25–48 hours was essentially unchanged.
These small differences do not establish statistically reliable improvement
or deterioration. See [series metrics](diagnostic-metrics.csv),
[lead-band metrics](diagnostic-lead-metrics.csv), [evaluation audit](diagnostic-audit.json),
and [component development metrics](development-metrics.csv).

## Decision

Keep the public model, ranges, explanations and existing trial unchanged.
DK1 wind's 2.10% diagnostic reduction is below the prospective protocol's 5%
requirement, and these inspected historical observations cannot serve as the
new holdout. No settings or per-target choices were changed after this result.

The separate prospective comparison records November 13–December 10 forecasts
before their outcomes, using measured source availability. Final review is
after December 15 at 21:00 UTC, subject to sufficient complete evidence. Its
collector and review are automatic; model promotion is not. The existing trial
retains its original December review.
