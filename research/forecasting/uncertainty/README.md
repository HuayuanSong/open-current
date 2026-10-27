# Prediction ranges and demand explanations

The public panel has a checkbox for experimental 80%-target hourly prediction
ranges. Solar shading is daylight-only. An expandable demand explanation shows
the LightGBM component, explicitly separate from the published ensemble.

## Range results and scope

The existing point models and blend weights are unchanged. Method selection
used expanding August–September validation of archived out-of-fold predictions.
We compared absolute-error, output-scaled and signed-error ranges using interval
score (width plus a penalty for misses), then rolling windows of 14/28/56 days.
Every simulated calibration uses only complete 48-hour outcomes available at
least 15 minutes before that forecast origin. See the original [protocol](PROTOCOL.md)
and the subsequent [rolling follow-up protocol](ROLLING_PROTOCOL.md).

| Region | Target | Published hours | Observed coverage | Mean range width |
| --- | --- | --- | ---: | ---: |
| DK1 | Wind | All | 85.4% | 1,249 MW |
| DK2 | Wind | All | 84.9% | 315 MW |
| DK1 | Solar | Daylight | 82.6% | 529 MW |
| DK2 | Solar | Daylight | 78.7% | 189 MW |
| DK1 | Demand | All | 77.6% | 490 MW |
| DK2 | Demand | All | 76.6% | 185 MW |

These are descriptive checks on 13 October origins per region. Forecast hours
overlap and are not independent trials. The initial fixed calibration covered
only 38.3% / 44.1% of solar outcomes; those failures are retained in `initial/`.
The source reports non-zero overnight solar readings, with errors that the
older calibration substantially underestimated. Rolling calibration improved
the full-day results to 70.2% / 71.8%, but DK2's 7–24-hour band still covered
only 61.1%. We consequently restricted **both** solar displays to daylight.
All daylight lead bands covered 72.6–87.5% of observations. This is a disclosed
post-audit scope restriction, not independent confirmation of the design.

October had already been used to assess point forecasts and then diagnose the
initial interval failure. It is not an untouched test of the follow-up method.
The blend weights themselves were selected on the development predictions.
Neither finite-sample conformal guarantees nor calibrated probabilities for
every hour, season or complete 48-hour path are claimed. Coverage at other
issue times and with filled inputs must be assessed prospectively. The panel
shows measured coverage alongside the 80% target.

The deployed calibration gradually replaces historical out-of-sample errors
with actually issued forecasts. It selects one completed forecast per region
and UTC day closest to 12 UTC, requires at least seven daily origins, and can
extend a short window to 56 days if data are sparse. The selected method/window
does not change automatically. Each range keeps its calibration version,
cutoff, origin count and actual window size. Saved bounds never change after
issue. Old forecasts receive no retrospective bounds. Live coverage excludes
missing observations, unavailable nighttime solar ranges and expired versions.

## SHAP stability results

We audited the refitted original-feature LightGBM component on all 13 October
origins per region at leads 1, 6, 12, 24 and 48: 65 examples per series.
Interventional TreeSHAP was evaluated against five reference samples (32/64/96
rows and three seeds), matched by region and lead band; solar also matched
daylight. References came only from August–September inputs. Feature
contributions were grouped into recent electricity, wind, sunlight/cloud,
temperature and calendar/horizon.

| Series | Worst median attribution change | Worst 90th-percentile change | Public |
| --- | ---: | ---: | --- |
| DK1 wind | 20.3% | 61.3% | No |
| DK2 wind | 19.0% | 55.1% | No |
| DK1 solar | 28.1% | 54.3% | No |
| DK2 solar | 22.0% | 48.4% | No |
| DK1 demand | 12.1% | 18.5% | Yes |
| DK2 demand | 10.5% | 16.5% | Yes |

Changes are normalized by total absolute attribution, with a floor tied to
typical prediction magnitude. Near-baseline cases are excluded from rank/sign
gates; counts are in `shap-audit.json`. Demand had 100% agreement on material
contribution signs and median top-two overlap of 1.0 in every reference
comparison. All six series reconstructed raw model outputs within 0.001 MW.
Solar additionally lacked sufficient matched references for 13 cases per area.
Stable leading groups did not rescue unstable contribution magnitudes for
wind or solar; those explanations remain internal.

We also compared TreeSHAP with exact five-group Shapley calculations that keep
history/wind transformations together and recompute the temperature–radiation
interaction when its inputs are drawn from different groups. Demand's coherent
grouping changed median attributions by 1.55% in DK1 and 0.86% in DK2, with matching leading
groups and material signs. This checks robustness to one important engineered
feature issue; it does not model the true joint conditional distribution or
establish causal effects. Different reference populations answer different
questions. The reference and component are always named on the public panel.

At every live issue, demand explanations are recomputed with 64-row references
and checked against separate 96-row samples. An hour is withheld if its
material signs, top-two groups or attribution magnitude change too much, or
if the explanation fails additive reconstruction. Clipped negative predictions
and references are not explained. Only five audited horizons are offered.

## Reproduce

Run from the repository root, with the original experiment cache present:

```sh
.venv/bin/python research/forecasting/uncertainty/ranges.py select
.venv/bin/python research/forecasting/uncertainty/ranges.py audit
.venv/bin/python research/forecasting/uncertainty/rolling.py select
.venv/bin/python research/forecasting/uncertainty/rolling.py audit
.venv/bin/python research/forecasting/uncertainty/shap_audit.py
.venv/bin/python research/forecasting/uncertainty/export_explanations.py
```

Selection scripts refuse to overwrite different frozen selections. Re-run
`rolling.py audit` last to restore the production calibration after reproducing
the initial failed audit. SHAP 0.54.1 is pinned; other inference dependencies
are in `forecast/requirements-main.txt`. Local detailed attribution arrays
remain under `.cache/forecasting/uncertainty/`; aggregate metrics and all failed
series are preserved here.

References: [adaptive time-series conformal inference](https://arxiv.org/abs/2202.07282),
[dependent-feature Shapley values](https://arxiv.org/abs/1903.10464),
[TreeSHAP](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html).
