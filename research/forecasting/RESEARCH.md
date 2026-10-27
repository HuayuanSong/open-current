# Weather-driven electricity forecasting: research and candidate selection

Reviewed 27 October 2026 for Open Current. The question is next-day/two-day **power
and load in Danish bidding areas**, not predicting wind speed, weather itself,
electricity prices, or a single solar panel's output. This is a focused literature
review, not an exhaustive systematic review or a claim to reproduce a leaderboard.

## Findings that determine the experiment

**There is no defensible universal winner before testing on our data.** Boosted
trees are credible candidates for weather-to-power conversion; compact neural
models can learn the whole horizon together; foundation models offer a strong
pretrained prior. Their relative value depends on forecast horizon, history,
weather quality, target aggregation and changes in installed capacity.

**Issued weather matters more than the label “state of the art.”** Reanalysis,
observed future temperatures, and artificially perturbed observations are useful
research inputs, but they do not recreate operational forecast errors. We use
archived individual NWP runs with their original initialization times. We cannot
reconstruct historical API publication delays exactly, so availability remains
an explicitly stated assumption, not a guarantee.

**The three targets deserve separate judgments.** Wind has a nonlinear power
curve and geographic weather variation. Solar depends on radiation, daylight and
the expanding PV fleet. Demand has strong calendar effects and temperature
dependence. A single best overall score can hide a failure on one of them.

## Primary evidence and what transfers

| Work | Evidence relevant here | Limits and resulting decision |
|---|---|---|
| [Bruninx et al., 2026, probabilistic wind forecasting](https://arxiv.org/abs/2602.13010) | Compares tree-compatible conformal quantiles, natural gradient boosting and conditional diffusion using four years of Belgian offshore data and multiple weather providers. Explicitly studies weather-to-power conversion and probabilistic accuracy. | Offshore farms have more detailed asset information than our area aggregates. Test quantile boosting first; weather-provider ensembles and diffusion are follow-ups, not assumed improvements in Denmark. |
| [Santos et al., 2026, hybrid PV forecasting](https://doi.org/10.1016/j.solener.2026.114951) | Combines physical predictors with ML for day-ahead PV forecasts at Brazilian and Dutch installations. ECMWF forecasts, rather than target-time observed weather, drive the forecast. | Two installations do not establish national-scale performance. Include simple radiation/temperature conversion features; do not pretend that approximate area coordinates constitute a detailed PV fleet model. |
| [Stüber et al., 2021, PV model comparison](https://doi.org/10.3389/fenrg.2021.639346) | Evaluates physics, feed-forward/LSTM neural models and tree ensembles with NWP in a small-scale PV setting; model families have comparable forecast quality. | Site-specific evidence argues against declaring deep learning or boosting categorically superior. Keep meaningful baselines and a neural comparator. |
| [Das et al., 2023/2024, TiDE](https://arxiv.org/abs/2304.08424) | Dense residual encoder/decoder, projected covariates, temporal decoder and linear history residual support multi-horizon forecasts without requiring attention. | Generic benchmark efficiency does not prove Danish power accuracy. Implement a small **TiDE-inspired adaptation** for origin-specific future weather, with no claim to reproduce original benchmark scores. |
| [Lim et al., 2019/2020, TFT](https://arxiv.org/abs/1912.09363) | Explicitly accommodates static, observed-past and known-future inputs; recurrent and attention components support multi-horizon predictions. | Appropriate architecture, but a larger implementation/tuning burden for our small daily-origin dataset. Defer until a compact deep model establishes value. Attention weights would not establish causality. |
| [Ansari et al., 2025, Chronos-2](https://arxiv.org/abs/2510.15821) | A pretrained forecasting model that natively accepts multivariate targets and covariates, including future inputs; reports strong broad-benchmark results. | Generic leaderboard success is not a power-system guarantee. Evaluate the public checkpoint locally, without fine-tuning or assuming contamination-free pretraining. |
| [Kaljevic et al., October 2026, load under covariate uncertainty](https://arxiv.org/abs/2610.07232) | Compares four foundation models and four trained models on load datasets; explicitly changes future-covariate availability and noise. | Its uncertain-covariate scenario adds Gaussian noise to observations, including ERA5 temperature for CAISO. It motivates testing robustness but is **not an archived-NWP operational benchmark**. |
| [Hoo et al., revised January 2026, TabPFN-TS](https://arxiv.org/abs/2501.02945) | Turns forecasting into tabular regression with temporal features and a pretrained tabular backbone; evaluates covariate-informed tasks. | A worthwhile second foundation-model family. Not included in this first bounded experiment: prioritize one native multivariate foundation comparator plus locally trained statistical/tree/neural models. |
| [Ke et al., 2017, LightGBM](https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html) | Efficient gradient-boosted decision trees, suitable for a compact nonlinear tabular forecast using weather, history and calendar predictors. | General ML evidence, not electricity-specific superiority. Test two compact settings and an otherwise comparable no-weather ablation. |
| [Romano, Patterson and Candès, 2019, conformalized quantile regression](https://arxiv.org/abs/1905.03222) | Combines quantile regression with held-out calibration of prediction intervals. | Classical coverage guarantees need assumptions our overlapping, drifting time series does not satisfy. Report measured interval coverage, not a guaranteed “80% confidence.” |
| [Hong et al., February 2026, US grid neural benchmark](https://arxiv.org/abs/2602.21415) | Compares state-space, transformer and recurrent models on several grids; rankings depend on target and how weather enters the architecture. | The paper explicitly uses observed/reanalysis weather to isolate architecture effects. Treat its weather results as an upper-information experiment, not evidence of deployable NWP skill. |
| [2026 unified NWP solar benchmark](https://doi.org/10.1016/j.egyai.2026.100833) | The accessible abstract reports twelve model comparisons, with recurrent nets and ExtraTrees offering attractive speed/accuracy trade-offs and no blanket transformer advantage. | Full text was inaccessible in this review; do not infer its exact protocol. Recurrent networks and ExtraTrees are reasonable next comparisons, not tested in the first bounded shortlist. |
| [Google Research, TimesFM-3, August 2026](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/) | A newer multivariate foundation model with native future covariates, rather than only the external covariate regression used by version 2.5. | Its [official repository](https://github.com/google-research/timesfm) restricts downloaded 3.0 weights to non-commercial, non-production use. Excluded from the intended public deployment shortlist; earlier 2.5 weights remain Apache-2.0. |

The research also surfaced work on monthly resource prediction and models using
target-time ERA5/reanalysis. Those do not answer this deployment question and are
not used to rank candidates. A potentially relevant ICLR workshop PDF on simple
target–covariate relationships could not be retrieved; its conclusions are not
used as evidence.

## Models actually selected

1. **Persistence, daily and weekly repetition.** Real benchmarks for whether the
   extra complexity helps. Day two repeats the last observed daily profile; it
   never reads tomorrow's actual value as a “lag.”
2. **Ridge with weather and calendar features.** A cheap statistical reference.
   Engineered heating/cooling, approximate wind-power-curve and PV-temperature
   terms let a linear estimator represent useful nonlinear effects.
3. **LightGBM point and quantile forecasts.** Six separate target/area models,
   direct multi-horizon rows with lead-time features. Compare against the same
   family without weather. Fit P10/P90 models and calibrate on separate months.
4. **Compact residual weather MLP.** Train three models pooling the two areas,
   with area-specific training-only normalization. Seven days of target history
   and the issued 48-hour weather/calendar trajectory feed a projected-feature
   encoder/decoder and a per-hour decoder, plus a linear history connection.
   Unlike original TiDE, this version omits past weather from the encoder.
5. **Chronos-2, zero-shot on CPU.** Jointly forecast the three targets for each
   area with past forecast-derived weather and known future weather/calendar.
   Disable cross-origin information sharing: tomorrow's historical context must
   not help today's forecast. Use the official library and pin the checkpoint.
6. **A fixed equal-weight tree/neural blend.** Include as a predeclared candidate;
   choose it only through validation, not by inspecting final-test results.

The two neural approaches answer different questions: whether a small network
trained on Danish data helps, and whether broad pretraining helps without local
training. Running a pretrained checkpoint is not described as training it.

## Data feasibility and integrity

The [Open-Meteo Single Runs API](https://open-meteo.com/en/docs/single-runs-api)
provides initialization-specific ECMWF IFS runs back to March 2024. The experiment
explicitly requests `ecmwf_ifs`, six coordinates and six variables; it does not
use a changing “best match” product. Radiation is averaged over the preceding
hour, so electrical values are aggregated on the same hour-ending convention.
Missing archived runs are recorded and excluded, not replaced by reanalysis.

Energinet's [operational dataset metadata](https://api.energidataservice.dk/meta/dataset/ElectricityProdex5MinRealtime)
warns that SCADA estimates contain errors that generally are not corrected. Its
[settlement metadata](https://api.energidataservice.dk/meta/dataset/ProductionConsumptionSettlement)
states a 9–15-day reporting delay and subsequent revisions. Consequently, finalized
settlement values cannot be treated as live lag features. This experiment uses
operational values for both inputs and targets and explicitly documents source
quality and revision limitations. “Load” includes losses and is derived from the
system balance, not a direct household-consumption measurement.

Open-Meteo publishes current [availability metadata](https://open-meteo.com/en/docs/model-updates),
but initialization is not publication. Our daily forecast follows the 00 UTC run
by twelve hours and includes a 15-minute observation buffer. This conservative
assumption still does not prove every historical run arrived on time. A production
system should archive actual retrieval timestamps and model-run identifiers.

The [free Open-Meteo tier](https://open-meteo.com/en/pricing) permits non-commercial
use without a key, including archive endpoints, within its rate limits. Multiple
locations count toward usage; a bulk HTTP call is not necessarily one billable
call. The backfill caches each response and resumes without refetching successes.
The model/library is separate from its data licence: “open source” is not a synonym
for unlimited free API use.

## Evaluation and deployment reasoning

The [frozen protocol](PROTOCOL.md) specifies temporal splits, label-boundary
purging, hyperparameter selection, independent calibration and an untouched test.
All families use the same eligible electricity/weather origins. Report horizon
and area results, daylight solar errors, ramp errors, quantile scores and coverage.
Resample blocks of forecast days when comparing accuracy; overlapping hourly
labels are not independent experimental observations.

Training and inference are different deployment costs. Half-hourly scoring could
load a saved model, fetch six weather locations plus recent operational history,
and emit small forecast JSON files consumed by a static page. It need not train
from scratch or serve Python in visitors' browsers. Model quality, data freshness
and reliable scheduling remain separate from whether CPU inference fits.

Measurements in the results are from this local Apple Silicon machine with four
CPU threads, not from a GitHub Actions runner. No web deployment is authorized
by this research task. The eventual production decision should also consider
dependency install/download time, model size, data retrieval and the existing
collection job's runtime.

## Search record

Searches covered arXiv, primary publisher pages, NeurIPS proceedings and authors'
official code/model cards. Topics included NWP-to-power comparisons, probabilistic
wind, physical/ML solar hybrids, covariate-informed load forecasting, compact
multi-horizon networks and recent forecasting foundation models. Google Scholar
was attempted directly but was inaccessible through the research tool. No claim
is made that its search results were exhaustively reviewed.

Authoritative implementation references: [Chronos repository](https://github.com/amazon-science/chronos-forecasting),
[Chronos-2 model card](https://huggingface.co/amazon/chronos-2), and the locally
installed source pinned in [requirements.lock.txt](requirements.lock.txt).
