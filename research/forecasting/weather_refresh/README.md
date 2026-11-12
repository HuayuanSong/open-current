# Latest ECMWF weather: implemented, not promoted

The latest-run selector and local inference tool are ready. The unchanged ensemble
did **not** pass the predeclared accuracy gate, so the public forecast and existing
prospective trial retain their daily 00 UTC inputs.

## Completed paired replay

48 complete origins, four per day from 29 October through 9 November 2026, with
48-hour outcomes in both regions. Model weights and all historical inputs were
identical; only future weather changed. Positive reductions mean lower MAE.

| Region | Target | Daily weather MAE (MW) | Latest-run MAE (MW) | Error reduction |
| --- | --- | ---: | ---: | ---: |
| DK1 | Wind | 226.62 | 227.65 | −0.46% |
| DK2 | Wind | 63.04 | 63.42 | −0.60% |
| DK1 | Solar | 78.79 | 75.41 | +4.29% |
| DK2 | Solar | 24.74 | 22.80 | +7.83% |
| DK1 | Demand | 160.00 | 161.20 | −0.75% |
| DK2 | Demand | 66.32 | 66.53 | −0.33% |

DK2 wind at 7–24 hours worsened by **7.79%**, and DK1 wind at 25–48 hours by
**6.89%**. This fails the no-overall-regression and 5% lead-band limits, and does
not deliver the required 5% DK1 wind improvement. The small overall differences
do not establish statistically reliable deterioration; they establish that this
short replay provides no basis for claiming a wind improvement.

This is a retrospective sensitivity check. It assumes an eight-hour publication
lag, uses revised observations, and contains dependent overlapping horizons over
12 days. It does not reproduce historical per-run availability or every live
observation vintage. These data are now inspected development evidence. See the
[frozen protocol](PROTOCOL.md), [machine-readable result](results.json),
[series metrics](metrics.csv), [lead bands](lead-metrics.csv), and
[daily metrics](daily-metrics.csv). No per-target policy was selected afterward.

## Availability-aware implementation

`forecast.weather_runs.refresh` reads Open-Meteo's actual availability metadata,
waits for its ten-minute replication margin, requests an exact ECMWF run, and
validates six locations and the complete 48-hour feature horizon. Successful
downloads are cached separately from the daily historical weather. Failed or
partial downloads preserve earlier complete runs and remain retryable.

`forecast.data.inputs(..., latest_weather=True)` opts into that cache and falls
back to eligible daily weather if a complete newer run is unavailable. The
default call is unchanged. Production and the frozen trial use the default;
neither calls the latest-run downloader. Trial hashes, weights, issued records,
ranges and explanations have not been changed.

The live probe on 12 November successfully downloaded and scored that day's
00 UTC run, versus 11 November 00 UTC in the baseline. That confirms access to
fresher inputs; it does not establish improved forecast accuracy. Validation
passed 87 Python tests, 13 JavaScript tests, production model export parity and
frozen trial parity. Default input arrays and metadata also matched the prior
implementation at five archived issuance times, and all eight frozen trial
source hashes remained unchanged.

## Local use

Download a recent public `data/forecast-state.json.gz` into an ignored cache.
The state should be less than two hours old for a live probe. With the existing
forecast environments installed:

```sh
.venv/bin/python -m research.forecasting.weather_refresh.probe \
  --state .cache/forecast-audit-20261112/forecast-state.json.gz --score

.venv/bin/python -m research.forecasting.weather_refresh.evaluate \
  --state .cache/forecast-audit-20261112/forecast-state.json.gz
```

The probe writes only `.cache/weather-refresh/live/`, including an explicitly
diagnostic comparison. It neither issues a production forecast nor updates the
trial. The replay resumes cached downloads and scoring batches whose input,
protocol, model release and scoring-source fingerprints still match.

Before future public promotion, test a model trained for the fresher weather
policy on new evaluation data, then recheck prediction ranges and explanations.
Keep the existing prospective trial's December review and frozen comparison intact.

Sources: [model availability](https://open-meteo.com/en/docs/model-updates),
[exact model runs](https://open-meteo.com/en/docs/single-runs-api).
