# Forecast inputs and adaptation: local results

All four experiments were implemented and run with real public data. The
deployed v2 ensemble is preserved. These are retrospective comparisons:
June–October 2026 was already inspected during previous research. Hyperparameter
selection uses pre-October labels, but October is **not a fresh independent
holdout** for the project. Overlapping 48-hour windows are not independent.

Read [protocol](PROTOCOL.md), [implementation notes](IMPLEMENTATION_NOTES.md),
and [reproduction instructions](README.md). Positive improvement means lower
MAE; negative means worse. WAPE and lead-band results are in the CSVs. MAE is
in MW and is not a percentage accuracy score.

## Findings

- Capacity-aware wind inputs reduced noon DK1 MAE from **345.69 to 327.97 MW**
  versus the deployed ensemble (5.1%); DK2 became worse (5.9%). This is not a
  universal benefit from capacity weighting.
- Newer ECMWF weather plus capacity features at evening origins reduced wind
  error by **8.6% in DK1 and 7.5% in DK2**. The same newer-weather pipeline
  reduced demand error by **6.1% and 2.2%**. Capacity inputs only affect wind.
- Combined ECMWF/DMI inputs reduced evening solar error by **13.3% and 10.5%**
  versus the deployed ensemble. Against matched shorter-history ECMWF tree
  controls, the reductions are smaller; use the regional table below to isolate
  the additional weather source. DMI alone performed poorly on wind.
- Online correction reduced demand error by about **0.6%** in both areas but
  worsened DK2 wind. Adaptive weights gave small mixed changes, including a
  1.5% DK2 wind improvement and worse DK2 solar/demand. Neither is a broad win.

These are observations about tested candidates, not an independently verified
selection of the best future model. **Prioritize weather-input challengers for
a prospective trial; keep the current public model until that trial supplies
new evidence.** Freeze the candidate specification before those outcomes
arrive. Selecting different winners from these October results would require
another independent evaluation period.

## Data and comparison scope

- Turbine register: 4,922 usable records; 7
  duplicate technical records removed, 2,250 excluded for
  unavailable/invalid required fields (35.78 MW of
  known capacity, including retired records). This is a revised September
  register, not an archive of what was published on each historical date.
  Municipality-zero North Sea offshore records are explicitly mapped to DK1.
  Connection dates do not establish when a registry update became public;
  this source-vintage limitation requires prospective validation.
- Geography uses only pre-January-2025 active turbines to choose six fixed
  representatives. Later capacity is assigned to the same clusters. Land and
  offshore wind stay separate. No production columns are used; solar/demand
  retain their original location sampling.
- Matched electricity eligibility: 469 noon dates and
  466 evening dates per area across the full dataset.
  The regional comparison has 125 complete dates per area.
  Input exclusions are in weather-exclusions.csv; unavailable archived runs
  are recorded separately in source-availability.json.
- Evening experiments compare 00 and 06 weather runs at 18:15 issuance. A
  conservative 12h15 publication assumption is used in replay. A separate
  tested live selector uses actual availability metadata plus ten minutes.
- DMI comparisons have shorter matched histories from April 2026 onward.
  The combined model receives both ECMWF and DMI fields. Missing DMI runs are
  not replaced with later runs or observed weather. DMI covers the first 47
  hours; the terminal hour uses the already-available same-issue ECMWF forecast
  and an explicit source indicator. See [horizon addendum](REGIONAL_ADDENDUM.md).
- Corrections use only matured earlier labels; overlapping errors are
  deduplicated by target timestamp and lead band. Daily historical issues are
  a limited test of a correction intended eventually for hourly operation.
  Base-family settings were already selected in v2 using the development
  periods, so this is not fully nested validation of the entire model stack.
  Results are conditional on complete input histories; missing-source periods
  may differ from the dates retained in the comparison.

## fleet

| area | target | variant | origins | mae_mw | improvement_vs_control_pct |
| --- | --- | --- | --- | --- | --- |
| DK1 | wind | control | 13 | 408.72 | 0.00 |
| DK2 | wind | control | 13 | 94.86 | 0.00 |
| DK1 | wind | fleet | 13 | 327.97 | 19.76 |
| DK2 | wind | fleet | 13 | 96.56 | -1.80 |

## evening

| area | target | variant | origins | mae_mw | improvement_vs_control_pct |
| --- | --- | --- | --- | --- | --- |
| DK1 | wind | control | 13 | 420.62 | 0.00 |
| DK1 | solar | control | 13 | 85.79 | 0.00 |
| DK1 | load | control | 13 | 211.46 | 0.00 |
| DK2 | wind | control | 13 | 105.42 | 0.00 |
| DK2 | solar | control | 13 | 36.20 | 0.00 |
| DK2 | load | control | 13 | 72.98 | 0.00 |
| DK1 | wind | fresh | 13 | 395.34 | 6.01 |
| DK1 | solar | fresh | 13 | 87.68 | -2.21 |
| DK1 | load | fresh | 13 | 200.10 | 5.37 |
| DK2 | wind | fresh | 13 | 88.70 | 15.86 |
| DK2 | solar | fresh | 13 | 33.79 | 6.67 |
| DK2 | load | fresh | 13 | 72.81 | 0.23 |
| DK1 | wind | fresh_fleet | 13 | 349.40 | 16.93 |
| DK1 | solar | fresh_fleet | 13 | 87.68 | -2.21 |
| DK1 | load | fresh_fleet | 13 | 200.10 | 5.37 |
| DK2 | wind | fresh_fleet | 13 | 89.04 | 15.54 |
| DK2 | solar | fresh_fleet | 13 | 33.79 | 6.67 |
| DK2 | load | fresh_fleet | 13 | 72.81 | 0.23 |

## regional

| area | target | variant | origins | mae_mw | improvement_vs_control_pct |
| --- | --- | --- | --- | --- | --- |
| DK1 | wind | control | 13 | 389.15 | 0.00 |
| DK1 | solar | control | 13 | 85.41 | 0.00 |
| DK1 | load | control | 13 | 217.42 | 0.00 |
| DK2 | wind | control | 13 | 101.10 | 0.00 |
| DK2 | solar | control | 13 | 33.62 | 0.00 |
| DK2 | load | control | 13 | 80.04 | 0.00 |
| DK1 | wind | regional | 13 | 520.20 | -33.68 |
| DK1 | solar | regional | 13 | 93.50 | -9.48 |
| DK1 | load | regional | 13 | 225.62 | -3.77 |
| DK2 | wind | regional | 13 | 127.90 | -26.51 |
| DK2 | solar | regional | 13 | 32.80 | 2.41 |
| DK2 | load | regional | 13 | 85.28 | -6.55 |
| DK1 | wind | multi_weather | 13 | 403.00 | -3.56 |
| DK1 | solar | multi_weather | 13 | 80.69 | 5.52 |
| DK1 | load | multi_weather | 13 | 226.68 | -4.26 |
| DK2 | wind | multi_weather | 13 | 90.25 | 10.73 |
| DK2 | solar | multi_weather | 13 | 31.18 | 7.26 |
| DK2 | load | multi_weather | 13 | 86.06 | -7.53 |

## adapt

| area | target | candidate | origins | mae_mw | improvement_pct |
| --- | --- | --- | --- | --- | --- |
| DK1 | wind | fixed | 13 | 345.69 | 0.00 |
| DK1 | wind | combined | 13 | 344.00 | 0.49 |
| DK1 | wind | selected_correction | 13 | 345.69 | 0.00 |
| DK1 | wind | selected_gate | 13 | 344.00 | 0.49 |
| DK1 | solar | fixed | 13 | 81.40 | 0.00 |
| DK1 | solar | combined | 13 | 81.38 | 0.02 |
| DK1 | solar | selected_correction | 13 | 81.40 | 0.00 |
| DK1 | solar | selected_gate | 13 | 81.38 | 0.02 |
| DK1 | load | fixed | 13 | 181.19 | 0.00 |
| DK1 | load | combined | 13 | 179.83 | 0.75 |
| DK1 | load | selected_correction | 13 | 180.01 | 0.65 |
| DK1 | load | selected_gate | 13 | 180.99 | 0.11 |
| DK2 | wind | fixed | 13 | 91.16 | 0.00 |
| DK2 | wind | combined | 13 | 90.56 | 0.66 |
| DK2 | wind | selected_correction | 13 | 92.39 | -1.35 |
| DK2 | wind | selected_gate | 13 | 89.81 | 1.48 |
| DK2 | solar | fixed | 13 | 31.63 | 0.00 |
| DK2 | solar | combined | 13 | 31.91 | -0.89 |
| DK2 | solar | selected_correction | 13 | 31.63 | 0.00 |
| DK2 | solar | selected_gate | 13 | 31.91 | -0.89 |
| DK2 | load | fixed | 13 | 65.77 | 0.00 |
| DK2 | load | combined | 13 | 65.50 | 0.40 |
| DK2 | load | selected_correction | 13 | 65.38 | 0.59 |
| DK2 | load | selected_gate | 13 | 65.89 | -0.19 |

## Comparison with the actual deployed ensemble

Weather tables above isolate input changes against matched refitted tree
controls. This table compares them with the deployed nine-family ensemble,
which is a stronger baseline. Evening ensemble predictions were recomputed
using the real deployed artifacts and matching evening histories. The deployed
models were trained at noon origins; evening tree controls were refitted at
evening origins. This is a practical pipeline comparison, not an architecture
comparison. No winner was selected using these October scores.

| study | area | target | candidate | origins | baseline_mae_mw | candidate_mae_mw | improvement_pct | origins_improved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fleet | DK1 | wind | control | 13 | 345.69 | 408.72 | -18.23 | 3 |
| fleet | DK2 | wind | control | 13 | 91.16 | 94.86 | -4.06 | 5 |
| fleet | DK1 | wind | fleet | 13 | 345.69 | 327.97 | 5.13 | 7 |
| fleet | DK2 | wind | fleet | 13 | 91.16 | 96.56 | -5.93 | 5 |
| evening | DK1 | wind | control | 13 | 382.16 | 420.62 | -10.06 | 4 |
| evening | DK1 | solar | control | 13 | 93.04 | 85.79 | 7.79 | 9 |
| evening | DK1 | load | control | 13 | 213.12 | 211.46 | 0.78 | 7 |
| evening | DK2 | wind | control | 13 | 96.28 | 105.42 | -9.49 | 5 |
| evening | DK2 | solar | control | 13 | 34.84 | 36.20 | -3.91 | 5 |
| evening | DK2 | load | control | 13 | 74.47 | 72.98 | 2.00 | 9 |
| evening | DK1 | wind | fresh | 13 | 382.16 | 395.34 | -3.45 | 5 |
| evening | DK1 | solar | fresh | 13 | 93.04 | 87.68 | 5.76 | 11 |
| evening | DK1 | load | fresh | 13 | 213.12 | 200.10 | 6.11 | 10 |
| evening | DK2 | wind | fresh | 13 | 96.28 | 88.70 | 7.87 | 8 |
| evening | DK2 | solar | fresh | 13 | 34.84 | 33.79 | 3.02 | 8 |
| evening | DK2 | load | fresh | 13 | 74.47 | 72.81 | 2.22 | 8 |
| evening | DK1 | wind | fresh_fleet | 13 | 382.16 | 349.40 | 8.57 | 8 |
| evening | DK1 | solar | fresh_fleet | 13 | 93.04 | 87.68 | 5.76 | 11 |
| evening | DK1 | load | fresh_fleet | 13 | 213.12 | 200.10 | 6.11 | 10 |
| evening | DK2 | wind | fresh_fleet | 13 | 96.28 | 89.04 | 7.52 | 7 |
| evening | DK2 | solar | fresh_fleet | 13 | 34.84 | 33.79 | 3.02 | 8 |
| evening | DK2 | load | fresh_fleet | 13 | 74.47 | 72.81 | 2.22 | 8 |
| regional | DK1 | wind | control | 13 | 382.16 | 389.15 | -1.83 | 5 |
| regional | DK1 | solar | control | 13 | 93.04 | 85.41 | 8.20 | 10 |
| regional | DK1 | load | control | 13 | 213.12 | 217.42 | -2.02 | 5 |
| regional | DK2 | wind | control | 13 | 96.28 | 101.10 | -5.00 | 5 |
| regional | DK2 | solar | control | 13 | 34.84 | 33.62 | 3.52 | 9 |
| regional | DK2 | load | control | 13 | 74.47 | 80.04 | -7.48 | 7 |
| regional | DK1 | wind | regional | 13 | 382.16 | 520.20 | -36.12 | 1 |
| regional | DK1 | solar | regional | 13 | 93.04 | 93.50 | -0.50 | 6 |
| regional | DK1 | load | regional | 13 | 213.12 | 225.62 | -5.87 | 5 |
| regional | DK2 | wind | regional | 13 | 96.28 | 127.90 | -32.83 | 2 |
| regional | DK2 | solar | regional | 13 | 34.84 | 32.80 | 5.84 | 6 |
| regional | DK2 | load | regional | 13 | 74.47 | 85.28 | -14.52 | 7 |
| regional | DK1 | wind | multi_weather | 13 | 382.16 | 403.00 | -5.45 | 4 |
| regional | DK1 | solar | multi_weather | 13 | 93.04 | 80.69 | 13.27 | 9 |
| regional | DK1 | load | multi_weather | 13 | 213.12 | 226.68 | -6.36 | 3 |
| regional | DK2 | wind | multi_weather | 13 | 96.28 | 90.25 | 6.27 | 7 |
| regional | DK2 | solar | multi_weather | 13 | 34.84 | 31.18 | 10.52 | 10 |
| regional | DK2 | load | multi_weather | 13 | 74.47 | 86.06 | -15.57 | 6 |
| adapt | DK1 | wind | selected_correction | 13 | 345.69 | 345.69 | 0.00 | 0 |
| adapt | DK1 | solar | selected_correction | 13 | 81.40 | 81.40 | 0.00 | 0 |
| adapt | DK1 | load | selected_correction | 13 | 181.19 | 180.01 | 0.65 | 9 |
| adapt | DK2 | wind | selected_correction | 13 | 91.16 | 92.39 | -1.35 | 2 |
| adapt | DK2 | solar | selected_correction | 13 | 31.63 | 31.63 | 0.00 | 0 |
| adapt | DK2 | load | selected_correction | 13 | 65.77 | 65.38 | 0.59 | 10 |
| adapt | DK1 | wind | selected_gate | 13 | 345.69 | 344.00 | 0.49 | 8 |
| adapt | DK1 | solar | selected_gate | 13 | 81.40 | 81.38 | 0.02 | 6 |
| adapt | DK1 | load | selected_gate | 13 | 181.19 | 180.99 | 0.11 | 11 |
| adapt | DK2 | wind | selected_gate | 13 | 91.16 | 89.81 | 1.48 | 10 |
| adapt | DK2 | solar | selected_gate | 13 | 31.63 | 31.91 | -0.89 | 4 |
| adapt | DK2 | load | selected_gate | 13 | 65.77 | 65.89 | -0.19 | 5 |
| adapt | DK1 | wind | combined | 13 | 345.69 | 344.00 | 0.49 | 8 |
| adapt | DK1 | solar | combined | 13 | 81.40 | 81.38 | 0.02 | 6 |
| adapt | DK1 | load | combined | 13 | 181.19 | 179.83 | 0.75 | 9 |
| adapt | DK2 | wind | combined | 13 | 91.16 | 90.56 | 0.66 | 8 |
| adapt | DK2 | solar | combined | 13 | 31.63 | 31.91 | -0.89 | 4 |
| adapt | DK2 | load | combined | 13 | 65.77 | 65.50 | 0.40 | 6 |

## Measured local CPU cost

| stage | seconds |
| --- | --- |
| adapt-evaluate-runtime | 0.63 |
| evening-select-runtime | 222.89 |
| baseline-evening-runtime | 16.47 |
| regional-evaluate-runtime | 35.28 |
| adapt-select-runtime | 0.75 |
| evening-evaluate-runtime | 35.59 |
| fleet-select-runtime | 126.98 |
| regional-select-runtime | 163.76 |
| fleet-evaluate-runtime | 11.06 |

Training ran on the local Mac, sometimes concurrently; figures are elapsed
times, not isolated hardware benchmarks. Downloads and environment setup are
separate. New tree and gate inference requires CPU only. All model weights,
source hashes, selections and origin-level errors are retained locally.

## Release decision

No automatic promotion from these reused historical periods. The prospective
step is to freeze a challenger and record its issued predictions alongside
the deployed model, then score newly matured outcomes. A point-forecast change
also requires new range calibration and a renewed explanation audit. Existing
prediction ranges must not be attached to a different point model unchanged.
