# Implementation notes

- The turbine workbook has reversed `Kommune` / `Type af placering` headings.
  Values are validated before repairing the headings. Seven duplicate technical
  rows are removed. Two small relocated turbines have distinct historical rows
  but lack usable coordinates and are excluded, rather than arbitrarily chosen.
  Bornholm municipality 400 is explicitly assigned to DK2 and UTM33.
- Initial timestamp tests caught pandas 3 retaining microsecond-resolution
  DatetimeIndex values while Timestamp.value returns nanoseconds. Initial error
  features were consequently empty. Normalize to nanoseconds explicitly and
  rerun both selection and evaluation without changing candidate settings.
  Invalid outputs are retained under the ignored cache's `invalid-timestamps/`
  folder; they are not experimental evidence. Gate-only outputs did not use
  this function, but the entire adaptation pipeline is rerun consistently.
- A final geographic audit caught municipality code `0` on 51 new North Sea
  offshore turbine records. A simple numerical municipality split had incorrectly
  assigned them to DK2. Their coordinates place them west of 9E, in the Danish
  North Sea fleet connected to DK1. Add an explicit rule and regression test;
  fail on any other unmapped code-0 record. Discard and archive the initial
  weather experiment outputs under `invalid-offshore-area/` and rerun all
  weather selection/evaluation with the unchanged hyperparameter grid. The
  six weather points remain identical because clustering uses pre-2025 assets.
  Previously reported preliminary fleet development scores are superseded.
