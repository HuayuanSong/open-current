# Regional horizon adjustment, before weather training/evaluation

Source inspection found that archived DMI 06 UTC runs contain 60 timestamps:
06 UTC through run+59h. At the planned 18 UTC electricity origin, the final
48-hour target is run+60h. Requiring 48 complete DMI hours would exclude every
date, although the first 47 forecast hours are usable.

Use DMI for those 47 hours and the already-available **same-issue 06 UTC ECMWF**
forecast for the final hour. Add an explicit native-DMI/source indicator to
the regional and combined-weather models. Never fill an interior gap; missing
archived runs remain exclusions. The ECMWF comparison uses exactly the same
eligible dates and labels. The result is a DMI/ECMWF forecast, not a pure
48-hour DMI forecast. No later run or observed weather is substituted.

This adjustment follows the source horizon, before any new weather-model
training or error inspection. It is not a tuning decision based on scores.
