# Rolling calibration follow-up

The first fixed-calibration audit failed for solar (38.3% / 44.1% coverage),
mostly overnight. The preliminary source reports substantial non-zero solar
readings at night and that baseline shifts. Physical zero clipping would not
solve disagreement with the measured target. The initial failure is retained.

Before running a second audit, add a trailing calibration window of 14, 28 or
56 days and choose window/method only on expanding August–September validation
using the existing interval score. Keep the other protocol gates unchanged.
Use complete 48-hour forecast outcomes available before the new origin minus
15 minutes. The October check is explicitly a reused diagnostic sample, not
independent confirmation of this follow-up design. Do not tune on its errors.

Only past forecast/outcome pairs enter each rolling calibration. Require at
least 7 distinct daily origins; if the chosen window has fewer, extend to
56 days, and fail closed if still insufficient. Preserve point forecasts.
For deployment, seed with historical out-of-sample residuals and gradually
replace with saved operational forecasts, selecting one origin per UTC day
closest to 12 UTC. Retain 64 days of daily calibration history in published
state, independently of the 35-day full forecast archive. Each issued range
records its cutoff and calibration count. Historical and live coverage remain
separate. Freeze the live range at issuance; never retrofit old records.
