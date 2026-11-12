# Source-inventory freeze before fitting

On 12 November, the archive backfill encountered transport timeouts and then
an HTTP 429 rate-limit response. Bulk requests were stopped. No model fitting
or error-based selection had begun. The available source inventory is frozen
for this experiment using `prepare --cached-only`.

Every planned date is still considered. A pair is admitted only when both
weather policies have complete archived horizons and the electricity history
and all labels pass the predefined checks. Missing runs are explicitly listed
in `exclusions.csv`; no observed future weather, alternative origin, or one-sided
comparison replaces them. The two policies use the same eligible samples.

Sources include the new exact-run downloads, matching v2/v3 request caches,
the prior weather-refresh replay's exact-run cache, and the archived production
state's daily 00 UTC ECMWF records. The latter's provenance is the state-file
hash and the production collector, rather than a fabricated original request
URL. Full source hashes remain in the local source manifest; its hash and
dataset hash are recorded in the data audit.

This is an acquisition constraint, not selection based on model errors.
Coverage is incomplete and must be reported. In particular, the final weeks
have fewer available non-midnight runs. Model fitting still observes the
predeclared label cutoffs; the prospective comparison remains unchanged.
Later source acquisition or refitting requires a new experiment rather than
silently modifying the candidate during the trial.
