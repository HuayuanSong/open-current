# Freshness watchdog

A Cloudflare Worker independently checks the published electricity snapshot every
15 minutes. It starts the existing GitHub refresh workflow when publication is
over 60 minutes old and no refresh is active. Models, source collection and Pages
hosting continue to run in the existing pipeline.

## Recovery and monitoring

- Checks run at minutes **11, 26, 41 and 56 UTC**.
- `dist/data/health.json` keeps the original snapshot, forecast issue and source
  observation times. Deploying code does not make old readings fresh.
- All active GitHub run states are checked before dispatching `pages.yml` on `main`.
- One SQLite-backed Durable Object coordinates this repository. A persisted lease
  prevents overlapping checks. Attempts are recorded **before** dispatch, so an
  ambiguous network timeout cannot bypass the cooldown after a restart.
- Recovery has a 45-minute cooldown and at most three attempts per outage.
  A fresh snapshot resets the budget. Invalid metadata and GitHub errors fail closed.
- Stale source readings alone do not trigger another collection.
- Seven days of timestamped check results are retained for review.

Publication older than two hours, forecasts older than three hours or past their
horizon, and Danish source readings older than two hours after a recent collection
raise an alert. Repeated metadata failures and exhausted recovery attempts also
raise alerts. European source ages are reported independently because their
normal reporting delays differ.

An alert dispatches `watchdog-alert.yml`, a deliberately failing GitHub Actions
run with an explanation. Alert delivery is deduplicated per incident and retries
are bounded. **Email delivery depends on the owner's GitHub Actions notification
settings**. Authentication failure can also prevent alert delivery; `/health`
and Cloudflare error logs expose that failure.

## Endpoints

- `GET /status`: latest check, action, errors and separate freshness measurements.
- `GET /health`: HTTP 503 when monitoring is overdue, unconfigured or unhealthy.
- `GET /history`: up to seven days of check results, with no credentials.
- `POST /check`: an immediate check, requiring the configured GitHub token as a
  Bearer credential. It respects the same lease, cooldown and recovery limits.

Scheduled checks and requests do not take target URLs or repository names from
visitors. Upstream responses are size-limited and redirects are rejected.

## Setup and deployment

Use Node **26.12+**, **24.22+**, or **22.23+**, and the pinned dependencies:

```sh
cd watchdog
npm ci
npm test
npm run build
npx cf auth login
```

Create a **fine-grained** GitHub token scoped only to `HuayuanSong/open-current`,
with **Actions: read and write** and the automatic Metadata read permission.
No Contents, Issues or other repository permissions are required. Choose an
expiration and renew it before that date.

The local helper provides a prefilled GitHub link and a one-use browser form:

```sh
python3 setup_token.py --output ../.cache/watchdog-setup/secrets.json
npx cf deploy --secrets-file ../.cache/watchdog-setup/secrets.json
```

The helper binds only to loopback, validates the request origin and saves the
token in an ignored file with mode 0600. **Delete that temporary file immediately
after successful deployment.** Never put tokens in source, command arguments,
screenshots, issue comments or chat messages. Deployment stores the value as the
Worker's `GITHUB_TOKEN` secret. Cloudflare logs and traces are enabled.

Changing Worker code requires a separate `cf deploy`; a Pages deployment does not
deploy this Worker. `watchdog-tests.yml` validates the Worker on GitHub.

## Review after 48 hours

```sh
python3 watchdog/report.py https://YOUR-WATCHDOG.workers.dev
```

Review the report and `/history` for polling gaps, publication ages, dispatches,
forecast/source delays and failed recovery. The report explicitly states whether
48 hours have elapsed. It measures observations at polling times, not every
intermediate publication. The independent schedule reduces reliance on GitHub's
cron scheduler; GitHub API/runner availability and upstream reporting delays
still affect recovery.
