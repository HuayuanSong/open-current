# Freshness watchdog

A Cloudflare Worker checks the published snapshot, Danish forecasts and independent European collection every 15 minutes. GitHub Actions collects and publishes the electricity data. [Live status](https://open-current-watchdog.naermad.workers.dev/status) · [Health](https://open-current-watchdog.naermad.workers.dev/health) · [Check history](https://open-current-watchdog.naermad.workers.dev/history)

## Freshness and recovery

Checks run at minutes **04, 19, 34 and 49 UTC**. The watchdog can recover a publication after 60 minutes. At minute 17 it also starts checking whether both Danish regions have a wind forecast for the new reference hour. Routine dispatches use minutes 17–49 so the hourly observation buffer has elapsed and the runner has time to finish. A publication over two hours old can trigger recovery at any check.

If the last imported European collection is 90 minutes old, the watchdog checks GitHub for a completed artifact or an active collector. A fresh artifact triggers publication. An overdue collector can be dispatched independently. Source timestamps exceeding normal provider lag trigger alerts while the collector keeps its normal pacing.

The Durable Object serializes checks, records attempts before dispatch and limits retries to three per outage with a 45-minute cooldown. It keeps seven days of checks. Alerts cover a publication over two hours old, a baseline or regional wind issue over three hours old, an imported European collection over 150 minutes old, and delayed provider readings. The Danish source threshold is two hours. The European threshold is a four-hour median across countries with generation readings after a recent collection. Missing health metadata and exhausted recovery attempts are recorded separately.

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
python3 watchdog/report.py https://open-current-watchdog.naermad.workers.dev
```

Review the report and `/history` for polling gaps, publication ages, dispatches,
forecast/source delays and failed recovery. The report explicitly states whether
48 hours have elapsed. It measures observations at polling times, not every
intermediate publication. The independent schedule reduces reliance on GitHub's
cron scheduler; GitHub API/runner availability and upstream reporting delays
still affect recovery.

Monitoring began on **2 November 2026 at 18:08 UTC**. The first full 48-hour
review is possible after **4 November at 18:08 UTC** (19:08 Copenhagen).
The initial deployment passed an authenticated live check; this alone does not
establish scheduled recovery reliability. The current GitHub credential expires
on **30 January 2027 at 23:00 UTC** and must be renewed before then.
