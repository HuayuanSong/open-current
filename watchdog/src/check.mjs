export const MINUTE = 60_000;
export const POLICY = Object.freeze({
  refreshAfter: 60 * MINUTE,
  publicationAlertAfter: 120 * MINUTE,
  forecastAlertAfter: 180 * MINUTE,
  sourceAlertAfter: 120 * MINUTE,
  cooldown: 45 * MINUTE,
  maxAttempts: 3,
});
export const SITE = "https://huayuansong.github.io/open-current/";
export const API = "https://api.github.com/repos/HuayuanSong/open-current/actions/workflows/";
const ACTIVE_STATUSES = ["in_progress", "queued", "pending", "waiting", "requested"];

export class RemoteError extends Error {
  constructor(service, status = 0) {
    super(`${service}_${status ? `http_${status}` : "unavailable"}`);
    this.status = status;
  }
}

export async function request(url, init = {}, fetcher = fetch) {
  const service = url.startsWith(API) ? "github" : "publication";
  let response;
  try {
    response = await fetcher(url, {
      ...init,
      redirect: "manual",
      signal: AbortSignal.timeout(10_000),
    });
  } catch {
    throw new RemoteError(service);
  }
  if (!response.ok) {
    await response.body?.cancel();
    throw new RemoteError(service, response.status);
  }
  if (response.status === 204) return null;
  const limit = service === "publication" ? 32_768 : 131_072;
  const reader = response.body?.getReader();
  if (!reader) throw new RemoteError(service);
  const chunks = [];
  let length = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      length += value.byteLength;
      if (length > limit) throw new Error("Oversized response");
      chunks.push(value);
    }
    const bytes = new Uint8Array(length);
    let offset = 0;
    for (const chunk of chunks) {
      bytes.set(chunk, offset);
      offset += chunk.byteLength;
    }
    return JSON.parse(new TextDecoder().decode(bytes));
  } catch {
    await reader.cancel().catch(() => {});
    throw new RemoteError(service);
  }
}

function timestamp(value, now, required = false) {
  const time = typeof value === "number" ? value
    : typeof value === "string" && /(?:Z|[+-]\d\d:\d\d)$/.test(value) ? Date.parse(value) : NaN;
  if (!Number.isFinite(time) || time < 0 || time > now + 5 * MINUTE) {
    if (required) throw new Error("Invalid publication timestamp");
    return null;
  }
  return time;
}

export function assess(payload, now) {
  if (payload?.schema !== 1) throw new Error("Invalid health schema");
  const snapshot = timestamp(payload.snapshotAt, now, true);
  const issue = timestamp(payload.forecast?.issuedAt, now);
  // Forecast horizons are intentionally allowed to be in the future.
  const horizon = typeof payload.forecast?.horizonEnd === "string"
    ? Date.parse(payload.forecast.horizonEnd) : NaN;
  const age = (time) => time === null ? null : Math.max(0, Math.round((now - time) / MINUTE));
  const sourceAges = {};
  for (const provider of ["energinet", "energyCharts"]) {
    sourceAges[provider] = {};
    const rows = payload.sources?.[provider] || {};
    if (Object.keys(rows).length > 50) throw new Error("Unexpected source count");
    for (const [area, feeds] of Object.entries(rows)) {
      sourceAges[provider][area] = Object.fromEntries(
        Object.entries(feeds).map(([feed, value]) => [feed, age(timestamp(value, now))]),
      );
    }
  }
  const publicationStale = now - snapshot > POLICY.refreshAfter;
  const forecastStale = issue === null || now - issue > POLICY.forecastAlertAfter
    || !Number.isFinite(horizon) || horizon <= now;
  const sourceDelayed = ["DK1", "DK2"].some((area) => ["power", "co2"].some((feed) => {
    const value = sourceAges.energinet[area]?.[feed];
    return value == null || value > POLICY.sourceAlertAfter / MINUTE;
  }));
  let reason = null;
  if (now - snapshot > POLICY.publicationAlertAfter) reason = "publication_stale";
  else if (forecastStale) reason = "forecast_stale";
  else if (!publicationStale && sourceDelayed) reason = "source_delayed";
  return {
    snapshotAt: new Date(snapshot).toISOString(),
    snapshotAgeMinutes: age(snapshot),
    forecastIssuedAt: issue === null ? null : new Date(issue).toISOString(),
    forecastAgeMinutes: age(issue),
    forecastHorizonEnd: Number.isFinite(horizon) ? new Date(horizon).toISOString() : null,
    forecastStatus: payload.forecast?.status || "unknown",
    sourceAgeMinutes: sourceAges,
    publicationStale,
    reason,
  };
}

function headers(token) {
  return {
    Authorization: `Bearer ${token}`,
    Accept: "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "OpenCurrent-Freshness-Watchdog",
    "Content-Type": "application/json",
  };
}

async function activeRun(token, fetcher) {
  const responses = await Promise.all(ACTIVE_STATUSES.map((status) => request(
    `${API}pages.yml/runs?branch=main&status=${status}&per_page=1`,
    { headers: headers(token) }, fetcher,
  )));
  for (const result of responses) {
    if (!Number.isInteger(result?.total_count) || result.total_count < 0
        || !Array.isArray(result.workflow_runs)
        || (result.total_count > 0 && !Number.isSafeInteger(result.workflow_runs[0]?.id))) {
      throw new RemoteError("github");
    }
  }
  return responses.find((result) => result.total_count > 0)?.workflow_runs[0] || null;
}

async function dispatch(workflow, inputs, token, fetcher) {
  return request(`${API}${workflow}/dispatches`, {
    method: "POST", headers: headers(token),
    body: JSON.stringify({ ref: "main", inputs }),
  }, fetcher);
}

/** Persist attempts before network side effects, including ambiguous timeouts. */
export async function check(env, store, now = Date.now(), fetcher = fetch) {
  const state = store.load();
  const result = { checkedAt: new Date(now).toISOString(), configured: Boolean(env.GITHUB_TOKEN),
    action: "checked", health: null, error: null, alert: null };
  try {
    const payload = await request(`${SITE}data/health.json?check=${Math.floor(now / MINUTE)}`, {
      headers: { "Cache-Control": "no-cache", "User-Agent": "OpenCurrent-Freshness-Watchdog" },
    }, fetcher);
    result.health = assess(payload, now);
    state.healthFailures = 0;
    if (!result.health.publicationStale) {
      state.attempts = 0;
      state.lastAttemptAt = null;
      if (!result.health.reason) state.alertSent = false;
    }
  } catch (error) {
    state.healthFailures = (state.healthFailures || 0) + 1;
    result.error = error instanceof RemoteError ? error.message : "health_invalid";
  }
  store.save(state);

  // Missing/invalid metadata is an alert, never permission to dispatch blindly.
  if (!env.GITHUB_TOKEN) {
    result.action = "credential_required";
  } else if (result.health?.publicationStale) {
    try {
      const active = await activeRun(env.GITHUB_TOKEN, fetcher);
      if (active) {
        result.action = "workflow_busy";
        result.runUrl = `https://github.com/HuayuanSong/open-current/actions/runs/${Number(active.id)}`;
      } else if ((state.attempts || 0) >= POLICY.maxAttempts) {
        result.action = "recovery_exhausted";
      } else if (state.lastAttemptAt && now - state.lastAttemptAt < POLICY.cooldown) {
        result.action = "cooldown";
      } else {
        state.attempts = (state.attempts || 0) + 1;
        state.lastAttemptAt = now;
        store.save(state);
        const run = await dispatch("pages.yml", { refresh: true }, env.GITHUB_TOKEN, fetcher);
        result.action = "recovery_dispatched";
        if (Number.isSafeInteger(run?.workflow_run_id)) {
          result.runUrl = `https://github.com/HuayuanSong/open-current/actions/runs/${run.workflow_run_id}`;
        }
      }
    } catch (error) {
      result.action = "github_error";
      result.error = error instanceof RemoteError ? error.message : "github_unavailable";
    }
  }
  result.attempts = state.attempts || 0;
  result.reason = result.action === "recovery_exhausted" ? "recovery_exhausted"
    : result.health?.reason || (state.healthFailures >= 2 ? "health_unavailable" : null);

  if (result.reason && env.GITHUB_TOKEN && !state.alertSent
      && (!state.lastAlertAttemptAt || now - state.lastAlertAttemptAt >= POLICY.cooldown)
      && (state.alertAttempts || 0) < POLICY.maxAttempts) {
    state.lastAlertAttemptAt = now;
    state.alertAttempts = (state.alertAttempts || 0) + 1;
    store.save(state);
    try {
      await dispatch("watchdog-alert.yml", { reason: result.reason, checked_at: result.checkedAt },
        env.GITHUB_TOKEN, fetcher);
      state.alertSent = true;
      result.alert = "github_workflow_dispatched";
    } catch (error) {
      result.alert = "delivery_failed";
      result.error ||= error instanceof RemoteError ? error.message : "github_unavailable";
    }
  }
  if (result.health && !result.health.publicationStale && !result.reason) {
    state.alertAttempts = 0;
    state.lastAlertAttemptAt = null;
  }
  store.save(state);
  store.record(result, now);
  return result;
}
