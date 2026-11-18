export const MINUTE = 60_000;
export const POLICY = Object.freeze({
  refreshAfter: 60 * MINUTE,
  publicationAlertAfter: 120 * MINUTE,
  forecastAlertAfter: 180 * MINUTE,
  sourceAlertAfter: 120 * MINUTE,
  windRefreshAfter: 55 * MINUTE,
  collectionRefreshAfter: 90 * MINUTE,
  collectionAlertAfter: 150 * MINUTE,
  europeSourceAlertAfter: 240 * MINUTE,
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
  const tracksWind = payload.windForecast != null;
  const regionalIssues = Object.values(payload.windForecast?.areas || {}).map((r) => timestamp(r?.issuedAt, now));
  const windIssue = regionalIssues.length
    ? (regionalIssues.some((t) => t === null) ? null : Math.min(...regionalIssues))
    : timestamp(payload.windForecast?.issuedAt, now);
  const windAge = age(windIssue);
  const windDue = tracksWind && (windIssue === null || now - windIssue >= POLICY.windRefreshAfter);
  const tracksCollection = Object.hasOwn(payload, "europeCollection");
  const collection = timestamp(payload.europeCollection?.collectedAt, now);
  const collectionDue = tracksCollection && (collection === null || now - collection >= POLICY.collectionRefreshAfter);
  const powerAges = Object.values(sourceAges.energyCharts).map((feed) => feed.power)
    .filter(Number.isFinite).sort((a, b) => a - b);
  const middle = Math.floor(powerAges.length / 2);
  const medianPowerAge = powerAges.length % 2 ? powerAges[middle]
    : powerAges.length ? (powerAges[middle - 1] + powerAges[middle]) / 2 : null;
  let reason = null;
  if (now - snapshot > POLICY.publicationAlertAfter) reason = "publication_stale";
  else if (forecastStale) reason = "forecast_stale";
  else if (tracksWind && (windAge === null || windAge * MINUTE > POLICY.forecastAlertAfter)) reason = "wind_forecast_stale";
  else if (tracksCollection && (collection === null || now - collection > POLICY.collectionAlertAfter)) reason = "europe_collection_stale";
  else if (!publicationStale && sourceDelayed) reason = "source_delayed";
  else if (tracksCollection && !collectionDue && (medianPowerAge === null || medianPowerAge * MINUTE > POLICY.europeSourceAlertAfter)) reason = "europe_source_delayed";
  return {
    snapshotAt: new Date(snapshot).toISOString(),
    snapshotAgeMinutes: age(snapshot),
    forecastIssuedAt: issue === null ? null : new Date(issue).toISOString(),
    forecastAgeMinutes: age(issue),
    forecastHorizonEnd: Number.isFinite(horizon) ? new Date(horizon).toISOString() : null,
    forecastStatus: payload.forecast?.status || "unknown",
    sourceAgeMinutes: sourceAges,
    windIssuedAt: windIssue === null ? null : new Date(windIssue).toISOString(),
    windAgeMinutes: windAge,
    windAttempt: payload.windForecast?.lastAttempt?.status || null,
    windDue, tracksWind, collectionDue,
    europeCollectedAt: collection === null ? null : new Date(collection).toISOString(),
    europeCollectionAgeMinutes: age(collection),
    europeMedianPowerAgeMinutes: medianPowerAge,
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

async function activeRun(workflow, token, fetcher) {
  const responses = await Promise.all(ACTIVE_STATUSES.map((status) => request(
    `${API}${workflow}/runs?branch=main&status=${status}&per_page=1`,
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

async function recover(workflow, inputs, budget, persist, now, token, fetcher) {
  try {
    if ((budget.attempts || 0) >= POLICY.maxAttempts) return { action: "recovery_exhausted" };
    if (budget.lastAttemptAt && now - budget.lastAttemptAt < POLICY.cooldown) return { action: "cooldown" };
    const active = await activeRun(workflow, token, fetcher);
    if (active) return { action: "workflow_busy", runUrl: `https://github.com/HuayuanSong/open-current/actions/runs/${active.id}` };
    budget.attempts = (budget.attempts || 0) + 1;
    budget.lastAttemptAt = now;
    persist();
    const run = await dispatch(workflow, inputs, token, fetcher);
    const result = { action: "recovery_dispatched" };
    if (Number.isSafeInteger(run?.workflow_run_id)) result.runUrl = `https://github.com/HuayuanSong/open-current/actions/runs/${run.workflow_run_id}`;
    return result;
  } catch (error) {
    return { action: "github_error", error: error instanceof RemoteError ? error.message : "github_unavailable" };
  }
}

async function recoverCollection(health, state, persist, now, token, fetcher) {
  const budget = state.collectionRecovery ||= {};
  if (budget.collectedAt !== health.europeCollectedAt) {
    Object.assign(budget, { attempts: 0, lastAttemptAt: null, collectedAt: health.europeCollectedAt });
  }
  if (!health.collectionDue) return { action: "current" };
  try {
    // A completed artifact may still be waiting for the next Pages publisher.
    const runs = await request(`${API}collect-europe.yml/runs?branch=main&status=success&per_page=1`,
      { headers: headers(token) }, fetcher);
    if (!Array.isArray(runs?.workflow_runs)) throw new RemoteError("github");
    const completed = timestamp(runs.workflow_runs[0]?.updated_at, now);
    if (completed !== null && now - completed < POLICY.collectionRefreshAfter
        && (health.europeCollectedAt === null || completed > Date.parse(health.europeCollectedAt))) {
      return { action: "awaiting_publication", completedAt: completed };
    }
    return await recover("collect-europe.yml", {}, budget, persist, now, token, fetcher);
  } catch (error) {
    return { action: "github_error", error: error instanceof RemoteError ? error.message : "github_unavailable" };
  }
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
    if (!result.health.publicationStale && !result.health.windDue) {
      state.attempts = 0;
      state.lastAttemptAt = null;
    }

  } catch (error) {
    state.healthFailures = (state.healthFailures || 0) + 1;
    result.error = error instanceof RemoteError ? error.message : "health_invalid";
  }
  store.save(state);

  // Missing/invalid metadata is an alert, never permission to dispatch blindly.
  if (!env.GITHUB_TOKEN) {
    result.action = "credential_required";
  } else if (result.health) {
    const h = result.health;
    const persist = () => store.save(state);
    if (h.collectionDue || state.collectionRecovery) {
      result.collection = await recoverCollection(h, state, persist, now, env.GITHUB_TOKEN, fetcher);
      if (result.collection.error) result.error = result.collection.error;
    }
    const minute = new Date(now).getUTCMinutes();
    const eligible = !h.tracksWind || (minute >= 17 && minute <= 49);
    const importReady = result.collection?.action === "awaiting_publication";
    if (h.publicationStale || h.windDue || importReady) {
      if (eligible || h.snapshotAgeMinutes > POLICY.publicationAlertAfter / MINUTE) {
        let budget = state;
        if (importReady && !h.publicationStale && !h.windDue) {
          if (state.importRecovery?.completedAt !== result.collection.completedAt) state.importRecovery = { completedAt: result.collection.completedAt };
          budget = state.importRecovery;
        }
        Object.assign(result, await recover("pages.yml", { refresh: true }, budget, persist, now, env.GITHUB_TOKEN, fetcher));
      } else {
        result.action = "waiting_for_source_window";
      }
    }
  }
  result.attempts = state.attempts || 0;
  result.reason = (result.action === "recovery_exhausted" || result.collection?.action === "recovery_exhausted") ? "recovery_exhausted"
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
    state.alertSent = false;
    state.alertAttempts = 0;
    state.lastAlertAttemptAt = null;
  }
  store.save(state);
  store.record(result, now);
  return result;
}
