import test from "node:test";
import assert from "node:assert/strict";
import { assess, check, MINUTE, POLICY, SITE } from "../src/check.mjs";

const NOW = Date.parse("2026-11-02T18:00:00Z");
const env = { GITHUB_TOKEN: "test-only-not-a-real-token" };
export function payload(now = NOW, age = 10) {
  return { schema: 1, snapshotAt: new Date(now - age * MINUTE).toISOString(),
    forecast: { issuedAt: new Date(now - 20 * MINUTE).toISOString(),
      horizonEnd: new Date(now + 24 * 60 * MINUTE).toISOString(), status: "ready" },
    sources: { energinet: Object.fromEntries(["DK1", "DK2"].map((area) =>
      [area, { power: now - 5 * MINUTE, co2: now - 10 * MINUTE }])), energyCharts: {} } };
}
function store() {
  let value = {};
  return { load: () => structuredClone(value), save: (state) => { value = structuredClone(state); },
    records: [], record(result) { this.records.push(structuredClone(result)); } };
}
function source(data, options = {}) {
  const calls = [];
  const fetcher = async (url, init = {}) => {
    calls.push({ url, ...init });
    if (url.startsWith(SITE)) return options.malformed
      ? new Response("invalid") : Response.json(data);
    if (options.authError) return Response.json({}, { status: 401 });
    if (url.includes("/runs?")) {
      const busy = new URL(url).searchParams.get("status") === options.busy;
      return Response.json({ total_count: busy ? 1 : 0, workflow_runs: busy ? [{ id: 123 }] : [] });
    }
    if (options.dispatchError && url.includes("pages.yml/dispatches")) throw new Error("network timeout");
    return new Response(null, { status: 204 });
  };
  return { fetcher, calls, dispatches: () => calls.filter((x) => x.url.includes("pages.yml/dispatches")),
    alerts: () => calls.filter((x) => x.url.includes("watchdog-alert.yml/dispatches")) };
}

test("fresh publication needs no GitHub request", async () => {
  const mock = source(payload());
  const result = await check(env, store(), NOW, mock.fetcher);
  assert.equal(result.action, "checked");
  assert.equal(result.reason, null);
  assert.equal(mock.calls.length, 1);
});

test("stale publication dispatches the existing refresh on main", async () => {
  const mock = source(payload(NOW, 70));
  const s = store();
  const result = await check(env, s, NOW, mock.fetcher);
  assert.equal(result.action, "recovery_dispatched");
  assert.equal(s.load().attempts, 1);
  assert.deepEqual(JSON.parse(mock.dispatches()[0].body), { ref: "main", inputs: { refresh: true } });
  await check(env, s, NOW + 15 * MINUTE, mock.fetcher);
  assert.equal(mock.dispatches().length, 1);
});

for (const status of ["in_progress", "queued", "pending", "waiting", "requested"]) {
  test(`a ${status} workflow prevents another refresh`, async () => {
    const mock = source(payload(NOW, 70), { busy: status });
    assert.equal((await check(env, store(), NOW, mock.fetcher)).action, "workflow_busy");
    assert.equal(mock.dispatches().length, 0);
  });
}

test("ambiguous dispatch failures still consume cooldown and bounded attempt budget", async () => {
  const mock = source(payload(NOW, 70), { dispatchError: true });
  const s = store();
  assert.equal((await check(env, s, NOW, mock.fetcher)).action, "github_error");
  await check(env, s, NOW + MINUTE, mock.fetcher);
  assert.equal(mock.dispatches().length, 1);
  await check(env, s, NOW + POLICY.cooldown, mock.fetcher);
  await check(env, s, NOW + 2 * POLICY.cooldown, mock.fetcher);
  const result = await check(env, s, NOW + 3 * POLICY.cooldown, mock.fetcher);
  assert.equal(result.action, "recovery_exhausted");
  assert.equal(mock.dispatches().length, 3);
  assert.equal(mock.alerts().length, 1);
});

test("a fresh snapshot restores the recovery budget", async () => {
  const s = store();
  s.save({ attempts: 3, lastAttemptAt: NOW, alertSent: true, alertAttempts: 3 });
  await check(env, s, NOW, source(payload()).fetcher);
  assert.equal(s.load().attempts, 0);
  assert.equal(s.load().alertAttempts, 0);
  assert.equal(s.load().alertSent, false);
});

test("delayed upstream readings alert without refreshing a current snapshot", async () => {
  const data = payload();
  data.sources.energinet.DK1.power = NOW - 180 * MINUTE;
  const mock = source(data);
  const result = await check(env, store(), NOW, mock.fetcher);
  assert.equal(result.reason, "source_delayed");
  assert.equal(mock.dispatches().length, 0);
  assert.equal(mock.alerts().length, 1);
});

test("new metadata does not make an old forecast fresh", async () => {
  const data = payload();
  data.forecast.issuedAt = new Date(NOW - 200 * MINUTE).toISOString();
  data.forecast.checkedAt = new Date(NOW).toISOString();
  assert.equal(assess(data, NOW).reason, "forecast_stale");
});

test("invalid metadata fails closed and alerts after consecutive failures", async () => {
  const mock = source({}, { malformed: true });
  const s = store();
  await check(env, s, NOW, mock.fetcher);
  assert.equal(mock.alerts().length, 0);
  await check(env, s, NOW + 15 * MINUTE, mock.fetcher);
  assert.equal(mock.alerts().length, 1);
  assert.equal(mock.dispatches().length, 0);
});

test("missing token is explicit and never sent to GitHub", async () => {
  const mock = source(payload(NOW, 240));
  const result = await check({}, store(), NOW, mock.fetcher);
  assert.equal(result.action, "credential_required");
  assert.equal(mock.calls.length, 1);
});

test("invalid GitHub credentials fail closed without exposing secrets", async () => {
  const mock = source(payload(NOW, 70), { authError: true });
  const result = await check(env, store(), NOW, mock.fetcher);
  assert.equal(result.error, "github_http_401");
  assert.equal(mock.dispatches().length, 0);
  assert.ok(!JSON.stringify(result).includes(env.GITHUB_TOKEN));
});

test("future and offset-free publication timestamps are rejected", () => {
  for (const snapshotAt of ["2026-11-03T00:00:00Z", "2026-11-02T18:00:00", null]) {
    assert.throws(() => assess({ ...payload(), snapshotAt }, NOW));
  }
});

test("a malformed active-run response cannot permit a refresh", async () => {
  const mock = source(payload(NOW, 70));
  const result = await check(env, store(), NOW, (url, init) => url.includes("/runs?")
    ? Promise.resolve(Response.json({ total_count: 1, workflow_runs: [] })) : mock.fetcher(url, init));
  assert.equal(result.action, "github_error");
  assert.equal(mock.dispatches().length, 0);
});

test("redirects fail closed instead of forwarding credentials", async () => {
  const mock = source(payload(NOW, 70));
  const result = await check(env, store(), NOW, (url, init) => {
    if (url.includes("/runs?")) {
      assert.equal(init.redirect, "manual");
      return Promise.resolve(new Response(null, { status: 302, headers: { Location: "https://unrelated.example" } }));
    }
    return mock.fetcher(url, init);
  });
  assert.equal(result.error, "github_http_302");
  assert.equal(mock.dispatches().length, 0);
});

test("wind recovery waits for source publication and runs even with a fresh snapshot", async () => {
  const s = store();
  const at = NOW + 11 * MINUTE;
  const data = payload(at);
  data.windForecast = { issuedAt: new Date(at - 125 * MINUTE).toISOString(),
    lastAttempt: { status: "waiting-for-source-hour" } };
  const mock = source(data);
  assert.equal((await check(env, s, at, mock.fetcher)).action, "waiting_for_source_window");
  const result = await check(env, s, NOW + 19 * MINUTE, mock.fetcher);
  assert.equal(result.action, "recovery_dispatched");
  assert.equal(result.health.windAttempt, "waiting-for-source-hour");
  assert.equal(mock.dispatches().length, 1);
  await check(env, s, NOW + 34 * MINUTE, mock.fetcher);
  assert.equal(mock.dispatches().length, 1);
});

test("a late wind issue covers its own hour and leaves the next hour eligible", async () => {
  const hour = Date.parse("2026-11-18T00:00:00Z");
  const stamp = (offset) => new Date(hour + offset * MINUTE).toISOString();
  const s = store();
  const data = payload(hour);
  data.windForecast = { areas: { DK1: { issuedAt: stamp(-1) }, DK2: { issuedAt: stamp(-2) } } };
  const mock = source(data);
  assert.equal((await check(env, s, hour + 4 * MINUTE, mock.fetcher)).action, "checked");
  assert.equal((await check(env, s, hour + 19 * MINUTE, mock.fetcher)).action, "recovery_dispatched");
  data.windForecast.areas.DK2.issuedAt = stamp(23);
  assert.equal((await check(env, s, hour + 34 * MINUTE, mock.fetcher)).action, "cooldown");
  data.windForecast.areas.DK1.issuedAt = stamp(39);
  assert.equal((await check(env, s, hour + 49 * MINUTE, mock.fetcher)).action, "checked");
  assert.equal(s.load().attempts, 0);
  assert.equal(mock.dispatches().length, 1);
});

test("Europe recovery has its own attempt budget and a completed artifact waits for import", async () => {
  const at = NOW + 19 * MINUTE;
  const s = store();
  const data = payload(at);
  data.europeCollection = { collectedAt: new Date(at - 360 * MINUTE).toISOString() };
  const mock = source(data);
  const one = await check(env, s, at, mock.fetcher);
  assert.equal(one.reason, "europe_collection_stale");
  assert.equal(one.collection.action, "recovery_dispatched");
  assert.equal(one.action, "checked");
  assert.equal(mock.calls.filter(x => x.url.includes("collect-europe.yml/dispatches")).length, 1);
  const two = await check(env, s, at + 15 * MINUTE, mock.fetcher);
  assert.equal(two.collection.action, "cooldown");
  const fetcher = (url, init) => url.includes("status=success")
    ? Promise.resolve(Response.json({ workflow_runs: [{ updated_at: new Date(at).toISOString() }] }))
    : mock.fetcher(url, init);
  const three = await check(env, s, at + 30 * MINUTE, fetcher);
  assert.equal(three.collection.action, "awaiting_publication");
  assert.equal(three.action, "recovery_dispatched");
});

test("fresh collection with old provider data alerts without collecting again", async () => {
  const data = payload();
  data.europeCollection = { collectedAt: new Date(NOW - 10 * MINUTE).toISOString() };
  data.sources.energyCharts = { dk: {power: null}, de: {power: NOW - 420*MINUTE}, fr: {power: NOW - 300*MINUTE} };
  const mock = source(data);
  const result = await check(env, store(), NOW, mock.fetcher);
  assert.equal(result.reason, "europe_source_delayed");
  assert.equal(result.health.europeMedianPowerAgeMinutes, 360);
  assert.equal(result.action, "checked");
  assert.equal(mock.calls.filter(x => x.url.includes("collect-europe")).length, 0);
});

test("the oldest wind region can raise its own alert", () => {
  const data = payload();
  data.windForecast = { issuedAt: new Date(NOW).toISOString(), areas: {
    DK1: { issuedAt: new Date(NOW - 240*MINUTE).toISOString() }, DK2: { issuedAt: new Date(NOW).toISOString() }
  } };
  assert.equal(assess(data, NOW).reason, "wind_forecast_stale");
});
