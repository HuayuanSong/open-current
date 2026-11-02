import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { build } from "esbuild";
import { Miniflare, convertV4MiniflareOptions, Response as WorkerResponse } from "miniflare";

test("real Workers runtime: concurrent checks, authorization, SQLite persistence and restart", async () => {
  const directory = await mkdtemp(join(tmpdir(), "open-current-watchdog-"));
  const bundled = await build({ entryPoints: ["src/index.mjs"], bundle: true, write: false,
    format: "esm", platform: "neutral", external: ["cloudflare:workers"] });
  let launches = 0;
  let pagesReads = 0;
  const now = Date.now();
  const health = { schema: 1, snapshotAt: new Date(now - 70 * 60_000).toISOString(),
    forecast: { issuedAt: new Date(now).toISOString(), horizonEnd: new Date(now + 86400_000).toISOString() },
    sources: { energinet: { DK1: { power: now, co2: now }, DK2: { power: now, co2: now } } } };
  const options = convertV4MiniflareOptions({
    name: "open-current-watchdog", modules: true, script: bundled.outputFiles[0].text,
    compatibilityDate: "2026-11-02",
    bindings: { GITHUB_TOKEN: "test-token" },
    durableObjects: { MONITOR: { className: "FreshnessMonitor", useSQLite: true } },
    durableObjectsPersist: directory,
    outboundService: async (request) => {
      if (request.url.startsWith("https://huayuansong.github.io/")) {
        pagesReads++;
        await new Promise((resolve) => setTimeout(resolve, 30));
        return new WorkerResponse(JSON.stringify(health));
      }
      if (request.url.includes("/runs?")) {
        return new WorkerResponse(JSON.stringify({ total_count: 0, workflow_runs: [] }));
      }
      assert.match(request.url, /pages\.yml\/dispatches$/);
      launches++;
      return new WorkerResponse(null, { status: 204 });
    },
  });
  options.resourcePersistencePath = directory;
  let mf = new Miniflare(options);
  try {
    assert.equal((await mf.dispatchFetch("http://localhost/check", { method: "POST" })).status, 401);
    assert.equal(pagesReads, 0);
    const checks = await Promise.all(Array.from({ length: 4 }, () => mf.dispatchFetch(
      "http://localhost/check", { method: "POST", headers: { Authorization: "Bearer test-token" } },
    )));
    for (const response of checks) assert.equal(response.status, 200);
    assert.equal(launches, 1);
    assert.equal(pagesReads, 1);
    const status = await (await mf.dispatchFetch("http://localhost/status")).json();
    assert.equal(status.latest.action, "recovery_dispatched");
    assert.equal((await (await mf.dispatchFetch("http://localhost/history")).json()).length, 1);
    await mf.dispose();
    mf = new Miniflare(options);
    const after = await (await mf.dispatchFetch("http://localhost/status")).json();
    assert.equal(after.latest.attempts, 1);
    await mf.dispatchFetch("http://localhost/check", { method: "POST", headers: { Authorization: "Bearer test-token" } });
    assert.equal(launches, 1);
    assert.equal((await mf.dispatchFetch("http://localhost/health")).status, 200);
  } finally {
    await mf.dispose();
    await rm(directory, { recursive: true, force: true });
  }
});
