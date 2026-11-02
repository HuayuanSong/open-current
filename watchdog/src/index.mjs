import { DurableObject } from "cloudflare:workers";
import { check, MINUTE } from "./check.mjs";

/** One coordinator per monitored repository; no state is shared with other apps. */
export class FreshnessMonitor extends DurableObject {
  constructor(ctx, env) {
    super(ctx, env);
    ctx.storage.sql.exec("CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY CHECK (id = 1), value TEXT NOT NULL)");
    ctx.storage.sql.exec("INSERT OR IGNORE INTO state VALUES (1, '{}')");
    ctx.storage.sql.exec("CREATE TABLE IF NOT EXISTS checks (checked_at INTEGER PRIMARY KEY, payload TEXT NOT NULL)");
  }

  load() {
    return JSON.parse(this.ctx.storage.sql.exec("SELECT value FROM state WHERE id = 1").one().value);
  }

  save(state) {
    this.ctx.storage.sql.exec("UPDATE state SET value = ? WHERE id = 1", JSON.stringify(state));
  }

  record(result, now) {
    this.ctx.storage.sql.exec("INSERT OR REPLACE INTO checks VALUES (?, ?)", now, JSON.stringify(result));
    this.ctx.storage.sql.exec("DELETE FROM checks WHERE checked_at < ?", now - 7 * 24 * 60 * MINUTE);
  }

  async inspect() {
    const now = Date.now();
    const claimed = this.ctx.storage.transactionSync(() => {
      const state = this.load();
      if ((state.leaseUntil || 0) > now || (state.lastCheckAt || 0) > now - 5 * MINUTE) return false;
      state.leaseUntil = now + 2 * MINUTE;
      state.lastCheckAt = now;
      this.save(state);
      return true;
    });
    if (!claimed) return { action: "check_already_started" };
    try {
      const result = await check(this.env, this, now);
      const log = { event: "freshness_check", ...result };
      if (result.reason || result.error) console.error(JSON.stringify(log));
      else console.log(JSON.stringify(log));
      return result;
    } finally {
      const state = this.load();
      state.leaseUntil = 0;
      this.save(state);
    }
  }

  status() {
    const rows = this.ctx.storage.sql.exec("SELECT payload FROM checks ORDER BY checked_at DESC LIMIT 1").toArray();
    const latest = rows.length ? JSON.parse(rows[0].payload) : null;
    const overdue = !latest || Date.now() - Date.parse(latest.checkedAt) > 30 * MINUTE;
    return { schema: 1, overdue, latest, historyUrl: "/history", retentionDays: 7 };
  }

  history() {
    return this.ctx.storage.sql.exec(
      "SELECT payload FROM checks WHERE checked_at >= ? ORDER BY checked_at DESC LIMIT 672",
      Date.now() - 7 * 24 * 60 * MINUTE,
    ).toArray().map((row) => JSON.parse(row.payload));
  }
}

async function authorized(request, token) {
  if (!token) return false;
  const supplied = request.headers.get("Authorization") || "";
  const encoder = new TextEncoder();
  const [left, right] = await Promise.all([supplied, `Bearer ${token}`].map((value) =>
    crypto.subtle.digest("SHA-256", encoder.encode(value))));
  return crypto.subtle.timingSafeEqual(left, right);
}

export default {
  async scheduled(controller, env, ctx) {
    await env.MONITOR.getByName("HuayuanSong/open-current").inspect();
  },
  async fetch(request, env) {
    const path = new URL(request.url).pathname;
    const monitor = env.MONITOR.getByName("HuayuanSong/open-current");
    const options = { headers: { "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff" } };
    if (request.method === "POST" && path === "/check") {
      if (!await authorized(request, env.GITHUB_TOKEN)) return new Response("Unauthorized", { status: 401 });
      return Response.json(await monitor.inspect(), options);
    }
    if (request.method !== "GET") return new Response("Method not allowed", { status: 405 });
    if (path === "/history") return Response.json(await monitor.history(), options);
    if (!["/", "/status", "/health"].includes(path)) return new Response("Not found", { status: 404 });
    const status = await monitor.status();
    const unhealthy = status.overdue || !status.latest?.configured || status.latest.reason || status.latest.error;
    return Response.json(status, { ...options, status: path === "/health" && unhealthy ? 503 : 200 });
  },
};
