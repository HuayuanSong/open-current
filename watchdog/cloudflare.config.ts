import { bindings, defineConfig, exports, triggers } from "cf/config";

export default defineConfig({
  accountId: "0bd8d9ef9102676df70229fe0d83f223",
  worker: {
    name: "open-current-watchdog",
    entrypoint: "./src/index.mjs",
    compatibilityDate: "2026-11-02",
    workersDev: true,
    env: {
      MONITOR: bindings.durableObject({ worker: "open-current-watchdog", exportName: "FreshnessMonitor" }),
      GITHUB_TOKEN: bindings.secret(),
    },
    exports: { FreshnessMonitor: exports.durableObject({ storage: "sqlite" }) },
    triggers: [triggers.scheduled({ schedule: "11,26,41,56 * * * *" })],
    observability: {
      enabled: true,
      logs: { enabled: true, headSamplingRate: 1, invocationLogs: true },
      traces: { enabled: true, headSamplingRate: 0.01 },
    },
  },
});
