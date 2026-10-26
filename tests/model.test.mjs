import test from "node:test";
import assert from "node:assert/strict";
import {
  readingAt,
  priceAt,
  summaryAt,
  parseState,
  powerText,
  metricValue,
} from "../dist/model.js";

test("historical selection never carries a reading across an empty hour", () => {
  const rows = [{ t: 0 }, { t: 900000 }, { t: 7200000 }];
  assert.equal(readingAt(rows, 0).t, 900000);
  assert.equal(readingAt(rows, 3600000), null);
  assert.equal(readingAt(rows, null).t, 7200000);
});
test("prices expire at the end of their market interval and zero is valid", () => {
  const series = {
    intervalMinutes: 15,
    rows: [
      { t: 0, value: 0 },
      { t: 1800000, value: -15 },
    ],
  };
  assert.equal(priceAt(series, 0).value, 0);
  assert.equal(priceAt(series, 900000), null);
  assert.equal(priceAt(series, 1900000).value, -15);
});
test("a historical map gap does not fall back to latest values", () => {
  const country = {
    power: { lowCarbon: 100 },
    history: [{ t: 0, power: { lowCarbon: 50 } }],
  };
  assert.equal(metricValue(summaryAt(country, 3600000), "lowCarbon"), null);
  assert.equal(metricValue(summaryAt(country, 0), "lowCarbon"), 50);
});
test("invalid URL state is bounded to known countries and views", () => {
  assert.deepEqual(
    parseState("#country=nope&metric=oops&chart=nope&time=NaN", ["dk", "de"]),
    {
      country: "dk",
      metric: "lowCarbon",
      chart: "power",
      time: null,
      zone: null,
    },
  );
});
test("missing power stays missing and signed flows preserve their sign", () => {
  assert.equal(powerText(null), "—");
  assert.equal(powerText(0), "0 MW");
  assert.equal(powerText(-1500), "-1.50 GW");
});
