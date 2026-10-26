export const GROUPS = {
  wind: { name: "Wind", color: "#47856b" },
  solar: { name: "Solar", color: "#d2b751" },
  hydro: { name: "Hydro", color: "#6c9da8" },
  nuclear: { name: "Nuclear", color: "#9b91b2" },
  bio: { name: "Bioenergy & other renewables", color: "#a0af72" },
  gas: { name: "Gas", color: "#bf8966" },
  coal: { name: "Coal", color: "#61655d" },
  other: { name: "Other & waste", color: "#b5b5a3" },
  storage: { name: "Storage discharge", color: "#98a9b6" },
};
export const valid = (value) =>
  typeof value === "number" && Number.isFinite(value);
export const formatNumber = (value, digits = 1) =>
  valid(value)
    ? value.toLocaleString("en-GB", {
        maximumFractionDigits: digits,
        minimumFractionDigits: digits,
      })
    : "—";
export const powerText = (value) =>
  !valid(value)
    ? "—"
    : Math.abs(value) >= 1000
      ? `${formatNumber(value / 1000, 2)} GW`
      : `${formatNumber(value, 0)} MW`;
export const dateText = (time, options = {}) =>
  !valid(time)
    ? "No reading"
    : new Intl.DateTimeFormat("en-GB", {
        timeZone: "UTC",
        day: "numeric",
        month: "short",
        hour: "2-digit",
        minute: "2-digit",
        ...options,
      }).format(time) + " UTC";
export function ageText(time, now = Date.now()) {
  if (!valid(time)) return "No reading";
  const minutes = Math.max(0, Math.floor((now - time) / 60000));
  return minutes < 60
    ? `${minutes} min ago`
    : minutes < 1440
      ? `${Math.floor(minutes / 60)}h ${minutes % 60}m ago`
      : `${Math.floor(minutes / 1440)}d ago`;
}
export function readingAt(rows, hour = null) {
  if (!rows?.length) return null;
  if (hour === null) return rows[rows.length - 1];
  // The time slider chooses an hour, not an instant. Never carry an earlier
  // observation into an empty hour, or use the following hour's reading.
  const index = upperBound(rows, hour + 3600000 - 1) - 1;
  return index >= 0 && rows[index].t >= hour ? rows[index] : null;
}
export function upperBound(rows, time) {
  let lo = 0,
    hi = rows.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (rows[mid].t <= time) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}
export function priceAt(series, time = Date.now()) {
  if (!series?.rows?.length) return null;
  const row = series.rows[upperBound(series.rows, time) - 1];
  return row &&
    time < row.t + series.intervalMinutes * 60000 &&
    valid(row.value)
    ? row
    : null;
}
export function summaryAt(country, hour = null) {
  if (hour === null) return country;
  const row = country.history?.find((r) => r.t === hour);
  return { ...country, power: row?.power ?? null, flows: row?.flows ?? null };
}
export function metricValue(country, metric) {
  return metric === "net"
    ? (country?.flows?.net ?? null)
    : (country?.power?.[metric] ?? null);
}
export const escapeHTML = (value) =>
  String(value).replace(
    /[&<>"']/g,
    (ch) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        ch
      ],
  );
export function parseState(hash, codes) {
  const params = new URLSearchParams(hash.replace(/^#/, ""));
  const code = params.get("country");
  const metric = params.get("metric");
  const time = params.get("time");
  const chart = params.get("chart");
  const zone = params.get("zone");
  return {
    country: codes.includes(code) ? code : "dk",
    metric: ["lowCarbon", "renewable", "net"].includes(metric)
      ? metric
      : "lowCarbon",
    time: time && /^\d{13}$/.test(time) ? Number(time) : null,
    chart: ["power", "share", "price"].includes(chart) ? chart : "power",
    zone: zone && /^[A-Za-z0-9-]{1,20}$/.test(zone) ? zone : null,
  };
}
