export const HOUR = 3600000;
export const usable = (value) => typeof value === "number" && Number.isFinite(value);

export function forecastState(payload, now = Date.now()) {
  if (!payload?.latest) return "unavailable";
  if (payload.status !== "ready" || now - Date.parse(payload.latest.issuedAt) > 2 * HOUR) return "delayed";
  return "ready";
}

export function chartSeries(payload, area, target, mode = "outlook") {
  const record = mode === "comparison" ? payload?.comparison : payload?.latest;
  if (!record) return { record: null, points: [] };
  const observed = new Map((payload.observations?.[area] || []).map(r => [Date.parse(r.time), r[target]]));
  const forecast = new Map(record.times.map((t, i) => [Date.parse(t), record.areas[area][target][i]]));
  const band=record.ranges?.areas?.[area]?.[target];
  const bounds=new Map(record.times.map((t,i)=>[Date.parse(t),{lower:band?.lower?.[i],upper:band?.upper?.[i]}]));
  const origin = Date.parse(record.origin);
  const start = mode === "comparison" ? Date.parse(record.times[0]) : origin - 24 * HOUR;
  const end = Date.parse(record.times.at(-1));
  const times = new Set([...forecast.keys(), ...observed.keys()]);
  const points = [...times].filter(t => t >= start && t <= end).sort((a,b) => a-b).map(time => ({time,
    observed: usable(observed.get(time)) ? observed.get(time) : null,
    predicted: usable(forecast.get(time)) ? forecast.get(time) : null,
    lower: usable(bounds.get(time)?.lower) ? bounds.get(time).lower : null,
    upper: usable(bounds.get(time)?.upper) ? bounds.get(time).upper : null}));
  return {record, points};
}

export function forecastStats(record, area, target) {
  if (!record) return null;
  const values = record.areas[area][target];
  if (!values.length || !values.every(usable)) return null;
  const peak = Math.max(...values);
  return {average: values.reduce((a,b) => a+b,0)/values.length,peak,peakTime: record.times[values.indexOf(peak)]};
}
