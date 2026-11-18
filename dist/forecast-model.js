export const HOUR = 3600000;
export const usable = (value) => typeof value === "number" && Number.isFinite(value);

export function targetPayload(payload, target, area) {
  if (target !== "wind" || !payload?.windOutlook) return payload;
  const regional = payload.windOutlook.latestByArea?.[area];
  if (regional) return { ...payload.windOutlook, latest: regional, status: payload.windOutlook.message ? "delayed" : "ready" };
  if (payload.windOutlook.latest) return payload.windOutlook;
  return { ...payload, status: payload.windOutlook.message ? "delayed" : payload.status,
    windUpdatePending: true };
}

export function matchedObservationCount(payload, area, target) {
  const record = payload?.comparison;
  if (!record) return 0;
  const observations = new Map((payload.observations?.[area] || []).map(r => [Date.parse(r.time), r[target]]));
  const checked = Date.parse(payload.checkedAt);
  const cutoff = Number.isFinite(checked) ? checked - 15 * 60000 : Infinity;
  return record.times.filter((time, index) => Date.parse(time) > Date.parse(payload.pairedCutoff || record.issuedAt) &&
    Date.parse(time) <= cutoff && usable(observations.get(Date.parse(time))) &&
    usable(record.areas[area]?.[target]?.[index])).length;
}

export function viewPayload(payload, area, target, mode = "outlook", preference = "auto") {
  const current = targetPayload(payload, target, area);
  if (mode !== "comparison" || target !== "wind" || !(payload?.windOutlook?.latest || payload?.windOutlook?.latestByArea?.[area])) return current;
  const paired = payload.windComparison?.[area];
  if (paired?.comparison) {
    const previous = preference === "previous";
    const source = previous ? payload : current;
    const comparison = paired.comparison[previous ? "previous" : "current"];
    return { ...source, comparison, comparisonSource: previous ? "previous" : "current",
      comparisonHasPrevious: true, pairedComparison: true, pairedCutoff: paired.comparison.issuedAfter };
  }
  const currentHours = matchedObservationCount(current, area, target);
  const previousHours = matchedObservationCount(payload, area, target);
  const hasPrevious = Boolean(payload.comparison);
  const previous = hasPrevious && (preference === "previous" ||
    (preference === "auto" && currentHours < 24 && previousHours > currentHours));
  return { ...(previous ? payload : current), comparisonSource: previous ? "previous" : "current",
    comparisonCurrentHours: currentHours, comparisonHasPrevious: hasPrevious };
}

export function isolatedPoints(points, key) {
  return points.filter((point, index) => usable(point[key]) &&
    !usable(points[index - 1]?.[key]) && !usable(points[index + 1]?.[key]));
}

export function rangeUndercoverage(stats) {
  return stats?.rangeHours >= 24 && stats?.rangeOrigins >= 3 &&
    usable(stats.rangeCoverage) && stats.rangeCoverage < 0.75;
}

export function forecastTransition(points, origin) {
  const start = Date.parse(origin);
  const observed = points.find((point) => point.time === start);
  const forecast = points.find((point) => point.time === start + HOUR);
  if (
    !usable(observed?.observed) ||
    !usable(forecast?.predicted) ||
    usable(observed.predicted) ||
    points.some((point) => point.time > start && usable(point.observed))
  ) {
    return null;
  }
  return [
    { time: start, value: observed.observed },
    { time: start + HOUR, value: forecast.predicted },
  ];
}

export function forecastState(payload, now = Date.now()) {
  if (!payload?.latest) return "unavailable";
  if (payload.status !== "ready" || now - Date.parse(payload.latest.issuedAt) > 2 * HOUR) return "delayed";
  return "ready";
}

export function chartSeries(payload, area, target, mode = "outlook", preference = "auto") {
  payload = viewPayload(payload, area, target, mode, preference);
  const record = mode === "comparison" ? payload?.comparison : payload?.latest;
  if (!record) return { record: null, points: [] };
  const observed = new Map((payload.observations?.[area] || []).map(r => [Date.parse(r.time), r[target]]));
  const issued = Date.parse(payload.pairedCutoff || record.issuedAt);
  const forecast = new Map(record.times.map((t, i) => [Date.parse(t), record.areas[area][target][i]])
    .filter(([time]) => time > issued));
  const band=record.ranges?.areas?.[area]?.[target];
  const bounds=new Map(record.times.map((t,i)=>[Date.parse(t),{lower:band?.lower?.[i],upper:band?.upper?.[i]}]));
  const origin = Date.parse(record.origin);
  const start = mode === "comparison" ? Date.parse(record.times[0]) : origin - 24 * HOUR;
  const end = Date.parse(record.times.at(-1));
  const times = new Set([...forecast.keys(), ...observed.keys()]);
  const points = [...times].filter(t => t >= start && t <= end).sort((a,b) => a-b).map(time => ({time,
    observed: usable(observed.get(time)) ? observed.get(time) : null,
    predicted: usable(forecast.get(time)) ? forecast.get(time) : null,
    lower: time > issued && usable(bounds.get(time)?.lower) ? bounds.get(time).lower : null,
    upper: time > issued && usable(bounds.get(time)?.upper) ? bounds.get(time).upper : null}));
  return {record, points};
}

export function forecastStats(record, area, target, cutoff = record?.issuedAt) {
  if (!record) return null;
  const eligible = record.times.map((time, i) => ({ time, value: record.areas[area][target][i] }))
    .filter((point) => Date.parse(point.time) > Date.parse(cutoff));
  if (!eligible.length || !eligible.every((point) => usable(point.value))) return null;
  const peak = eligible.reduce((best, point) => point.value > best.value ? point : best);
  return { hours: eligible.length, average: eligible.reduce((sum, point) => sum + point.value, 0) / eligible.length,
    peak: peak.value, peakTime: peak.time };
}
