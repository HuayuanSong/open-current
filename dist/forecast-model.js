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
  if (mode !== "comparison" || target !== "wind" || !payload?.windOutlook) return current;
  const hasPrevious = Boolean(payload.comparison?.areas?.[area]?.wind);
  if (!(payload.windOutlook.latest || payload.windOutlook.latestByArea?.[area])) {
    const previous = hasPrevious && preference !== "current";
    return { ...(previous ? payload : payload.windOutlook), comparison: previous ? payload.comparison : null,
      comparisonSource: previous ? "previous" : "current", comparisonHasPrevious: hasPrevious };
  }
  const paired = payload.windComparison?.[area];
  if (paired?.comparison) {
    const previous = preference === "previous";
    const source = previous ? payload : current;
    const comparison = paired.comparison[previous ? "previous" : "current"];
    return { ...source, comparison, comparisonSource: previous ? "previous" : "current",
      alternateComparison: paired.comparison[previous ? "current" : "previous"],
      observations: payload.observations ?? current.observations,
      comparisonHasPrevious: true, pairedComparison: true, pairedCutoff: paired.comparison.issuedAfter };
  }
  const currentHours = matchedObservationCount(current, area, target);
  const previousHours = matchedObservationCount(payload, area, target);
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

function issuedHours(record, area, target, cutoff) {
  const hours = new Map();
  if (!record) return hours;
  const values = record.areas[area]?.[target];
  const range = record.ranges?.areas?.[area]?.[target];
  record.times.forEach((stamp, index) => {
    const time = Date.parse(stamp);
    if (time > cutoff) hours.set(time, {
      predicted: values?.[index], lower: range?.lower?.[index], upper: range?.upper?.[index],
    });
  });
  return hours;
}

export function chartSeries(payload, area, target, mode = "outlook", preference = "auto", includeAlternate = true) {
  payload = viewPayload(payload, area, target, mode, preference);
  const record = mode === "comparison" ? payload?.comparison : payload?.latest;
  if (!record) return { record: null, points: [] };
  const observed = new Map((payload.observations?.[area] || []).map(r => [Date.parse(r.time), r[target]]));
  const issued = Date.parse(payload.pairedCutoff || record.issuedAt);
  const forecast = issuedHours(record, area, target, issued);
  const alternate = includeAlternate && payload.alternateComparison
    ? issuedHours(payload.alternateComparison, area, target, issued) : null;
  const origin = Date.parse(record.origin);
  const start = mode === "comparison" ? Date.parse(record.times[0]) : origin - 24 * HOUR;
  const end = Date.parse(record.times.at(-1));
  const times = new Set([...forecast.keys(), ...observed.keys()]);
  const points = [...times].filter(t => t >= start && t <= end).sort((a,b) => a-b).map(time => {
    const hour = forecast.get(time);
    const point = {time, observed: usable(observed.get(time)) ? observed.get(time) : null,
      predicted: usable(hour?.predicted) ? hour.predicted : null,
      lower: usable(hour?.lower) ? hour.lower : null, upper: usable(hour?.upper) ? hour.upper : null};
    if (alternate) {
      const other = alternate.get(time);
      point.alternate = usable(other?.predicted) ? other.predicted : null;
      point.alternateUpper = usable(other?.upper) ? other.upper : null;
    }
    return point;
  });
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
