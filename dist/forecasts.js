import {
  forecastState,
  chartSeries,
  forecastStats,
  forecastTransition,
  HOUR,
  usable,
} from "./forecast-model.js?v=b0ffa21872d5";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const d3 = window.d3;
const state = {
  area: "DK1",
  target: "wind",
  view: "outlook",
  range: true,
  explainLead: 24,
};
try {
  state.range = localStorage.getItem("open-current-prediction-range") !== "off";
} catch {}
const names = {
  wind: "Wind generation",
  solar: "Solar generation",
  load: "Electricity demand",
};
let payload = null;
let activeIndex = 0;
const utcFormatter = new Intl.DateTimeFormat("en-GB", {
  day: "numeric",
  month: "short",
  hour: "2-digit",
  minute: "2-digit",
  timeZone: "UTC",
});

function utc(value) {
  return utcFormatter.format(new Date(value)) + " UTC";
}

function hourInterval(value) {
  const end = new Date(value);
  const start = new Date(+end - HOUR);
  return utcFormatter.formatRange(start, end) + " UTC · hourly average";
}

function power(value) {
  if (!usable(value)) return "—";
  if (value >= 1000) return `${(value / 1000).toFixed(2)} GW`;
  return `${Math.round(value)} MW`;
}

function text(selector, value) {
  $(selector).textContent = value;
}

function hasPredictionRange(point) {
  return usable(point.lower) && usable(point.upper);
}

function stat(label, value, note) {
  const element = document.createElement("div");
  for (const [tag, className, content] of [
    ["span", "metric-label", label],
    ["strong", "", value],
    ["span", "muted", note],
  ]) {
    const child = document.createElement(tag);
    child.className = className;
    child.textContent = content;
    element.append(child);
  }
  return element;
}

function render() {
  const status = forecastState(payload);
  $("#forecast-status").dataset.status = status;
  if (!payload?.latest) {
    text(
      "#forecast-status",
      "The next forecast is waiting for complete source data.",
    );
  } else {
    const missing = payload.latest.quality?.[state.area]?.filledInputHours || 0;
    text(
      "#forecast-status",
      `${status === "delayed" ? "Update delayed · Last forecast" : "Forecast issued"} ${utc(payload.latest.issuedAt)}${missing ? ` · ${missing} missing input hours filled from earlier readings` : ""}`,
    );
  }
  $$("[data-forecast-area]").forEach((button) =>
    button.setAttribute(
      "aria-pressed",
      String(button.dataset.forecastArea === state.area),
    ),
  );
  $$("[data-forecast-target]").forEach((button) =>
    button.setAttribute(
      "aria-pressed",
      String(button.dataset.forecastTarget === state.target),
    ),
  );
  $$("[data-forecast-view]").forEach((button) =>
    button.setAttribute(
      "aria-pressed",
      String(button.dataset.forecastView === state.view),
    ),
  );
  const { record, points } = chartSeries(
    payload,
    state.area,
    state.target,
    state.view,
  );
  $("#forecast-empty").hidden = Boolean(record);
  text(
    "#forecast-empty",
    state.view === "comparison"
      ? "We’re saving forecasts now. Comparisons will appear as the next hourly observations arrive."
      : "A forecast will appear when electricity and weather data are available.",
  );
  const stats = forecastStats(record, state.area, state.target);
  const live = payload?.performance?.[state.area]?.[state.target];
  $("#forecast-stats").replaceChildren(
    ...(stats
      ? [
          stat("48-hour average", power(stats.average), names[state.target]),
          stat("Forecast peak", power(stats.peak), utc(stats.peakTime)),
          stat(
            "Live mean error",
            live?.forecastHours >= 24
              ? power(live.maeMW)
              : "Collecting outcomes",
            live?.forecastHours
              ? `${live.forecastHours} forecasted hours · ${live.origins} issued outlooks · last 7 days`
              : "Measured after forecasts are issued",
          ),
        ]
      : []),
  );
  text(
    "#forecast-timing",
    record
      ? `${state.view === "comparison" ? `Comparing the forecast issued ${utc(record.issuedAt)}. ` : ""}Hourly average power · points mark hour ends · all times UTC. Weather run: ${utc(record.weatherRun)}.`
      : "Hourly average power · points mark hour ends · all times UTC.",
  );
  $("#forecast-download").disabled = !record;
  const hasRange = points.some(hasPredictionRange);
  $("#forecast-range-toggle").disabled = !hasRange;
  $("#forecast-range-toggle").checked = state.range && hasRange;
  $("#forecast-range-key").hidden = !(state.range && hasRange);
  const audit = payload?.rangeValidation?.series?.[state.area]?.[state.target];
  let rangeNote =
    "This outlook has no saved prediction range. Ranges are never added to old forecasts.";
  if (hasRange) {
    rangeNote = `Experimental, 80% target for each hour. ${state.target === "solar" ? "Daylight only; overnight ranges did not pass validation. " : ""}`;
    if (audit && record.ranges.version === payload.rangeValidation.version) {
      rangeNote += `Historical ${state.target === "solar" ? "daylight " : ""}coverage: ${Math.round(audit.coverage * 100)}% across ${audit.origins} forecast dates. `;
    }
    rangeNote +=
      live?.rangeHours >= 24
        ? `Live coverage: ${Math.round(live.rangeCoverage * 100)}% of ${live.rangeHours} forecasted hours from ${live.rangeOrigins} outlooks (last 7 days).`
        : "Live coverage: collecting outcomes.";
  }
  text("#forecast-range-note", rangeNote);
  renderExplanation(record);
  draw(record, points);
}

function renderExplanation(record) {
  const items = record?.explanations?.areas?.[state.area]?.load;
  const visible = state.target === "load" && Boolean(items?.length);
  $("#forecast-explanation").hidden = !visible;
  if (!visible) return;
  const select = $("#forecast-explanation-hour");
  select.replaceChildren();
  for (const item of items) {
    const option = document.createElement("option");
    option.value = item.lead;
    option.textContent = utc(record.times[item.lead - 1]);
    select.append(option);
  }
  if (!items.some((item) => item.lead === state.explainLead)) {
    state.explainLead = items[0].lead;
  }
  select.value = String(state.explainLead);
  const item = items.find((item) => item.lead === state.explainLead);
  const ready = item.status === "ready";
  text(
    "#forecast-explanation-status",
    ready
      ? "Positive contributions raise this model’s prediction; negative contributions lower it."
      : "This hour’s explanation changes too much with the reference sample to display reliably.",
  );
  $("#forecast-explanation-values").replaceChildren();
  $("#forecast-explanation-bars").replaceChildren();
  $(".forecast-explanation-caption").hidden = !ready;
  if (!ready) return;
  $("#forecast-explanation-values").append(
    stat("Reference prediction", power(item.baselineMW), "Tree model"),
    stat("Explained prediction", power(item.predictionMW), "Tree model"),
    stat(
      "Published forecast",
      power(record.areas[state.area].load[item.lead - 1]),
      "Full ensemble",
    ),
  );
  const maxContribution = Math.max(
    ...item.contributions.map((contribution) => Math.abs(contribution.mw)),
    1,
  );
  for (const contribution of [...item.contributions].sort(
    (a, b) => Math.abs(b.mw) - Math.abs(a.mw),
  )) {
    const row = document.createElement("div");
    row.className = "forecast-contribution";
    const name = document.createElement("span");
    name.textContent = contribution.name;
    const track = document.createElement("span");
    track.className = "forecast-contribution-track";
    track.setAttribute("aria-hidden", "true");
    const bar = document.createElement("i");
    const width = (Math.abs(contribution.mw) / maxContribution) * 50;
    bar.style.width = `${width}%`;
    bar.style.left = `${contribution.mw < 0 ? 50 - width : 50}%`;
    bar.className = contribution.mw < 0 ? "negative" : "positive";
    track.append(bar);
    const value = document.createElement("strong");
    value.textContent = `${contribution.mw < 0 ? "−" : "+"}${power(Math.abs(contribution.mw))}`;
    row.append(name, track, value);
    $("#forecast-explanation-bars").append(row);
  }
}

function draw(record, points) {
  const svg = d3.select("#forecast-chart");
  svg.selectAll("*").remove();
  $("#forecast-tooltip").hidden = true;
  svg.on("pointermove", null).on("pointerleave", null).on("keydown", null);
  const transition =
    record && state.view === "outlook"
      ? forecastTransition(points, record.origin)
      : null;
  const transitionKey = $("#forecast-transition-key");
  if (transitionKey) transitionKey.hidden = !transition;
  const width = $("#forecast-chart-wrap").clientWidth;
  const height = width < 600 ? 300 : 360;
  svg.attr("viewBox", `0 0 ${width} ${height}`);
  if (!record || !points.length) {
    activeIndex = 0;
    return;
  }
  svg
    .append("title")
    .text(`${names[state.target]} · ${state.area} · hourly power in gigawatts`);
  svg
    .append("desc")
    .text(
      `Solid line shows recorded readings. Dashed green line shows predictions.${transition ? " The faint dotted line is a visual transition between consecutive observed and forecast hours." : ""}${state.range ? " Shading shows the experimental 80%-target hourly prediction range, where available." : ""} Points mark hour ends. Use arrow keys to inspect each hourly average.`,
    );
  const margin = { top: 24, right: 24, bottom: 48, left: 58 };
  const x = d3
    .scaleUtc()
    .domain(d3.extent(points, (point) => new Date(point.time)))
    .range([margin.left, width - margin.right]);
  const maxPower =
    d3.max(points, (point) =>
      Math.max(
        point.observed ?? 0,
        point.predicted ?? 0,
        state.range ? (point.upper ?? 0) : 0,
      ),
    ) || 1;
  const y = d3
    .scaleLinear()
    .domain([0, (maxPower / 1000) * 1.08])
    .nice()
    .range([height - margin.bottom, margin.top]);
  const axis = svg.append("g").attr("class", "forecast-axis");
  axis
    .append("g")
    .attr("transform", `translate(${margin.left},0)`)
    .call(
      d3
        .axisLeft(y)
        .ticks(4)
        .tickSize(-(width - margin.left - margin.right))
        .tickFormat((d) => `${d}`),
    );
  axis
    .append("g")
    .attr("transform", `translate(0,${height - margin.bottom})`)
    .call(
      d3
        .axisBottom(x)
        .ticks(width < 600 ? 4 : 8)
        .tickFormat(d3.utcFormat("%d %b %Hh")),
    );
  svg
    .append("text")
    .attr("x", margin.left)
    .attr("y", 15)
    .attr("class", "forecast-unit")
    .text("GW");
  if (state.range && points.some(hasPredictionRange)) {
    const band = d3
      .area()
      .defined(hasPredictionRange)
      .x((point) => x(point.time))
      .y0((point) => y(point.lower / 1000))
      .y1((point) => y(point.upper / 1000));
    svg
      .append("path")
      .datum(points)
      .attr("d", band)
      .attr("class", "forecast-range-band");
  }
  const origin = Date.parse(record.origin);
  if (origin > points[0].time && origin < points.at(-1).time) {
    svg
      .append("line")
      .attr("x1", x(origin))
      .attr("x2", x(origin))
      .attr("y1", margin.top)
      .attr("y2", height - margin.bottom)
      .attr("class", "forecast-origin");
    svg
      .append("text")
      .attr("x", x(origin) + 8)
      .attr("y", margin.top + 13)
      .attr("class", "forecast-unit")
      .text("Outlook →");
  }
  for (const key of ["observed", "predicted"]) {
    const line = d3
      .line()
      .defined((point) => usable(point[key]))
      .x((point) => x(point.time))
      .y((point) => y(point[key] / 1000));
    svg
      .append("path")
      .datum(points)
      .attr("d", line)
      .attr("class", `forecast-line forecast-${key}`);
  }
  if (transition) {
    svg
      .append("path")
      .datum(transition)
      .attr("class", "forecast-transition")
      .attr(
        "d",
        d3
          .line()
          .x((point) => x(point.time))
          .y((point) => y(point.value / 1000)),
      )
      .append("title")
      .text(
        "Visual transition between consecutive observed and forecast hours",
      );
  }
  const cursor = svg
    .append("line")
    .attr("class", "forecast-cursor")
    .attr("y1", margin.top)
    .attr("y2", height - margin.bottom)
    .style("display", "none");
  function inspect(index) {
    activeIndex = Math.max(0, Math.min(points.length - 1, index));
    const point = points[activeIndex];
    cursor
      .attr("x1", x(point.time))
      .attr("x2", x(point.time))
      .style("display", null);
    const tooltip = $("#forecast-tooltip");
    tooltip.replaceChildren();
    const lines = [
      hourInterval(point.time),
      `Observed: ${power(point.observed)}`,
      `Forecast: ${power(point.predicted)}`,
    ];
    if (state.range && hasPredictionRange(point)) {
      lines.push(
        `80% target range: ${power(point.lower)} – ${power(point.upper)}`,
      );
    }
    if (transition?.some((endpoint) => endpoint.time === point.time)) {
      lines.push("Dotted link: visual transition only.");
    }
    for (const content of lines) {
      const row = document.createElement("div");
      row.textContent = content;
      tooltip.append(row);
    }
    tooltip.hidden = false;
    tooltip.style.left = `${Math.max(8, Math.min(width - tooltip.offsetWidth - 8, x(point.time) + 12))}px`;
  }
  svg.on("pointermove", (event) => {
    const [px] = d3.pointer(event);
    const stamp = +x.invert(px);
    inspect(d3.bisector((point) => point.time).center(points, stamp));
  });
  svg.on("pointerleave", () => {
    cursor.style("display", "none");
    $("#forecast-tooltip").hidden = true;
  });
  svg.on("keydown", (event) => {
    if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
      event.preventDefault();
      inspect(activeIndex + (event.key === "ArrowRight" ? 1 : -1));
    }
    if (event.key === "Escape") {
      $("#forecast-tooltip").hidden = true;
      cursor.style("display", "none");
    }
  });
}
$("#forecast-range-toggle").addEventListener("change", (event) => {
  state.range = event.target.checked;
  try {
    localStorage.setItem(
      "open-current-prediction-range",
      state.range ? "on" : "off",
    );
  } catch {}
  render();
});
$("#forecast-explanation-hour").addEventListener("change", (event) => {
  state.explainLead = Number(event.target.value);
  render();
});

for (const key of ["area", "target", "view"]) {
  $$(`[data-forecast-${key}]`).forEach((button) =>
    button.addEventListener("click", () => {
      state[key] =
        button.dataset[`forecast${key[0].toUpperCase() + key.slice(1)}`];
      render();
    }),
  );
}
$("#forecast-download").onclick = () => {
  const record =
    state.view === "comparison" ? payload.comparison : payload.latest;
  if (!record) return;
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(record, null, 2)], { type: "application/json" }),
  );
  const link = document.createElement("a");
  link.href = url;
  link.download = `open-current-forecast-${record.origin.slice(0, 10)}.json`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
async function refresh() {
  if (document.hidden) return;
  try {
    const response = await fetch("data/forecasts.json", { cache: "no-cache" });
    if (!response.ok) throw new Error(response.status);
    const next = await response.json();
    if (next.schema !== 1) throw new Error("schema");
    payload = next;
    render();
  } catch {
    if (payload) {
      render();
    } else {
      text(
        "#forecast-status",
        "Forecasts are temporarily unavailable. Please try again shortly.",
      );
      $("#forecast-status").dataset.status = "unavailable";
    }
  }
}
let resizeTimer;
new ResizeObserver(() => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(render, 80);
}).observe($("#forecast-chart-wrap"));
refresh();
setInterval(refresh, 5 * 60 * 1000);
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) refresh();
});
