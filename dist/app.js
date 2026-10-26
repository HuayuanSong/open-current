import {
  GROUPS,
  valid,
  formatNumber as fmt,
  powerText,
  dateText,
  ageText,
  readingAt,
  priceAt,
  summaryAt,
  metricValue,
  escapeHTML as esc,
  parseState,
} from "./model.js?v=ce1e63c9b335";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const d3 = window.d3;
const state = {
  country: "dk",
  metric: "lowCarbon",
  time: null,
  chart: "power",
  panel: "mix",
  zone: null,
  showAll: false,
  flows: true,
};
let index,
  geography,
  danish,
  countryData,
  mapWidth,
  mapHeight,
  projection,
  zoom,
  transform = d3.zoomIdentity,
  hours = [],
  playTimer,
  toastTimer,
  countryRequest = 0;
const countryCache = new Map();
const svg = d3.select("#map");
const defs = svg.append("defs");
defs
  .append("marker")
  .attr("id", "flow-arrow")
  .attr("viewBox", "0 -4 8 8")
  .attr("refX", 7)
  .attr("refY", 0)
  .attr("markerWidth", 6)
  .attr("markerHeight", 6)
  .attr("orient", "auto")
  .append("path")
  .attr("d", "M0,-3L7,0L0,3")
  .attr("fill", "none")
  .attr("stroke", "#204d45")
  .attr("stroke-width", 1.4);
const land = svg.append("g");
const labelLayer = svg.append("g").attr("aria-hidden", "true");
const flowLayer = svg.append("g").attr("aria-hidden", "true");
const lowScale = d3
  .scaleLinear()
  .domain([0, 25, 50, 75, 100])
  .range(["#e6e9c5", "#bbce91", "#7eaa71", "#43825a", "#154b42"])
  .clamp(true);
const netScale = d3
  .scaleLinear()
  .domain([-12000, 0, 12000])
  .range(["#638c91", "#e7e9d4", "#bb815a"])
  .clamp(true);
const metrics = {
  lowCarbon: "Low-carbon generation",
  renewable: "Renewable generation",
  net: "Net imports",
};
const hourMs = 3600000;

async function getJSON(path) {
  const response = await fetch(path, { cache: "no-cache" });
  if (!response.ok) throw new Error(`${path}: ${response.status}`);
  return response.json();
}
function toast(message) {
  $("#toast").textContent = message;
  $("#toast").hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => ($("#toast").hidden = true), 2700);
}
function hashState() {
  const p = new URLSearchParams({
    country: state.country,
    metric: state.metric,
    chart: state.chart,
  });
  if (state.time !== null) p.set("time", state.time);
  if (state.zone) p.set("zone", state.zone);
  history.replaceState(null, "", "#" + p);
}
function formatMetric(value, metric = state.metric) {
  return valid(value)
    ? metric === "net"
      ? `${value > 0 ? "+" : ""}${powerText(value)}`
      : `${fmt(value, 1)}%`
    : "No reading";
}
function selectedSummary(code = state.country) {
  return summaryAt(index.countries[code], state.time);
}
function timeFor(metric, summary) {
  return metric === "net" ? summary?.flows?.t : summary?.power?.t;
}
function isStale(time) {
  return state.time === null && valid(time) && Date.now() - time > 6 * hourMs;
}

function initMap() {
  zoom = d3
    .zoom()
    .scaleExtent([0.7, 14])
    .filter(
      (event) => !event.button && (event.type !== "wheel" || !event.ctrlKey),
    )
    .on("zoom", (event) => {
      transform = event.transform;
      land.attr("transform", transform);
      renderMapOverlay();
      $("#map-tooltip").hidden = true;
    });
  svg.call(zoom).on("dblclick.zoom", null);
  land
    .selectAll("path")
    .data(geography.features)
    .join("path")
    .attr(
      "class",
      (d) =>
        "country-shape" +
        (index.countries[d.properties.code] ? "" : " unavailable"),
    )
    .attr("data-country", (d) => d.properties.code)
    .on("click", (event, d) => {
      if (!event.defaultPrevented && index.countries[d.properties.code])
        selectCountry(d.properties.code);
    })
    .on("mousemove", (event, d) => mapTooltip(event, d))
    .on("mouseleave", () => ($("#map-tooltip").hidden = true));
  resizeMap();
  $("#zoom-in").onclick = () =>
    svg.transition().duration(220).call(zoom.scaleBy, 1.4);
  $("#zoom-out").onclick = () =>
    svg
      .transition()
      .duration(220)
      .call(zoom.scaleBy, 1 / 1.4);
  $("#zoom-reset").onclick = () =>
    svg.transition().duration(350).call(zoom.transform, d3.zoomIdentity);
  new ResizeObserver(() => resizeMap()).observe($("#map-stage"));
}
function resizeMap() {
  const width = $("#map-stage").clientWidth,
    height = $("#map-stage").clientHeight;
  if (!width || (width === mapWidth && height === mapHeight)) return;
  mapWidth = width;
  mapHeight = height;
  svg.attr("viewBox", `0 0 ${width} ${height}`);
  projection = d3.geoMercator().scale(1).translate([0, 0]);
  const a = projection([-13, 34]),
    b = projection([35.5, 71.5]);
  const scale = Math.min(
    (width - 30) / (b[0] - a[0]),
    (height - 64) / (a[1] - b[1]),
  );
  projection
    .scale(scale)
    .translate([
      width / 2 - (scale * (a[0] + b[0])) / 2,
      height / 2 + 15 - (scale * (a[1] + b[1])) / 2,
    ]);
  land.selectAll("path").attr("d", d3.geoPath(projection));
  zoom.extent([
    [0, 0],
    [width, height],
  ]);
  renderMap();
}
function renderMap() {
  if (!projection) return;
  land
    .selectAll("path")
    .attr("fill", (d) => {
      const c = index.countries[d.properties.code];
      if (!c) return "#d6ddd6";
      const summary = summaryAt(c, state.time),
        value = metricValue(summary, state.metric);
      return valid(value)
        ? state.metric === "net"
          ? netScale(value)
          : lowScale(value)
        : "#d6ddd6";
    })
    .classed("selected", (d) => d.properties.code === state.country);
  $("#map-title").textContent =
    state.metric === "net"
      ? "Net cross-border power flow"
      : `Share of ${state.metric === "lowCarbon" ? "low-carbon" : "renewable"} generation`;
  $("#map-subtitle").textContent =
    `${state.time === null ? "Latest available readings" : "Last reading in the selected UTC hour"}${state.metric === "net" ? " · imports minus exports" : " · storage excluded"}`;
  $("#legend-low").textContent =
    state.metric === "net" ? "−12 GW · export" : "0%";
  $("#legend-high").textContent =
    state.metric === "net" ? "+12 GW · import" : "100%";
  $("#legend-ramp").style.background =
    state.metric === "net"
      ? "linear-gradient(to right,#638c91,#e7e9d4,#bb815a)"
      : "";
  $$("[data-metric]").forEach((b) =>
    b.setAttribute("aria-pressed", b.dataset.metric === state.metric),
  );
  renderMapOverlay();
}
function screenPoint(code) {
  const coords = index.countries[code]?.coordinates;
  return coords ? transform.apply(projection(coords)) : null;
}
function renderMapOverlay() {
  if (!projection || !index) return;
  const prominent = [
    "dk",
    "de",
    "fr",
    "es",
    "pt",
    "it",
    "pl",
    "se",
    "no",
    "fi",
    "uk",
    "ie",
    "ro",
    "gr",
    "nl",
    "at",
    "ch",
  ];
  const codes = Object.keys(index.countries).filter(
    (code) =>
      code === state.country || transform.k > 1.8 || prominent.includes(code),
  );
  const labels = labelLayer
    .selectAll("g")
    .data(codes, (d) => d)
    .join("g")
    .attr("transform", (code) => `translate(${screenPoint(code)})`);
  labels.each(function (code) {
    const group = d3.select(this),
      c = selectedSummary(code),
      value = metricValue(c, state.metric);
    const countryName = index.countries[code].name;
    group
      .selectAll("text")
      .data([0, 1])
      .join("text")
      .attr("class", (n) => (n ? "map-label map-label-value" : "map-label"))
      .attr("y", (n) => (n ? 13 : 0))
      .style("font-weight", code === state.country ? 700 : 400)
      .text((n) =>
        n
          ? valid(value)
            ? state.metric === "net"
              ? `${value > 0 ? "+" : ""}${fmt(value / 1000, 1)} GW`
              : `${fmt(value, 0)}%`
            : "—"
          : countryName === "United Kingdom"
            ? "UK"
            : countryName,
      );
  });
  flowLayer.selectAll("*").remove();
  if (!state.flows || !countryData || countryData.code !== state.country)
    return;
  const flow = readingAt(countryData.flows, state.time);
  if (!flow) return;
  const selected = screenPoint(state.country);
  for (const border of flow.borders) {
    if (!border.code || !valid(border.mw) || Math.abs(border.mw) < 5) continue;
    const other = screenPoint(border.code);
    if (!other) continue;
    const [from, to] = border.mw > 0 ? [other, selected] : [selected, other];
    const dx = to[0] - from[0],
      dy = to[1] - from[1],
      length = Math.hypot(dx, dy);
    if (length < 25) continue;
    const nx = -dy / length,
      ny = dx / length,
      offset = Math.min(35, length * 0.15);
    const start = [from[0] + (dx / length) * 14, from[1] + (dy / length) * 14],
      end = [to[0] - (dx / length) * 15, to[1] - (dy / length) * 15];
    const control = [
      (from[0] + to[0]) / 2 + nx * offset,
      (from[1] + to[1]) / 2 + ny * offset,
    ];
    flowLayer
      .append("path")
      .attr("class", "flow-path")
      .attr("d", `M${start} Q${control} ${end}`)
      .attr("marker-end", "url(#flow-arrow)")
      .style(
        "stroke-width",
        Math.max(1, Math.min(3, Math.sqrt(Math.abs(border.mw) / 600))),
      );
    if (length > 70)
      flowLayer
        .append("text")
        .attr("class", "flow-label")
        .attr("x", (from[0] + to[0]) / 2 + nx * (offset / 2 + 8))
        .attr("y", (from[1] + to[1]) / 2 + ny * (offset / 2 + 8))
        .text(powerText(Math.abs(border.mw)));
  }
}
function mapTooltip(event, feature) {
  const c = index.countries[feature.properties.code],
    tip = $("#map-tooltip");
  if (!c)
    tip.innerHTML = `<strong>${esc(feature.properties.name)}</strong><small>No feed included in this atlas.</small>`;
  else {
    const s = summaryAt(c, state.time),
      t = timeFor(state.metric, s);
    tip.innerHTML = `<strong>${esc(c.name)}</strong>${esc(formatMetric(metricValue(s, state.metric)))}<small>${esc(dateText(t))}${isStale(t) ? " · delayed" : ""}</small><small>Click to explore</small>`;
  }
  const box = $("#map-stage").getBoundingClientRect();
  tip.style.left =
    Math.min(mapWidth - 235, Math.max(8, event.clientX - box.left + 14)) + "px";
  tip.style.top =
    Math.min(mapHeight - 100, Math.max(8, event.clientY - box.top - 20)) + "px";
  tip.hidden = false;
}
function initTimeline() {
  const last =
    Math.floor(new Date(index.generatedAt).getTime() / hourMs) * hourMs;
  hours = Array.from({ length: 168 }, (_, i) => last - (167 - i) * hourMs);
  if (state.time !== null && !hours.includes(state.time)) state.time = null;
  $("#time-start").textContent = dateText(hours[0], {
    hour: undefined,
    minute: undefined,
  }).replace(" UTC", "");
  $("#time-end").textContent = dateText(hours.at(-1), {
    hour: undefined,
    minute: undefined,
  }).replace(" UTC", "");
  renderTimeline();
}
function renderTimeline() {
  $("#time-label").textContent =
    state.time === null
      ? "Latest available"
      : dateText(state.time).replace(" UTC", "") +
        "–" +
        String(new Date(state.time).getUTCHours()).padStart(2, "0") +
        ":59 UTC";
  $("#time-slider").value =
    state.time === null
      ? hours.length - 1
      : Math.max(0, hours.indexOf(state.time));
  $("#latest-time").classList.toggle("active", state.time === null);
}
function setTime(time) {
  state.time = time;
  hashState();
  renderTimeline();
  renderMap();
  renderCountry();
  renderRanking();
  renderDanish();
}
function stopPlaying() {
  clearInterval(playTimer);
  playTimer = null;
  $("#play-time").textContent = "▶";
  $("#play-time").setAttribute("aria-label", "Play historical map");
}
function play() {
  if (playTimer) {
    stopPlaying();
    return;
  }
  if (state.time === null || state.time >= hours.at(-1)) setTime(hours[0]);
  $("#play-time").textContent = "Ⅱ";
  $("#play-time").setAttribute("aria-label", "Pause historical map");
  playTimer = setInterval(() => {
    const i = hours.indexOf(state.time);
    if (i >= hours.length - 1) {
      stopPlaying();
      return;
    }
    setTime(hours[i + 1]);
  }, 650);
}

async function selectCountry(code, writeHash = true) {
  if (!index.countries[code]) return;
  if (state.country !== code) state.zone = null;
  state.country = code;
  if (writeHash) hashState();
  const request = ++countryRequest;
  $("#country-select").value = code;
  $("#country-code").textContent = code.toUpperCase();
  $("#country-time").textContent = "Loading readings…";
  $("#chart-country").textContent = "/ " + index.countries[code].name;
  if (!countryCache.has(code)) {
    $("#mix-list").innerHTML =
      '<p class="empty-note">Loading country readings…</p>';
    $("#detail-value").textContent = "—";
    $("#generation-total").textContent = "—";
    $("#country-story").textContent = "";
    $("#mix-strip").replaceChildren();
    $("#flows-list").replaceChildren();
  }
  renderMap();
  renderRanking();
  try {
    const data =
      countryCache.get(code) || (await getJSON(`data/countries/${code}.json`));
    countryCache.set(code, data);
    if (request !== countryRequest) return;
    countryData = data;
    renderCountry();
    renderMapOverlay();
    renderChart();
    renderDanish();
  } catch (error) {
    if (request !== countryRequest) return;
    countryData = null;
    $("#country-time").textContent = "Could not load this country";
    $("#mix-list").innerHTML =
      '<p class="empty-note">The country file could not be loaded. Choose another country or reload to retry.</p>';
    toast("Country data unavailable. Please try again.");
  }
}
function renderCountry() {
  if (!countryData || countryData.code !== state.country) return;
  const power = readingAt(countryData.power, state.time),
    flow = readingAt(countryData.flows, state.time);
  const summary = { power, flows: flow },
    value = metricValue(summary, state.metric),
    time = timeFor(state.metric, summary);
  $("#detail-metric-label").textContent = metrics[state.metric];
  $("#detail-value").innerHTML = valid(value)
    ? state.metric === "net"
      ? `${value > 0 ? "+" : ""}${fmt(value / 1000, 2)}<span>GW</span>`
      : `${fmt(value, 1)}<span>%</span>`
    : "—";
  $("#generation-total").textContent = powerText(power?.total);
  $("#country-time").textContent = valid(time)
    ? `${dateText(time)}${state.time === null ? " · " + ageText(time) : ""}`
    : "No reading in this period";
  $("#country-time").classList.toggle("stale", isStale(time));
  const fuels = Object.entries(power?.g || {})
    .filter(([key]) => key !== "storage")
    .sort((a, b) => (b[1] ?? -1) - (a[1] ?? -1));
  const largest = fuels.find(([, v]) => valid(v) && v > 0);
  let story = "No complete generation reading is available for this period.";
  if (largest && valid(power.total) && power.total > 0)
    story = `${GROUPS[largest[0]].name} supplies ${fmt((100 * largest[1]) / power.total, 0)}% of reported generation.`;
  if (valid(flow?.net))
    story += ` ${flow.net >= 0 ? "Importing" : "Exporting"} ${powerText(Math.abs(flow.net))} net at ${new Date(flow.t).toISOString().slice(11, 16)} UTC.`;
  if (isStale(time)) story = "Delayed reading. " + story;
  $("#country-story").textContent = story;
  $("#mix-strip").innerHTML =
    valid(power?.total) && power.total > 0
      ? fuels
          .filter(([, v]) => valid(v) && v > 0)
          .map(
            ([key, v]) =>
              `<span style="width:${(100 * v) / power.total}%;background:${GROUPS[key].color}" title="${esc(GROUPS[key].name)} ${fmt((100 * v) / power.total, 1)}%"></span>`,
          )
          .join("")
      : "";
  const mixRows = [...fuels];
  if (valid(power?.g?.storage) && power.g.storage > 0)
    mixRows.push(["storage", power.g.storage]);
  $("#mix-list").innerHTML = mixRows.length
    ? mixRows
        .map(
          ([key, v]) =>
            `<div class="mix-row"><i style="background:${GROUPS[key].color}"></i><span>${esc(GROUPS[key].name)}</span><span class="mix-value">${esc(powerText(v))}</span><span class="mix-percent">${key !== "storage" && valid(v) && power.total > 0 ? fmt((100 * v) / power.total, 0) + "%" : "—"}</span></div>`,
        )
        .join("")
    : '<p class="empty-note">No generation data available for this period.</p>';
  $("#mix-note").textContent =
    power?.missing?.length && !valid(power.total)
      ? `${power.missing.length} source series missing. Shares are withheld rather than treating missing data as zero.`
      : power?.missing?.length
        ? "Storage data are incomplete. Generation shares exclude storage discharge."
        : "Reported generation. Shares exclude storage discharge. Totals may differ from consumption.";
  $("#flow-time").textContent = flow
    ? `Physical flows · ${dateText(flow.t)}${isStale(flow.t) ? " · delayed" : ""}`
    : "No border-flow reading in this period.";
  $("#flows-list").innerHTML = flow
    ? flow.borders
        .slice()
        .sort((a, b) => Math.abs(b.mw || 0) - Math.abs(a.mw || 0))
        .map(
          (b) =>
            `<button class="flow-row ${b.mw > 0 ? "import" : "export"}" ${b.code && index.countries[b.code] ? `data-neighbour="${b.code}"` : "disabled"}><span>${esc(b.name)}</span><span>${valid(b.mw) ? esc(powerText(Math.abs(b.mw))) : "No reading"}</span><span class="direction" aria-label="${valid(b.mw) ? (b.mw > 0 ? "Import" : b.mw < 0 ? "Export" : "No flow") : "Unavailable"}">${valid(b.mw) ? (b.mw > 0 ? "↙" : b.mw < 0 ? "↗" : "–") : "—"}</span></button>`,
        )
        .join("")
    : '<p class="empty-note">Exchange data are published independently from generation data.</p>';
  $$("[data-neighbour]").forEach(
    (button) =>
      (button.onclick = () => selectCountry(button.dataset.neighbour)),
  );
  renderPanel();
  renderPrice();
  renderChart();
}
function renderPanel() {
  $("#mix-panel").hidden = state.panel !== "mix";
  $("#flows-panel").hidden = state.panel !== "flows";
  $$("[data-panel]").forEach((b) =>
    b.setAttribute("aria-pressed", b.dataset.panel === state.panel),
  );
}
function renderPrice() {
  const zones = Object.keys(countryData?.prices || {});
  if (!zones.includes(state.zone)) state.zone = zones[0] || null;
  $("#price-zone").innerHTML = zones.length
    ? zones
        .map(
          (z) =>
            `<option value="${esc(z)}">${esc(countryData.prices[z].label)} · ${esc(z)}</option>`,
        )
        .join("")
    : "<option>No open price feed</option>";
  $("#price-zone").disabled = !zones.length;
  if (state.zone) $("#price-zone").value = state.zone;
  const row = priceAt(
    countryData?.prices[state.zone],
    state.time === null ? Date.now() : state.time,
  );
  $("#price-current").innerHTML = row
    ? `${fmt(row.value, 2)}<small>€/MWh</small>`
    : "—";
  $("#price-time").textContent = row
    ? new Date(row.t).toISOString().slice(11, 16) + " UTC · wholesale"
    : zones.length
      ? "No price for this interval"
      : "Coverage varies by bidding zone";
}

function chartSeries() {
  if (!countryData) return [];
  if (state.chart === "price")
    return [
      {
        label: state.zone || "Price",
        color: "#a06c43",
        key: "price",
        rows: (countryData.prices[state.zone]?.rows || []).map((r) => ({
          t: r.t,
          v: r.value,
        })),
      },
    ];
  if (state.chart === "share")
    return [
      {
        label: "Low-carbon",
        color: "#245b49",
        key: "lowCarbon",
        rows: countryData.power.map((r) => ({ t: r.t, v: r.lowCarbon })),
      },
      {
        label: "Renewables",
        color: "#a2a754",
        key: "renewable",
        rows: countryData.power.map((r) => ({ t: r.t, v: r.renewable })),
      },
    ];
  return [
    {
      label: "Generation",
      color: "#245b49",
      key: "generation",
      rows: countryData.power.map((r) => ({
        t: r.t,
        v: valid(r.total) ? r.total / 1000 : null,
      })),
    },
    {
      label: "Demand",
      color: "#ac825d",
      key: "load",
      rows: countryData.power.map((r) => ({
        t: r.t,
        v: valid(r.load) ? r.load / 1000 : null,
      })),
    },
    {
      label: "Wind",
      color: "#91a990",
      key: "wind",
      rows: countryData.power.map((r) => ({
        t: r.t,
        v: valid(r.g.wind) ? r.g.wind / 1000 : null,
      })),
    },
  ];
}
function withGaps(rows) {
  const output = [];
  // Source resolutions vary. Use the observed modal interval, not an assumed 15 min.
  const differences = d3
    .rollups(
      rows.slice(1).map((r, i) => r.t - rows[i].t),
      (v) => v.length,
      (d) => d,
    )
    .sort((a, b) => b[1] - a[1]);
  const interval = differences[0]?.[0] || 900000;
  rows.forEach((r, i) => {
    if (i && r.t - rows[i - 1].t > interval * 1.5)
      output.push({ t: rows[i - 1].t + interval, v: null });
    output.push(r);
  });
  return output;
}
function renderChart() {
  if (!countryData) return;
  $$("[data-chart]").forEach((b) =>
    b.setAttribute("aria-pressed", b.dataset.chart === state.chart),
  );
  $("#chart-country").textContent = "/ " + countryData.name;
  $("#chart-caption").textContent =
    state.chart === "power"
      ? "Reported generation and demand, in gigawatts."
      : state.chart === "share"
        ? "Share of reported generation, excluding storage."
        : `Day-ahead wholesale price${state.zone ? " · " + state.zone : ""}, in EUR/MWh.`;
  const series = chartSeries();
  $("#chart-legend").innerHTML = series
    .map((s) => `<span style="color:${s.color}"><i></i>${esc(s.label)}</span>`)
    .join("");
  $("#history-note").textContent =
    state.chart === "price"
      ? "Includes published day-ahead prices. Excludes retail charges and taxes."
      : "Gaps mean unavailable readings. Generation excludes storage discharge.";
  const chart = d3.select("#history-chart");
  chart.selectAll("*").remove();
  const all = series.flatMap((s) => s.rows.filter((r) => valid(r.v)));
  $("#chart-empty").hidden = all.length > 0;
  if (!all.length) return;
  const width = $("#chart-wrap").clientWidth,
    height = $("#chart-wrap").clientHeight,
    margin = { left: 40, right: 12, top: 16, bottom: 28 };
  chart.attr("viewBox", `0 0 ${width} ${height}`);
  const extent = d3.extent(all, (r) => r.t);
  if (extent[0] === extent[1]) extent[1] += hourMs;
  const x = d3
    .scaleUtc()
    .domain(extent.map((t) => new Date(t)))
    .range([margin.left, width - margin.right]);
  const ymin =
      state.chart === "share"
        ? 0
        : Math.min(
            0,
            d3.min(all, (r) => r.v),
          ),
    ymax =
      state.chart === "share"
        ? 100
        : Math.max(1, d3.max(all, (r) => r.v) * 1.08);
  const y = d3
    .scaleLinear()
    .domain([ymin, ymax])
    .nice()
    .range([height - margin.bottom, margin.top]);
  chart
    .append("g")
    .attr("class", "chart-axis")
    .attr("transform", `translate(${margin.left},0)`)
    .call(
      d3
        .axisLeft(y)
        .ticks(4)
        .tickSize(-(width - margin.left - margin.right))
        .tickPadding(9),
    );
  chart
    .append("g")
    .attr("class", "chart-axis")
    .attr("transform", `translate(0,${height - margin.bottom})`)
    .call(
      d3
        .axisBottom(x)
        .ticks(width < 500 ? 4 : 8)
        .tickSize(0)
        .tickPadding(12)
        .tickFormat(d3.utcFormat("%d %b")),
    );
  if (state.chart === "power") {
    const s = series[0];
    chart
      .append("path")
      .datum(withGaps(s.rows))
      .attr("fill", "#c7d6bd")
      .attr("opacity", 0.36)
      .attr(
        "d",
        d3
          .area()
          .defined((r) => valid(r.v))
          .x((r) => x(r.t))
          .y0(y(0))
          .y1((r) => y(r.v)),
      );
  }
  series.forEach((s) =>
    chart
      .append("path")
      .datum(withGaps(s.rows))
      .attr("class", "chart-line")
      .attr("stroke", s.color)
      .attr("stroke-dasharray", s.key === "load" ? "5 4" : null)
      .attr(
        "d",
        d3
          .line()
          .defined((r) => valid(r.v))
          .x((r) => x(r.t))
          .y((r) => y(r.v)),
      ),
  );
  if (state.time !== null && state.time >= extent[0] && state.time <= extent[1])
    chart
      .append("line")
      .attr("x1", x(state.time))
      .attr("x2", x(state.time))
      .attr("y1", margin.top)
      .attr("y2", height - margin.bottom)
      .attr("stroke", "#455f4e")
      .attr("stroke-dasharray", "3 3");
  const cross = chart
    .append("line")
    .attr("stroke", "#8e9e8e")
    .attr("y1", margin.top)
    .attr("y2", height - margin.bottom)
    .attr("visibility", "hidden");
  chart
    .append("rect")
    .attr("x", margin.left)
    .attr("y", margin.top)
    .attr("width", width - margin.left - margin.right)
    .attr("height", height - margin.top - margin.bottom)
    .attr("fill", "transparent")
    .on("mousemove", (event) => {
      const [px] = d3.pointer(event),
        t = +x.invert(px),
        values = series.map((s) => {
          const i = d3.bisector((r) => r.t).center(s.rows, t);
          const r = s.rows[i];
          return { s, r: r && Math.abs(r.t - t) < hourMs ? r : null };
        });
      const observation = values.find((v) => v.r)?.r;
      cross.attr("x1", px).attr("x2", px).attr("visibility", "visible");
      const tip = $("#chart-tooltip");
      tip.innerHTML =
        `<strong>${esc(dateText(observation?.t ?? t))}</strong>` +
        values
          .map(
            ({ s, r }) =>
              `${esc(s.label)}: ${valid(r?.v) ? fmt(r.v, 1) + (state.chart === "power" ? " GW" : state.chart === "share" ? "%" : " €/MWh") : "No reading"}`,
          )
          .join("<br>");
      tip.style.left = Math.max(2, Math.min(width - 210, px + 12)) + "px";
      tip.style.top = "12px";
      tip.hidden = false;
    })
    .on("mouseleave", () => {
      cross.attr("visibility", "hidden");
      $("#chart-tooltip").hidden = true;
    })
    .on("click", (event) => {
      const t = Math.floor(+x.invert(d3.pointer(event)[0]) / hourMs) * hourMs;
      if (hours.includes(t)) {
        stopPlaying();
        setTime(t);
      }
    });
}
function renderDanish() {
  $("#denmark-section").hidden = state.country !== "dk";
  if (!danish) return;
  $("#danish-areas").innerHTML = Object.entries(danish.areas)
    .map(([area, data]) => {
      const p = readingAt(data.power, state.time),
        c = readingAt(data.co2, state.time);
      const share =
        valid(p?.wind) && p.total > 0 ? (100 * p.wind) / p.total : null;
      return `<article class="danish-card"><p class="eyebrow">${area} · ENERGINET</p><h3>${area === "DK1" ? "West Denmark" : "East Denmark"}</h3><div class="co2-value">${fmt(c?.value, 0)}<small>g CO₂/kWh</small></div><p class="co2-caption">Operational consumption intensity</p><div class="danish-mini"><span>Wind <strong>${valid(share) ? fmt(share, 0) + "%" : "—"}</strong></span><span>Generation <strong>${esc(powerText(p?.total))}</strong></span></div><time class="${isStale(c?.t) ? "stale" : ""}">CO₂: ${esc(dateText(c?.t))}${isStale(c?.t) ? " · delayed" : ""}<br>Power: ${esc(dateText(p?.t))}</time></article>`;
    })
    .join("");
}
function renderRanking() {
  if (!index) return;
  const metric = $("#ranking-metric").value;
  const rows = Object.entries(index.countries)
    .map(([code, c]) => {
      const s = summaryAt(c, state.time);
      return {
        code,
        name: c.name,
        value: metricValue(s, metric),
        time: timeFor(metric, s),
      };
    })
    .sort(
      (a, b) =>
        (b.value ?? -Infinity) - (a.value ?? -Infinity) ||
        a.name.localeCompare(b.name),
    );
  const visible = state.showAll ? rows : rows.slice(0, 9),
    max =
      metric === "total" || metric === "net"
        ? Math.max(1, ...rows.map((r) => Math.abs(r.value || 0)))
        : 100;
  $("#compare-note").textContent =
    state.time === null
      ? "Latest reported readings. Reporting times vary; hover over a country for its timestamp."
      : "Last reported reading within the selected UTC hour. Countries with gaps remain unavailable.";
  $("#ranking-grid").innerHTML = visible
    .map(
      (r, i) =>
        `<button class="ranking-row ${r.code === state.country ? "selected" : ""}" data-rank-country="${r.code}" title="${esc(dateText(r.time))}${isStale(r.time) ? " · delayed" : ""}"><span class="rank-no">${String(i + 1).padStart(2, "0")}</span><span class="ranking-name">${esc(r.name)}</span><span class="ranking-bar"><i style="width:${Math.min(100, (100 * Math.abs(r.value || 0)) / max)}%"></i></span><span class="ranking-value">${valid(r.value) ? (metric === "total" || metric === "net" ? fmt(r.value / 1000, 1) + "<small> GW</small>" : fmt(r.value, 0) + "%") : "—"}</span></button>`,
    )
    .join("");
  $$("[data-rank-country]").forEach(
    (button) =>
      (button.onclick = () => {
        selectCountry(button.dataset.rankCountry);
        $("#explorer").scrollIntoView({
          behavior: matchMedia("(prefers-reduced-motion: reduce)").matches
            ? "auto"
            : "smooth",
          block: "start",
        });
      }),
  );
  $("#show-all").textContent = state.showAll
    ? "Show fewer countries ↑"
    : `Show all ${rows.length} countries ↓`;
}
function bindEvents() {
  $("#country-select").onchange = (event) => selectCountry(event.target.value);
  $$("[data-metric]").forEach(
    (b) =>
      (b.onclick = () => {
        state.metric = b.dataset.metric;
        hashState();
        renderMap();
        renderCountry();
      }),
  );
  $$("[data-panel]").forEach(
    (b) =>
      (b.onclick = () => {
        state.panel = b.dataset.panel;
        renderPanel();
      }),
  );
  $$("[data-chart]").forEach(
    (b) =>
      (b.onclick = () => {
        state.chart = b.dataset.chart;
        hashState();
        renderChart();
      }),
  );
  $("#show-flows").onchange = (event) => {
    state.flows = event.target.checked;
    renderMapOverlay();
  };
  $("#time-slider").oninput = (event) => {
    stopPlaying();
    setTime(hours[Number(event.target.value)]);
  };
  $("#latest-time").onclick = () => {
    stopPlaying();
    setTime(null);
  };
  $("#play-time").onclick = play;
  $("#price-zone").onchange = (event) => {
    state.zone = event.target.value;
    hashState();
    renderPrice();
    renderChart();
  };
  $("#ranking-metric").onchange = renderRanking;
  $("#show-all").onclick = () => {
    state.showAll = !state.showAll;
    renderRanking();
  };
  $("#share-link").onclick = async () => {
    hashState();
    try {
      await navigator.clipboard.writeText(location.href);
      toast("Link copied to this view.");
    } catch {
      toast("This view is saved in your address bar.");
    }
  };
  $$("[data-about]").forEach(
    (b) => (b.onclick = () => $("#about-dialog").showModal()),
  );
  $("#close-about").onclick = () => $("#about-dialog").close();
  $("#about-dialog").addEventListener("click", (event) => {
    if (event.target === $("#about-dialog")) {
      const r = event.target.getBoundingClientRect();
      if (
        event.clientX < r.left ||
        event.clientX > r.right ||
        event.clientY < r.top ||
        event.clientY > r.bottom
      )
        event.target.close();
    }
  });
  $("#download-data").onclick = () => {
    if (!countryData) return;
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(countryData, null, 2)], {
        type: "application/json",
      }),
    );
    const a = document.createElement("a");
    a.href = url;
    a.download = `grid-in-motion-${state.country}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  window.addEventListener("hashchange", () => {
    if (!location.hash.includes("=")) return;
    stopPlaying();
    Object.assign(
      state,
      parseState(location.hash, Object.keys(index.countries)),
    );
    initTimeline();
    selectCountry(state.country, false);
  });
  let resize;
  new ResizeObserver(() => {
    clearTimeout(resize);
    resize = setTimeout(renderChart, 80);
  }).observe($("#chart-wrap"));
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) stopPlaying();
  });
}
function renderCollection() {
  const complete = Object.values(index.countries).filter((c) =>
    valid(c.power?.lowCarbon),
  ).length;
  $("#coverage").textContent = `${complete} countries with a complete mix`;
  $("#collection-time").textContent =
    "Snapshot prepared " + dateText(new Date(index.generatedAt).getTime());
  const hoursOld =
    (Date.now() - new Date(index.generatedAt).getTime()) / hourMs;
  if (hoursOld > 2) {
    $("#load-error").textContent =
      `This snapshot was collected ${ageText(new Date(index.generatedAt).getTime())}. Automatic collection may be delayed. Every panel shows the time of its source reading.`;
    $("#load-error").hidden = false;
  } else $("#load-error").hidden = true;
}
async function refresh() {
  if (document.hidden) return;
  try {
    const next = await getJSON("data/index.json");
    if (next.generatedAt === index.generatedAt) {
      renderCollection();
      return;
    }
    index = next;
    countryCache.clear();
    danish = await getJSON("data/denmark.json");
    initTimeline();
    renderCollection();
    await selectCountry(state.country, false);
  } catch {
    /* The current snapshot remains usable during an interrupted refresh. */
  }
}
async function start() {
  try {
    [index, geography, danish] = await Promise.all([
      getJSON("data/index.json"),
      getJSON("data/europe.geojson"),
      getJSON("data/denmark.json"),
    ]);
    Object.assign(
      state,
      parseState(location.hash, Object.keys(index.countries)),
    );
    $("#country-select").innerHTML = Object.entries(index.countries)
      .sort((a, b) => a[1].name.localeCompare(b[1].name))
      .map(([code, c]) => `<option value="${code}">${esc(c.name)}</option>`)
      .join("");
    initTimeline();
    initMap();
    bindEvents();
    renderCollection();
    await selectCountry(state.country, false);
    setInterval(refresh, 300000);
  } catch (error) {
    $("#load-error").textContent =
      "The electricity snapshot could not be loaded. Check your connection and reload the page. Source data and collection scripts are available on GitHub.";
    $("#load-error").hidden = false;
    $("#coverage").textContent = "Snapshot unavailable";
    console.error(error);
  }
}
start();
