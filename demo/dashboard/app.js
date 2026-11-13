// The live API (served by `python -m ds_greenland.demo`) and the saved copy of it
// written by `python -m ds_greenland.demo.snapshot`, which is used instead
// whenever the live one can't be reached.
const LIVE_API = "./api";
const SAVED_API = "./static-api";
// How long to wait for the live catalog before falling back to the saved copy.
const LIVE_TIMEOUT_MS = 20000;
let offline = false;
let useSaved = false; // set by the "Use saved copy" button
let savedMaps = {}; // offline only: experiment -> variable -> ["YYYY-MM-DD", ...]

const q = encodeURIComponent;
// Nested experiment names ("project/run1") are nested folders in the saved copy.
const savedPath = (name) => name.split("/").map(q).join("/");

const api = {
  catalog: (opts) => getJson(offline ? `${SAVED_API}/catalog.json` : `${LIVE_API}/catalog`, opts),
  repo: (opts) => getJson(offline ? `${SAVED_API}/repo.json` : `${LIVE_API}/repo`, opts),
  series: (experiment, variable) =>
    getJson(offline
      ? `${SAVED_API}/series/${savedPath(experiment)}/${q(variable)}.json`
      : `${LIVE_API}/series?experiment=${q(experiment)}&variable=${q(variable)}`),
  mapUrl: (experiment, variable, date, scale) =>
    offline
      ? `${SAVED_API}/maps/${savedPath(experiment)}/${q(variable)}/${date}${SAVED_MAP_SUFFIXES[scale]}.png`
      : `${LIVE_API}/map?experiment=${q(experiment)}&variable=${q(variable)}&date=${q(date)}` +
        `&scale=${q(scale)}`,
};

// Saved map file name suffix per colour scale; must match MAP_SCALE_SUFFIXES
// in src/ds_greenland/demo/snapshot.py.
const SAVED_MAP_SUFFIXES = { shared: "", full: "-full", date: "-date" };

function getJson(url, opts) {
  return fetch(url, opts).then(assertOk);
}

function assertOk(r) {
  if (!r.ok) return r.text().then((t) => { throw new Error(`${r.status} ${r.statusText}: ${t}`); });
  return r.json();
}

// The saved map date nearest `isoDate` for one experiment/variable.
function nearestSavedMapDate(experiment, variable, isoDate) {
  const dates = savedMaps[experiment]?.[variable] ?? [];
  return dates.length ? dates[nearestIndexForDate(dates, isoDate)] : null;
}

// Variables shown on first load, matched case-insensitively on long_name
// (the sidebar title).
const DEFAULT_VARIABLE_NAMES = ["Acc snow melt", "2-m Temperature"];

// Variables listed under "Output variable" in the sidebar, keyed
// case-insensitively on long_name, mapped to the title shown in the
// sidebar and chart headings; everything else is an input variable.
const OUTPUT_VARIABLES = new Map([["acc snow melt", "Surface melt"]]);

// Experiments that appear in the time-series charts but are never offered
// on the maps.
const MAP_EXCLUDED_EXPERIMENTS = new Set(["baseline_10year"]);

let catalog = null;
let selected = new Set();
let selectedVariables = new Set();
// long_name -> when it was last ticked in the sidebar, so newly added plots
// go to the top (just under the output variables). Defaults have no entry
// and keep catalog order below anything the user has added.
const variableSelectedAt = new Map();
let selectionCounter = 0;
let seriesCache = new Map(); // `${experiment}::${variable}` -> series payload
// Every series is shown as a running mean over this many points.
const SMOOTH_WINDOW = 11;
let mapState = new Map(); // group title -> { experiment, date }
// How maps' colour bars are chosen (the sidebar's "Map colour scale"):
// "shared" across dates by percentiles, "full" across dates by min to max,
// or "date" for each date's own; applies to every map.
let mapScale = "shared";
let mapRefreshers = []; // each rendered map's updateMapImg, for that choice

const $ = (sel) => document.querySelector(sel);
const errorBox = $("#error");
const outputVarList = $("#output-variable-list");
const inputVarList = $("#input-variable-list");
const plotsDiv = $("#plots");
const footerInfo = $("#footer-info");
const refreshBtn = $("#refresh-btn");
const savedBtn = $("#saved-btn");
const lightbox = $("#lightbox");
const lightboxImg = $("#lightbox-img");

function openLightbox(src, alt) {
  lightboxImg.src = src;
  lightboxImg.alt = alt || "";
  lightbox.classList.remove("hidden");
}
function closeLightbox() {
  lightbox.classList.add("hidden");
  lightboxImg.src = "";
}
lightbox.addEventListener("click", closeLightbox);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeLightbox();
});

function showError(msg) {
  errorBox.textContent =
    `Failed to load data from Arraylake: ${msg}. ` +
    "The dashboard needs the local demo server (run `uv run python -m ds_greenland.demo`), " +
    "or a saved copy (run `uv run python -m ds_greenland.demo.snapshot` while online).";
  errorBox.classList.remove("hidden");
}
function clearError() {
  errorBox.classList.add("hidden");
}

function isIsoDateString(t) {
  return typeof t === "string" && /^\d{4}-\d{2}-\d{2}/.test(t);
}

function rollingMean(values, window) {
  if (window <= 1) return values;
  const half = Math.floor(window / 2);
  const out = new Array(values.length).fill(null);
  for (let i = 0; i < values.length; i++) {
    let sum = 0, count = 0;
    for (let j = Math.max(0, i - half); j <= Math.min(values.length - 1, i + half); j++) {
      const v = values[j];
      if (v !== null && v !== undefined && !Number.isNaN(v)) { sum += v; count++; }
    }
    out[i] = count > 0 ? sum / count : null;
  }
  return out;
}

function selectedExperimentVariableKeys() {
  const keys = new Set();
  for (const exp of catalog.experiments) {
    if (selected.has(exp.name)) {
      for (const k of exp.variables) keys.add(k);
    }
  }
  return keys;
}

// Variables that represent the same physical quantity (e.g. "tas" on
// baseline and "tas_10yr-avg" on baseline_10year) share the same long_name,
// so grouping on it merges them into a single sidebar entry / chart without
// needing to know either dataset's naming scheme.
function variableGroups() {
  const groups = new Map();
  for (const v of catalog.variables) {
    const groupKey = v.long_name || v.key;
    if (!groups.has(groupKey)) {
      groups.set(groupKey, {
        title: OUTPUT_VARIABLES.get(groupKey.toLowerCase()) ?? groupKey,
        longName: groupKey,
        units: v.units,
        spatial_mean: v.spatial_mean,
        keys: new Set(),
      });
    }
    groups.get(groupKey).keys.add(v.key);
  }
  return [...groups.values()];
}

function renderVariableList() {
  outputVarList.innerHTML = "";
  inputVarList.innerHTML = "";
  const available = selectedExperimentVariableKeys();
  for (const k of [...selectedVariables]) {
    if (!available.has(k)) selectedVariables.delete(k);
  }
  const groups = variableGroups()
    .map((g) => ({ ...g, availableKeys: [...g.keys].filter((k) => available.has(k)) }))
    .filter((g) => g.availableKeys.length)
    .sort((a, b) => a.title.localeCompare(b.title, undefined, { sensitivity: "base", numeric: true }));
  const isOutput = (g) => OUTPUT_VARIABLES.has(g.longName.toLowerCase());
  for (const [list, label, inList] of [
    [outputVarList, "output", isOutput],
    [inputVarList, "input", (g) => !isOutput(g)],
  ]) {
    if (!groups.some(inList)) {
      list.innerHTML = `<p class="muted">No ${label} variables in the dataset.</p>`;
    }
  }
  for (const g of groups) {
    const varList = isOutput(g) ? outputVarList : inputVarList;
    const row = document.createElement("label");
    row.className = "experiment-row";
    const checked = g.availableKeys.some((k) => selectedVariables.has(k)) ? "checked" : "";
    row.innerHTML = `<input type="checkbox" ${checked}/> <span>${g.title}</span>`;
    row.querySelector("input").addEventListener("change", (e) => {
      for (const k of g.availableKeys) {
        if (e.target.checked) selectedVariables.add(k); else selectedVariables.delete(k);
      }
      if (e.target.checked) variableSelectedAt.set(g.longName, ++selectionCounter);
      loadAndRender();
    });
    varList.appendChild(row);
  }
}

async function ensureSeriesLoaded() {
  const jobs = [];
  for (const exp of catalog.experiments) {
    if (!selected.has(exp.name)) continue;
    for (const varKey of exp.variables) {
      if (!selectedVariables.has(varKey)) continue;
      const key = `${exp.name}::${varKey}`;
      if (!seriesCache.has(key)) {
        jobs.push(
          api.series(exp.name, varKey)
            .then((s) => seriesCache.set(key, s))
            .catch((err) => console.error(`series ${key}`, err))
        );
      }
    }
  }
  await Promise.all(jobs);
}

// Trace labels: "baseline_10year" carries 10-year-average variables, so
// tag its traces accordingly wherever they end up sharing a chart with
// baseline's raw series.
function traceLabel(experimentName) {
  return experimentName === "baseline_10year" ? `${experimentName} (10-year average)` : experimentName;
}

// Nearest index in `times` (an array of ISO-ish date strings) to `isoDate`.
function nearestIndexForDate(times, isoDate) {
  const target = Date.parse(isoDate);
  let bestIdx = 0, bestDiff = Infinity;
  for (let i = 0; i < times.length; i++) {
    const diff = Math.abs(Date.parse(String(times[i]).slice(0, 10)) - target);
    if (diff < bestDiff) { bestDiff = diff; bestIdx = i; }
  }
  return bestIdx;
}

function renderPlots() {
  plotsDiv.innerHTML = "";
  mapRefreshers = [];
  const groups = variableGroups()
    .map((g) => ({ ...g, activeKeys: [...g.keys].filter((k) => selectedVariables.has(k)) }))
    .filter((g) => g.activeKeys.length);
  // Output variables always come first, then the most recently ticked
  // variables; defaults (never ticked) keep catalog order (stable sort).
  const isOutput = (g) => OUTPUT_VARIABLES.has(g.longName.toLowerCase());
  groups.sort((a, b) =>
    (isOutput(b) - isOutput(a)) ||
    ((variableSelectedAt.get(b.longName) ?? 0) - (variableSelectedAt.get(a.longName) ?? 0))
  );
  if (!groups.length) {
    plotsDiv.innerHTML = `<p class="muted">Select one or more variables in the sidebar to plot them.</p>`;
    return;
  }

  // Every rendered chart registers itself here so a click on any one of them
  // can highlight the same date on all the others (and drive every map to
  // it), not just the chart that was clicked.
  const chartHandles = [];
  let selectedDateKey = null;

  // Move `h`'s highlight marker to the point on each of its traces nearest
  // `isoDate`, draw the date line, and (unless `mapExperiment` is given, in
  // which case also switch experiment) push the date to its map.
  function applyDateToChart(h, isoDate, mapExperiment) {
    const hx = [], hy = [];
    for (const meta of h.traceMeta) {
      if (!meta.time.length || !isIsoDateString(meta.time[0])) continue;
      const idx = nearestIndexForDate(meta.time, isoDate);
      hx.push(meta.time[idx]);
      hy.push(meta.y[idx]);
    }
    Plotly.restyle(h.gd, { x: [hx], y: [hy] }, [h.highlightIndex]);
    Plotly.relayout(h.gd, {
      shapes: hx.length ? [{
        type: "line", xref: "x", yref: "paper",
        x0: isoDate, x1: isoDate, y0: 0, y1: 1,
        line: { color: "rgba(230,237,243,0.35)", width: 1, dash: "dot" },
      }] : [],
    });
    if (h.mapRow) {
      if (mapExperiment) h.mapRow.setSelection(mapExperiment, isoDate);
      else h.mapRow.setDate(isoDate);
    }
  }

  function clearChartHighlight(h) {
    Plotly.restyle(h.gd, { x: [[]], y: [[]] }, [h.highlightIndex]);
    Plotly.relayout(h.gd, { shapes: [] });
  }

  for (const g of groups) {
    const container = document.createElement("div");
    container.className = "plot-card";
    const title = document.createElement("h3");
    title.textContent = `${g.title}${g.units ? ` (${g.units})` : ""}${g.spatial_mean ? " · domain mean" : ""}`;
    container.appendChild(title);
    const body = document.createElement("div");
    body.className = "plot-body";
    container.appendChild(body);
    const plotEl = document.createElement("div");
    plotEl.className = "plot";
    body.appendChild(plotEl);
    let mapRow = null;
    if (g.spatial_mean) {
      mapRow = buildMapRow(g);
      if (mapRow) body.appendChild(mapRow.el);
    }
    plotsDiv.appendChild(container);

    const traces = [];
    const traceMeta = []; // parallel to traces: which experiment/series each data trace came from
    for (const exp of catalog.experiments) {
      if (!selected.has(exp.name)) continue;
      // An experiment names its own variant of this group (e.g. "tas" vs
      // "tas_10yr-avg"); pick whichever of the group's keys actually
      // belongs to it so matching variables land on the same chart.
      const varKey = g.activeKeys.find((k) => exp.variables.includes(k));
      if (!varKey) continue;
      const s = seriesCache.get(`${exp.name}::${varKey}`);
      if (!s) continue;
      const y = rollingMean(s.values, SMOOTH_WINDOW);
      traces.push({ x: s.time, y, mode: "lines", name: traceLabel(exp.name) });
      traceMeta.push({ experiment: exp.name, time: s.time, y });
    }

    // A dedicated marker trace that click events move to the clicked point,
    // so the selected date/value stays visible. Clicking the same point again clears it.
    const highlightIndex = traces.length;
    traces.push({
      x: [], y: [], mode: "markers", type: "scatter",
      marker: { size: 11, color: "#f97316", line: { color: "#0f1720", width: 2 } },
      hoverinfo: "skip", showlegend: false,
    });

    Plotly.newPlot(plotEl, traces, {
      margin: { t: 10, r: 10, b: 40, l: 55 },
      xaxis: { title: "time" },
      yaxis: { title: g.units || "" },
      // Only one experiment is ever plotted, so a legend adds nothing (Plotly
      // would otherwise show one once the highlight marker gets a point).
      showlegend: false,
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
      font: { color: "#e6edf3" },
    }, { responsive: true, displayModeBar: false }).then((gd) => {
      const handle = { gd, highlightIndex, traceMeta, mapRow };
      chartHandles.push(handle);
      gd.on("plotly_click", (ev) => {
        const pt = ev.points.find((p) => p.curveNumber !== highlightIndex);
        if (!pt) return;
        const meta = traceMeta[pt.curveNumber];
        if (!meta) return;
        const idx = pt.pointIndex ?? pt.pointNumber;
        const iso = meta.time.length && isIsoDateString(meta.time[0]);
        // A real date can be matched across every other chart's own time
        // axis; a fallback sample index can't, so key + scope it to this
        // chart alone (`experiment:idx` keeps it unique per chart/click).
        const dateKey = iso ? String(meta.time[idx]).slice(0, 10) : `${meta.experiment}:${idx}`;

        if (selectedDateKey === dateKey) {
          for (const h of chartHandles) clearChartHighlight(h);
          selectedDateKey = null;
          return;
        }
        selectedDateKey = dateKey;

        if (iso) {
          for (const h of chartHandles) applyDateToChart(h, dateKey, h === handle ? meta.experiment : undefined);
        } else {
          Plotly.restyle(gd, { x: [[pt.x]], y: [[pt.y]] }, [highlightIndex]);
          Plotly.relayout(gd, {
            shapes: [{
              type: "line", xref: "x", yref: "paper",
              x0: pt.x, x1: pt.x, y0: 0, y1: 1,
              line: { color: "rgba(230,237,243,0.35)", width: 1, dash: "dot" },
            }],
          });
          if (mapRow) mapRow.setSelection(meta.experiment, String(idx));
        }
      });
    });
  }
}

// Experiments (among those selected) that actually carry a variable from
// group `g`, paired with which of the group's keys is theirs (e.g. "tas" vs
// "tas_10yr-avg" — see variableGroups()).
function experimentsForGroup(g) {
  const out = [];
  for (const exp of catalog.experiments) {
    if (!selected.has(exp.name)) continue;
    const varKey = g.activeKeys.find((k) => exp.variables.includes(k));
    if (varKey) out.push({ experiment: exp.name, variable: varKey });
  }
  return out;
}

// A static (server-rendered) lon/lat map of one date's snapshot for a
// variable group, alongside the group's time-series chart. Each group keeps
// its own { experiment, date } selection in `mapState` so switching tabs /
// re-rendering doesn't reset it.
function buildMapRow(g) {
  const options = experimentsForGroup(g).filter((o) =>
    !MAP_EXCLUDED_EXPERIMENTS.has(o.experiment) &&
    // The saved copy may only have maps for some variables (`--map-variables`).
    (!offline || savedMaps[o.experiment]?.[o.variable]));
  if (!options.length) return null;

  let ms = mapState.get(g.title);
  if (!ms || !options.some((o) => o.experiment === ms.experiment)) {
    ms = { experiment: options[0].experiment, date: null };
    mapState.set(g.title, ms);
  }

  const currentVar = () => options.find((o) => o.experiment === ms.experiment).variable;
  const seriesForCurrent = () => seriesCache.get(`${ms.experiment}::${currentVar()}`);

  const s = seriesForCurrent();
  const dateIsIso = s && s.time.length && isIsoDateString(s.time[0]);
  if (ms.date === null && s && s.time.length) {
    ms.date = dateIsIso ? s.time[0].slice(0, 10) : String(s.time[0]);
  }

  const row = document.createElement("div");
  row.className = "map-row";

  const controls = document.createElement("div");
  controls.className = "map-controls";

  let expSelect = null;
  if (options.length > 1) {
    expSelect = document.createElement("select");
    for (const o of options) {
      const opt = document.createElement("option");
      opt.value = o.experiment;
      opt.textContent = traceLabel(o.experiment);
      if (o.experiment === ms.experiment) opt.selected = true;
      expSelect.appendChild(opt);
    }
    expSelect.addEventListener("change", (e) => {
      ms.experiment = e.target.value;
      const ns = seriesForCurrent();
      if (ns && ns.time.length) {
        ms.date = dateIsIso ? ns.time[0].slice(0, 10) : String(ns.time[0]);
      }
      updateMapImg();
    });
    controls.appendChild(expSelect);
  }

  const dateInput = document.createElement("input");
  if (dateIsIso) {
    dateInput.type = "date";
    dateInput.min = s.time[0].slice(0, 10);
    dateInput.max = s.time[s.time.length - 1].slice(0, 10);
  } else if (s && s.time.length) {
    dateInput.type = "range";
    dateInput.min = "0";
    dateInput.max = String(s.time.length - 1);
  }
  dateInput.value = ms.date ?? "";
  dateInput.addEventListener("change", (e) => {
    ms.date = e.target.value;
    updateMapImg();
  });
  controls.appendChild(dateInput);
  row.appendChild(controls);

  const imgWrap = document.createElement("div");
  imgWrap.className = "map-img-wrap";
  row.appendChild(imgWrap);

  const img = document.createElement("img");
  img.className = "map-img";
  img.alt = `${g.title} map`;
  img.addEventListener("click", () => openLightbox(img.src, img.alt));
  imgWrap.appendChild(img);

  const spinner = document.createElement("div");
  spinner.className = "map-spinner";
  imgWrap.appendChild(spinner);

  // Dim the outgoing image and show a spinner while a new date/experiment's
  // map is loading, rather than clearing it (which would flash blank and
  // collapse the column) or leaving it looking up to date when it isn't.
  const stopLoading = () => {
    img.classList.remove("loading");
    spinner.classList.remove("visible");
  };
  img.addEventListener("load", stopLoading);
  img.addEventListener("error", stopLoading);

  function updateMapImg() {
    if (ms.date === null || ms.date === "") return;
    if (offline && dateIsIso) {
      // Only some dates were saved: show (and say) which one is mapped.
      ms.date = nearestSavedMapDate(ms.experiment, currentVar(), ms.date);
      dateInput.value = ms.date;
    }
    img.classList.add("loading");
    spinner.classList.add("visible");
    img.src = api.mapUrl(ms.experiment, currentVar(), ms.date, mapScale);
  }
  updateMapImg();
  mapRefreshers.push(updateMapImg);

  // Lets other UI (e.g. clicking a point on the time-series chart) drive the
  // map to a specific experiment/date without reaching into this closure.
  function setSelection(experiment, date) {
    // Clicking a point from an experiment that isn't mapped (e.g.
    // baseline_10year) still moves this map to that date.
    if (!options.some((o) => o.experiment === experiment)) {
      setDate(date);
      return;
    }
    ms.experiment = experiment;
    ms.date = date;
    if (expSelect) expSelect.value = experiment;
    dateInput.value = date;
    updateMapImg();
  }

  // Same as setSelection but keeps whichever experiment is already chosen
  // for this map — used when a date picked on another chart propagates here.
  // A no-op if this map's current experiment isn't on a real (ISO) date
  // axis, since its date/index values then aren't comparable to one.
  function setDate(date) {
    if (!dateIsIso) return;
    ms.date = date;
    dateInput.value = date;
    updateMapImg();
  }

  return { el: row, setSelection, setDate };
}

async function loadAndRender() {
  renderVariableList();
  await ensureSeriesLoaded();
  renderPlots();
}

// Load the catalog and repo info from the live API, falling back to the saved
// copy (and switching every later request to it) if that fails — or straight
// from the saved copy if the "Use saved copy" button chose it.
async function loadCatalog() {
  if (useSaved) return loadSaved();
  offline = false;
  const opts = { signal: AbortSignal.timeout(LIVE_TIMEOUT_MS) };
  try {
    return await Promise.all([api.catalog(opts), api.repo(opts)]);
  } catch (liveErr) {
    console.warn("live data unavailable, trying the saved copy", liveErr);
    try {
      return await loadSaved();
    } catch {
      offline = false;
      throw liveErr;
    }
  }
}

async function loadSaved() {
  offline = true;
  const [c, r, maps] = await Promise.all([
    api.catalog(), api.repo(), getJson(`${SAVED_API}/maps/index.json`),
  ]);
  savedMaps = maps;
  return [c, r];
}

async function refresh() {
  refreshBtn.disabled = true;
  savedBtn.disabled = true;
  refreshBtn.textContent = "Refreshing…";
  try {
    const [c, r] = await loadCatalog();
    catalog = c;
    seriesCache = new Map();

    selected = new Set(catalog.experiments.map((e) => e.name));
    if (selectedVariables.size === 0) {
      // Match on long_name so every experiment's key for the same quantity
      // (e.g. "tas" and "tas_10yr-avg") is picked up.
      const wanted = new Set(DEFAULT_VARIABLE_NAMES.map((n) => n.toLowerCase()));
      const chosen = new Set(
        catalog.variables
          .filter((v) => wanted.has((v.long_name || v.key).toLowerCase()))
          .map((v) => v.key)
      );
      if (chosen.size === 0) {
        // None of the defaults exist in this dataset: seed with each selected
        // experiment's own first variable so every checked experiment (e.g.
        // "baseline_10year", whose variables are all named differently from
        // "baseline"'s) shows up, then fill out to 4 from the combined pool.
        for (const exp of catalog.experiments) {
          if (selected.has(exp.name) && exp.variables.length) chosen.add(exp.variables[0]);
        }
        const availableKeys = selectedExperimentVariableKeys();
        for (const v of catalog.variables) {
          if (chosen.size >= 4) break;
          if (availableKeys.has(v.key)) chosen.add(v.key);
        }
      }
      selectedVariables = chosen;
    }
    footerInfo.innerHTML =
      `Data: <code>${r.repo}</code>${r.root_path ? ` · <code>${r.root_path}</code>` : ""} · snapshot ` +
      `<code>${r.snapshot.id.slice(0, 12)}</code> · ${new Date(r.snapshot.written_at).toLocaleString()}` +
      (offline ? ` · saved copy from ${new Date(r.saved_at).toLocaleString()}` : "");
    document.getElementById("live-label").hidden = offline;
    clearError();
    await loadAndRender();
  } catch (e) {
    showError(e.message || String(e));
  } finally {
    refreshBtn.disabled = false;
    savedBtn.disabled = false;
    refreshBtn.textContent = "Refresh data";
    savedBtn.textContent = offline ? "Use live data" : "Use saved copy";
  }
}

refreshBtn.addEventListener("click", refresh);

$("#map-scale").addEventListener("change", (e) => {
  mapScale = e.target.value;
  for (const update of mapRefreshers) update();
});

// Switch between live data and the saved copy. From an automatic fallback
// (offline but not chosen), this retries live data.
savedBtn.addEventListener("click", () => {
  useSaved = !offline;
  refresh();
});

// Only offer the switch when a saved copy exists.
fetch(`${SAVED_API}/repo.json`, { method: "HEAD" })
  .then((r) => { savedBtn.hidden = !r.ok; })
  .catch(() => {});

refresh();
