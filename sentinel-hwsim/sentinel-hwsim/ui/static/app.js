// Sentinel M3 hardware console. Vanilla ES module, no dependencies.
// Live data arrives over Server-Sent Events (/api/stream); commands are JSON POSTs.
// All device data is rendered with textContent, never innerHTML (payloads can be hostile).

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

function h(tag, props = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "text") el.textContent = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) if (c != null && c !== false) el.append(c);
  return el;
}

// ------------------------------------------------------------------ status
const STATUS = {
  ONLINE: ["ok", "Online"], HEALTHY: ["ok", "Healthy"],
  DEGRADED: ["warn", "Degraded"], THROTTLED: ["warn", "Throttled"], RESTARTING: ["warn", "Restarting"],
  RECONNECTING: ["warn", "Reconnecting"], STARTING: ["muted", "Starting"], CONNECTING: ["muted", "Connecting"],
  OFFLINE: ["crit", "Offline"], AUTH_FAILED: ["crit", "Auth failed"], RUNTIME_DOWN: ["crit", "Runtime down"],
  INFERENCE_DEGRADED: ["crit", "Inference stalled"], PIPE_DISCONNECTED: ["crit", "Pipe disconnected"],
  NODE_OFFLINE: ["crit", "Node offline"],
};
const GLYPH = { ok: "●", warn: "▲", crit: "■", muted: "○" };
const tone = (s) => (STATUS[s] ?? ["muted"])[0];
const label = (s) => (STATUS[s] ?? [null, s ?? "—"])[1];

function pill(state) {
  const t = tone(state);
  return h("span", { class: `pill pill--${t}` }, h("span", { class: "pill__glyph", "aria-hidden": "true", text: GLYPH[t] }), label(state));
}
function setPill(slot, state) {
  if (slot.dataset.state === state) return;   // avoid churn (and screen-reader noise) on every tick
  slot.dataset.state = state;
  slot.replaceChildren(pill(state));
}

// ------------------------------------------------------------------ format
const timeFmt = new Intl.DateTimeFormat(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
const fmtTime = (iso) => (iso ? timeFmt.format(new Date(iso)) : "—");
const fmtNum = (n, d = 1) => (n == null ? "—" : Number(n).toFixed(d));
const signed = (n) => (n >= 0 ? "+" : "−") + Math.abs(n).toFixed(2);
function fmtDuration(s) {
  if (s == null) return "—";
  if (s < 90) return `${Math.round(s)} s`;
  const m = s / 60, hrs = m / 60, d = hrs / 24;
  if (d >= 1) return `${Math.floor(d)} d ${Math.floor(hrs % 24)} h`;
  if (hrs >= 1) return `${Math.floor(hrs)} h ${Math.floor(m % 60)} m`;
  return `${Math.floor(m)} m`;
}
function eventDetails(ev) {
  const a = ev.attributes ?? {};
  const frag = document.createDocumentFragment();
  if (a.plate) frag.append(h("span", { class: "chip", text: a.plate }), " ");
  const rest = Object.entries(a)
    .filter(([k, v]) => !["plate", "bbox"].includes(k) && v != null)
    .map(([k, v]) => (k === "riders" ? `${v} rider${v === 1 ? "" : "s"}` : String(v).toLowerCase().replaceAll("_", " ")));
  frag.append(rest.join(" · "));
  return frag;
}

// ------------------------------------------------------------------- store
const store = {
  meta: null, cameras: new Map(), node: null, events: [], alerts: new Map(), dlq: [],
  selectedCam: null, view: null, chaos: new Map(), ready: false,
};
const MAX_EVENTS = 300;

// --------------------------------------------------------------- announce
function announce(msg, urgent = false) {
  const region = urgent ? $("#announce-assertive") : $("#announce-polite");
  region.textContent = "";
  setTimeout(() => { region.textContent = msg; }, 60);   // re-set so repeats are re-announced
}
function notify(msg, kind = "ok") {
  const box = $("#toasts");
  while (box.children.length >= 3) box.firstElementChild.remove();
  const t = h("div", { class: `toast${kind === "error" ? " toast--error" : ""}`, text: msg });
  box.append(t);
  setTimeout(() => { t.classList.add("toast--out"); setTimeout(() => t.remove(), 250); }, 4500);
  announce(msg, kind === "error");
}

// --------------------------------------------------------------------- api
async function api(path, body) {
  const opts = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const res = await fetch(path, opts);
  const isJson = (res.headers.get("content-type") ?? "").includes("json");
  const data = isJson ? await res.json() : await res.blob();
  if (!res.ok) throw Object.assign(new Error(data?.message ?? res.statusText), { code: data?.error ?? res.status });
  return data;
}

// Loading state: aria-busy drives the spinner; aria-disabled (not `disabled`) keeps focus on the button.
async function busy(btn, fn) {
  if (btn.getAttribute("aria-busy") === "true") return;
  btn.setAttribute("aria-busy", "true");
  btn.setAttribute("aria-disabled", "true");
  try { return await fn(); }
  finally { btn.removeAttribute("aria-busy"); btn.removeAttribute("aria-disabled"); }
}

async function command(btn, device, action, body = {}, { quiet = false } = {}) {
  return busy(btn, async () => {
    try {
      const r = await api(`/api/devices/${device}/${action}`, body);
      if (!quiet) notify(r.message);
      return r;
    } catch (e) {
      notify(`${e.code}: ${e.message}`, "error");
      throw e;
    }
  });
}

// ------------------------------------------------------------------ router
const TITLES = { overview: "Overview", cameras: "Cameras", node: "AI node", events: "Events", trace: "Wire trace" };

function route() {
  const m = location.hash.match(/^#\/([a-z]+)(?:\/([A-Z0-9-]+))?$/);
  const view = m && TITLES[m[1]] ? m[1] : "overview";
  const changed = view !== store.view;
  if (view === "cameras" && m?.[2] && store.cameras.has(m[2])) selectCamera(m[2]);
  store.view = view;

  for (const s of $$("[data-view]")) s.hidden = s.dataset.view !== view;
  for (const a of $$("[data-route]")) {
    if (a.dataset.route === view) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  }
  document.title = `${view === "cameras" && store.selectedCam ? store.selectedCam : TITLES[view]} · Sentinel console`;

  if (changed && route.started) {           // move focus to the new page heading (not on first load)
    window.scrollTo({ top: 0 });
    $(`[data-view="${view}"] h1`).focus({ preventScroll: true });
  }
  route.started = true;
  if (view === "events") renderEventTable();
  if (view !== "cameras") feedVideo.pause();
  trace.toggle(view === "trace");
  renderAll();
}

// --------------------------------------------------------------- live feed
function setLive(on) {
  const el = $("#live");
  el.classList.toggle("live--on", on);
  el.classList.toggle("live--off", !on);
  $("#live-label").textContent = on ? "Live" : "Reconnecting…";
}

function connect() {
  const es = new EventSource("/api/stream");
  es.addEventListener("open", () => setLive(true));
  es.addEventListener("error", () => setLive(false));
  es.addEventListener("state", (e) => hydrate(JSON.parse(e.data)));
  es.addEventListener("camera", (e) => { const c = JSON.parse(e.data); store.cameras.set(c.id, c); schedule(); });
  es.addEventListener("node", (e) => { store.node = JSON.parse(e.data); schedule(); });
  es.addEventListener("ai_event", (e) => onEvent(JSON.parse(e.data)));
  es.addEventListener("dlq", (e) => { store.dlq.push(JSON.parse(e.data)); renderDlq(); });
  es.addEventListener("alert", (e) => {
    const a = JSON.parse(e.data);
    const isNew = !store.alerts.has(a.id);
    store.alerts.set(a.id, a);
    if (isNew) announce(`${a.severity === "critical" ? "Critical alert" : "Warning"}: ${a.message}`, a.severity === "critical");
    else if (a.resolved_at) announce(`Resolved: ${a.device} ${label(a.state)}`);
    renderAlerts();
    schedule();
  });
}

function hydrate(s) {
  // full resync (first connect and after every reconnect)
  store.meta = s.meta;
  store.cameras = new Map(s.cameras.map((c) => [c.id, c]));
  store.node = s.node;
  store.events = s.events;
  store.alerts = new Map(s.alerts.map((a) => [a.id, a]));
  store.dlq = s.dlq;
  if (!store.ready) {
    store.ready = true;
    buildCameraLists();
    selectCamera(store.selectedCam ?? s.cameras[0]?.id);
    $("#main").removeAttribute("aria-busy");
    route();
  }
  const m = s.meta;
  $("#footer-meta").textContent = `simulation ${m.speed}× real time · seed ${m.seed} · started ${fmtTime(m.started_at)}`;
  renderAlerts();
  renderLatest();
  renderEventTable();
  renderDlq();
  schedule();
}

let frame = 0;
function schedule() {
  if (!frame) frame = requestAnimationFrame(() => { frame = 0; renderAll(); });
}
function renderAll() {
  if (!store.ready) return;
  renderKpis();
  renderCameraLists();
  if (store.view === "cameras") renderCameraDetail();
  if (store.view === "node") renderNode();
}

// ---------------------------------------------------------------- overview
function kpi(id, value, note, crit = false) {
  const el = $(id);
  el.classList.toggle("kpi--crit", crit);
  $(".kpi__value", el).textContent = value;
  $(".kpi__note", el).textContent = note;
}

function renderKpis() {
  const cams = [...store.cameras.values()];
  const up = cams.filter((c) => ["ONLINE", "DEGRADED"].includes(c.health)).length;
  const down = cams.filter((c) => tone(c.health) === "crit");
  const degraded = cams.filter((c) => c.health === "DEGRADED").length;
  kpi("#kpi-cameras", `${up} / ${cams.length}`,
    down.length ? `${down.map((c) => c.id).join(", ")} down` : degraded ? `${degraded} degraded` : "All streams reachable",
    down.length > 0);

  const n = store.node, h = n?.ai_health;
  kpi("#kpi-node", label(n?.health), h ? `${fmtNum(h.inference_fps)} fps · ${fmtNum(h.temp_c)} °C` : "No telemetry",
    tone(n?.health) === "crit");
  kpi("#kpi-events", String(n?.events_last_min ?? 0), `${n?.events_received ?? 0} received since start`);

  const all = [...store.alerts.values()];
  const open = all.filter((a) => !a.resolved_at).length;
  kpi("#kpi-alerts", String(open), `${all.length - open} resolved`, open > 0);
}

function renderAlerts() {
  const list = $("#alert-list");
  const alerts = [...store.alerts.values()].sort((a, b) => b.id - a.id);
  $("#alert-empty").hidden = alerts.length > 0;
  list.replaceChildren(...alerts.map((a) => {
    const kind = a.resolved_at ? "resolved" : a.severity;
    const icon = a.resolved_at ? "✓" : a.severity === "critical" ? "■" : "▲";
    const sev = a.resolved_at ? "Resolved" : a.severity === "critical" ? "Critical" : "Warning";
    return h("li", { class: `alert alert--${kind}` },
      h("span", { class: "alert__icon", "aria-hidden": "true", text: icon }),
      h("p", { class: "alert__msg" }, h("span", { class: "visually-hidden", text: `${sev}: ` }), a.message),
      h("p", { class: "alert__time", text: `Raised ${fmtTime(a.at)}${a.resolved_at ? ` · resolved ${fmtTime(a.resolved_at)}` : ""}` }));
  }));
}

function eventRow(ev, { model = false, fresh = false } = {}) {
  const hit = ev.event_type === "WATCHLIST_HIT";
  return h("tr", { class: [fresh && "row--new", hit && "row--hit"].filter(Boolean).join(" ") || null },
    h("td", { class: "nowrap", text: fmtTime(ev.occurred_at) }),
    h("td", { class: "nowrap", text: ev.camera_external_id }),
    h("td", { class: "nowrap" }, hit ? h("span", { class: "pill pill--crit", text: "WATCHLIST HIT" }) : ev.event_type),
    h("td", {}, eventDetails(ev)),
    h("td", { class: "num", text: `${Math.round(ev.confidence * 100)}%` }),
    model ? h("td", { class: "mono nowrap", text: ev.model_version }) : null);
}

function renderLatest() {
  const rows = store.events.slice(-8).reverse().map((ev) => eventRow(ev));
  $("#latest-events").replaceChildren(...rows);
  $("#latest-empty").hidden = rows.length > 0;
}

function onEvent(ev) {
  store.events.push(ev);
  if (store.events.length > MAX_EVENTS) store.events.shift();
  const latest = $("#latest-events");
  latest.prepend(eventRow(ev, { fresh: true }));
  while (latest.rows.length > 8) latest.deleteRow(-1);
  $("#latest-empty").hidden = true;
  if (store.view === "events" && !$("#f-pause").checked && matches(ev)) {
    const tb = $("#event-rows");
    tb.prepend(eventRow(ev, { model: true, fresh: true }));
    while (tb.rows.length > 100) tb.deleteRow(-1);
    $("#event-empty").hidden = true;
    updateFeedCount();
  }
  if (ev.event_type === "WATCHLIST_HIT") notify(`Watchlist hit: ${ev.attributes?.plate ?? "?"} on ${ev.camera_external_id}`, "error");
}

// ----------------------------------------------------------------- cameras
const camEls = new Map();   // id -> { tile, tilePill, tileMeta, pick, pickPill }

function buildCameraLists() {
  const tiles = [], picks = [];
  for (const id of store.cameras.keys()) {
    const tilePill = h("span"), tileMeta = h("span", { class: "tile__meta" });
    const tile = h("li", {}, h("a", { class: "tile", href: `#/cameras/${id}` },
      h("span", { class: "tile__head" }, h("span", { class: "tile__id", text: id }), tilePill), tileMeta));
    const pickPill = h("span");
    const link = h("a", { class: "picker__item", href: `#/cameras/${id}` }, h("span", { text: id }), pickPill);
    tiles.push(tile); picks.push(h("li", {}, link));
    camEls.set(id, { tilePill, tileMeta, link, pickPill });
    store.chaos.set(id, { offline: false, revoke: false });
  }
  $("#camera-tiles").replaceChildren(...tiles);
  $("#camera-picker").replaceChildren(...picks);
  const opts = (sel, ids) => $(sel).append(...ids.map((id) => h("option", { value: id, text: id })));
  opts("#f-cam", [...store.cameras.keys()]);
  opts("#t-dev", [...store.cameras.keys(), store.node.id]);
}

function renderCameraLists() {
  for (const [id, c] of store.cameras) {
    const els = camEls.get(id);
    if (!els) continue;
    setPill(els.tilePill, c.health);
    setPill(els.pickPill, c.health);
    const s = c.status;
    els.tileMeta.textContent = tone(c.health) === "crit"
      ? (c.last_error?.code ?? "unreachable")
      : s ? `${fmtNum(s.fps)} fps · ${Math.round(s.bitrate_kbps)} kbps` : "Connecting…";
    if (id === store.selectedCam) els.link.setAttribute("aria-current", "page");
    else els.link.removeAttribute("aria-current");
  }
}

function selectCamera(id) {
  if (!id || id === store.selectedCam && $("#cam-title").textContent === id) return;
  store.selectedCam = id;
  $("#enc-error").textContent = "";
  $$("#enc-form .input").forEach((i) => i.removeAttribute("aria-invalid"));
  const ptz = store.cameras.get(id)?.ptz;
  for (const k of ["pan", "tilt", "zoom"]) setRange(`#ptz-${k}`, ptz?.[k] ?? 0);
  $("#cam-streams").dataset.sig = "";
}

function setRange(sel, v) {
  const input = $(sel);
  input.value = v;
  input.nextElementSibling.textContent = Number(v).toFixed(2);
}

function renderCameraDetail() {
  const id = store.selectedCam, c = store.cameras.get(id);
  if (!c) return;
  $("#cam-title").textContent = id;
  setPill($("#cam-pill"), c.health);
  const info = c.info;
  $("#cam-sub").textContent = info
    ? `${info.vendor} ${info.model} · firmware ${info.firmware} · serial ${info.serial}`
    : "Identity not read yet";

  const errEl = $("#cam-error");
  errEl.hidden = !c.last_error;
  if (c.last_error) errEl.replaceChildren("Last error ", h("strong", { text: c.last_error.code }), ` at ${fmtTime(c.last_error.at)}: ${c.last_error.message}`);

  const s = c.status;
  $("#cam-fps").textContent = s ? `${fmtNum(s.fps)} fps` : "—";
  $("#cam-bitrate").textContent = s ? `${Math.round(s.bitrate_kbps)} kbps` : "—";
  $("#cam-uptime").textContent = s ? fmtDuration(s.uptime_s) : "—";
  $("#cam-state").textContent = s?.state?.toLowerCase() ?? "—";
  $("#cam-errors").textContent = String(c.errors);

  // streams: rebuild only when they change, so a focused Copy button is not destroyed
  const dl = $("#cam-streams"), sig = JSON.stringify(c.streams);
  if (dl.dataset.sig !== sig) {
    dl.dataset.sig = sig;
    dl.replaceChildren(...Object.values(c.streams).map((st) => h("div", { class: "group" },
      h("dt", { class: "group__label", text: `${st.profile} · ${st.codec} ${st.width}×${st.height} @ ${st.fps} fps · ${st.bitrate_kbps} kbps` }),
      h("dd", { class: "uri" }, h("code", { text: st.uri }),
        h("button", { type: "button", class: "btn btn--sm btn--ghost", "data-copy": st.uri, "aria-label": `Copy ${st.profile} stream URI` }, "Copy")))));
    if (!dl.children.length) dl.append(h("p", { class: "empty", text: "Stream URIs not resolved yet." }));
  }

  const hasPtz = info?.capabilities?.includes("PTZ");
  $("#ptz-form").hidden = !hasPtz;
  $("#ptz-none").hidden = !info || hasPtz;
  const p = c.ptz;
  $("#ptz-pos").textContent = p
    ? `Position: pan ${signed(p.pan)} · tilt ${signed(p.tilt)} · zoom ${p.zoom.toFixed(2)}${p.moving ? " · moving" : ""}`
    : "Position: —";

  const chaos = store.chaos.get(id);
  $("#cam-offline").setAttribute("aria-pressed", String(chaos.offline));
  $("#cam-offline").textContent = chaos.offline ? "Bring online" : "Take offline";
  $("#cam-revoke").setAttribute("aria-pressed", String(chaos.revoke));
  $("#cam-revoke").textContent = chaos.revoke ? "Restore credentials" : "Revoke credentials";
  syncFeed(c);
}

// ------------------------------------------------------------- camera feed
// Recorded MAIN / SUB clips stand in for the cameras' video (the simulator has no pixels).
// The server alternates them by camera (CAM-001 MAIN, CAM-002 SUB, ...) and lists the clips it has in meta.feeds.
const feedVideo = $("#feed-video");
let feedPaused = matchMedia("(prefers-reduced-motion: reduce)").matches;   // the viewer's choice; starts paused if motion is reduced

function setText(el, text) { if (el.textContent !== text) el.textContent = text; }

function syncFeed(c) {
  const src = store.meta?.feeds?.[c.feed], empty = $("#feed-empty");
  let note = null;
  if (!src) note = "No feed clip on the server. Put camera_main.mp4 and camera_sub.mp4 in the camera_feed folder.";
  else if (tone(c.health) === "crit") note = `No signal · ${c.id} is ${label(c.health).toLowerCase()}`;
  else if (feedVideo.dataset.failed === src) note = "This browser could not play the feed clip.";

  setText($("#feed-badge"), `${c.id} · ${c.feed}`);
  setText($("#feed-caption"), `${c.feed} feed · looping sample clip`);
  setText($("#feed-toggle"), feedPaused ? "Play feed" : "Pause feed");
  empty.hidden = !note;
  if (note) setText(empty, note);
  feedVideo.hidden = !!note;
  if (note) return feedVideo.pause();

  feedVideo.setAttribute("aria-label", `${c.id} ${c.feed} camera feed, looping sample clip`);
  if (feedVideo.getAttribute("src") !== src) feedVideo.src = src;    // CAM-001 and CAM-003 share a clip: no reload
  if (!feedPaused && feedVideo.paused) feedVideo.play().catch(() => {});   // muted, so autoplay is allowed
}

function fieldError(input, errEl, msg) {
  input.setAttribute("aria-invalid", "true");
  errEl.textContent = msg;
  input.focus();
}

function bindCameraControls() {
  $("#feed-toggle").addEventListener("click", () => {
    feedPaused = !feedPaused;
    if (feedPaused) feedVideo.pause();
    syncFeed(store.cameras.get(store.selectedCam));
    announce(feedPaused ? "Feed paused" : "Feed playing");
  });
  feedVideo.addEventListener("error", () => {
    feedVideo.dataset.failed = feedVideo.getAttribute("src");
    syncFeed(store.cameras.get(store.selectedCam));
  });
  feedVideo.addEventListener("loadedmetadata", () => { if (feedVideo.paused) feedVideo.currentTime = 0.1; });   // paint a frame while paused

  for (const k of ["pan", "tilt", "zoom"]) {
    $(`#ptz-${k}`).addEventListener("input", (e) => { e.target.nextElementSibling.textContent = Number(e.target.value).toFixed(2); });
  }
  $("#ptz-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const f = new FormData(e.target);
    command(e.submitter, store.selectedCam, "ptz", { pan: +f.get("pan"), tilt: +f.get("tilt"), zoom: +f.get("zoom") }).catch(() => {});
  });

  $("#enc-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const fps = $("#enc-fps"), br = $("#enc-bitrate"), err = $("#enc-error");
    for (const i of [fps, br]) i.removeAttribute("aria-invalid");
    err.textContent = "";
    const fpsV = Number(fps.value), brV = Number(br.value);
    if (!Number.isInteger(fpsV) || fpsV < 1 || fpsV > 30) return fieldError(fps, err, "Frame rate must be a whole number from 1 to 30.");
    if (!Number.isInteger(brV) || brV < 64 || brV > 16384) return fieldError(br, err, "Bitrate must be a whole number from 64 to 16384 kbps.");
    try {
      await command(e.submitter, store.selectedCam, "encoder", { profile: $("#enc-profile").value, fps: fpsV, bitrate_kbps: brV });
    } catch (ex) {
      err.textContent = `The camera rejected the change: ${ex.message}`;
    }
  });

  $("#fault-form").addEventListener("submit", (e) => {
    e.preventDefault();
    command(e.submitter, store.selectedCam, "fault", { fault: $("#fault-kind").value }).catch(() => {});
  });
  for (const [btnId, key, action] of [["#cam-offline", "offline", "offline"], ["#cam-revoke", "revoke", "revoke"]]) {
    $(btnId).addEventListener("click", async (e) => {
      const id = store.selectedCam, st = store.chaos.get(id), next = !st[key];
      await command(e.currentTarget, id, action, { on: next }).then(() => { st[key] = next; renderCameraDetail(); }, () => {});
    });
  }
  $("#cam-heal").addEventListener("click", async (e) => {
    const id = store.selectedCam;
    await command(e.currentTarget, id, "heal").then(() => { store.chaos.set(id, { offline: false, revoke: false }); renderCameraDetail(); }, () => {});
  });

  // copy buttons are rebuilt with the stream list, so delegate
  $("#cam-streams").addEventListener("click", async (e) => {
    const btn = e.target.closest("[data-copy]");
    if (!btn) return;
    try { await navigator.clipboard.writeText(btn.dataset.copy); notify("Stream URI copied"); }
    catch { notify("Copy failed: clipboard not available", "error"); }
  });
}

// -------------------------------------------------------------------- node
const LAYERS = [
  ["node", "Node reachable", "agent answers health calls"],
  ["runtime", "Runtime running", "Kaaval runtime not crashed or restarting"],
  ["inference", "Inference producing frames", "fps above zero"],
  ["pipe", "Event pipe delivering", `heartbeat within 3 intervals`],
];

function meter(id, value, max, text, crit, markAt) {
  const el = $(id);
  $(".meter__head > span:last-child", el).textContent = text;
  $(".meter__fill", el).style.width = value == null ? "0" : `${Math.min(100, (value / max) * 100)}%`;
  el.classList.toggle("meter--crit", !!crit);
  if (markAt != null) $(".meter__mark", el).style.left = `${(markAt / max) * 100}%`;
}

function nextModelVersion(v) {
  const m = v?.match(/^(.*?)(\d{4})\.(\d{2})$/);
  if (!m) return v ?? "";
  let [y, mo] = [+m[2], +m[3] + 1];
  if (mo > 12) { y += 1; mo = 1; }
  return `${m[1]}${y}.${String(mo).padStart(2, "0")}`;
}

function renderNode() {
  const n = store.node;
  if (!n) return;
  $("#node-title").textContent = n.id;
  setPill($("#node-pill"), n.health);
  const info = n.info, model = n.model;
  $("#node-sub").textContent = info
    ? `${info.vendor} ${info.model} · ${info.firmware}${model ? ` · model ${model.model_version}` : ""}`
    : "Identity not read yet";
  const errEl = $("#node-error");
  const showErr = n.last_error && tone(n.health) === "crit";
  errEl.hidden = !showErr;
  if (showErr) errEl.replaceChildren("Last error ", h("strong", { text: n.last_error.code }), ` at ${fmtTime(n.last_error.at)}: ${n.last_error.message}`);

  // layered check: first failing layer is the cause; layers after it are not evaluated
  const layers = n.layers ?? {};
  const firstFail = LAYERS.findIndex(([k]) => !layers[k]);
  const known = ["CONNECTING", "STARTING"].includes(n.health) ? -1 : firstFail;
  $("#node-layers").replaceChildren(...LAYERS.map(([k, name, hint], i) => {
    const state = layers[k] ? "ok" : i === known ? "fail" : "skip";
    const word = { ok: "passing", fail: "failing", skip: "not checked" }[state];
    return h("li", { class: `layer layer--${state}` },
      h("span", { class: "layer__icon", "aria-hidden": "true", text: { ok: "✓", fail: "✕", skip: "–" }[state] }),
      h("span", {}, h("span", { class: "layer__name", text: name }), h("br"), h("span", { class: "layer__state", text: hint })),
      h("span", { class: "layer__state", text: word }));
  }));

  const hl = n.ai_health;
  meter("#m-cpu", hl?.cpu_pct, 100, hl ? `${fmtNum(hl.cpu_pct)} %` : "—", hl?.cpu_pct >= 90);
  meter("#m-mem", hl?.mem_pct, 100, hl ? `${fmtNum(hl.mem_pct)} %` : "—", hl?.mem_pct >= 90);
  meter("#m-temp", hl?.temp_c, 100, hl ? `${fmtNum(hl.temp_c)} °C${hl.throttled ? " · throttled" : ""}` : "—", hl?.throttled, 85);
  $("#n-fps").textContent = hl ? `${fmtNum(hl.inference_fps)} fps` : "—";
  $("#n-latency").textContent = hl ? `${fmtNum(hl.latency_ms)} ms` : "—";
  $("#n-hb").textContent = n.heartbeat_age_s == null ? "none yet" : `${fmtDuration(n.heartbeat_age_s)} ago (sim time)`;
  $("#n-events").textContent = String(n.events_received);
  $("#n-hbs").textContent = String(n.heartbeats_received);
  $("#n-dlq").textContent = String(n.dead_letters);

  const caps = n.capabilities;
  const chips = (list) => (list ?? []).flatMap((x) => [h("span", { class: "chip", text: x }), " "]);
  const fn = $("#n-functions"), cams = $("#n-cams");
  if (fn.dataset.sig !== JSON.stringify(caps)) {
    fn.dataset.sig = JSON.stringify(caps);
    fn.replaceChildren(...(caps ? chips(caps.functions) : ["—"]));
    cams.replaceChildren(...(caps ? chips(caps.assigned_cameras) : ["—"]));
  }

  const mv = $("#model-version");
  if (!mv.value && document.activeElement !== mv && model) mv.value = nextModelVersion(model.model_version);
  const wl = $("#watch-plates");
  if (document.activeElement !== wl && !wl.dataset.dirty) wl.value = (n.watchlist ?? []).join(", ");
  const heat = $("#heat");
  if (document.activeElement !== heat && !heat.dataset.dirty) { heat.value = n.ambient_c; $("#heat-out").textContent = `${n.ambient_c} °C`; }
}

function bindNodeControls() {
  const nodeId = () => store.node?.id ?? "EDGE-NODE-001";
  for (const btn of $$("[data-node]")) {
    btn.addEventListener("click", () => {
      const body = btn.dataset.on ? { on: true } : {};
      command(btn, nodeId(), btn.dataset.node, body).catch(() => {});
    });
  }
  const heat = $("#heat");
  heat.addEventListener("input", () => { heat.dataset.dirty = "1"; $("#heat-out").textContent = `${heat.value} °C`; });
  $("#heat-form").addEventListener("submit", (e) => {
    e.preventDefault();
    command(e.submitter, nodeId(), "ambient", { celsius: Number(heat.value) })
      .then(() => { delete heat.dataset.dirty; }, () => {});
  });

  $("#model-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const input = $("#model-version"), err = $("#model-error");
    input.removeAttribute("aria-invalid");
    err.textContent = "";
    const v = input.value.trim();
    if (!/^[a-z0-9][a-z0-9._-]{2,63}$/.test(v)) {
      return fieldError(input, err, "Use 3–64 lowercase letters, digits, dots, dashes or underscores, starting with a letter or digit.");
    }
    announce(`Deploying ${v}. The runtime will restart.`);
    try { await command(e.submitter, nodeId(), "model", { model_version: v }); input.value = ""; }
    catch (ex) { err.textContent = `Deployment failed: ${ex.message}`; }
  });

  const wl = $("#watch-plates");
  wl.addEventListener("input", () => { wl.dataset.dirty = "1"; });
  $("#watch-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const err = $("#watch-error");
    wl.removeAttribute("aria-invalid");
    err.textContent = "";
    const plates = wl.value.split(/[\s,]+/).map((p) => p.trim().toUpperCase()).filter(Boolean);
    const bad = plates.filter((p) => !/^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}$/.test(p));
    if (bad.length) return fieldError(wl, err, `Not a valid plate: ${bad.join(", ")} (format like GJ01AB1234).`);
    try { await command(e.submitter, nodeId(), "watchlist", { plates }); delete wl.dataset.dirty; }
    catch (ex) { err.textContent = ex.message; }
  });
}

// ------------------------------------------------------------------ events
function matches(ev) {
  const type = $("#f-type").value, cam = $("#f-cam").value, q = $("#f-q").value.trim().toUpperCase();
  return (!type || ev.event_type === type) && (!cam || ev.camera_external_id === cam) &&
    (!q || (ev.attributes?.plate ?? "").includes(q));
}
function updateFeedCount() {
  $("#feed-count").textContent = `${$("#event-rows").rows.length} shown · ${store.events.length} buffered${$("#f-pause").checked ? " · paused" : ""}`;
}
function renderEventTable() {
  const rows = store.events.filter(matches).slice(-100).reverse().map((ev) => eventRow(ev, { model: true }));
  $("#event-rows").replaceChildren(...rows);
  $("#event-empty").hidden = rows.length > 0;
  updateFeedCount();
}
function renderDlq() {
  const rows = store.dlq.slice().reverse().map((d) => h("tr", {},
    h("td", { class: "nowrap", text: fmtTime(d.at) }), h("td", { class: "nowrap", text: d.reason }),
    h("td", {}, h("code", { text: d.sample }))));
  $("#dlq-rows").replaceChildren(...rows);
  $("#dlq-empty").hidden = rows.length > 0;
}
function bindEventFilters() {
  $("#event-filters").addEventListener("submit", (e) => e.preventDefault());
  for (const id of ["#f-type", "#f-cam", "#f-q", "#f-pause"]) $(id).addEventListener("input", renderEventTable);
}

// ------------------------------------------------------------------- trace
const trace = {
  timer: 0,
  toggle(on) {
    clearInterval(this.timer);
    this.timer = 0;
    if (on) { this.load(); this.timer = setInterval(() => !document.hidden && this.load(), 2000); }
  },
  data: [],
  async load() {
    try { this.data = await api("/api/trace?limit=300"); this.render(); }
    catch { $("#trace-meta").textContent = "Trace unavailable"; }
  },
  render() {
    const dev = $("#t-dev").value, faults = $("#t-faults").checked, noPolls = $("#t-polls").checked;
    const rows = this.data
      .filter((t) => (!dev || t.device === dev) && (!faults || t.outcome !== "OK") && (!noPolls || t.op !== "events.poll"))
      .slice(-150).reverse()
      .map((t) => h("tr", { class: t.outcome !== "OK" ? "row--fault" : null },
        h("td", { class: "num mono", text: String(t.id) }),
        h("td", { class: "nowrap", text: fmtTime(t.at) }),
        h("td", { class: "nowrap", text: t.device }),
        h("td", {}, h("code", { text: t.op }), t.params ? ` ${JSON.stringify(t.params)}` : ""),
        h("td", { class: "num nowrap", text: `${fmtNum(t.latency_ms)} ms` }),
        h("td", { class: "nowrap", text: t.outcome })));
    $("#trace-rows").replaceChildren(...rows);
    $("#trace-empty").hidden = rows.length > 0;
    const nFaults = this.data.filter((t) => t.outcome !== "OK").length;
    $("#trace-meta").textContent = `${this.data.length} requests · ${nFaults} faults · updated ${fmtTime(new Date().toISOString())}`;
  },
};

// -------------------------------------------------------------------- boot
bindCameraControls();
bindNodeControls();
bindEventFilters();
for (const id of ["#t-dev", "#t-faults", "#t-polls"]) $(id).addEventListener("input", () => trace.render());
$("#trace-filters").addEventListener("submit", (e) => e.preventDefault());
window.addEventListener("hashchange", route);
route();     // show the deep-linked view (with skeletons) before the first data arrives
connect();
