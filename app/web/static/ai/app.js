/**
 * KAAVAL AI console.
 *
 * Talks only to this app's own API (same origin): /ai/helmet/{config,health,process,jobs}
 * and /ai/helmet/evidence for the frames. Every value from the server is written with
 * textContent, never innerHTML — a garbled detector payload must not become markup.
 */
const API = "/ai/helmet";
const POLL_MS = 1200;

const $ = (id) => document.getElementById(id);
const state = { poll: null, camera: null, videos: {}, total: {}, model: null };

// --------------------------------------------------------------------- helpers
function el(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k === "dataset") Object.assign(node.dataset, v);
    else node.setAttribute(k, v);
  }
  for (const child of [].concat(children)) if (child) node.append(child);
  return node;
}

async function api(path, options) {
  const res = await fetch(path, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail ? String(body.detail) : `${res.status} ${res.statusText}`);
  return body;
}

function toast(text, kind) {
  const box = $("toasts");
  const node = el("div", { class: "toast", text });
  if (kind === "err") node.style.borderLeftColor = "var(--alert)";
  box.append(node);
  setTimeout(() => node.remove(), 6000);
}

function status(kind, text) {
  const pill = $("job-state");
  pill.className = "pill " + (kind === "ok" ? "pill--ok" : kind === "err" ? "pill--err" : kind === "run" ? "pill--run" : "pill--idle");
  pill.textContent = text;
}

function setMsg(text, kind) {
  const msg = $("run-msg");
  msg.textContent = text || "";
  msg.dataset.kind = kind || "";
}

function busy(on) {
  const btn = $("run");
  btn.setAttribute("aria-busy", String(on));
  btn.setAttribute("aria-disabled", String(on));
}

// ------------------------------------------------------------------- rendering
function renderHealth(h) {
  const box = $("health");
  box.setAttribute("aria-busy", "false");
  box.replaceChildren();
  const yolo = h.yolo_model || {};
  const rows = [
    ["YOLO model", yolo.status || "UNKNOWN", yolo.status === "READY" ? "ok" : "err"],
    ["KAAVAL AI logic", h.kaaval_ai || "UNKNOWN", h.kaaval_ai === "ONLINE" ? "ok" : "err"],
    ["Event pipeline", h.event_pipeline || "UNKNOWN", h.event_pipeline === "ONLINE" ? "ok" : "err"],
    ["Inference size", (h.yolo_model?.inference_size_4k ? `${h.yolo_model.inference_size_4k} px @ 4K` : "—"), "warn"],
    ["Device", yolo.device || "—", "warn"],
    ["Tracking", yolo.tracking_fallback ? `fallback: ${yolo.tracking_fallback}` : (yolo.tracking_enabled ? "ByteTrack" : "off"), "warn"],
  ];
  for (const [label, value, kind] of rows) {
    const dd = el("dd", { class: kind === "ok" ? "state-ok" : kind === "err" ? "state-err" : "state-warn", text: String(value) });
    box.append(el("div", {}, [el("dt", { text: label }), dd]));
  }
  if (Array.isArray(h.cameras) && h.cameras.length) {
    const cams = h.cameras.map((c) => `${c.camera_id}:${c.camera_status}`).join("  ");
    box.append(el("div", {}, [el("dt", { text: "Cameras" }), el("dd", { class: "mono", text: cams })]));
  }
}

function renderMetrics(metrics) {
  const box = $("metrics");
  const keys = [
    ["frames_decoded", "Frames"], ["inference_frames", "Observations"],
    ["detections", "Detections"], ["violations", "Violations"],
    ["events_ingested", "Accepted"], ["inference_size", "imgsz"],
    ["processing_time", "Seconds"], ["avg_latency_ms", "ms/frame"],
  ];
  box.replaceChildren();
  box.hidden = false;
  for (const [key, label] of keys) {
    const value = metrics?.[key];
    if (value === undefined || value === null) continue;
    box.append(el("div", {}, [
      el("dt", { text: label }),
      el("dd", { text: key === "processing_time" ? `${value}s` : String(value) }),
    ]));
  }
}

function renderViolations(violations, cameraId) {
  const box = $("violations");
  $("violation-count").textContent = String(violations.length);
  if (!violations.length) {
    box.replaceChildren(el("p", { class: "muted", text: "No confirmed violations in this job." }));
    return;
  }
  box.replaceChildren(...violations.map((v) => {
    const uri = v.evidence_ref && v.evidence_ref.uri;
    const img = uri
      ? el("img", { src: `${API}/evidence?uri=${encodeURIComponent(uri)}`, alt: `Evidence frame for ${v.event_id}`, loading: "lazy" })
      : null;
    const meta = [
      `${cameraId || v.camera_id} · conf ${v.confidence}`,
      `frame ${v.frame_number} · t ${v.timestamp}s · ${v.confirmations} observations`,
      v.track_id !== null && v.track_id !== undefined ? `track ${v.track_id}` : "no track id",
    ];
    return el("article", { class: "card" }, [
      img,
      el("div", { class: "card__body" }, [
        el("span", { class: "badge", text: "HELMET_VIOLATION" }),
        el("p", { class: "card__id", text: v.event_id }),
        ...meta.map((line) => el("p", { class: "card__meta", text: line })),
      ]),
    ]);
  }));
}

function renderJobs(jobs) {
  const body = $("jobs");
  body.replaceChildren();
  if (!jobs.length) {
    body.append(el("tr", {}, [el("td", { class: "muted", colspan: "7", text: "No jobs yet." })]));
    return;
  }
  for (const job of jobs) {
    const m = job.metrics || {};
    body.append(el("tr", {}, [
      el("td", { class: "mono", text: job.job_id }),
      el("td", { text: job.camera_id }),
      el("td", { text: job.status }),
      el("td", { class: "mono", text: String(m.frames_decoded ?? "—") }),
      el("td", { class: "mono", text: String(m.detections ?? "—") }),
      el("td", { class: "mono", text: String(m.violations ?? "—") }),
      el("td", { class: "mono", text: m.processing_time != null ? `${m.processing_time}s` : "—" }),
    ]));
  }
}

// ---------------------------------------------------------------------- polling
async function refreshOverview() {
  try {
    const [health, jobs] = await Promise.all([api(`${API}/health`), api(`${API}/jobs?limit=10`)]);
    renderHealth(health);
    renderJobs(jobs.jobs || []);
  } catch (e) {
    $("health").setAttribute("aria-busy", "false");
    $("health").replaceChildren(el("p", { class: "state-err", text: `Health unavailable: ${e.message}` }));
  }
}

async function trackJob(jobId, cameraId) {
  const wrap = $("progress-wrap");
  wrap.hidden = false;
  wrap.dataset.state = "run";
  clearInterval(state.poll);
  state.poll = setInterval(async () => {
    let job;
    try {
      job = await api(`${API}/jobs/${jobId}`);
    } catch (e) {
      setMsg(`Could not read job ${jobId}: ${e.message}`, "err");
      return;
    }
    const metrics = job.metrics || {};
    $("progress-label").textContent =
      `${metrics.frames_decoded ?? 0} frames · ${metrics.detections ?? 0} detections · ${metrics.violations ?? 0} violations`;
    renderMetrics(metrics);

    if (job.status === "RUNNING" || job.status === "QUEUED") {
      status("run", job.status.toLowerCase());
      return;
    }
    clearInterval(state.poll);
    state.poll = null;
    busy(false);
    wrap.dataset.state = job.status === "COMPLETED" ? "done" : "err";
    renderMetrics(metrics);
    renderViolations(job.violations || [], cameraId);
    if (job.status === "COMPLETED") {
      status("ok", "completed");
      const found = (job.violations || []).length;
      setMsg(`${found} violation${found === 1 ? "" : "s"} confirmed and ingested in ${metrics.processing_time ?? "?"}s.`, "ok");
      toast(`${job.camera_id}: ${found} violation${found === 1 ? "" : "s"} ingested`, "ok");
    } else {
      status("err", "failed");
      setMsg(job.error || "Job failed.", "err");
      toast(`${job.camera_id}: job failed`, "err");
    }
    refreshOverview();
  }, POLL_MS);
}

// ------------------------------------------------------------------------ setup
async function start() {
  try {
    const cfg = await api(`${API}/config`);
    state.model = cfg;
    state.videos = cfg.video_sources || {};
    const select = $("camera");
    const ids = Object.keys(state.videos).sort();
    select.replaceChildren(...ids.map((id) => el("option", { value: id, text: id })));
    state.camera = ids[0] || null;
    showVideo();
    setMsg(`Model ${cfg.model_name} v${cfg.model_version} · inference size up to ${cfg.image_size} px (auto: ${cfg.image_size_auto}).`, "");
  } catch (e) {
    setMsg(`Could not load AI configuration: ${e.message}`, "err");
  }
  refreshOverview();
}

function showVideo() {
  const id = $("camera").value;
  const path = state.videos[id];
  $("camera-video").textContent = path ? `source: ${path}` : "no video mapped for this camera";
}

$("camera").addEventListener("change", showVideo);

$("run-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const cameraId = $("camera").value;
  if (!cameraId) return;
  busy(true);
  status("run", "starting");
  setMsg("Job accepted; decoding video…", "");
  renderViolations([], cameraId);
  try {
    const job = await api(`${API}/process`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ camera_id: cameraId }),
    });
    toast(`Job ${job.job_id} started on ${job.camera_id}`, "ok");
    trackJob(job.job_id, job.camera_id);
  } catch (e) {
    busy(false);
    status("err", "rejected");
    setMsg(e.message, "err");
  }
});

start();
setInterval(refreshOverview, 15000);
