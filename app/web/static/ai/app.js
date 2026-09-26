/**
 * KAAVAL AI console.
 *
 * Talks only to this app's own API (same origin): /ai/helmet/{config,health,process,jobs}
 * (local YOLO path), /ai/video/{health,jobs} (cloud route) and /ai/helmet/evidence for the
 * frames. Every value from the server is written with textContent, never innerHTML — a
 * garbled detector payload must not become markup.
 */
const API = "/ai/helmet";
const VIDEO_API = "/ai/video";
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
  const total = metrics?.total_frames;
  const processed = metrics?.frames_processed;
  const keys = [
    ["duration_seconds", "Video length", (v) => `${v}s`],
    ["video_fps", "Video FPS"],
    ["total_frames", "Total frames"],
    ["frames_processed", "Frames processed"],
    ["frames_skipped", "Frames skipped"],
    ["violation_detections", "Violation detections"],
    ["unique_violators", "Unique violators"],
    ["evidence_frames", "Evidence frames"],
    ["tracked_riders", "Tracked riders"],
    ["events_ingested", "M3 accepted"],
    ["processing_fps", "Processing FPS", (v) => String(v)],
    ["processing_time", "Seconds", (v) => `${v}s`],
    ["sampling_mode", "Sampling"],
    ["inference_size", "imgsz"],
    ["avg_latency_ms", "ms/frame"],
  ];
  box.replaceChildren();
  box.hidden = false;
  for (const [key, label, fmt] of keys) {
    let value = metrics?.[key];
    if (value === undefined || value === null) continue;
    if (key === "frames_processed" && total) value = `${value} / ${total}`;
    box.append(el("div", {}, [
      el("dt", { text: label }),
      el("dd", { text: fmt ? fmt(value) : String(value) }),
    ]));
  }
}

/** The number that matters: distinct riders, never frame-level detections. */
function renderSummary(metrics, done) {
  const hero = $("hero");
  const unique = metrics?.unique_violators;
  if (unique === undefined || unique === null) return;
  hero.hidden = false;
  $("hero-unique").textContent = String(unique);
  hero.dataset.state = done ? "done" : "run";
  const frames = metrics.total_frames
    ? `${metrics.frames_processed ?? 0} / ${metrics.total_frames} frames`
    : `${metrics.frames_processed ?? 0} frames`;
  $("hero-sub").textContent = [
    `${metrics.violation_detections ?? 0} frame-level violation detections`,
    frames,
    metrics.processing_fps != null ? `${metrics.processing_fps} fps` : null,
  ].filter(Boolean).join(" · ");
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
      v.track_id !== null && v.track_id !== undefined ? `unique rider · track ${v.track_id}` : "unique rider · no track id",
      `seen ${v.observations ?? "?"} frames without a helmet (${v.violation_detections ?? "?"} detections)`,
      `first ${v.first_seen ?? "?"}s → last ${v.last_seen ?? "?"}s`,
      `evidence frame ${v.frame_number} · t ${v.timestamp}s · confirmed after ${v.confirmations} observations`,
    ];
    return el("article", { class: "card" }, [
      img,
      el("div", { class: "card__body" }, [
        el("span", { class: "badge", text: "UNIQUE HELMET VIOLATOR" }),
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
    body.append(el("tr", {}, [el("td", { class: "muted", colspan: "8", text: "No jobs yet." })]));
    return;
  }
  for (const job of jobs) {
    const m = job.metrics || {};
    const backend = m.analysis_effective
      ? (m.analysis_fallback ? `${m.analysis_effective}*` : m.analysis_effective)
      : job.mode;
    const frames = m.frames_processed != null && m.total_frames != null
      ? `${m.frames_processed}/${m.total_frames}` : String(m.frames_processed ?? "—");
    body.append(el("tr", {}, [
      el("td", { class: "mono", text: job.job_id }),
      el("td", { text: job.camera_id }),
      el("td", { class: "mono", title: m.analysis_fallback_reason || "", text: backend }),
      el("td", { text: job.status }),
      el("td", { class: "mono", text: frames }),
      el("td", { class: "mono", text: String(m.violation_detections ?? "—") }),
      el("td", { class: "mono", text: String(m.unique_violators ?? "—") }),
      el("td", { class: "mono", text: m.processing_time != null ? `${m.processing_time}s` : "—" }),
    ]));
  }
}

// ---------------------------------------------------------------------- polling
/** Cloud analysis layer: requested vs effective backend, honest about any fallback. */
function renderCloud(vs) {
  const box = $("cloud");
  const pill = $("cloud-state");
  box.setAttribute("aria-busy", "false");
  box.replaceChildren();
  const analysis = vs.analysis || {};
  const vi = vs.video_intelligence || {};
  const anpr = vs.anpr || {};
  const classifier = vs.helmet_classifier || {};
  const effective = analysis.effective || "unknown";
  const fellBack = String(effective).includes("fallback");
  pill.className = "pill " + (fellBack ? "pill--run" : "pill--ok");
  pill.textContent = fellBack ? "fallback: local" : effective;
  const rows = [
    ["Requested backend", analysis.requested || "—", fellBack ? "err" : "ok"],
    ["Effective backend", effective, fellBack ? "err" : "ok"],
    // the reason a fallback happened is the Video Intelligence error itself
    ["Fallback reason", analysis.fallback_reason || (fellBack ? (vi.error || "see Video Intelligence below") : "—"), fellBack ? "err" : "warn"],
    ["Video Intelligence", `${vi.status || "—"}${vi.sdk_installed === false ? " (SDK not installed)" : ""}`, vi.status === "AVAILABLE" ? "ok" : "err"],
    ["Helmet classifier", `${classifier.status || "—"} (${classifier.classifier || "local YOLO"})`, classifier.status === "READY" ? "ok" : "err"],
    ["Plate OCR (ANPR)", anpr.status || "—", anpr.status === "AVAILABLE" ? "ok" : "warn"],
  ];
  for (const [label, value, kind] of rows) {
    const dd = el("dd", { class: kind === "ok" ? "state-ok" : kind === "err" ? "state-err" : "state-warn", text: String(value) });
    box.append(el("div", {}, [el("dt", { text: label }), dd]));
  }
  if (vi.deprecation_note) {
    box.append(el("div", {}, [el("dt", { text: "Deprecation" }),
                             el("dd", { class: "state-warn", text: String(vi.deprecation_note) })]));
  }
}

async function refreshCloud() {
  try {
    renderCloud(await api(`${VIDEO_API}/health`));
  } catch (e) {
    $("cloud").setAttribute("aria-busy", "false");
    $("cloud").replaceChildren(el("p", { class: "state-err", text: `Cloud route unavailable: ${e.message}` }));
  }
}

async function refreshOverview() {
  try {
    const [health, jobs, videoJobs] = await Promise.all([
      api(`${API}/health`),
      api(`${API}/jobs?limit=10`),
      api(`${VIDEO_API}/jobs?limit=10`).catch(() => ({ jobs: [] })),
    ]);
    renderHealth(health);
    const merged = [...(jobs.jobs || []), ...(videoJobs.jobs || [])]
      .sort((a, b) => (b.created_at || 0) - (a.created_at || 0)).slice(0, 12);
    renderJobs(merged);
  } catch (e) {
    $("health").setAttribute("aria-busy", "false");
    $("health").replaceChildren(el("p", { class: "state-err", text: `Health unavailable: ${e.message}` }));
  }
  refreshCloud();
}

async function trackJob(jobId, cameraId) {
  const wrap = $("progress-wrap");
  wrap.hidden = false;
  wrap.dataset.state = "run";
  clearInterval(state.poll);
  state.poll = setInterval(async () => {
    let job;
    try {
      // both modes live in the same job registry: one endpoint serves either job id
      job = await api(`${API}/jobs/${jobId}`);
    } catch (e) {
      setMsg(`Could not read job ${jobId}: ${e.message}`, "err");
      return;
    }
    const m = job.metrics || {};
    const progress = m.progress_percent;
    const done = m.frames_processed ?? m.frames_analysed_by_provider ?? 0;
    $("progress-label").textContent =
      `${done} / ${m.total_frames ?? "?"} frames · ` +
      `${m.unique_violators ?? 0} unique violator(s) so far`;
    $("progress-fill").style.width = progress != null ? `${progress}%` : "35%";
    renderMetrics(metrics);
    renderSummary(metrics, false);

    if (job.status === "RUNNING" || job.status === "QUEUED") {
      status("run", job.status.toLowerCase());
      return;
    }
    clearInterval(state.poll);
    state.poll = null;
    busy(false);
    wrap.dataset.state = job.status === "COMPLETED" ? "done" : "err";
    renderMetrics(m);
    renderSummary(m, true);
    renderViolations(job.violations || [], cameraId);
    if (job.status === "COMPLETED") {
      status("ok", "completed");
      const found = m.unique_violators ?? (job.violations || []).length;
      const backend = m.analysis_effective
        ? ` · backend ${m.analysis_effective}${m.analysis_fallback ? " (fallback)" : ""}` : "";
      const anpr = job.mode === "video" && m.anpr_attempts != null
        ? ` · ANPR ${m.anpr_matches ?? 0}/${m.anpr_attempts}` : "";
      setMsg(
        `${found} unique helmet violator${found === 1 ? "" : "s"} across ` +
        `${m.frames_processed ?? m.frames_analysed_by_provider ?? "?"} frames ` +
        `(${m.violation_detections ?? 0} frame-level detections) in ${m.processing_time ?? "?"}s` +
        `${backend}${anpr}.`,
        found ? "ok" : "");
      if (job.mode === "video") refreshCloud();
      toast(`${job.camera_id}: ${found} unique helmet violator${found === 1 ? "" : "s"} ingested`, "ok");
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
  const cloud = $("backend").value === "cloud";
  busy(true);
  status("run", "starting");
  setMsg(cloud
    ? "Job accepted; analyzing the whole video with the configured provider…"
    : "Job accepted; decoding video…", "");
  renderViolations([], cameraId);
  $("hero").hidden = true;
  $("progress-fill").style.width = "0%";
  try {
    const job = await api(cloud ? `${VIDEO_API}/process` : `${API}/process`, {
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
