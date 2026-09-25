// Live camera wall (real video mode).
//
// Every byte of video and every address comes from OUR side: the inventory and stream URLs from Module 3 (through this
// console), the pictures from OUR relay. This page never contacts the camera grid, and nothing here can: the Content
// Security Policy only allows the console itself and the relay.
//
// Player fallback chain, per camera:  WebRTC (WHEP)  ->  HLS  ->  retry with backoff (2 s doubling to 30 s).
// Only tiles that are on screen stream; a tile that scrolls away, a page change or leaving the Overview stops it.

const RETRY_MIN_MS = 2000, RETRY_MAX_MS = 30000;
const WHEP_TIMEOUT_MS = 15000, KEYFRAME_WAIT_MS = 40000, STALL_MS = 12000;   // some grid cameras send a keyframe only every 20 s or more
const $ = (sel, root = document) => root.querySelector(sel);

function h(tag, props = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (v == null || v === false) continue;
    if (k === "text") el.textContent = v;
    else if (v === true) el.setAttribute(k, "");
    else el.setAttribute(k, v);
  }
  el.append(...children.flat().filter((c) => c != null));
  return el;
}

const sleep = (ms, signal) => new Promise((res) => {
  const t = setTimeout(res, ms);
  signal?.addEventListener("abort", () => { clearTimeout(t); res(); }, { once: true });
});

class Cancelled extends Error {}

// ------------------------------------------------------------------- player
// States: idle | connecting | playing | offline.  `protocol` is what is playing now ("WebRTC" | "HLS").
export class LivePlayer {
  constructor(video, onChange = () => {}) {
    this.video = video;
    this.onChange = onChange;
    this.state = "idle";
    this.protocol = null;
    this.info = null;
    this.note = "";
    this.camId = null;
    this.abort = null;
    this.pc = null;
    this.whepSession = null;
  }

  start(camId) {
    if (this.camId === camId && this.abort && !this.abort.signal.aborted) return;   // already running for this camera
    this.stop();
    this.camId = camId;
    this.abort = new AbortController();
    this.#run(this.abort.signal);
  }

  stop() {
    this.abort?.abort();
    this.abort = null;
    this.#teardown();
    this.camId = null;
    this.#set("idle", null, "");
  }

  #set(state, protocol = this.protocol, note = "") {
    this.state = state; this.protocol = protocol; this.note = note;
    this.onChange(this);
  }

  #teardown() {
    if (this.pc) { this.pc.ontrack = this.pc.onconnectionstatechange = null; this.pc.close(); this.pc = null; }
    if (this.whepSession) { fetch(this.whepSession, { method: "DELETE" }).catch(() => {}); this.whepSession = null; }
    const v = this.video;
    v.pause();
    v.srcObject = null;
    if (v.getAttribute("src")) { v.removeAttribute("src"); v.load(); }
  }

  async #run(signal) {
    let attempt = 0;
    while (!signal.aborted) {
      this.#set("connecting", null, "Asking Module 3 for the stream…");
      let lost;
      try {
        this.info = await this.#sources(signal);
        lost = await this.#establish(this.info, signal);      // resolves once frames flow; returns a promise that settles on loss
        attempt = 0;
        await lost;
      } catch (e) {
        if (e instanceof Cancelled || signal.aborted) return;
        this.note = e.message;
      }
      this.#teardown();
      if (signal.aborted) return;
      const wait = Math.min(RETRY_MAX_MS, RETRY_MIN_MS * 2 ** attempt++) * (0.8 + Math.random() * 0.4);   // backoff + jitter, never a tight loop
      this.#set("offline", null, `${this.note || "Stream lost"} · retrying in ${Math.round(wait / 1000)} s`);
      await sleep(wait, signal);
    }
  }

  async #sources(signal) {
    const res = await fetch(`/api/wall/cameras/${encodeURIComponent(this.camId)}/stream-sources?profile=web`, { signal });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.message ?? `stream broker answered ${res.status}`);
    return data;
  }

  // Try WebRTC, then HLS. Returns a promise that rejects when the picture is later lost.
  async #establish(info, signal) {
    const errors = [];
    if (typeof RTCPeerConnection === "function") {
      try { return await this.#whep(info.urls.whep, signal); }
      catch (e) { if (e instanceof Cancelled) throw e; errors.push(`WebRTC: ${e.message}`); this.#teardown(); }
    } else {
      errors.push("WebRTC: not available in this browser");
    }
    if (!this.video.canPlayType("application/vnd.apple.mpegurl")) {
      errors.push("HLS: this browser cannot play it natively");
      throw new Error(errors.join(" · "));
    }
    this.#set("connecting", null, `${errors.join(" · ")} → trying HLS`);
    try { return await this.#hls(info.urls.hls, signal); }
    catch (e) { if (e instanceof Cancelled) throw e; errors.push(`HLS: ${e.message}`); throw new Error(errors.join(" · ")); }
  }

  async #whep(url, signal) {
    const pc = this.pc = new RTCPeerConnection({ iceServers: [] });     // relay is reachable directly: host candidates suffice
    pc.addTransceiver("video", { direction: "recvonly" });
    let gotTrack;
    const track = new Promise((res) => { gotTrack = res; });
    pc.ontrack = (e) => { this.video.srcObject = e.streams[0] ?? new MediaStream([e.track]); gotTrack(); };

    await pc.setLocalDescription(await pc.createOffer());
    await new Promise((res) => {                                          // wait (briefly) for ICE candidates to be in the offer
      if (pc.iceGatheringState === "complete") return res();
      const done = () => { if (pc.iceGatheringState === "complete") { pc.removeEventListener("icegatheringstatechange", done); res(); } };
      pc.addEventListener("icegatheringstatechange", done);
      setTimeout(res, 1500);
    });
    if (signal.aborted) throw new Cancelled();

    const ctl = AbortSignal.any([signal, AbortSignal.timeout(WHEP_TIMEOUT_MS)]);
    let res;
    try {
      res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/sdp" }, body: pc.localDescription.sdp, signal: ctl });
    } catch (e) { throw signal.aborted ? new Cancelled() : new Error(e.name === "TimeoutError" ? "timed out" : "relay unreachable"); }
    if (res.status !== 201 && !res.ok) throw new Error(`relay answered ${res.status}`);
    const loc = res.headers.get("Location");
    if (loc) this.whepSession = new URL(loc, url).href;
    await pc.setRemoteDescription({ type: "answer", sdp: await res.text() });
    this.#set("connecting", null, "Connected to the relay · waiting for the camera's next keyframe (some send one only every 20 s or more)…");

    await Promise.race([track, sleep(WHEP_TIMEOUT_MS, signal).then(() => { throw signal.aborted ? new Cancelled() : new Error("no video arrived"); })]);
    await this.#firstFrame(KEYFRAME_WAIT_MS, signal);
    this.#set("playing", "WebRTC", "");
    return this.#watch(signal, (fail) => { pc.onconnectionstatechange = () => { if (["failed", "disconnected", "closed"].includes(pc.connectionState)) fail("WebRTC connection lost"); }; });
  }

  async #hls(url, signal) {
    const v = this.video;
    v.src = url;
    v.play().catch(() => {});
    await this.#firstFrame(KEYFRAME_WAIT_MS, signal, () => v.error && `player error ${v.error.code}`);
    this.#set("playing", "HLS", "");
    return this.#watch(signal, (fail) => { v.onerror = () => fail("HLS playback error"); v.onended = () => fail("HLS ended"); });
  }

  #firstFrame(timeoutMs, signal, failed = () => null) {
    const v = this.video;
    return new Promise((res, rej) => {
      const t0 = performance.now();
      const tick = () => {
        if (signal.aborted) return rej(new Cancelled());
        if (v.readyState >= 2 && v.videoWidth > 0) return res();
        const why = failed();
        if (why) return rej(new Error(why));
        if (performance.now() - t0 > timeoutMs) return rej(new Error("no picture in time"));
        setTimeout(tick, 150);
      };
      v.play().catch(() => {});
      tick();
    });
  }

  // A promise that rejects when the picture stops: the transport says so, or time stops advancing for STALL_MS.
  #watch(signal, hook) {
    return new Promise((_, rej) => {
      let last = this.video.currentTime, since = performance.now(), done = false;
      const fail = (msg) => { if (!done) { done = true; clearInterval(iv); rej(new Error(msg)); } };
      hook(fail);
      const iv = setInterval(() => {
        if (signal.aborted) { done = true; clearInterval(iv); return; }
        if (this.video.paused && !document.hidden) this.video.play().catch(() => {});
        if (this.video.currentTime !== last) { last = this.video.currentTime; since = performance.now(); }
        else if (performance.now() - since > STALL_MS) fail("picture stalled");
      }, 1500);
    });
  }
}

// --------------------------------------------------------------------- wall
const glyph = { live: "●", connecting: "▲", offline: "■", idle: "○" };
const codecLabel = (cam, info) => {
  const src = (info?.source_codec ?? cam.codec ?? "").toUpperCase().replace("H", "H.");
  if (!src) return "codec unknown";
  return info?.transcoded || (src && src !== "H.264") ? `${src} → H.264` : src;
};

export function createWall({ announce = () => {} } = {}) {
  const grid = $("#wall-grid"), sizeSel = $("#wall-size"), prev = $("#wall-prev"), next = $("#wall-next");
  const pageLabel = $("#wall-page"), meta = $("#wall-meta"), empty = $("#wall-empty");
  const dialog = $("#wall-dialog"), dVideo = $("#wall-dialog-video");
  let cameras = [], page = 0, active = false, pollTimer = null, loaded = false;
  const tiles = new Map();            // cam id -> { cam, video, chips..., player, visible }
  const dialogPlayer = new LivePlayer(dVideo, (p) => paintDialog(p));
  let dialogCam = null, opener = null;

  const observer = new IntersectionObserver((entries) => {
    for (const e of entries) {
      const t = tiles.get(e.target.dataset.cam);
      if (!t) continue;
      t.visible = e.isIntersecting;
      sync(t);
    }
  }, { threshold: 0.25 });

  const perPage = () => Number(sizeSel.value) || 9;
  const pages = () => Math.max(1, Math.ceil(cameras.length / perPage()));

  function sync(t) {                  // stream exactly when: the wall is showing, the tile is on screen, the tab is visible
    if (active && t.visible && !document.hidden) t.player.start(t.cam.id);
    else t.player.stop();
  }

  function paint(t) {
    const p = t.player, cam = t.cam;
    let kind = p.state === "playing" ? "live" : p.state === "offline" || (p.state === "idle" && cam.status === "OFFLINE") ? "offline" : p.state === "connecting" ? "connecting" : "idle";
    if (cam.status === "OFFLINE" && p.state !== "playing") kind = "offline";
    const label = { live: "LIVE", connecting: "CONNECTING", offline: "OFFLINE", idle: "PAUSED" }[kind];
    t.status.className = `pill pill--${kind === "live" ? "ok" : kind === "offline" ? "crit" : kind === "connecting" ? "warn" : "muted"}`;
    t.status.replaceChildren(h("span", { class: "pill__glyph", "aria-hidden": "true", text: glyph[kind] + " " }), label);
    const info = p.info;
    t.codec.textContent = codecLabel(cam, info);
    t.res.textContent = (info?.resolution ?? cam.resolution ?? "—").replace("x", "×");
    const grid_ = info?.active_source ?? cam.active_source;
    t.proto.textContent = p.state === "playing" ? `${p.protocol}${grid_ ? ` · from ${grid_}` : ""}` : "—";
    t.empty.hidden = p.state === "playing";
    t.empty.textContent = kind === "offline" ? (cam.status === "OFFLINE" && !p.note ? "No signal: the grid did not answer for this camera" : p.note || "No signal")
      : kind === "connecting" ? (p.note || "Connecting…") : "Stream paused (off screen)";
    t.video.hidden = p.state !== "playing";
  }

  function buildTile(cam) {
    const video = h("video", { class: "feed__video", playsinline: true, autoplay: true, disablepictureinpicture: true, hidden: true, "aria-label": `${cam.id} live` });
    video.muted = true;
    const empty = h("p", { class: "feed__empty", text: "Stream paused (off screen)" });
    const status = h("span", { class: "pill pill--muted" }), codec = h("span", { class: "wall__chip" }), res = h("span", { class: "wall__chip" }), proto = h("span", { class: "wall__chip" });
    const open = h("button", { type: "button", class: "wall__open", "aria-label": `Open ${cam.id}, ${cam.name}, full size` });
    const frame = h("div", { class: "feed__frame", "data-cam": cam.id }, video,
      h("p", { class: "feed__badge", text: cam.id }), empty, open);
    const li = h("li", { class: "wall__tile" }, h("figure", { class: "feed" }, frame,
      h("figcaption", { class: "wall__cap" }, h("span", { class: "wall__name", text: cam.name }), status),
      h("div", { class: "wall__chips" }, codec, res, proto)));
    const t = { cam, li, frame, video, empty, status, codec, res, proto, visible: false, player: null };
    t.player = new LivePlayer(video, () => paint(t));
    open.addEventListener("click", () => openDialog(cam.id, open));
    observer.observe(frame);
    tiles.set(cam.id, t);
    paint(t);
    return li;
  }

  function render() {
    for (const t of tiles.values()) { observer.unobserve(t.frame); t.player.stop(); }     // page change stops every stream of the old page
    tiles.clear();
    page = Math.min(page, pages() - 1);
    const slice = cameras.slice(page * perPage(), (page + 1) * perPage());
    grid.dataset.size = String(perPage());
    grid.replaceChildren(...slice.map(buildTile));
    empty.hidden = cameras.length > 0;
    pageLabel.textContent = cameras.length ? `Page ${page + 1} of ${pages()} · cameras ${page * perPage() + 1}–${page * perPage() + slice.length} of ${cameras.length}` : "";
    prev.disabled = page === 0;
    next.disabled = page >= pages() - 1;
    const up = cameras.filter((c) => c.status !== "OFFLINE").length;
    meta.textContent = cameras.length ? `${cameras.length} cameras from Module 3 · ${up} reachable · video served by our relay` : "";
  }

  async function load() {
    try {
      const res = await fetch("/api/wall/cameras");
      const data = await res.json();
      if (!res.ok) throw new Error(data.message ?? `HTTP ${res.status}`);
      const first = !loaded;
      loaded = true;
      const changed = JSON.stringify(data.map((c) => [c.id, c.name])) !== JSON.stringify(cameras.map((c) => [c.id, c.name]));
      cameras = data;
      empty.textContent = "No cameras in Module 3 yet. Sync the grid catalogue: POST /api/v1/grid/sync.";
      if (first || changed) render();
      else for (const t of tiles.values()) { t.cam = cameras.find((c) => c.id === t.cam.id) ?? t.cam; paint(t); }   // statuses only: never disturb playing tiles
    } catch (e) {
      cameras = loaded ? cameras : [];
      empty.hidden = false;
      empty.textContent = `Cannot load the camera inventory: ${e.message}`;
    }
  }

  // ------------------------------------------------------------ full size
  function paintDialog(p) {
    $("#wall-dialog-status").textContent = p.state === "playing" ? `LIVE · ${p.protocol}${p.info?.active_source ? ` · from ${p.info.active_source}` : ""}`
      : p.state === "offline" ? p.note : p.state === "connecting" ? (p.note || "Connecting…") : "";
    dVideo.hidden = p.state !== "playing";
  }

  function openDialog(camId, from) {
    const cam = cameras.find((c) => c.id === camId);
    if (!cam) return;
    dialogCam = cam; opener = from ?? document.activeElement;
    $("#wall-dialog-title").textContent = `${cam.id} · ${cam.name}`;
    $("#wall-dialog-meta").textContent = `${cam.codec ? cam.codec.toUpperCase().replace("H", "H.") : "codec unknown"} · ${(cam.resolution ?? "—").replace("x", "×")}`;
    if (!dialog.open) dialog.showModal();
    dialogPlayer.start(cam.id);
  }

  dialog.addEventListener("close", () => { dialogPlayer.stop(); dialogCam = null; opener?.focus?.(); });
  $("#wall-close").addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", (e) => { if (e.target === dialog) dialog.close(); });   // click on the backdrop

  // ---------------------------------------------------------------- wiring
  prev.addEventListener("click", () => { page = Math.max(0, page - 1); render(); });
  next.addEventListener("click", () => { page = Math.min(pages() - 1, page + 1); render(); });
  sizeSel.addEventListener("change", () => { page = 0; render(); announce(`${perPage()} cameras per page`); });
  document.addEventListener("visibilitychange", () => { for (const t of tiles.values()) sync(t); });

  return {
    get cameras() { return cameras; },
    open: openDialog,
    load,
    // The wall streams only while its page is showing.
    setActive(on) {
      active = on;
      clearInterval(pollTimer);
      if (on) { load(); pollTimer = setInterval(load, 30000); }
      for (const t of tiles.values()) sync(t);
    },
    /** Live tiles currently streaming (for diagnostics and tests). */
    get streaming() { return [...tiles.values()].filter((t) => t.player.state === "playing").map((t) => ({ id: t.cam.id, protocol: t.player.protocol })); },
  };
}
