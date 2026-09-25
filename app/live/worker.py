"""
Live detection worker: reads live camera streams, detects events in them, and sends those events
into Module 3 (C4 ingestion) in place of the mock Kaaval events.

    # 1. once: fetch the detector model files
    python -m app.live.detector --fetch
    # 2. start the M3 API
    uvicorn app.main:app --port 8000
    # 3. start the worker (grid credentials come from .env)
    python -m app.live.worker
    # try it without the grid, on a video file (loops like a live feed):
    python -m app.live.worker --source cam01=camera_feed/camera_main.mp4 --dry-run

Events reach M3 exactly like Kaaval events do (signed batches to /federation/ingest/<node>), so
deduplication, storage, provenance and delivery to Module 4 all apply unchanged.
"""
from __future__ import annotations

import argparse
import json
import logging
import queue
import sys
import threading
import time
import uuid

from app.core.config import Settings, get_settings
from app.live.detector import MODEL_VERSION, Detector, ModelMissing
from app.live.grid import GridCamera, GridClient, GridError
from app.live.m3 import M3Client, M3Error, canonical_camera_id
from app.live.reader import StreamReader
from app.live.redact import Redactor, install_log_redaction, redact_url
from app.live.tracker import Tracker

log = logging.getLogger("sentinel.m3.worker")

BATCH_MAX, BATCH_EVERY_S, HEARTBEAT_EVERY_S, STATS_EVERY_S, OUTBOX_MAX = 25, 1.0, 10.0, 30.0, 2000


class LatestFrame:
    """
    Hand-off between the thread that reads the stream and the thread that runs detection.

    Reading must never wait for detection: if the connection is not drained fast enough, the server
    discards data ("reader is too slow") and the picture corrupts. So the reader always keeps going and
    overwrites this slot; detection takes the newest frame and skips the ones it had no time for.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._frame = None
        self.skipped = 0

    def put(self, frame) -> None:
        with self._cond:
            if self._frame is not None:
                self.skipped += 1
                frame.discontinuity = frame.discontinuity or self._frame.discontinuity   # never lose a "loop happened" flag
            self._frame = frame
            self._cond.notify()

    def get(self, timeout: float):
        with self._cond:
            if self._frame is None:
                self._cond.wait(timeout)
            frame, self._frame = self._frame, None
            return frame


class CameraWorker(threading.Thread):
    """Reading thread + detection loop, for one camera."""

    def __init__(self, cam_id: str, source: str, settings: Settings, out: "queue.Queue[dict]",
                 stop: threading.Event, run_id: str):
        super().__init__(name=f"cam-{cam_id}", daemon=True)
        self.cam_id, self._out, self._stop_evt, self._run_id = cam_id, out, stop, run_id
        self._s = settings
        self.reader = StreamReader(cam_id, source, settings.live_sample_fps, stop)
        self.slot = LatestFrame()
        self.processed = 0
        self.events = 0
        self.latency_ms = 0.0
        self.started = time.monotonic()

    def _pump(self) -> None:
        try:
            for frame in self.reader.frames():
                self.slot.put(frame)
        except Exception:
            log.exception("%s: stream reader crashed", self.cam_id)

    def run(self) -> None:
        threading.Thread(target=self._pump, name=f"read-{self.cam_id}", daemon=True).start()
        try:
            detector, tracker = Detector(self._s.live_model_dir, self._s.live_min_confidence), Tracker()
            while not self._stop_evt.is_set():
                frame = self.slot.get(timeout=0.5)
                if frame is None:
                    continue
                if frame.discontinuity:
                    tracker.reset()                 # a loop or reconnect: objects are new, nothing to report
                t0 = time.perf_counter()
                detections = detector.detect(frame.image)
                took_ms = (time.perf_counter() - t0) * 1000
                self.latency_ms = took_ms if not self.processed else 0.9 * self.latency_ms + 0.1 * took_ms
                self.processed += 1
                h, w = frame.image.shape[:2]
                for e in tracker.update(detections):
                    self.events += 1
                    self._out.put({
                        "kind": "event", "provider": "SENTINEL_GRID", "adapter": "LIVE_DETECTOR",
                        "evt_id": f"live-{self._run_id}-{self.cam_id}-{e.track_id}-{e.type}",   # stable, so a retried batch dedups
                        "node": self._s.live_node_id, "cam": canonical_camera_id(self.cam_id), "type": e.type,
                        "ts_ms": int(time.time() * 1000), "pts_ms": int(frame.pts_s * 1000), "conf": e.conf,
                        "data": {"bbox": list(e.box), "track_id": e.track_id, "frame_width": w, "frame_height": h, **e.data},
                        "model_ver": MODEL_VERSION, "fps": round(self.inference_fps, 2), "latency_ms": round(self.latency_ms, 1)})
        except Exception:                                # a crashed camera must not take the others down
            log.exception("%s: worker crashed", self.cam_id)

    @property
    def inference_fps(self) -> float:
        return self.processed / max(1e-6, time.monotonic() - self.started)


def _select_cameras(catalogue: list[GridCamera], wanted: list[str], limit: int) -> list[GridCamera]:
    by_id = {c.id: c for c in catalogue}
    if wanted:
        chosen = []
        for w in wanted:
            if w in by_id:
                chosen.append(by_id[w])
            else:
                log.warning("camera '%s' is not in the grid catalogue (available: %s); skipping", w, ", ".join(sorted(by_id)[:12]))
    else:
        chosen = catalogue[:limit]
    if len(chosen) > limit:
        log.warning("%d cameras requested but LIVE_MAX_CAMERAS is %d (each open camera costs the grid one stream): using the first %d",
                    len(chosen), limit, limit)
        chosen = chosen[:limit]
    return chosen


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Detect events in live camera streams and feed them to Module 3")
    ap.add_argument("--cameras", help="comma separated grid camera ids (default: LIVE_CAMERAS from .env)")
    ap.add_argument("--source", action="append", default=[], metavar="ID=PATH_OR_URL",
                    help="use this video file or rtsp URL for a camera instead of the grid (repeatable); skips the grid login")
    ap.add_argument("--dry-run", action="store_true", help="print events instead of sending them to M3")
    ap.add_argument("--seconds", type=float, default=0, help="stop after this many seconds (default: run until Ctrl+C)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    s = get_settings()
    grid = GridClient(s)
    install_log_redaction(Redactor(*grid.secrets()))                 # before anything can log a credential
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)             # one line per request is noise

    # ---- decide what to watch, and where each stream comes from
    overrides = dict(x.split("=", 1) for x in args.source if "=" in x)
    if overrides:
        cameras = [GridCamera(cid.lower(), cid.lower()) for cid in overrides]
        sources = {c.id: overrides[c.id] for c in cameras}
        log.info("test mode: using %d local source(s); the grid is not contacted", len(cameras))
    else:
        if not grid.configured:
            print("SENTINEL_GRID_EMAIL and SENTINEL_GRID_PASSWORD are not set. Add them to .env, then run again.", file=sys.stderr)
            return 2
        try:                                                          # C7 connection test gates everything else
            catalogue = grid.connection_test()
        except GridError as e:
            print(f"grid connection test failed: {e}", file=sys.stderr)
            return 2
        wanted = [c.strip().lower() for c in (args.cameras or s.live_cameras).split(",") if c.strip()]
        cameras = _select_cameras(catalogue, wanted, s.live_max_cameras)
        if not cameras:
            print("no cameras selected: check LIVE_CAMERAS against the grid catalogue", file=sys.stderr)
            return 2
        sources = {c.id: grid.rtsp_url(c.id) for c in cameras}       # credentials exist only in this dict, in memory
    log.info("watching: %s", ", ".join(c.id for c in cameras))

    try:
        Detector(s.live_model_dir)        # fail fast, with instructions, if the model files are missing
    except ModelMissing as e:
        print(str(e), file=sys.stderr)
        return 2

    m3: M3Client | None = None
    if not args.dry_run:
        m3 = M3Client(s.m3_ingest_base_url, s.kaaval_hmac_secret, s.live_node_id)
        try:
            m3.register(cameras, s.sentinel_grid_host, s.sentinel_grid_rtsp_port, s.sentinel_grid_web_base)
        except M3Error as e:
            print(str(e), file=sys.stderr)
            return 2

    # ---- run
    stop, out = threading.Event(), queue.Queue(maxsize=5000)
    run_id = uuid.uuid4().hex[:6]
    workers = [CameraWorker(c.id, sources[c.id], s, out, stop, run_id) for c in cameras]
    for w in workers:
        w.start()
    log.info("running (Ctrl+C to stop)%s", " [dry run]" if args.dry_run else "")

    outbox: list[dict] = []
    last_send = last_hb = last_stats = time.monotonic()
    deadline = time.monotonic() + args.seconds if args.seconds else None
    try:
        while not stop.is_set() and not (deadline and time.monotonic() > deadline):
            try:
                outbox.append(out.get(timeout=0.25))
            except queue.Empty:
                pass
            now = time.monotonic()
            if now - last_hb >= HEARTBEAT_EVERY_S:
                last_hb = now
                outbox.append({"kind": "heartbeat", "node": s.live_node_id, "runtime": "RUNNING",
                               "fps": round(sum(w.inference_fps for w in workers), 2), "ts_ms": int(time.time() * 1000)})
            if outbox and (len(outbox) >= BATCH_MAX or now - last_send >= BATCH_EVERY_S):
                last_send = now
                if args.dry_run:
                    for item in outbox:
                        print(json.dumps(item, separators=(",", ":")), flush=True)
                    outbox.clear()
                else:
                    try:
                        r = m3.send(outbox[:BATCH_MAX * 4])
                        log.info("sent %d item(s): accepted=%s duplicates=%s rejected=%s", r["processed"], r["accepted"], r["duplicates"], r["rejected"])
                        del outbox[:BATCH_MAX * 4]
                    except M3Error as e:
                        log.warning("%s; keeping %d item(s) to retry", e, len(outbox))
                        del outbox[:-OUTBOX_MAX]
            if now - last_stats >= STATS_EVERY_S:
                last_stats = now
                for w in workers:
                    st = w.reader.stats
                    log.info("%s: %.1f detections/s, %.0f ms each, %d event(s), %d frame(s) skipped, %d reconnect(s), %d loop(s)%s",
                             w.cam_id, w.inference_fps, w.latency_ms, w.events, w.slot.skipped, st.reconnects, st.loops,
                             f", last problem: {st.last_error}" if st.last_error else "")
    except KeyboardInterrupt:
        log.info("stopping...")
    finally:
        stop.set()
        for w in workers:
            w.join(timeout=8)
        while not out.empty():
            outbox.append(out.get_nowait())
        if outbox and m3 and not args.dry_run:
            try:
                m3.send(outbox[-OUTBOX_MAX:])
            except M3Error as e:
                log.warning("final flush failed: %s", e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
