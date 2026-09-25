"""
Live grid integration: credential safety, catalogue/login, stream reading, tracking and the path
into the existing C4 ingestion pipeline. Everything here is offline (no grid, no network).
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.adapters.live_detector import LiveDetectorAdapter
from app.core.config import Settings
from app.live import reader as reader_mod
from app.live.detector import Detection
from app.live.grid import GridCamera, GridClient, GridError, parse_catalogue
from app.live.m3 import M3Client, M3Error, canonical_camera_id
from app.live.reader import StreamReader
from app.live.redact import Redactor, install_log_redaction, redact_url
from app.live.tracker import Tracker
from app.main import app

EMAIL, PASSWORD = "tester@example.com", "ABCD-1234-WXYZ"
SECRETS = (EMAIL, PASSWORD, "tester%40example.com")


def make_settings(**kw) -> Settings:
    return Settings(_env_file=None, sentinel_grid_email=EMAIL, sentinel_grid_password=PASSWORD, **kw)


# ------------------------------------------------------------------ credential safety
def test_rtsp_url_encodes_the_at_sign_and_public_url_has_no_credentials():
    grid = GridClient(make_settings(sentinel_grid_host="10.0.0.9", sentinel_grid_rtsp_port=8554))
    assert grid.rtsp_url("cam01") == "rtsp://tester%40example.com:ABCD-1234-WXYZ@10.0.0.9:8554/stream/cam01"
    public = grid.rtsp_url_public("cam01")
    assert public == "rtsp://10.0.0.9:8554/stream/cam01"
    assert not any(s in public for s in SECRETS)


def test_password_is_masked_in_settings_repr():
    assert PASSWORD not in repr(make_settings()) and PASSWORD not in str(make_settings().model_dump())


def test_redactor_removes_every_encoding():
    r = Redactor(EMAIL, PASSWORD)
    for text in (f"rtsp://tester%40example.com:{PASSWORD}@h/stream/x", f"user {EMAIL} pw {PASSWORD}", "tester%40example.com",
                 f"http://{EMAIL}:{PASSWORD}@h:8889/x"):
        out = r.scrub(text)
        assert not any(s in out for s in SECRETS), out
    assert redact_url("rtsp://a:b@host/x") == "rtsp://***@host/x"
    assert Redactor("ab").scrub("about") == "about"          # too short to redact safely: left alone


def test_log_records_and_tracebacks_never_contain_credentials(caplog):
    install_log_redaction(Redactor(EMAIL, PASSWORD))
    log = logging.getLogger("sentinel.m3.test")
    with caplog.at_level(logging.DEBUG, logger="sentinel.m3.test"):
        log.info("opening %s", f"rtsp://tester%40example.com:{PASSWORD}@grid/stream/cam01")
        try:
            raise RuntimeError(f"boom {PASSWORD} for {EMAIL}")
        except RuntimeError:
            log.exception("stream failed for %s", EMAIL)
    everything = caplog.text + "\n".join(r.getMessage() + (r.exc_text or "") for r in caplog.records)
    assert not any(s in everything for s in SECRETS), everything


# ----------------------------------------------------------------------- catalogue + login
def test_parse_catalogue_accepts_the_shapes_a_catalogue_might_have():
    a = parse_catalogue([{"id": "cam01", "name": "One", "codec": "h264", "width": 1920, "height": 1080, "fps": 25}, {"id": "cam02"}])
    assert [c.id for c in a] == ["cam01", "cam02"] and a[0].width == 1920 and a[1].name == "cam02"
    assert [c.id for c in parse_catalogue({"cameras": ["cam03", {"camera_id": "CAM04", "resolution": "1280x720"}]})] == ["cam03", "cam04"]
    assert [c.id for c in parse_catalogue({"cam05": {"name": "x"}, "cam06": {}})] == ["cam05", "cam06"]
    assert [c.id for c in parse_catalogue([{"id": "ok"}, {"id": "bad id!"}, {"id": ""}])] == ["ok"]      # unusable ids are skipped
    with pytest.raises(GridError):
        parse_catalogue({"cameras": []})
    with pytest.raises(GridError):
        parse_catalogue(42)


def fake_grid_site(password: str = PASSWORD, expire_once: bool = False):
    state = {"logins": 0, "expire": expire_once}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/login" and request.method == "POST":
            form = dict(x.split("=", 1) for x in request.content.decode().split("&"))
            if form.get("password") == password:
                state["logins"] += 1
                return httpx.Response(302, headers={"location": "/", "set-cookie": f"session=s{state['logins']}; Path=/"})
            return httpx.Response(200, text="<p class=err>Wrong password</p>")
        if request.url.path == "/cameras.json":
            if state["expire"] and "session=s1" in request.headers.get("cookie", ""):
                state["expire"] = False
                return httpx.Response(302, headers={"location": "/auth/login"})      # session expired mid-run
            if "session=" not in request.headers.get("cookie", ""):
                return httpx.Response(302, headers={"location": "/auth/login"})
            return httpx.Response(200, json={"cameras": [{"id": "cam01"}, {"id": "cam02"}]})
        return httpx.Response(404)

    return state, handler


def grid_with(handler) -> GridClient:
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://grid.test", follow_redirects=False)
    return GridClient(make_settings(), http=http)


def test_login_then_catalogue():
    state, handler = fake_grid_site()
    cams = grid_with(handler).fetch_catalogue()
    assert [c.id for c in cams] == ["cam01", "cam02"] and state["logins"] == 1


def test_wrong_password_is_reported_without_echoing_it():
    _, handler = fake_grid_site(password="something-else")
    with pytest.raises(GridError) as e:
        grid_with(handler).fetch_catalogue()
    assert not any(s in str(e.value) for s in SECRETS) and "rejected" in str(e.value)


def test_expired_session_logs_in_again():
    state, handler = fake_grid_site(expire_once=True)
    assert len(grid_with(handler).fetch_catalogue()) == 2 and state["logins"] == 2


def test_missing_credentials_are_reported_clearly():
    with pytest.raises(GridError, match="not set"):
        GridClient(Settings(_env_file=None, sentinel_grid_email="", sentinel_grid_password="")).rtsp_url("cam01")


# ------------------------------------------------------------------------------ tracker
def det(label, box, conf=0.9):
    return Detection(label, conf, box)


def test_one_event_per_vehicle_not_one_per_frame():
    t, events = Tracker(min_hits=3), []
    for i in range(8):                                                     # a car creeping along for 8 sampled frames
        events += t.update([det("car", (100 + i * 10, 200, 300, 150))])
    assert [e.type for e in events] == ["VEHICLE_DETECTION"] and events[0].data["vehicle_class"] == "CAR"


def test_fast_vehicle_is_still_followed():
    t, events = Tracker(min_hits=3), []
    for i in range(5):                                                     # moves ~0.7 of its own width between samples
        events += t.update([det("car", (100 + i * 210, 300, 300, 200))])
    assert len(events) == 1


def test_triple_riding_needs_three_people_on_one_bike_for_several_frames():
    bike = det("motorbike", (500, 500, 150, 120))
    riders = [det("person", (490 + i * 40, 380, 50, 140)) for i in range(3)]
    t, types = Tracker(min_hits=3), []
    for _ in range(4):
        types += [e.type for e in t.update([bike, *riders])]
    assert types.count("TRIPLE_RIDING") == 1 and types.count("VEHICLE_DETECTION") == 1
    t2, types2 = Tracker(min_hits=3), []
    for _ in range(6):
        types2 += [e.type for e in t2.update([bike, *riders[:2]])]         # two riders is legal
    assert "TRIPLE_RIDING" not in types2


def test_reset_at_a_loop_point_emits_nothing_and_ids_stay_unique():
    t = Tracker(min_hits=2)
    first = [e for _ in range(3) for e in t.update([det("car", (10, 10, 100, 80))])]
    t.reset()
    second = [e for _ in range(3) for e in t.update([det("car", (10, 10, 100, 80))])]
    assert len(first) == len(second) == 1 and first[0].track_id != second[0].track_id


# -------------------------------------------------------------------------------- reader
@pytest.fixture
def clip(tmp_path: Path) -> str:
    path = str(tmp_path / "feed.mp4")
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (320, 240))
    for i in range(75):                                                    # 3 s at 25 fps
        out.write(np.full((240, 320, 3), i * 3 % 255, np.uint8))
    out.release()
    return path


def test_sampling_uses_stream_time_and_a_loop_is_a_discontinuity_not_an_error(clip):
    stop, frames = threading.Event(), []
    r = StreamReader("t", clip, sample_fps=5, stop=stop)
    for f in r.frames():
        frames.append(f)
        if len(frames) >= 20:
            stop.set()
    loop_at = next(i for i, f in enumerate(frames) if f.discontinuity and i > 0)      # the loop was noticed ...
    first_pass = [f.pts_s for f in frames[:loop_at]]
    gaps = np.diff(first_pass)
    assert len(first_pass) >= 10 and np.all(gaps > 0)                        # timestamps only move forward within a pass
    assert abs(float(np.mean(gaps)) - 0.2) < 0.05                            # 5 per second of VIDEO, whatever the read speed
    assert frames[loop_at].pts_s < first_pass[-1]                            # and the clock really did jump back
    assert r.stats.reconnects == 0                                           # ... and never counted as a failure


class _ClosedCapture:
    def isOpened(self): return False
    def release(self): pass


def test_a_dead_stream_backs_off_instead_of_spinning(monkeypatch):
    opens = []
    monkeypatch.setattr(reader_mod.cv2, "VideoCapture", lambda *a, **k: opens.append(time.monotonic()) or _ClosedCapture())
    stop = threading.Event()
    r = StreamReader("t", "rtsp://127.0.0.1:1/x", 4, stop)
    th = threading.Thread(target=lambda: list(r.frames()), daemon=True)
    th.start()
    time.sleep(1.2)
    stop.set(); th.join(timeout=5)
    assert len(opens) == 1                                                   # the retry waits >= 1.6 s (2 s -20 %), it does not spin


def test_backoff_doubles_to_the_cap():
    stop = threading.Event(); stop.set()                                     # wait() returns at once
    r, delay, seen = StreamReader("t", "rtsp://x/y", 4, stop), 2.0, []
    for _ in range(7):
        delay = r._backoff(delay, "test"); seen.append(delay)
    assert seen == [4.0, 8.0, 16.0, 30.0, 30.0, 30.0, 30.0]


# ------------------------------------------------------- adapter and the C4 pipeline
def live_event(evt_id="live-t-cam01-1-VEHICLE_DETECTION", cam="GRID-CAM01", typ="VEHICLE_DETECTION") -> dict:
    return {"kind": "event", "provider": "SENTINEL_GRID", "adapter": "LIVE_DETECTOR", "evt_id": evt_id, "node": "LIVE-DETECTOR-001",
            "cam": cam, "type": typ, "ts_ms": int(time.time() * 1000), "pts_ms": 4200, "conf": 0.91,
            "data": {"bbox": [1, 2, 30, 40], "track_id": 7, "vehicle_class": "TWO_WHEELER", "riders": 3, "frame_width": 1920, "frame_height": 1080},
            "model_ver": "yolov4-tiny-coco", "fps": 3.9, "latency_ms": 31.5}


def test_adapter_reports_honest_provenance():
    env = LiveDetectorAdapter().normalize(live_event(typ="TRIPLE_RIDING"), "GRID-CAM01", "LIVE-DETECTOR-001")
    assert env.source.provider == "SENTINEL_GRID" and env.source.adapter == "LIVE_DETECTOR"
    assert env.processing.model_version == "yolov4-tiny-coco" and env.processing.inference_fps == 3.9 and env.processing.latency_ms == 31.5
    assert env.attributes["vehicle_class"] == "TWO_WHEELER" and env.attributes["riders"] == 3 and env.attributes["stream_pts_ms"] == 4200
    assert env.evidence_ref == {}                                            # no fake mock:// evidence links
    with pytest.raises(ValueError):
        LiveDetectorAdapter().normalize({"type": "X"}, "GRID-CAM01")


@pytest.fixture(scope="module")
def api():
    with TestClient(app) as c:
        yield c


def test_live_events_flow_through_ingestion_and_no_credential_reaches_the_api(api):
    secret = "test-hmac-secret-not-the-default"
    from app.core.config import get_settings
    real_secret = get_settings().kaaval_hmac_secret
    grid = GridClient(make_settings())
    m3 = M3Client("http://testserver", real_secret, "LIVE-DETECTOR-001", http=api)
    m3.register([GridCamera("cam01", "Junction 1", width=1920, height=1080, fps=25.0)], grid._s.sentinel_grid_host,
                grid._s.sentinel_grid_rtsp_port, grid._s.sentinel_grid_web_base)
    m3.register([GridCamera("cam01", "Junction 1")], "h", 1, "w")            # a second start must be harmless (idempotent)

    evt = live_event(evt_id=f"live-pytest-{time.time_ns()}")
    r1 = m3.send([{"kind": "heartbeat", "node": "LIVE-DETECTOR-001", "runtime": "RUNNING", "fps": 3.9, "ts_ms": 0}, evt])
    assert (r1["accepted"], r1["heartbeats"], r1["rejected"]) == (1, 1, 0)
    assert m3.send([evt])["duplicates"] == 1                                 # a retried batch does not create a second event

    stored = [e for e in api.get("/api/v1/events", params={"limit": 500}).json() if e["source"]["source_event_id"] == evt["evt_id"]]
    assert len(stored) == 1 and stored[0]["camera_id"] == canonical_camera_id("cam01")
    assert stored[0]["processing"]["model_name"] == "yolov4-tiny-coco" and stored[0]["attributes"]["vehicle_class"] == "TWO_WHEELER"

    dump = "".join(api.get(p).text for p in ("/api/v1/systems", "/api/v1/cameras", "/api/v1/nodes", "/api/v1/assignments",
                                             "/api/v1/events?limit=500", "/health", "/api/v1/cameras/GRID-CAM01/stream-source"))
    assert "sentinel-grid" in dump and "GRID-CAM01" in dump
    assert not any(s in dump for s in SECRETS), "a grid credential reached an API response"
    bad = M3Client("http://testserver", "wrong-" + secret, "LIVE-DETECTOR-001", http=api)
    with pytest.raises(M3Error, match="signature"):
        bad.send([evt])
