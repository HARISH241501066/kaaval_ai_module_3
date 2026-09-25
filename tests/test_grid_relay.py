"""SENTINEL_GRID adapter, C2 sync, relay config and source failover, broker endpoint. All offline (fakes)."""
import json
import logging

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete

from app.adapters.sentinel_grid import SentinelGridAdapter
from app.core.config import Settings
from app.db.database import SessionLocal, init_db
from app.grid.breaker import CircuitBreaker
from app.grid.probe import ProbeResult
from app.live.grid import GridCamera, GridClient, GridError
from app.live.m3 import SYSTEM_ID
from app.models.grid import CameraMappingModel, StreamSourceModel
from app.models.system import CameraModel, SystemModel
from app.relay import config as cfg
from app.relay.manager import RelayManager
from app.services import grid_sync
from contract import CapabilityNotSupported, ConnectionFailed, DeviceTimeout, MappingError

EMAIL, PASSWORD = "seeded.user@example.com", "S33ded:Pa/ss#1"


def make_settings(**kw) -> Settings:
    base = dict(sentinel_grid_email=EMAIL, sentinel_grid_password=PASSWORD, sentinel_grid_host="grid.test",
                relay_public_host="relay.test", ffmpeg_path="C:\\tools\\ffmpeg.exe", relay_config_path="unused.yml",
                _env_file=None)
    return Settings(**{**base, **kw})


class FakeGrid(GridClient):
    """GridClient whose network calls are scripted."""

    def __init__(self, settings, cameras=None, fail_times=0):
        super().__init__(settings)
        self.cameras, self.fail_times, self.calls = cameras or [], fail_times, 0

    def fetch_catalogue(self):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise GridError("grid web endpoint unreachable (ConnectError)")
        return list(self.cameras)


def cams(*ids):
    return [GridCamera(i, f"Camera {i}") for i in ids]


@pytest.fixture(autouse=True)
def clean_grid_rows():
    init_db()

    def wipe():
        with SessionLocal() as db:
            db.execute(delete(StreamSourceModel))
            db.execute(delete(CameraMappingModel))
            db.execute(delete(CameraModel).where(CameraModel.system_id == SYSTEM_ID))
            db.execute(delete(SystemModel).where(SystemModel.id == SYSTEM_ID))
            db.commit()
    wipe()
    yield
    wipe()


# ---------------------------------------------------------------- adapter (C1)
def test_capabilities_are_live_stream_and_health_only():
    a = SentinelGridAdapter(make_settings())
    assert a.capabilities == {"LIVE_STREAM", "HEALTH"}
    for call in (a.ptz, a.playback, a.export_clip, a.snapshot):
        with pytest.raises(CapabilityNotSupported) as ei:
            call()
        assert ei.value.code.value == "CAPABILITY_NOT_SUPPORTED"
    with pytest.raises(CapabilityNotSupported):
        a.normalize({}, "GRID-CAM01")
    assert a.validate_payload({"anything": 1}) is False


def test_stream_uri_candidates_are_ordered_and_credential_free():
    a = SentinelGridAdapter(make_settings())
    got = a.get_stream_uri("cam04")
    assert [c.protocol for c in got] == ["RTSP", "WHEP"]           # HLS is not offered: not usable server-side
    assert got[0].url == "rtsp://grid.test:8554/stream/cam04"
    assert got[1].url == "http://grid.test:8889/stream/cam04/whep"
    assert all("@" not in c.url and "seeded" not in c.url for c in got)
    a.hls_server_side = True                                        # only if it was measured to work
    assert [c.protocol for c in a.get_stream_uri("cam04")] == ["RTSP", "WHEP", "HLS"]


def test_source_url_for_the_relay_is_encoded_once_and_hls_is_refused():
    a = SentinelGridAdapter(make_settings())
    assert a.source_url("RTSP", "cam04").startswith("rtsp://seeded.user%40example.com:S33ded%3APa%2Fss%231@grid.test:8554/")
    assert a.source_url("WHEP", "cam04").startswith("whep://seeded.user%40example.com:")
    with pytest.raises(CapabilityNotSupported):
        a.source_url("HLS", "cam04")


def test_guard_retries_twice_with_backoff_then_succeeds():
    slept = []
    s = make_settings()
    a = SentinelGridAdapter(s, grid=FakeGrid(s, cams("cam01"), fail_times=2), sleep=slept.append)
    assert [c.id for c in a.inventory()] == ["cam01"]
    assert len(slept) == 2 and slept[1] > slept[0] * 1.2            # exponential (0.5 s, then 1 s), jittered


def test_guard_gives_up_after_two_retries_and_the_breaker_opens():
    s = make_settings()
    grid = FakeGrid(s, cams("cam01"), fail_times=99)
    a = SentinelGridAdapter(s, grid=grid, breaker=CircuitBreaker(threshold=2, reset_s=30), sleep=lambda _: None)
    for _ in range(2):
        with pytest.raises(ConnectionFailed):
            a.inventory()
    assert grid.calls == 6                                          # 2 calls x (1 try + 2 retries)
    with pytest.raises(ConnectionFailed, match="circuit breaker"):
        a.inventory()
    assert grid.calls == 6                                          # open: the grid is not touched at all


def test_wrong_credentials_are_not_retried():
    s = make_settings()

    class Rejecting(FakeGrid):
        def fetch_catalogue(self):
            self.calls += 1
            raise GridError("the grid rejected the login (check SENTINEL_GRID_EMAIL / SENTINEL_GRID_PASSWORD)")
    grid = Rejecting(s)
    with pytest.raises(Exception) as ei:
        SentinelGridAdapter(s, grid=grid, sleep=lambda _: None).inventory()
    assert ei.value.code.value == "VENDOR_AUTH_FAILED" and grid.calls == 1


# ------------------------------------------------------------------- C7 + C2
class StubAdapter(SentinelGridAdapter):
    def __init__(self, catalogue, probes=None):
        super().__init__(make_settings())
        self.catalogue, self.probes = catalogue, probes or {}
        self.hls_server_side, self.hls_note = False, 'HTTP 403 "browser required"'

    def inventory(self):
        return list(self.catalogue)

    def probe(self, cam_id):
        r = self.probes[cam_id]
        if isinstance(r, Exception):
            raise r
        return r

    def connection_test(self, sample_camera=None):
        if self.probes.get("gate") == "fail":
            raise ConnectionFailed("cannot reach the grid")
        return {"cameras": len(self.catalogue), "probe_camera": "cam01", "codec": "h264", "resolution": "1x1",
                "hls_server_side": False, "hls_note": self.hls_note}


def test_system_becomes_active_only_after_the_connection_test_passes():
    with SessionLocal() as db:
        res = grid_sync.register_system(db, StubAdapter(cams("cam01"), {"gate": "fail"}))
        assert res["status"] == "INACTIVE" and res["error_code"] == "CONNECTION_FAILED"
        assert db.get(SystemModel, SYSTEM_ID).status == "INACTIVE"
        res = grid_sync.register_system(db, StubAdapter(cams("cam01")))
        system = db.get(SystemModel, SYSTEM_ID)
    assert res["status"] == "ACTIVE" and system.status == "ACTIVE"
    params = system.connection_params
    assert params["adapter_key"] == "SENTINEL_GRID"
    assert set(params["endpoints"]) == {"rtsp", "hls", "whep"}      # all three templates stored
    assert params["endpoints"]["hls"] == "https://cctv.corp8.cloud/{id}/index.m3u8"
    assert params["hls_server_side"] is False
    blob = json.dumps(params)
    assert EMAIL not in blob and PASSWORD not in blob and "@" not in blob


def test_sync_creates_every_catalogue_camera_with_external_ids_and_codecs():
    ids = [f"cam{i:02d}" for i in range(1, 31)]
    probes = {i: ProbeResult("h265" if i in ("cam06", "cam12") else "h264", 1920, 1080, 25.0) for i in ids}
    with SessionLocal() as db:
        rep = grid_sync.sync_inventory(db, StubAdapter(cams(*ids), probes), probe_limit=60)
        maps = db.query(CameraMappingModel).all()
        cameras = db.query(CameraModel).filter(CameraModel.system_id == SYSTEM_ID).all()
    assert rep.catalogue == 30 and rep.created == 30 and rep.probed == 30
    assert len(maps) == 30 and len(cameras) == 30                   # the correct count, not a hard-coded one
    assert {m.system_id for m in maps} == {"sentinel-grid"} and {m.external_id for m in maps} == set(ids)
    assert {m.external_id for m in maps if m.codec == "h265"} == {"cam06", "cam12"}
    assert all(c.rtsp_url.startswith("rtsp://grid.test:8554/") and "@" not in c.rtsp_url for c in cameras)


def test_the_set_of_cameras_is_taken_from_the_catalogue_not_hard_coded():
    with SessionLocal() as db:
        rep = grid_sync.sync_inventory(db, StubAdapter(cams("cam01", "cam02", "cam03")))
        assert rep.catalogue == 3 and db.query(CameraMappingModel).count() == 3


def test_camera_missing_for_three_runs_goes_inactive_but_is_never_deleted():
    with SessionLocal() as db:
        grid_sync.sync_inventory(db, StubAdapter(cams("cam01", "cam02")))
        for run in (1, 2, 3):
            rep = grid_sync.sync_inventory(db, StubAdapter(cams("cam01")))
            m = db.query(CameraMappingModel).filter_by(external_id="cam02").one()
            assert m.missed_runs == run
            assert m.status == ("INACTIVE" if run == 3 else "ACTIVE")
        assert rep.newly_inactive == ["cam02"]
        assert db.get(CameraModel, "GRID-CAM02") is not None        # still there
        rep = grid_sync.sync_inventory(db, StubAdapter(cams("cam01", "cam02")))     # it comes back
        m = db.query(CameraMappingModel).filter_by(external_id="cam02").one()
        assert rep.reactivated == 1 and m.status == "ACTIVE" and m.missed_runs == 0


def test_a_failed_catalogue_fetch_marks_nothing_missing():
    class Down(StubAdapter):
        def inventory(self):
            raise ConnectionFailed("grid unreachable")
    with SessionLocal() as db:
        grid_sync.sync_inventory(db, StubAdapter(cams("cam01")))
        with pytest.raises(ConnectionFailed):
            grid_sync.sync_inventory(db, Down([]))
        assert db.query(CameraMappingModel).one().missed_runs == 0


def test_probe_failure_marks_the_camera_offline_without_leaking():
    probes = {"cam01": ProbeResult("h264", 1280, 720, 25.0), "cam02": DeviceTimeout("probe timed out")}
    with SessionLocal() as db:
        rep = grid_sync.sync_inventory(db, StubAdapter(cams("cam01", "cam02"), probes), probe_limit=5)
        assert rep.probe_failed == ["cam02"]
        assert db.get(CameraModel, "GRID-CAM02").status == "OFFLINE"
        assert db.get(CameraModel, "GRID-CAM01").resolution == "1280x720"
        assert db.query(CameraMappingModel).filter_by(external_id="cam02").one().probe_error.startswith("VENDOR_TIMEOUT")


# ------------------------------------------------------------- relay config (C3)
def _paths(codec, protocol="RTSP"):
    s = make_settings()
    a = SentinelGridAdapter(s)
    return cfg.path_settings(cfg.CameraPlan("cam04", codec, protocol), s, a.source_url(protocol, "cam04"), "C:/tools/ffmpeg.exe"), s


def test_h264_camera_web_path_is_a_copy_pulled_from_our_own_main_path():
    paths, s = _paths("h264")
    main, web = paths["cam/cam04/main"], paths["cam/cam04/web"]
    assert main["source"].startswith("rtsp://") and main["rtspTransport"] == "tcp"
    assert web["source"] == f"rtsp://127.0.0.1:{s.relay_rtsp_port}/cam/cam04/main"      # one upstream, however many paths
    assert "runOnDemand" not in web


def test_h265_camera_web_path_is_an_on_demand_libx264_transcode():
    paths, s = _paths("h265")
    web = paths["cam/cam04/web"]
    assert "source" not in web
    cmd = web["runOnDemand"]
    assert cmd.startswith("C:/tools/ffmpeg.exe ") and "libx264" in cmd and "-preset veryfast" in cmd
    assert "-tune zerolatency" in cmd and "-g 30" in cmd
    assert f"-i rtsp://127.0.0.1:{s.relay_rtsp_port}/cam/cam04/main" in cmd and cmd.endswith("/$MTX_PATH")
    assert web["runOnDemandRestart"] is True


def test_every_path_is_on_demand_and_closes_after_60s():
    for codec in ("h264", "h265"):
        for path in _paths(codec)[0].values():
            assert (path.get("sourceOnDemand") or path.get("runOnDemand")) and "60s" in json.dumps(path)
    assert _paths("h264")[0]["cam/cam04/main"]["sourceOnDemandCloseAfter"] == "60s"


def test_whep_main_path_has_no_rtsp_transport_and_unknown_codec_is_transcoded():
    assert "rtspTransport" not in _paths("h264", "WHEP")[0]["cam/cam04/main"]
    assert "libx264" in _paths(None)[0]["cam/cam04/web"]["runOnDemand"]


def test_rendered_yaml_is_valid_quotes_secrets_and_locks_down_access():
    yaml = pytest.importorskip("yaml")
    paths, s = _paths("h265")
    doc = yaml.safe_load(cfg.render(s, paths))
    assert doc["paths"]["cam/cam04/main"]["source"].startswith("rtsp://seeded.user%40example.com:")
    assert doc["rtspTransports"] == ["tcp"] and doc["rtmp"] is False and doc["srt"] is False
    users = doc["authInternalUsers"]
    assert users[0]["permissions"] == [{"action": "read"}] and users[0]["ips"] == []      # the world can only watch
    assert {"action": "publish"} in users[1]["permissions"] and users[1]["ips"] == ["127.0.0.1", "::1"]
    assert yaml.safe_load(cfg.render(s, {})) ["paths"] == {}


def test_generated_config_location_is_git_ignored():
    import subprocess
    r = subprocess.run(["git", "check-ignore", "-q", "relay/generated/mediamtx.yml"], capture_output=True)
    assert r.returncode == 0


# ------------------------------------------------------------- failover (C3)
class RelayFake:
    """MediaMTX API stand-in: records path replacements, says whether a path is ready."""

    def __init__(self, ready=()):
        self.ready, self.puts, self.requests = set(ready), [], []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append((request.method, request.url.path))
        p = request.url.path
        if p.startswith("/v3/paths/list"):
            return httpx.Response(200, json={"items": []})
        if p.startswith("/v3/paths/get/"):
            name = p.removeprefix("/v3/paths/get/")
            return httpx.Response(200, json={"name": name, "ready": name in self.ready})
        if p.startswith("/v3/config/paths/replace/"):
            self.puts.append((p.removeprefix("/v3/config/paths/replace/"), json.loads(request.content)))
            return httpx.Response(200)
        return httpx.Response(404)


def make_manager(relay: RelayFake, probe_results):
    s = make_settings()
    mgr = RelayManager(s, SentinelGridAdapter(s), http=httpx.Client(base_url="http://relay-api", transport=httpx.MockTransport(relay)))
    calls = []

    def fake_probe(url, redactor, **kw):
        calls.append(url)
        r = probe_results.pop(0)
        if isinstance(r, Exception):
            raise r
        return r
    return mgr, calls, fake_probe


def seed_camera(codec="h264"):
    with SessionLocal() as db:
        grid_sync.sync_inventory(db, StubAdapter(cams("cam04")))
        m = db.query(CameraMappingModel).filter_by(external_id="cam04").one()
        m.codec = codec
        db.commit()


def main_source_of(relay: RelayFake, idx=-1):
    puts = [c for n, c in relay.puts if n == "cam/cam04/main"]
    return puts[idx]["source"]


def test_rtsp_is_tried_first_and_recorded_as_active(monkeypatch):
    seed_camera()
    relay = RelayFake()
    mgr, calls, fake = make_manager(relay, [ProbeResult("h264", 1920, 1080, 25.0)])
    monkeypatch.setattr("app.relay.manager.probe_rtsp", fake)
    with SessionLocal() as db:
        st = mgr.ensure(db, "cam04")
    assert st["active_source"] == "RTSP" and main_source_of(relay).startswith("rtsp://")
    assert calls == ["rtsp://127.0.0.1:8654/cam/cam04/main"]        # the relay was asked, not the grid


def test_blocked_rtsp_fails_over_to_whep_and_the_status_says_so(monkeypatch):
    seed_camera()
    relay = RelayFake()
    mgr, calls, fake = make_manager(relay, [ConnectionFailed("probe failed (Connection refused)"), ProbeResult("h264", 1920, 1080, 25.0)])
    monkeypatch.setattr("app.relay.manager.probe_rtsp", fake)
    with SessionLocal() as db:
        st = mgr.ensure(db, "cam04")
        rows = {r.protocol: r for r in db.query(StreamSourceModel).all()}
    assert st["active_source"] == "WHEP"
    assert main_source_of(relay).startswith("whep://")
    assert rows["RTSP"].available is False and "CONNECTION_FAILED" in rows["RTSP"].detail
    assert rows["WHEP"].active is True
    assert rows["HLS"].available is False and "not usable server-side" in rows["HLS"].detail   # explicit, never a silent 401


def test_camera_returns_to_rtsp_when_rtsp_recovers(monkeypatch):
    seed_camera()
    relay = RelayFake()
    mgr, _, fake = make_manager(relay, [ConnectionFailed("down"), ProbeResult("h264", 1, 1, 1.0),
                                        ProbeResult("h264", 1, 1, 1.0)])
    monkeypatch.setattr("app.relay.manager.probe_rtsp", fake)
    with SessionLocal() as db:
        assert mgr.ensure(db, "cam04")["active_source"] == "WHEP"
        assert mgr.ensure(db, "cam04")["active_source"] == "RTSP"   # RTSP is always tried first again


def test_a_camera_that_is_already_flowing_is_not_touched(monkeypatch):
    seed_camera()
    relay = RelayFake(ready={"cam/cam04/main"})
    mgr, calls, fake = make_manager(relay, [])
    monkeypatch.setattr("app.relay.manager.probe_rtsp", fake)
    with SessionLocal() as db:
        mgr.ensure(db, "cam04")
    assert calls == [] and relay.puts == []                         # no probe, no reconfiguration: no extra grid connection


def test_all_protocols_failing_reports_unavailable_without_raising(monkeypatch):
    seed_camera()
    relay = RelayFake()
    mgr, _, fake = make_manager(relay, [DeviceTimeout("probe timed out"), DeviceTimeout("probe timed out")])
    monkeypatch.setattr("app.relay.manager.probe_rtsp", fake)
    with SessionLocal() as db:
        st = mgr.ensure(db, "cam04")
    assert st["active_source"] is None and not any(s["available"] for s in st["sources"])


def test_real_codec_from_the_stream_replaces_the_catalogue_guess(monkeypatch):
    seed_camera("h264")
    relay = RelayFake()
    mgr, _, fake = make_manager(relay, [ProbeResult("h265", 2560, 1440, 25.0)])
    monkeypatch.setattr("app.relay.manager.probe_rtsp", fake)
    with SessionLocal() as db:
        mgr.ensure(db, "cam04")
        assert db.query(CameraMappingModel).one().codec == "h265"
    web = [c for n, c in relay.puts if n == "cam/cam04/web"][-1]
    assert "libx264" in web["runOnDemand"]                          # the web path was rebuilt as a transcode


def test_unknown_camera_is_a_mapping_error():
    mgr, _, _ = make_manager(RelayFake(), [])
    with SessionLocal() as db, pytest.raises(MappingError):
        mgr.ensure(db, "cam99")


# ------------------------------------------------------------- broker endpoint
@pytest.fixture()
def api_client(monkeypatch):
    from app.main import app
    with TestClient(app) as c:
        yield c


def test_broker_returns_only_relay_urls_and_no_credentials(api_client, monkeypatch):
    seed_camera("h265")
    s = make_settings()
    mgr = RelayManager(s, SentinelGridAdapter(s))
    monkeypatch.setattr("app.api.grid.get_relay_manager", lambda: mgr)
    body = api_client.get("/federation/cameras/cam04/stream-sources?profile=web")
    assert body.status_code == 200
    j = body.json()
    assert j["urls"]["whep"] == "http://relay.test:8889/cam/cam04/web/whep"
    assert j["urls"]["hls"] == "http://relay.test:8888/cam/cam04/web/index.m3u8"
    assert j["urls"]["rtsp"] == "rtsp://relay.test:8654/cam/cam04/web"
    assert j["codec"] == "h264" and j["source_codec"] == "h265" and j["transcoded"] is True
    for needle in (EMAIL, PASSWORD, "seeded", "grid.test", "cctv.corp8.cloud", "103.250.160.189"):
        assert needle not in body.text
    main = api_client.get("/federation/cameras/GRID-CAM04/stream-sources?profile=main").json()   # canonical id works too
    assert main["codec"] == "h265" and main["transcoded"] is False and main["urls"]["whep"].endswith("/cam/cam04/main/whep")


def test_broker_404s_for_cameras_that_are_not_grid_cameras(api_client):
    r = api_client.get("/federation/cameras/CAM-001/stream-sources")
    assert r.status_code == 404 and r.json()["detail"]["error"] == "MAPPING_ERROR"
    assert api_client.get("/federation/cameras/cam04/stream-sources?profile=hd").status_code == 422


def test_wall_inventory_lists_cameras_from_module_3(api_client):
    with SessionLocal() as db:
        grid_sync.sync_inventory(db, StubAdapter(cams("cam01", "cam02"), {"cam01": ProbeResult("h264", 1920, 1080, 25.0), "cam02": ConnectionFailed("x")}), probe_limit=5)
    rows = {r["id"]: r for r in api_client.get("/api/v1/grid/cameras").json()}
    assert set(rows) == {"cam01", "cam02"}
    assert rows["cam01"]["codec"] == "h264" and rows["cam01"]["resolution"] == "1920x1080" and rows["cam01"]["status"] == "ONLINE"
    assert rows["cam02"]["status"] == "OFFLINE" and rows["cam01"]["name"] == "Camera cam01"


def test_openapi_documents_the_new_endpoints(api_client):
    paths = api_client.get("/openapi.json").json()["paths"]
    for p in ("/api/v1/grid/connect", "/api/v1/grid/sync", "/api/v1/grid/cameras", "/federation/cameras/{camera_id}/stream-sources"):
        assert p in paths
