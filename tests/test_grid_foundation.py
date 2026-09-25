"""Grid foundation: URL encoding, pipeline selection by codec, circuit breaker, probe safety. All offline."""
import subprocess
from urllib.parse import unquote, urlsplit

import pytest

from app.grid import urls
from app.grid.breaker import CircuitBreaker
from app.grid.probe import probe_rtsp
from app.live.redact import Redactor
from contract import AuthFailed, ConnectionFailed, DeviceTimeout

NASTY = ["a@b.com", "p@ss:w/rd#1?x%2", "100%", "we ird+na me@x.io", "colon:slash/hash#q?pct%40"]


@pytest.mark.parametrize("email", NASTY)
@pytest.mark.parametrize("password", NASTY)
def test_credentials_round_trip_through_a_url(email, password):
    """Whatever the raw email/password contain, a URL parser gives them back exactly."""
    for url in (urls.rtsp_url(email, password, "1.2.3.4", 8554, "cam04"),
                urls.whep_source_url(email, password, "1.2.3.4", 8889, "cam04")):
        parts = urlsplit(url)
        assert unquote(parts.username) == email
        assert unquote(parts.password) == password
        assert parts.hostname == "1.2.3.4"
        assert parts.path.endswith("cam04") or parts.path.endswith("/whep")


def test_at_sign_is_encoded_exactly_once():
    url = urls.rtsp_url("kaaval-ai.kanyakumari@gmail.com", "EU9Q-7FE7-SBJ9", "103.250.160.189", 8554, "cam04")
    assert url == "rtsp://kaaval-ai.kanyakumari%40gmail.com:EU9Q-7FE7-SBJ9@103.250.160.189:8554/stream/cam04"
    assert "%2540" not in url                         # the classic double-encoding bug
    assert url.count("@") == 1                        # only the userinfo/host separator is a raw "@"


def test_percent_in_a_raw_password_is_encoded_not_reinterpreted():
    url = urls.rtsp_url("u@x.io", "50%40off", "h", 8554, "c")
    assert "50%2540off" in url                        # the RAW value contained "%40": that "%" must become %25
    assert unquote(urlsplit(url).password) == "50%40off"


def test_public_urls_carry_no_credentials():
    for u in (urls.rtsp_public("h", 8554, "cam04"), urls.whep_public("h", 8889, "cam04"),
              urls.hls_public("https://cctv.corp8.cloud/", "cam04")):
        assert "@" not in u
    assert urls.hls_public("https://cctv.corp8.cloud/", "cam04") == "https://cctv.corp8.cloud/cam04/index.m3u8"


@pytest.mark.parametrize("name,expected", [("h264", "h264"), ("H.264", "h264"), ("avc1", "h264"), ("hevc", "h265"),
                                           ("H265", "h265"), ("hvc1", "h265"), ("mjpeg", "mjpeg"), (None, None), ("", None)])
def test_codec_names_are_normalised(name, expected):
    assert urls.norm_codec(name) == expected


def test_gstreamer_pipeline_follows_the_codec():
    h264 = urls.gstreamer_pipeline("rtsp://x/y", "h264")
    h265 = urls.gstreamer_pipeline("rtsp://x/y", "hevc")
    unknown = urls.gstreamer_pipeline("rtsp://x/y", None)
    assert "rtph264depay ! h264parse ! avdec_h264" in h264 and "265" not in h264
    assert "rtph265depay ! h265parse ! avdec_h265" in h265 and "264" not in h265
    assert "decodebin" in unknown and "depay" not in unknown        # never guess H.264
    assert all("protocols=tcp" in p for p in (h264, h265, unknown))


def test_ffmpeg_web_args_copy_h264_and_transcode_the_rest():
    assert urls.ffmpeg_web_args("h264") == ["-c", "copy"]
    for codec in ("hevc", "h265", "mjpeg", None):
        args = urls.ffmpeg_web_args(codec)
        assert "libx264" in args and "zerolatency" in args and "veryfast" in args
        assert args[args.index("-g") + 1] == "30"
    assert not urls.needs_web_transcode("H.264") and urls.needs_web_transcode("hevc") and urls.needs_web_transcode(None)


def test_circuit_breaker_opens_half_opens_and_closes():
    now = [0.0]
    b = CircuitBreaker(threshold=3, reset_s=30, clock=lambda: now[0])
    assert b.state == "closed" and b.allow()
    b.failure(); b.failure()
    assert b.allow()                                   # still closed at 2 failures
    b.failure()
    assert b.state == "open" and not b.allow()         # 3rd failure opens it
    now[0] = 29
    assert not b.allow()
    now[0] = 31
    assert b.state == "half-open" and b.allow()        # one trial call
    assert not b.allow()                               # ...and only one at a time
    b.failure()                                        # trial failed: straight back to open
    assert b.state == "open"
    now[0] = 70
    assert b.allow()
    b.success()
    assert b.state == "closed" and b.allow()


class _Done:
    def __init__(self, rc=0, out="", err=""):
        self.returncode, self.stdout, self.stderr = rc, out, err


def _patch_ffprobe(monkeypatch, result=None, raises=None):
    monkeypatch.setattr("app.grid.probe.shutil.which", lambda _: "ffprobe")
    calls = []

    def run(cmd, **kw):
        calls.append((cmd, kw))
        if raises:
            raise raises
        return result
    monkeypatch.setattr("app.grid.probe.subprocess.run", run)
    return calls


def test_probe_uses_an_argument_list_tcp_and_timeouts(monkeypatch):
    calls = _patch_ffprobe(monkeypatch, _Done(out='{"streams":[{"codec_name":"hevc","width":1920,"height":1080,"avg_frame_rate":"25/1"}]}'))
    res = probe_rtsp("rtsp://u:p@h:8554/stream/cam06", Redactor("u", "p"))
    cmd, kw = calls[0]
    assert isinstance(cmd, list) and not kw.get("shell")
    assert cmd[cmd.index("-rtsp_transport") + 1] == "tcp"
    assert cmd[cmd.index("-timeout") + 1] == "5000000"       # 5 s, in microseconds
    assert kw["timeout"] > 5                                 # a hard subprocess timeout as well
    assert (res.codec, res.resolution, res.fps) == ("h265", "1920x1080", 25.0)


def test_probe_errors_never_contain_credentials(monkeypatch):
    secret_email, secret_pw = "victim@example.com", "S3cr3t-Pa55"
    url = f"rtsp://victim%40example.com:S3cr3t-Pa55@h:8554/stream/cam1"
    err = f"{url}: Server returned 401 Unauthorized (authorization failed)"
    _patch_ffprobe(monkeypatch, _Done(rc=1, err=err))
    with pytest.raises(AuthFailed) as ei:
        probe_rtsp(url, Redactor(secret_email, secret_pw))
    assert secret_pw not in str(ei.value) and "victim" not in str(ei.value)


def test_probe_classifies_timeout_and_failure(monkeypatch):
    _patch_ffprobe(monkeypatch, raises=subprocess.TimeoutExpired("ffprobe", 15))
    with pytest.raises(DeviceTimeout):
        probe_rtsp("rtsp://h/x", Redactor())
    _patch_ffprobe(monkeypatch, _Done(rc=1, err="rtsp://h/x: Connection refused"))
    with pytest.raises(ConnectionFailed):
        probe_rtsp("rtsp://h/x", Redactor())
    _patch_ffprobe(monkeypatch, _Done(out='{"streams":[]}'))
    with pytest.raises(ConnectionFailed):
        probe_rtsp("rtsp://h/x", Redactor())
