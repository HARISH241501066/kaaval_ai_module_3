"""
Endpoint URLs for the Sentinel Camera Grid, and which pipeline suits which codec.

Credentials live raw in .env ("@" and all) and are percent-encoded exactly once, here, when a URL is
built. `enc()` never touches an already-encoded value twice, because it is only ever given raw ones.

Credentialed builders (`rtsp_url`, `whep_source_url`) return secrets: use them in memory or write them to
the git-ignored relay config, never log or return them. The `*_public` builders carry no credentials.
"""
from __future__ import annotations

from urllib.parse import quote

RTSP_PORT_DEFAULT = 8554
WHEP_PORT_DEFAULT = 8889


def enc(raw: str) -> str:
    """Percent-encode one credential component. '@' -> %40, ':' -> %3A, '/' -> %2F, '%' -> %25 ..."""
    return quote(raw, safe="")


# ------------------------------------------------------------------ credentialed
def rtsp_url(email: str, password: str, host: str, port: int, cam_id: str) -> str:
    return f"rtsp://{enc(email)}:{enc(password)}@{host}:{port}/stream/{cam_id}"


def whep_source_url(email: str, password: str, host: str, port: int, cam_id: str) -> str:
    """Form MediaMTX accepts as a pull source (`whep://user:pass@host:port/path`)."""
    return f"whep://{enc(email)}:{enc(password)}@{host}:{port}/stream/{cam_id}/whep"


# ---------------------------------------------------------------------- public
def rtsp_public(host: str, port: int, cam_id: str) -> str:
    return f"rtsp://{host}:{port}/stream/{cam_id}"


def whep_public(host: str, port: int, cam_id: str) -> str:
    return f"http://{host}:{port}/stream/{cam_id}/whep"


def hls_public(web_base: str, cam_id: str) -> str:
    return f"{web_base.rstrip('/')}/{cam_id}/index.m3u8"


# ------------------------------------------------------------------- codecs
def norm_codec(name: str | None) -> str | None:
    """ffprobe/catalogue spellings -> 'h264' | 'h265' | other lower-case name | None."""
    if not name:
        return None
    n = name.strip().lower().replace(".", "").replace("-", "")
    if n in ("h264", "avc", "avc1"):
        return "h264"
    if n in ("h265", "hevc", "hev1", "hvc1"):
        return "h265"
    return n


def gstreamer_pipeline(url: str, codec: str | None) -> str:
    """
    A GStreamer pipeline for an RTSP URL (TCP only). Never hard-codes H.264: the depay/parse/decode
    elements follow the probed codec, and an unknown codec falls back to decodebin, which works for both.
    """
    src = f'rtspsrc location="{url}" protocols=tcp latency=200'
    c = norm_codec(codec)
    if c == "h264":
        chain = "rtph264depay ! h264parse ! avdec_h264"
    elif c == "h265":
        chain = "rtph265depay ! h265parse ! avdec_h265"
    else:
        chain = "decodebin"
    return f"{src} ! {chain} ! videoconvert ! appsink"


WEB_TRANSCODE_ARGS = ["-an", "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency", "-pix_fmt", "yuv420p", "-g", "30"]


def needs_web_transcode(codec: str | None) -> bool:
    """Browsers need H.264. Unknown codec is transcoded too: always playable, at the price of some CPU."""
    return norm_codec(codec) != "h264"


def ffmpeg_web_args(codec: str | None) -> list[str]:
    """Output-side ffmpeg arguments that turn a camera stream into what a browser can play."""
    return WEB_TRANSCODE_ARGS if needs_web_transcode(codec) else ["-c", "copy"]
