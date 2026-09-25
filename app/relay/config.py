"""
MediaMTX configuration, generated from the camera registry (never hand-written).

For every ACTIVE grid camera two paths exist:

    cam/<id>/main   the camera as it is (original codec, stream copy): for backend consumers
    cam/<id>/web    always H.264: for browsers.
                    H.264 camera -> pulled from our own `main` path (stream copy)
                    H.265 camera -> ffmpeg libx264 transcode, started on demand and restarted if it dies

`web` is fed from `main`, never from the grid, so however many viewers and paths there are, the grid sees
ONE upstream connection per camera. Both are on-demand: the upstream opens with the first viewer and
closes `relay_idle_close_s` (60 s) after the last one leaves.

The output holds the grid credentials (they are the pull-source URLs), so it is written to a git-ignored
file (relay/generated/) and nowhere else.
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.grid import urls
from contract import MappingError

MAIN, WEB = "main", "web"


def path_name(cam_id: str, profile: str) -> str:
    return f"cam/{cam_id}/{profile}"


@dataclass(frozen=True)
class CameraPlan:
    cam_id: str                  # grid id, e.g. "cam04"
    codec: str | None            # probed source codec
    protocol: str = "RTSP"       # which source the `main` path pulls over: "RTSP" | "WHEP"


def find_ffmpeg(settings: Settings) -> str:
    """
    Absolute ffmpeg path in POSIX form. MediaMTX splits `runOnDemand` shell-style, which eats Windows
    backslashes, and its process launcher cannot follow WinGet's shim links: resolve both here.
    """
    exe = settings.ffmpeg_path or shutil.which("ffmpeg")
    if not exe:
        raise MappingError("ffmpeg was not found: install it or set FFMPEG_PATH (needed to transcode H.265 cameras for browsers)")
    return Path(os.path.realpath(exe)).as_posix()


def local_rtsp(settings: Settings, path: str) -> str:
    return f"rtsp://127.0.0.1:{settings.relay_rtsp_port}/{path}"


def path_settings(plan: CameraPlan, settings: Settings, source_url: str, ffmpeg: str) -> dict[str, dict[str, Any]]:
    """The two path definitions for one camera, in MediaMTX's own parameter names (used for both the file and the API)."""
    idle = f"{settings.relay_idle_close_s}s"
    start = f"{settings.relay_probe_timeout_s}s"
    main: dict[str, Any] = {"source": source_url, "sourceOnDemand": True, "sourceOnDemandStartTimeout": start,
                            "sourceOnDemandCloseAfter": idle}
    if source_url.startswith("rtsp"):
        main["rtspTransport"] = "tcp"
    if urls.needs_web_transcode(plan.codec):
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "warning", "-rtsp_transport", "tcp",
               "-i", local_rtsp(settings, path_name(plan.cam_id, MAIN)), *urls.ffmpeg_web_args(plan.codec),
               "-f", "rtsp", "-rtsp_transport", "tcp", f"rtsp://127.0.0.1:{settings.relay_rtsp_port}/$MTX_PATH"]
        web: dict[str, Any] = {"runOnDemand": " ".join(shlex.quote(c) if " " in c else c for c in cmd),
                               "runOnDemandRestart": True, "runOnDemandStartTimeout": "20s",
                               "runOnDemandCloseAfter": idle}
    else:
        web = {"source": local_rtsp(settings, path_name(plan.cam_id, MAIN)), "rtspTransport": "tcp",
               "sourceOnDemand": True, "sourceOnDemandStartTimeout": start, "sourceOnDemandCloseAfter": idle}
    return {path_name(plan.cam_id, MAIN): main, path_name(plan.cam_id, WEB): web}


def all_paths(plans: Iterable[CameraPlan], settings: Settings, source_for: Mapping[tuple[str, str], str],
              ffmpeg: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for plan in plans:
        out.update(path_settings(plan, settings, source_for[(plan.protocol, plan.cam_id)], ffmpeg))
    return out


# ------------------------------------------------------------------ YAML output
def _scalar(v: Any) -> str:
    return json.dumps(v)                     # JSON scalars are valid YAML and quote every special character safely


def _yaml(d: Mapping[str, Any], indent: int = 0) -> list[str]:
    pad, lines = "  " * indent, []
    for k, v in d.items():
        if isinstance(v, Mapping):
            lines.append(f"{pad}{_scalar(k)}:")
            lines += _yaml(v, indent + 1)
        else:
            lines.append(f"{pad}{_scalar(k)}: {_scalar(v)}")
    return lines


def render(settings: Settings, paths: Mapping[str, Mapping[str, Any]]) -> str:
    origins = [o.strip() for o in settings.relay_allow_origins.split(",") if o.strip()]
    api_host = settings.relay_api_url.split("://", 1)[-1]
    head: dict[str, Any] = {
        "logLevel": "info", "logDestinations": ["stdout"],
        "api": True, "apiAddress": api_host, "metrics": True, "metricsAddress": "127.0.0.1:9998",
        "pprof": False, "playback": False, "rtmp": False, "srt": False, "moq": False,
        "writeQueueSize": settings.relay_write_queue_size,
        "rtsp": True, "rtspAddress": f":{settings.relay_rtsp_port}", "rtspTransports": ["tcp"],
        "hls": True, "hlsAddress": f":{settings.relay_hls_port}", "hlsAlwaysRemux": False, "hlsAllowOrigins": origins,
        "webrtc": True, "webrtcAddress": f":{settings.relay_webrtc_port}", "webrtcAllowOrigins": origins,
    }
    lines = ["# GENERATED by app.relay: do not edit, do not commit. Contains the Sentinel Grid credentials."]
    lines += [f"{_scalar(k)}: {_scalar(v)}" for k, v in head.items()]
    # Anyone may WATCH; only this machine may publish (the ffmpeg transcode) or use the API / metrics.
    lines.append(_scalar("authInternalUsers") + ":")
    lines += ['  - {"user": "any", "pass": "", "ips": [], "permissions": [{"action": "read"}]}',
              '  - {"user": "any", "pass": "", "ips": ["127.0.0.1", "::1"], "permissions": '
              '[{"action": "publish"}, {"action": "api"}, {"action": "metrics"}]}']
    lines.append(_scalar("paths") + ":" + ("" if paths else " {}"))
    lines += _yaml(paths, 1)
    return "\n".join(lines) + "\n"


def write_config(settings: Settings, text: str) -> Path:
    path = Path(settings.relay_config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    try:
        path.chmod(0o600)                    # owner only; a no-op on Windows
    except OSError:
        pass
    return path
