"""
Check which Sentinel Grid cameras you can reach, over which protocol, and what codec they use.
Probes cameras ONE AT A TIME (each connection costs the grid a stream copy).

    set SENTINEL_GRID_EMAIL / SENTINEL_GRID_PASSWORD in your environment or .env, then:
    python check_grid.py                 # cam01..cam30
    python check_grid.py cam04 cam17     # specific cameras
"""
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

RTSP_HOST = "103.250.160.189:8554"
HLS_BASE = "https://cctv.corp8.cloud"


def load_env(path: str = ".env") -> None:
    """Minimal .env loader (no extra dependency). Existing env vars win."""
    p = Path(path)
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def rtsp_url(cam: str) -> str:
    email = quote(os.environ["SENTINEL_GRID_EMAIL"], safe="")      # @ -> %40
    password = quote(os.environ["SENTINEL_GRID_PASSWORD"], safe="")  # handles @ : / # ? %
    return f"rtsp://{email}:{password}@{RTSP_HOST}/stream/{cam}"


def probe(url: str, rtsp: bool) -> dict | None:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=codec_name,width,height,avg_frame_rate", "-of", "json"]
    if rtsp:
        cmd += ["-rtsp_transport", "tcp", "-timeout", "5000000"]  # 5 s, microseconds
    try:
        out = subprocess.run(cmd + [url], capture_output=True, text=True, timeout=15)
    except subprocess.TimeoutExpired:
        return None
    if out.returncode != 0:
        return {"error": out.stderr.strip().splitlines()[-1][:120] if out.stderr.strip() else "failed"}
    streams = json.loads(out.stdout).get("streams", [])
    return streams[0] if streams else {"error": "no video stream"}


def fmt(r: dict | None) -> str:
    if r is None:
        return "TIMEOUT"
    if "error" in r:
        return "FAIL (" + r["error"].replace(os.environ.get("SENTINEL_GRID_PASSWORD", "\0"), "***") + ")"
    return f"{r['codec_name']:<5} {r['width']}x{r['height']}"


def main() -> None:
    load_env()
    if not os.environ.get("SENTINEL_GRID_EMAIL") or not os.environ.get("SENTINEL_GRID_PASSWORD"):
        sys.exit("Set SENTINEL_GRID_EMAIL and SENTINEL_GRID_PASSWORD (raw email, with @)")
    cams = sys.argv[1:] or [f"cam{i:02d}" for i in range(1, 31)]
    print(f"{'camera':<7} {'RTSP (tcp)':<28} HLS")
    ok = 0
    for cam in cams:
        r = probe(rtsp_url(cam), rtsp=True)
        h = probe(f"{HLS_BASE}/{cam}/index.m3u8", rtsp=False)
        ok += r is not None and "error" not in r
        print(f"{cam:<7} {fmt(r):<28} {fmt(h)}")
    print(f"\nRTSP reachable: {ok}/{len(cams)}")
    if ok == 0:
        print("-> Port 8554 is probably blocked on this network. Try a mobile hotspot, or rely on HLS.")


if __name__ == "__main__":
    main()
