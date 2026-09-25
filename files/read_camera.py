"""
Robust live reader for one Sentinel Grid camera: the guide's OpenCV snippet,
plus everything its do's-and-don'ts section requires.

    python read_camera.py cam04            # RTSP over TCP
    python read_camera.py cam04 --hls      # HLS fallback
    python read_camera.py cam04 --show     # open a preview window (q to quit)
"""
import argparse
import os
import random
import time
from urllib.parse import quote

# MUST be set before cv2 opens any capture: force RTSP over TCP (UDP -> corrupt frames)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from check_grid import HLS_BASE, load_env, rtsp_url  # noqa: E402


def open_capture(cam: str, use_hls: bool) -> cv2.VideoCapture:
    url = f"{HLS_BASE}/{cam}/index.m3u8" if use_hls else rtsp_url(cam)
    return cv2.VideoCapture(url, cv2.CAP_FFMPEG)


def run(cam: str, use_hls: bool, show: bool) -> None:
    backoff = 2.0
    last_pts, last_small = None, None
    while True:
        cap = open_capture(cam, use_hls)
        if not cap.isOpened():
            wait = backoff * random.uniform(0.8, 1.2)
            print(f"[{cam}] connect failed, retry in {wait:.1f}s")
            time.sleep(wait)
            backoff = min(backoff * 2, 30.0)         # 2s -> 30s cap, never tight-loop
            continue
        print(f"[{cam}] connected via {'HLS' if use_hls else 'RTSP/TCP'}")
        backoff = 2.0
        frames = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                print(f"[{cam}] stream ended/dropped after {frames} frames, reconnecting")
                break
            frames += 1
            pts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)   # use THIS for timing, never CAP_PROP_FPS
            if last_pts is not None and pts_ms < last_pts:
                print(f"[{cam}] PTS went backwards ({last_pts:.0f} -> {pts_ms:.0f} ms): feed restarted")
            last_pts = pts_ms

            # loop-point / hard scene cut detection: reset trackers here in a real pipeline
            small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36))
            if last_small is not None and np.mean(cv2.absdiff(small, last_small)) > 40:
                print(f"[{cam}] scene cut at pts={pts_ms:.0f} ms (loop point?) -> reset state")
            last_small = small

            if frames % 100 == 0:
                h, w = frame.shape[:2]
                print(f"[{cam}] {frames} frames, {w}x{h}, pts={pts_ms / 1000:.1f}s")
            if show:
                cv2.imshow(cam, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cap.release()
                    return
        cap.release()                                 # always close: each open = one grid stream copy
        time.sleep(backoff)


if __name__ == "__main__":
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("cam")
    ap.add_argument("--hls", action="store_true")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    run(a.cam, a.hls, a.show)
