"""
Robust reader for one live camera stream.

Built for how the grid's feeds behave:
* Timing uses the stream's own presentation timestamps (PTS), never arrival time, and never
  CAP_PROP_FPS (unreliable). Sampling "4 frames per second" means every 0.25 s of *video*.
* A buffered GOP replays on connect, so the first second arrives faster than real time. PTS
  sampling handles that: frames are still picked per 0.25 s of video, whatever the arrival speed.
* Feeds loop with a hard scene cut. PTS jumps backwards; that is reported as a `discontinuity`
  (the tracker resets) and is NEVER treated as a failure.
* Decoder warnings on join ("Could not find ref with POC") are normal and never fatal. Only a
  failed open, or a read that times out, counts as a disconnect.
* On disconnect: reconnect with exponential backoff (2 s doubling to a 30 s cap, +/-20 % jitter),
  never a tight loop. Backoff resets only after the stream proved healthy.
* TCP only, 5 s open/read timeouts.

The source is either an rtsp:// URL or a local video file (used by tests, where reaching the end
of the file behaves like a feed looping).
"""
from __future__ import annotations

import logging
import os
import random
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.live.redact import redact_url

log = logging.getLogger("sentinel.m3.reader")

# rtsp_transport;tcp  -> TCP only (UDP is blocked on many networks and drops packets)
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

BACKOFF_START_S, BACKOFF_CAP_S = 2.0, 30.0
OPEN_TIMEOUT_MS = READ_TIMEOUT_MS = 5000
HEALTHY_AFTER_S = 5.0            # seconds of video read before we trust the connection again
LOOP_JUMP_S = 0.5                # a backwards PTS jump larger than this is a loop / restart


@dataclass
class Frame:
    image: np.ndarray
    pts_s: float
    discontinuity: bool          # first frame after a loop point or a reconnect


@dataclass
class ReaderStats:
    opens: int = 0
    reconnects: int = 0
    loops: int = 0
    frames_seen: int = 0
    frames_sampled: int = 0
    last_error: str = ""


class StreamReader:
    def __init__(self, name: str, source: str, sample_fps: float, stop: threading.Event):
        self.name, self._source, self._interval = name, source, 1.0 / max(0.1, sample_fps)
        self._stop = stop
        self.stats = ReaderStats()
        self._is_file = "://" not in source and Path(source).exists()

    def frames(self) -> Iterator[Frame]:
        delay = BACKOFF_START_S
        while not self._stop.is_set():
            cap = self._open()
            if cap is None:
                delay = self._backoff(delay, "could not open the stream")
                continue
            healthy = False
            try:
                for frame in self._read(cap):
                    if not healthy and frame.pts_s >= HEALTHY_AFTER_S:
                        healthy, delay = True, BACKOFF_START_S       # proven healthy: forgive earlier failures
                    yield frame
                    if self._stop.is_set():
                        return
            finally:
                cap.release()
            if self._is_file:
                self.stats.loops += 1                                 # end of a local file == the feed looping
                continue
            delay = self._backoff(delay, "stream ended or timed out" if healthy else "stream failed right after opening")

    # ------------------------------------------------------------------ internals
    def _open(self) -> cv2.VideoCapture | None:
        self.stats.opens += 1
        cap = cv2.VideoCapture(self._source, cv2.CAP_FFMPEG,
                               [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, OPEN_TIMEOUT_MS,
                                cv2.CAP_PROP_READ_TIMEOUT_MSEC, READ_TIMEOUT_MS])
        if not cap.isOpened():
            cap.release()
            return None
        log.info("%s: connected to %s", self.name, redact_url(self._source) if not self._is_file else self._source)
        return cap

    def _read(self, cap: cv2.VideoCapture) -> Iterator[Frame]:
        last_pts, next_sample = -1.0, 0.0
        index, discontinuity, warned = 0, self.stats.opens > 1, False
        while not self._stop.is_set():
            if not cap.grab():                         # decode without the colour conversion: cheap for skipped frames
                return
            self.stats.frames_seen += 1
            pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if pts <= 0 and index > 0 and last_pts <= 0:
                # some builds report no timestamps; count frames at a nominal rate instead (still not arrival time)
                if not warned:
                    log.warning("%s: stream reports no presentation timestamps; counting frames at 25 fps", self.name)
                    warned = True
                pts = index / 25.0
            index += 1
            if last_pts >= 0 and pts < last_pts - LOOP_JUMP_S:
                self.stats.loops += 1
                log.info("%s: loop point (timestamp %.1fs -> %.1fs): resetting trackers, health unaffected", self.name, last_pts, pts)
                discontinuity, next_sample = True, pts
            last_pts = pts
            if pts < next_sample:
                continue
            ok, image = cap.retrieve()
            if not ok:
                continue                               # a bad frame is a decoder warning, not a disconnect
            next_sample = pts + self._interval
            self.stats.frames_sampled += 1
            yield Frame(image, pts, discontinuity)
            discontinuity = False

    def _backoff(self, delay: float, why: str) -> float:
        self.stats.reconnects += 1
        self.stats.last_error = why
        wait = delay * random.uniform(0.8, 1.2)
        log.warning("%s: %s; reconnecting in %.1fs", self.name, why, wait)
        self._stop.wait(wait)
        return min(BACKOFF_CAP_S, delay * 2)
