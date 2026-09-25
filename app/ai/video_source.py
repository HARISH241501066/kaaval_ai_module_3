"""
Video sources for the KAAVAL AI layer.

The detector only ever sees frames:  VideoSource -> frames -> HelmetDetector.

LocalVideoSource reads a .mp4 road video (the prototype input, treated as a simulated
CCTV feed). RTSPVideoSource is the reserved slot for the camera architecture discussed
for M3 (ONVIF -> stream URI -> RTSP -> video processing -> YOLO): adding it later must
not change the detector or the Kaaval logic interfaces.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger("sentinel.m3.ai.video")

Frame = Tuple[np.ndarray, int, float]  # (BGR frame, frame_number, video timestamp in seconds)


class VideoError(RuntimeError):
    """Raised for missing / unreadable / empty / corrupted video inputs."""


class VideoSource(ABC):
    """Anything that can produce frames: an .mp4 today, an RTSP stream tomorrow."""

    @property
    @abstractmethod
    def describe(self) -> str:
        """Human-readable source description for logs and metrics."""

    @abstractmethod
    def frames(self) -> Iterator[Frame]:
        """Yield (frame, frame_number, timestamp_seconds) for every decoded frame."""

    def close(self) -> None:  # pragma: no cover - default no-op
        """Release underlying resources (safe to call twice)."""


class LocalVideoSource(VideoSource):
    """Local video file (.mp4 / any codec OpenCV can open)."""

    def __init__(self, path: str):
        self.path = str(path)
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: float = 0.0
        self._frame_count: int = 0

    @property
    def describe(self) -> str:
        return self.path

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def _open(self) -> None:
        p = Path(self.path)
        if not p.exists():
            raise VideoError(f"video file not found: '{self.path}'")
        if not p.is_file():
            raise VideoError(f"video path is not a file: '{self.path}'")
        if p.stat().st_size == 0:
            raise VideoError(f"video file is empty (0 bytes): '{self.path}'")
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            cap.release()
            raise VideoError(
                f"cannot open video '{self.path}': unsupported format or corrupted file "
                "(check the codec with ffprobe)"
            )
        self._cap = cap
        self._fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self._frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if self._fps <= 0:
            log.warning("%s: frame rate unreadable, assuming 25 fps for timestamps", self.path)
            self._fps = 25.0

    def frames(self) -> Iterator[Frame]:
        self._open()
        assert self._cap is not None
        index = 0
        try:
            while True:
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    break
                if frame.size == 0:
                    log.warning("%s: decoded an empty frame at #%d — stopping", self.path, index)
                    break
                yield frame, index, index / self._fps
                index += 1
        finally:
            self.close()
        if index == 0:
            raise VideoError(f"video '{self.path}' contains zero decodable frames")

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class RTSPVideoSource(VideoSource):
    """
    Reserved for the ONVIF -> RTSP camera path. Not implemented in this prototype:
    local MP4 is the input for now (no ONVIF discovery was in scope for this task).
    """

    def __init__(self, url: str):
        self.url = url

    @property
    def describe(self) -> str:
        return self.url

    def frames(self) -> Iterator[Frame]:
        raise NotImplementedError(
            "RTSPVideoSource is a reserved slot: resolve the stream URI via ONVIF/relay, "
            "then decode it here. The detector interface does not change."
        )


def open_source(path_or_url: str) -> VideoSource:
    """Factory: rtsp:// -> RTSPVideoSource, everything else -> LocalVideoSource."""
    if str(path_or_url).lower().startswith("rtsp://"):
        return RTSPVideoSource(str(path_or_url))
    return LocalVideoSource(str(path_or_url))
