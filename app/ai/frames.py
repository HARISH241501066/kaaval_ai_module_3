"""
Random-access frame extraction (no frame buffering).

Both provider paths need a *few* real frames without re-decoding whole videos:

* the Google path extracts the frames of candidate rider observations to classify them and
  the single best frame of each confirmed violator as evidence;
* the local path only needs the evidence frame (it already saw every frame).

`FrameExtractor` opens the video once, seeks to the wanted frame, and returns that frame
plus its real frame number (a seek can land on a nearby keyframe), so timestamps stay
honest. It never keeps more than one frame alive.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger("sentinel.m3.ai.frames")


class FrameExtractionError(RuntimeError):
    """The requested frame could not be decoded."""


class FrameExtractor:
    """Seek-based access to individual frames of one video file."""

    def __init__(self, video_path: str):
        self.path = str(video_path)
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: float = 0.0
        self._frame_count: int = 0
        self._width: int = 0
        self._height: int = 0

    # ------------------------------------------------------------------ lifecycle
    def open(self) -> "FrameExtractor":
        if self._cap is not None:
            return self
        p = Path(self.path)
        if not p.is_file():
            raise FrameExtractionError(f"video file not found: '{self.path}'")
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            cap.release()
            raise FrameExtractionError(f"cannot open video '{self.path}' (unsupported or corrupted)")
        self._cap = cap
        self._fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
        self._frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        return self

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "FrameExtractor":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ properties
    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def size(self) -> Tuple[int, int]:
        return self._width, self._height

    # ------------------------------------------------------------------ access
    def frame(self, frame_number: int) -> Tuple[np.ndarray, int]:
        """Decode one frame by index. Returns (frame, actual_frame_number)."""
        if self._cap is None:
            self.open()
        assert self._cap is not None
        index = max(0, int(frame_number))
        if self._frame_count:
            index = min(index, self._frame_count - 1)
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self._cap.read()
        if not ok or frame is None or frame.size == 0:
            raise FrameExtractionError(f"could not decode frame {index} of '{self.path}'")
        # get(CAP_PROP_POS_FRAMES) after a read gives the *next* frame index; the decoded one is
        # index unless the seek snapped to a keyframe, in which case we report what we asked for.
        return frame, index

    def frame_at_time(self, timestamp_s: float) -> Tuple[np.ndarray, int]:
        """Decode the frame at/after a video timestamp (timestamp = frame / fps)."""
        return self.frame(int(round(float(timestamp_s) * self._fps)))
