"""
A small circuit breaker, one per upstream system.

closed    -> calls pass; `threshold` consecutive failures open it
open      -> calls are refused for `reset_s` (so a dead grid is not hammered)
half-open -> one trial call is let through; success closes it, failure re-opens it
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable


class CircuitBreaker:
    def __init__(self, threshold: int = 5, reset_s: float = 30.0, clock: Callable[[], float] = time.monotonic):
        self.threshold, self.reset_s, self._clock = threshold, reset_s, clock
        self._failures = 0
        self._opened_at: float | None = None
        self._trial_out = False
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            if self._opened_at is None:
                return "closed"
            return "half-open" if self._clock() - self._opened_at >= self.reset_s else "open"

    def allow(self) -> bool:
        with self._lock:
            if self._opened_at is None:
                return True
            if self._clock() - self._opened_at < self.reset_s:
                return False
            if self._trial_out:                    # one trial at a time while half-open
                return False
            self._trial_out = True
            return True

    def success(self) -> None:
        with self._lock:
            self._failures, self._opened_at, self._trial_out = 0, None, False

    def failure(self) -> None:
        with self._lock:
            self._failures += 1
            self._trial_out = False
            if self._failures >= self.threshold or self._opened_at is not None:
                self._opened_at = self._clock()
