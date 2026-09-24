"""
Simulation clock.

All simulated physics (sensor drift, PTZ travel, event arrivals) and all
simulated latencies run on this clock. `speed` > 1 compresses time so tests
that need "60 seconds of device behaviour" finish in well under a second.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone


class SimClock:
    def __init__(self, speed: float = 1.0, start: datetime | None = None):
        if speed <= 0:
            raise ValueError("speed must be > 0")
        self.speed = speed
        self._real0 = time.monotonic()
        self._wall0 = start or datetime.now(timezone.utc)

    def now(self) -> float:
        """Simulated seconds since the clock started."""
        return (time.monotonic() - self._real0) * self.speed

    def wall(self) -> datetime:
        """Simulated UTC wall-clock time."""
        return self._wall0 + timedelta(seconds=self.now())

    async def sleep(self, sim_seconds: float) -> None:
        if sim_seconds > 0:
            await asyncio.sleep(sim_seconds / self.speed)
