"""
C2 on a schedule: keep the grid's catalogue in step with the registry while the API runs.

Each run: make sure the system is registered ACTIVE, sync the catalogue, probe a few cameras that have no
codec yet (one at a time, so the grid never sees a burst), and bring a running relay in line. A failure only
skips that run: the loop keeps its own pace and never spins.
"""
from __future__ import annotations

import logging
import threading

from app.core.config import Settings
from app.db.database import SessionLocal
from app.relay.manager import RelayManager
from app.services import grid_sync
from contract import DeviceError

log = logging.getLogger("sentinel.m3.grid.scheduler")
PROBES_PER_RUN = 3


def run_once(manager: RelayManager, settings: Settings) -> None:
    with SessionLocal() as db:
        grid_sync.register_system(db, manager.adapter)       # the connection test doubles as the grid's health check each cycle
        grid_sync.sync_inventory(db, manager.adapter, probe_limit=PROBES_PER_RUN)
        if settings.relay_enabled:
            manager.sync_paths(db)


class GridScheduler:
    def __init__(self, manager: RelayManager, settings: Settings):
        self._m, self._s = manager, settings
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="grid-sync", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        self._stop.wait(2)                                   # let the API finish starting
        while not self._stop.is_set():
            try:
                run_once(self._m, self._s)
            except DeviceError as e:
                log.warning("grid sync run skipped: %s", e)
            except Exception:
                log.exception("grid sync run failed")
            self._stop.wait(self._s.grid_sync_interval_s)
