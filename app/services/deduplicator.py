"""
Event Deduplication Service.
Deduplicates incoming events using provider + source_event_id within a configurable time window.
Uses in-memory cache for speed, backed by database repository checks.
"""
from __future__ import annotations

import time
import logging
from typing import Tuple, Optional, Dict
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.repositories import EventRepository

logger = logging.getLogger("sentinel.m3.dedup")


class EventDeduplicator:
    def __init__(self, window_s: Optional[int] = None):
        self.settings = get_settings()
        self.window_s = window_s or self.settings.dedup_window_s
        # In-memory dedup cache: key -> (timestamp, canonical_event_id)
        self._cache: Dict[str, Tuple[float, str]] = {}

    def _make_key(self, provider: str, source_event_id: str) -> str:
        return f"{provider.upper()}:{source_event_id.strip()}"

    def _prune(self) -> None:
        """Prune cache entries older than window_s."""
        now = time.time()
        expired = [k for k, (ts, _) in self._cache.items() if now - ts > self.window_s]
        for k in expired:
            del self._cache[k]

    def check_and_record(
        self,
        db: Session,
        provider: str,
        source_event_id: str,
        canonical_event_id: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if an event is a duplicate.
        Returns:
            (is_duplicate, existing_canonical_event_id)
        """
        self._prune()
        key = self._make_key(provider, source_event_id)
        now = time.time()

        # 1. Fast in-memory cache check
        if key in self._cache:
            ts, existing_id = self._cache[key]
            if now - ts <= self.window_s:
                logger.info(f"Duplicate event detected in-memory for {key} (existing: {existing_id})")
                return True, existing_id

        # 2. Database check for persistence across restarts
        repo = EventRepository(db)
        existing_record = repo.find_duplicate(provider, source_event_id, window_s=self.window_s)
        if existing_record:
            self._cache[key] = (now, existing_record.event_id)
            logger.info(f"Duplicate event detected in DB for {key} (existing: {existing_record.event_id})")
            return True, existing_record.event_id

        # 3. Not a duplicate: record it in memory
        self._cache[key] = (now, canonical_event_id)
        return False, None


# Global deduplicator instance
deduplicator = EventDeduplicator()


def get_deduplicator() -> EventDeduplicator:
    return deduplicator
