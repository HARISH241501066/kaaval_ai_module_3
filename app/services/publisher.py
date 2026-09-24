"""
Event Publisher and Mock Module 4 Consumer.
Module 4 (Intelligence / Events) receives Canonical Sentinel Event Envelopes.
"""
from __future__ import annotations

import json
import logging
from collections import deque
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import httpx

from app.core.config import get_settings
from app.schemas.event import CanonicalEventEnvelope

logger = logging.getLogger("sentinel.m3.publisher")


class MockM4Consumer:
    """
    Mock Module 4 Ingestion Consumer.
    Buffers delivered Canonical Sentinel Event Envelopes in memory
    so they can be inspected in tests and via Swagger UI (/api/v1/mock-m4/events).
    """
    def __init__(self, max_buffer_size: int = 500):
        self._buffer: deque = deque(maxlen=max_buffer_size)

    def receive(self, event: CanonicalEventEnvelope) -> None:
        event_dict = event.model_dump(mode="json")
        self._buffer.append({
            "received_at": datetime.now(timezone.utc).isoformat(),
            "event": event_dict,
        })
        logger.info(
            f"[MOCK M4 CONSUMER] Successfully received canonical event {event.event_id} "
            f"type={event.event_type} cam={event.camera_id} conf={event.confidence}"
        )

    def get_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        items = list(self._buffer)
        items.reverse()
        return items[:limit]

    def clear(self) -> None:
        self._buffer.clear()


# Global Mock M4 consumer singleton
mock_m4_consumer = MockM4Consumer()


class EventPublisher:
    """
    Publishes Canonical Sentinel Event Envelopes to downstream consumers (Module 4).
    """
    def __init__(self, m4_consumer: Optional[MockM4Consumer] = None):
        self.settings = get_settings()
        self.mock_consumer = m4_consumer or mock_m4_consumer

    async def publish(self, envelope: CanonicalEventEnvelope) -> bool:
        mode = self.settings.m4_delivery_mode.lower()
        payload = envelope.model_dump(mode="json")

        # Prototype standard: deliver to Mock M4 consumer
        self.mock_consumer.receive(envelope)

        if mode == "http" and self.settings.m4_endpoint_url:
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    resp = await client.post(self.settings.m4_endpoint_url, json=payload)
                    if 200 <= resp.status_code < 300:
                        logger.info(f"Published event {envelope.event_id} to M4 HTTP endpoint")
                        return True
                    else:
                        logger.warning(f"M4 HTTP endpoint returned status {resp.status_code}")
                        return False
            except Exception as e:
                logger.warning(f"Could not forward event {envelope.event_id} to M4 HTTP endpoint: {e}")
                return False

        logger.info(f"Published event {envelope.event_id} to downstream Sentinel Module 4 (mode: {mode})")
        return True


# Global publisher instance
event_publisher = EventPublisher()


def get_publisher() -> EventPublisher:
    return event_publisher


def get_mock_m4_consumer() -> MockM4Consumer:
    return mock_m4_consumer
