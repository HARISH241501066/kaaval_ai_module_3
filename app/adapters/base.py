"""
Base Adapter interface (C1).
All vendor, hardware, and AI adapters implement this uniform contract.
Adapters isolate vendor-specific quirks and guarantee that only canonical Sentinel
objects flow downstream.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from app.schemas.event import CanonicalEventEnvelope


class BaseAdapter(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Adapter unique identifier (e.g., 'KAAVAL_EDGE', 'KAAVAL_SERVER', 'ONVIF')."""
        pass

    @property
    @abstractmethod
    def protocol(self) -> str:
        """Integration protocol (e.g., 'KAAVAL_EDGE', 'ONVIF', 'RTSP')."""
        pass

    @abstractmethod
    def validate_payload(self, raw_payload: Dict[str, Any]) -> bool:
        """Validate raw incoming payload structure."""
        pass

    @abstractmethod
    def normalize(
        self,
        raw_payload: Dict[str, Any],
        camera_id: str,
        processing_node_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> CanonicalEventEnvelope:
        """
        Normalize vendor payload into the Canonical Sentinel Event Envelope.
        Must not leak raw unstandardized vendor structures downstream.
        """
        pass

    @abstractmethod
    def check_health(self, target_id: str) -> Dict[str, Any]:
        """Perform protocol/adapter health check for a given target entity."""
        pass
