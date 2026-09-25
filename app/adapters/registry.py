"""
Adapter Registry (C1).
Maintains active adapter instances for protocol and vendor resolution.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from app.adapters.base import BaseAdapter
from app.adapters.kaaval_edge import KaavalEdgeAdapter
from app.adapters.kaaval_server import KaavalServerAdapter
from app.adapters.live_detector import LiveDetectorAdapter
from app.adapters.sentinel_grid import SentinelGridAdapter

logger = logging.getLogger("sentinel.m3.adapters")


class AdapterRegistry:
    def __init__(self):
        self._adapters: Dict[str, BaseAdapter] = {}
        self._default_adapters_registered = False

    def register(self, adapter: BaseAdapter) -> None:
        self._adapters[adapter.name.upper()] = adapter
        self._adapters[adapter.protocol.upper()] = adapter
        logger.info(f"Registered adapter: {adapter.name} (Protocol: {adapter.protocol})")

    def get(self, key: str) -> Optional[BaseAdapter]:
        self._ensure_defaults()
        return self._adapters.get(key.upper())

    def list_adapters(self) -> List[Dict[str, str]]:
        self._ensure_defaults()
        unique = {}
        for adapter in self._adapters.values():
            unique[adapter.name] = {"name": adapter.name, "protocol": adapter.protocol}
        return list(unique.values())

    def _ensure_defaults(self) -> None:
        if not self._default_adapters_registered:
            self.register(KaavalEdgeAdapter())
            self.register(KaavalServerAdapter())
            self.register(LiveDetectorAdapter())
            self.register(SentinelGridAdapter())
            self._default_adapters_registered = True


# Global registry singleton
adapter_registry = AdapterRegistry()


def get_adapter_registry() -> AdapterRegistry:
    return adapter_registry
