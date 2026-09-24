"""
Pydantic schemas for Integration Health & Connectivity Monitoring (C6).
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, Any, List
from pydantic import BaseModel, Field


class ComponentHealth(BaseModel):
    entity_id: str = Field(..., example="RDK-X5-001")
    entity_type: str = Field(..., example="PROCESSING_NODE")  # CAMERA, PROCESSING_NODE, AI_RUNTIME, EVENT_PIPE
    status: str = Field(..., example="ONLINE")
    last_seen: datetime
    details: Dict[str, Any] = Field(default_factory=dict)


class HealthSummaryResponse(BaseModel):
    status: str = Field(..., example="HEALTHY")
    timestamp: datetime
    version: str
    database: str = Field(..., example="CONNECTED")
    summary: Dict[str, int] = Field(..., example={
        "cameras_total": 4,
        "cameras_online": 4,
        "nodes_total": 3,
        "nodes_online": 3,
        "ai_runtimes_running": 3,
        "event_pipes_up": 3
    })
    components: List[ComponentHealth]
