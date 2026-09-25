"""
Schemas for the Sentinel Camera Grid endpoints. Nothing here can carry credentials: there is no field for a
username, password or upstream URL, only relay URLs and status.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class SourceStatus(BaseModel):
    protocol: Literal["RTSP", "WHEP", "HLS"]
    priority: int
    available: bool
    active: bool
    detail: Optional[str] = Field(None, description="Why the protocol is unavailable, if it is.")


class RelayUrls(BaseModel):
    whep: str = Field(..., description="WebRTC (WHEP) playback URL on OUR relay.")
    hls: str = Field(..., description="HLS playlist on OUR relay.")
    rtsp: str = Field(..., description="RTSP on OUR relay, for backend consumers.")


class StreamSourcesRead(BaseModel):
    camera_id: str = Field(..., example="cam04")
    canonical_id: str = Field(..., example="GRID-CAM04")
    profile: Literal["main", "web"]
    codec: Optional[str] = Field(None, description="Codec of THIS profile: web is always h264.")
    source_codec: Optional[str] = Field(None, description="Codec of the camera itself.")
    transcoded: bool = False
    resolution: Optional[str] = None
    urls: RelayUrls
    active_source: Optional[Literal["RTSP", "WHEP", "HLS"]] = Field(None, description="Grid protocol the relay is pulling over.")
    state: Literal["AVAILABLE", "UNKNOWN", "UNAVAILABLE", "RELAY_DOWN"]
    sources: List[SourceStatus] = []


class GridCameraRead(BaseModel):
    id: str = Field(..., example="cam04")
    canonical_id: str = Field(..., example="GRID-CAM04")
    name: str
    codec: Optional[str] = None
    resolution: Optional[str] = None
    fps: Optional[float] = None
    status: str = Field(..., description="ONLINE | OFFLINE (from the last probe)")
    mapping_status: Literal["ACTIVE", "INACTIVE"]
    probed: bool
    active_source: Optional[str] = None
    probe_error: Optional[str] = None


class GridSyncRead(BaseModel):
    catalogue: int
    created: int
    updated: int
    reactivated: int
    newly_inactive: List[str]
    probed: int
    probe_failed: List[str]
    relay: Dict[str, int] = {}
