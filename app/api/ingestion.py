"""
Event Ingestion Router (C4).
Provides:
1. POST /mock/kaaval/events — Compatible with sentinel_hwsim.push with HMAC verification
2. POST /api/v1/events/ingest — Generic single-event ingestion endpoint
"""
from __future__ import annotations

import json
import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, Header, Request, HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.database import get_db
from app.schemas.event import (
    IngestResult,
    BatchIngestResult,
    GenericEventIngestRequest,
)
from app.services.pipeline import get_pipeline, verify_kaaval_hmac

logger = logging.getLogger("sentinel.m3.ingest")
router = APIRouter(tags=["C4 Event Ingestion"])


@router.post("/mock/kaaval/events", response_model=BatchIngestResult)
@router.post("/federation/ingest/{node_id}", response_model=BatchIngestResult)
async def ingest_kaaval_push_events(
    request: Request,
    node_id: Optional[str] = None,
    db: Session = Depends(get_db),
    x_kaaval_node: Optional[str] = Header(None, alias="X-Kaaval-Node"),
    x_kaaval_timestamp: Optional[str] = Header(None, alias="X-Kaaval-Timestamp"),
    x_kaaval_signature: Optional[str] = Header(None, alias="X-Kaaval-Signature"),
):
    """
    Primary ingestion endpoint for Kaaval edge & server push dispatches.
    Directly compatible with sentinel_hwsim.push.
    Validates HMAC-SHA256 signature when headers are provided.
    """
    settings = get_settings()
    raw_body = await request.body()

    # 1. HMAC Verification (if signature provided)
    if x_kaaval_signature and x_kaaval_timestamp:
        valid, reason = verify_kaaval_hmac(
            secret=settings.kaaval_hmac_secret,
            timestamp_str=x_kaaval_timestamp,
            raw_body=raw_body,
            signature=x_kaaval_signature,
            max_skew_s=settings.kaaval_hmac_max_skew_s,
        )
        if not valid:
            logger.warning(f"HMAC authentication failed for node {x_kaaval_node}: {reason}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"HMAC authentication failed: {reason}",
            )

    # 2. Parse JSON payload
    try:
        data = json.loads(raw_body.decode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {e}")

    resolved_node_id = data.get("node") or node_id or x_kaaval_node or "KAAVAL-NODE-DEFAULT"
    batch = data.get("batch") or []

    if not isinstance(batch, list):
        raise HTTPException(status_code=400, detail="'batch' must be a list of items")

    # 3. Process batch through C4 pipeline
    pipeline = get_pipeline()
    result = await pipeline.ingest_kaaval_batch(db=db, node_id=resolved_node_id, batch_items=batch)
    return result


@router.post("/api/v1/events/ingest", response_model=IngestResult)
async def ingest_generic_event(
    request: GenericEventIngestRequest,
    db: Session = Depends(get_db),
):
    """
    Generic single-event ingestion endpoint.
    Accepts arbitrary vendor or test payloads with provider tag.
    """
    pipeline = get_pipeline()
    return await pipeline.ingest_single_event(
        db=db,
        raw_payload=request.payload,
        provider=request.provider,
        adapter_name=request.adapter,
    )
