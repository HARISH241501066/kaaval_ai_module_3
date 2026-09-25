"""
Sentinel Gujarat — Module 3 (Kaaval AI) Federation Backend.
Main FastAPI Application Entrypoint.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.core.config import get_settings
from app.db.database import init_db, SessionLocal
from app.db.seed import seed_initial_topology
from app.adapters.registry import get_adapter_registry
from app.api.health import router as health_router
from app.api.systems import router as systems_router
from app.api.ingestion import router as ingestion_router
from app.api.events import router as events_router
from app.api.provenance import router as provenance_router
from app.api.grid import router as grid_router, federation_router as broker_router
from app.live.redact import Redactor, install_log_redaction
from app.relay.manager import get_relay_manager
from app.services.grid_scheduler import GridScheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sentinel.m3.main")
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager:
    1. Initialize database schema
    2. Seed initial Gujarat CCTV topology
    3. Initialize and verify adapter registry
    """
    logger.info("Initializing Sentinel Module 3 Federation Backend...")
    init_db()

    # Seed demo topology
    with SessionLocal() as db:
        seed_initial_topology(db)

    # Initialize adapter registry
    registry = get_adapter_registry()
    logger.info(f"Loaded adapters: {[a['name'] for a in registry.list_adapters()]}")

    # The grid login must never reach a log line, from any logger (uvicorn, httpx, our own).
    manager = get_relay_manager()
    if manager.adapter.grid.configured:
        install_log_redaction(Redactor(*manager.adapter.grid.secrets()))
    scheduler = None
    if manager.adapter.grid.configured and settings.grid_sync_interval_s > 0:
        scheduler = GridScheduler(manager, settings)
        scheduler.start()
        logger.info("Sentinel Grid catalogue sync every %ss", settings.grid_sync_interval_s)

    logger.info("Sentinel Module 3 Backend ready. Swagger docs available at /docs")
    yield
    if scheduler:
        scheduler.stop()
    logger.info("Shutting down Sentinel Module 3 Backend.")


app = FastAPI(
    title="Sentinel Gujarat — Module 3 Federation API",
    description="""
# Sentinel Gujarat CCTV Hackathon — Module 3

### Federation, Middleware & Kaaval AI Integration Layer

Module 3 abstracts physical cameras, VMS systems, Kaaval RDK-X5 edge processing nodes, 
and central GPU servers into unified Sentinel-level interfaces.

---
### Key Capabilities Built in this Vertical Slice:
1. **Canonical Sentinel Event Envelope (C4)**: Contractual normalized schema for Module 4 (Intelligence).
2. **C4 Ingestion Pipeline**: Ingests Kaaval edge/server batches, verifies HMAC-SHA256 signatures, normalizes payloads, enriches AI provenance, and deduplicates.
3. **Multi-Dimension Health Monitoring (C6)**: Tracks Camera, Processing Node, AI Runtime, and Event Pipe states independently.
4. **Topology & Failure Impact**: Maps Camera -> Processing Node -> AI Model. Answers which cameras fail if a node goes down.
5. **C3 Stream Source Broker**: Exposes internal relay stream endpoints without exposing upstream vendor credentials.
6. **Live AI Provenance**: Traces complete AI lineage from camera to inference metrics and model versions.
7. **Mock Module 4 Delivery**: Verifies real-time event publishing downstream.
    """,
    version=settings.version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for hackathon evaluator frontends & testing tools
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health_router)
app.include_router(systems_router)
app.include_router(ingestion_router)
app.include_router(events_router)
app.include_router(provenance_router)
app.include_router(grid_router)
app.include_router(broker_router)


@app.get("/", include_in_schema=False)
def root():
    """Redirect root to interactive Swagger UI documentation."""
    return RedirectResponse(url="/docs")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
