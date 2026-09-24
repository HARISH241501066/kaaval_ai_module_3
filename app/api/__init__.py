from app.api.health import router as health_router
from app.api.systems import router as systems_router
from app.api.ingestion import router as ingestion_router
from app.api.events import router as events_router
from app.api.provenance import router as provenance_router

__all__ = [
    "health_router",
    "systems_router",
    "ingestion_router",
    "events_router",
    "provenance_router",
]
