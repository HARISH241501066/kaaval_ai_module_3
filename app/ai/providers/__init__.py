"""
Provider factory + selection policy.

    AI_PROVIDER=google  -> GoogleVideoIntelligenceProvider (with an explicit, reported
                           fallback to LocalYOLOProvider when Google is not configured
                           and AI_ALLOW_LOCAL_FALLBACK=true)
    AI_PROVIDER=yolo    -> LocalYOLOProvider

Helmet status and ANPR are separate selections on purpose:

    HELMET_PROVIDER=yolo          the only helmet-capable backend here (Cloud VI has none)
    ANPR_PROVIDER=google_vision   Cloud Vision OCR on the evidence frame, one call per violation
    ANPR_PROVIDER=none            explicit "no OCR" with ANPR_STATUS=NOT_CONFIGURED

Every selection is reported with the *effective* backend and the reason for any fallback, so
a demo can never silently claim it used Google when it actually ran locally.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from app.ai.config import AIConfig, get_ai_config
from app.ai.providers.base import (
    AnprProvider,
    HelmetClassifier,
    ProviderNotConfigured,
    VideoAnalyticsProvider,
)
from app.ai.providers.google_video import GoogleVideoIntelligenceProvider
from app.ai.providers.google_vision import GoogleVisionAnprProvider, NoAnprProvider
from app.ai.providers.local_yolo import LocalYOLOClassifier, LocalYOLOProvider

log = logging.getLogger("sentinel.m3.ai.providers")

__all__ = [
    "LocalYOLOProvider",
    "LocalYOLOClassifier",
    "GoogleVideoIntelligenceProvider",
    "GoogleVisionAnprProvider",
    "NoAnprProvider",
    "ProviderSelection",
    "select_video_provider",
    "get_video_provider",
    "get_helmet_classifier",
    "get_anpr_provider",
    "provider_status",
]


@dataclass
class ProviderSelection:
    """The backend that WILL run, plus why it differs from the requested one (if it does)."""

    requested: str
    effective: str
    provider: VideoAnalyticsProvider
    fallback_reason: Optional[str] = None
    google_health: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requested": self.requested,
            "effective": self.effective,
            "fallback": bool(self.fallback_reason),
            "fallback_reason": self.fallback_reason,
            "capabilities": self.provider.capabilities(),
        }


def select_video_provider(
    config: Optional[AIConfig] = None,
    google_client: Any = None,
    detector: Any = None,
) -> ProviderSelection:
    """Resolve the provider, applying the documented fallback policy."""
    cfg = config or get_ai_config()
    requested = cfg.video_provider
    if requested == "yolo":
        return ProviderSelection(requested="yolo", effective="yolo",
                                 provider=LocalYOLOProvider(cfg, detector=detector))

    google = GoogleVideoIntelligenceProvider(cfg, client=google_client)
    health = google.health()
    if health.get("status") == "AVAILABLE":
        return ProviderSelection(requested=requested, effective="google",
                                 provider=google, google_health=health)

    reason = f"{health.get('status', 'ERROR')}: {health.get('error') or 'unavailable'}"
    if not cfg.allow_local_fallback:
        raise ProviderNotConfigured(
            f"AI_PROVIDER=google is not usable and AI_ALLOW_LOCAL_FALLBACK=false — {reason}"
        )
    log.warning("Google Video Intelligence unavailable (%s) — falling back to the local YOLO "
                "provider for this job; the metrics will say so.", reason)
    return ProviderSelection(requested=requested, effective="local_yolo",
                             provider=LocalYOLOProvider(cfg, detector=detector),
                             fallback_reason=reason, google_health=health)


def get_video_provider(config: Optional[AIConfig] = None, google_client: Any = None,
                       detector: Any = None) -> VideoAnalyticsProvider:
    return select_video_provider(config, google_client=google_client, detector=detector).provider


def get_helmet_classifier(config: Optional[AIConfig] = None, detector: Any = None) -> HelmetClassifier:
    """The helmet classifier used for tracked riders the analysis stage did not classify."""
    cfg = config or get_ai_config()
    if cfg.helmet_provider in ("yolo", "local", "local_yolo"):
        return LocalYOLOClassifier(cfg, detector=detector)
    raise ProviderNotConfigured(
        f"HELMET_PROVIDER='{cfg.helmet_provider}' is not supported: Cloud Video Intelligence "
        "detects and tracks people but does NOT classify helmet status. Use HELMET_PROVIDER=yolo "
        "(the local helmet model) as the second stage."
    )


def get_anpr_provider(config: Optional[AIConfig] = None, vision_client: Any = None) -> AnprProvider:
    cfg = config or get_ai_config()
    if not cfg.anpr_enabled or cfg.anpr_provider in ("none", "off", "disabled"):
        return NoAnprProvider("ANPR disabled (ANPR_ENABLED=false or ANPR_PROVIDER=none)")
    if cfg.anpr_provider == "google_vision":
        return GoogleVisionAnprProvider(cfg, client=vision_client)
    return NoAnprProvider(f"unsupported ANPR_PROVIDER='{cfg.anpr_provider}'")


def provider_status(
    config: Optional[AIConfig] = None,
    google_client: Any = None,
    vision_client: Any = None,
    detector: Any = None,
) -> Dict[str, Any]:
    """Independent health for every AI layer (never raises, never touches a camera)."""
    cfg = config or get_ai_config()
    google = GoogleVideoIntelligenceProvider(cfg, client=google_client)
    local = LocalYOLOProvider(cfg, detector=detector)
    classifier = LocalYOLOClassifier(cfg, detector=detector)
    anpr = get_anpr_provider(cfg, vision_client=vision_client)

    selection = {
        "requested": cfg.video_provider,
        "effective": "google" if (cfg.video_provider == "google" and google.health()["status"] == "AVAILABLE")
        else ("yolo" if cfg.video_provider == "yolo" else "local_yolo(fallback)"),
        "allow_local_fallback": cfg.allow_local_fallback,
    }
    return {
        "analysis": selection,
        "video_intelligence": google.health(),
        "local_yolo_provider": local.health(),
        "helmet_classifier": classifier.health(),
        "anpr": anpr.health(),
        "note": (
            "Cloud Video Intelligence = person/object detection + tracking (no helmet verdict). "
            "Helmet status = local YOLO helmet classifier. ANPR = generic Cloud Vision OCR."
        ),
    }
