"""
M3 application settings.
All values can be overridden through environment variables or a .env file.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- database ---
    database_url: str = "sqlite:///./sentinel_m3.db"

    # --- security / auth ---
    # Secret used to verify HMAC-SHA256 signatures from push.py / real Kaaval nodes.
    # Must match the --secret argument passed to sentinel_hwsim.push.
    kaaval_hmac_secret: str = "dev-secret"

    # Maximum allowed clock skew (seconds) for HMAC timestamp check
    kaaval_hmac_max_skew_s: int = 300

    # --- deduplication ---
    # Time window (seconds) within which duplicate events are suppressed
    dedup_window_s: int = 3600

    # --- logging ---
    log_level: str = "INFO"

    # --- project meta ---
    project_name: str = "Sentinel Gujarat — Module 3 Federation"
    version: str = "0.1.0"

    # --- mock M4 ---
    # 'log'  → print to console (prototype default)
    # 'http' → POST to m4_endpoint_url
    m4_delivery_mode: str = "log"
    m4_endpoint_url: str = "http://localhost:9000/events"

    # --- Sentinel Camera Grid (live CCTV feeds) ---
    # Credentials come only from the environment / .env and are never returned by the API or logged.
    sentinel_grid_email: str = ""
    sentinel_grid_password: SecretStr = SecretStr("")
    sentinel_grid_host: str = "103.250.160.189"
    sentinel_grid_rtsp_port: int = 8554
    sentinel_grid_web_base: str = "https://cctv.corp8.cloud"

    sentinel_grid_whep_port: int = 8889
    grid_sync_interval_s: int = 900   # C2 re-sync period; 0 = only when POST /api/v1/grid/sync is called

    # --- stream relay (C3): our MediaMTX, the only thing browsers and Module 2 ever talk to ---
    relay_enabled: bool = True
    relay_bin: str = "relay/bin/mediamtx.exe"
    relay_config_path: str = "relay/generated/mediamtx.yml"      # holds grid credentials: git-ignored
    relay_api_url: str = "http://127.0.0.1:9997"
    relay_public_host: str = "localhost"                          # host the BROWSER uses to reach the relay
    relay_rtsp_port: int = 8654
    relay_hls_port: int = 8888
    relay_webrtc_port: int = 8889
    relay_allow_origins: str = "http://127.0.0.1:8765,http://localhost:8765"   # who may call the relay from a browser
    relay_idle_close_s: int = 60      # upstream is closed this long after the last viewer leaves
    relay_probe_timeout_s: int = 12   # how long a source gets to start before we try the next protocol
    ffmpeg_path: str = ""             # empty = look on PATH

    # --- live detection worker (python -m app.live.worker) ---
    live_cameras: str = ""            # comma separated grid ids; empty = first live_max_cameras of the catalogue
    live_max_cameras: int = 3         # every open camera costs the grid one stream copy
    live_sample_fps: float = 4.0      # detections per second of video, on the stream's own timestamps
    live_min_confidence: float = 0.45
    live_node_id: str = "LIVE-DETECTOR-001"
    m3_ingest_base_url: str = "http://localhost:8000"
    live_model_dir: str = "models"


@lru_cache
def get_settings() -> Settings:
    return Settings()
