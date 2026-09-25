"""
Sentinel Camera Grid client (C7 connection + C1 inventory for the SENTINEL_GRID system).

* The web side (catalogue, HLS) sits behind a login form: POST /auth/login with `email` and
  `password`, which sets a session cookie.
* RTSP (what the detector reads) uses Basic auth inside the URL, with the "@" of the email
  percent-encoded (%40), TCP only.

The catalogue schema is not fixed in advance, so parsing is deliberately tolerant.
"""
from __future__ import annotations

import logging
import random
import re
import socket
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

import httpx

from app.core.config import Settings

log = logging.getLogger("sentinel.m3.grid")

TIMEOUT_S = 5.0                      # every vendor call
RETRIES = 2                          # then give up (retry x2, exponential backoff + jitter)
_ID_OK = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_LIST_KEYS = ("cameras", "items", "data", "streams", "list", "results")
_ID_KEYS = ("id", "camera_id", "cam_id", "key", "slug", "stream", "path", "name")
_NAME_KEYS = ("name", "title", "label", "location", "description")


class GridError(Exception):
    """Raised with a message that is safe to print (never contains credentials)."""


@dataclass(frozen=True)
class GridCamera:
    id: str                          # grid id, e.g. "cam01": the external id in M3
    name: str
    codec: str | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    extra: dict[str, Any] = field(default_factory=dict, compare=False)


def parse_catalogue(data: Any) -> list[GridCamera]:
    """Accept a list, a dict wrapping a list, or a dict keyed by camera id."""
    if isinstance(data, dict):
        for key in _LIST_KEYS:
            if isinstance(data.get(key), (list, dict)):
                return parse_catalogue(data[key])
        entries: list[Any] = [{"id": k, **v} if isinstance(v, dict) else k for k, v in data.items()]
    elif isinstance(data, list):
        entries = data
    else:
        raise GridError(f"catalogue has an unexpected shape ({type(data).__name__})")

    cameras: list[GridCamera] = []
    for entry in entries:
        if isinstance(entry, str):
            entry = {"id": entry}
        if not isinstance(entry, dict):
            continue
        cam_id = next((str(entry[k]).strip() for k in _ID_KEYS if entry.get(k) not in (None, "")), "")
        if not _ID_OK.match(cam_id):
            log.warning("skipping catalogue entry with an unusable id")
            continue
        name = next((str(entry[k]) for k in _NAME_KEYS if entry.get(k)), cam_id)
        width, height = _int(entry.get("width")), _int(entry.get("height"))
        if not (width and height) and isinstance(entry.get("resolution"), str):
            m = re.match(r"^(\d+)\s*[x×]\s*(\d+)$", entry["resolution"].strip())
            width, height = (int(m[1]), int(m[2])) if m else (width, height)
        cameras.append(GridCamera(cam_id.lower(), name, str(entry["codec"]) if entry.get("codec") else None,
                                  width, height, _float(entry.get("fps"))))
    if not cameras:
        raise GridError("catalogue contained no usable cameras")
    return cameras


def _int(v: Any) -> int | None:
    try:
        return int(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _float(v: Any) -> float | None:
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


class GridClient:
    def __init__(self, settings: Settings, http: httpx.Client | None = None):
        self._s = settings
        self._http = http or httpx.Client(base_url=settings.sentinel_grid_web_base, timeout=TIMEOUT_S,
                                          follow_redirects=False)
        self._logged_in = False

    # ------------------------------------------------------------ credentials
    @property
    def configured(self) -> bool:
        return bool(self._s.sentinel_grid_email and self._s.sentinel_grid_password.get_secret_value())

    def secrets(self) -> list[str]:
        """Everything that must be scrubbed from logs (the log redactor is built from this)."""
        return [self._s.sentinel_grid_email, self._s.sentinel_grid_password.get_secret_value()]

    # -------------------------------------------------------------------- URLs
    def rtsp_url_public(self, cam_id: str) -> str:
        """Credential-free address: safe to store, log, return from an API."""
        return f"rtsp://{self._s.sentinel_grid_host}:{self._s.sentinel_grid_rtsp_port}/stream/{cam_id}"

    def rtsp_url(self, cam_id: str) -> str:
        """
        The ONLY place a credentialed URL is built. Use it in memory to open a stream, then drop it.
        Never log it, store it or return it (use rtsp_url_public for that).
        """
        if not self.configured:
            raise GridError("SENTINEL_GRID_EMAIL / SENTINEL_GRID_PASSWORD are not set")
        user = quote(self._s.sentinel_grid_email, safe="")           # "@" -> %40
        password = quote(self._s.sentinel_grid_password.get_secret_value(), safe="")
        return f"rtsp://{user}:{password}@{self._s.sentinel_grid_host}:{self._s.sentinel_grid_rtsp_port}/stream/{cam_id}"

    # ------------------------------------------------------- connection test
    def ping(self) -> None:
        """TCP reachability of the RTSP port (the port the detector needs)."""
        try:
            socket.create_connection((self._s.sentinel_grid_host, self._s.sentinel_grid_rtsp_port), TIMEOUT_S).close()
        except OSError as e:
            raise GridError(f"cannot reach {self._s.sentinel_grid_host}:{self._s.sentinel_grid_rtsp_port} ({e.__class__.__name__})") from None

    def login(self) -> None:
        if not self.configured:
            raise GridError("SENTINEL_GRID_EMAIL / SENTINEL_GRID_PASSWORD are not set in .env")
        r = self._request("POST", "/auth/login", data={"email": self._s.sentinel_grid_email,
                                                       "password": self._s.sentinel_grid_password.get_secret_value()})
        # success redirects away from the login page and sets a cookie; failure re-renders the form (200)
        if r.status_code not in (301, 302, 303, 307) or "/auth/login" in r.headers.get("location", ""):
            raise GridError("the grid rejected the login (check SENTINEL_GRID_EMAIL / SENTINEL_GRID_PASSWORD)")
        self._logged_in = True

    def fetch_catalogue(self) -> list[GridCamera]:
        if not self._logged_in:
            self.login()
        r = self._request("GET", "/cameras.json")
        if r.status_code in (301, 302, 303, 307) and "/auth/login" in r.headers.get("location", ""):
            self._logged_in = False          # session expired: log in once more, then try again
            self.login()
            r = self._request("GET", "/cameras.json")
        if r.status_code != 200:
            raise GridError(f"catalogue request failed (HTTP {r.status_code})")
        try:
            return parse_catalogue(r.json())
        except ValueError:
            raise GridError("catalogue is not valid JSON") from None

    def connection_test(self) -> list[GridCamera]:
        """C7 gate: ping + auth + first catalogue page. Returns the catalogue on success."""
        self.ping()
        cameras = self.fetch_catalogue()
        log.info("grid connection test passed: %d cameras in the catalogue", len(cameras))
        return cameras

    # ----------------------------------------------------------------- helpers
    def _request(self, method: str, path: str, **kw: Any) -> httpx.Response:
        last: Exception | None = None
        for attempt in range(RETRIES + 1):
            try:
                return self._http.request(method, path, **kw)
            except httpx.HTTPError as e:
                last = e
                if attempt < RETRIES:
                    time.sleep(min(30.0, 0.5 * 2 ** attempt) * random.uniform(0.8, 1.2))
        raise GridError(f"grid web endpoint unreachable ({last.__class__.__name__})") from None
