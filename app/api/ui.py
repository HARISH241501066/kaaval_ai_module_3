"""
KAAVAL AI console — the browser surface for the local helmet detector.

    GET /ui/ai            the page
    GET /ui/ai/app.js     module
    GET /ui/ai/styles.css tokens

Static, build-step-free files served by M3 itself (the same approach sentinel-hwsim's
console uses: three files, no bundler, no framework). The page calls only this app's own
API — /ai/helmet/* — so nothing here needs CORS, and the CSP forbids anything else.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter(prefix="/ui/ai", tags=["KAAVAL AI — Console (HTML)"], include_in_schema=False)

STATIC_DIR = Path(__file__).resolve().parent.parent / "web" / "static" / "ai"

# Fixed allow-list: these routes can only ever serve these three files.
ASSETS: dict[str, tuple[str, str]] = {
    "index.html": ("index.html", "text/html; charset=utf-8"),
    "app.js": ("app.js", "text/javascript; charset=utf-8"),
    "styles.css": ("styles.css", "text/css; charset=utf-8"),
}

# No inline script/style, no remote anything: the page ships as three local files.
CSP = (
    "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; "
    "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)


def _serve(asset: str) -> FileResponse:
    filename, content_type = ASSETS[asset]
    path = STATIC_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"console asset missing: {path}")
    return FileResponse(
        path,
        media_type=content_type,
        headers={"Content-Security-Policy": CSP, "X-Content-Type-Options": "nosniff",
                 "Referrer-Policy": "no-referrer", "Cache-Control": "no-cache"},
    )


@router.get("", summary="KAAVAL AI helmet-detection console")
def console() -> FileResponse:
    """Interactive page for the helmet detector: pick a camera, start a job, watch the live
    counters, and see every confirmed violation with its evidence frame."""
    return _serve("index.html")


@router.get("/{asset}", summary="Console static asset")
def asset(asset: str) -> FileResponse:
    if asset not in ASSETS:
        raise HTTPException(status_code=404, detail=f"no such console asset: {asset}")
    return _serve(asset)
