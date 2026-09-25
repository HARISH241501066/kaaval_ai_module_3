"""
Credential safety for the live grid integration.

Rule: the grid email and password must never appear in logs, exceptions, API responses,
URLs handed to consumers or files. The only place a credentialed URL exists is the one
function that builds it (GridClient.rtsp_url), and it is used in memory only.

`Redactor` removes the secrets (in every encoding they are likely to be logged in) from any
text. `install_log_redaction` applies it to every log record in the process, including
records from third-party libraries and formatted tracebacks.
"""
from __future__ import annotations

import logging
import re
import traceback
from urllib.parse import quote, quote_plus

MASK = "***"
_URL_USERINFO = re.compile(r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+.-]*)://[^/@\s]+@")


def redact_url(url: str) -> str:
    """rtsp://user:pass@host/x -> rtsp://***@host/x (also works when nothing is configured)."""
    return _URL_USERINFO.sub(lambda m: f"{m.group('scheme')}://{MASK}@", url)


class Redactor:
    def __init__(self, *secrets: str):
        self._needles: list[str] = []
        self.add(*secrets)

    def add(self, *secrets: str) -> None:
        for secret in secrets:
            if not secret or len(secret) < 3:      # too short to redact safely without wrecking normal text
                continue
            for variant in {secret, quote(secret, safe=""), quote_plus(secret), secret.replace("@", "%40")}:
                if variant not in self._needles:
                    self._needles.append(variant)
        self._needles.sort(key=len, reverse=True)   # longest first, so a needle never leaves a fragment

    def scrub(self, text: str) -> str:
        text = redact_url(text)
        for needle in self._needles:
            if needle in text:
                text = text.replace(needle, MASK)
        return text


_installed: Redactor | None = None
_original_factory = logging.getLogRecordFactory()


def install_log_redaction(redactor: Redactor) -> None:
    """Scrub every log record, whichever logger emits it. Safe to call again with more secrets."""
    global _installed
    if _installed is not None:
        _installed._needles = redactor._needles      # same process, new secrets: keep one live needle list
        return
    _installed = redactor

    def factory(*args, **kwargs):
        record = _original_factory(*args, **kwargs)
        record.msg = redactor.scrub(record.getMessage())
        record.args = None
        if record.exc_info and record.exc_info[0] is not None:
            record.exc_text = redactor.scrub("".join(traceback.format_exception(*record.exc_info)).rstrip("\n"))
        return record

    logging.setLogRecordFactory(factory)
