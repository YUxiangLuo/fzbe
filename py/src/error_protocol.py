"""Stable error-tag formatting shared by all Python model entry points."""

from typing import Optional


KNOWN_ERROR_TAGS = ("[VALIDATION]", "[DATA]", "[MODEL]", "[TIMEOUT]")


def format_known_error(exc: Exception) -> Optional[str]:
    message = str(exc).strip()
    if isinstance(exc, TimeoutError):
        detail = message.removeprefix("[TIMEOUT]").strip()
        return f"[TIMEOUT] {detail}".rstrip()
    for tag in KNOWN_ERROR_TAGS:
        if message.startswith(tag):
            detail = message.removeprefix(tag).strip()
            return f"{tag} {detail}".rstrip()
    return None
