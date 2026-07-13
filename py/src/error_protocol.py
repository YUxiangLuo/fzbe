"""Stable error-tag formatting shared by all Python model entry points."""

from typing import Optional


KNOWN_ERROR_TAGS = ("[VALIDATION]", "[DATA]")


def format_known_error(exc: Exception) -> Optional[str]:
    message = str(exc).strip()
    for tag in KNOWN_ERROR_TAGS:
        if message.startswith(tag):
            detail = message.removeprefix(tag).strip()
            return f"{tag} {detail}".rstrip()
    return None
