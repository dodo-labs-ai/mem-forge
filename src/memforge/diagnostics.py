"""Bounded diagnostic coordinates shared by producers and event validation."""

import re
from collections.abc import Iterable

_DIAGNOSTIC_PATH = re.compile(r"[a-zA-Z0-9_.$\[\]:-]{1,255}")


def is_diagnostic_path(value: str) -> bool:
    return _DIAGNOSTIC_PATH.fullmatch(value) is not None


def diagnostic_path(parts: Iterable[str | int]) -> str:
    # Unrepresentable segments are opaque: do not log portions of arbitrary keys
    # or Pydantic branch descriptions. Preserve ordinary schema coordinates.
    segments = [str(part) if is_diagnostic_path(str(part)) else "_" for part in parts]
    path = ".".join(segments) or "$"
    return path if len(path) <= 255 else path[:252] + "..."
