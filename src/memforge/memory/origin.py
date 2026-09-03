"""Derived origin classification for Memory provenance."""

from __future__ import annotations

from enum import Enum

from memforge.models import VIRTUAL_DOCUMENT_SOURCE_IDS


class MemoryOriginKind(str, Enum):
    """How a provenance source introduced knowledge into MemForge."""

    DIRECT_USER = "direct_user"
    CONFIGURED_SOURCE = "configured_source"
    MANAGED_CAPTURE = "managed_capture"


def classify_memory_origin(source_type: str) -> MemoryOriginKind:
    """Classify an existing provenance source type without storing parallel state."""

    if source_type in VIRTUAL_DOCUMENT_SOURCE_IDS:
        return MemoryOriginKind.DIRECT_USER
    if source_type == "agent_session":
        return MemoryOriginKind.MANAGED_CAPTURE
    return MemoryOriginKind.CONFIGURED_SOURCE


def references_source_row(source_id: str | None) -> bool:
    """Whether a provenance reference names a configured or managed Source row.

    Direct User document namespaces are provenance, not Source identities. All
    other non-null identifiers remain Source dependencies, even if unresolved.
    This classification does not establish current provenance or grant access.
    """

    return source_id is not None and source_id not in VIRTUAL_DOCUMENT_SOURCE_IDS
