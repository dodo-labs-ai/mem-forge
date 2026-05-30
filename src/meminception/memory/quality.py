"""Deterministic quality checks for extracted memory candidates."""

from __future__ import annotations

import re
from dataclasses import dataclass

from meminception.models import RawMemory

__all__ = [
    "MemoryCandidateQuality",
    "classify_memory_candidate",
    "should_keep_memory",
]


@dataclass(frozen=True)
class MemoryCandidateQuality:
    """Keep/skip decision for a raw memory before persistence."""

    keep: bool
    skip_reason: str | None = None


def classify_memory_candidate(raw: RawMemory) -> MemoryCandidateQuality:
    """Classify whether a raw memory is useful enough to persist."""
    content = _normalize(raw.content)
    context = _normalize(raw.extraction_context or "")

    if not content:
        return MemoryCandidateQuality(keep=False, skip_reason="empty")
    if _is_self_referential(content):
        return MemoryCandidateQuality(keep=False, skip_reason="self_referential")
    if _is_reference_only(content, context):
        return MemoryCandidateQuality(keep=False, skip_reason="reference_only")
    if _is_metadata_only(content, context):
        return MemoryCandidateQuality(keep=False, skip_reason="metadata_only")
    if _is_open_question(content, context):
        return MemoryCandidateQuality(keep=False, skip_reason="open_question")

    return MemoryCandidateQuality(keep=True)


def should_keep_memory(raw: RawMemory) -> bool:
    """Return True when a raw memory should proceed to persistence."""
    return classify_memory_candidate(raw).keep


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def _is_self_referential(content: str) -> bool:
    """Drop candidates that narrate the memory system itself or cite internal memory ids.

    These arise when a generated agent-session summary describes its own tooling
    (context injection, warm-context loading) or embeds internal ``mem-...`` ids.
    They are never durable team knowledge, and "do not extract" prompt rules do
    not reliably suppress them, so this is enforced deterministically.
    """
    if re.search(r"\bmem-[0-9a-f]{6,}\b", content):
        return True
    patterns = [
        r"\bmeminception memories\b",
        r"\bmemories are loaded\b",
        r"\bloaded at session\s*start\b",
        r"\bused as warm context\b",
        r"\bas warm context\b",
        r"\brelevant facts were already present\b",
    ]
    return any(re.search(pattern, content) for pattern in patterns)


def _is_metadata_only(content: str, context: str) -> bool:
    """Detect document provenance/header facts masquerading as memories."""
    document_subject = bool(re.search(r"\b(acd|document|doc|page|wiki page|confluence page)\b", content))
    metadata_markers = [
        r"\bauthored by\b",
        r"\bauthor\b",
        r"\blast modified\b",
        r"\bdocument status\b",
        r"\brevision history\b",
        r"\breviewer(?:s)?\b",
        r"\bcreated by\b",
        r"\bupdated by\b",
    ]
    marker_count = sum(1 for pattern in metadata_markers if re.search(pattern, content))
    context_has_header = bool(
        re.search(r"\bauthor\s*:", context)
        or re.search(r"\bdocument status\b", context)
        or re.search(r"\blast modified\s*:", context)
    )

    return document_subject and (marker_count >= 2 or (marker_count >= 1 and context_has_header))


def _is_reference_only(content: str, context: str) -> bool:
    """Detect source link/reference rows that belong in provenance."""
    has_url = "http://" in content or "https://" in content
    context_is_link_list = bool(
        re.search(r"\blink to (concept|design|document|doc|source)\b", context)
        or re.search(r"\blink list\b", context)
    )
    content_is_link_sentence = bool(
        has_url
        and (
            re.search(r"\blinks? to\b.+\bat\s*:", content)
            or re.search(r"\breferences?\b.+\bat\s*:", content)
        )
    )
    content_describes_link = bool(
        has_url
        or re.search(r"\blinks? to\b", content)
        or re.search(r"\breferences?\b", content)
    )
    return content_is_link_sentence or (context_is_link_list and content_describes_link)


def _is_open_question(content: str, context: str) -> bool:
    """Detect unresolved discussion prompts that are not settled memories."""
    combined = f"{content} {context}"
    if re.search(r"\b(discuss whether|bear it in mind|to be discussed|open question)\b", combined):
        return True
    if "should be considered" in content and re.search(r"\b(discuss|discussion|consideration|whether)\b", context):
        return True
    return False
