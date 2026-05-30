"""Canonical payload builders for derived memory and document indexes."""

from __future__ import annotations

import json
from collections.abc import Sequence

from memforge.models import DocumentMetadata, Memory, content_hash

__all__ = [
    "document_embedding_text",
    "embedding_text_hash",
    "embedding_vector_hash",
    "memory_embedding_text",
]


def memory_embedding_text(memory: Memory, entity_names: list[str] | None = None) -> str:
    """Build the canonical text embedded for a memory vector."""
    prefix = {
        "fact": "Fact",
        "decision": "Decision",
        "convention": "Convention",
        "procedure": "Procedure",
    }.get(memory.memory_type, "Knowledge")
    names = entity_names if entity_names is not None else memory.entity_refs
    entities = ", ".join(names) if names else ""
    return f"{prefix}: {memory.content}\nEntities: {entities}"


def document_embedding_text(metadata: DocumentMetadata) -> str:
    """Build the canonical text embedded for a document vector."""
    return (
        f"{metadata.summary}\n"
        f"{' '.join(metadata.tags)}\n"
        f"{metadata.doc_type}\n"
        f"{metadata.complexity}"
    )


def embedding_text_hash(text: str) -> str:
    """Hash the exact text used as vector embedding input."""
    return content_hash(text)


def embedding_vector_hash(embedding: Sequence[float]) -> str:
    """Hash the exact vector payload stored in Chroma."""
    payload = json.dumps([float(value) for value in embedding], separators=(",", ":"))
    return content_hash(payload)
