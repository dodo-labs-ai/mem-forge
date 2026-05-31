"""Helpers for source-document provenance exposed to agent clients."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

from memforge.config import AppConfig
from memforge.models import DocumentRecord


def document_content_uri(doc: DocumentRecord | None) -> str | None:
    """Return the best stored content artifact for a document."""
    if doc is None:
        return None
    return doc.normalized_content_uri or doc.raw_content_uri


def document_content_url(doc: DocumentRecord | None) -> str | None:
    """Return a service URL for the stored content artifact, if present."""
    if doc is None or document_content_uri(doc) is None:
        return None
    return f"/api/documents/{quote(doc.doc_id, safe='')}/content"


def document_pdf_url(doc: DocumentRecord | None) -> str | None:
    """Return a service URL for the stored PDF artifact, if present."""
    if doc is None or doc.pdf_content_uri is None:
        return None
    return f"/api/documents/{quote(doc.doc_id, safe='')}/pdf"


def resolve_document_artifact_path(uri: str | None, config: AppConfig) -> Path | None:
    """Resolve a stored artifact path if it belongs to MemForge storage."""
    if not uri:
        return None

    candidate = Path(uri).expanduser()
    if not candidate.is_absolute():
        candidate = Path(config.storage.docs_path) / candidate

    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        return None

    docs_root = Path(config.storage.docs_path).expanduser().resolve()
    if not (resolved == docs_root or docs_root in resolved.parents):
        return None
    if not resolved.is_file():
        return None
    return resolved
