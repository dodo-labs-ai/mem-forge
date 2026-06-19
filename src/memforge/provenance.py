"""Helpers for source-document provenance exposed to agent clients."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from memforge.config import AppConfig
from memforge.models import DocumentRecord
from memforge.storage.document_store import DocumentStore, LocalDocumentStore, StoredDocumentArtifact


DocumentArtifactStore = DocumentStore


def document_content_uri(doc: DocumentRecord | None) -> str | None:
    """Return the best stored content artifact for a document."""
    if doc is None:
        return None
    return doc.normalized_content_uri or doc.raw_content_uri


def document_content_url(doc: DocumentRecord | None, config: AppConfig | None = None) -> str | None:
    """Return a service URL for the stored content artifact, if present."""
    if doc is None:
        return None
    if config is not None and select_document_artifact(doc, "content", config) is None:
        return None
    if config is None and document_content_uri(doc) is None:
        return None
    return f"/api/documents/{quote(doc.doc_id, safe='')}/content"


def document_pdf_url(doc: DocumentRecord | None, config: AppConfig | None = None) -> str | None:
    """Return a service URL for the stored PDF artifact, if present."""
    if doc is None:
        return None
    if config is not None and select_document_artifact(doc, "pdf", config) is None:
        return None
    if config is None and doc.pdf_content_uri is None:
        return None
    return f"/api/documents/{quote(doc.doc_id, safe='')}/pdf"


def document_content_url_for_store(
    doc: DocumentRecord | None,
    config: AppConfig,
    artifact_store: DocumentArtifactStore | None,
) -> str | None:
    """Return a content URL when the configured store can serve content."""
    if doc is None:
        return None
    if select_document_artifact(doc, "content", config, artifact_store) is None:
        return None
    return f"/api/documents/{quote(doc.doc_id, safe='')}/content"


def document_pdf_url_for_store(
    doc: DocumentRecord | None,
    config: AppConfig,
    artifact_store: DocumentArtifactStore | None,
) -> str | None:
    """Return a PDF URL when the configured store can serve a PDF."""
    if doc is None:
        return None
    if select_document_artifact(doc, "pdf", config, artifact_store) is None:
        return None
    return f"/api/documents/{quote(doc.doc_id, safe='')}/pdf"


@dataclass(frozen=True)
class DocumentArtifact:
    kind: str
    stored: StoredDocumentArtifact
    media_type: str
    url: str

    @property
    def filename(self) -> str:
        return self.stored.filename

    @property
    def size_bytes(self) -> int | None:
        return self.stored.size_bytes

    @property
    def uri(self) -> str:
        return self.stored.uri

    def metadata(self) -> dict[str, object]:
        data: dict[str, object] = {
            "kind": self.kind,
            "url": self.url,
            "content_type": self.media_type,
            "filename": self.filename,
        }
        if self.size_bytes is not None:
            data["size_bytes"] = self.size_bytes
        return data


def list_document_artifacts(
    doc: DocumentRecord,
    config: AppConfig,
    artifact_store: DocumentArtifactStore | None = None,
) -> dict[str, DocumentArtifact]:
    """Return available source artifacts keyed by explicit artifact kind."""
    artifacts: dict[str, DocumentArtifact] = {}
    store = artifact_store or LocalDocumentStore(config.storage.docs_path)

    normalized = store.get_artifact(
        doc.normalized_content_uri,
        "text/markdown; charset=utf-8",
    )
    if normalized is not None:
        artifacts["normalized_markdown"] = DocumentArtifact(
            kind="normalized_markdown",
            stored=normalized,
            media_type="text/markdown; charset=utf-8",
            url=_document_artifact_url(doc.doc_id, "normalized_markdown"),
        )

    raw = store.get_artifact(
        doc.raw_content_uri,
        doc.raw_content_type or "application/octet-stream",
    )
    if raw is not None:
        artifacts["raw_source"] = DocumentArtifact(
            kind="raw_source",
            stored=raw,
            media_type=doc.raw_content_type or "application/octet-stream",
            url=_document_artifact_url(doc.doc_id, "raw_source"),
        )

    pdf = store.get_artifact(doc.pdf_content_uri, "application/pdf")
    if pdf is not None:
        artifacts["pdf"] = DocumentArtifact(
            kind="pdf",
            stored=pdf,
            media_type="application/pdf",
            url=_document_artifact_url(doc.doc_id, "pdf"),
        )

    return artifacts


def select_document_artifact(
    doc: DocumentRecord,
    kind: str,
    config: AppConfig,
    artifact_store: DocumentArtifactStore | None = None,
) -> DocumentArtifact | None:
    """Select an artifact by explicit kind, with a content alias for text fallback."""
    artifacts = list_document_artifacts(doc, config, artifact_store)
    if kind == "content":
        return artifacts.get("normalized_markdown") or artifacts.get("raw_source")
    return artifacts.get(kind)


def _document_artifact_url(doc_id: str, kind: str) -> str:
    return f"/api/documents/{quote(doc_id, safe='')}/artifacts/{quote(kind, safe='')}"


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
