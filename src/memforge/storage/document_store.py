"""Local filesystem document store.

Stores raw content, normalized markdown, and optional PDFs for synced documents.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Protocol

from memforge.models import slugify
from memforge.source_artifacts import extension_for_media_type

logger = logging.getLogger(__name__)

__all__ = [
    "ArtifactNotOwnedError",
    "DocumentStore",
    "LocalDocumentStore",
    "StoredDocumentArtifact",
    "document_artifact_identity",
]


class ArtifactNotOwnedError(ValueError):
    """The requested artifact URI is outside this store's ownership boundary."""


@dataclass(frozen=True)
class StoredDocumentArtifact:
    """A readable document artifact in the configured artifact store."""

    uri: str
    filename: str
    media_type: str
    size_bytes: int | None = None


def document_artifact_identity(doc_id: str) -> str:
    """Return the stable collision-resistant namespace for one Document."""
    return sha256(doc_id.encode("utf-8")).hexdigest()


class DocumentStore(Protocol):
    def store_raw(
        self,
        source_id: str,
        doc_id: str,
        title: str,
        content: bytes,
        content_type: str,
        extension: str | None = None,
    ) -> str: ...

    def store_normalized(
        self,
        source_id: str,
        doc_id: str,
        title: str,
        markdown: str,
    ) -> str: ...

    def store_pdf(
        self,
        source_id: str,
        doc_id: str,
        title: str,
        pdf_bytes: bytes,
    ) -> str: ...
    def store_source_artifact(
        self,
        *,
        source_id: str,
        artifact_id: str,
        filename: str,
        content: bytes,
        content_type: str,
    ) -> str: ...
    def read_normalized(self, stored_path: str) -> str | None: ...
    def get_artifact(self, uri: str | None, media_type: str) -> StoredDocumentArtifact | None: ...
    def read_artifact(self, uri: str) -> bytes: ...
    def delete_artifact(self, uri: str) -> None: ...


class LocalDocumentStore:
    """Filesystem-based document content storage.

    Directory layout:
        {docs_path}/{source_id_slug}/{document_identity}/{title_slug}.raw.html
        {docs_path}/{source_id_slug}/{document_identity}/{title_slug}.md
        {docs_path}/{source_id_slug}/{document_identity}/{title_slug}.pdf
    """

    def __init__(self, docs_path: str) -> None:
        self._root = Path(docs_path)

    def _resolve_artifact_path(self, uri: str | None) -> Path | None:
        if not uri:
            return None

        candidate = Path(uri).expanduser()
        if not candidate.is_absolute():
            candidate = self._root / candidate

        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            return None

        docs_root = self._root.expanduser().resolve()
        if not (resolved == docs_root or docs_root in resolved.parents):
            return None
        if not resolved.is_file():
            return None
        return resolved

    def _document_dir(self, source_id: str, doc_id: str) -> Path:
        return self._root / slugify(source_id) / document_artifact_identity(doc_id)

    def _doc_stem(self, title: str) -> str:
        return slugify(title)

    def store_raw(
        self,
        source_id: str,
        doc_id: str,
        title: str,
        content: bytes,
        content_type: str,
        extension: str | None = None,
    ) -> str:
        """Store raw document content. Returns the stored file path."""
        document_dir = self._document_dir(source_id, doc_id)
        document_dir.mkdir(parents=True, exist_ok=True)
        if extension:
            ext = extension
        elif "pdf" in content_type:
            ext = ".pdf"
        elif "json" in content_type:
            ext = ".raw.json"
        else:
            ext = ".raw.html"
        path = document_dir / f"{self._doc_stem(title)}{ext}"
        path.write_bytes(content)
        return str(path)

    def store_normalized(
        self,
        source_id: str,
        doc_id: str,
        title: str,
        markdown: str,
    ) -> str:
        """Store normalized markdown content. Returns the stored file path."""
        document_dir = self._document_dir(source_id, doc_id)
        document_dir.mkdir(parents=True, exist_ok=True)
        path = document_dir / f"{self._doc_stem(title)}.md"
        path.write_text(markdown, encoding="utf-8")
        return str(path)

    def store_pdf(
        self,
        source_id: str,
        doc_id: str,
        title: str,
        pdf_bytes: bytes,
    ) -> str:
        """Store PDF export. Returns the stored file path."""
        document_dir = self._document_dir(source_id, doc_id)
        document_dir.mkdir(parents=True, exist_ok=True)
        path = document_dir / f"{self._doc_stem(title)}.pdf"
        path.write_bytes(pdf_bytes)
        return str(path)

    def store_source_artifact(
        self,
        *,
        source_id: str,
        artifact_id: str,
        filename: str,
        content: bytes,
        content_type: str,
    ) -> str:
        """Store an exact provider Artifact under stable Artifact identity."""

        artifact_dir = self._root / slugify(source_id) / "source-artifacts" / document_artifact_identity(artifact_id)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        safe_stem = slugify(Path(filename).stem) or "artifact"
        path = artifact_dir / f"{safe_stem}{extension_for_media_type(content_type)}"
        path.write_bytes(content)
        return str(path)

    def read_normalized(self, stored_path: str) -> str | None:
        """Read normalized markdown from a stored file path."""
        path = self._resolve_artifact_path(stored_path)
        if path is not None:
            return path.read_text(encoding="utf-8")
        return None

    def get_artifact(self, uri: str | None, media_type: str) -> StoredDocumentArtifact | None:
        """Resolve a readable local artifact if it is still present."""
        path = self._resolve_artifact_path(uri)
        if path is None:
            return None
        return StoredDocumentArtifact(
            uri=str(path),
            filename=path.name,
            media_type=media_type,
            size_bytes=path.stat().st_size,
        )

    def read_artifact(self, uri: str) -> bytes:
        """Read a stored local artifact."""
        path = self._resolve_artifact_path(uri)
        if path is None:
            raise FileNotFoundError(uri)
        return path.read_bytes()

    def delete_artifact(self, uri: str) -> None:
        """Idempotently delete one exact artifact owned by this store."""
        candidate = Path(uri).expanduser()
        if not candidate.is_absolute():
            candidate = self._root / candidate
        path = candidate.resolve()
        docs_root = self._root.expanduser().resolve()
        if path != docs_root and docs_root not in path.parents:
            raise ArtifactNotOwnedError("artifact URI is outside the document store")
        path.unlink(missing_ok=True)
