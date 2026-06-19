"""Local filesystem document store.

Stores raw content, normalized markdown, and optional PDFs for synced documents.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from memforge.models import slugify

logger = logging.getLogger(__name__)

__all__ = ["DocumentStore", "LocalDocumentStore", "StoredDocumentArtifact"]


@dataclass(frozen=True)
class StoredDocumentArtifact:
    """A readable document artifact in the configured artifact store."""

    uri: str
    filename: str
    media_type: str
    size_bytes: int | None = None


class DocumentStore(Protocol):
    def store_raw(
        self,
        source_name: str,
        title: str,
        content: bytes,
        content_type: str,
        extension: str | None = None,
    ) -> str: ...

    def store_normalized(self, source_name: str, title: str, markdown: str) -> str: ...
    def store_pdf(self, source_name: str, title: str, pdf_bytes: bytes) -> str: ...
    def read_normalized(self, stored_path: str) -> str | None: ...
    def get_artifact(self, uri: str | None, media_type: str) -> StoredDocumentArtifact | None: ...
    def read_artifact(self, uri: str) -> bytes: ...
    def delete_document_files(self, source_name: str, title: str) -> None: ...


class LocalDocumentStore:
    """Filesystem-based document content storage.

    Directory layout:
        {docs_path}/{source_slug}/{doc_slug}.raw.html   (or .raw.json)
        {docs_path}/{source_slug}/{doc_slug}.md
        {docs_path}/{source_slug}/{doc_slug}.pdf         (optional)
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

    def _source_dir(self, source_name: str) -> Path:
        return self._root / slugify(source_name)

    def _doc_stem(self, title: str) -> str:
        return slugify(title)

    def store_raw(
        self,
        source_name: str,
        title: str,
        content: bytes,
        content_type: str,
        extension: str | None = None,
    ) -> str:
        """Store raw document content. Returns the stored file path."""
        source_dir = self._source_dir(source_name)
        source_dir.mkdir(parents=True, exist_ok=True)
        if extension:
            ext = extension
        elif "pdf" in content_type:
            ext = ".pdf"
        elif "json" in content_type:
            ext = ".raw.json"
        else:
            ext = ".raw.html"
        path = source_dir / f"{self._doc_stem(title)}{ext}"
        path.write_bytes(content)
        return str(path)

    def store_normalized(
        self,
        source_name: str,
        title: str,
        markdown: str,
    ) -> str:
        """Store normalized markdown content. Returns the stored file path."""
        source_dir = self._source_dir(source_name)
        source_dir.mkdir(parents=True, exist_ok=True)
        path = source_dir / f"{self._doc_stem(title)}.md"
        path.write_text(markdown, encoding="utf-8")
        return str(path)

    def store_pdf(
        self,
        source_name: str,
        title: str,
        pdf_bytes: bytes,
    ) -> str:
        """Store PDF export. Returns the stored file path."""
        source_dir = self._source_dir(source_name)
        source_dir.mkdir(parents=True, exist_ok=True)
        path = source_dir / f"{self._doc_stem(title)}.pdf"
        path.write_bytes(pdf_bytes)
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

    def delete_document_files(self, source_name: str, title: str) -> None:
        """Delete all files for a document."""
        source_dir = self._source_dir(source_name)
        stem = self._doc_stem(title)
        for ext in [".raw.html", ".raw.json", ".md", ".pdf"]:
            path = source_dir / f"{stem}{ext}"
            if path.exists():
                path.unlink()
                logger.debug("Deleted %s", path)
