"""Local filesystem document store.

Stores raw content, normalized markdown, and optional PDFs for synced documents.
"""

from __future__ import annotations

import logging
from pathlib import Path

from meminception.models import slugify

logger = logging.getLogger(__name__)

__all__ = ["LocalDocumentStore"]


class LocalDocumentStore:
    """Filesystem-based document content storage.

    Directory layout:
        {docs_path}/{source_slug}/{doc_slug}.raw.html   (or .raw.json)
        {docs_path}/{source_slug}/{doc_slug}.md
        {docs_path}/{source_slug}/{doc_slug}.pdf         (optional)
    """

    def __init__(self, docs_path: str) -> None:
        self._root = Path(docs_path)

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
        """Store raw document content. Returns the file URI."""
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
        """Store normalized markdown content. Returns the file URI."""
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
        """Store PDF export. Returns the file URI."""
        source_dir = self._source_dir(source_name)
        source_dir.mkdir(parents=True, exist_ok=True)
        path = source_dir / f"{self._doc_stem(title)}.pdf"
        path.write_bytes(pdf_bytes)
        return str(path)

    def read_normalized(self, file_uri: str) -> str | None:
        """Read normalized markdown from a file URI."""
        path = Path(file_uri)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return None

    def delete_document_files(self, source_name: str, title: str) -> None:
        """Delete all files for a document."""
        source_dir = self._source_dir(source_name)
        stem = self._doc_stem(title)
        for ext in [".raw.html", ".raw.json", ".md", ".pdf"]:
            path = source_dir / f"{stem}{ext}"
            if path.exists():
                path.unlink()
                logger.debug("Deleted %s", path)
