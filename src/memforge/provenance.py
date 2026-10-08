"""Links to the stored content of source Documents exposed to agent clients.

Each Source stores its own copy of a Document it syncs, recorded as the stored
input of its Source Unit (:class:`memforge.models.SourceUnitInput`). A Unit
link (``/api/v1/source-units/{source_unit_id}/...``) serves the copy stored
with that Unit's current revision; Memory Evidence names the Unit that supports
the Memory, so its links read that Source's copy. A Document link
(``/api/v1/documents/{doc_id}/...``) serves, among the Sources that hold the
Document and that the caller can read, the most recently recorded copy whose
requested object is still stored.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote

from memforge.config import AppConfig
from memforge.models import SourceUnitInput
from memforge.storage.document_store import DocumentStore, LocalDocumentStore, StoredDocumentArtifact


DocumentArtifactStore = DocumentStore

NORMALIZED_MARKDOWN_MEDIA_TYPE = "text/markdown; charset=utf-8"
PDF_MEDIA_TYPE = "application/pdf"


def source_unit_resource_path(source_unit_id: str) -> str:
    return f"/api/v1/source-units/{quote(source_unit_id, safe='')}"


def document_resource_path(doc_id: str) -> str:
    return f"/api/v1/documents/{quote(doc_id, safe='')}"


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


def list_stored_input_artifacts(
    unit_input: SourceUnitInput,
    config: AppConfig | None,
    artifact_store: DocumentArtifactStore | None = None,
    *,
    resource_path: str,
) -> dict[str, DocumentArtifact]:
    """Return the readable objects of one stored input, keyed by artifact kind.

    ``resource_path`` is the Unit or Document path the artifact URLs extend.
    """
    artifacts: dict[str, DocumentArtifact] = {}
    if artifact_store is not None:
        store = artifact_store
    elif config is not None:
        store = LocalDocumentStore(config.storage.docs_path)
    else:
        return artifacts
    candidates = (
        ("normalized_markdown", unit_input.normalized_content_uri, NORMALIZED_MARKDOWN_MEDIA_TYPE),
        ("raw_source", unit_input.raw_content_uri, unit_input.raw_content_type),
        ("pdf", unit_input.pdf_content_uri, PDF_MEDIA_TYPE),
    )
    for kind, uri, media_type in candidates:
        if not store.belongs_to_document(uri, source_id=unit_input.source_id, doc_id=unit_input.document_id):
            continue
        stored = store.get_artifact(uri, media_type)
        if stored is not None:
            artifacts[kind] = DocumentArtifact(
                kind=kind,
                stored=stored,
                media_type=media_type,
                url=f"{resource_path}/artifacts/{quote(kind, safe='')}",
            )
    return artifacts


def select_stored_input_artifact(
    unit_input: SourceUnitInput,
    kind: str,
    config: AppConfig | None,
    artifact_store: DocumentArtifactStore | None = None,
    *,
    resource_path: str,
) -> DocumentArtifact | None:
    """Select an artifact by explicit kind, with a content alias for text fallback."""
    artifacts = list_stored_input_artifacts(unit_input, config, artifact_store, resource_path=resource_path)
    if kind == "content":
        return artifacts.get("normalized_markdown") or artifacts.get("raw_source")
    return artifacts.get(kind)


def source_unit_content_url(
    unit_input: SourceUnitInput | None,
    config: AppConfig | None,
    artifact_store: DocumentArtifactStore | None = None,
) -> str | None:
    """Return the Unit's content URL when its stored input has readable content.

    A store or configuration is required to establish the object ownership.
    """
    if unit_input is None:
        return None
    resource_path = source_unit_resource_path(unit_input.source_unit_id)
    if config is None and artifact_store is None:
        return None
    if select_stored_input_artifact(unit_input, "content", config, artifact_store, resource_path=resource_path) is None:
        return None
    return f"{resource_path}/content"


def source_unit_pdf_url(
    unit_input: SourceUnitInput | None,
    config: AppConfig | None,
    artifact_store: DocumentArtifactStore | None = None,
) -> str | None:
    """Return the Unit's PDF URL when its stored input has a readable PDF."""
    if unit_input is None:
        return None
    resource_path = source_unit_resource_path(unit_input.source_unit_id)
    if config is None and artifact_store is None:
        return None
    if select_stored_input_artifact(unit_input, "pdf", config, artifact_store, resource_path=resource_path) is None:
        return None
    return f"{resource_path}/pdf"
