"""Deterministic health checks for memory search indexes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from memforge.memory.index_payloads import (
    document_embedding_text,
    embedding_text_hash,
    embedding_vector_hash,
    memory_embedding_text,
)
from memforge.memory.lifecycle import allowed_search_statuses

__all__ = ["MemoryIndexHealthChecker", "MemoryIndexHealthIssue", "MemoryIndexHealthReport"]


@dataclass
class MemoryIndexHealthIssue:
    kind: str
    severity: str
    memory_id: str | None = None
    detail: str = ""


@dataclass
class MemoryIndexHealthReport:
    issues: list[MemoryIndexHealthIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues


class MemoryIndexHealthChecker:
    """Compare SQLite state with FTS5, memory vectors, and document vectors."""

    def __init__(
        self,
        db,
        memory_collection: Any | None = None,
        document_collection: Any | None = None,
    ) -> None:
        self.db = db
        self.memory_collection = memory_collection
        self.document_collection = document_collection

    async def check(self) -> MemoryIndexHealthReport:
        statuses = await self._memory_statuses()
        try:
            fts_records, duplicate_fts_ids = await self._fts_records()
            fts_ids = set(fts_records)
            fts_error = None
        except Exception as exc:
            fts_ids = set()
            duplicate_fts_ids = set()
            fts_error = MemoryIndexHealthIssue(
                kind="fts_unavailable",
                severity="P0",
                detail=str(exc),
            )
        chroma_records, chroma_error = self._collection_records(
            self.memory_collection,
            issue_prefix="memory",
        )
        chroma_ids = set(chroma_records)
        document_ids = await self._document_ids()
        document_records, document_chroma_error = self._collection_records(
            self.document_collection,
            issue_prefix="document",
        )
        document_chroma_ids = set(document_records)
        confluence_documents_missing_pdf = await self._confluence_documents_missing_pdf()
        search_visible_statuses = set(allowed_search_statuses())

        issues: list[MemoryIndexHealthIssue] = []
        if fts_error:
            issues.append(fts_error)
        if chroma_error:
            issues.append(chroma_error)
        if document_chroma_error:
            issues.append(document_chroma_error)
        for memory_id in sorted(duplicate_fts_ids):
            issues.append(MemoryIndexHealthIssue(
                kind="fts_duplicate",
                severity="P0",
                memory_id=memory_id,
                detail="FTS5 contains more than one row for the same memory ID",
            ))

        for memory_id, status in statuses.items():
            search_visible = status in search_visible_statuses
            if search_visible and memory_id not in fts_ids:
                issues.append(MemoryIndexHealthIssue(
                    kind="active_missing_fts",
                    severity="P0",
                    memory_id=memory_id,
                    detail="Search-visible memory is missing from FTS5",
                ))
            if search_visible and self.memory_collection is not None and memory_id not in chroma_ids:
                issues.append(MemoryIndexHealthIssue(
                    kind="active_missing_chroma",
                    severity="P0",
                    memory_id=memory_id,
                    detail="Search-visible memory is missing from Chroma",
                ))
            if not search_visible and memory_id in fts_ids:
                issues.append(MemoryIndexHealthIssue(
                    kind="non_active_present_fts",
                    severity="P0",
                    memory_id=memory_id,
                    detail=f"Non-active memory with status {status!r} is still in FTS5",
                ))
            if not search_visible and memory_id in chroma_ids:
                issues.append(MemoryIndexHealthIssue(
                    kind="non_active_present_chroma",
                    severity="P0",
                    memory_id=memory_id,
                    detail=f"Non-active memory with status {status!r} is still in Chroma",
                ))

            fts_record = fts_records.get(memory_id)
            if search_visible and fts_record:
                expected = await self._memory_search_text(memory_id)
                if expected and fts_record.get("content") != expected["content"]:
                    issues.append(MemoryIndexHealthIssue(
                        kind="fts_content_mismatch",
                        severity="P0",
                        memory_id=memory_id,
                        detail="SQLite memory content differs from FTS5 content",
                    ))
                if expected and fts_record.get("tags_text") != expected["tags_text"]:
                    issues.append(MemoryIndexHealthIssue(
                        kind="fts_tags_mismatch",
                        severity="P0",
                        memory_id=memory_id,
                        detail="SQLite memory tags differ from FTS5 tags text",
                    ))
                if expected and fts_record.get("entities_text") != expected["entities_text"]:
                    issues.append(MemoryIndexHealthIssue(
                        kind="fts_entities_mismatch",
                        severity="P0",
                        memory_id=memory_id,
                        detail="SQLite memory entities differ from FTS5 entities text",
                    ))

            metadata_status = chroma_records.get(memory_id, {}).get("status")
            if search_visible and self.memory_collection is not None and not metadata_status:
                issues.append(MemoryIndexHealthIssue(
                    kind="chroma_status_missing",
                    severity="P0",
                    memory_id=memory_id,
                    detail="Search-visible memory Chroma metadata has no status",
                ))
            elif metadata_status and metadata_status != status:
                issues.append(MemoryIndexHealthIssue(
                    kind="chroma_status_mismatch",
                    severity="P0",
                    memory_id=memory_id,
                    detail=f"SQLite status {status!r} differs from Chroma metadata {metadata_status!r}",
                ))
            metadata_content_hash = chroma_records.get(memory_id, {}).get("content_hash")
            memory_content_hash = await self._memory_content_hash(memory_id)
            if search_visible and memory_content_hash and not metadata_content_hash:
                issues.append(MemoryIndexHealthIssue(
                    kind="chroma_content_hash_missing",
                    severity="P0",
                    memory_id=memory_id,
                    detail="Search-visible memory Chroma metadata has no content hash",
                ))
            if (
                metadata_content_hash
                and memory_content_hash
                and metadata_content_hash != memory_content_hash
            ):
                issues.append(MemoryIndexHealthIssue(
                    kind="chroma_content_hash_mismatch",
                    severity="P0",
                    memory_id=memory_id,
                    detail="SQLite content hash differs from Chroma metadata",
                ))
            metadata_embedding_text_hash = chroma_records.get(memory_id, {}).get("embedding_text_hash")
            memory_embedding_text_hash = await self._memory_embedding_text_hash(memory_id)
            if search_visible and memory_embedding_text_hash and not metadata_embedding_text_hash:
                issues.append(MemoryIndexHealthIssue(
                    kind="chroma_embedding_text_hash_missing",
                    severity="P0",
                    memory_id=memory_id,
                    detail="Search-visible memory Chroma metadata has no embedding text hash",
                ))
            if (
                metadata_embedding_text_hash
                and memory_embedding_text_hash
                and metadata_embedding_text_hash != memory_embedding_text_hash
            ):
                issues.append(MemoryIndexHealthIssue(
                    kind="chroma_embedding_text_hash_mismatch",
                    severity="P0",
                    memory_id=memory_id,
                    detail="SQLite memory embedding text differs from Chroma metadata",
                ))
            actual_embedding_vector_hash = chroma_records.get(memory_id, {}).get("_embedding_vector_hash")
            metadata_embedding_vector_hash = chroma_records.get(memory_id, {}).get("embedding_vector_hash")
            if search_visible and actual_embedding_vector_hash and not metadata_embedding_vector_hash:
                issues.append(MemoryIndexHealthIssue(
                    kind="chroma_embedding_vector_hash_missing",
                    severity="P0",
                    memory_id=memory_id,
                    detail="Search-visible memory Chroma metadata has no embedding vector hash",
                ))
            if (
                actual_embedding_vector_hash
                and metadata_embedding_vector_hash
                and actual_embedding_vector_hash != metadata_embedding_vector_hash
            ):
                issues.append(MemoryIndexHealthIssue(
                    kind="chroma_embedding_vector_hash_mismatch",
                    severity="P0",
                    memory_id=memory_id,
                    detail="Chroma memory embedding payload differs from Chroma metadata",
                ))

        for memory_id in chroma_ids - set(statuses):
            issues.append(MemoryIndexHealthIssue(
                kind="chroma_orphan",
                severity="P0",
                memory_id=memory_id,
                detail="Chroma contains a memory ID missing from SQLite",
            ))
        for memory_id in fts_ids - set(statuses):
            issues.append(MemoryIndexHealthIssue(
                kind="fts_orphan",
                severity="P0",
                memory_id=memory_id,
                detail="FTS5 contains a memory ID missing from SQLite",
            ))

        for doc_id in sorted(confluence_documents_missing_pdf):
            issues.append(MemoryIndexHealthIssue(
                kind="confluence_pdf_uri_missing",
                severity="P1",
                memory_id=doc_id,
                detail="Confluence document is missing its PDF provenance URI",
            ))

        if self.document_collection is not None and not document_chroma_error:
            document_state = await self._document_state()
            for doc_id in document_ids - document_chroma_ids:
                issues.append(MemoryIndexHealthIssue(
                    kind="document_missing_chroma",
                    severity="P0",
                    memory_id=doc_id,
                    detail="SQLite document is missing from document Chroma",
                ))
            for doc_id in document_chroma_ids - document_ids:
                issues.append(MemoryIndexHealthIssue(
                    kind="document_chroma_orphan",
                    severity="P0",
                    memory_id=doc_id,
                    detail="Document Chroma contains a document ID missing from SQLite",
                ))
            for doc_id in document_ids & document_chroma_ids:
                metadata = document_records.get(doc_id, {})
                db_state = document_state.get(doc_id, {})
                if (
                    metadata.get("content_hash")
                    and db_state.get("content_hash")
                    and metadata["content_hash"] != db_state["content_hash"]
                ):
                    issues.append(MemoryIndexHealthIssue(
                        kind="document_chroma_content_hash_mismatch",
                        severity="P0",
                        memory_id=doc_id,
                        detail="SQLite document content hash differs from document Chroma metadata",
                    ))
                if db_state.get("content_hash") and not metadata.get("content_hash"):
                    issues.append(MemoryIndexHealthIssue(
                        kind="document_chroma_content_hash_missing",
                        severity="P0",
                        memory_id=doc_id,
                        detail="Document Chroma metadata has no content hash",
                    ))
                if (
                    metadata.get("version")
                    and db_state.get("version")
                    and metadata["version"] != db_state["version"]
                ):
                    issues.append(MemoryIndexHealthIssue(
                        kind="document_chroma_version_mismatch",
                        severity="P0",
                        memory_id=doc_id,
                        detail="SQLite document version differs from document Chroma metadata",
                    ))
                if db_state.get("version") and not metadata.get("version"):
                    issues.append(MemoryIndexHealthIssue(
                        kind="document_chroma_version_missing",
                        severity="P0",
                        memory_id=doc_id,
                        detail="Document Chroma metadata has no document version",
                    ))
                expected_embedding_text_hash = await self._document_embedding_text_hash(doc_id)
                actual_embedding_text_hash = metadata.get("embedding_text_hash")
                if expected_embedding_text_hash and not actual_embedding_text_hash:
                    issues.append(MemoryIndexHealthIssue(
                        kind="document_chroma_embedding_text_hash_missing",
                        severity="P0",
                        memory_id=doc_id,
                        detail="Document Chroma metadata has no embedding text hash",
                    ))
                if (
                    expected_embedding_text_hash
                    and actual_embedding_text_hash
                    and actual_embedding_text_hash != expected_embedding_text_hash
                ):
                    issues.append(MemoryIndexHealthIssue(
                        kind="document_chroma_embedding_text_hash_mismatch",
                        severity="P0",
                        memory_id=doc_id,
                        detail="SQLite document embedding text differs from document Chroma metadata",
                    ))
                actual_embedding_vector_hash = metadata.get("_embedding_vector_hash")
                metadata_embedding_vector_hash = metadata.get("embedding_vector_hash")
                if actual_embedding_vector_hash and not metadata_embedding_vector_hash:
                    issues.append(MemoryIndexHealthIssue(
                        kind="document_chroma_embedding_vector_hash_missing",
                        severity="P0",
                        memory_id=doc_id,
                        detail="Document Chroma metadata has no embedding vector hash",
                    ))
                if (
                    actual_embedding_vector_hash
                    and metadata_embedding_vector_hash
                    and actual_embedding_vector_hash != metadata_embedding_vector_hash
                ):
                    issues.append(MemoryIndexHealthIssue(
                        kind="document_chroma_embedding_vector_hash_mismatch",
                        severity="P0",
                        memory_id=doc_id,
                        detail="Document Chroma embedding payload differs from document Chroma metadata",
                    ))

        return MemoryIndexHealthReport(issues=issues)

    async def _memory_statuses(self) -> dict[str, str]:
        statuses: dict[str, str] = {}
        async with self.db.db.execute("SELECT id, status FROM memories") as cursor:
            async for row in cursor:
                statuses[row[0]] = row[1]
        return statuses

    async def _fts_records(self) -> tuple[dict[str, dict[str, str]], set[str]]:
        records: dict[str, dict[str, str]] = {}
        duplicate_ids: set[str] = set()
        async with self.db.db.execute(
            "SELECT memory_id, content, entities_text, tags_text FROM memories_fts"
        ) as cursor:
            async for row in cursor:
                if row[0] in records:
                    duplicate_ids.add(row[0])
                records[row[0]] = {
                    "content": row[1],
                    "entities_text": row[2],
                    "tags_text": row[3],
                }
        return records, duplicate_ids

    async def _document_ids(self) -> set[str]:
        ids: set[str] = set()
        async with self.db.db.execute("SELECT doc_id FROM documents") as cursor:
            async for row in cursor:
                ids.add(row[0])
        return ids

    async def _confluence_documents_missing_pdf(self) -> set[str]:
        doc_ids: set[str] = set()
        async with self.db.db.execute(
            """SELECT d.doc_id
               FROM documents d
               JOIN sources s ON s.id = d.source
               WHERE s.type = 'confluence'
                 AND d.normalized_content_uri IS NOT NULL
                 AND d.normalized_content_uri <> ''
                 AND (d.pdf_content_uri IS NULL OR d.pdf_content_uri = '')"""
        ) as cursor:
            async for row in cursor:
                doc_ids.add(row[0])
        return doc_ids

    async def _memory_content_hash(self, memory_id: str) -> str | None:
        async with self.db.db.execute(
            "SELECT content_hash FROM memories WHERE id = ?",
            (memory_id,),
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else None

    async def _memory_search_text(self, memory_id: str) -> dict[str, str] | None:
        async with self.db.db.execute(
            "SELECT content, tags FROM memories WHERE id = ?",
            (memory_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if not row:
            return None
        entity_names: list[str] = []
        async with self.db.db.execute(
            """SELECT e.canonical_name
               FROM memory_entities me
               JOIN entities e ON me.entity_id = e.id
               WHERE me.memory_id = ?
               ORDER BY e.id""",
            (memory_id,),
        ) as cursor:
            async for entity_row in cursor:
                entity_names.append(entity_row[0])
        import json

        tags = json.loads(row[1] or "[]")
        return {
            "content": row[0],
            "tags_text": " ".join(tags),
            "entities_text": " ".join(entity_names),
        }

    async def _memory_embedding_text_hash(self, memory_id: str) -> str | None:
        memory = await self.db.get_memory(memory_id)
        if memory is None:
            return None
        entity_names = await self.db.get_memory_entity_names(memory_id)
        return embedding_text_hash(memory_embedding_text(memory, entity_names))

    async def _document_embedding_text_hash(self, doc_id: str) -> str | None:
        metadata = await self.db.get_metadata(doc_id)
        if metadata is None:
            return None
        return embedding_text_hash(document_embedding_text(metadata))

    async def _document_state(self) -> dict[str, dict[str, str | None]]:
        state: dict[str, dict[str, str | None]] = {}
        async with self.db.db.execute("SELECT doc_id, content_hash, version FROM documents") as cursor:
            async for row in cursor:
                state[row[0]] = {"content_hash": row[1], "version": row[2]}
        return state

    def _collection_records(
        self,
        collection: Any | None,
        *,
        issue_prefix: str,
    ) -> tuple[dict[str, dict[str, Any]], MemoryIndexHealthIssue | None]:
        if collection is None:
            return {}, None
        try:
            raw = collection.get(include=["metadatas", "embeddings"])
        except Exception as exc:
            return {}, MemoryIndexHealthIssue(
                kind=f"{issue_prefix}_chroma_unavailable",
                severity="P0",
                detail=str(exc),
            )
        ids = raw.get("ids") or []
        metadatas = raw.get("metadatas") or []
        embeddings = raw.get("embeddings")
        if embeddings is None:
            embeddings = []
        records: dict[str, dict[str, Any]] = {}
        for index, record_id in enumerate(ids):
            metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
            records[record_id] = dict(metadata)
            if index < len(embeddings) and embeddings[index] is not None:
                records[record_id]["_embedding_vector_hash"] = embedding_vector_hash(embeddings[index])
        return records, None
