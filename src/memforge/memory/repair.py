"""Deterministic repair routines for memory search indexes."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from memforge.memory.index_payloads import (
    document_embedding_text,
    embedding_text_hash,
    embedding_vector_hash,
    memory_embedding_text,
)
from memforge.memory.lifecycle import allowed_search_statuses
from memforge.models import Memory
from memforge.retrieval.embeddings import embed_texts
from memforge.retrieval.vector_metadata import upsert_with_stored_vector_hash

__all__ = ["MemoryIndexRepairResult", "MemoryIndexRepairer"]


@dataclass
class MemoryIndexRepairResult:
    fts_rows_rebuilt: int = 0
    fts_rows_deleted: int = 0
    memory_vectors_repaired: int = 0
    memory_vectors_deleted: int = 0
    document_vectors_repaired: int = 0
    document_vectors_created: int = 0
    unrepaired_memories: list[str] = field(default_factory=list)
    unrepaired_documents: list[str] = field(default_factory=list)


class MemoryIndexRepairer:
    """Repair FTS5 and Chroma indexes from SQLite, the source of truth."""

    def __init__(
        self,
        *,
        db,
        memory_collection: Any | None,
        document_collection: Any | None,
        embed_cfg: dict[str, Any],
    ) -> None:
        self.db = db
        self.memory_collection = memory_collection
        self.document_collection = document_collection
        self.embed_cfg = embed_cfg
        self.search_visible_statuses = set(allowed_search_statuses())

    async def repair(self) -> MemoryIndexRepairResult:
        result = MemoryIndexRepairResult()
        memories = await self._memories()
        result.fts_rows_deleted = await self.db.prune_memory_fts_orphans()
        result.fts_rows_rebuilt = await self._rebuild_memory_fts(memories)
        if self.memory_collection is not None:
            await self._repair_memory_vectors(memories, result)
        if self.document_collection is not None:
            await self._repair_document_vectors(result)
        return result

    async def _memories(self) -> list[Memory]:
        rows: list[Memory] = []
        async with self.db.db.execute("SELECT id FROM memories ORDER BY id") as cursor:
            ids = [row["id"] async for row in cursor]
        for memory_id in ids:
            memory = await self.db.get_memory(memory_id)
            if memory:
                rows.append(memory)
        return rows

    async def _rebuild_memory_fts(self, memories: list[Memory]) -> int:
        rebuilt = 0
        for memory in memories:
            if await self.db.rebuild_memory_fts(
                memory.id,
                search_visible_statuses=self.search_visible_statuses,
            ):
                rebuilt += 1
        return rebuilt

    async def _repair_memory_vectors(
        self,
        memories: list[Memory],
        result: MemoryIndexRepairResult,
    ) -> None:
        assert self.memory_collection is not None
        records = self._collection_records(self.memory_collection)
        by_id = {memory.id: memory for memory in memories}

        for record_id in set(records) - set(by_id):
            self.memory_collection.delete(ids=[record_id])
            result.memory_vectors_deleted += 1

        for memory in memories:
            search_visible = memory.status in self.search_visible_statuses
            existing = records.get(memory.id)
            if not search_visible:
                if existing is not None:
                    self.memory_collection.delete(ids=[memory.id])
                    result.memory_vectors_deleted += 1
                continue

            expected_embedding_text = await self._memory_embedding_text(memory)
            if existing is None:
                if not self._has_embedding_config():
                    result.unrepaired_memories.append(memory.id)
                    continue
                vector = await self._embed_memory(memory)
                metadata = await self._memory_metadata(
                    memory,
                    None,
                    expected_embedding_text=expected_embedding_text,
                    embedding=vector,
                )
                upsert_with_stored_vector_hash(
                    self.memory_collection,
                    ids=[memory.id],
                    embeddings=[vector],
                    metadatas=[metadata],
                )
                result.memory_vectors_repaired += 1
                continue

            actual_metadata = existing.get("metadata") or {}
            existing_embedding = existing.get("embedding")
            metadata = await self._memory_metadata(
                memory,
                actual_metadata,
                expected_embedding_text=expected_embedding_text,
                embedding=existing_embedding,
            )
            vector_is_stale = (
                actual_metadata.get("embedding_text_hash") != metadata["embedding_text_hash"]
                or actual_metadata.get("content_hash") != metadata["content_hash"]
            )
            if vector_is_stale:
                if not self._has_embedding_config():
                    result.unrepaired_memories.append(memory.id)
                    continue
                vector = await self._embed_memory(memory)
                metadata = await self._memory_metadata(
                    memory,
                    actual_metadata,
                    expected_embedding_text=expected_embedding_text,
                    embedding=vector,
                )
                upsert_with_stored_vector_hash(
                    self.memory_collection,
                    ids=[memory.id],
                    embeddings=[vector],
                    metadatas=[metadata],
                )
                result.memory_vectors_repaired += 1
            elif self._metadata_needs_update(actual_metadata, metadata):
                self._update_collection_metadata(
                    self.memory_collection,
                    memory.id,
                    metadata,
                    existing,
                )
                result.memory_vectors_repaired += 1

    async def _memory_metadata(
        self,
        memory: Memory,
        existing_metadata: dict[str, Any] | None,
        *,
        expected_embedding_text: str,
        embedding: Any | None,
    ) -> dict[str, Any]:
        metadata = dict(existing_metadata or {})
        sources = await self.db.get_memory_sources(memory.id)
        if sources and not metadata.get("source_doc_id"):
            metadata["source_doc_id"] = sources[0].doc_id
        metadata.update({
            "memory_type": memory.memory_type,
            "space_or_project": memory.project_key or "",
            "confidence": memory.confidence,
            "status": memory.status,
            "content_hash": memory.content_hash,
            "embedding_text_hash": embedding_text_hash(expected_embedding_text),
        })
        if embedding is not None:
            metadata["embedding_vector_hash"] = embedding_vector_hash(embedding)
        return metadata

    async def _repair_document_vectors(self, result: MemoryIndexRepairResult) -> None:
        assert self.document_collection is not None
        records = self._collection_records(self.document_collection)
        docs = await self._documents()
        doc_ids = {doc["doc_id"] for doc in docs}

        for record_id in set(records) - doc_ids:
            self.document_collection.delete(ids=[record_id])

        for doc in docs:
            doc_id = doc["doc_id"]
            expected_embedding_text = await self._document_embedding_text(doc_id)
            existing = records.get(doc_id)
            if existing is None:
                if expected_embedding_text is None or not self._has_embedding_config():
                    result.unrepaired_documents.append(doc_id)
                    continue
                vector = await self._embed_document(expected_embedding_text)
                expected = self._document_metadata(
                    doc,
                    expected_embedding_text=expected_embedding_text,
                    embedding=vector,
                )
                upsert_with_stored_vector_hash(
                    self.document_collection,
                    ids=[doc_id],
                    embeddings=[vector],
                    documents=[expected_embedding_text],
                    metadatas=[expected],
                )
                result.document_vectors_created += 1
                continue

            actual_metadata = existing.get("metadata") or {}
            expected = self._document_metadata(
                doc,
                expected_embedding_text=expected_embedding_text,
                embedding=existing.get("embedding"),
            )
            vector_is_stale = (
                expected_embedding_text is not None
                and actual_metadata.get("embedding_text_hash") != expected["embedding_text_hash"]
            )
            if vector_is_stale:
                if not self._has_embedding_config():
                    result.unrepaired_documents.append(doc_id)
                    continue
                vector = await self._embed_document(expected_embedding_text)
                expected = self._document_metadata(
                    doc,
                    expected_embedding_text=expected_embedding_text,
                    embedding=vector,
                )
                upsert_with_stored_vector_hash(
                    self.document_collection,
                    ids=[doc_id],
                    embeddings=[vector],
                    documents=[expected_embedding_text],
                    metadatas=[expected],
                )
                result.document_vectors_repaired += 1
            elif self._metadata_needs_update(actual_metadata, expected):
                metadata = dict(actual_metadata)
                metadata.update(expected)
                self._update_collection_metadata(
                    self.document_collection,
                    doc_id,
                    metadata,
                    existing,
                )
                result.document_vectors_repaired += 1

    async def _documents(self) -> list[dict[str, Any]]:
        docs: list[dict[str, Any]] = []
        async with self.db.db.execute(
            """SELECT d.*, s.type AS source_type
               FROM documents d
               LEFT JOIN sources s ON s.id = d.source
               ORDER BY d.doc_id"""
        ) as cursor:
            async for row in cursor:
                docs.append(dict(row))
        return docs

    def _document_metadata(
        self,
        doc: dict[str, Any],
        *,
        expected_embedding_text: str | None,
        embedding: Any | None,
    ) -> dict[str, Any]:
        metadata = {
            "source": doc.get("source") or "",
            "source_type": doc.get("source_type") or "",
            "space": doc.get("space_or_project") or "",
            "token_count": int(doc.get("token_count") or 0),
            "content_hash": doc.get("content_hash") or "",
            "version": doc.get("version") or "",
        }
        if expected_embedding_text is not None:
            metadata["embedding_text_hash"] = embedding_text_hash(expected_embedding_text)
        if embedding is not None:
            metadata["embedding_vector_hash"] = embedding_vector_hash(embedding)
        return metadata

    async def _document_embedding_text(self, doc_id: str) -> str | None:
        metadata = await self.db.get_metadata(doc_id)
        if metadata is None:
            return None
        return document_embedding_text(metadata)

    async def _embed_memory(self, memory: Memory) -> list[float]:
        if not self._has_embedding_config():
            raise RuntimeError(f"Cannot repair missing memory vector for {memory.id}: embedding config is missing")
        text = await self._memory_embedding_text(memory)
        return (await asyncio.to_thread(
            embed_texts,
            [text],
            self.embed_cfg["base_url"],
            self.embed_cfg["api_key"],
            self.embed_cfg["model"],
        ))[0]

    async def _memory_embedding_text(self, memory: Memory) -> str:
        entity_names = await self.db.get_memory_entity_names(memory.id)
        return memory_embedding_text(memory, entity_names)

    async def _embed_document(self, text: str) -> list[float]:
        return (await asyncio.to_thread(
            embed_texts,
            [text],
            self.embed_cfg["base_url"],
            self.embed_cfg["api_key"],
            self.embed_cfg["model"],
        ))[0]

    def _has_embedding_config(self) -> bool:
        return bool(
            self.embed_cfg.get("base_url")
            and self.embed_cfg.get("api_key")
            and self.embed_cfg.get("model")
        )

    def _collection_records(self, collection: Any) -> dict[str, dict[str, Any]]:
        raw = collection.get(include=["embeddings", "documents", "metadatas"])
        ids = raw.get("ids") or []
        metadatas = raw.get("metadatas")
        embeddings = raw.get("embeddings")
        documents = raw.get("documents")
        if metadatas is None:
            metadatas = []
        if embeddings is None:
            embeddings = []
        if documents is None:
            documents = []
        records: dict[str, dict[str, Any]] = {}
        for index, record_id in enumerate(ids):
            records[record_id] = {
                "metadata": dict(metadatas[index] if index < len(metadatas) and metadatas[index] else {}),
                "embedding": embeddings[index] if index < len(embeddings) else None,
                "document": documents[index] if index < len(documents) else None,
            }
        return records

    def _metadata_needs_update(self, actual: dict[str, Any], expected: dict[str, Any]) -> bool:
        return any(actual.get(key) != value for key, value in expected.items())

    def _update_collection_metadata(
        self,
        collection: Any,
        record_id: str,
        metadata: dict[str, Any],
        existing: dict[str, Any],
    ) -> None:
        if hasattr(collection, "update"):
            collection.update(ids=[record_id], metadatas=[metadata])
            return
        kwargs: dict[str, Any] = {"ids": [record_id], "metadatas": [metadata]}
        if existing.get("embedding") is not None:
            kwargs["embeddings"] = [existing["embedding"]]
        if existing.get("document") is not None:
            kwargs["documents"] = [existing["document"]]
        collection.upsert(**kwargs)
