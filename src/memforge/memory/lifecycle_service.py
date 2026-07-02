"""User-facing memory lifecycle orchestration.

This service is the boundary for MCP/API lifecycle intents. Routes and MCP
tools should say what the user wants; this service decides how to perform the
structured lifecycle transition without raw status mutation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from memforge.memory.store import MemoryStore
from memforge.models import (
    DocumentRecord,
    Memory,
    MemoryType,
    ReplacementKind,
    UNSORTED_PROJECT_KEY,
    Visibility,
    content_hash,
    generate_memory_id,
)
from memforge.storage.database import Database


class MemoryLifecycleError(Exception):
    """Base class for lifecycle-service errors."""


class MemoryLifecycleNotFound(MemoryLifecycleError):
    """Raised when the target memory does not exist."""


class MemoryLifecycleConflict(MemoryLifecycleError):
    """Raised when the requested lifecycle action is stale or conflicts."""


@dataclass(frozen=True)
class RetireMemoryResult:
    memory_id: str
    status: str


@dataclass(frozen=True)
class ReplaceMemoryResult:
    memory_id: str
    replacement_memory_id: str
    status: str
    replacement_kind: ReplacementKind


@dataclass(frozen=True)
class CreateMemoryResult:
    memory_id: str
    status: str


class MemoryLifecycleService:
    """Apply user-confirmed memory lifecycle actions through store primitives."""

    def __init__(self, *, db: Database, memory_store: MemoryStore) -> None:
        self.db = db
        self.memory_store = memory_store

    async def create_memory(
        self,
        *,
        content: str,
        provenance: str,
        owner_user_id: str,
        client: str,
        memory_type: str = MemoryType.FACT.value,
        tags: list[str] | None = None,
        confidence: float = 0.95,
        repo_identifier: str | None = None,
        idempotency_key: str | None = None,
    ) -> CreateMemoryResult:
        content = content.strip()
        provenance = provenance.strip() if provenance else None
        if not content:
            raise MemoryLifecycleConflict("content_required")
        if not provenance:
            raise MemoryLifecycleConflict("provenance_required")
        if not owner_user_id.strip():
            raise MemoryLifecycleConflict("owner_user_id_required")
        memory_type = self._validate_memory_type(memory_type)
        normalized_tags = [tag.strip() for tag in (tags or []) if tag.strip()]

        now = datetime.now(timezone.utc)
        memory = Memory(
            id=generate_memory_id(),
            memory_type=memory_type,
            content=content,
            content_hash=content_hash(content),
            visibility=Visibility.PRIVATE.value,
            owner_user_id=owner_user_id.strip(),
            project_key=UNSORTED_PROJECT_KEY,
            repo_identifier=repo_identifier.strip() if repo_identifier else None,
            tags=normalized_tags,
            confidence=confidence,
            created_at=now,
            updated_at=now,
            status="active",
            extraction_context=provenance,
        )
        doc_id = self._user_memory_doc_id(memory.id, idempotency_key=idempotency_key)
        await self._write_user_memory_document(
            doc_id=doc_id,
            memory=memory,
            provenance=provenance,
            client=client,
            observed_at=now,
        )
        status = await self.memory_store.deduplicate_and_insert(
            memory,
            doc_id,
            "user_memory",
            source_updated_at=now,
            excerpt=content,
        )
        memory_id = memory.id
        if status != "inserted":
            memory_ids = await self.db.get_memory_ids_for_doc(doc_id)
            if memory_ids:
                memory_id = memory_ids[0]
        return CreateMemoryResult(memory_id=memory_id, status=status)

    async def retire_memory(
        self,
        memory_id: str,
        *,
        reason: str,
        expected_content_hash: str,
    ) -> RetireMemoryResult:
        memory = await self._active_target(memory_id, expected_content_hash=expected_content_hash)
        await self.memory_store.retire_memory(memory.id, reason=reason)
        return RetireMemoryResult(memory_id=memory.id, status="retired")

    async def replace_memory(
        self,
        memory_id: str,
        *,
        replacement_content: str,
        provenance: str,
        reason: str,
        expected_content_hash: str,
        replacement_kind: ReplacementKind = "supersession",
    ) -> ReplaceMemoryResult:
        replacement_kind = self._validate_replacement_kind(replacement_kind)
        replacement_content = replacement_content.strip()
        if not replacement_content:
            raise MemoryLifecycleConflict("replacement_content_required")
        provenance = provenance.strip() if provenance else None
        if not provenance:
            raise MemoryLifecycleConflict("provenance_required")

        old = await self._active_target(memory_id, expected_content_hash=expected_content_hash)
        now = datetime.now(timezone.utc)
        new_memory = Memory(
            id=generate_memory_id(),
            memory_type=old.memory_type,
            content=replacement_content,
            content_hash=content_hash(replacement_content),
            visibility=old.visibility,
            owner_user_id=old.owner_user_id,
            project_key=old.project_key,
            repo_identifier=old.repo_identifier,
            tags=list(old.tags),
            confidence=old.confidence,
            created_at=now,
            updated_at=now,
            status="active",
        )

        claim = await self.db.get_agent_claim_by_memory_id(old.id)
        if claim is not None:
            await self.memory_store.supersede_agent_claim_memory(
                old.id,
                new_memory,
                claim["concept_id"],
                "agent_session",
                replacement_kind=replacement_kind,
                claim_id=claim["id"],
                concept_id=claim["concept_id"],
                display_anchor=claim["display_anchor"],
                claim_text=replacement_content,
                memory_type=new_memory.memory_type,
                tags=list(new_memory.tags),
                confidence=new_memory.confidence,
                observed_at=now,
                source_updated_at=now,
                excerpt=replacement_content,
                replacement_reason=reason,
            )
        else:
            correction_doc_id = f"correction-{new_memory.id}"
            await self._write_correction_document(
                doc_id=correction_doc_id,
                old_memory=old,
                replacement_content=replacement_content,
                provenance=provenance,
                reason=reason,
                replacement_kind=replacement_kind,
                observed_at=now,
            )
            await self.memory_store.supersede_memory(
                old.id,
                new_memory,
                correction_doc_id,
                "user_correction",
                replacement_kind=replacement_kind,
                replacement_reason=reason,
                source_updated_at=now,
                excerpt=replacement_content,
                carry_revision_sources=False,
            )

        return ReplaceMemoryResult(
            memory_id=old.id,
            replacement_memory_id=new_memory.id,
            status="superseded",
            replacement_kind=replacement_kind,
        )

    async def _active_target(self, memory_id: str, *, expected_content_hash: str) -> Memory:
        memory = await self.db.get_memory(memory_id)
        if memory is None:
            raise MemoryLifecycleNotFound("memory_not_found")
        if memory.status != "active":
            raise MemoryLifecycleConflict("memory_not_active")
        if memory.content_hash != expected_content_hash:
            raise MemoryLifecycleConflict("content_hash_mismatch")
        return memory

    async def _write_correction_document(
        self,
        *,
        doc_id: str,
        old_memory: Memory,
        replacement_content: str,
        provenance: str,
        reason: str,
        replacement_kind: ReplacementKind,
        observed_at: datetime,
    ) -> None:
        lines = [
            f"Target memory: {old_memory.id}",
            f"Replacement kind: {replacement_kind}",
            f"Reason: {reason}",
        ]
        lines.extend(["", "Provenance:", provenance])
        lines.extend(["", "Replacement content:", replacement_content])
        document_body = "\n".join(lines)
        await self.db.upsert_document(
            DocumentRecord(
                doc_id=doc_id,
                source="user_correction",
                source_url=f"memforge://memory-corrections/{doc_id}",
                title=f"User correction for {old_memory.id}",
                space_or_project=old_memory.project_key or "UNSORTED",
                author=None,
                last_modified=observed_at,
                labels=["user_correction"],
                version=content_hash(document_body),
                content_hash=content_hash(document_body),
                token_count=len(document_body.split()),
                raw_content_uri=None,
                raw_content_type=None,
                normalized_content_uri=None,
                pdf_content_uri=None,
                last_synced=observed_at,
            )
        )

    async def _write_user_memory_document(
        self,
        *,
        doc_id: str,
        memory: Memory,
        provenance: str,
        client: str,
        observed_at: datetime,
    ) -> None:
        document_body = "\n".join(
            [
                f"Client: {client}",
                "",
                "Provenance:",
                provenance,
                "",
                "Memory:",
                memory.content,
            ]
        )
        await self.db.upsert_document(
            DocumentRecord(
                doc_id=doc_id,
                source="user_memory",
                source_url=f"memforge://user-memory/{doc_id}",
                title=f"User memory {memory.id}",
                space_or_project=memory.project_key or UNSORTED_PROJECT_KEY,
                author=memory.owner_user_id,
                last_modified=observed_at,
                labels=["user_memory"],
                version=content_hash(document_body),
                content_hash=content_hash(document_body),
                token_count=len(document_body.split()),
                raw_content_uri=None,
                raw_content_type=None,
                normalized_content_uri=None,
                pdf_content_uri=None,
                last_synced=observed_at,
                client=client,
            )
        )

    @staticmethod
    def _user_memory_doc_id(memory_id: str, *, idempotency_key: str | None) -> str:
        if idempotency_key:
            return f"user-memory-{content_hash(idempotency_key)[:16]}"
        return f"user-memory-{memory_id}"

    @staticmethod
    def _validate_replacement_kind(value: str) -> ReplacementKind:
        if value not in {"revision", "supersession"}:
            raise MemoryLifecycleConflict("invalid_replacement_kind")
        return value  # type: ignore[return-value]

    @staticmethod
    def _validate_memory_type(value: str) -> str:
        allowed = {item.value for item in MemoryType}
        if value not in allowed:
            raise MemoryLifecycleConflict("invalid_memory_type")
        return value
