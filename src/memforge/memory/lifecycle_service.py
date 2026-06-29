"""User-facing memory lifecycle orchestration.

This service is the boundary for MCP/API lifecycle intents. Routes and MCP
tools should say what the user wants; this service decides how to perform the
structured lifecycle transition without raw status mutation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from memforge.memory.store import MemoryStore
from memforge.models import DocumentRecord, Memory, ReplacementKind, content_hash, generate_memory_id
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


class MemoryLifecycleService:
    """Apply user-confirmed memory lifecycle actions through store primitives."""

    def __init__(self, *, db: Database, memory_store: MemoryStore) -> None:
        self.db = db
        self.memory_store = memory_store

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
        reason: str,
        expected_content_hash: str,
        replacement_kind: ReplacementKind = "supersession",
    ) -> ReplaceMemoryResult:
        replacement_kind = self._validate_replacement_kind(replacement_kind)
        replacement_content = replacement_content.strip()
        if not replacement_content:
            raise MemoryLifecycleConflict("replacement_content_required")

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
        reason: str,
        replacement_kind: ReplacementKind,
        observed_at: datetime,
    ) -> None:
        document_body = "\n".join(
            [
                f"Target memory: {old_memory.id}",
                f"Replacement kind: {replacement_kind}",
                f"Reason: {reason}",
                "",
                replacement_content,
            ]
        )
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

    @staticmethod
    def _validate_replacement_kind(value: str) -> ReplacementKind:
        if value not in {"revision", "supersession"}:
            raise MemoryLifecycleConflict("invalid_replacement_kind")
        return value  # type: ignore[return-value]
