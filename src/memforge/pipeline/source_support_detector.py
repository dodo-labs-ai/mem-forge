"""Attach corroborating source support to existing memories.

This stage runs after extraction/reconciliation. It does not create or change
memory content. It only links an existing active memory to the current document
when the document directly supports that memory with a concrete excerpt.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from memforge.llm.structured import SourceSupportStructuredClient, StructuredLlmError
from memforge.models import Memory

if TYPE_CHECKING:
    from memforge.memory.store import MemoryStore
    from memforge.storage.database import Database

logger = logging.getLogger(__name__)

__all__ = ["SourceSupportDetector"]


SOURCE_SUPPORT_PROMPT = """You are verifying whether a source document directly supports existing team memories.

Only mark a memory supported when the document explicitly states the same durable fact, decision, convention, or procedure.
Do not use link-only evidence, metadata rows, vague topic similarity, or inference beyond the document.
supported=true means the excerpt entails the full memory content.
Return supported=false when the source is merely related, narrower, broader, uses different actors, or requires domain inference.

<candidate_memories>
{candidates_json}
</candidate_memories>

<document>
{document}
</document>

Return ONLY a JSON object matching this schema:
{{"decisions": [{{"memory_id": "mem-id", "supported": true, "excerpt": "exact quote from the document", "reason": "short reason"}}]}}

For unsupported candidates, omit them or return supported=false. The excerpt must be copied from the document text."""


@dataclass(frozen=True)
class SourceSupportVerification:
    decisions: list[dict[str, Any]] | None
    error: str | None = None


class SourceSupportDetector:
    """Find existing memories directly supported by the current document."""

    def __init__(
        self,
        *,
        structured_llm_client: SourceSupportStructuredClient | None = None,
        llm_model: str = "claude-sonnet-4-20250514",
        max_candidates: int = 30,
        max_excerpt_chars: int = 500,
    ) -> None:
        self.structured_llm_client = structured_llm_client
        self.llm_model = llm_model
        self.max_candidates = max(1, max_candidates)
        self.max_excerpt_chars = max(80, max_excerpt_chars)

    async def detect_and_persist(
        self,
        *,
        doc_id: str,
        source_type: str,
        document: str,
        entity_ids: list[int],
        project_key: str | None,
        db: Database,
        memory_store: MemoryStore,
        writer_visibility: str | None = None,
        writer_owner_user_id: str | None = None,
        writer_project_key: str | None = None,
    ) -> dict[str, int]:
        """Attach corroborated provenance for supported existing memories."""
        stats = {
            "checked": 0,
            "added": 0,
            "updated": 0,
            "removed_stale": 0,
            "skipped": 0,
        }
        audit_context = memory_store.operation_context(
            doc_id=doc_id,
            model=self.llm_model,
            prompt_hash=self._prompt_hash(),
        )

        existing_sources = await db.get_corroborated_sources_by_doc(doc_id)

        if not entity_ids:
            stats["removed_stale"] += await self._remove_stale_sources(
                memory_store=memory_store,
                audit_context=audit_context,
                doc_id=doc_id,
                sources=existing_sources,
                document=document,
                refreshed_ids=set(),
                remove_ids=set(),
            )
            return stats

        existing_by_id: dict[str, Memory] = {}
        for source in existing_sources:
            memory = await db.get_memory(source.memory_id)
            if memory and memory.status == "active":
                existing_by_id[memory.id] = memory

        if existing_by_id:
            verification = await self._verify_support(
                document=document,
                candidates=list(existing_by_id.values()),
            )
            if verification.decisions is None:
                await self._record_verification_failure(
                    memory_store=memory_store,
                    audit_context=audit_context,
                    doc_id=doc_id,
                    error=verification.error,
                    phase="existing_support_refresh",
                )
                stats["removed_stale"] += await self._remove_stale_sources(
                    memory_store=memory_store,
                    audit_context=audit_context,
                    doc_id=doc_id,
                    sources=existing_sources,
                    document=document,
                    refreshed_ids=set(),
                    remove_ids=set(),
                )
                return stats
            stats["removed_stale"] += await self._apply_existing_decisions(
                decisions=verification.decisions,
                existing_by_id=existing_by_id,
                existing_sources=existing_sources,
                db=db,
                memory_store=memory_store,
                audit_context=audit_context,
                doc_id=doc_id,
                source_type=source_type,
                document=document,
                stats=stats,
                writer_visibility=writer_visibility,
                writer_owner_user_id=writer_owner_user_id,
                writer_project_key=writer_project_key,
            )
        elif existing_sources:
            stats["removed_stale"] += await self._remove_stale_sources(
                memory_store=memory_store,
                audit_context=audit_context,
                doc_id=doc_id,
                sources=existing_sources,
                document=document,
                refreshed_ids=set(),
                remove_ids=set(),
            )

        candidates = await db.get_source_support_candidates(
            doc_id=doc_id,
            entity_ids=entity_ids,
            project_key=project_key,
            limit=self.max_candidates,
            writer_visibility=writer_visibility,
            writer_owner_user_id=writer_owner_user_id,
            writer_project_key=writer_project_key,
        )
        candidates_by_id: dict[str, Memory] = {memory.id: memory for memory in candidates}

        if not candidates_by_id:
            return stats

        verification = await self._verify_support(
            document=document,
            candidates=list(candidates_by_id.values()),
        )
        if verification.decisions is None:
            await self._record_verification_failure(
                memory_store=memory_store,
                audit_context=audit_context,
                doc_id=doc_id,
                error=verification.error,
                phase="candidate_support",
            )
            return stats

        for decision in verification.decisions:
            memory_id = str(decision.get("memory_id") or "")
            if memory_id not in candidates_by_id:
                stats["skipped"] += 1
                await memory_store.record_audit_event(
                    "source_support_rejected",
                    "skipped",
                    context=audit_context,
                    memory_id=memory_id or None,
                    doc_id=doc_id,
                    reason="unknown_candidate",
                    payload={"verifier_reason": decision.get("reason")},
                )
                continue
            if decision.get("supported") is not True:
                await memory_store.record_audit_event(
                    "source_support_rejected",
                    "skipped",
                    context=audit_context,
                    memory_id=memory_id,
                    doc_id=doc_id,
                    reason="unsupported",
                    payload={"verifier_reason": decision.get("reason")},
                )
                continue

            excerpt = str(decision.get("excerpt") or "").strip()
            if len(excerpt) > self.max_excerpt_chars:
                excerpt = excerpt[: self.max_excerpt_chars].rstrip()
            if not self._is_valid_support_excerpt(excerpt, document):
                stats["skipped"] += 1
                await memory_store.record_audit_event(
                    "source_support_rejected",
                    "skipped",
                    context=audit_context,
                    memory_id=memory_id,
                    doc_id=doc_id,
                    reason="invalid_excerpt",
                    payload={"verifier_reason": decision.get("reason")},
                    evidence_refs=[{"excerpt": excerpt}] if excerpt else [],
                )
                continue

            await memory_store.record_audit_event(
                "source_support_verified",
                "committed",
                context=audit_context,
                memory_id=memory_id,
                doc_id=doc_id,
                support_kind="corroborated",
                reason=str(decision.get("reason") or ""),
                evidence_refs=[{"excerpt": excerpt}],
            )
            outcome = await memory_store.add_source_support(
                memory_id,
                doc_id,
                source_type,
                excerpt,
                support_kind="corroborated",
                context=audit_context,
                writer_visibility=writer_visibility,
                writer_owner_user_id=writer_owner_user_id,
                writer_project_key=writer_project_key,
            )
            stats["checked"] += 1
            if outcome == "inserted":
                stats["added"] += 1
            elif outcome == "updated":
                stats["updated"] += 1

        return stats

    async def _apply_existing_decisions(
        self,
        *,
        decisions: list[dict[str, Any]],
        existing_by_id: dict[str, Memory],
        existing_sources: list,
        db: Database,
        memory_store: MemoryStore,
        audit_context,
        doc_id: str,
        source_type: str,
        document: str,
        stats: dict[str, int],
        writer_visibility: str | None = None,
        writer_owner_user_id: str | None = None,
        writer_project_key: str | None = None,
    ) -> int:
        refreshed_existing_ids: set[str] = set()
        remove_ids: set[str] = set()

        for decision in decisions:
            memory_id = str(decision.get("memory_id") or "")
            if memory_id not in existing_by_id:
                stats["skipped"] += 1
                await memory_store.record_audit_event(
                    "source_support_rejected",
                    "skipped",
                    context=audit_context,
                    memory_id=memory_id or None,
                    doc_id=doc_id,
                    reason="unknown_existing_candidate",
                    payload={"verifier_reason": decision.get("reason")},
                )
                continue
            if decision.get("supported") is False:
                await memory_store.record_audit_event(
                    "source_support_rejected",
                    "skipped",
                    context=audit_context,
                    memory_id=memory_id,
                    doc_id=doc_id,
                    reason="unsupported",
                    payload={"verifier_reason": decision.get("reason")},
                )
                remove_ids.add(memory_id)
                continue
            excerpt = str(decision.get("excerpt") or "").strip()
            if len(excerpt) > self.max_excerpt_chars:
                excerpt = excerpt[: self.max_excerpt_chars].rstrip()
            if not self._is_valid_support_excerpt(excerpt, document):
                stats["skipped"] += 1
                await memory_store.record_audit_event(
                    "source_support_rejected",
                    "skipped",
                    context=audit_context,
                    memory_id=memory_id,
                    doc_id=doc_id,
                    reason="invalid_excerpt",
                    payload={"verifier_reason": decision.get("reason")},
                    evidence_refs=[{"excerpt": excerpt}] if excerpt else [],
                )
                continue

            await memory_store.record_audit_event(
                "source_support_verified",
                "committed",
                context=audit_context,
                memory_id=memory_id,
                doc_id=doc_id,
                support_kind="corroborated",
                reason=str(decision.get("reason") or ""),
                evidence_refs=[{"excerpt": excerpt}],
            )
            outcome = await memory_store.add_source_support(
                memory_id,
                doc_id,
                source_type,
                excerpt,
                support_kind="corroborated",
                context=audit_context,
                writer_visibility=writer_visibility,
                writer_owner_user_id=writer_owner_user_id,
                writer_project_key=writer_project_key,
            )
            stats["checked"] += 1
            if outcome == "updated":
                stats["updated"] += 1
            elif outcome == "inserted":
                stats["added"] += 1
            refreshed_existing_ids.add(memory_id)

        return await self._remove_stale_sources(
            memory_store=memory_store,
            audit_context=audit_context,
            doc_id=doc_id,
            sources=existing_sources,
            document=document,
            refreshed_ids=refreshed_existing_ids,
            remove_ids=remove_ids,
        )

    async def _verify_support(
        self,
        *,
        document: str,
        candidates: list[Memory],
    ) -> SourceSupportVerification:
        candidates_json = json.dumps(
            [
                {
                    "memory_id": memory.id,
                    "content": memory.content,
                    "memory_type": memory.memory_type,
                    "tags": memory.tags,
                    "confidence": memory.confidence,
                    "corroboration_count": memory.corroboration_count,
                }
                for memory in candidates
            ],
            indent=2,
        )
        prompt = SOURCE_SUPPORT_PROMPT.format(
            candidates_json=candidates_json,
            document=document[:100_000],
        )

        if self.structured_llm_client is None:
            return SourceSupportVerification(
                decisions=None,
                error="structured source-support LLM unavailable",
            )

        try:
            response = await self.structured_llm_client.verify_source_support(prompt)
        except StructuredLlmError as exc:
            logger.warning("Source support detection failed: %s", exc)
            return SourceSupportVerification(decisions=None, error=str(exc))
        except Exception as exc:
            logger.warning("Unexpected source support detection failure: %s", exc)
            return SourceSupportVerification(decisions=None, error=str(exc))
        return SourceSupportVerification(decisions=[decision.model_dump() for decision in response.decisions])

    async def _record_verification_failure(
        self,
        *,
        memory_store: MemoryStore,
        audit_context,
        doc_id: str,
        error: str | None,
        phase: str,
    ) -> None:
        await memory_store.record_audit_event(
            "source_support_verification_failed",
            "failed",
            context=audit_context,
            doc_id=doc_id,
            reason=phase,
            payload_class="llm_response_error",
            error=error,
        )

    @staticmethod
    def _prompt_hash() -> str:
        return hashlib.sha256(SOURCE_SUPPORT_PROMPT.encode("utf-8")).hexdigest()

    async def _remove_stale_sources(
        self,
        *,
        memory_store: MemoryStore,
        audit_context=None,
        doc_id: str,
        sources: list,
        document: str,
        refreshed_ids: set[str],
        remove_ids: set[str],
    ) -> int:
        removed = 0
        for source in sources:
            if source.memory_id in refreshed_ids:
                continue
            if source.memory_id not in remove_ids and source.excerpt and self._excerpt_in_document(source.excerpt, document):
                continue
            await self._remove_source_support(memory_store, source.memory_id, doc_id, context=audit_context)
            removed += 1
        return removed

    @staticmethod
    async def _remove_source_support(
        memory_store: MemoryStore,
        memory_id: str,
        doc_id: str,
        *,
        context=None,
    ) -> None:
        await memory_store.remove_source_support(memory_id, doc_id, reason="no_support", context=context)

    def _is_valid_support_excerpt(self, excerpt: str, document: str) -> bool:
        if not excerpt:
            return False
        if not self._excerpt_in_document(excerpt, document):
            return False
        if self._is_link_only(excerpt):
            return False
        if self._is_metadata_only(excerpt):
            return False
        return True

    @staticmethod
    def _excerpt_in_document(excerpt: str, document: str) -> bool:
        return _normalize_ws(excerpt) in _normalize_ws(document)

    @staticmethod
    def _is_link_only(excerpt: str) -> bool:
        stripped = excerpt.strip()
        return bool(re.fullmatch(r"(https?://\S+|www\.\S+)", stripped))

    @staticmethod
    def _is_metadata_only(excerpt: str) -> bool:
        stripped = excerpt.strip()
        if len(stripped.split()) > 8:
            return False
        metadata_patterns = (
            r"^(author|created|updated|status|priority|assignee|reporter)\s*[:=]",
            r"^last\s+(modified|updated)\s*[:=]",
        )
        lower = stripped.lower()
        return any(re.search(pattern, lower) for pattern in metadata_patterns)


def _normalize_ws(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
