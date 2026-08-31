"""User-facing memory lifecycle orchestration.

This service is the boundary for MCP/API lifecycle intents. Routes and MCP
tools should say what the user wants; this service decides how to perform the
structured lifecycle transition without raw status mutation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from memforge.agent_knowledge import (
    AgentClaimLifecycleConflict,
    AgentKnowledgeBundleService,
)
from memforge.memory.audit import MemoryAuditEvent
from memforge.memory.correction_authority import CorrectionAuthority
from memforge.memory.store import MemoryStore
from memforge.models import (
    DocumentRecord,
    Memory,
    MemoryReview,
    MemoryType,
    ReviewKind,
    ReviewStatus,
    ReplacementKind,
    UNSORTED_PROJECT_KEY,
    Visibility,
    content_hash,
    generate_deterministic_review_id,
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


@dataclass(frozen=True, slots=True)
class MaintenanceClosureEntry:
    memory_id: str
    expected_content_hash: str
    reason: str


@dataclass(frozen=True, slots=True)
class MaintenanceClosureReportEntry:
    memory_id: str
    disposition: str


@dataclass(frozen=True, slots=True)
class MaintenanceClosureReport:
    id: str
    entries: tuple[MaintenanceClosureReportEntry, ...]

    @property
    def ready_count(self) -> int:
        return sum(item.disposition in {"ready", "already_closed"} for item in self.entries)


@dataclass(frozen=True, slots=True)
class MaintenanceClosureReceipt:
    id: str
    report_id: str
    memory_id: str
    status: str
    authority: str
    operator_actor_id: str


@dataclass(frozen=True)
class _ReplaceOwnedMemoryResult:
    memory_id: str
    replacement_memory_id: str
    status: str
    replacement_kind: ReplacementKind


@dataclass(frozen=True)
class CreateMemoryResult:
    memory_id: str
    status: str


@dataclass(frozen=True)
class ProposeMemoryCorrectionResult:
    memory_id: str
    replacement_memory_id: str
    outcome: Literal["applied", "review_created"]
    status: str
    replacement_kind: ReplacementKind
    review_id: str


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
            excerpt=provenance,
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
        actor_user_id: str,
    ) -> RetireMemoryResult:
        actor_user_id = actor_user_id.strip()
        if not actor_user_id:
            raise MemoryLifecycleConflict("actor_user_id_required")
        memory = await self._active_target(memory_id, expected_content_hash=expected_content_hash)
        if memory.visibility == Visibility.PRIVATE.value and memory.owner_user_id != actor_user_id:
            raise MemoryLifecycleConflict("memory_owner_authority_required")
        claim = await self.db.get_agent_claim_by_memory_id(memory.id)
        if claim is not None:
            try:
                await AgentKnowledgeBundleService(
                    db=self.db,
                    memory_store=self.memory_store,
                ).retire_claim_from_user_request(
                    old_memory_id=memory.id,
                    reason=reason,
                    observed_at=datetime.now(timezone.utc),
                )
            except AgentClaimLifecycleConflict as exc:
                raise MemoryLifecycleConflict(exc.code) from exc
            return RetireMemoryResult(memory_id=memory.id, status="retired")
        support_state = (await self.db.get_active_memory_support_states((memory.id,)))[memory.id]
        if support_state.support_ids:
            raise MemoryLifecycleConflict("source_backed_memory_requires_lifecycle_review")
        try:
            await self.memory_store.retire_memory(memory.id, reason=reason)
        except ValueError as exc:
            if "active source support" in str(exc):
                raise MemoryLifecycleConflict("source_backed_memory_requires_lifecycle_review") from exc
            raise
        return RetireMemoryResult(memory_id=memory.id, status="retired")

    async def report_maintenance_closures(
        self,
        manifest: tuple[MaintenanceClosureEntry, ...],
    ) -> MaintenanceClosureReport:
        normalized = self._normalize_maintenance_manifest(manifest)
        entries: list[MaintenanceClosureReportEntry] = []
        for item in normalized:
            memory = await self.db.get_memory(item.memory_id)
            disposition = "ready"
            if memory is None:
                disposition = "not_found"
            elif memory.content_hash != item.expected_content_hash:
                disposition = "content_hash_mismatch"
            elif memory.status == "retired" and memory.retirement_reason == item.reason:
                disposition = "already_closed"
            elif memory.status != "active":
                disposition = "not_active"
            elif memory.visibility != Visibility.PRIVATE.value:
                disposition = "not_private"
            else:
                support_state = (await self.db.get_active_memory_support_states((memory.id,)))[memory.id]
                if support_state.support_ids:
                    disposition = "active_support"
            entries.append(
                MaintenanceClosureReportEntry(
                    memory_id=item.memory_id,
                    disposition=disposition,
                )
            )
        payload = [
            {
                "memory_id": item.memory_id,
                "expected_content_hash": item.expected_content_hash,
                "reason": item.reason,
                "disposition": (
                    "ready" if report_item.disposition in {"ready", "already_closed"} else report_item.disposition
                ),
            }
            for item, report_item in zip(normalized, entries, strict=True)
        ]
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        return MaintenanceClosureReport(
            id=f"maintenance-closure-report-{digest}",
            entries=tuple(entries),
        )

    async def apply_maintenance_closures(
        self,
        manifest: tuple[MaintenanceClosureEntry, ...],
        *,
        expected_report_id: str,
        operator_actor_id: str,
    ) -> tuple[MaintenanceClosureReceipt, ...]:
        operator_actor_id = operator_actor_id.strip()
        if not operator_actor_id:
            raise MemoryLifecycleConflict("operator_actor_id_required")
        normalized = self._normalize_maintenance_manifest(manifest)
        report = await self.report_maintenance_closures(normalized)
        if report.id != expected_report_id:
            raise MemoryLifecycleConflict("maintenance_closure_report_mismatch")
        blocked = [item for item in report.entries if item.disposition not in {"ready", "already_closed"}]
        if blocked:
            raise MemoryLifecycleConflict("maintenance_closure_manifest_not_ready")

        receipts: list[MaintenanceClosureReceipt] = []
        report_by_id = {item.memory_id: item for item in report.entries}
        for item in normalized:
            status = "already_retired"
            if report_by_id[item.memory_id].disposition == "ready":
                memory = await self._active_target(
                    item.memory_id,
                    expected_content_hash=item.expected_content_hash,
                )
                claim = await self.db.get_agent_claim_by_memory_id(memory.id)
                if claim is not None:
                    try:
                        await AgentKnowledgeBundleService(
                            db=self.db,
                            memory_store=self.memory_store,
                        ).retire_claim_from_maintenance_operator(
                            old_memory_id=memory.id,
                            reason=item.reason,
                            observed_at=datetime.now(timezone.utc),
                        )
                    except AgentClaimLifecycleConflict as exc:
                        raise MemoryLifecycleConflict(exc.code) from exc
                else:
                    await self.memory_store.retire_memory(memory.id, reason=item.reason)
                status = "retired"
            receipt_id = (
                "maintenance-closure-receipt-"
                + hashlib.sha256(f"{report.id}\x1f{item.memory_id}".encode("utf-8")).hexdigest()
            )
            await self.db.insert_memory_audit_event(
                MemoryAuditEvent(
                    event_type="maintenance_memory_closure_committed",
                    status="committed",
                    actor_type="maintenance_operator",
                    actor_id=operator_actor_id,
                    memory_id=item.memory_id,
                    reason=item.reason,
                    payload={
                        "authority": "maintenance_operator",
                        "report_id": report.id,
                        "receipt_id": receipt_id,
                        "result": status,
                    },
                )
            )
            receipts.append(
                MaintenanceClosureReceipt(
                    id=receipt_id,
                    report_id=report.id,
                    memory_id=item.memory_id,
                    status=status,
                    authority="maintenance_operator",
                    operator_actor_id=operator_actor_id,
                )
            )
        return tuple(receipts)

    @staticmethod
    def _normalize_maintenance_manifest(
        manifest: tuple[MaintenanceClosureEntry, ...],
    ) -> tuple[MaintenanceClosureEntry, ...]:
        if not manifest:
            raise MemoryLifecycleConflict("maintenance_closure_manifest_required")
        normalized: list[MaintenanceClosureEntry] = []
        seen: set[str] = set()
        for item in manifest:
            memory_id = item.memory_id.strip()
            expected_hash = item.expected_content_hash.strip()
            reason = item.reason.strip()
            if not memory_id or not expected_hash or not reason:
                raise MemoryLifecycleConflict("maintenance_closure_entry_incomplete")
            if memory_id in seen:
                raise MemoryLifecycleConflict("maintenance_closure_duplicate_memory")
            seen.add(memory_id)
            normalized.append(
                MaintenanceClosureEntry(
                    memory_id=memory_id,
                    expected_content_hash=expected_hash,
                    reason=reason,
                )
            )
        return tuple(normalized)

    async def _replace_owned_memory(
        self,
        memory_id: str,
        *,
        replacement_content: str,
        provenance: str,
        reason: str,
        expected_content_hash: str,
        replacement_kind: ReplacementKind = "supersession",
    ) -> _ReplaceOwnedMemoryResult:
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
            confidence=old.confidence,
            created_at=now,
            updated_at=now,
            status="active",
            extraction_context=provenance,
        )

        claim = await self.db.get_agent_claim_by_memory_id(old.id)
        if claim is not None:
            try:
                replacement_id = await AgentKnowledgeBundleService(
                    db=self.db,
                    memory_store=self.memory_store,
                ).replace_claim_from_user_correction(
                    old_memory_id=old.id,
                    replacement_content=replacement_content,
                    provenance=provenance,
                    reason=reason,
                    replacement_kind=replacement_kind,
                    observed_at=now,
                )
            except AgentClaimLifecycleConflict as exc:
                raise MemoryLifecycleConflict(exc.code) from exc
            new_memory.id = replacement_id
        else:
            support_state = (await self.db.get_active_memory_support_states((old.id,)))[old.id]
            if support_state.support_ids:
                raise MemoryLifecycleConflict("source_backed_memory_requires_lifecycle_review")
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
            try:
                await self.memory_store.supersede_memory(
                    old.id,
                    new_memory,
                    correction_doc_id,
                    "user_correction",
                    replacement_kind=replacement_kind,
                    replacement_reason=reason,
                    source_updated_at=now,
                    excerpt=provenance,
                    carry_revision_sources=False,
                )
            except ValueError as exc:
                # A concurrent projected write may attach support after the
                # preflight.  Remove the not-yet-authoritative correction
                # document and return the same explicit conflict.
                await self.db.delete_document(correction_doc_id)
                if "active source support" in str(exc):
                    raise MemoryLifecycleConflict("source_backed_memory_requires_lifecycle_review") from exc
                raise

        return _ReplaceOwnedMemoryResult(
            memory_id=old.id,
            replacement_memory_id=new_memory.id,
            status="superseded",
            replacement_kind=replacement_kind,
        )

    async def propose_memory_correction(
        self,
        memory_id: str,
        *,
        replacement_content: str,
        provenance: str,
        reason: str,
        expected_content_hash: str,
        authority: CorrectionAuthority,
        replacement_kind: ReplacementKind = "supersession",
    ) -> ProposeMemoryCorrectionResult:
        """Apply an authorized correction or stage it for an authorized reviewer.

        Every ordinary correction first becomes a hidden challenger plus a
        stale-guarded supersede Review.  A caller with authority over the full
        active Support Set resolves that Review immediately; other visible
        callers leave the same Review pending for a Source manager.  Managed
        Capture claims keep using their projection-owned correction path.
        """

        replacement_kind = self._validate_replacement_kind(replacement_kind)
        replacement_content = replacement_content.strip()
        provenance = provenance.strip() if provenance else None
        reason = reason.strip()
        actor_user_id = authority.actor_user_id.strip()
        if not replacement_content:
            raise MemoryLifecycleConflict("replacement_content_required")
        if not provenance:
            raise MemoryLifecycleConflict("provenance_required")
        if not reason:
            raise MemoryLifecycleConflict("reason_required")
        if not actor_user_id:
            raise MemoryLifecycleConflict("actor_user_id_required")

        old = await self._active_target(memory_id, expected_content_hash=expected_content_hash)
        claim = await self.db.get_agent_claim_by_memory_id(old.id)
        if claim is not None:
            if old.owner_user_id != actor_user_id:
                raise MemoryLifecycleNotFound("memory_not_found")
            result = await self._replace_owned_memory(
                old.id,
                replacement_content=replacement_content,
                provenance=provenance,
                reason=reason,
                expected_content_hash=expected_content_hash,
                replacement_kind=replacement_kind,
            )
            return ProposeMemoryCorrectionResult(
                memory_id=result.memory_id,
                replacement_memory_id=result.replacement_memory_id,
                outcome="applied",
                status=result.status,
                replacement_kind=result.replacement_kind,
                review_id="",
            )

        support_state = (await self.db.get_active_memory_support_states((old.id,)))[old.id]
        expected_support_set_hash = support_state.support_set_hash if support_state.support_ids else None
        legacy_configured_source_ids: tuple[str, ...] = ()
        if not support_state.support_ids:
            memory_sources = await self.db.get_memory_sources(old.id)
            legacy_configured_source_ids = tuple(
                sorted(
                    {
                        source.source_id
                        for source in memory_sources
                        if source.source_type not in {"user_memory", "user_correction"}
                    }
                )
            )
        can_apply = await self._can_apply_correction(
            old,
            authority=authority,
            supporting_source_ids=support_state.source_ids,
            legacy_configured_source_ids=legacy_configured_source_ids,
        )
        if not can_apply and not support_state.support_ids and not legacy_configured_source_ids:
            raise MemoryLifecycleConflict("workspace_memory_correction_requires_management_authority")

        now = datetime.now(timezone.utc)
        challenger = Memory(
            id=generate_memory_id(),
            memory_type=old.memory_type,
            content=replacement_content,
            content_hash=content_hash(replacement_content),
            visibility=old.visibility,
            owner_user_id=old.owner_user_id,
            project_key=old.project_key,
            repo_identifier=old.repo_identifier,
            confidence=old.confidence,
            created_at=now,
            updated_at=now,
            status="active" if can_apply else "pending_review",
            extraction_context=provenance,
        )
        correction_doc_id = f"correction-{challenger.id}"
        review = MemoryReview(
            id=generate_deterministic_review_id(
                kind=ReviewKind.SUPERSEDE.value,
                incumbent_memory_id=old.id,
                challenger_memory_id=challenger.id,
                review_case="user_correction",
            ),
            kind=ReviewKind.SUPERSEDE.value,
            status=ReviewStatus.PENDING.value,
            incumbent_memory_id=old.id,
            challenger_memory_id=challenger.id,
            reason=reason,
            expected_incumbent_updated_at=(old.updated_at.isoformat() if old.updated_at else None),
            expected_challenger_updated_at=now.isoformat(),
            expected_support_set_hash=expected_support_set_hash,
            replacement_kind=replacement_kind,
            created_at=now,
        )
        correction_document = self._build_correction_document(
            doc_id=correction_doc_id,
            old_memory=old,
            replacement_content=replacement_content,
            provenance=provenance,
            reason=reason,
            replacement_kind=replacement_kind,
            observed_at=now,
        )
        if not can_apply:
            await self.db.upsert_document(correction_document)
            try:
                await self.memory_store.insert_memory(
                    challenger,
                    correction_doc_id,
                    "user_correction",
                    source_updated_at=now,
                    excerpt=provenance,
                    review=review,
                )
            except Exception:
                await self.db.delete_document(correction_doc_id)
                raise
            return ProposeMemoryCorrectionResult(
                memory_id=old.id,
                replacement_memory_id=challenger.id,
                outcome="review_created",
                status="pending",
                replacement_kind=replacement_kind,
                review_id=review.id,
            )

        try:
            await self.memory_store.apply_authorized_memory_correction(
                document=correction_document,
                incumbent=old,
                challenger=challenger,
                review=review,
                reviewer=actor_user_id,
                review_note=reason,
            )
        except ValueError as exc:
            if "support set changed" in str(exc):
                raise MemoryLifecycleConflict("memory_correction_support_set_changed") from exc
            if "active source support" in str(exc):
                raise MemoryLifecycleConflict("source_backed_memory_requires_lifecycle_review") from exc
            raise MemoryLifecycleConflict("memory_correction_target_changed") from exc
        return ProposeMemoryCorrectionResult(
            memory_id=old.id,
            replacement_memory_id=challenger.id,
            outcome="applied",
            status="superseded",
            replacement_kind=replacement_kind,
            review_id=review.id,
        )

    async def _can_apply_correction(
        self,
        memory: Memory,
        *,
        authority: CorrectionAuthority,
        supporting_source_ids: tuple[str, ...],
        legacy_configured_source_ids: tuple[str, ...],
    ) -> bool:
        if supporting_source_ids:
            for source_id in supporting_source_ids:
                source = await self.db.get_source(source_id)
                if source is None:
                    raise MemoryLifecycleConflict("source_backed_memory_lineage_incomplete")
                if not authority.can_manage_source(source):
                    return False
            return True

        if legacy_configured_source_ids:
            return False
        if memory.visibility == Visibility.PRIVATE.value:
            if memory.owner_user_id != authority.actor_user_id:
                raise MemoryLifecycleNotFound("memory_not_found")
            return True
        return authority.can_manage_workspace_memory()

    async def _active_target(self, memory_id: str, *, expected_content_hash: str) -> Memory:
        memory = await self.db.get_memory(memory_id)
        if memory is None:
            raise MemoryLifecycleNotFound("memory_not_found")
        if memory.status != "active":
            raise MemoryLifecycleConflict("memory_not_active")
        if memory.content_hash != expected_content_hash:
            raise MemoryLifecycleConflict("content_hash_mismatch")
        return memory

    def _build_correction_document(
        self,
        *,
        doc_id: str,
        old_memory: Memory,
        replacement_content: str,
        provenance: str,
        reason: str,
        replacement_kind: ReplacementKind,
        observed_at: datetime,
    ) -> DocumentRecord:
        lines = [
            f"Target memory: {old_memory.id}",
            f"Replacement kind: {replacement_kind}",
            f"Reason: {reason}",
        ]
        lines.extend(["", "Provenance:", provenance])
        lines.extend(["", "Replacement content:", replacement_content])
        document_body = "\n".join(lines)
        return DocumentRecord(
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

    async def _write_correction_document(
        self,
        **kwargs,
    ) -> None:
        await self.db.upsert_document(self._build_correction_document(**kwargs))

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
