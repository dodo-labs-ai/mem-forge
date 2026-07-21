from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest
import pytest_asyncio

from memforge.llm.structured import (
    CandidateLedgerDecision,
    CandidateLedgerResponse,
    ContradictionDecision,
    ContradictionResponse,
    MemoryEquivalenceResponse,
    MemorySupportValidationResponse,
    ReconciliationDecision,
    ReconciliationResponse,
)
from memforge.memory.audit import MemoryAuditLogger
from memforge.memory.engine import (
    MEMORY_EQUIVALENCE_PROMPT,
    MemoryEngine,
    _memory_equivalence_pair_json,
)
from memforge.memory.evidence import (
    EvidenceContentProvenance,
    EvidenceReference,
    EvidenceRole,
    EvidenceUnit,
    MemorySupportAssertion,
)
from memforge.memory.lifecycle_plan import (
    CoverageProof,
    CutoverFindingReason,
    CutoverFindingStatus,
    LifecycleCutoverFinding,
    LifecycleGateState,
    LifecycleMutation,
    LifecycleMutationType,
    LifecyclePlan,
    LifecycleReviewStatus,
    LifecycleVectorDeliveryResult,
    LifecycleVectorDeliveryState,
    LifecycleVectorOperation,
    LifecycleVectorTaskStatus,
    ReconciliationScope,
    StaleGuard,
)
from memforge.memory.lifecycle_planner import (
    NewMemoryDefaults,
    build_lifecycle_plan,
    lifecycle_access_context_hash,
    lifecycle_plan_id,
)
from memforge.memory.store import MemoryStore
from memforge.models import (
    ContentItem,
    DocumentRecord,
    Memory,
    NormalizedContent,
    RawContent,
    RawMemory,
    ReconcileAction,
    ReconcileOperation,
    SourceLifecycleResetResult,
    content_hash,
)
from memforge.pipeline.projection_evidence import build_projected_claim_evidence
from memforge.pipeline.source_projection_adapters import (
    project_source_item,
    project_source_unit_tombstone,
)
from memforge.source_projection import AnchorKind, SourceAnchor, SourceProjection
from memforge.storage.adapters.sqlite import build_sqlite_adapters
from memforge.storage.database import Database


@pytest_asyncio.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "projected-lifecycle.db"))
    await database.connect()
    await database.upsert_source(
        id="src-1",
        type="confluence",
        name="Engineering",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="owner-1",
    )
    now = datetime(2026, 7, 15, tzinfo=timezone.utc).isoformat()
    await database.db.execute(
        """INSERT INTO documents (
               doc_id, source, source_url, title, space_or_project,
               last_modified, version, content_hash, last_synced
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("confluence-123", "src-1", "https://example.test/123", "Page", "ENG", now, "2", "h", now),
    )
    try:
        yield database
    finally:
        await database.close()


def _projection(
    *,
    run_id: str,
    body: str,
    item_id: str = "confluence-123",
    page_id: str = "123",
    source_id: str = "src-1",
    prior=None,
    prior_observations=None,
):
    item = ContentItem(
        item_id=item_id,
        title="Page",
        source_url="https://example.test/123",
        last_modified=datetime(2026, 7, 15, tzinfo=timezone.utc),
        version="2",
        extra={"page_id": page_id, "space_key": "ENG"},
    )
    raw = RawContent(item=item, body=body.encode(), content_type="text/html")
    normalized = NormalizedContent(item=item, markdown_body=body)
    return project_source_item(
        source_id=source_id,
        source_type="confluence",
        run_id=run_id,
        item=item,
        raw=raw,
        normalized=normalized,
        prior_unit_revision=prior,
        prior_observation_revisions=prior_observations,
    )


def _jira_projection(
    *,
    run_id: str,
    description: str,
    comment_body: str = "Decision: retain A7",
    comments_truncated: bool = False,
    source_id: str = "src-1",
    item_id: str = "confluence-123",
    issue_key: str = "PAY-12",
    issue_id: str = "10012",
    prior=None,
    prior_observations=None,
):
    item = ContentItem(
        item_id=item_id,
        title=issue_key,
        source_url=f"https://jira.example.test/browse/{issue_key}",
        last_modified=datetime(2026, 7, 15, tzinfo=timezone.utc),
        version="2",
        extra={"issue_key": issue_key, "issue_id": issue_id},
    )
    payload = {
        "id": issue_id,
        "key": issue_key,
        "fields": {
            "summary": "Payroll",
            "description": description,
            "status": None,
            "priority": None,
            "assignee": None,
            "labels": [],
            "resolution": None,
            "updated": "2026-07-15T10:00:00Z",
        },
        "_comments": [{"id": "502", "body": comment_body}],
        "_comments_included": True,
        "_comments_total": 2 if comments_truncated else 1,
        "changelog": {"startAt": 0, "histories": [], "total": 0},
    }
    if comments_truncated:
        payload["_comments_truncated"] = {"returned": 1, "total": 2}
    return project_source_item(
        source_id=source_id,
        source_type="jira",
        run_id=run_id,
        item=item,
        raw=RawContent(
            item=item,
            body=json.dumps(payload).encode(),
            content_type="application/json",
        ),
        normalized=NormalizedContent(item=item, markdown_body="PAY-12"),
        prior_unit_revision=prior,
        prior_observation_revisions=prior_observations,
    )
class _ReplacementClient:
    def __init__(self, incumbent_id: str) -> None:
        self.incumbent_id = incumbent_id

    async def reconcile_memories(self, prompt: str, **kwargs):
        del prompt, kwargs
        return ReconciliationResponse(
            decisions=[
                ReconciliationDecision(
                    index=0,
                    action="SUPERSEDE",
                    memory_id=self.incumbent_id,
                    reason="The source now retains A7.",
                )
            ]
        )

    async def detect_contradictions(self, prompt: str, **kwargs):
        del prompt, kwargs
        return ContradictionResponse(
            decisions=[
                ContradictionDecision(
                    pair_index=0,
                    classification="contradiction",
                    reason="Independent source still says A7 is removed.",
                )
            ]
        )


class _NoopClient:
    def __init__(self, incumbent_id: str) -> None:
        self.incumbent_id = incumbent_id

    async def reconcile_memories(self, prompt: str, **kwargs):
        del prompt, kwargs
        return ReconciliationResponse(
            decisions=[
                ReconciliationDecision(
                    action="NOOP",
                    memory_id=self.incumbent_id,
                    reason="The exact claim remains in the revised page.",
                )
            ]
        )

    async def detect_contradictions(self, prompt: str, **kwargs):
        del prompt, kwargs
        return ContradictionResponse(
            decisions=[
                ContradictionDecision(
                    pair_index=0,
                    classification="contradiction",
                    reason="Independent source still says A7 is removed.",
                )
            ]
        )


class _DeleteClient:
    def __init__(self, incumbent_id: str) -> None:
        self.incumbent_id = incumbent_id

    async def reconcile_memories(self, prompt: str, **kwargs):
        del prompt, kwargs
        return ReconciliationResponse(
            decisions=[
                ReconciliationDecision(
                    action="DELETE",
                    memory_id=self.incumbent_id,
                    reason="The incomplete rendering appears to omit the claim.",
                )
            ]
        )


class _PersistentlyIndexlessReplacementClient:
    def __init__(self, incumbent_id: str) -> None:
        self.incumbent_id = incumbent_id
        self.calls = 0

    async def reconcile_memories(self, prompt: str, **kwargs):
        del prompt, kwargs
        self.calls += 1
        return ReconciliationResponse(
            decisions=[
                ReconciliationDecision(
                    action="SUPERSEDE",
                    memory_id=self.incumbent_id,
                    reason="The incumbent appears stale but no replacement candidate was selected.",
                )
            ]
        )


class _OutboxDrainer:
    def __init__(self, database: Database) -> None:
        self.db = database

    async def attempt_lifecycle_vector_delivery(
        self, lifecycle_plan_id: str
    ) -> LifecycleVectorDeliveryResult:
        for task in await self.db.list_lifecycle_vector_tasks(
            lifecycle_plan_id=lifecycle_plan_id
        ):
            await self.db.complete_lifecycle_vector_task(task.id)
        return LifecycleVectorDeliveryResult(
            state=LifecycleVectorDeliveryState.DELIVERED
        )

    async def find_access_compatible_equivalence_candidates(
        self,
        memory: Memory,
        **kwargs,
    ) -> tuple[Memory, ...]:
        del kwargs
        candidate = await self.db.find_rebaseline_reactivation_candidate(
            memory.content_hash,
            visibility=memory.visibility,
            owner_user_id=memory.owner_user_id,
            repo_identifier=memory.repo_identifier,
        )
        return (candidate,) if candidate is not None else ()

    async def find_access_compatible_exact_candidate(
        self,
        memory: Memory,
        *,
        excluded_memory_ids=frozenset(),
    ) -> Memory | None:
        return await self.db.find_active_exact_claim_candidate(
            memory.content_hash,
            visibility=memory.visibility,
            owner_user_id=memory.owner_user_id,
            repo_identifier=memory.repo_identifier,
            excluded_memory_ids=tuple(sorted(excluded_memory_ids)),
        )


class _AuditedOutboxDrainer(_OutboxDrainer):
    def __init__(self, database: Database) -> None:
        super().__init__(database)
        self.audit_logger = MemoryAuditLogger(database)

    def operation_context(self, **fields):
        return self.audit_logger.default_context.child(**fields)

    async def record_audit_event(self, event_type: str, status: str, **fields) -> None:
        await self.audit_logger.emit(event_type, status, **fields)


class _CandidateLedgerClient:
    def __init__(self, response: CandidateLedgerResponse) -> None:
        self.response = response
        self.calls = 0

    async def select_memory_candidates(self, prompt: str, **kwargs):
        del prompt, kwargs
        self.calls += 1
        return self.response


class _FailingOutboxDrainer(_OutboxDrainer):
    async def attempt_lifecycle_vector_delivery(
        self, lifecycle_plan_id: str
    ) -> LifecycleVectorDeliveryResult:
        del lifecycle_plan_id
        return LifecycleVectorDeliveryResult(
            state=LifecycleVectorDeliveryState.PENDING,
            attempted_tasks=1,
            failed_tasks=1,
            error_types=("RuntimeError",),
        )


class _EquivalentMemoryStore(_OutboxDrainer):
    def __init__(self, database: Database, target: Memory) -> None:
        super().__init__(database)
        self.target = target

    async def find_access_compatible_equivalence_candidates(
        self,
        memory: Memory,
        *,
        excluded_memory_ids=frozenset(),
        scope=None,
        doc_id=None,
        entity_ids=(),
    ) -> tuple[Memory, ...]:
        del memory, excluded_memory_ids, scope, doc_id, entity_ids
        return (self.target,)


@pytest.mark.asyncio
async def test_cold_baseline_collapses_exact_duplicates_before_lifecycle_writes(
    db: Database,
) -> None:
    projection = _projection(
        run_id="projection-candidate-ledger-1",
        body="The payroll trigger remained OPEN and was not processed.",
    )
    observation_id = projection.observations[0].id
    canonical = RawMemory(
        content=projection.observation_revisions[0].content,
        memory_type="fact",
        evidence_quote=projection.observation_revisions[0].content,
        source_observation_id=observation_id,
    )
    duplicate = RawMemory(
        content="  # Page\n\nThe   payroll trigger remained OPEN and was not processed. ",
        memory_type="fact",
        evidence_quote=projection.observation_revisions[0].content,
        source_observation_id=observation_id,
    )
    adapters = build_sqlite_adapters(db, object())
    store = _AuditedOutboxDrainer(db)
    engine = MemoryEngine(
        relational=adapters.relational,
        vector=adapters.vector,
        db=db,
        memory_store=store,
        structured_llm_client=None,
    )

    stats = await engine.apply_projected_lifecycle(
        projection=projection,
        doc_id="confluence-123",
        raw_memories=[canonical, duplicate],
        doc_type="ticket",
        project_key="ENG",
        repo_identifier=None,
        entity_ids=[],
        document_content=projection.observation_revisions[0].content,
        update_mode="full_document",
        changed_hunks=None,
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 17, tzinfo=timezone.utc),
    )

    async with db.db.execute("SELECT content FROM memories") as cursor:
        rows = await cursor.fetchall()
    events = await db.list_memory_audit_events(event_type="candidate_ledger_completed")

    assert stats["added"] == 1
    assert stats["skipped"] == 1
    assert [row["content"] for row in rows] == [canonical.content]
    assert len(events) == 1
    assert events[0].source_id == "src-1"
    assert events[0].doc_id == "confluence-123"
    assert events[0].payload == {
        "input_count": 2,
        "semantic_input_count": 1,
        "selected_count": 1,
        "dropped_exact_count": 1,
        "dropped_redundant_count": 0,
        "drops": [
            {
                "candidate_content_hash": content_hash(duplicate.content),
                "candidate_source_observation_id": observation_id,
                "canonical_content_hash": content_hash(canonical.content),
                "canonical_source_observation_id": observation_id,
                "method": "exact_content",
                "reason": "normalized content is identical",
            }
        ],
    }


@pytest.mark.asyncio
async def test_projected_create_persists_validity_as_dates(db: Database) -> None:
    projection = _projection(
        run_id="projection-validity",
        body="The policy is effective during June 2026.",
    )
    revision = projection.observation_revisions[0]
    raw = RawMemory(
        content="The policy is effective during June 2026.",
        memory_type="fact",
        evidence_quote=revision.content,
        source_observation_id=projection.observations[0].id,
        valid_from="2026-06-01",
        valid_until="2026-06-30T12:00:00+08:00",
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        relational=adapters.relational,
        vector=adapters.vector,
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=None,
    )

    stats = await engine.apply_projected_lifecycle(
        projection=projection,
        doc_id="confluence-123",
        raw_memories=[raw],
        doc_type="document",
        project_key="ENG",
        repo_identifier=None,
        entity_ids=[],
        document_content=revision.content,
        update_mode="full_document",
        changed_hunks=None,
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 17, tzinfo=timezone.utc),
    )

    [memory] = await db.list_memories()
    assert stats["added"] == 1
    assert memory.valid_from == date(2026, 6, 1)
    assert memory.valid_until == date(2026, 6, 30)


@pytest.mark.asyncio
async def test_incomplete_candidate_ledger_is_audited_and_writes_no_memory(
    db: Database,
) -> None:
    projection = _projection(
        run_id="projection-candidate-ledger-failed",
        body="The trigger remained OPEN. The trigger was not processed.",
    )
    observation_id = projection.observations[0].id
    client = _CandidateLedgerClient(
        CandidateLedgerResponse(
            decisions=[CandidateLedgerDecision(index=0, action="KEEP")]
        )
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        relational=adapters.relational,
        vector=adapters.vector,
        db=db,
        memory_store=_AuditedOutboxDrainer(db),
        structured_llm_client=client,
    )

    with pytest.raises(RuntimeError, match="candidate ledger failed closed: invalid_ledger"):
        await engine.apply_projected_lifecycle(
            projection=projection,
            doc_id="confluence-123",
            raw_memories=[
                RawMemory(
                    content="The trigger remained OPEN.",
                    memory_type="fact",
                    evidence_quote="The trigger remained OPEN.",
                    source_observation_id=observation_id,
                ),
                RawMemory(
                    content="The trigger was not processed.",
                    memory_type="fact",
                    evidence_quote="The trigger was not processed.",
                    source_observation_id=observation_id,
                ),
            ],
            doc_type="ticket",
            project_key="ENG",
            repo_identifier=None,
            entity_ids=[],
            document_content=projection.observation_revisions[0].content,
            update_mode="full_document",
            changed_hunks=None,
            update_plan_stats=None,
            source_updated_at=datetime(2026, 7, 17, tzinfo=timezone.utc),
        )

    async with db.db.execute("SELECT COUNT(*) AS total FROM memories") as cursor:
        row = await cursor.fetchone()
    events = await db.list_memory_audit_events(event_type="candidate_ledger_failed")

    assert row["total"] == 0
    assert client.calls == 2
    assert len(events) == 1
    assert events[0].status == "failed"
    assert events[0].reason == "invalid_ledger"
    assert events[0].payload["input_count"] == 2
    assert events[0].payload["semantic_input_count"] == 2
    assert events[0].payload["selected_count"] == 0
    assert events[0].payload["fingerprints_truncated"] is False
    assert events[0].payload["candidate_fingerprints"] == [
        {
            "content_hash": content_hash("The trigger remained OPEN."),
            "source_observation_id": observation_id,
        },
        {
            "content_hash": content_hash("The trigger was not processed."),
            "source_observation_id": observation_id,
        },
    ]


class _SemanticEquivalentClient:
    async def reconcile_memories(self, prompt: str, **kwargs):
        del prompt, kwargs
        return ReconciliationResponse(
            decisions=[ReconciliationDecision(action="ADD", index=0)]
        )

    async def classify_memory_equivalence(self, prompt: str, **kwargs):
        del kwargs
        pair = prompt.split("<claim_pair>", 1)[1].split("</claim_pair>", 1)[0]
        payload = json.loads(pair)
        assert set(payload) == {"claim_a", "claim_b"}
        assert set(payload.values()) == {"A7 is removed.", "A7 remains excluded."}
        return MemoryEquivalenceResponse(
            equivalent=True,
            reason="Both claims state that A7 is excluded.",
        )


class _SupportValidatingNoopClient(_NoopClient):
    def __init__(
        self,
        memory_id: str,
        *,
        supported: bool,
        evidence_quote: str = "",
    ) -> None:
        super().__init__(memory_id)
        self.supported = supported
        self.evidence_quote = evidence_quote

    async def validate_memory_support(self, prompt: str, **kwargs):
        del kwargs
        assert '"memory_claim"' in prompt
        assert (
            "A7 is retained for regular payroll." in prompt
            or "A7 is removed." in prompt
        )
        return MemorySupportValidationResponse(
            supported=self.supported,
            evidence_quote=self.evidence_quote,
            reason=(
                "The applicability remains regular payroll."
                if self.supported
                else "The applicability changed from regular to off-cycle payroll."
            ),
        )


def test_lifecycle_access_identity_treats_project_as_relevance_only() -> None:
    pay = lifecycle_access_context_hash(
        visibility="workspace",
        owner_user_id=None,
        project_key="PAY",
        repo_identifier=None,
    )
    risk = lifecycle_access_context_hash(
        visibility="workspace",
        owner_user_id=None,
        project_key="RISK",
        repo_identifier=None,
    )

    assert pay == risk


def test_memory_equivalence_pair_payload_is_order_independent_and_neutral() -> None:
    first = "The deployment policy permits at most three retry attempts."
    second = "The maximum retry count permitted by the deployment policy is 3."

    forward = _memory_equivalence_pair_json(first, second)
    reverse = _memory_equivalence_pair_json(second, first)

    assert forward == reverse
    payload = json.loads(forward)
    assert set(payload) == {"claim_a", "claim_b"}
    assert set(payload.values()) == {first, second}
    assert "document, case, or record states that P" in MEMORY_EQUIVALENCE_PROMPT
    assert "neither claim is about the act, completeness, or authority of recording" in (
        MEMORY_EQUIVALENCE_PROMPT
    )


async def _seed_incumbent_support(
    db: Database,
    *,
    projection,
    memory_id: str = "mem-old",
    memory_content: str = "A7 is removed.",
    observation_index: int = 0,
    source_type: str = "confluence",
) -> Memory:
    incumbent = Memory(
        id=memory_id,
        memory_type="decision",
        content=memory_content,
        content_hash=content_hash(memory_content),
    )
    await db.insert_memory(incumbent)
    await db.add_memory_source(
        incumbent.id,
        "confluence-123",
        source_type,
        memory_content,
        source_updated_at=None,
    )
    observation = projection.observations[observation_index]
    revisions_by_observation = {
        item.observation_id: item for item in projection.observation_revisions
    }
    revision = revisions_by_observation[observation.id]
    unit = EvidenceUnit(
        id=f"eu-{memory_id}",
        source_id="src-1",
        doc_id="confluence-123",
        doc_revision_id=projection.source_unit_revisions[0].id,
        source_type=source_type,
        source_anchor=observation.id,
        source_lineage_id=projection.source_units[0].id,
        project_key="ENG",
        visibility="workspace",
        owner_user_id=None,
        repo_identifier=None,
        content=revision.content,
        excerpt=memory_content,
        evidence_provenance=EvidenceContentProvenance.SOURCE_EXCERPT,
        access_context_hash="workspace-eng",
    )
    await db.upsert_evidence_unit(unit)
    reference = (
        await db.record_evidence_references(
            unit.id,
            (
                EvidenceReference(
                    role=EvidenceRole.PRIMARY,
                    anchor=SourceAnchor(
                        kind=AnchorKind.WHOLE_OBSERVATION,
                        observation_id=observation.id,
                        observation_revision_id=revision.id,
                    ),
                ),
            ),
        )
    )[0]
    await db.upsert_memory_support_assertion(
        MemorySupportAssertion(
            id=f"support-{memory_id}",
            memory_id=incumbent.id,
            evidence_reference_id=reference.id or "",
            source_id="src-1",
            access_context_hash="workspace-eng",
        )
    )
    return incumbent


@pytest.mark.asyncio
async def test_noop_rebinds_support_to_current_source_revision(db: Database) -> None:
    first = _projection(run_id="projection-noop-1", body="A7 is removed.")
    await db.record_source_projection(first)
    incumbent = await _seed_incumbent_support(db, projection=first)
    incumbent = await db.get_memory(incumbent.id)
    assert incumbent is not None
    await db.enable_lifecycle_gate("src-1")
    old_support = await db.get_active_memory_support_reference_ids(incumbent.id)

    second = _projection(
        run_id="projection-noop-2",
        body="A7 is removed.\n\nThe page now also documents rollout ownership.",
        prior=first.source_unit_revisions[0],
        prior_observations={
            revision.observation_id: revision for revision in first.observation_revisions
        },
    )
    raw = RawMemory(
        content=incumbent.content,
        memory_type=incumbent.memory_type,
        confidence=incumbent.confidence,
        tags=list(incumbent.tags),
        extraction_context="A7 is removed.",
        evidence_quote="A7 is removed.",
    )
    evidence = build_projected_claim_evidence(
        projection=second,
        raw_memories=(raw,),
        doc_id="confluence-123",
        source_type="confluence",
        project_key="ENG",
        visibility="workspace",
        owner_user_id=None,
        repo_identifier=None,
        access_context_hash="workspace-eng",
        extractor_run_id=second.run_id,
    )
    delta = second.deltas[0]
    scope = ReconciliationScope(
        id="scope-noop-rebind",
        source_id="src-1",
        source_unit_id=delta.source_unit_id,
        base_unit_revision_id=delta.previous_unit_revision_id,
        target_unit_revision_id=delta.current_unit_revision_id,
    )
    plan = build_lifecycle_plan(
        plan_id="plan-noop-rebind",
        scope=scope,
        gate_state=LifecycleGateState.ENABLED,
        operations=(
            ReconcileOperation(
                action=ReconcileAction.NOOP,
                memory_id=incumbent.id,
                memory=raw,
                reason="claim remains valid",
            ),
        ),
        incumbents={incumbent.id: incumbent},
        source_support_reference_ids={incumbent.id: old_support},
        all_active_support_reference_ids={incumbent.id: old_support},
        support_set_hashes={
            incumbent.id: await db.get_memory_support_set_hash(incumbent.id)
        },
        observation_revision_ids=tuple(
            revision.id for revision in second.observation_revisions
        ),
        new_evidence_reference_ids=(),
        evidence_reference_ids_by_claim_hash=evidence.reference_ids_by_claim_hash,
        defaults=NewMemoryDefaults(
            visibility="workspace",
            owner_user_id=None,
            project_key="ENG",
            repo_identifier=None,
            doc_id="confluence-123",
            source_type="confluence",
            access_context_hash="workspace-eng",
        ),
        evidence_units=evidence.units,
        evidence_references=evidence.references,
    )

    await db.apply_source_projection_lifecycle(second, plan)

    current_support = await db.get_active_memory_support_reference_ids(incumbent.id)
    expected_support = evidence.reference_ids_by_claim_hash[content_hash(raw.content)]
    assert current_support == expected_support
    assert set(current_support).isdisjoint(old_support)
    current_unit = await db.get_current_source_unit_revision(first.source_units[0].id)
    assert current_unit is not None
    assert current_unit.id == second.source_unit_revisions[0].id


@pytest.mark.asyncio
async def test_noop_without_current_evidence_rolls_back_stale_support(db: Database) -> None:
    first = _projection(run_id="projection-stale-1", body="A7 is removed.")
    await db.record_source_projection(first)
    seeded = await _seed_incumbent_support(db, projection=first)
    incumbent = await db.get_memory(seeded.id)
    assert incumbent is not None
    await db.enable_lifecycle_gate("src-1")
    old_support = await db.get_active_memory_support_reference_ids(incumbent.id)
    second = _projection(
        run_id="projection-stale-2",
        body="A7 is retained.",
        prior=first.source_unit_revisions[0],
        prior_observations={
            revision.observation_id: revision for revision in first.observation_revisions
        },
    )
    delta = second.deltas[0]
    scope = ReconciliationScope(
        id="scope-stale-noop",
        source_id="src-1",
        source_unit_id=delta.source_unit_id,
        base_unit_revision_id=delta.previous_unit_revision_id,
        target_unit_revision_id=delta.current_unit_revision_id,
    )
    plan = build_lifecycle_plan(
        plan_id="plan-stale-noop",
        scope=scope,
        gate_state=LifecycleGateState.ENABLED,
        operations=(
            ReconcileOperation(
                action=ReconcileAction.NOOP,
                memory_id=incumbent.id,
                reason="incorrectly kept without current evidence",
            ),
        ),
        incumbents={incumbent.id: incumbent},
        source_support_reference_ids={incumbent.id: old_support},
        all_active_support_reference_ids={incumbent.id: old_support},
        support_set_hashes={
            incumbent.id: await db.get_memory_support_set_hash(incumbent.id)
        },
        observation_revision_ids=tuple(
            revision.id for revision in second.observation_revisions
        ),
        new_evidence_reference_ids=(),
        defaults=NewMemoryDefaults(
            visibility="workspace",
            owner_user_id=None,
            project_key="ENG",
            repo_identifier=None,
            doc_id="confluence-123",
            source_type="confluence",
            access_context_hash="workspace-eng",
        ),
    )

    with pytest.raises(ValueError, match="stale or ambiguous source support"):
        await db.apply_source_projection_lifecycle(second, plan)

    current_unit = await db.get_current_source_unit_revision(first.source_units[0].id)
    assert current_unit is not None
    assert current_unit.id == first.source_unit_revisions[0].id
    assert await db.get_active_memory_support_reference_ids(incumbent.id) == old_support


@pytest.mark.asyncio
async def test_review_exempts_only_the_incumbent_support_staged_for_removal(
    db: Database,
) -> None:
    first = _projection(run_id="projection-review-scope-1", body="A7 is removed.")
    await db.record_source_projection(first)
    reviewed = await _seed_incumbent_support(
        db,
        projection=first,
        memory_id="mem-reviewed",
        memory_content="A7 is removed.",
    )
    unrelated = await _seed_incumbent_support(
        db,
        projection=first,
        memory_id="mem-unrelated",
        memory_content="A separate control remains enabled.",
    )
    reviewed = await db.get_memory(reviewed.id)
    unrelated = await db.get_memory(unrelated.id)
    assert reviewed is not None
    assert unrelated is not None
    reviewed_support = await db.get_active_memory_support_reference_ids(reviewed.id)
    unrelated_support = await db.get_active_memory_support_reference_ids(unrelated.id)
    await db.enable_lifecycle_gate("src-1")

    second = _projection(
        run_id="projection-review-scope-2",
        body="A7 is retained.",
        prior=first.source_unit_revisions[0],
        prior_observations={
            revision.observation_id: revision
            for revision in first.observation_revisions
        },
    )
    delta = second.deltas[0]
    scope = ReconciliationScope(
        id="scope-review-support-exemption",
        source_id="src-1",
        source_unit_id=delta.source_unit_id,
        base_unit_revision_id=delta.previous_unit_revision_id,
        target_unit_revision_id=delta.current_unit_revision_id,
    )
    plan = build_lifecycle_plan(
        plan_id="plan-review-support-exemption",
        scope=scope,
        gate_state=LifecycleGateState.ENABLED,
        operations=(
            ReconcileOperation(
                action=ReconcileAction.DELETE,
                memory_id=reviewed.id,
                reason="the source now disputes this claim",
                flag_for_review=True,
            ),
            ReconcileOperation(
                action=ReconcileAction.NOOP,
                memory_id=unrelated.id,
                reason="incorrectly kept without current evidence",
            ),
        ),
        incumbents={reviewed.id: reviewed, unrelated.id: unrelated},
        source_support_reference_ids={
            reviewed.id: reviewed_support,
            unrelated.id: unrelated_support,
        },
        all_active_support_reference_ids={
            reviewed.id: reviewed_support,
            unrelated.id: unrelated_support,
        },
        support_set_hashes={
            reviewed.id: await db.get_memory_support_set_hash(reviewed.id),
            unrelated.id: await db.get_memory_support_set_hash(unrelated.id),
        },
        observation_revision_ids=tuple(
            revision.id for revision in second.observation_revisions
        ),
        new_evidence_reference_ids=(),
        defaults=NewMemoryDefaults(
            visibility="workspace",
            owner_user_id=None,
            project_key="ENG",
            repo_identifier=None,
            doc_id="confluence-123",
            source_type="confluence",
            access_context_hash="workspace-eng",
        ),
    )

    with pytest.raises(ValueError, match="stale or ambiguous source support"):
        await db.apply_source_projection_lifecycle(second, plan)

    current_unit = await db.get_current_source_unit_revision(first.source_units[0].id)
    assert current_unit is not None
    assert current_unit.id == first.source_unit_revisions[0].id
    assert await db.get_lifecycle_review(
        str(plan.mutations[0].payload["review_id"])
    ) is None


@pytest.mark.asyncio
async def test_review_preserves_its_exact_contested_incumbent_support(
    db: Database,
) -> None:
    first = _projection(run_id="projection-reviewed-support-1", body="A7 is removed.")
    await db.record_source_projection(first)
    incumbent = await _seed_incumbent_support(db, projection=first)
    incumbent = await db.get_memory(incumbent.id)
    assert incumbent is not None
    old_support = await db.get_active_memory_support_reference_ids(incumbent.id)
    await db.enable_lifecycle_gate("src-1")

    second = _projection(
        run_id="projection-reviewed-support-2",
        body="A7 is retained.",
        prior=first.source_unit_revisions[0],
        prior_observations={
            revision.observation_id: revision
            for revision in first.observation_revisions
        },
    )
    delta = second.deltas[0]
    scope = ReconciliationScope(
        id="scope-reviewed-support",
        source_id="src-1",
        source_unit_id=delta.source_unit_id,
        base_unit_revision_id=delta.previous_unit_revision_id,
        target_unit_revision_id=delta.current_unit_revision_id,
    )
    plan = build_lifecycle_plan(
        plan_id="plan-reviewed-support",
        scope=scope,
        gate_state=LifecycleGateState.ENABLED,
        operations=(
            ReconcileOperation(
                action=ReconcileAction.DELETE,
                memory_id=incumbent.id,
                reason="the source now disputes this claim",
                flag_for_review=True,
            ),
        ),
        incumbents={incumbent.id: incumbent},
        source_support_reference_ids={incumbent.id: old_support},
        all_active_support_reference_ids={incumbent.id: old_support},
        support_set_hashes={
            incumbent.id: await db.get_memory_support_set_hash(incumbent.id)
        },
        observation_revision_ids=tuple(
            revision.id for revision in second.observation_revisions
        ),
        new_evidence_reference_ids=(),
        defaults=NewMemoryDefaults(
            visibility="workspace",
            owner_user_id=None,
            project_key="ENG",
            repo_identifier=None,
            doc_id="confluence-123",
            source_type="confluence",
            access_context_hash="workspace-eng",
        ),
    )

    await db.apply_source_projection_lifecycle(second, plan)

    current_unit = await db.get_current_source_unit_revision(first.source_units[0].id)
    assert current_unit is not None
    assert current_unit.id == second.source_unit_revisions[0].id
    assert await db.get_active_memory_support_reference_ids(incumbent.id) == old_support
    review = await db.get_lifecycle_review(
        str(plan.mutations[0].payload["review_id"])
    )
    assert review is not None
    assert review.status is LifecycleReviewStatus.PENDING


@pytest.mark.asyncio
async def test_later_unit_plan_preserves_exact_support_contested_by_durable_review(
    db: Database,
) -> None:
    first = _projection(run_id="projection-durable-review-1", body="A7 is removed.")
    await db.record_source_projection(first)
    incumbent = await _seed_incumbent_support(db, projection=first)
    incumbent = await db.get_memory(incumbent.id)
    assert incumbent is not None
    old_support = await db.get_active_memory_support_reference_ids(incumbent.id)
    await db.enable_lifecycle_gate("src-1")

    changed = _projection(
        run_id="projection-durable-review-2",
        body="A7 is retained.",
        prior=first.source_unit_revisions[0],
        prior_observations={
            revision.observation_id: revision
            for revision in first.observation_revisions
        },
    )
    changed_delta = changed.deltas[0]
    review_plan = build_lifecycle_plan(
        plan_id="plan-durable-review",
        scope=ReconciliationScope(
            id="scope-durable-review",
            source_id="src-1",
            source_unit_id=changed_delta.source_unit_id,
            base_unit_revision_id=changed_delta.previous_unit_revision_id,
            target_unit_revision_id=changed_delta.current_unit_revision_id,
        ),
        gate_state=LifecycleGateState.ENABLED,
        operations=(
            ReconcileOperation(
                action=ReconcileAction.DELETE,
                memory_id=incumbent.id,
                reason="the source now disputes this claim",
                flag_for_review=True,
            ),
        ),
        incumbents={incumbent.id: incumbent},
        source_support_reference_ids={incumbent.id: old_support},
        all_active_support_reference_ids={incumbent.id: old_support},
        support_set_hashes={
            incumbent.id: await db.get_memory_support_set_hash(incumbent.id)
        },
        observation_revision_ids=tuple(
            revision.id for revision in changed.observation_revisions
        ),
        new_evidence_reference_ids=(),
        defaults=NewMemoryDefaults(
            visibility="workspace",
            owner_user_id=None,
            project_key="ENG",
            repo_identifier=None,
            doc_id="confluence-123",
            source_type="confluence",
            access_context_hash="workspace-eng",
        ),
    )
    await db.apply_source_projection_lifecycle(changed, review_plan)

    later = _projection(
        run_id="projection-durable-review-later-unit",
        body="A separate page changes.",
        item_id="confluence-456",
        page_id="456",
    )
    later_delta = later.deltas[0]
    later_plan = build_lifecycle_plan(
        plan_id="plan-after-durable-review",
        scope=ReconciliationScope(
            id="scope-after-durable-review",
            source_id="src-1",
            source_unit_id=later_delta.source_unit_id,
            base_unit_revision_id=later_delta.previous_unit_revision_id,
            target_unit_revision_id=later_delta.current_unit_revision_id,
        ),
        gate_state=LifecycleGateState.ENABLED,
        operations=(
            ReconcileOperation(
                action=ReconcileAction.NOOP,
                memory_id=incumbent.id,
                reason="the unrelated unit does not resolve the pending review",
            ),
        ),
        incumbents={incumbent.id: incumbent},
        source_support_reference_ids={incumbent.id: old_support},
        all_active_support_reference_ids={incumbent.id: old_support},
        support_set_hashes={
            incumbent.id: await db.get_memory_support_set_hash(incumbent.id)
        },
        observation_revision_ids=tuple(
            revision.id for revision in later.observation_revisions
        ),
        new_evidence_reference_ids=(),
        defaults=NewMemoryDefaults(
            visibility="workspace",
            owner_user_id=None,
            project_key="ENG",
            repo_identifier=None,
            doc_id="confluence-456",
            source_type="confluence",
            access_context_hash="workspace-eng",
        ),
    )

    await db.apply_source_projection_lifecycle(later, later_plan)

    assert await db.get_lifecycle_plan_status(later_plan.id) == "applied"
    assert await db.get_active_memory_support_reference_ids(incumbent.id) == old_support
    review = await db.get_lifecycle_review(
        str(review_plan.mutations[0].payload["review_id"])
    )
    assert review is not None
    assert review.status is LifecycleReviewStatus.PENDING


@pytest.mark.asyncio
async def test_review_does_not_exempt_support_from_another_source_unit(
    db: Database,
) -> None:
    first = _projection(run_id="projection-review-unit-1", body="A7 is removed.")
    await db.record_source_projection(first)
    incumbent = await _seed_incumbent_support(db, projection=first)
    old_support = await db.get_active_memory_support_reference_ids(incumbent.id)
    changed = _projection(
        run_id="projection-review-unit-2",
        body="A7 is retained.",
        prior=first.source_unit_revisions[0],
        prior_observations={
            revision.observation_id: revision
            for revision in first.observation_revisions
        },
    )
    await db.record_source_projection(changed)
    other = _projection(
        run_id="projection-review-other-unit",
        body="An unrelated page changes.",
        item_id="confluence-456",
        page_id="456",
    )
    await db.record_source_projection(other)
    plan = SimpleNamespace(
        mutations=(
            LifecycleMutation(
                mutation_type=LifecycleMutationType.CREATE_REVIEW,
                memory_id=incumbent.id,
                source_id="src-1",
                payload={
                    "staged_evidence": {
                        "proposed_mutations": [
                            {
                                "mutation_type": "remove_support",
                                "memory_id": incumbent.id,
                                "source_id": "src-1",
                                "evidence_reference_ids": list(old_support),
                            }
                        ]
                    }
                },
            ),
        ),
        coverage_proof=SimpleNamespace(
            mandatory_incumbent_ids=(incumbent.id,)
        ),
        scope=SimpleNamespace(
            source_id="src-1",
            source_unit_id=other.source_units[0].id,
        ),
    )

    with pytest.raises(ValueError, match="stale or ambiguous source support"):
        await db._validate_projected_support_invariant_unlocked(plan)


@pytest.mark.asyncio
async def test_review_does_not_exempt_mismatched_observation_revision_lineage(
    db: Database,
) -> None:
    first = _projection(run_id="projection-review-lineage-1", body="A7 is removed.")
    await db.record_source_projection(first)
    incumbent = await _seed_incumbent_support(db, projection=first)
    support = await db.get_active_memory_support_reference_ids(incumbent.id)
    other = _projection(
        run_id="projection-review-lineage-other",
        body="An unrelated page changes.",
        item_id="confluence-456",
        page_id="456",
    )
    await db.record_source_projection(other)
    await db.db.execute(
        "UPDATE evidence_references SET observation_revision_id = ? WHERE id = ?",
        (other.observation_revisions[0].id, support[0]),
    )
    await db.db.commit()
    plan = SimpleNamespace(
        mutations=(
            LifecycleMutation(
                mutation_type=LifecycleMutationType.CREATE_REVIEW,
                memory_id=incumbent.id,
                source_id="src-1",
                payload={
                    "staged_evidence": {
                        "proposed_mutations": [
                            {
                                "mutation_type": "remove_support",
                                "memory_id": incumbent.id,
                                "source_id": "src-1",
                                "evidence_reference_ids": list(support),
                            }
                        ]
                    }
                },
            ),
        ),
        coverage_proof=SimpleNamespace(
            mandatory_incumbent_ids=(incumbent.id,)
        ),
        scope=SimpleNamespace(
            source_id="src-1",
            source_unit_id=first.source_units[0].id,
        ),
    )

    with pytest.raises(ValueError, match="stale or ambiguous source support"):
        await db._validate_projected_support_invariant_unlocked(plan)


@pytest.mark.parametrize(
    ("broken_hop", "table", "id_column"),
    (
        ("evidence_reference", "evidence_references", "reference_id"),
        ("evidence_unit", "evidence_units", "evidence_unit_id"),
        ("observation", "source_observations", "observation_id"),
        (
            "observation_revision",
            "source_observation_revisions",
            "observation_revision_id",
        ),
        ("source_unit", "source_units", "source_unit_id"),
    ),
)
@pytest.mark.asyncio
async def test_projected_support_invariant_cannot_hide_a_missing_lineage_hop(
    db: Database,
    broken_hop: str,
    table: str,
    id_column: str,
) -> None:
    projection = _projection(
        run_id=f"projection-missing-{broken_hop}",
        body="A7 is removed.",
    )
    await db.record_source_projection(projection)
    incumbent = await _seed_incumbent_support(db, projection=projection)
    [reference_id] = await db.get_active_memory_support_reference_ids(incumbent.id)
    async with db.db.execute(
        """SELECT er.evidence_unit_id, er.observation_id,
                  er.observation_revision_id, so.source_unit_id
             FROM evidence_references er
             JOIN source_observations so ON so.id = er.observation_id
            WHERE er.id = ?""",
        (reference_id,),
    ) as cursor:
        lineage = await cursor.fetchone()
    assert lineage is not None
    ids = {
        "reference_id": reference_id,
        "evidence_unit_id": lineage["evidence_unit_id"],
        "observation_id": lineage["observation_id"],
        "observation_revision_id": lineage["observation_revision_id"],
        "source_unit_id": lineage["source_unit_id"],
    }
    await db.db.commit()
    await db.db.execute("PRAGMA foreign_keys = OFF")
    await db.db.execute(f"DELETE FROM {table} WHERE id = ?", (ids[id_column],))
    await db.db.commit()
    plan = SimpleNamespace(
        mutations=(),
        coverage_proof=SimpleNamespace(
            mandatory_incumbent_ids=(incumbent.id,)
        ),
        scope=SimpleNamespace(
            source_id="src-1",
            source_unit_id=projection.source_units[0].id,
        ),
    )

    with pytest.raises(ValueError, match="stale or ambiguous source support"):
        await db._validate_projected_support_invariant_unlocked(plan)


@pytest.mark.asyncio
async def test_projected_support_invariant_accepts_other_valid_same_source_unit(
    db: Database,
) -> None:
    first = _projection(run_id="projection-multi-unit-1", body="A7 is removed.")
    await db.record_source_projection(first)
    incumbent = await _seed_incumbent_support(db, projection=first)
    other_item = ContentItem(
        item_id="confluence-456",
        title="Independent Page",
        source_url="https://example.test/456",
        last_modified=datetime(2026, 7, 15, tzinfo=timezone.utc),
        version="1",
        extra={"page_id": "456", "space_key": "ENG"},
    )
    other_body = "Independent note: A7 is removed."
    other = project_source_item(
        source_id="src-1",
        source_type="confluence",
        run_id="projection-multi-unit-2",
        item=other_item,
        raw=RawContent(
            item=other_item,
            body=other_body.encode(),
            content_type="text/html",
        ),
        normalized=NormalizedContent(
            item=other_item,
            markdown_body=other_body,
        ),
    )
    await db.record_source_projection(other)
    now = datetime(2026, 7, 15, tzinfo=timezone.utc)
    await db.upsert_document(
        DocumentRecord(
            doc_id="confluence-456",
            source="src-1",
            source_url="https://example.test/456",
            title="Independent Page",
            space_or_project="ENG",
            author=None,
            last_modified=now,
            labels=[],
            version="1",
            content_hash="independent-page-hash",
            token_count=10,
            raw_content_uri=None,
            raw_content_type=None,
            normalized_content_uri=None,
            pdf_content_uri=None,
            last_synced=now,
        )
    )
    await db.add_memory_source(
        incumbent.id,
        "confluence-456",
        "confluence",
        "A7 is removed.",
        source_updated_at=now,
    )
    other_observation = other.observations[0]
    other_revision = other.observation_revisions[0]
    other_unit = EvidenceUnit(
        id="eu-multi-unit-other",
        source_id="src-1",
        doc_id="confluence-456",
        doc_revision_id=other.source_unit_revisions[0].id,
        source_type="confluence",
        source_anchor=other_observation.id,
        source_lineage_id=other.source_units[0].id,
        project_key="ENG",
        visibility="workspace",
        owner_user_id=None,
        repo_identifier=None,
        content=other_revision.content,
        excerpt="A7 is removed.",
        evidence_provenance=EvidenceContentProvenance.SOURCE_EXCERPT,
        access_context_hash="workspace-eng",
    )
    await db.upsert_evidence_unit(other_unit)
    other_reference = (
        await db.record_evidence_references(
            other_unit.id,
            (
                EvidenceReference(
                    role=EvidenceRole.PRIMARY,
                    anchor=SourceAnchor(
                        kind=AnchorKind.WHOLE_OBSERVATION,
                        observation_id=other_observation.id,
                        observation_revision_id=other_revision.id,
                    ),
                ),
            ),
        )
    )[0]
    await db.upsert_memory_support_assertion(
        MemorySupportAssertion(
            id="support-multi-unit-other",
            memory_id=incumbent.id,
            evidence_reference_id=other_reference.id or "",
            source_id="src-1",
            access_context_hash="workspace-eng",
        )
    )
    plan = SimpleNamespace(
        mutations=(),
        coverage_proof=SimpleNamespace(mandatory_incumbent_ids=(incumbent.id,)),
        scope=SimpleNamespace(
            source_id="src-1",
            source_unit_id=first.source_units[0].id,
        ),
    )
    support_rows = await db.db.execute_fetchall(
        """SELECT eu.source_id AS evidence_source_id,
                  so.source_id AS observation_source_id,
                  su.source_id AS unit_source_id,
                  eu.source_lineage_id, so.source_unit_id,
                  er.observation_revision_id, so.current_revision_id
             FROM memory_support_assertions msa
             JOIN evidence_references er ON er.id = msa.evidence_reference_id
             JOIN evidence_units eu ON eu.id = er.evidence_unit_id
             JOIN source_observations so ON so.id = er.observation_id
             JOIN source_units su ON su.id = so.source_unit_id
            WHERE msa.memory_id = ? AND msa.source_id = ? AND msa.active = 1
            ORDER BY so.source_unit_id""",
        (incumbent.id, "src-1"),
    )
    assert {
        (
            row["evidence_source_id"],
            row["observation_source_id"],
            row["unit_source_id"],
            row["source_lineage_id"],
            row["source_unit_id"],
            row["observation_revision_id"],
            row["current_revision_id"],
        )
        for row in support_rows
    } == {
        (
            "src-1",
            "src-1",
            "src-1",
            first.source_units[0].id,
            first.source_units[0].id,
            first.observation_revisions[0].id,
            first.observation_revisions[0].id,
        ),
        (
            "src-1",
            "src-1",
            "src-1",
            other.source_units[0].id,
            other.source_units[0].id,
            other.observation_revisions[0].id,
            other.observation_revisions[0].id,
        ),
    }

    await db._validate_projected_support_invariant_unlocked(plan)

    second = _projection(
        run_id="projection-multi-unit-rebind",
        body="A7 is removed.\n\nThe page now names an owner.",
        prior=first.source_unit_revisions[0],
        prior_observations={
            revision.observation_id: revision
            for revision in first.observation_revisions
        },
    )
    second = replace(
        second,
        observations=other.observations + second.observations,
        observation_revisions=(
            other.observation_revisions + second.observation_revisions
        ),
        source_units=other.source_units + second.source_units,
        source_unit_revisions=(
            other.source_unit_revisions + second.source_unit_revisions
        ),
        relations=other.relations + second.relations,
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        relational=adapters.relational,
        vector=adapters.vector,
        db=db,
        memory_store=_OutboxDrainer(db),
    )

    [rebound] = await engine._rebind_noop_evidence_to_current_revision(
        operations=(
            ReconcileOperation(
                action=ReconcileAction.NOOP,
                memory_id=incumbent.id,
                reason="claim remains valid in this Unit",
            ),
        ),
        incumbents={incumbent.id: incumbent},
        unit_support=await db.get_source_unit_support_reference_ids(
            first.source_units[0].id
        ),
        projection=second,
    )

    assert rebound.action is ReconcileAction.NOOP
    assert rebound.memory is not None
    assert rebound.memory.source_observation_id == first.observations[0].id
    assert other_reference.id in (
        await db.get_active_memory_support_reference_ids(incumbent.id)
    )


@pytest.mark.asyncio
async def test_incremental_noop_rebinds_exact_unchanged_claim_without_new_extraction(
    db: Database,
) -> None:
    first = _projection(
        run_id="projection-incremental-keep-1",
        body="A7 is removed.\nOld deployment note.",
    )
    await db.record_source_projection(first)
    incumbent = await _seed_incumbent_support(db, projection=first)
    await db.enable_lifecycle_gate("src-1")
    old_support = await db.get_active_memory_support_reference_ids(incumbent.id)
    second = _projection(
        run_id="projection-incremental-keep-2",
        body="A7 is removed.\nNew deployment note.",
        prior=first.source_unit_revisions[0],
        prior_observations={
            first.observations[0].id: first.observation_revisions[0]
        },
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        relational=adapters.relational,
        vector=adapters.vector,
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=_NoopClient(incumbent.id),
    )

    stats = await engine.apply_projected_lifecycle(
        projection=second,
        doc_id="confluence-123",
        raw_memories=[],
        doc_type="design-doc",
        project_key="ENG",
        repo_identifier=None,
        entity_ids=[],
        document_content=second.observation_revisions[0].content,
        update_mode="diff_guided",
        changed_hunks="Old deployment note -> New deployment note",
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 15, 10, 36, tzinfo=timezone.utc),
    )

    current_support = await db.get_active_memory_support_reference_ids(incumbent.id)
    assert stats["noop"] == 1
    assert current_support
    assert set(current_support).isdisjoint(old_support)
    [evidence] = await db.get_active_memory_support_evidence(
        incumbent.id,
        source_id="src-1",
    )
    assert evidence.anchor.observation_revision_id == second.observation_revisions[0].id


@pytest.mark.asyncio
async def test_incremental_noop_revalidates_reworded_primary_evidence(
    db: Database,
) -> None:
    first = _projection(
        run_id="projection-primary-reword-1",
        body="A7 is removed.",
    )
    await db.record_source_projection(first)
    incumbent = await _seed_incumbent_support(db, projection=first)
    await db.enable_lifecycle_gate("src-1")
    old_support = await db.get_active_memory_support_reference_ids(incumbent.id)
    current_quote = "The A7 slot remains excluded."
    second = _projection(
        run_id="projection-primary-reword-2",
        body=current_quote,
        prior=first.source_unit_revisions[0],
        prior_observations={
            first.observations[0].id: first.observation_revisions[0]
        },
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        relational=adapters.relational,
        vector=adapters.vector,
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=_SupportValidatingNoopClient(
            incumbent.id,
            supported=True,
            evidence_quote=current_quote,
        ),
    )

    stats = await engine.apply_projected_lifecycle(
        projection=second,
        doc_id="confluence-123",
        raw_memories=[],
        doc_type="design-doc",
        project_key="ENG",
        repo_identifier=None,
        entity_ids=[],
        document_content=current_quote,
        update_mode="diff_guided",
        changed_hunks="A7 is removed. -> The A7 slot remains excluded.",
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 16, 10, 36, tzinfo=timezone.utc),
    )

    current_support = await db.get_active_memory_support_reference_ids(incumbent.id)
    assert stats["noop"] == 1
    assert set(current_support).isdisjoint(old_support)
    [evidence] = await db.get_active_memory_support_evidence(
        incumbent.id,
        source_id="src-1",
    )
    assert evidence.excerpt == current_quote
    assert evidence.anchor.observation_revision_id == second.observation_revisions[0].id


@pytest.mark.asyncio
async def test_incremental_noop_reworded_primary_requires_exact_current_quote(
    db: Database,
) -> None:
    first = _projection(
        run_id="projection-primary-bad-quote-1",
        body="A7 is removed.",
    )
    await db.record_source_projection(first)
    incumbent = await _seed_incumbent_support(db, projection=first)
    await db.enable_lifecycle_gate("src-1")
    second = _projection(
        run_id="projection-primary-bad-quote-2",
        body="The A7 slot remains excluded.",
        prior=first.source_unit_revisions[0],
        prior_observations={
            first.observations[0].id: first.observation_revisions[0]
        },
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        relational=adapters.relational,
        vector=adapters.vector,
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=_SupportValidatingNoopClient(
            incumbent.id,
            supported=True,
            evidence_quote="A quote that is not in the current source.",
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="support validation lacks exact current PRIMARY evidence",
    ):
        await engine.apply_projected_lifecycle(
            projection=second,
            doc_id="confluence-123",
            raw_memories=[],
            doc_type="design-doc",
            project_key="ENG",
            repo_identifier=None,
            entity_ids=[],
            document_content=second.observation_revisions[0].content,
            update_mode="diff_guided",
            changed_hunks="primary wording changed",
            update_plan_stats=None,
            source_updated_at=datetime(2026, 7, 16, 10, 36, tzinfo=timezone.utc),
        )


@pytest.mark.asyncio
async def test_incremental_noop_invalidated_primary_creates_review(
    db: Database,
) -> None:
    first = _projection(
        run_id="projection-primary-invalid-1",
        body="A7 is removed.",
    )
    await db.record_source_projection(first)
    incumbent = await _seed_incumbent_support(db, projection=first)
    await db.enable_lifecycle_gate("src-1")
    second = _projection(
        run_id="projection-primary-invalid-2",
        body="A7 is now retained.",
        prior=first.source_unit_revisions[0],
        prior_observations={
            first.observations[0].id: first.observation_revisions[0]
        },
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        relational=adapters.relational,
        vector=adapters.vector,
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=_SupportValidatingNoopClient(
            incumbent.id,
            supported=False,
        ),
    )

    stats = await engine.apply_projected_lifecycle(
        projection=second,
        doc_id="confluence-123",
        raw_memories=[],
        doc_type="design-doc",
        project_key="ENG",
        repo_identifier=None,
        entity_ids=[],
        document_content=second.observation_revisions[0].content,
        update_mode="diff_guided",
        changed_hunks="removed -> retained",
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 16, 10, 36, tzinfo=timezone.utc),
    )

    assert stats["pending_review"] == 1
    current = await db.get_memory(incumbent.id)
    assert current is not None and current.status == "active"
    assert await db.get_active_memory_support_reference_ids(incumbent.id)


@pytest.mark.asyncio
async def test_persistent_indexless_replacement_creates_review_without_mutating_incumbent(
    db: Database,
) -> None:
    first = _projection(
        run_id="projection-indexless-replacement-1",
        body="A7 is removed.",
    )
    await db.record_source_projection(first)
    incumbent = await _seed_incumbent_support(db, projection=first)
    await db.enable_lifecycle_gate("src-1")
    second = _projection(
        run_id="projection-indexless-replacement-2",
        body="A7 is now retained.",
        prior=first.source_unit_revisions[0],
        prior_observations={
            first.observations[0].id: first.observation_revisions[0]
        },
    )
    client = _PersistentlyIndexlessReplacementClient(incumbent.id)
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        relational=adapters.relational,
        vector=adapters.vector,
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=client,
    )

    stats = await engine.apply_projected_lifecycle(
        projection=second,
        doc_id="confluence-123",
        raw_memories=[],
        doc_type="design-doc",
        project_key="ENG",
        repo_identifier=None,
        entity_ids=[],
        document_content=second.observation_revisions[0].content,
        update_mode="diff_guided",
        changed_hunks="removed -> retained",
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 16, 10, 36, tzinfo=timezone.utc),
    )

    assert client.calls == 2
    assert stats["pending_review"] == 1
    current = await db.get_memory(incumbent.id)
    assert current is not None and current.status == "active"
    assert await db.get_active_memory_support_reference_ids(incumbent.id)
    reviews = await db.list_lifecycle_reviews("src-1")
    assert len(reviews) == 1
    assert reviews[0].incumbent_memory_id == incumbent.id


@pytest.mark.asyncio
async def test_explicit_empty_revision_deterministically_removes_incumbent_support(
    db: Database,
) -> None:
    first = _projection(
        run_id="projection-empty-1",
        body="A7 is removed.",
    )
    await db.record_source_projection(first)
    incumbent = await _seed_incumbent_support(db, projection=first)
    await db.enable_lifecycle_gate("src-1")
    second = _projection(
        run_id="projection-empty-2",
        body="",
        prior=first.source_unit_revisions[0],
        prior_observations={
            first.observations[0].id: first.observation_revisions[0]
        },
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        relational=adapters.relational,
        vector=adapters.vector,
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=None,
    )

    stats = await engine.apply_projected_lifecycle(
        projection=second,
        doc_id="confluence-123",
        raw_memories=[],
        doc_type="design-doc",
        project_key="ENG",
        repo_identifier=None,
        entity_ids=[],
        document_content="",
        update_mode="diff_guided",
        changed_hunks="A7 is removed. -> empty",
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 16, 10, 36, tzinfo=timezone.utc),
    )

    assert stats["deleted"] == 1
    assert await db.get_active_memory_support_reference_ids(incumbent.id) == ()
    retired = await db.get_memory(incumbent.id)
    assert retired is not None
    assert retired.status == "retired"


async def _seed_jira_required_incumbent(
    db: Database,
    first: SourceProjection,
) -> Memory:
    incumbent = Memory(
        id="mem-jira-required",
        memory_type="decision",
        content="A7 is retained for regular payroll.",
        content_hash=content_hash("A7 is retained for regular payroll."),
        project_key="ENG",
    )
    await db.insert_memory(incumbent)
    await db.add_memory_source(
        incumbent.id,
        "confluence-123",
        "jira",
        "Decision: retain A7",
        source_updated_at=None,
    )
    primary = first.observations[1]
    required = first.observations[0]
    revisions = {
        item.observation_id: item for item in first.observation_revisions
    }
    unit = EvidenceUnit(
        id="eu-jira-required",
        source_id="src-1",
        doc_id="confluence-123",
        doc_revision_id=first.source_unit_revisions[0].id,
        source_type="jira",
        source_anchor=primary.id,
        source_lineage_id=first.source_units[0].id,
        project_key="ENG",
        visibility="workspace",
        owner_user_id=None,
        repo_identifier=None,
        content=revisions[primary.id].content,
        excerpt="Decision: retain A7",
        evidence_provenance=EvidenceContentProvenance.SOURCE_EXCERPT,
        access_context_hash="workspace-eng",
    )
    await db.upsert_evidence_unit(unit)
    references = await db.record_evidence_references(
        unit.id,
        (
            EvidenceReference(
                role=EvidenceRole.PRIMARY,
                anchor=SourceAnchor(
                    kind=AnchorKind.WHOLE_OBSERVATION,
                    observation_id=primary.id,
                    observation_revision_id=revisions[primary.id].id,
                ),
            ),
            EvidenceReference(
                role=EvidenceRole.REQUIRED,
                anchor=SourceAnchor(
                    kind=AnchorKind.WHOLE_OBSERVATION,
                    observation_id=required.id,
                    observation_revision_id=revisions[required.id].id,
                ),
            ),
        ),
    )
    for index, reference in enumerate(references):
        await db.upsert_memory_support_assertion(
            MemorySupportAssertion(
                id=f"support-jira-required-{index}",
                memory_id=incumbent.id,
                evidence_reference_id=reference.id or "",
                source_id="src-1",
                access_context_hash="workspace-eng",
            )
        )
    return incumbent


@pytest.mark.asyncio
async def test_partial_jira_projection_forces_disjoint_incumbent_keep(db: Database) -> None:
    first = _jira_projection(
        run_id="projection-jira-partial-fence-1",
        description="Initial issue description.",
        comment_body="Decision: retain A7",
    )
    await db.record_source_projection(first)
    incumbent = await _seed_incumbent_support(
        db,
        projection=first,
        memory_id="mem-jira-disjoint",
        memory_content="Decision: retain A7",
        observation_index=1,
        source_type="jira",
    )
    await db.enable_lifecycle_gate("src-1")
    old_support = await db.get_active_memory_support_reference_ids(incumbent.id)
    second = _jira_projection(
        run_id="projection-jira-partial-fence-2",
        description="Changed issue description.",
        comment_body="Decision: retain A7",
        comments_truncated=True,
        prior=first.source_unit_revisions[0],
        prior_observations={
            revision.observation_id: revision
            for revision in first.observation_revisions
        },
    )
    assert second.coverage.value == "partial_projection"
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        relational=adapters.relational,
        vector=adapters.vector,
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=_DeleteClient(incumbent.id),
    )

    stats = await engine.apply_projected_lifecycle(
        projection=second,
        doc_id="confluence-123",
        raw_memories=[],
        doc_type="ticket",
        project_key="ENG",
        repo_identifier=None,
        entity_ids=[],
        document_content="PAY-12 changed description",
        update_mode="diff_guided",
        changed_hunks="description changed; comment page is truncated",
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 16, tzinfo=timezone.utc),
    )

    current = await db.get_memory(incumbent.id)
    assert current is not None and current.status == "active"
    assert stats["deleted"] == 0
    assert stats["noop"] == 1
    assert await db.get_active_memory_support_reference_ids(incumbent.id) == old_support


@pytest.mark.asyncio
async def test_partial_jira_projection_admits_directly_affected_incumbent_delete(
    db: Database,
) -> None:
    first = _jira_projection(
        run_id="projection-jira-partial-affected-1",
        description="A7 is retained.",
        comment_body="Unrelated comment",
    )
    await db.record_source_projection(first)
    incumbent = await _seed_incumbent_support(
        db,
        projection=first,
        memory_id="mem-jira-affected",
        memory_content="A7 is retained.",
        observation_index=0,
        source_type="jira",
    )
    await db.enable_lifecycle_gate("src-1")
    second = _jira_projection(
        run_id="projection-jira-partial-affected-2",
        description="A7 is removed.",
        comment_body="Unrelated comment",
        comments_truncated=True,
        prior=first.source_unit_revisions[0],
        prior_observations={
            revision.observation_id: revision
            for revision in first.observation_revisions
        },
    )
    assert second.coverage.value == "partial_projection"
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        relational=adapters.relational,
        vector=adapters.vector,
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=_DeleteClient(incumbent.id),
    )

    stats = await engine.apply_projected_lifecycle(
        projection=second,
        doc_id="confluence-123",
        raw_memories=[],
        doc_type="ticket",
        project_key="ENG",
        repo_identifier=None,
        entity_ids=[],
        document_content="PAY-12 changed description",
        update_mode="diff_guided",
        changed_hunks="description changed; comment page is truncated",
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 16, tzinfo=timezone.utc),
    )

    current = await db.get_memory(incumbent.id)
    assert current is not None and current.status == "retired"
    assert stats["deleted"] == 1


@pytest.mark.asyncio
async def test_noop_revalidates_revised_required_jira_description(db: Database) -> None:
    first = _jira_projection(
        run_id="projection-jira-required-1",
        description="A7 applies only to regular payroll.",
    )
    await db.record_source_projection(first)
    incumbent = await _seed_jira_required_incumbent(db, first)
    await db.enable_lifecycle_gate("src-1")
    old_support = await db.get_active_memory_support_reference_ids(incumbent.id)
    revisions = {
        item.observation_id: item for item in first.observation_revisions
    }
    second = _jira_projection(
        run_id="projection-jira-required-2",
        description="A7 remains limited to regular payroll runs.",
        prior=first.source_unit_revisions[0],
        prior_observations=revisions,
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        relational=adapters.relational,
        vector=adapters.vector,
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=_SupportValidatingNoopClient(
            incumbent.id,
            supported=True,
        ),
    )

    stats = await engine.apply_projected_lifecycle(
        projection=second,
        doc_id="confluence-123",
        raw_memories=[],
        doc_type="ticket",
        project_key="ENG",
        repo_identifier=None,
        entity_ids=[],
        document_content="PAY-12",
        update_mode="diff_guided",
        changed_hunks="wording clarified; scope remains regular payroll",
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 15, 10, 36, tzinfo=timezone.utc),
    )

    current_support = await db.get_active_memory_support_reference_ids(incumbent.id)
    assert stats["noop"] == 1
    assert set(current_support).isdisjoint(old_support)
    evidence = await db.get_active_memory_support_evidence(
        incumbent.id,
        source_id="src-1",
    )
    assert {item.role for item in evidence} == {
        EvidenceRole.PRIMARY,
        EvidenceRole.REQUIRED,
    }
    current_revisions = {
        item.observation_id: item.id for item in second.observation_revisions
    }
    assert all(
        item.anchor.observation_revision_id
        == current_revisions[item.anchor.observation_id]
        for item in evidence
    )
    current = await db.get_current_source_unit_revision(first.source_units[0].id)
    assert current is not None
    assert current.id == second.source_unit_revisions[0].id
    [plan_row] = await db.db.execute_fetchall(
        "SELECT payload_json FROM lifecycle_plans WHERE source_id = ?",
        ("src-1",),
    )
    plan_payload = json.loads(str(plan_row["payload_json"]))
    attach = next(
        mutation
        for mutation in plan_payload["mutations"]
        if mutation["mutation_type"] == "attach_support"
    )
    assert attach["payload"]["support_validation"]["supported"] is True


@pytest.mark.asyncio
async def test_noop_with_invalidated_required_evidence_creates_review(db: Database) -> None:
    first = _jira_projection(
        run_id="projection-jira-invalid-required-1",
        description="A7 applies only to regular payroll.",
    )
    await db.record_source_projection(first)
    incumbent = await _seed_jira_required_incumbent(db, first)
    await db.enable_lifecycle_gate("src-1")
    second = _jira_projection(
        run_id="projection-jira-invalid-required-2",
        description="A7 now applies only to off-cycle payroll.",
        prior=first.source_unit_revisions[0],
        prior_observations={
            revision.observation_id: revision
            for revision in first.observation_revisions
        },
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        relational=adapters.relational,
        vector=adapters.vector,
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=_SupportValidatingNoopClient(
            incumbent.id,
            supported=False,
        ),
    )

    stats = await engine.apply_projected_lifecycle(
        projection=second,
        doc_id="confluence-123",
        raw_memories=[],
        doc_type="ticket",
        project_key="ENG",
        repo_identifier=None,
        entity_ids=[],
        document_content="PAY-12",
        update_mode="diff_guided",
        changed_hunks="regular -> off-cycle",
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 15, 10, 36, tzinfo=timezone.utc),
    )

    assert stats["pending_review"] == 1
    current = await db.get_memory(incumbent.id)
    assert current is not None and current.status == "active"
    assert await db.get_active_memory_support_reference_ids(incumbent.id)


@pytest.mark.asyncio
async def test_source_rebaseline_preserves_source_and_documents_but_resets_derived_lifecycle(
    db: Database,
) -> None:
    projection = _projection(run_id="projection-before-rebaseline", body="A7 is removed.")
    await db.record_source_projection(projection)
    incumbent = await _seed_incumbent_support(db, projection=projection)
    await db.set_source_subscription("src-1", "user-1", False)
    await db.enable_lifecycle_gate("src-1")
    await db.upsert_lifecycle_cutover_finding(
        LifecycleCutoverFinding(
            id="finding-before-rebaseline",
            source_id="src-1",
            memory_id=incumbent.id,
            reason=CutoverFindingReason.AMBIGUOUS_OBSERVATION,
            status=CutoverFindingStatus.OPEN,
            available_provenance={"doc_id": "confluence-123"},
            mapping_attempt={"strategy": "exact"},
        )
    )

    result = await db.rebaseline_source_lifecycle("src-1")

    assert result.retired_memory_ids == (incumbent.id,)
    assert await db.get_source("src-1") is not None
    assert await db.get_document("confluence-123") is not None
    assert await db.is_source_enabled_for_user("src-1", "user-1") is False
    reset_memory = await db.get_memory(incumbent.id)
    assert reset_memory is not None
    assert reset_memory.status == "retired"
    assert reset_memory.retirement_reason == "source_rebaseline"
    assert await db.get_active_memory_support_reference_ids(incumbent.id) == ()
    assert await db.get_source_projection(projection.run_id) is None
    assert await db.find_source_unit_by_document_id("src-1", "confluence-123") is None
    gate = await db.get_lifecycle_gate("src-1")
    assert gate.state is LifecycleGateState.GATED
    assert gate.reason == "source rebaseline requires a complete successful replay"
    finding = await db.get_lifecycle_cutover_finding("finding-before-rebaseline")
    assert finding is not None
    assert finding.status is CutoverFindingStatus.RESOLVED
    assert finding.mapping_attempt["resolution"] == "source_rebaseline"
    cleanup_tasks = await db.list_lifecycle_vector_tasks(source_id="src-1")
    assert len(cleanup_tasks) == 1
    assert cleanup_tasks[0].memory_id == incumbent.id
    assert cleanup_tasks[0].operation is LifecycleVectorOperation.DELETE


@pytest.mark.asyncio
async def test_source_rebaseline_preserves_failed_vector_cleanup_for_retry(db: Database) -> None:
    projection = _projection(run_id="projection-before-rebaseline-retry", body="A7 is removed.")
    await db.record_source_projection(projection)
    await _seed_incumbent_support(db, projection=projection)

    await db.rebaseline_source_lifecycle("src-1")
    [task] = await db.list_lifecycle_vector_tasks(source_id="src-1")
    await db.fail_lifecycle_vector_task(task.id, "temporary Chroma failure")

    # Retrying the relational reset has no remaining source associations, but
    # it must not erase the failed external cleanup task.
    await db.rebaseline_source_lifecycle("src-1")

    [retryable] = await db.list_lifecycle_vector_tasks(source_id="src-1")
    assert retryable.id == task.id
    assert retryable.status is LifecycleVectorTaskStatus.FAILED
    assert retryable.attempts == 1

    await db.fail_lifecycle_vector_task(task.id, "secondary lifecycle state failure")
    [retried] = await db.list_lifecycle_vector_tasks(source_id="src-1")
    assert retried.attempts == 2
    assert retried.error == "temporary Chroma failure"

    await db.complete_lifecycle_vector_task(task.id)
    async with db.db.execute(
        "SELECT status, error FROM lifecycle_vector_outbox WHERE id = ?",
        (task.id,),
    ) as cursor:
        recovered = await cursor.fetchone()
    assert recovered is not None
    assert (recovered["status"], recovered["error"]) == (
        "completed",
        "temporary Chroma failure",
    )


@pytest.mark.asyncio
async def test_memory_store_rebaseline_drains_only_its_source_vector_tasks() -> None:
    class _Relational:
        async def rebaseline_source_lifecycle(
            self,
            source_id: str,
            *,
            source_activity=None,
        ) -> SourceLifecycleResetResult:
            assert source_id == "src-1"
            assert source_activity is None
            return SourceLifecycleResetResult(
                retired_memory_ids=("mem-1",),
                retired_search_cleanup_required=True,
            )

    store = object.__new__(MemoryStore)
    store.relational = _Relational()
    store._operation_context = lambda **_kwargs: None
    store._emit = lambda *_args, **_kwargs: _async_none()
    drained: list[tuple[str | None, str | None]] = []

    async def record_delivery(
        lifecycle_plan_id: str | None = None,
        *,
        source_id: str | None = None,
    ) -> LifecycleVectorDeliveryResult:
        drained.append((lifecycle_plan_id, source_id))
        return LifecycleVectorDeliveryResult(
            state=LifecycleVectorDeliveryState.DELIVERED
        )

    store.attempt_lifecycle_vector_delivery = record_delivery

    assert await store.rebaseline_source_lifecycle("src-1") == ["mem-1"]
    assert drained == [(None, "src-1")]


async def _async_none() -> None:
    return None


@pytest.mark.asyncio
async def test_cross_source_keep_persists_provenance_and_survives_other_source_rebaseline(
    db: Database,
) -> None:
    first = _projection(run_id="projection-cross-source-1", body="A7 is removed.")
    await db.record_source_projection(first)
    incumbent = await _seed_incumbent_support(db, projection=first)
    incumbent = await db.get_memory(incumbent.id)
    assert incumbent is not None

    await db.upsert_source(
        id="src-2",
        type="confluence",
        name="Second source",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="owner-2",
    )
    now = datetime(2026, 7, 15, tzinfo=timezone.utc).isoformat()
    await db.db.execute(
        """INSERT INTO documents (
               doc_id, source, source_url, title, space_or_project,
               last_modified, version, content_hash, last_synced
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "confluence-456",
            "src-2",
            "https://example.test/456",
            "Second page",
            "ENG",
            now,
            "1",
            "h2",
            now,
        ),
    )
    await db.db.commit()
    await db.enable_lifecycle_gate("src-2")
    second = _projection(
        run_id="projection-cross-source-2",
        body="A7 is removed.",
        item_id="confluence-456",
        source_id="src-2",
    )
    raw = RawMemory(
        content=incumbent.content,
        memory_type=incumbent.memory_type,
        confidence=incumbent.confidence,
        tags=list(incumbent.tags),
        evidence_quote="A7 is removed.",
    )
    evidence = build_projected_claim_evidence(
        projection=second,
        raw_memories=(raw,),
        doc_id="confluence-456",
        source_type="confluence",
        project_key="ENG",
        visibility="workspace",
        owner_user_id=None,
        repo_identifier=None,
        access_context_hash="workspace-eng",
        extractor_run_id=second.run_id,
    )
    delta = second.deltas[0]
    source_one_refs = await db.get_active_memory_support_reference_ids(incumbent.id)
    plan = build_lifecycle_plan(
        plan_id="plan-cross-source-keep",
        scope=ReconciliationScope(
            id="scope-cross-source-keep",
            source_id="src-2",
            source_unit_id=delta.source_unit_id,
            base_unit_revision_id=delta.previous_unit_revision_id,
            target_unit_revision_id=delta.current_unit_revision_id,
        ),
        gate_state=LifecycleGateState.ENABLED,
        operations=(
            ReconcileOperation(
                action=ReconcileAction.NOOP,
                memory_id=incumbent.id,
                memory=raw,
                reason="independent source corroborates the claim",
            ),
        ),
        incumbents={incumbent.id: incumbent},
        source_support_reference_ids={incumbent.id: ()},
        all_active_support_reference_ids={incumbent.id: source_one_refs},
        support_set_hashes={
            incumbent.id: await db.get_memory_support_set_hash(incumbent.id)
        },
        observation_revision_ids=tuple(
            revision.id for revision in second.observation_revisions
        ),
        new_evidence_reference_ids=(),
        evidence_reference_ids_by_claim_hash=evidence.reference_ids_by_claim_hash,
        defaults=NewMemoryDefaults(
            visibility="workspace",
            owner_user_id=None,
            project_key="ENG",
            repo_identifier=None,
            doc_id="confluence-456",
            source_type="confluence",
            access_context_hash="workspace-eng",
        ),
        evidence_units=evidence.units,
        evidence_references=evidence.references,
    )

    await db.apply_source_projection_lifecycle(second, plan)

    sources = await db.get_memory_sources(incumbent.id)
    assert {(item.source_id, item.doc_id) for item in sources} == {
        ("src-1", "confluence-123"),
        ("src-2", "confluence-456"),
    }
    assert (await db.get_memory(incumbent.id)).corroboration_count == 2

    await db.rebaseline_source_lifecycle("src-1")

    surviving = await db.get_memory(incumbent.id)
    assert surviving is not None
    assert surviving.status == "active"
    assert surviving.corroboration_count == 1
    assert {(item.source_id, item.doc_id) for item in await db.get_memory_sources(incumbent.id)} == {
        ("src-2", "confluence-456")
    }


@pytest.mark.asyncio
async def test_cross_source_semantic_equivalent_add_reuses_memory_id_and_attaches_support(
    db: Database,
) -> None:
    first = _projection(run_id="projection-equivalent-source-1", body="A7 is removed.")
    await db.record_source_projection(first)
    incumbent = await _seed_incumbent_support(db, projection=first)
    incumbent = await db.get_memory(incumbent.id)
    assert incumbent is not None
    await db.upsert_source(
        id="src-2",
        type="confluence",
        name="Independent Engineering",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="owner-1",
    )
    now = datetime(2026, 7, 15, tzinfo=timezone.utc).isoformat()
    await db.db.execute(
        """INSERT INTO documents (
               doc_id, source, source_url, title, space_or_project,
               last_modified, version, content_hash, last_synced
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "confluence-456",
            "src-2",
            "https://example.test/456",
            "Independent Page",
            "ENG",
            now,
            "1",
            "h2",
            now,
        ),
    )
    await db.db.commit()
    second = _projection(
        run_id="projection-equivalent-source-2",
        body="A7 remains excluded.",
        item_id="confluence-456",
        source_id="src-2",
    )
    raw = RawMemory(
        content="A7 remains excluded.",
        memory_type="decision",
        confidence=0.9,
        evidence_quote="A7 remains excluded.",
        extraction_context="A7 remains excluded.",
        source_observation_id="obs-from-unrelated-source",
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        relational=adapters.relational,
        vector=adapters.vector,
        db=db,
        memory_store=_EquivalentMemoryStore(db, incumbent),
        structured_llm_client=_SemanticEquivalentClient(),
    )

    stats = await engine.apply_projected_lifecycle(
        projection=second,
        doc_id="confluence-456",
        raw_memories=[raw],
        doc_type="design-doc",
        project_key=None,
        repo_identifier=None,
        entity_ids=[],
        document_content="A7 remains excluded.",
        update_mode="full_document",
        changed_hunks=None,
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 15, 11, 0, tzinfo=timezone.utc),
    )

    assert stats["added"] == 0
    assert stats["corroborated"] == 1
    sources = await db.get_memory_sources(incumbent.id)
    assert {source.source_id for source in sources} == {"src-1", "src-2"}
    plan_rows = await db.db.execute_fetchall(
        "SELECT payload_json FROM lifecycle_plans WHERE source_id = ?",
        ("src-2",),
    )
    assert len(plan_rows) == 1
    plan_payload = json.loads(str(plan_rows[0]["payload_json"]))
    attach = next(
        mutation
        for mutation in plan_payload["mutations"]
        if mutation["mutation_type"] == "attach_support"
    )
    assert attach["payload"]["equivalence_proof"] == {
        "candidate_content_hash": content_hash("A7 remains excluded."),
        "incumbent_content_hash": content_hash("A7 is removed."),
        "method": "structured_classifier",
        "model": engine.llm_model,
        "reason": "Both claims state that A7 is excluded.",
    }
    support = await db.get_active_memory_support_evidence(
        incumbent.id,
        source_id="src-2",
    )
    assert len(support) == 1
    assert support[0].anchor.observation_revision_id == second.observation_revisions[0].id


@pytest.mark.asyncio
async def test_same_source_cross_unit_semantic_equivalent_claim_reuses_memory_id(
    db: Database,
) -> None:
    first = _projection(run_id="projection-same-source-semantic-1", body="A7 is removed.")
    await db.record_source_projection(first)
    incumbent = await _seed_incumbent_support(db, projection=first)
    incumbent = await db.get_memory(incumbent.id)
    assert incumbent is not None
    await db.enable_lifecycle_gate("src-1")
    now = datetime(2026, 7, 15, tzinfo=timezone.utc).isoformat()
    await db.db.execute(
        """INSERT INTO documents (
               doc_id, source, source_url, title, space_or_project,
               last_modified, version, content_hash, last_synced
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "confluence-456",
            "src-1",
            "https://example.test/456",
            "Independent confirmation",
            "ENG",
            now,
            "1",
            "h2",
            now,
        ),
    )
    await db.db.commit()
    second = _projection(
        run_id="projection-same-source-semantic-2",
        body="A7 remains excluded.",
        item_id="confluence-456",
        page_id="456",
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        relational=adapters.relational,
        vector=adapters.vector,
        db=db,
        memory_store=_EquivalentMemoryStore(db, incumbent),
        structured_llm_client=_SemanticEquivalentClient(),
    )

    stats = await engine.apply_projected_lifecycle(
        projection=second,
        doc_id="confluence-456",
        raw_memories=[
            RawMemory(
                content="A7 remains excluded.",
                memory_type="decision",
                confidence=0.9,
                evidence_quote="A7 remains excluded.",
                source_observation_id=second.observations[0].id,
            )
        ],
        doc_type="design-doc",
        project_key="ENG",
        repo_identifier=None,
        entity_ids=[],
        document_content="A7 remains excluded.",
        update_mode="full_document",
        changed_hunks=None,
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 15, 11, 0, tzinfo=timezone.utc),
    )

    assert stats["added"] == 0
    assert stats["corroborated"] == 1
    assert len(await db.list_memories(source="src-1", status="active")) == 1
    assert {
        (source.source_id, source.doc_id)
        for source in await db.get_memory_sources(incumbent.id)
    } == {
        ("src-1", "confluence-123"),
        ("src-1", "confluence-456"),
    }
    supports = await db.get_active_memory_support_evidence(
        incumbent.id,
        source_id="src-1",
    )
    assert len(supports) == 2
    lineage_rows = await db.db.execute_fetchall(
        """SELECT COUNT(DISTINCT EU.SOURCE_LINEAGE_ID) AS lineage_count
             FROM MEMORY_SUPPORT_ASSERTIONS MSA
             JOIN EVIDENCE_REFERENCES ER ON ER.ID = MSA.EVIDENCE_REFERENCE_ID
             JOIN EVIDENCE_UNITS EU ON EU.ID = ER.EVIDENCE_UNIT_ID
            WHERE MSA.MEMORY_ID = ? AND MSA.ACTIVE = 1""",
        (incumbent.id,),
    )
    assert lineage_rows[0]["lineage_count"] == 2


@pytest.mark.asyncio
async def test_same_source_cross_unit_exact_claim_reuses_memory_id_and_preserves_both_lineages(
    db: Database,
) -> None:
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        relational=adapters.relational,
        vector=adapters.vector,
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=None,
    )
    first = _projection(
        run_id="projection-same-source-exact-1",
        body="A7 is retained for regular payroll.",
    )
    first_raw = RawMemory(
        content="A7 is retained for regular payroll.",
        memory_type="decision",
        confidence=0.95,
        evidence_quote="A7 is retained for regular payroll.",
        source_observation_id=first.observations[0].id,
    )
    first_stats = await engine.apply_projected_lifecycle(
        projection=first,
        doc_id="confluence-123",
        raw_memories=[first_raw],
        doc_type="design-doc",
        project_key="ENG",
        repo_identifier=None,
        entity_ids=[],
        document_content="A7 is retained for regular payroll.",
        update_mode="full_document",
        changed_hunks=None,
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 15, 10, 0, tzinfo=timezone.utc),
    )
    now = datetime(2026, 7, 15, tzinfo=timezone.utc).isoformat()
    await db.db.execute(
        """INSERT INTO documents (
               doc_id, source, source_url, title, space_or_project,
               last_modified, version, content_hash, last_synced
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "confluence-456",
            "src-1",
            "https://example.test/456",
            "Second page",
            "ENG",
            now,
            "1",
            "h2",
            now,
        ),
    )
    await db.db.commit()
    second = _projection(
        run_id="projection-same-source-exact-2",
        body="A7 is retained for regular payroll.",
        item_id="confluence-456",
        page_id="456",
    )
    second_raw = replace(
        first_raw,
        source_observation_id=second.observations[0].id,
    )

    second_stats = await engine.apply_projected_lifecycle(
        projection=second,
        doc_id="confluence-456",
        raw_memories=[second_raw],
        doc_type="design-doc",
        project_key="ENG",
        repo_identifier=None,
        entity_ids=[],
        document_content="A7 is retained for regular payroll.",
        update_mode="full_document",
        changed_hunks=None,
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 15, 11, 0, tzinfo=timezone.utc),
    )

    memories = await db.list_memories(source="src-1", status="active")
    assert first_stats["added"] == 1
    assert second_stats["added"] == 0
    assert second_stats["corroborated"] == 1
    assert len(memories) == 1
    memory = memories[0]
    assert {(source.source_id, source.doc_id) for source in await db.get_memory_sources(memory.id)} == {
        ("src-1", "confluence-123"),
        ("src-1", "confluence-456"),
    }
    lineage_rows = await db.db.execute_fetchall(
        """SELECT COUNT(DISTINCT EU.SOURCE_LINEAGE_ID) AS lineage_count,
                  COUNT(DISTINCT EU.DOC_ID) AS document_count
             FROM MEMORY_SUPPORT_ASSERTIONS MSA
             JOIN EVIDENCE_REFERENCES ER ON ER.ID = MSA.EVIDENCE_REFERENCE_ID
             JOIN EVIDENCE_UNITS EU ON EU.ID = ER.EVIDENCE_UNIT_ID
            WHERE MSA.MEMORY_ID = ? AND MSA.ACTIVE = 1""",
        (memory.id,),
    )
    assert lineage_rows[0]["lineage_count"] == 2
    assert lineage_rows[0]["document_count"] == 2
    plan_rows = await db.db.execute_fetchall(
        "SELECT payload_json FROM lifecycle_plans WHERE source_unit_id = ?",
        (second.deltas[0].source_unit_id,),
    )
    assert len(plan_rows) == 1
    payload = json.loads(str(plan_rows[0]["payload_json"]))
    attach = next(
        mutation
        for mutation in payload["mutations"]
        if mutation["mutation_type"] == "attach_support"
    )
    assert attach["memory_id"] == memory.id
    assert attach["payload"]["equivalence_proof"]["method"] == "exact_content"


@pytest.mark.asyncio
async def test_cross_source_exact_claim_reuses_memory_without_llm_and_preserves_both_lineages(
    db: Database,
) -> None:
    first = _projection(
        run_id="projection-cross-source-exact-1",
        body="A7 is retained for regular payroll.",
    )
    await db.record_source_projection(first)
    incumbent = await _seed_incumbent_support(
        db,
        projection=first,
        memory_content="A7 is retained for regular payroll.",
    )
    await db.upsert_source(
        id="src-2",
        type="confluence",
        name="Independent Engineering",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="owner-1",
    )
    now = datetime(2026, 7, 15, tzinfo=timezone.utc).isoformat()
    await db.db.execute(
        """INSERT INTO documents (
               doc_id, source, source_url, title, space_or_project,
               last_modified, version, content_hash, last_synced
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "confluence-456",
            "src-2",
            "https://example.test/456",
            "Independent Page",
            "ENG",
            now,
            "1",
            "h2",
            now,
        ),
    )
    await db.db.commit()
    second = _projection(
        run_id="projection-cross-source-exact-2",
        body="A7 is retained for regular payroll.",
        item_id="confluence-456",
        page_id="456",
        source_id="src-2",
    )
    raw = RawMemory(
        content="A7 is retained for regular payroll.",
        memory_type="decision",
        confidence=0.95,
        evidence_quote="A7 is retained for regular payroll.",
        source_observation_id=second.observations[0].id,
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        relational=adapters.relational,
        vector=adapters.vector,
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=None,
    )

    stats = await engine.apply_projected_lifecycle(
        projection=second,
        doc_id="confluence-456",
        raw_memories=[raw],
        doc_type="design-doc",
        project_key="ENG",
        repo_identifier=None,
        entity_ids=[],
        document_content=raw.content,
        update_mode="full_document",
        changed_hunks=None,
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 15, 11, 0, tzinfo=timezone.utc),
    )

    memories = await db.list_memories(status="active")
    assert stats["added"] == 0
    assert stats["corroborated"] == 1
    assert [memory.id for memory in memories] == [incumbent.id]
    support = await db.get_active_memory_support_evidence(incumbent.id)
    assert {item.source_id for item in support} == {"src-1", "src-2"}
    assert {
        item.anchor.observation_revision_id for item in support
    } == {
        first.observation_revisions[0].id,
        second.observation_revisions[0].id,
    }


@pytest.mark.asyncio
async def test_ordinary_exact_admission_preserves_agent_claim_identity(
    db: Database,
) -> None:
    claim_text = "A7 is retained for regular payroll."
    now = datetime(2026, 7, 15, 9, 0, tzinfo=timezone.utc)
    await db.upsert_source(
        id="src-agent",
        type="agent_session",
        name="Agent Knowledge",
        config_json="{}",
        access_policy="private",
        owner_user_id="owner-1",
    )
    await db.upsert_source(
        id="src-private-doc",
        type="confluence",
        name="Private Engineering",
        config_json="{}",
        access_policy="private",
        owner_user_id="owner-1",
    )
    await db.db.execute(
        """INSERT INTO documents (
               doc_id, source, source_url, title, space_or_project,
               last_modified, version, content_hash, last_synced
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "private-confluence-123",
            "src-private-doc",
            "https://example.test/private/123",
            "Private page",
            "ENG",
            now.isoformat(),
            "1",
            "private-hash",
            now.isoformat(),
        ),
    )
    agent_memory = Memory(
        id="mem-agent-explicit-claim",
        memory_type="decision",
        content=claim_text,
        content_hash=content_hash(claim_text),
        visibility="private",
        owner_user_id="owner-1",
        project_key="ENG",
        repo_identifier="repo-a",
    )
    await db.insert_memory(agent_memory)
    await db.upsert_agent_concept(
        concept_id="agent-concept-explicit",
        source_id="src-agent",
        owner_user_id="owner-1",
        workspace="/workspace",
        repo_identifier="repo-a",
        concept_type="decision",
        concept_path="decisions/a7.md",
        title="A7 handling",
        markdown_body=claim_text,
        frontmatter={},
        observed_at=now,
    )
    await db.upsert_agent_claim(
        claim_id="agent-claim-explicit",
        concept_id="agent-concept-explicit",
        display_anchor="A7 handling",
        claim_text=claim_text,
        memory_type="decision",
        tags=[],
        confidence=0.95,
        memory_id=agent_memory.id,
        observed_at=now,
    )

    assert (
        await db.find_active_exact_claim_candidate(
            agent_memory.content_hash,
            visibility=agent_memory.visibility,
            owner_user_id=agent_memory.owner_user_id,
            repo_identifier=agent_memory.repo_identifier,
        )
        is None
    )

    projection = _projection(
        run_id="projection-private-doc-after-agent-claim",
        body=claim_text,
        item_id="private-confluence-123",
        page_id="private-123",
        source_id="src-private-doc",
    )
    raw = RawMemory(
        content=claim_text,
        memory_type="decision",
        confidence=0.95,
        evidence_quote=claim_text,
        source_observation_id=projection.observations[0].id,
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        relational=adapters.relational,
        vector=adapters.vector,
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=None,
    )

    stats = await engine.apply_projected_lifecycle(
        projection=projection,
        doc_id="private-confluence-123",
        raw_memories=[raw],
        doc_type="design-doc",
        project_key="ENG",
        repo_identifier="repo-a",
        entity_ids=[],
        document_content=claim_text,
        update_mode="full_document",
        changed_hunks=None,
        update_plan_stats=None,
        source_updated_at=now,
        user_id="owner-1",
    )

    ordinary_memory = await db.find_active_exact_claim_candidate(
        agent_memory.content_hash,
        visibility=agent_memory.visibility,
        owner_user_id=agent_memory.owner_user_id,
        repo_identifier=agent_memory.repo_identifier,
    )
    claim = await db.get_agent_claim("agent-claim-explicit")
    exact_rows = await db.db.execute_fetchall(
        """SELECT id FROM memories
           WHERE content_hash = ? AND status = 'active'
           ORDER BY id""",
        (agent_memory.content_hash,),
    )
    assert stats["added"] == 1
    assert ordinary_memory is not None
    assert ordinary_memory.id != agent_memory.id
    assert claim is not None
    assert claim["memory_id"] == agent_memory.id
    assert {row["id"] for row in exact_rows} == {
        agent_memory.id,
        ordinary_memory.id,
    }


@pytest.mark.asyncio
async def test_stale_parallel_cross_unit_create_fails_closed_before_duplicate_write(
    db: Database,
) -> None:
    now = datetime(2026, 7, 15, tzinfo=timezone.utc).isoformat()
    await db.db.execute(
        """INSERT INTO documents (
               doc_id, source, source_url, title, space_or_project,
               last_modified, version, content_hash, last_synced
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "confluence-456",
            "src-1",
            "https://example.test/456",
            "Second page",
            "ENG",
            now,
            "1",
            "h2",
            now,
        ),
    )
    await db.db.commit()
    projections = (
        (
            _projection(
                run_id="projection-stale-exact-1",
                body="A7 is retained for regular payroll.",
            ),
            "confluence-123",
        ),
        (
            _projection(
                run_id="projection-stale-exact-2",
                body="A7 is retained for regular payroll.",
                item_id="confluence-456",
                page_id="456",
            ),
            "confluence-456",
        ),
    )

    def plan_for(projection: SourceProjection, doc_id: str) -> LifecyclePlan:
        raw = RawMemory(
            content="A7 is retained for regular payroll.",
            memory_type="decision",
            confidence=0.95,
            evidence_quote="A7 is retained for regular payroll.",
            source_observation_id=projection.observations[0].id,
        )
        access_context_hash = lifecycle_access_context_hash(
            visibility="workspace",
            owner_user_id=None,
            project_key="ENG",
            repo_identifier=None,
        )
        evidence = build_projected_claim_evidence(
            projection=projection,
            raw_memories=(raw,),
            doc_id=doc_id,
            source_type="confluence",
            project_key="ENG",
            visibility="workspace",
            owner_user_id=None,
            repo_identifier=None,
            access_context_hash=access_context_hash,
            extractor_run_id=projection.run_id,
        )
        delta = projection.deltas[0]
        scope = ReconciliationScope(
            id=f"scope:{projection.run_id}",
            source_id=projection.source_id,
            source_unit_id=delta.source_unit_id,
            base_unit_revision_id=delta.previous_unit_revision_id,
            target_unit_revision_id=delta.current_unit_revision_id,
        )
        return build_lifecycle_plan(
            plan_id=lifecycle_plan_id(scope),
            scope=scope,
            gate_state=LifecycleGateState.GATED,
            operations=(
                ReconcileOperation(
                    action=ReconcileAction.ADD,
                    memory=raw,
                ),
            ),
            incumbents={},
            source_support_reference_ids={},
            all_active_support_reference_ids={},
            support_set_hashes={},
            observation_revision_ids=tuple(
                revision.id for revision in projection.observation_revisions
            ),
            new_evidence_reference_ids=(),
            evidence_reference_ids_by_claim_hash=evidence.reference_ids_by_claim_hash,
            defaults=NewMemoryDefaults(
                visibility="workspace",
                owner_user_id=None,
                project_key="ENG",
                repo_identifier=None,
                doc_id=doc_id,
                source_type="confluence",
                access_context_hash=access_context_hash,
            ),
            evidence_units=evidence.units,
            evidence_references=evidence.references,
        )

    first_projection, first_doc_id = projections[0]
    second_projection, second_doc_id = projections[1]
    first_plan = plan_for(first_projection, first_doc_id)
    stale_second_plan = plan_for(second_projection, second_doc_id)

    await db.apply_source_projection_lifecycle(first_projection, first_plan)
    with pytest.raises(
        ValueError,
        match="exact claim stale guard failed",
    ):
        await db.apply_source_projection_lifecycle(
            second_projection,
            stale_second_plan,
        )

    memories = await db.list_memories(source="src-1", status="active")
    assert len(memories) == 1
    async with db.db.execute(
        "SELECT COUNT(*) AS total FROM source_units WHERE id = ?",
        (second_projection.deltas[0].source_unit_id,),
    ) as cursor:
        row = await cursor.fetchone()
    assert row["total"] == 0


@pytest.mark.asyncio
async def test_new_projected_memory_persists_explicit_source_observation_support(
    db: Database,
) -> None:
    projection = _jira_projection(
        run_id="projection-jira-new-memory",
        description="A7 applies only to regular payroll.",
        comment_body="The rollout note is unrelated.",
    )
    primary = projection.observations[0]
    primary_revision = projection.observation_revisions[0]
    raw = RawMemory(
        content="A7 applies only to regular payroll.",
        memory_type="decision",
        confidence=0.95,
        evidence_quote="A7 applies only to regular payroll.",
        extraction_context="A7 applies only to regular payroll.",
        evidence_anchor="projection_batch",
        source_observation_id=primary.id,
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        relational=adapters.relational,
        vector=adapters.vector,
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=_SemanticEquivalentClient(),
    )

    stats = await engine.apply_projected_lifecycle(
        projection=projection,
        doc_id="confluence-123",
        raw_memories=[raw],
        doc_type="ticket",
        project_key="ENG",
        repo_identifier=None,
        entity_ids=[],
        document_content="PAY-12",
        update_mode="full_document",
        changed_hunks=None,
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 15, 11, 0, tzinfo=timezone.utc),
    )

    [memory] = await db.list_memories(source="src-1", status="active")
    support = await db.get_active_memory_support_evidence(memory.id, source_id="src-1")
    assert stats["added"] == 1
    assert len(support) == 1
    assert support[0].anchor.observation_id == primary.id
    assert support[0].anchor.observation_revision_id == primary_revision.id


@pytest.mark.asyncio
async def test_new_projected_memory_commit_survives_vector_outbox_delivery_failure(
    db: Database,
) -> None:
    projection = _projection(
        run_id="projection-vector-outbox-failure",
        body="A7 applies only to regular payroll.",
    )
    raw = RawMemory(
        content="A7 applies only to regular payroll.",
        memory_type="decision",
        confidence=0.95,
        evidence_quote="A7 applies only to regular payroll.",
        extraction_context="A7 applies only to regular payroll.",
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        relational=adapters.relational,
        vector=adapters.vector,
        db=db,
        memory_store=_FailingOutboxDrainer(db),
        structured_llm_client=_SemanticEquivalentClient(),
    )

    stats = await engine.apply_projected_lifecycle(
        projection=projection,
        doc_id="confluence-123",
        raw_memories=[raw],
        doc_type="design-doc",
        project_key="ENG",
        repo_identifier=None,
        entity_ids=[],
        document_content=projection.observation_revisions[0].content,
        update_mode="full_document",
        changed_hunks=None,
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 15, 11, 0, tzinfo=timezone.utc),
    )

    [memory] = await db.list_memories(source="src-1", status="active")
    support = await db.get_active_memory_support_evidence(memory.id, source_id="src-1")
    assert stats["added"] == 1
    assert stats["vector_delivery_pending"] == 1
    assert len(support) == 1
    assert support[0].anchor.observation_revision_id == projection.observation_revisions[0].id


@pytest.mark.asyncio
async def test_generic_document_delete_rejects_active_projected_support(
    db: Database,
) -> None:
    projection = _projection(
        run_id="projection-delete-fail-closed",
        body="A7 applies only to regular payroll.",
    )
    await db.record_source_projection(projection)
    incumbent = await _seed_incumbent_support(db, projection=projection)

    with pytest.raises(
        ValueError,
        match="active projected support remains",
    ):
        await db.delete_document("confluence-123")

    current = await db.get_memory(incumbent.id)
    assert current is not None and current.status == "active"
    assert await db.get_document("confluence-123") is not None
    assert await db.get_active_memory_support_reference_ids(incumbent.id)
    assert await db.get_evidence_unit(
        (await db.get_active_memory_support_evidence(incumbent.id))[0].evidence_unit_id
    ) is not None


@pytest.mark.asyncio
async def test_rebaseline_replay_reuses_memory_with_explicit_observation_support(
    db: Database,
) -> None:
    projection = _jira_projection(
        run_id="projection-jira-before-rebaseline",
        description="A7 applies only to regular payroll.",
        comment_body="The rollout note is unrelated.",
    )
    primary = projection.observations[0]
    raw = RawMemory(
        content="A7 applies only to regular payroll.",
        memory_type="decision",
        confidence=0.95,
        evidence_quote="A7 applies only to regular payroll.",
        extraction_context="A7 applies only to regular payroll.",
        evidence_anchor="projection_batch",
        source_observation_id=primary.id,
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        relational=adapters.relational,
        vector=adapters.vector,
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=_SemanticEquivalentClient(),
    )
    arguments = {
        "doc_id": "confluence-123",
        "raw_memories": [raw],
        "doc_type": "ticket",
        "project_key": "ENG",
        "repo_identifier": None,
        "entity_ids": [],
        "document_content": "PAY-12",
        "update_mode": "full_document",
        "changed_hunks": None,
        "update_plan_stats": None,
        "source_updated_at": datetime(2026, 7, 15, 11, 0, tzinfo=timezone.utc),
    }

    first = await engine.apply_projected_lifecycle(projection=projection, **arguments)
    [memory] = await db.list_memories(source="src-1", status="active")
    assert first["added"] == 1
    assert await db.get_active_memory_support_evidence(memory.id, source_id="src-1")

    await db.rebaseline_source_lifecycle("src-1")
    replay = _jira_projection(
        run_id="projection-jira-after-rebaseline",
        description="A7 applies only to regular payroll.",
        comment_body="The rollout note is unrelated.",
    )
    second = await engine.apply_projected_lifecycle(projection=replay, **arguments)

    replayed = await db.get_memory(memory.id)
    support = await db.get_active_memory_support_evidence(memory.id, source_id="src-1")
    assert replayed is not None and replayed.status == "active"
    assert second["reactivated"] == 1
    assert second["corroborated"] == 1
    assert second["contradictions_found"] == 0
    assert len(support) == 1
    assert support[0].anchor.observation_id == primary.id
    plan_rows = await db.db.execute_fetchall(
        "SELECT payload_json FROM lifecycle_plans WHERE id = ?",
        (
            lifecycle_plan_id(
                ReconciliationScope(
                    id=f"scope:{replay.run_id}",
                    source_id=replay.source_id,
                    source_unit_id=replay.deltas[0].source_unit_id,
                    base_unit_revision_id=replay.deltas[0].previous_unit_revision_id,
                    target_unit_revision_id=replay.deltas[0].current_unit_revision_id,
                )
            ),
        ),
    )
    assert len(plan_rows) == 1
    plan_payload = json.loads(str(plan_rows[0]["payload_json"]))
    assert [item["mutation_type"] for item in plan_payload["mutations"]] == [
        "reactivate_memory",
        "attach_support",
        "refresh_memory_index",
    ]
    assert plan_payload["mutations"][0]["payload"]["expected_content_hash"] == memory.content_hash


@pytest.mark.asyncio
async def test_reactivation_rejects_stale_content_hash_and_keeps_memory_retired(
    db: Database,
) -> None:
    memory = Memory(
        id="mem-rebaseline-stale",
        memory_type="decision",
        content="A7 applies only to regular payroll.",
        content_hash=content_hash("A7 applies only to regular payroll."),
        status="retired",
        retirement_reason="source_rebaseline",
    )
    await db.insert_memory(memory)
    plan = LifecyclePlan(
        id="plan-rebaseline-stale",
        scope=ReconciliationScope(
            id="scope-rebaseline-stale",
            source_id="src-1",
            source_unit_id="unit-rebaseline-stale",
            base_unit_revision_id=None,
            target_unit_revision_id=None,
        ),
        gate_state=LifecycleGateState.GATED,
        coverage_proof=CoverageProof((), (), (), ()),
        stale_guard=StaleGuard((), {}),
        mutations=(
            LifecycleMutation(
                LifecycleMutationType.REACTIVATE_MEMORY,
                memory_id=memory.id,
                source_id="src-1",
                payload={"expected_content_hash": "stale-content-hash"},
            ),
        ),
    )

    with pytest.raises(ValueError, match="reactivate Memory stale guard failed"):
        await db.apply_lifecycle_plan(plan)

    persisted = await db.get_memory(memory.id)
    assert persisted is not None and persisted.status == "retired"
    assert await db.get_lifecycle_plan_payload(plan.id) is None


@pytest.mark.asyncio
async def test_reactivation_without_new_source_support_rolls_back(
    db: Database,
) -> None:
    memory = Memory(
        id="mem-rebaseline-no-support",
        memory_type="decision",
        content="A7 applies only to regular payroll.",
        content_hash=content_hash("A7 applies only to regular payroll."),
        status="retired",
        retirement_reason="source_rebaseline",
    )
    await db.insert_memory(memory)
    plan = LifecyclePlan(
        id="plan-rebaseline-no-support",
        scope=ReconciliationScope(
            id="scope-rebaseline-no-support",
            source_id="src-1",
            source_unit_id="unit-rebaseline-no-support",
            base_unit_revision_id=None,
            target_unit_revision_id=None,
        ),
        gate_state=LifecycleGateState.GATED,
        coverage_proof=CoverageProof((), (), (), ()),
        stale_guard=StaleGuard((), {}),
        mutations=(
            LifecycleMutation(
                LifecycleMutationType.REACTIVATE_MEMORY,
                memory_id=memory.id,
                source_id="src-1",
                payload={"expected_content_hash": memory.content_hash},
            ),
        ),
    )

    with pytest.raises(ValueError, match="activated Memory without source support"):
        await db.apply_lifecycle_plan(plan)

    persisted = await db.get_memory(memory.id)
    assert persisted is not None and persisted.status == "retired"
    assert await db.get_lifecycle_plan_payload(plan.id) is None


@pytest.mark.asyncio
async def test_rebaseline_reset_retires_legacy_edge_identified_by_document_source(
    db: Database,
) -> None:
    memory = Memory(
        id="mem-legacy-null-source-edge",
        memory_type="fact",
        content="Legacy Jira fact without canonical source identity.",
        content_hash=content_hash("Legacy Jira fact without canonical source identity."),
        project_key="ENG",
    )
    await db.insert_memory(memory)
    await db.add_memory_source(
        memory.id,
        "confluence-123",
        "confluence",
        memory.content,
        source_updated_at=None,
    )
    await db.db.execute(
        "UPDATE memory_sources SET source_id = NULL WHERE memory_id = ?",
        (memory.id,),
    )
    await db.db.commit()

    result = await db.rebaseline_source_lifecycle("src-1")

    retired = await db.get_memory(memory.id)
    assert result.retired_memory_ids == (memory.id,)
    assert retired is not None and retired.status == "retired"
    assert await db.get_memory_sources(memory.id) == []


@pytest.mark.asyncio
async def test_post_cutover_direct_source_memory_write_is_rejected(db: Database) -> None:
    await db.enable_lifecycle_gate("src-1")
    memory = Memory(
        id="mem-direct-bypass",
        memory_type="fact",
        content="This bypass has no Source Observation lineage.",
        content_hash=content_hash("This bypass has no Source Observation lineage."),
    )

    with pytest.raises(ValueError, match="projected lifecycle required"):
        await db.insert_memory_with_source_and_relation(
            memory,
            doc_id="confluence-123",
            source_type="confluence",
            excerpt=memory.content,
            entity_ids=None,
            relation_outcome=None,
            source_updated_at=None,
        )

    assert await db.get_memory(memory.id) is None


@pytest.mark.asyncio
async def test_direct_terminal_transition_rejects_active_source_support(db: Database) -> None:
    projection = _projection(run_id="projection-before-direct-terminal", body="A7 is removed.")
    await db.record_source_projection(projection)
    incumbent = await _seed_incumbent_support(db, projection=projection)

    with pytest.raises(ValueError, match="active source support"):
        await db.update_memory_status(incumbent.id, "retired", reason="direct bypass")
    with pytest.raises(ValueError, match="active source support"):
        await db.update_memory_status(incumbent.id, "pending_review", reason="direct bypass")
    with pytest.raises(ValueError, match="active source support"):
        await db.update_memory_content(
            incumbent.id,
            "A7 was mutated in place.",
            None,
            None,
        )

    # Non-semantic metadata tuning does not invalidate source evidence.
    await db.update_memory_content(
        incumbent.id,
        incumbent.content,
        0.8,
        list(incumbent.tags),
    )

    replacement = Memory(
        id="mem-direct-replacement",
        memory_type="decision",
        content="A7 is retained.",
        content_hash=content_hash("A7 is retained."),
    )
    with pytest.raises(ValueError, match="active source support"):
        await db.supersede_memory(
            incumbent.id,
            replacement,
            replacement_reason="direct bypass",
            replacement_kind="revision",
        )

    stored = await db.get_memory(incumbent.id)
    assert stored is not None and stored.status == "active"
    assert stored.content == incumbent.content
    assert stored.confidence == 0.8
    assert await db.get_memory(replacement.id) is None


@pytest.mark.asyncio
async def test_enabled_source_supersedes_incumbent_in_one_atomic_plan(db: Database) -> None:
    first = _projection(
        run_id="projection-1",
        body="A7 is removed.",
        item_id="confluence-old-path",
    )
    await db.record_source_projection(first)
    now = datetime(2026, 7, 14, tzinfo=timezone.utc)
    await db.upsert_document(
        DocumentRecord(
            doc_id="confluence-old-path",
            source="src-1",
            source_url="https://example.test/123",
            title="Page",
            space_or_project="ENG",
            author=None,
            last_modified=now,
            labels=[],
            version="1",
            content_hash="old-hash",
            token_count=10,
            raw_content_uri=None,
            raw_content_type=None,
            normalized_content_uri=None,
            pdf_content_uri=None,
            last_synced=now,
        )
    )
    incumbent = Memory(
        id="mem-old",
        memory_type="decision",
        content="A7 is removed.",
        content_hash=content_hash("A7 is removed."),
    )
    await db.insert_memory(incumbent)
    await db.add_memory_source(
        incumbent.id,
        "confluence-old-path",
        "confluence",
        "A7 is removed.",
        source_updated_at=None,
    )
    old_revision = first.observation_revisions[0]
    old_unit = EvidenceUnit(
        id="eu-old",
        source_id="src-1",
        doc_id="confluence-old-path",
        doc_revision_id=first.source_unit_revisions[0].id,
        source_type="confluence",
        source_anchor=first.observations[0].id,
        source_lineage_id=first.source_units[0].id,
        project_key="ENG",
        visibility="workspace",
        owner_user_id=None,
        repo_identifier=None,
        content=old_revision.content,
        excerpt="A7 is removed.",
        evidence_provenance=EvidenceContentProvenance.SOURCE_EXCERPT,
        access_context_hash="workspace-eng",
    )
    await db.upsert_evidence_unit(old_unit)
    old_reference = (
        await db.record_evidence_references(
            old_unit.id,
            (
                EvidenceReference(
                    role=EvidenceRole.PRIMARY,
                    anchor=SourceAnchor(
                        kind=AnchorKind.WHOLE_OBSERVATION,
                        observation_id=first.observations[0].id,
                        observation_revision_id=old_revision.id,
                    ),
                ),
            ),
        )
    )[0]
    await db.upsert_memory_support_assertion(
        MemorySupportAssertion(
            id="support-old",
            memory_id=incumbent.id,
            evidence_reference_id=old_reference.id or "",
            source_id="src-1",
            access_context_hash="workspace-eng",
        )
    )
    await db.enable_lifecycle_gate("src-1")

    entity_id = await db.upsert_entity("A7", "A7", ["payroll-result-slot"])
    await db.upsert_source(
        id="src-2",
        type="jira",
        name="Payroll Jira",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="owner-1",
    )
    now = datetime(2026, 7, 15, tzinfo=timezone.utc)
    await db.upsert_document(
        DocumentRecord(
            doc_id="jira-456",
            source="src-2",
            source_url="https://example.test/browse/PAY-456",
            title="A7 behavior",
            space_or_project="PAY",
            author=None,
            last_modified=now,
            labels=[],
            version="1",
            content_hash="jira-hash",
            token_count=20,
            raw_content_uri=None,
            raw_content_type=None,
            normalized_content_uri=None,
            pdf_content_uri=None,
            last_synced=now,
        )
    )
    cross_source_memory = Memory(
        id="mem-cross-source",
        memory_type="decision",
        content="A7 is removed.",
        content_hash=content_hash("A7 is removed."),
        project_key="ENG",
    )
    await db.insert_memory(cross_source_memory)
    await db.add_memory_source(
        cross_source_memory.id,
        "jira-456",
        "jira",
        "A7 is removed.",
        source_updated_at=now,
    )
    await db.link_memory_entity(cross_source_memory.id, entity_id)
    cross_source_projection = _jira_projection(
        run_id="projection-cross-source",
        description="A7 is removed.",
        source_id="src-2",
        item_id="jira-456",
        issue_key="PAY-456",
        issue_id="456",
    )
    await db.record_source_projection(cross_source_projection)
    cross_source_observation = cross_source_projection.observations[0]
    cross_source_revision = next(
        item
        for item in cross_source_projection.observation_revisions
        if item.observation_id == cross_source_observation.id
    )
    cross_source_unit = EvidenceUnit(
        id="eu-cross-source",
        source_id="src-2",
        doc_id="jira-456",
        doc_revision_id=cross_source_projection.source_unit_revisions[0].id,
        source_type="jira",
        source_anchor=cross_source_observation.id,
        source_lineage_id=cross_source_projection.source_units[0].id,
        project_key="ENG",
        visibility="workspace",
        owner_user_id=None,
        repo_identifier=None,
        content=cross_source_revision.content,
        excerpt="A7 is removed.",
        evidence_provenance=EvidenceContentProvenance.SOURCE_EXCERPT,
        access_context_hash="workspace-eng",
    )
    await db.upsert_evidence_unit(cross_source_unit)
    cross_source_reference = (
        await db.record_evidence_references(
            cross_source_unit.id,
            (
                EvidenceReference(
                    role=EvidenceRole.PRIMARY,
                    anchor=SourceAnchor(
                        kind=AnchorKind.WHOLE_OBSERVATION,
                        observation_id=cross_source_observation.id,
                        observation_revision_id=cross_source_revision.id,
                    ),
                ),
            ),
        )
    )[0]
    await db.upsert_memory_support_assertion(
        MemorySupportAssertion(
            id="support-cross-source",
            memory_id=cross_source_memory.id,
            evidence_reference_id=cross_source_reference.id or "",
            source_id="src-2",
            access_context_hash="workspace-eng",
        )
    )

    second = _projection(
        run_id="projection-2",
        body="A7 is retained and marked as reduced retro chain.",
        prior=first.source_unit_revisions[0],
        prior_observations={first.observations[0].id: old_revision},
    )
    raw = RawMemory(
        content="A7 is retained and marked as reduced retro chain.",
        memory_type="decision",
        confidence=0.95,
        tags=["payroll", "retro"],
        extraction_context="A7 is retained and marked as reduced retro chain.",
    )
    evidence = build_projected_claim_evidence(
        projection=second,
        raw_memories=(raw,),
        doc_id="confluence-123",
        source_type="confluence",
        project_key="ENG",
        visibility="workspace",
        owner_user_id=None,
        repo_identifier=None,
        access_context_hash="workspace-eng",
        extractor_run_id="sync-2",
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        relational=adapters.relational,
        vector=adapters.vector,
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=_ReplacementClient(incumbent.id),
    )
    stats = await engine.apply_projected_lifecycle(
        projection=second,
        doc_id="confluence-123",
        raw_memories=[raw],
        doc_type="design-doc",
        project_key="ENG",
        repo_identifier=None,
        entity_ids=[entity_id],
        document_content=raw.content,
        update_mode="full_document",
        changed_hunks=None,
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 15, 10, 36, tzinfo=timezone.utc),
    )

    old = await db.get_memory(incumbent.id)
    assert old is not None and old.status == "superseded"
    assert old.superseded_by is not None
    replacement = await db.get_memory(old.superseded_by)
    assert replacement is not None and replacement.status == "active"
    assert stats["superseded"] == 1
    assert stats["contradictions_found"] == 1
    cross_source_review = await db.get_pending_review_for_challenger(replacement.id)
    assert cross_source_review is not None
    assert cross_source_review.kind == "cross_source_conflict"
    assert (await db.get_memory(cross_source_memory.id)).status == "active"
    assert (await db.get_memory(replacement.id)).status == "active"
    persisted_evidence = await db.get_evidence_unit(evidence.units[0].id)
    assert persisted_evidence is not None
    assert persisted_evidence.source_lineage_id == evidence.units[0].source_lineage_id
    assert persisted_evidence.content == evidence.units[0].content
    assert persisted_evidence.extractor_run_id == second.run_id
    assert persisted_evidence.observed_at == "2026-07-15T10:36:00+00:00"
    assert await db.list_lifecycle_vector_tasks() == []


@pytest.mark.asyncio
async def test_projected_quality_consumes_typed_observation_semantics(
    db: Database,
) -> None:
    projection = _projection(
        run_id="projection-operational-transition",
        body="The ticket priority changed from high to low.",
    )
    typed_revision = replace(
        projection.observation_revisions[0],
        metadata={
            **projection.observation_revisions[0].metadata,
            "semantic_class": "operational_transition",
        },
    )
    projection = replace(
        projection,
        observation_revisions=(typed_revision,),
    )
    raw = RawMemory(
        content="The ticket priority changed from high to low.",
        memory_type="fact",
        extraction_context="opaque provider payload",
        source_observation_id=projection.observations[0].id,
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        relational=adapters.relational,
        vector=adapters.vector,
        db=db,
        memory_store=_OutboxDrainer(db),
    )

    stats = await engine.apply_projected_lifecycle(
        projection=projection,
        doc_id="confluence-123",
        raw_memories=[raw],
        doc_type="document",
        project_key="ENG",
        repo_identifier=None,
        entity_ids=[],
        document_content=typed_revision.content,
        update_mode="full_document",
        changed_hunks=None,
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 15, tzinfo=timezone.utc),
    )

    assert stats["added"] == 0
    assert stats["skipped"] == 1
    assert await db.count_memories() == 0


@pytest.mark.parametrize(
    ("run_id", "content", "context", "expected_added", "expected_skipped"),
    [
        (
            "projection-quality-metadata",
            "The ACD document was authored by Alice and last modified on 2026-06-01.",
            "Author: Alice; Last modified: 2026-06-01",
            0,
            1,
        ),
        (
            "projection-quality-open-question",
            "The team should discuss whether the payroll cutoff moves to Thursday.",
            "Open question for the next design discussion.",
            0,
            1,
        ),
        (
            "projection-quality-conditional-rule",
            "The AP result is recalculated only when the retro trigger remains OPEN.",
            "The rule is conditional on the trigger remaining OPEN.",
            1,
            0,
        ),
    ],
)
@pytest.mark.asyncio
async def test_projected_lifecycle_enforces_candidate_quality_before_persistence(
    db: Database,
    run_id: str,
    content: str,
    context: str,
    expected_added: int,
    expected_skipped: int,
) -> None:
    projection = _projection(run_id=run_id, body=content)
    revision = projection.observation_revisions[0]
    raw = RawMemory(
        content=content,
        memory_type="fact",
        extraction_context=context,
        evidence_quote=revision.content,
        source_observation_id=projection.observations[0].id,
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        relational=adapters.relational,
        vector=adapters.vector,
        db=db,
        memory_store=_OutboxDrainer(db),
    )

    stats = await engine.apply_projected_lifecycle(
        projection=projection,
        doc_id="confluence-123",
        raw_memories=[raw],
        doc_type="document",
        project_key="ENG",
        repo_identifier=None,
        entity_ids=[],
        document_content=revision.content,
        update_mode="full_document",
        changed_hunks=None,
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 15, tzinfo=timezone.utc),
    )

    assert stats["added"] == expected_added
    assert stats["skipped"] == expected_skipped
    assert await db.count_memories() == expected_added


@pytest.mark.asyncio
async def test_exact_replay_schema_is_removed(db: Database) -> None:
    async with db.db.execute(
        "SELECT name FROM sqlite_master "
        "WHERE name IN ("
        "'lifecycle_replay_ledgers', 'lifecycle_replay_claims', "
        "'idx_lifecycle_plans_exact_replay', 'idx_lifecycle_replay_claims_memory'"
        ")"
    ) as cursor:
        assert await cursor.fetchall() == []


@pytest.mark.asyncio
async def test_enabled_source_tombstone_retires_last_supported_incumbent(db: Database) -> None:
    initial = _projection(run_id="projection-before-delete", body="A7 is removed.")
    await db.record_source_projection(initial)
    incumbent = await _seed_incumbent_support(db, projection=initial)
    await db.enable_lifecycle_gate("src-1")
    tombstone = project_source_unit_tombstone(
        source_type="confluence",
        run_id="projection-delete",
        source_unit=initial.source_units[0],
        prior_unit_revision=initial.source_unit_revisions[0],
        prior_observation_revisions={
            revision.observation_id: revision for revision in initial.observation_revisions
        },
        reason="not_returned_by_authoritative_snapshot",
    )
    engine = MemoryEngine(
        relational=build_sqlite_adapters(db, object()).relational,
        vector=build_sqlite_adapters(db, object()).vector,
        db=db,
        memory_store=_OutboxDrainer(db),
    )

    result = await engine.apply_projected_tombstone(
        projection=tombstone,
        doc_id="confluence-123",
        reason="not_returned_by_authoritative_snapshot",
        lifecycle_cycle_id="enabled-source-removal",
    )

    retired = await db.get_memory(incumbent.id)
    assert retired is not None and retired.status == "retired"
    assert result == {"retired": 1, "pending_review": 0, "can_delete_document": True}
    assert await db.list_lifecycle_vector_tasks() == []

    await db.delete_projected_document("confluence-123")

    assert await db.get_document("confluence-123") is None
    assert await db.get_memory_sources(incumbent.id) == []
    assert await db.get_evidence_unit(f"eu-{incumbent.id}") is not None
    assert await db.get_source_projection(initial.run_id) == initial


@pytest.mark.asyncio
async def test_gated_source_tombstone_only_opens_review(db: Database) -> None:
    initial = _projection(run_id="projection-before-gated-delete", body="A7 is removed.")
    await db.record_source_projection(initial)
    incumbent = await _seed_incumbent_support(db, projection=initial)
    tombstone = project_source_unit_tombstone(
        source_type="confluence",
        run_id="projection-gated-delete",
        source_unit=initial.source_units[0],
        prior_unit_revision=initial.source_unit_revisions[0],
        prior_observation_revisions={
            revision.observation_id: revision for revision in initial.observation_revisions
        },
        reason="not_returned_by_authoritative_snapshot",
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        relational=adapters.relational,
        vector=adapters.vector,
        db=db,
        memory_store=_OutboxDrainer(db),
    )

    result = await engine.apply_projected_tombstone(
        projection=tombstone,
        doc_id="confluence-123",
        reason="not_returned_by_authoritative_snapshot",
        lifecycle_cycle_id="gated-source-removal",
    )

    active = await db.get_memory(incumbent.id)
    assert active is not None and active.status == "active"
    assert result == {"retired": 0, "pending_review": 1, "can_delete_document": False}
    with pytest.raises(ValueError, match="active document support remains"):
        await db.delete_projected_document("confluence-123")
