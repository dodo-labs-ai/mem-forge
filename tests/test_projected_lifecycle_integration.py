from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
import pytest_asyncio

from memforge.llm.structured import (
    CandidateLedgerDecision,
    CandidateLedgerResponse,
    IncumbentSupportAuditDecision,
    IncumbentSupportAuditResponse,
    MemoryRelationDecision,
    MemoryRelationResponse,
    MemorySupportValidationRequiredEvidence,
    MemorySupportValidationResponse,
    RevisionCompositionDecision,
    RevisionCompositionResponse,
    StructuredLlmError,
)
from memforge.evals.agent_evaluation import (
    AgentAssessmentQuery,
    AgentRuntimeEventQuery,
    QualitySignal,
    bind_quality_signals,
    bind_source_lifecycle_outcome,
    record_quality_signal,
)
from memforge.memory.audit import MemoryAuditLogger
from memforge.memory.engine import (
    DeferredProjectedLifecycleHandle,
    MemoryEngine,
    SourceUnitLifecycleDeferred,
    SourceUnitLifecycleExecutionError,
)
from memforge.memory.evidence import (
    AuthorityCase,
    CandidateMemory,
    EvidenceContentProvenance,
    EvidenceRelationRecord,
    EvidenceReference,
    EvidenceRole,
    EvidenceUnit,
    LifecycleAction,
    MemorySupportAssertion,
    RelationDirection,
    RelationOutcomeBundle,
    RelationRunRecord,
    RelationType,
    SupportScopeVersion,
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
    ProjectedLifecycleDeferredError,
    ProjectedSupportInvariantError,
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
from memforge.memory.relation_candidate_retrieval import CrossDocumentCandidateRetriever
from memforge.memory.relation_candidate_retrieval import (
    CrossDocumentCandidateSelection,
    RetrievedRelationCandidate,
)
from memforge.memory.relation_classifier import (
    MemoryPairClassification,
    MemoryPairClassificationPlan,
    MemoryPairDecision,
    MemoryRelationType,
)
from memforge.memory.relation_discovery import RelationDiscovery
from memforge.models import (
    ContentItem,
    DocumentRecord,
    Memory,
    MemoryExtractionResult,
    MemorySource,
    NormalizedContent,
    RawContent,
    RawMemory,
    ReconcileAction,
    ReconcileOperation,
    SourceLifecycleResetResult,
    content_hash,
)
from memforge.pipeline.projection_evidence import build_projected_claim_evidence
from memforge.pipeline.projection_context import (
    ProjectionExtractionBatch,
)
from memforge.pipeline.source_projection_adapters import (
    project_source_item,
    project_source_unit_tombstone,
)
from memforge.source_projection import (
    AnchorKind,
    EvidenceCoordinateSpace,
    EvidenceRepresentationProfile,
    ProjectionCoverage,
    SourceAnchor,
    SourceProjection,
)
from memforge.source_artifacts import StoredSourceArtifact
from memforge.source_derivation import (
    SourceUnitDerivationRequest,
    SourceUnitDerivationContext,
    SourceUnitDeriver,
    plan_source_derivation_work,
    safe_derivation_error,
    source_derivation_manifest,
)
from memforge.storage.adapters.sqlite import build_sqlite_adapters
from memforge.storage.database import Database


def _candidate_retriever(adapters):
    return CrossDocumentCandidateRetriever(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
    )


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


async def _set_fixture_source_type(db: Database, source_type: str) -> None:
    await db.upsert_source(
        id="src-1",
        type=source_type,
        name="Engineering",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="owner-1",
    )


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


def _teams_projection(*, run_id: str, message_content: str) -> SourceProjection:
    item = ContentItem(
        item_id="teams-window-1",
        title="Group: PCC Agent Dev -- Jul 30, 10:00-10:00",
        source_url="https://teams.example.test/message/1",
        last_modified=datetime(2026, 7, 30, 10, tzinfo=timezone.utc),
        version="1",
        extra={
            "conversation_id": "19:conversation@thread.v2",
            "window_id": "teams-window-1",
        },
    )
    raw_payload = {
        "conversation_id": "19:conversation@thread.v2",
        "window_id": "teams-window-1",
        "messages": [
            {
                "id": "message-1",
                "content": message_content,
                "time": "2026-07-30T10:00:00+00:00",
            }
        ],
    }
    rendered_document = (
        f"# {item.title}\n\n"
        "**Group Chat**: PCC Agent Dev\n"
        "**Messages**: 1\n\n---\n\n"
        "**Alex** (2026-07-30T10:00):\n"
        f"{message_content}\n"
    )
    return project_source_item(
        source_id="src-teams",
        source_type="teams",
        run_id=run_id,
        item=item,
        raw=RawContent(
            item=item,
            body=json.dumps(raw_payload).encode(),
            content_type="application/json",
        ),
        normalized=NormalizedContent(item=item, markdown_body=rendered_document),
    )
def _projection_with_artifact(
    *,
    run_id: str,
    payload: bytes,
    provider_revision: str,
    inference_eligible: bool,
    body: str = "# Page",
    prior=None,
    prior_observations=None,
) -> SourceProjection:
    item = ContentItem(
        item_id="confluence-123",
        title="Page",
        source_url="https://example.test/123",
        last_modified=datetime(2026, 7, 15, tzinfo=timezone.utc),
        version=provider_revision,
        extra={"page_id": "123", "space_key": "ENG"},
    )
    return project_source_item(
        source_id="src-1",
        source_type="confluence",
        run_id=run_id,
        item=item,
        raw=RawContent(
            item=item,
            body=body.encode(),
            content_type="text/html",
        ),
        normalized=NormalizedContent(item=item, markdown_body=body),
        artifacts=(
            StoredSourceArtifact(
                id="artifact-diagram",
                provider_key="attachment-1",
                parent_observation_type="page_body",
                parent_provider_key="123:body",
                provider_revision=provider_revision,
                filename="diagram.png",
                media_type="image/png",
                size_bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
                uri=f"artifact://diagram-{provider_revision}.png",
                inference_eligible=inference_eligible,
                inference_ineligible_reason=(None if inference_eligible else "invalid_image_structure"),
            ),
        ),
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


def _audit_response(
    *decisions: IncumbentSupportAuditDecision,
) -> IncumbentSupportAuditResponse:
    return IncumbentSupportAuditResponse(decisions=list(decisions))


def _uniform_relation_response(
    prompt: str,
    *,
    classification: str,
    reason: str,
) -> MemoryRelationResponse:
    groups_json = prompt.split("<memory_pair_groups>\n", 1)[1].split(
        "\n</memory_pair_groups>",
        1,
    )[0]
    groups = json.loads(groups_json)
    pair_indices = [
        item["pair_index"]
        for group in groups
        for item in group["candidates"]
    ]
    return MemoryRelationResponse(
        decisions=[
            MemoryRelationDecision(
                pair_index=pair_index,
                classification=classification,
                direction="symmetric",
                same_subject_and_scope=classification == "contradicts",
                incompatible_assertions=("current assertions are incompatible" if classification == "contradicts" else ""),
                reason=reason,
            )
            for pair_index in pair_indices
        ]
    )


class _ReplacementClient:
    def __init__(self, incumbent_id: str) -> None:
        self.incumbent_id = incumbent_id

    async def classify_memory_relations(self, prompt: str, **kwargs):
        del kwargs
        return _uniform_relation_response(
            prompt,
            classification="contradicts",
            reason="The source now retains A7.",
        )

    async def audit_incumbent_support(self, prompt: str, **kwargs):
        del prompt, kwargs
        return _audit_response(
            IncumbentSupportAuditDecision(
                supported=False,
                reason="The old claim is replaced.",
            )
        )


class _ConflictingReplacementClient:
    async def classify_memory_relations(self, prompt: str, **kwargs):
        del kwargs
        return _uniform_relation_response(
            prompt,
            classification="contradicts",
            reason="The candidate appears to replace the incumbent.",
        )

    async def audit_incumbent_support(self, prompt: str, **kwargs):
        del prompt, kwargs
        return _audit_response(
            IncumbentSupportAuditDecision(
                supported=True,
                reason="The incumbent still appears supported.",
            )
        )


class _AdditiveRevisionClient:
    async def classify_memory_relations(self, prompt: str, **kwargs):
        del kwargs
        groups_json = prompt.split("<memory_pair_groups>\n", 1)[1].split(
            "\n</memory_pair_groups>",
            1,
        )[0]
        pair_index = json.loads(groups_json)[0]["candidates"][0]["pair_index"]
        return MemoryRelationResponse(
            decisions=[
                MemoryRelationDecision(
                    pair_index=pair_index,
                    classification="refines",
                    direction="challenger_to_candidate",
                    same_subject_and_scope=True,
                    incompatible_assertions="",
                    reason="The configuration key adds detail to the same timeout claim.",
                )
            ]
        )

    async def audit_incumbent_support(self, prompt: str, **kwargs):
        del prompt, kwargs
        return _audit_response(
            IncumbentSupportAuditDecision(
                supported=True,
                reason="The 30 second timeout remains current.",
            )
        )

    async def prove_revision_compositions(self, prompt: str, **kwargs):
        del prompt, kwargs
        return RevisionCompositionResponse(
            decisions=[
                RevisionCompositionDecision(
                    pair_index=0,
                    same_memory_identity=True,
                    preserves_incumbent_truth=True,
                    candidate_is_canonical_composite=True,
                    current_evidence_entails_candidate=True,
                    reason="The candidate is the complete current timeout claim.",
                )
            ]
        )


class _RunbookComponentFallbackClient:
    async def classify_memory_relations(self, prompt: str, **kwargs):
        del kwargs
        groups_json = prompt.split("<memory_pair_groups>\n", 1)[1].split(
            "\n</memory_pair_groups>",
            1,
        )[0]
        pair_indices = [item["pair_index"] for group in json.loads(groups_json) for item in group["candidates"]]
        return MemoryRelationResponse(
            decisions=[
                MemoryRelationDecision(
                    pair_index=pair_index,
                    classification="refines",
                    direction="challenger_to_candidate",
                    same_subject_and_scope=True,
                    incompatible_assertions="",
                    reason="The procedure is related but not a lossless revision of this branch.",
                )
                for pair_index in pair_indices
            ]
        )

    async def audit_incumbent_support(self, prompt: str, **kwargs):
        del kwargs
        incumbents_json = prompt.split("<incumbents>", 1)[1].split("</incumbents>", 1)[0]
        return _audit_response(
            *(
                IncumbentSupportAuditDecision(
                    supported=True,
                    reason="The branch remains supported in the current runbook.",
                )
                for _ in json.loads(incumbents_json)
            )
        )

    async def prove_revision_compositions(self, prompt: str, **kwargs):
        del prompt, kwargs
        raise StructuredLlmError("The current procedure does not losslessly replace each branch.")


class _RunbookComponentRevisionClient(_RunbookComponentFallbackClient):
    async def prove_revision_compositions(self, prompt: str, **kwargs):
        del kwargs
        pairs_json = prompt.split("<refinement_pairs>", 1)[1].split(
            "</refinement_pairs>",
            1,
        )[0]
        return RevisionCompositionResponse(
            decisions=[
                RevisionCompositionDecision(
                    pair_index=item["pair_index"],
                    same_memory_identity=True,
                    preserves_incumbent_truth=True,
                    candidate_is_canonical_composite=True,
                    current_evidence_entails_candidate=True,
                    reason="The canonical procedure preserves this branch verbatim.",
                )
                for item in json.loads(pairs_json)
            ]
        )


class _NoopClient:
    def __init__(self, incumbent_id: str) -> None:
        self.incumbent_id = incumbent_id

    async def audit_incumbent_support(self, prompt: str, **kwargs):
        del prompt, kwargs
        return _audit_response(
            IncumbentSupportAuditDecision(
                supported=True,
                reason="The exact claim remains in the revised page.",
            )
        )


class _DeleteClient:
    def __init__(self, incumbent_id: str) -> None:
        self.incumbent_id = incumbent_id

    async def audit_incumbent_support(self, prompt: str, **kwargs):
        del prompt, kwargs
        return _audit_response(
            IncumbentSupportAuditDecision(
                supported=False,
                reason="The incomplete rendering appears to omit the claim.",
            )
        )


class _UnexpectedReconciliationClient:
    async def classify_memory_relations(self, prompt: str, **kwargs):
        del prompt, kwargs
        raise AssertionError("proven-disjoint incumbent must not require LLM reconciliation")


class _CapturingRuntimeSink:
    def __init__(self) -> None:
        self.published = []

    def publish(self, events) -> None:
        self.published.append(events)


@pytest.mark.asyncio
async def test_lifecycle_commit_rejection_returns_failure_bundle_without_success_row(
    db: Database,
) -> None:
    projection = _projection(
        run_id="projection-stale-lifecycle-event",
        body="Service uses PostgreSQL 15.",
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_OutboxDrainer(db),
    )

    with pytest.raises(SourceUnitLifecycleExecutionError) as failure:
        await engine.prepare_and_commit_projected_lifecycle(
            projection=projection,
            doc_id="confluence-123",
            raw_memories=[
                RawMemory(
                    content="Service uses PostgreSQL 15.",
                    memory_type="fact",
                )
            ],
            doc_type="design-doc",
            project_key="ENG",
            repo_identifier=None,
            document_content="Service uses PostgreSQL 15.",
            update_mode="full_document",
            changed_hunks=None,
            update_plan_stats=None,
            source_updated_at=datetime(2026, 7, 15, tzinfo=timezone.utc),
            expected_source_activity_epoch=999,
            lifecycle_execution_owner_id="sync-run-stale:lease-1",
        )

    assert failure.value.runtime_bundle.event.reason_code == "lifecycle_commit_failed"
    assert await db.list_memories() == []
    assert await db.list_agent_runtime_events(
        AgentRuntimeEventQuery(
            occurred_from=datetime(2026, 8, 16, tzinfo=timezone.utc),
            occurred_to=datetime(2026, 8, 18, tzinfo=timezone.utc),
            source_id="src-1",
            event_name="source_unit_lifecycle_outcome",
            limit=10,
        )
    ) == []

    async def audit_incumbent_support(self, prompt: str, **kwargs):
        del prompt, kwargs
        raise AssertionError("proven-disjoint incumbent must not require LLM reconciliation")


@pytest.mark.asyncio
async def test_conflicting_reconciliation_judgments_commit_pending_review(
    db: Database,
) -> None:
    first = _projection(
        run_id="projection-conflict-review-1",
        body="Service uses PostgreSQL 15.",
    )
    adapters = build_sqlite_adapters(db, object())
    runtime_sink = _CapturingRuntimeSink()
    engine = MemoryEngine(
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_OutboxDrainer(db),
        runtime_event_trace_sink=runtime_sink,
    )
    await engine.prepare_and_commit_projected_lifecycle(
        projection=first,
        doc_id="confluence-123",
        raw_memories=[
            RawMemory(
                content="Service uses PostgreSQL 15.",
                memory_type="fact",
            )
        ],
        doc_type="design-doc",
        project_key="ENG",
        repo_identifier=None,
        document_content="Service uses PostgreSQL 15.",
        update_mode="full_document",
        changed_hunks=None,
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 15, tzinfo=timezone.utc),
        lifecycle_execution_owner_id="sync-run-review:lease-1",
        lifecycle_attempt_count=2,
    )
    [incumbent] = await db.list_memories()
    previous_support = await db.get_active_memory_support_reference_ids(incumbent.id)

    second = _projection(
        run_id="projection-conflict-review-2",
        body="Service uses PostgreSQL 16.",
        prior=first.source_unit_revisions[0],
        prior_observations={revision.observation_id: revision for revision in first.observation_revisions},
    )
    reviewing_engine = MemoryEngine(
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=_ConflictingReplacementClient(),
    )

    event_window_start = datetime.now(timezone.utc) - timedelta(seconds=1)
    stats = await reviewing_engine.prepare_and_commit_projected_lifecycle(
        projection=second,
        doc_id="confluence-123",
        raw_memories=[
            RawMemory(
                content="Service uses PostgreSQL 16.",
                memory_type="fact",
            )
        ],
        doc_type="design-doc",
        project_key="ENG",
        repo_identifier=None,
        document_content="Service uses PostgreSQL 16.",
        update_mode="full_document",
        changed_hunks=None,
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 16, tzinfo=timezone.utc),
    )
    event_window_end = datetime.now(timezone.utc) + timedelta(seconds=1)

    assert stats["pending_review"] == 1
    current = await db.get_memory(incumbent.id)
    assert current is not None and current.status == "active"
    assert await db.get_active_memory_support_reference_ids(incumbent.id) == previous_support
    assert [memory.id for memory in await db.list_memories()] == [incumbent.id]
    [review] = await db.list_lifecycle_reviews(
        "src-1",
        status=LifecycleReviewStatus.PENDING,
    )
    assert review.incumbent_memory_id == incumbent.id
    assert (
        second.source_unit_revisions[0].id == (await db.get_current_source_unit_revision(second.source_units[0].id)).id
    )
    runtime_events = await db.list_agent_runtime_events(
        AgentRuntimeEventQuery(
            occurred_from=event_window_start,
            occurred_to=event_window_end,
            source_id="src-1",
            event_name="source_unit_lifecycle_outcome",
            limit=10,
        )
    )
    assert len(runtime_events) == 1
    assert runtime_events[0].outcome == "expected"
    assert runtime_events[0].recovered is True
    assert runtime_sink.published == [(runtime_events[0],)]


@pytest.mark.asyncio
async def test_additive_refinement_commits_revision_with_candidate_local_evidence(
    db: Database,
) -> None:
    first_text = "The client timeout is 30 seconds."
    first = _projection(run_id="projection-revision-1", body=first_text)
    adapters = build_sqlite_adapters(db, object())
    initial_engine = MemoryEngine(
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_OutboxDrainer(db),
    )
    await initial_engine.prepare_and_commit_projected_lifecycle(
        projection=first,
        doc_id="confluence-123",
        raw_memories=[
            RawMemory(
                content=first_text,
                memory_type="fact",
                evidence_quote=first_text,
                evidence_anchor="projection_batch",
                source_observation_id=first.observations[0].id,
            )
        ],
        doc_type="design-doc",
        project_key="ENG",
        repo_identifier=None,
        document_content=first_text,
        update_mode="full_document",
        changed_hunks=None,
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 15, tzinfo=timezone.utc),
    )
    [incumbent] = await db.list_memories()
    await db.enable_lifecycle_gate("src-1")

    second_text = "The client timeout is 30 seconds and is configurable with CLIENT_TIMEOUT."
    second = _projection(
        run_id="projection-revision-2",
        body=second_text,
        prior=first.source_unit_revisions[0],
        prior_observations={revision.observation_id: revision for revision in first.observation_revisions},
    )
    revision_engine = MemoryEngine(
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=_AdditiveRevisionClient(),
    )
    stats = await revision_engine.prepare_and_commit_projected_lifecycle(
        projection=second,
        doc_id="confluence-123",
        raw_memories=[
            RawMemory(
                content=second_text,
                memory_type="fact",
                evidence_quote=second_text,
                evidence_anchor="projection_batch",
                evidence_resolved_from_block=True,
                source_observation_id=second.observations[0].id,
            )
        ],
        doc_type="design-doc",
        project_key="ENG",
        repo_identifier=None,
        document_content=second_text,
        update_mode="full_document",
        changed_hunks=None,
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 16, tzinfo=timezone.utc),
    )

    old = await db.get_memory(incumbent.id)
    assert old is not None and old.status == "superseded"
    assert old.replacement_kind == "revision"
    replacement = await db.get_memory(old.superseded_by or "")
    assert replacement is not None and replacement.content == second_text
    support = await db.get_active_memory_support_evidence(replacement.id, source_id="src-1")
    assert len([item for item in support if item.role is EvidenceRole.PRIMARY]) == 1
    assert support[0].excerpt == second_text
    assert stats["updated"] == 1
    assert stats["superseded"] == 0


@pytest.mark.asyncio
async def test_runbook_component_fallback_commits_candidate_once_and_keeps_branches(
    db: Database,
) -> None:
    branch_claims = [
        "For HTTP 404, check service health, retrigger, then open a DwC issue if it persists.",
        "For HTTP 502 or 503, wait for service recovery and then retrigger the process.",
        "For other invalid process map errors, create a design-time Jira defect.",
    ]
    first_text = "\n\n".join(branch_claims)
    first = _projection(run_id="projection-runbook-component-1", body=first_text)
    adapters = build_sqlite_adapters(db, object())
    initial_engine = MemoryEngine(
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_AuditedOutboxDrainer(db),
        structured_llm_client=_CandidateLedgerClient(
            _candidate_ledger_response(*(CandidateLedgerDecision(action="KEEP") for _ in branch_claims))
        ),
    )
    await initial_engine.prepare_and_commit_projected_lifecycle(
        projection=first,
        doc_id="confluence-123",
        raw_memories=[
            RawMemory(
                content=claim,
                memory_type="procedure",
                evidence_quote=claim,
                evidence_anchor="projection_batch",
                source_observation_id=first.observations[0].id,
            )
            for claim in branch_claims
        ],
        doc_type="runbook",
        project_key="ENG",
        repo_identifier=None,
        document_content=first_text,
        update_mode="full_document",
        changed_hunks=None,
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 15, tzinfo=timezone.utc),
    )
    incumbents = await db.list_memories()
    await db.enable_lifecycle_gate("src-1")

    current_procedure = (
        "Diagnose an invalid process map from its actual HTTP error, then follow the status-specific recovery path."
    )
    second_text = f"{first_text}\n\n{current_procedure}"
    second = _projection(
        run_id="projection-runbook-component-2",
        body=second_text,
        prior=first.source_unit_revisions[0],
        prior_observations={revision.observation_id: revision for revision in first.observation_revisions},
    )
    engine = MemoryEngine(
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=_RunbookComponentFallbackClient(),
    )

    stats = await engine.prepare_and_commit_projected_lifecycle(
        projection=second,
        doc_id="confluence-123",
        raw_memories=[
            RawMemory(
                content=current_procedure,
                memory_type="procedure",
                evidence_quote=current_procedure,
                evidence_anchor="projection_batch",
                evidence_resolved_from_block=True,
                source_observation_id=second.observations[0].id,
            )
        ],
        doc_type="runbook",
        project_key="ENG",
        repo_identifier=None,
        document_content=second_text,
        update_mode="full_document",
        changed_hunks=None,
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 16, tzinfo=timezone.utc),
    )

    memories = await db.list_memories()
    current_by_id = {memory.id: memory for memory in memories}
    assert stats["added"] == 1
    assert stats["noop"] == 3
    assert stats["pending_review"] == 0
    assert sorted(memory.content for memory in memories if memory.status == "active") == sorted(
        [*branch_claims, current_procedure]
    )
    assert all(current_by_id[memory.id].status == "active" for memory in incumbents)
    assert (
        second.source_unit_revisions[0].id == (await db.get_current_source_unit_revision(second.source_units[0].id)).id
    )


@pytest.mark.asyncio
async def test_runbook_component_revision_creates_one_replacement_for_all_branches(
    db: Database,
) -> None:
    branch_claims = [
        "For HTTP 404, check service health, retrigger, then open a DwC issue if it persists.",
        "For HTTP 502 or 503, wait for service recovery and then retrigger the process.",
        "For other invalid process map errors, create a design-time Jira defect.",
    ]
    first_text = "\n\n".join(branch_claims)
    first = _projection(run_id="projection-runbook-revision-1", body=first_text)
    adapters = build_sqlite_adapters(db, object())
    initial_engine = MemoryEngine(
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_AuditedOutboxDrainer(db),
        structured_llm_client=_CandidateLedgerClient(
            _candidate_ledger_response(*(CandidateLedgerDecision(action="KEEP") for _ in branch_claims))
        ),
    )
    await initial_engine.prepare_and_commit_projected_lifecycle(
        projection=first,
        doc_id="confluence-123",
        raw_memories=[
            RawMemory(
                content=claim,
                memory_type="procedure",
                evidence_quote=claim,
                evidence_anchor="projection_batch",
                source_observation_id=first.observations[0].id,
            )
            for claim in branch_claims
        ],
        doc_type="runbook",
        project_key="ENG",
        repo_identifier=None,
        document_content=first_text,
        update_mode="full_document",
        changed_hunks=None,
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 15, tzinfo=timezone.utc),
    )
    incumbents = await db.list_memories()
    await db.enable_lifecycle_gate("src-1")

    canonical_procedure = (
        "Diagnose an invalid process map from its actual HTTP error, then follow exactly one "
        "status-specific recovery path:\n- " + "\n- ".join(branch_claims)
    )
    second = _projection(
        run_id="projection-runbook-revision-2",
        body=canonical_procedure,
        prior=first.source_unit_revisions[0],
        prior_observations={revision.observation_id: revision for revision in first.observation_revisions},
    )
    engine = MemoryEngine(
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=_RunbookComponentRevisionClient(),
    )

    stats = await engine.prepare_and_commit_projected_lifecycle(
        projection=second,
        doc_id="confluence-123",
        raw_memories=[
            RawMemory(
                content=canonical_procedure,
                memory_type="procedure",
                evidence_quote=canonical_procedure,
                evidence_anchor="projection_batch",
                evidence_resolved_from_block=True,
                source_observation_id=second.observations[0].id,
            )
        ],
        doc_type="runbook",
        project_key="ENG",
        repo_identifier=None,
        document_content=canonical_procedure,
        update_mode="full_document",
        changed_hunks=None,
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 16, tzinfo=timezone.utc),
    )

    memories = await db.list_memories()
    old_memories = [memory for memory in memories if memory.id in {item.id for item in incumbents}]
    replacement_ids = {memory.superseded_by for memory in old_memories}
    active = [memory for memory in memories if memory.status == "active"]
    assert stats["added"] == 1
    assert stats["updated"] == 3
    assert stats["pending_review"] == 0
    assert len(replacement_ids) == 1
    assert all(memory.status == "superseded" for memory in old_memories)
    assert len(active) == 1
    assert active[0].id in replacement_ids
    assert active[0].content == canonical_procedure


@pytest.mark.asyncio
async def test_inference_ineligible_artifact_revision_preserves_incumbent_support(
    db: Database,
) -> None:
    first = _projection_with_artifact(
        run_id="projection-artifact-valid",
        payload=b"valid-image-revision",
        provider_revision="1",
        inference_eligible=True,
    )
    artifact_observation_id = next(
        observation.id for observation in first.observations if observation.observation_type == "binary_artifact"
    )
    page_observation_id = next(
        observation.id for observation in first.observations if observation.observation_type == "page_body"
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_OutboxDrainer(db),
    )
    await engine.prepare_and_commit_projected_lifecycle(
        projection=first,
        doc_id="confluence-123",
        raw_memories=[
            RawMemory(
                content=("The architecture diagram establishes that A7 remains enabled."),
                memory_type="decision",
                evidence_quote="# Page",
                evidence_anchor="projection_batch",
                source_observation_id=page_observation_id,
                required_source_observation_ids=(artifact_observation_id,),
            )
        ],
        doc_type="design-doc",
        project_key="ENG",
        repo_identifier=None,
        document_content="# Page",
        update_mode="full_document",
        changed_hunks=None,
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 15, tzinfo=timezone.utc),
    )
    [incumbent] = await db.list_memories()
    previous_support = await db.get_active_memory_support_reference_ids(incumbent.id)
    observation_ids = await db.get_active_memory_support_observation_ids_many(
        (incumbent.id,),
        source_id="src-1",
    )
    assert artifact_observation_id in observation_ids[incumbent.id]
    assert page_observation_id in observation_ids[incumbent.id]

    second = _projection_with_artifact(
        run_id="projection-artifact-invalid",
        payload=b"invalid-image-revision",
        provider_revision="2",
        inference_eligible=False,
        prior=first.source_unit_revisions[0],
        prior_observations={revision.observation_id: revision for revision in first.observation_revisions},
    )
    protected_engine = MemoryEngine(
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=_UnexpectedReconciliationClient(),
    )
    stats = await protected_engine.prepare_and_commit_projected_lifecycle(
        projection=second,
        doc_id="confluence-123",
        raw_memories=[],
        doc_type="design-doc",
        project_key="ENG",
        repo_identifier=None,
        document_content="# Page",
        update_mode="full_document",
        changed_hunks=None,
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 15, tzinfo=timezone.utc),
        protected_source_observation_ids=(artifact_observation_id,),
    )

    current = await db.get_memory(incumbent.id)
    current_support = await db.get_active_memory_support_reference_ids(incumbent.id)
    assert current is not None and current.status == "active"
    assert stats["pending_review"] == 1
    assert current_support == previous_support
    assert (
        second.source_unit_revisions[0].id == (await db.get_current_source_unit_revision(second.source_units[0].id)).id
    )


@pytest.mark.asyncio
async def test_context_artifact_does_not_become_active_support_dependency(
    db: Database,
) -> None:
    projection = _projection_with_artifact(
        run_id="projection-context-artifact",
        payload=b"valid-context-image",
        provider_revision="1",
        inference_eligible=True,
    )
    artifact_observation_id = next(
        observation.id
        for observation in projection.observations
        if observation.observation_type == "binary_artifact"
    )
    page_observation_id = next(
        observation.id
        for observation in projection.observations
        if observation.observation_type == "page_body"
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_OutboxDrainer(db),
    )
    await engine.prepare_and_commit_projected_lifecycle(
        projection=projection,
        doc_id="confluence-123",
        raw_memories=[
            RawMemory(
                content="A7 remains enabled.",
                memory_type="decision",
                evidence_quote="# Page",
                evidence_anchor="projection_batch",
                source_observation_id=page_observation_id,
            )
        ],
        doc_type="design-doc",
        project_key="ENG",
        repo_identifier=None,
        document_content="# Page",
        update_mode="full_document",
        changed_hunks=None,
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 15, tzinfo=timezone.utc),
    )
    [incumbent] = await db.list_memories()

    observation_ids = await db.get_active_memory_support_observation_ids_many(
        (incumbent.id,),
        source_id="src-1",
    )

    assert observation_ids[incumbent.id] == (page_observation_id,)
    assert artifact_observation_id not in observation_ids[incumbent.id]


@pytest.mark.asyncio
async def test_removed_artifact_dependency_commits_projection_with_pending_review(
    db: Database,
) -> None:
    first = _projection_with_artifact(
        run_id="projection-artifact-present",
        payload=b"current-body-image",
        provider_revision="1",
        inference_eligible=True,
    )
    artifact_observation_id = next(
        observation.id for observation in first.observations if observation.observation_type == "binary_artifact"
    )
    page_observation_id = next(
        observation.id for observation in first.observations if observation.observation_type == "page_body"
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_OutboxDrainer(db),
    )
    await engine.prepare_and_commit_projected_lifecycle(
        projection=first,
        doc_id="confluence-123",
        raw_memories=[
            RawMemory(
                content="The diagram records that A7 remains enabled.",
                memory_type="decision",
                evidence_quote="# Page",
                source_observation_id=page_observation_id,
                required_source_observation_ids=(artifact_observation_id,),
            )
        ],
        doc_type="design-doc",
        project_key="ENG",
        repo_identifier=None,
        document_content="# Page",
        update_mode="full_document",
        changed_hunks=None,
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 15, tzinfo=timezone.utc),
    )
    [incumbent] = await db.list_memories()
    previous_support = await db.get_active_memory_support_reference_ids(incumbent.id)
    second = _projection(
        run_id="projection-artifact-removed",
        body="# Page",
        prior=first.source_unit_revisions[0],
        prior_observations={revision.observation_id: revision for revision in first.observation_revisions},
    )
    reviewing_engine = MemoryEngine(
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=_NoopClient(incumbent.id),
    )

    stats = await reviewing_engine.prepare_and_commit_projected_lifecycle(
        projection=second,
        doc_id="confluence-123",
        raw_memories=[],
        doc_type="design-doc",
        project_key="ENG",
        repo_identifier=None,
        document_content="# Page",
        update_mode="full_document",
        changed_hunks=None,
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 16, tzinfo=timezone.utc),
    )

    current = await db.get_memory(incumbent.id)
    current_support = await db.get_active_memory_support_reference_ids(incumbent.id)
    assert current is not None and current.status == "active"
    assert stats["pending_review"] == 1
    assert current_support == previous_support
    assert (
        second.source_unit_revisions[0].id == (await db.get_current_source_unit_revision(second.source_units[0].id)).id
    )


class _RecordingAddClient:
    def __init__(self, incumbent_id: str) -> None:
        self.incumbent_id = incumbent_id
        self.prompts: list[str] = []

    async def classify_memory_relations(self, prompt: str, **kwargs):
        del kwargs
        self.prompts.append(prompt)
        return _uniform_relation_response(
            prompt,
            classification="unrelated",
            reason="The changed observation states a separate durable claim.",
        )

    async def audit_incumbent_support(self, prompt: str, **kwargs):
        del prompt, kwargs
        return _audit_response(
            IncumbentSupportAuditDecision(
                supported=True,
                reason="The unchanged incumbent remains supported.",
            )
        )


class _PersistentlyIncompleteAuditClient:
    def __init__(self, incumbent_id: str) -> None:
        self.incumbent_id = incumbent_id
        self.calls = 0

    async def audit_incumbent_support(self, prompt: str, **kwargs):
        del prompt, kwargs
        self.calls += 1
        return _audit_response()


class _OutboxDrainer:
    def __init__(self, database: Database) -> None:
        self.db = database

    async def attempt_lifecycle_vector_delivery(self, lifecycle_plan_id: str) -> LifecycleVectorDeliveryResult:
        for task in await self.db.list_lifecycle_vector_tasks(lifecycle_plan_id=lifecycle_plan_id):
            await self.db.complete_lifecycle_vector_task(task.id)
        return LifecycleVectorDeliveryResult(state=LifecycleVectorDeliveryState.DELIVERED)

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

    async def find_access_compatible_equivalence_candidates_batch(self, queries):
        candidates = []
        for query in queries:
            candidates.append(
                await self.find_access_compatible_equivalence_candidates(
                    query.memory,
                    excluded_memory_ids=query.excluded_memory_ids,
                    doc_id=query.doc_id,
                    entity_ids=query.entity_ids,
                )
            )
        return tuple(candidates)

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

    async def find_access_compatible_exact_candidates_batch(self, requests):
        return tuple(
            [
                await self.find_access_compatible_exact_candidate(
                    request.challenger,
                    excluded_memory_ids=request.excluded_memory_ids,
                )
                for request in requests
            ]
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


def _candidate_ledger_response(
    *decisions: CandidateLedgerDecision,
) -> CandidateLedgerResponse:
    return CandidateLedgerResponse(decisions=list(decisions))


class _FailingOutboxDrainer(_OutboxDrainer):
    async def attempt_lifecycle_vector_delivery(self, lifecycle_plan_id: str) -> LifecycleVectorDeliveryResult:
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

    async def find_access_compatible_equivalence_candidates_batch(self, queries):
        candidates = []
        for query in queries:
            candidates.append(
                await self.find_access_compatible_equivalence_candidates(
                    query.memory,
                    excluded_memory_ids=query.excluded_memory_ids,
                    doc_id=query.doc_id,
                    entity_ids=query.entity_ids,
                )
            )
        return tuple(candidates)


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
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=store,
        structured_llm_client=None,
    )

    stats = await engine.prepare_and_commit_projected_lifecycle(
        projection=projection,
        doc_id="confluence-123",
        raw_memories=[canonical, duplicate],
        doc_type="ticket",
        project_key="ENG",
        repo_identifier=None,
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
    assert stats["candidate_ledger_input_count"] == 2
    assert stats["candidate_ledger_selected_count"] == 1
    assert stats["candidate_ledger_llm_calls"] == 0
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
        "dropped_low_value_count": 0,
        "structured_llm_calls": 0,
        "structured_llm_elapsed_ms": 0,
        "validation_retries": 0,
        "fallback_batch_count": 0,
        "fallback_candidate_count": 0,
        "prompt_chars": 0,
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
async def test_projected_lifecycle_records_low_value_admission_without_content(
    db: Database,
) -> None:
    durable_content = "Enable the reduction toggle only after the compatibility suite passes."
    instance_content = "Test case 17 returned 204 rows in this run."
    projection = _projection(
        run_id="projection-candidate-quality",
        body=f"{durable_content}\n\n{instance_content}",
    )
    observation_id = projection.observations[0].id
    durable = RawMemory(
        content=durable_content,
        memory_type="procedure",
        evidence_quote=durable_content,
        source_observation_id=observation_id,
    )
    instance_output = RawMemory(
        content=instance_content,
        memory_type="fact",
        evidence_quote=instance_content,
        source_observation_id=observation_id,
    )
    client = _CandidateLedgerClient(
        _candidate_ledger_response(
            CandidateLedgerDecision(action="KEEP"),
            CandidateLedgerDecision(
                action="DROP_LOW_VALUE",
                reason=f"Do not persist: {instance_content}",
            ),
        )
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_AuditedOutboxDrainer(db),
        structured_llm_client=client,
    )

    stats = await engine.prepare_and_commit_projected_lifecycle(
        projection=projection,
        doc_id="confluence-123",
        raw_memories=[durable, instance_output],
        doc_type="document",
        project_key="ENG",
        repo_identifier=None,
        document_content=projection.observation_revisions[0].content,
        update_mode="full_document",
        changed_hunks=None,
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 17, tzinfo=timezone.utc),
    )

    memories = await db.list_memories()
    [event] = await db.list_memory_audit_events(event_type="candidate_ledger_completed")

    assert [memory.content for memory in memories] == [durable_content]
    assert stats["candidate_ledger_dropped_low_value_count"] == 1
    assert event.payload["dropped_low_value_count"] == 1
    [drop] = event.payload["drops"]
    assert drop["method"] == "structured_quality"
    assert drop["reason"] == "low_value_admission"
    assert instance_content not in str(event.payload)


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
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=None,
    )

    stats = await engine.prepare_and_commit_projected_lifecycle(
        projection=projection,
        doc_id="confluence-123",
        raw_memories=[raw],
        doc_type="document",
        project_key="ENG",
        repo_identifier=None,
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
async def test_incomplete_candidate_ledger_is_audited_as_fallback_and_keeps_memories(
    db: Database,
) -> None:
    projection = _projection(
        run_id="projection-candidate-ledger-failed",
        body="The trigger remained OPEN. The trigger was not processed.",
    )
    observation_id = projection.observations[0].id
    client = _CandidateLedgerClient(_candidate_ledger_response(CandidateLedgerDecision(action="KEEP")))
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_AuditedOutboxDrainer(db),
        structured_llm_client=client,
    )

    await engine.prepare_and_commit_projected_lifecycle(
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
        document_content=projection.observation_revisions[0].content,
        update_mode="full_document",
        changed_hunks=None,
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 17, tzinfo=timezone.utc),
    )

    async with db.db.execute("SELECT COUNT(*) AS total FROM memories") as cursor:
        row = await cursor.fetchone()
    events = await db.list_memory_audit_events(event_type="candidate_ledger_completed")

    assert row["total"] == 2
    assert client.calls == 2
    assert len(events) == 1
    assert events[0].status == "committed"
    assert events[0].reason == "candidate_admission_with_fallback"
    assert events[0].payload["input_count"] == 2
    assert events[0].payload["semantic_input_count"] == 2
    assert events[0].payload["selected_count"] == 2
    assert events[0].payload["fallback_batch_count"] == 1
    assert events[0].payload["fallback_candidate_count"] == 2


class _SemanticEquivalentClient:
    def __init__(self) -> None:
        self.relation_calls = 0

    async def audit_incumbent_support(self, prompt: str, **kwargs):
        del prompt, kwargs
        return _audit_response(IncumbentSupportAuditDecision(supported=True, reason="still supported"))

    async def classify_memory_relations(self, prompt: str, **kwargs):
        del kwargs
        self.relation_calls += 1
        groups = prompt.split("<memory_pair_groups>\n", 1)[1].split(
            "\n</memory_pair_groups>",
            1,
        )[0]
        payload = json.loads(groups)
        assert len(payload) == 1
        assert payload[0]["challenger"]["content"] == "A7 remains excluded."
        assert payload[0]["candidates"][0]["candidate"]["content"] == "A7 is removed."
        return MemoryRelationResponse(
            decisions=[
                MemoryRelationDecision(
                    pair_index=payload[0]["candidates"][0]["pair_index"],
                    classification="equivalent",
                    direction="symmetric",
                    same_subject_and_scope=True,
                    incompatible_assertions="",
                    reason="Both claims state that A7 is excluded.",
                )
            ]
        )


class _SupportValidatingNoopClient(_NoopClient):
    def __init__(
        self,
        memory_id: str,
        *,
        supported: bool,
        evidence_quote: str = "",
        required_evidence_quote: str = "",
        required_evidence_quotes: tuple[str, ...] = (),
    ) -> None:
        super().__init__(memory_id)
        self.supported = supported
        self.evidence_quote = evidence_quote
        self.required_evidence_quote = required_evidence_quote
        self.required_evidence_quotes = required_evidence_quotes
        self.validation_calls = 0

    async def validate_memory_support(self, prompt: str, **kwargs):
        del kwargs
        self.validation_calls += 1
        assert '"memory_claim"' in prompt
        assert "A7 is retained for regular payroll." in prompt or "A7 is removed." in prompt
        payload = json.loads(
            prompt.split("<case_json>\n", 1)[1].split("\n</case_json>", 1)[0]
        )
        return MemorySupportValidationResponse(
            supported=self.supported,
            evidence_quote=self.evidence_quote,
            required_evidence=[
                MemorySupportValidationRequiredEvidence(
                    selector=item["selector"],
                    evidence_quote=quote,
                )
                for item, quote in zip(
                    payload["required"],
                    self.required_evidence_quotes
                    or tuple(
                        self.required_evidence_quote
                        for _item in payload["required"]
                    ),
                    strict=True,
                )
                if quote
            ],
            reason=(
                "The applicability remains regular payroll."
                if self.supported
                else "The applicability changed from regular to off-cycle payroll."
            ),
        )


class _UnavailableSupportValidatingNoopClient(_NoopClient):
    async def validate_memory_support(self, prompt: str, **kwargs):
        del prompt, kwargs
        raise StructuredLlmError(
            "structured LLM returned an invalid response",
            error_code="ValidationError",
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


async def _seed_incumbent_support(
    db: Database,
    *,
    projection,
    memory_id: str = "mem-old",
    memory_content: str = "A7 is removed.",
    observation_index: int = 0,
    source_type: str = "confluence",
    access_context_hash: str = "workspace-eng",
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
    revisions_by_observation = {item.observation_id: item for item in projection.observation_revisions}
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
        access_context_hash=access_context_hash,
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
            access_context_hash=access_context_hash,
        )
    )
    return incumbent


async def _add_independent_legacy_support_alternative(
    db: Database,
    *,
    incumbent: Memory,
    projection: SourceProjection,
    doc_id: str,
    access_context_hash: str,
) -> str:
    await db.record_source_projection(projection)
    now = datetime(2026, 7, 15, tzinfo=timezone.utc)
    await db.upsert_document(
        DocumentRecord(
            doc_id=doc_id,
            source="src-1",
            source_url=f"https://example.test/{doc_id}",
            title="Independent Page",
            space_or_project="ENG",
            author=None,
            last_modified=now,
            labels=[],
            version="1",
            content_hash=f"{doc_id}-hash",
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
        doc_id,
        "confluence",
        incumbent.content,
        source_updated_at=now,
    )
    observation = projection.observations[0]
    revision = projection.observation_revisions[0]
    unit_id = f"eu-{incumbent.id}-{doc_id}"
    await db.upsert_evidence_unit(
        EvidenceUnit(
            id=unit_id,
            source_id="src-1",
            doc_id=doc_id,
            doc_revision_id=projection.source_unit_revisions[0].id,
            source_type="confluence",
            source_anchor=observation.id,
            source_lineage_id=projection.source_units[0].id,
            project_key="ENG",
            visibility="workspace",
            owner_user_id=None,
            repo_identifier=None,
            content=revision.content,
            excerpt=incumbent.content,
            evidence_provenance=EvidenceContentProvenance.SOURCE_EXCERPT,
            access_context_hash=access_context_hash,
        )
    )
    [reference] = await db.record_evidence_references(
        unit_id,
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
    await db.upsert_memory_support_assertion(
        MemorySupportAssertion(
            id=f"support-{incumbent.id}-{doc_id}",
            memory_id=incumbent.id,
            evidence_reference_id=reference.id or "",
            source_id="src-1",
            access_context_hash=access_context_hash,
        )
    )
    return unit_id


@dataclass(frozen=True, slots=True)
class _V2StaleCrossUnitScenario:
    access_context_hash: str
    first: SourceProjection
    incumbent: Memory
    alternative: SourceProjection
    alternative_unit_id: str
    alternative_current: SourceProjection


async def _seed_v2_stale_cross_unit_scenario(
    db: Database,
    *,
    prefix: str,
) -> _V2StaleCrossUnitScenario:
    access_context_hash = lifecycle_access_context_hash(
        visibility="workspace",
        owner_user_id=None,
        project_key="ENG",
        repo_identifier=None,
    )
    first = _projection(run_id=f"{prefix}-1", body="A7 is removed.")
    await db.record_source_projection(first)
    incumbent = await _seed_incumbent_support(
        db,
        projection=first,
        access_context_hash=access_context_hash,
    )
    alternative = _projection(
        run_id=f"{prefix}-alternative-1",
        body="A7 is removed.",
        item_id="confluence-456",
        page_id="456",
    )
    alternative_unit_id = await _add_independent_legacy_support_alternative(
        db,
        incumbent=incumbent,
        projection=alternative,
        doc_id="confluence-456",
        access_context_hash=access_context_hash,
    )
    cutover = await db.report_support_scope_cutover()
    await db.apply_support_scope_v2_cutover(
        expected_report_id=cutover.id,
        owner_id=prefix,
    )
    await db.enable_lifecycle_gate("src-1")
    alternative_current = _projection(
        run_id=f"{prefix}-alternative-2",
        body="A7 is excluded in the current release.",
        item_id="confluence-456",
        page_id="456",
        prior=alternative.source_unit_revisions[0],
        prior_observations={
            alternative.observations[0].id: alternative.observation_revisions[0]
        },
    )
    await db.record_source_projection(alternative_current)
    return _V2StaleCrossUnitScenario(
        access_context_hash=access_context_hash,
        first=first,
        incumbent=incumbent,
        alternative=alternative,
        alternative_unit_id=alternative_unit_id,
        alternative_current=alternative_current,
    )


async def _add_same_unit_legacy_support_alternative(
    db: Database,
    *,
    incumbent: Memory,
    projection: SourceProjection,
    access_context_hash: str,
) -> str:
    observation = projection.observations[0]
    revision = projection.observation_revisions[0]
    excerpt = "A7 is removed."
    start = revision.content.index(excerpt)
    unit_id = f"eu-{incumbent.id}-same-unit-alternative"
    await db.upsert_evidence_unit(
        EvidenceUnit(
            id=unit_id,
            source_id="src-1",
            doc_id="confluence-123",
            doc_revision_id=projection.source_unit_revisions[0].id,
            source_type="confluence",
            source_anchor=observation.id,
            source_lineage_id=projection.source_units[0].id,
            project_key="ENG",
            visibility="workspace",
            owner_user_id=None,
            repo_identifier=None,
            content=excerpt,
            excerpt=excerpt,
            evidence_provenance=EvidenceContentProvenance.SOURCE_EXCERPT,
            access_context_hash=access_context_hash,
        )
    )
    [reference] = await db.record_evidence_references(
        unit_id,
        (
            EvidenceReference(
                role=EvidenceRole.PRIMARY,
                anchor=SourceAnchor(
                    kind=AnchorKind.REVISION_RANGE,
                    observation_id=observation.id,
                    observation_revision_id=revision.id,
                    range_start=start,
                    range_end=start + len(excerpt),
                ),
            ),
        ),
    )
    await db.upsert_memory_support_assertion(
        MemorySupportAssertion(
            id=f"support-{incumbent.id}-same-unit-alternative",
            memory_id=incumbent.id,
            evidence_reference_id=reference.id or "",
            source_id="src-1",
            access_context_hash=access_context_hash,
        )
    )
    return unit_id


def test_exact_unique_quote_materializes_revision_range_primary_anchor() -> None:
    projection = _projection(
        run_id="projection-exact-range",
        body="Intro.\n\nA7 is removed.\n\nClosing note.",
    )
    quote = "A7 is removed."
    evidence = build_projected_claim_evidence(
        projection=projection,
        raw_memories=(
            RawMemory(
                content="A7 is removed.",
                memory_type="decision",
                evidence_quote=quote,
            ),
        ),
        doc_id="confluence-123",
        source_type="confluence",
        project_key="ENG",
        visibility="workspace",
        owner_user_id=None,
        repo_identifier=None,
        access_context_hash="workspace-eng",
        extractor_run_id=projection.run_id,
    )

    [primary] = [reference for reference in evidence.references if reference.role is EvidenceRole.PRIMARY]
    revision = projection.observation_revisions[0]
    expected_start = revision.content.index(quote)
    assert primary.anchor.kind is AnchorKind.REVISION_RANGE
    assert primary.anchor.observation_revision_id == revision.id
    assert primary.anchor.range_start == expected_start
    assert primary.anchor.range_end == expected_start + len(quote)


def test_repeated_quote_keeps_conservative_whole_observation_anchor() -> None:
    projection = _projection(
        run_id="projection-ambiguous-range",
        body="Repeated fact.\n\nRepeated fact.",
    )
    evidence = build_projected_claim_evidence(
        projection=projection,
        raw_memories=(
            RawMemory(
                content="Repeated fact.",
                memory_type="fact",
                evidence_quote="Repeated fact.",
            ),
        ),
        doc_id="confluence-123",
        source_type="confluence",
        project_key="ENG",
        visibility="workspace",
        owner_user_id=None,
        repo_identifier=None,
        access_context_hash="workspace-eng",
        extractor_run_id=projection.run_id,
    )

    [primary] = [reference for reference in evidence.references if reference.role is EvidenceRole.PRIMARY]
    assert primary.anchor.kind is AnchorKind.WHOLE_OBSERVATION


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
        prior_observations={revision.observation_id: revision for revision in first.observation_revisions},
    )
    raw = RawMemory(
        content=incumbent.content,
        memory_type=incumbent.memory_type,
        confidence=incumbent.confidence,
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
        support_set_hashes={incumbent.id: await db.get_memory_support_set_hash(incumbent.id)},
        observation_revision_ids=tuple(revision.id for revision in second.observation_revisions),
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
async def test_atomic_projection_lifecycle_commits_document_and_derivation(
    db: Database,
) -> None:
    first = _projection(
        run_id="projection-derivation-atomic-1",
        body="A7 is removed.",
    )
    await db.record_source_projection(first)
    second = _projection(
        run_id="projection-derivation-atomic-2",
        body="A7 remains removed.",
        prior=first.source_unit_revisions[0],
        prior_observations={revision.observation_id: revision for revision in first.observation_revisions},
    )
    original_document = await db.get_document("confluence-123")
    assert original_document is not None
    staged_document = replace(
        original_document,
        title="Atomically updated page",
        content_hash=content_hash("A7 remains removed."),
    )
    context = SourceUnitDerivationContext(
        document=staged_document,
        doc_type="confluence",
        project_key="ENG",
        repo_identifier=None,
        document_content="A7 remains removed.",
        update_mode="full_document",
        changed_hunks=None,
        update_plan_stats=None,
        source_updated_at=None,
        user_id=None,
        source_activity_epoch=None,
    )
    attempt = await db.stage_source_derivation(
        source_derivation_manifest(
            second,
            (),
            context=context,
        )
    )
    assert attempt.context.document == staged_document
    delta = second.deltas[0]
    scope = ReconciliationScope(
        id="scope-derivation-atomic",
        source_id="src-1",
        source_unit_id=delta.source_unit_id,
        base_unit_revision_id=delta.previous_unit_revision_id,
        target_unit_revision_id=delta.current_unit_revision_id,
    )
    plan = build_lifecycle_plan(
        plan_id="plan-derivation-atomic",
        scope=scope,
        gate_state=LifecycleGateState.GATED,
        operations=(),
        incumbents={},
        source_support_reference_ids={},
        all_active_support_reference_ids={},
        support_set_hashes={},
        observation_revision_ids=tuple(revision.id for revision in second.observation_revisions),
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

    with pytest.raises(
        ValueError,
        match="projection identity mismatch",
    ):
        await db.apply_source_projection_lifecycle(
            replace(second, source_type="jira"),
            plan,
            document=staged_document,
            derivation_id=attempt.id,
            derivation_context_identity_hash=(attempt.context_identity_hash),
        )

    with pytest.raises(
        ValueError,
        match="context identity mismatch",
    ):
        await db.apply_source_projection_lifecycle(
            second,
            plan,
            document=staged_document,
            derivation_id=attempt.id,
            derivation_context_identity_hash="wrong-context",
        )

    with pytest.raises(
        ValueError,
        match="Document identity mismatch",
    ):
        await db.apply_source_projection_lifecycle(
            second,
            plan,
            document=replace(staged_document, title="Unstaged title"),
            derivation_id=attempt.id,
            derivation_context_identity_hash=(attempt.context_identity_hash),
        )

    with pytest.raises(
        ValueError,
        match="requires its staged Document",
    ):
        await db.apply_source_projection_lifecycle(
            second,
            plan,
            derivation_id=attempt.id,
        )

    await db.apply_source_projection_lifecycle(
        second,
        plan,
        document=staged_document,
        derivation_id=attempt.id,
        derivation_context_identity_hash=(attempt.context_identity_hash),
        runtime_bundle=(
            runtime_bundle := bind_source_lifecycle_outcome(
                source_id="src-1",
                source_type="confluence",
                doc_id="confluence-123",
                source_unit_id=delta.source_unit_id,
                base_unit_revision_id=delta.previous_unit_revision_id,
                target_unit_revision_id=delta.current_unit_revision_id,
                projection_run_id=second.run_id,
                operation_input_hash="a" * 64,
                execution_owner_id="sync-run-atomic:lease-1",
                outcome="expected",
                reason_code="lifecycle_plan_applied",
                attempt_count=1,
                duration_ms=25,
                incumbent_count=0,
                relation_pair_count=0,
                mutation_count=0,
                review_count=0,
                model_call_count=0,
                occurred_at=datetime(2026, 8, 17, tzinfo=timezone.utc),
            )
        ),
    )

    committed_document = await db.get_document("confluence-123")
    [committed_attempt] = await db.list_source_derivation_attempts(source_id="src-1")
    current_unit = await db.get_current_source_unit_revision(second.source_units[0].id)
    assert committed_document is not None
    assert committed_document.title == "Atomically updated page"
    assert current_unit is not None
    assert current_unit.id == second.source_unit_revisions[0].id
    assert committed_attempt.status == "applied"
    events = await db.list_agent_runtime_events(
        AgentRuntimeEventQuery(
            occurred_from=datetime(2026, 8, 16, tzinfo=timezone.utc),
            occurred_to=datetime(2026, 8, 18, tzinfo=timezone.utc),
            event_id=runtime_bundle.event.event_id,
            limit=10,
        )
    )
    assessments = await db.list_agent_assessments(
        AgentAssessmentQuery(
            occurred_from=datetime(2026, 8, 16, tzinfo=timezone.utc),
            occurred_to=datetime(2026, 8, 18, tzinfo=timezone.utc),
            target_event_id=runtime_bundle.event.event_id,
            limit=10,
        )
    )
    assert events == [runtime_bundle.event]
    assert assessments == [runtime_bundle.assessment]


@pytest.mark.asyncio
async def test_source_deriver_persists_completed_batch_before_later_worker_failure(
    db: Database,
) -> None:
    projection = _projection(
        run_id="projection-durable-batch-progress",
        body="A7 remains removed.",
    )
    document = await db.get_document("confluence-123")
    assert document is not None
    observation_ids = tuple(observation.id for observation in projection.observations)
    batches = (
        ProjectionExtractionBatch(
            id="batch-first",
            source_unit_id=projection.source_units[0].id,
            primary_image_bytes=0,
            primary_observation_ids=(observation_ids[0],),
            primary_content_by_observation_id=((observation_ids[0], "first"),),
            context_observation_ids=(),
            context_observation_ids_by_primary=((observation_ids[0], ()),),
            primary_markdown="first",
            context_markdown="",
        ),
        ProjectionExtractionBatch(
            id="batch-second",
            source_unit_id=projection.source_units[0].id,
            primary_image_bytes=0,
            primary_observation_ids=(observation_ids[-1],),
            primary_content_by_observation_id=((observation_ids[-1], "second"),),
            context_observation_ids=(),
            context_observation_ids_by_primary=((observation_ids[-1], ()),),
            primary_markdown="second",
            context_markdown="",
        ),
    )

    async def extract(batch):
        if batch.id == "batch-second":
            raise RuntimeError("worker interrupted")
        return MemoryExtractionResult()

    with pytest.raises(RuntimeError, match="worker interrupted"):
        await SourceUnitDeriver(
            db,
            plan_work=lambda _projection, _context: batches,
        ).derive(
            SourceUnitDerivationRequest(
                projection=projection,
                context=SourceUnitDerivationContext(
                    document=document,
                    doc_type="confluence",
                    project_key="ENG",
                    repo_identifier=None,
                    document_content="A7 remains removed.",
                    update_mode="full_document",
                    changed_hunks=None,
                    update_plan_stats=None,
                    source_updated_at=None,
                    user_id=None,
                    source_activity_epoch=None,
                ),
                extract_batch=extract,
                max_concurrent=1,
            )
        )

    [attempt] = await db.list_source_derivation_attempts(source_id="src-1")
    assert {batch.batch_id: batch.status for batch in attempt.batches} == {
        "batch-first": "completed",
        "batch-second": "pending",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source_type",
    ("confluence", "jira", "github", "local_markdown", "teams", "agent_session", "extension"),
)
async def test_source_deriver_binds_provider_neutral_quality_events_to_current_lineage(
    db: Database,
    source_type: str,
) -> None:
    projection = replace(
        _projection(
            run_id=f"projection-agent-eval-{source_type}",
            body="The payroll tracing procedure remains active.",
        ),
        source_type=source_type,
    )
    document = await db.get_document("confluence-123")
    assert document is not None

    async def extract(batch):
        record_quality_signal(
            QualitySignal(
                event_name="evidence_admission_outcome",
                outcome="rejected",
                reason_code="unknown_evidence_block_id",
                observation_id=batch.primary_observation_ids[0],
                candidate_hash="a" * 64,
            )
        )
        return MemoryExtractionResult(
            metadata={
                "extraction_model": "anthropic/claude-sonnet",
                "invalid_evidence_block_count": 1,
            }
        )

    result = await SourceUnitDeriver(db).derive(
        SourceUnitDerivationRequest(
            projection=projection,
            context=SourceUnitDerivationContext(
                document=document,
                doc_type=source_type,
                project_key="ENG",
                repo_identifier=None,
                document_content="The payroll tracing procedure remains active.",
                update_mode="full_document",
                changed_hunks=None,
                update_plan_stats=None,
                source_updated_at=None,
                user_id=None,
                source_activity_epoch=None,
            ),
            extract_batch=extract,
            max_concurrent=1,
        )
    )
    rows = await db.list_agent_runtime_events(
        AgentRuntimeEventQuery(
            occurred_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
            occurred_to=datetime(2027, 1, 1, tzinfo=timezone.utc),
            requesting_user_id="user-1",
            include_private=True,
            source_type=source_type,
            event_name="evidence_admission_outcome",
        )
    )

    assert len(rows) == 1
    event = rows[0]
    assert event.source_id == projection.source_id
    assert event.source_type == source_type
    assert event.projection_run_id == projection.run_id
    assert event.derivation_id == result.derivation.id
    assert event.target_unit_revision_id == projection.source_unit_revisions[0].id
    assert event.observation_revision_id == projection.observation_revisions[0].id
    assert event.model == "anthropic/claude-sonnet"
    assert len(event.trace_id or "") == 32
    assert not hasattr(event, "memory_content")
    assessments = await db.list_agent_assessments(
        AgentAssessmentQuery(
            occurred_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
            occurred_to=datetime(2027, 1, 1, tzinfo=timezone.utc),
            requesting_user_id="user-1",
            include_private=True,
            source_id=projection.source_id,
        )
    )
    assert [(assessment.target_event_id, assessment.criterion, assessment.label) for assessment in assessments] == [
        (event.event_id, "evidence_reference_validity", "fail")
    ]


@pytest.mark.asyncio
async def test_source_derivation_separates_exact_payload_hash_from_stable_identity(
    db: Database,
) -> None:
    first = _projection(
        run_id="projection-identity-first",
        body="A7 remains removed.",
    )
    second = replace(
        first,
        run_id="projection-identity-retry",
        checkpoint={"cursor": "later-operational-cursor"},
        observation_revisions=tuple(
            replace(
                revision,
                observed_at="2026-07-27T12:00:00+00:00",
            )
            for revision in first.observation_revisions
        ),
        source_unit_revisions=tuple(
            replace(
                revision,
                observed_at="2026-07-27T12:00:00+00:00",
            )
            for revision in first.source_unit_revisions
        ),
    )
    document = await db.get_document("confluence-123")
    assert document is not None
    context = SourceUnitDerivationContext(
        document=document,
        doc_type="confluence",
        project_key="ENG",
        repo_identifier=None,
        document_content="A7 remains removed.",
        update_mode="full_document",
        changed_hunks=None,
        update_plan_stats=None,
        source_updated_at=None,
        user_id=None,
        source_activity_epoch=None,
    )
    first_manifest = source_derivation_manifest(
        first,
        (),
        context=context,
    )
    second_manifest = source_derivation_manifest(
        second,
        (),
        context=context,
    )

    assert first_manifest.id == second_manifest.id
    assert first_manifest.projection_identity_hash == second_manifest.projection_identity_hash
    assert first_manifest.projection_payload_hash != second_manifest.projection_payload_hash
    assert (
        first_manifest.projection_payload_hash
        == hashlib.sha256(
            json.dumps(
                first_manifest.projection_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    )

    first_attempt = await db.stage_source_derivation(first_manifest)
    retry_attempt = await db.stage_source_derivation(second_manifest)
    next_epoch_manifest = source_derivation_manifest(
        second,
        (),
        context=replace(context, source_activity_epoch=2),
    )
    next_epoch_attempt = await db.stage_source_derivation(next_epoch_manifest)

    assert retry_attempt.id == first_attempt.id
    assert retry_attempt.projection_payload_hash == first_manifest.projection_payload_hash
    assert retry_attempt.projection.run_id == first.run_id
    assert next_epoch_attempt.id != first_attempt.id
    assert next_epoch_attempt.target_unit_revision_id == (first_attempt.target_unit_revision_id)
    assert next_epoch_attempt.context.source_activity_epoch == 2


def test_single_observation_uses_projection_authority_when_document_view_differs():
    message_content = "Use contextId to find the traceId before following the request logs."
    projection = _teams_projection(
        run_id="teams-authority-view",
        message_content=message_content,
    )
    rendered_document = (
        "# Group: PCC Agent Dev -- Jul 30, 10:00-10:00\n\n"
        "**Group Chat**: PCC Agent Dev\n**Messages**: 1\n\n---\n\n"
        f"**Alex** (2026-07-30T10:00):\n{message_content}\n"
    )
    context = SourceUnitDerivationContext(
        document=SimpleNamespace(
            doc_id="teams-window-1",
            title="Group: PCC Agent Dev -- Jul 30, 10:00-10:00",
            source_url="https://teams.example.test/message/1",
        ),
        doc_type="teams",
        project_key=None,
        repo_identifier=None,
        document_content=rendered_document,
        update_mode="full_document",
        changed_hunks=None,
        update_plan_stats=None,
        source_updated_at=None,
        user_id=None,
        source_activity_epoch=None,
    )

    batches = plan_source_derivation_work(projection, context)

    assert len(batches) == 1
    assert isinstance(batches[0], ProjectionExtractionBatch)
    [revision] = projection.observation_revisions
    assert batches[0].primary_authority_spans == (
        (revision.observation_id, 0, revision.content),
    )


@pytest.mark.asyncio
async def test_projection_extraction_contract_change_invalidates_staged_derivation(
    db: Database,
) -> None:
    projection = _projection(
        run_id="projection-language-contract",
        body="决定：A7 保持启用。",
    )
    document = await db.get_document("confluence-123")
    assert document is not None
    context = SourceUnitDerivationContext(
        document=document,
        doc_type="confluence",
        project_key="ENG",
        repo_identifier=None,
        document_content="决定：A7 保持启用。",
        update_mode="full_document",
        changed_hunks=None,
        update_plan_stats=None,
        source_updated_at=None,
        user_id=None,
        source_activity_epoch=None,
    )

    current_batches = plan_source_derivation_work(projection, context)
    assert current_batches
    previous_batches = tuple(replace(batch, id=f"{batch.id}-v2") for batch in current_batches)
    previous = source_derivation_manifest(
        projection,
        previous_batches,
        context=context,
        extraction_contract_version="projection-extraction-v2",
    )
    previous_attempt = await db.stage_source_derivation(previous)
    for batch in previous_batches:
        previous_attempt = await db.record_source_derivation_batch_result(
            derivation_id=previous.id,
            batch_id=batch.id,
            result=MemoryExtractionResult(memories=[]),
        )
    assert previous_attempt.status == "completed"

    executed_batch_ids: list[str] = []

    async def extract_batch(
        batch: ProjectionExtractionBatch,
    ) -> MemoryExtractionResult:
        executed_batch_ids.append(batch.id)
        return MemoryExtractionResult(memories=[])

    result = await SourceUnitDeriver(db).derive(
        SourceUnitDerivationRequest(
            projection=projection,
            context=context,
            extract_batch=extract_batch,
            max_concurrent=1,
        )
    )

    current = source_derivation_manifest(
        projection,
        current_batches,
        context=context,
    )

    assert current.extraction_contract_version != previous.extraction_contract_version
    assert current.id != previous.id
    assert result.derivation.id == current.id
    assert result.reused_batch_count == 0
    assert result.executed_batch_count == len(current_batches)
    assert executed_batch_ids == [batch.id for batch in current_batches]


@pytest.mark.asyncio
async def test_batch_result_and_runtime_events_rollback_together(db: Database) -> None:
    projection = _projection(
        run_id="projection-agent-runtime-transaction",
        body="The payroll tracing procedure remains active.",
    )
    document = await db.get_document("confluence-123")
    assert document is not None
    context = SourceUnitDerivationContext(
        document=document,
        doc_type="confluence",
        project_key="ENG",
        repo_identifier=None,
        document_content="The payroll tracing procedure remains active.",
        update_mode="full_document",
        changed_hunks=None,
        update_plan_stats=None,
        source_updated_at=None,
        user_id=None,
        source_activity_epoch=None,
    )
    batches = plan_source_derivation_work(projection, context)
    manifest = source_derivation_manifest(projection, batches, context=context)
    await db.stage_source_derivation(manifest)
    [event] = bind_quality_signals(
        (
            QualitySignal(
                event_name="extraction_batch_outcome",
                outcome="expected",
                reason_code="zero_candidates",
            ),
        ),
        source_id=projection.source_id,
        source_type=projection.source_type,
        doc_id=document.doc_id,
        source_unit_id=manifest.source_unit_id,
        target_unit_revision_id=manifest.target_unit_revision_id,
        projection_run_id=projection.run_id,
        derivation_id=manifest.id,
        batch_id=batches[0].id,
        batch_attempt=1,
        extraction_contract_version=manifest.extraction_contract_version,
    )
    invalid_event = replace(event, source_id="missing-source")

    with pytest.raises(Exception, match="FOREIGN KEY"):
        await db.record_source_derivation_batch_result(
            derivation_id=manifest.id,
            batch_id=batches[0].id,
            result=MemoryExtractionResult(memories=[]),
            runtime_events=(invalid_event,),
        )

    attempt = await db.stage_source_derivation(manifest)
    assert attempt.batches[0].status == "pending"


def test_source_derivation_diagnostics_reject_content_and_bound_field_count() -> None:
    error_type, error_code, fields = safe_derivation_error(
        MemoryExtractionResult(
            error_type="structured_llm_error",
            metadata={
                "safe_error_code": "secret response content",
                "safe_validation_fields": [
                    {
                        "location": f"memories.{index}.memory_type",
                        "type": "literal_error",
                    }
                    for index in range(40)
                ],
            },
        )
    )

    assert error_type == "structured_llm_error"
    assert error_code is None
    assert len(fields) == 32
    assert fields[0] == (
        "memories.0.memory_type",
        "literal_error",
    )


@pytest.mark.asyncio
async def test_noop_without_current_evidence_rolls_back_stale_support(db: Database) -> None:
    first = _projection(run_id="projection-stale-1", body="A7 is removed.")
    await db.record_source_projection(first)
    seeded = await _seed_incumbent_support(db, projection=first)
    incumbent = await db.get_memory(seeded.id)
    assert incumbent is not None
    await db.enable_lifecycle_gate("src-1")
    old_support = await db.get_active_memory_support_reference_ids(incumbent.id)
    original_document = await db.get_document("confluence-123")
    assert original_document is not None
    second = _projection(
        run_id="projection-stale-2",
        body="A7 is retained.",
        prior=first.source_unit_revisions[0],
        prior_observations={revision.observation_id: revision for revision in first.observation_revisions},
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
        support_set_hashes={incumbent.id: await db.get_memory_support_set_hash(incumbent.id)},
        observation_revision_ids=tuple(revision.id for revision in second.observation_revisions),
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
    staged_document = replace(
        original_document,
        title="Must roll back",
        content_hash=content_hash("A7 is retained."),
    )
    attempt = await db.stage_source_derivation(
        source_derivation_manifest(
            second,
            (),
            context=SourceUnitDerivationContext(
                document=staged_document,
                doc_type="confluence",
                project_key="ENG",
                repo_identifier=None,
                document_content="A7 is retained.",
                update_mode="full_document",
                changed_hunks=None,
                update_plan_stats=None,
                source_updated_at=None,
                user_id=None,
                source_activity_epoch=None,
            ),
        )
    )

    with pytest.raises(ValueError, match="stale or ambiguous source support"):
        await db.apply_source_projection_lifecycle(
            second,
            plan,
            document=staged_document,
            derivation_id=attempt.id,
            derivation_context_identity_hash=(attempt.context_identity_hash),
        )

    current_unit = await db.get_current_source_unit_revision(first.source_units[0].id)
    assert current_unit is not None
    assert current_unit.id == first.source_unit_revisions[0].id
    assert await db.get_active_memory_support_reference_ids(incumbent.id) == old_support
    assert (await db.get_document("confluence-123")).title == (original_document.title)
    [rolled_back_attempt] = await db.list_source_derivation_attempts(source_id="src-1")
    assert rolled_back_attempt.status == "completed"


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
        prior_observations={revision.observation_id: revision for revision in first.observation_revisions},
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
        observation_revision_ids=tuple(revision.id for revision in second.observation_revisions),
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
    assert await db.get_lifecycle_review(str(plan.mutations[0].payload["review_id"])) is None


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
        prior_observations={revision.observation_id: revision for revision in first.observation_revisions},
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
        support_set_hashes={incumbent.id: await db.get_memory_support_set_hash(incumbent.id)},
        observation_revision_ids=tuple(revision.id for revision in second.observation_revisions),
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
    support_state = (await db.get_active_memory_support_states((incumbent.id,)))[incumbent.id]
    assert support_state.reference_ids == old_support
    assert support_state.current_reference_ids == ()
    review = await db.get_lifecycle_review(str(plan.mutations[0].payload["review_id"]))
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
        prior_observations={revision.observation_id: revision for revision in first.observation_revisions},
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
        support_set_hashes={incumbent.id: await db.get_memory_support_set_hash(incumbent.id)},
        observation_revision_ids=tuple(revision.id for revision in changed.observation_revisions),
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
        support_set_hashes={incumbent.id: await db.get_memory_support_set_hash(incumbent.id)},
        observation_revision_ids=tuple(revision.id for revision in later.observation_revisions),
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
    review = await db.get_lifecycle_review(str(review_plan.mutations[0].payload["review_id"]))
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
        prior_observations={revision.observation_id: revision for revision in first.observation_revisions},
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
        coverage_proof=SimpleNamespace(mandatory_incumbent_ids=(incumbent.id,)),
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
        coverage_proof=SimpleNamespace(mandatory_incumbent_ids=(incumbent.id,)),
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
        coverage_proof=SimpleNamespace(mandatory_incumbent_ids=(incumbent.id,)),
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
        prior_observations={revision.observation_id: revision for revision in first.observation_revisions},
    )
    second = replace(
        second,
        observations=other.observations + second.observations,
        observation_revisions=(other.observation_revisions + second.observation_revisions),
        source_units=other.source_units + second.source_units,
        source_unit_revisions=(other.source_unit_revisions + second.source_unit_revisions),
        relations=other.relations + second.relations,
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        cross_document_candidates=_candidate_retriever(adapters),
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
        unit_support=await db.get_source_unit_support_reference_ids(first.source_units[0].id),
        projection=second,
        access_context_hash="workspace-eng",
    )

    assert rebound.action is ReconcileAction.NOOP
    assert rebound.memory is not None
    assert rebound.memory.source_observation_id == first.observations[0].id
    assert other_reference.id in (await db.get_active_memory_support_reference_ids(incumbent.id))


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
        prior_observations={first.observations[0].id: first.observation_revisions[0]},
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=_NoopClient(incumbent.id),
    )

    stats = await engine.prepare_and_commit_projected_lifecycle(
        projection=second,
        doc_id="confluence-123",
        raw_memories=[],
        doc_type="design-doc",
        project_key="ENG",
        repo_identifier=None,
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
async def test_v2_incremental_noop_rebinds_complete_unit_to_current_revision(
    db: Database,
) -> None:
    first = _projection(
        run_id="projection-v2-incremental-keep-1",
        body="A7 is removed.\nOld deployment note.",
    )
    await db.record_source_projection(first)
    incumbent = await _seed_incumbent_support(
        db,
        projection=first,
        access_context_hash=lifecycle_access_context_hash(
            visibility="workspace",
            owner_user_id=None,
            project_key="ENG",
            repo_identifier=None,
        ),
    )
    cutover = await db.report_support_scope_cutover()
    await db.apply_support_scope_v2_cutover(
        expected_report_id=cutover.id,
        owner_id="test-v2-noop-rebind",
    )
    await db.enable_lifecycle_gate("src-1")
    old_support = await db.get_active_memory_support_unit_ids(incumbent.id)
    second = _projection(
        run_id="projection-v2-incremental-keep-2",
        body="A7 is removed.\nNew deployment note.",
        prior=first.source_unit_revisions[0],
        prior_observations={first.observations[0].id: first.observation_revisions[0]},
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=_SupportValidatingNoopClient(
            incumbent.id,
            supported=True,
            evidence_quote="A7 is removed.",
        ),
    )

    stats = await engine.prepare_and_commit_projected_lifecycle(
        projection=second,
        doc_id="confluence-123",
        raw_memories=[],
        doc_type="design-doc",
        project_key="ENG",
        repo_identifier=None,
        document_content=second.observation_revisions[0].content,
        update_mode="diff_guided",
        changed_hunks="Old deployment note -> New deployment note",
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 15, 10, 36, tzinfo=timezone.utc),
    )

    current_support = await db.get_active_memory_support_unit_ids(incumbent.id)
    assert stats["noop"] == 1
    assert current_support
    assert set(current_support).isdisjoint(old_support)
    evidence = await db.get_active_memory_support_evidence(
        incumbent.id,
        source_id="src-1",
    )
    assert {item.anchor.observation_revision_id for item in evidence} == {
        second.observation_revisions[0].id
    }


@pytest.mark.asyncio
async def test_v2_noop_rebind_preserves_independent_support_alternative(
    db: Database,
) -> None:
    access_context_hash = lifecycle_access_context_hash(
        visibility="workspace",
        owner_user_id=None,
        project_key="ENG",
        repo_identifier=None,
    )
    first = _projection(
        run_id="projection-v2-alternatives-1",
        body="A7 is removed.\nOld deployment note.",
    )
    await db.record_source_projection(first)
    incumbent = await _seed_incumbent_support(
        db,
        projection=first,
        access_context_hash=access_context_hash,
    )
    alternative = _projection(
        run_id="projection-v2-alternative-independent",
        body="A7 is removed.",
        item_id="confluence-456",
        page_id="456",
    )
    alternative_unit_id = await _add_independent_legacy_support_alternative(
        db,
        incumbent=incumbent,
        projection=alternative,
        doc_id="confluence-456",
        access_context_hash=access_context_hash,
    )
    cutover = await db.report_support_scope_cutover()
    await db.apply_support_scope_v2_cutover(
        expected_report_id=cutover.id,
        owner_id="test-v2-alternative-rebind",
    )
    await db.enable_lifecycle_gate("src-1")
    original_unit_id = f"eu-{incumbent.id}"
    assert set(await db.get_active_memory_support_unit_ids(incumbent.id)) == {
        original_unit_id,
        alternative_unit_id,
    }
    second = _projection(
        run_id="projection-v2-alternatives-2",
        body="A7 is removed.\nNew deployment note.",
        prior=first.source_unit_revisions[0],
        prior_observations={
            first.observations[0].id: first.observation_revisions[0]
        },
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=_SupportValidatingNoopClient(
            incumbent.id,
            supported=True,
            evidence_quote="A7 is removed.",
        ),
    )

    stats = await engine.prepare_and_commit_projected_lifecycle(
        projection=second,
        doc_id="confluence-123",
        raw_memories=[],
        doc_type="design-doc",
        project_key="ENG",
        repo_identifier=None,
        document_content=second.observation_revisions[0].content,
        update_mode="diff_guided",
        changed_hunks="Old deployment note -> New deployment note",
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 15, 10, 36, tzinfo=timezone.utc),
    )

    current_support = set(
        await db.get_active_memory_support_unit_ids(incumbent.id)
    )
    assert stats["noop"] == 1
    assert alternative_unit_id in current_support
    assert original_unit_id not in current_support
    assert len(current_support) == 2


@pytest.mark.asyncio
async def test_v2_noop_same_unit_alternatives_stage_review_without_collapsing(
    db: Database,
) -> None:
    access_context_hash = lifecycle_access_context_hash(
        visibility="workspace",
        owner_user_id=None,
        project_key="ENG",
        repo_identifier=None,
    )
    first = _projection(
        run_id="projection-v2-same-unit-alternatives-1",
        body="A7 is removed.\nOld deployment note.",
    )
    await db.record_source_projection(first)
    incumbent = await _seed_incumbent_support(
        db,
        projection=first,
        access_context_hash=access_context_hash,
    )
    alternative_unit_id = await _add_same_unit_legacy_support_alternative(
        db,
        incumbent=incumbent,
        projection=first,
        access_context_hash=access_context_hash,
    )
    cutover = await db.report_support_scope_cutover()
    await db.apply_support_scope_v2_cutover(
        expected_report_id=cutover.id,
        owner_id="test-v2-same-unit-alternatives",
    )
    await db.enable_lifecycle_gate("src-1")
    original_unit_id = f"eu-{incumbent.id}"
    old_support = await db.get_active_memory_support_unit_ids(incumbent.id)
    assert set(old_support) == {original_unit_id, alternative_unit_id}
    second = _projection(
        run_id="projection-v2-same-unit-alternatives-2",
        body="A7 is removed.\nNew deployment note.",
        prior=first.source_unit_revisions[0],
        prior_observations={
            first.observations[0].id: first.observation_revisions[0]
        },
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=_SupportValidatingNoopClient(
            incumbent.id,
            supported=True,
            evidence_quote="A7 is removed.",
        ),
    )

    stats = await engine.prepare_and_commit_projected_lifecycle(
        projection=second,
        doc_id="confluence-123",
        raw_memories=[],
        doc_type="design-doc",
        project_key="ENG",
        repo_identifier=None,
        document_content=second.observation_revisions[0].content,
        update_mode="diff_guided",
        changed_hunks="Old deployment note -> New deployment note",
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 15, 10, 36, tzinfo=timezone.utc),
    )

    assert stats["pending_review"] == 1
    assert await db.get_active_memory_support_unit_ids(incumbent.id) == old_support
    current = await db.get_current_source_unit_revision(first.source_units[0].id)
    assert current is not None
    assert current.id == second.source_unit_revisions[0].id
    [review] = await db.list_lifecycle_reviews("src-1")
    assert review.status is LifecycleReviewStatus.PENDING
    assert review.reason.endswith(": ambiguous")
    assert {
        unit_id
        for mutation in review.staged_evidence["proposed_mutations"]
        for unit_id in mutation["evidence_unit_ids"]
    } == {original_unit_id, alternative_unit_id}


@pytest.mark.asyncio
async def test_v2_noop_postcondition_failure_rolls_back_and_is_non_retryable(
    db: Database,
) -> None:
    access_context_hash = lifecycle_access_context_hash(
        visibility="workspace",
        owner_user_id=None,
        project_key="ENG",
        repo_identifier=None,
    )
    first = _projection(
        run_id="projection-v2-rollback-1",
        body="A7 is removed.\nOld deployment note.",
    )
    await db.record_source_projection(first)
    incumbent = await _seed_incumbent_support(
        db,
        projection=first,
        access_context_hash=access_context_hash,
    )
    alternative = _projection(
        run_id="projection-v2-rollback-alternative",
        body="A7 is removed.",
        item_id="confluence-456",
        page_id="456",
    )
    await _add_independent_legacy_support_alternative(
        db,
        incumbent=incumbent,
        projection=alternative,
        doc_id="confluence-456",
        access_context_hash=access_context_hash,
    )
    cutover = await db.report_support_scope_cutover()
    await db.apply_support_scope_v2_cutover(
        expected_report_id=cutover.id,
        owner_id="test-v2-rollback",
    )
    await db.enable_lifecycle_gate("src-1")
    old_support = await db.get_active_memory_support_unit_ids(incumbent.id)
    await db.db.execute(
        "DELETE FROM memory_sources WHERE memory_id = ? AND doc_id = ?",
        (incumbent.id, "confluence-456"),
    )
    await db.db.commit()
    second = _projection(
        run_id="projection-v2-rollback-2",
        body="A7 is removed.\nNew deployment note.",
        prior=first.source_unit_revisions[0],
        prior_observations={
            first.observations[0].id: first.observation_revisions[0]
        },
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=_SupportValidatingNoopClient(
            incumbent.id,
            supported=True,
            evidence_quote="A7 is removed.",
        ),
    )

    with pytest.raises(SourceUnitLifecycleExecutionError) as raised:
        await engine.prepare_and_commit_projected_lifecycle(
            projection=second,
            doc_id="confluence-123",
            raw_memories=[],
            doc_type="design-doc",
            project_key="ENG",
            repo_identifier=None,
            document_content=second.observation_revisions[0].content,
            update_mode="diff_guided",
            changed_hunks="Old deployment note -> New deployment note",
            update_plan_stats=None,
            source_updated_at=datetime(2026, 7, 15, 10, 36, tzinfo=timezone.utc),
            lifecycle_execution_owner_id="sync-v2-rollback:lease-1",
        )

    assert raised.value.retryable is False
    assert isinstance(raised.value.__cause__, ProjectedSupportInvariantError)
    assert await db.get_active_memory_support_unit_ids(incumbent.id) == old_support
    current = await db.get_current_source_unit_revision(first.source_units[0].id)
    assert current is not None
    assert current.id == first.source_unit_revisions[0].id


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
        prior_observations={first.observations[0].id: first.observation_revisions[0]},
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=_SupportValidatingNoopClient(
            incumbent.id,
            supported=True,
            evidence_quote=current_quote,
        ),
    )

    stats = await engine.prepare_and_commit_projected_lifecycle(
        projection=second,
        doc_id="confluence-123",
        raw_memories=[],
        doc_type="design-doc",
        project_key="ENG",
        repo_identifier=None,
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
async def test_incremental_noop_inexact_current_quote_creates_review(
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
        prior_observations={first.observations[0].id: first.observation_revisions[0]},
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=_SupportValidatingNoopClient(
            incumbent.id,
            supported=True,
            evidence_quote="A quote that is not in the current source.",
        ),
    )

    stats = await engine.prepare_and_commit_projected_lifecycle(
        projection=second,
        doc_id="confluence-123",
        raw_memories=[],
        doc_type="design-doc",
        project_key="ENG",
        repo_identifier=None,
        document_content=second.observation_revisions[0].content,
        update_mode="diff_guided",
        changed_hunks="primary wording changed",
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 16, 10, 36, tzinfo=timezone.utc),
    )

    assert stats["pending_review"] == 1
    current = await db.get_memory(incumbent.id)
    assert current is not None and current.status == "active"
    assert await db.get_active_memory_support_reference_ids(incumbent.id)


@pytest.mark.asyncio
async def test_incremental_noop_unavailable_support_validation_creates_review(
    db: Database,
) -> None:
    first = _projection(
        run_id="projection-primary-validation-unavailable-1",
        body="A7 is removed.",
    )
    await db.record_source_projection(first)
    incumbent = await _seed_incumbent_support(db, projection=first)
    await db.enable_lifecycle_gate("src-1")
    second = _projection(
        run_id="projection-primary-validation-unavailable-2",
        body="The A7 slot remains excluded.",
        prior=first.source_unit_revisions[0],
        prior_observations={first.observations[0].id: first.observation_revisions[0]},
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=_UnavailableSupportValidatingNoopClient(
            incumbent.id,
        ),
    )

    stats = await engine.prepare_and_commit_projected_lifecycle(
        projection=second,
        doc_id="confluence-123",
        raw_memories=[],
        doc_type="design-doc",
        project_key="ENG",
        repo_identifier=None,
        document_content=second.observation_revisions[0].content,
        update_mode="diff_guided",
        changed_hunks="primary wording changed",
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 16, 10, 36, tzinfo=timezone.utc),
    )

    assert stats["pending_review"] == 1
    current = await db.get_memory(incumbent.id)
    assert current is not None and current.status == "active"
    assert await db.get_active_memory_support_reference_ids(incumbent.id)


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
        prior_observations={first.observations[0].id: first.observation_revisions[0]},
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=_SupportValidatingNoopClient(
            incumbent.id,
            supported=False,
        ),
    )

    stats = await engine.prepare_and_commit_projected_lifecycle(
        projection=second,
        doc_id="confluence-123",
        raw_memories=[],
        doc_type="design-doc",
        project_key="ENG",
        repo_identifier=None,
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
async def test_persistent_incomplete_incumbent_audit_fails_closed_without_mutating_incumbent(
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
        prior_observations={first.observations[0].id: first.observation_revisions[0]},
    )
    client = _PersistentlyIncompleteAuditClient(incumbent.id)
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=client,
    )

    with pytest.raises(
        SourceUnitLifecycleExecutionError,
        match="incumbent support response count 0 does not match expected count 1",
    ) as failure:
        await engine.prepare_and_commit_projected_lifecycle(
            projection=second,
            doc_id="confluence-123",
            raw_memories=[],
            doc_type="design-doc",
            project_key="ENG",
            repo_identifier=None,
            document_content=second.observation_revisions[0].content,
            update_mode="diff_guided",
            changed_hunks="removed -> retained",
            update_plan_stats=None,
            source_updated_at=datetime(2026, 7, 16, 10, 36, tzinfo=timezone.utc),
            lifecycle_execution_owner_id="sync-run-incomplete:lease-1",
            lifecycle_attempt_count=3,
        )

    assert client.calls == 2
    current = await db.get_memory(incumbent.id)
    assert current is not None and current.status == "active"
    assert await db.get_active_memory_support_reference_ids(incumbent.id)
    assert failure.value.runtime_bundle.event.outcome == "failed"
    assert failure.value.runtime_bundle.event.reason_code == "support_response_incomplete"
    assert failure.value.runtime_bundle.event.attempt_count == 3
    assert failure.value.runtime_bundle.assessment.label == "fail"
    reviews = await db.list_lifecycle_reviews("src-1")
    assert reviews == []


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
        prior_observations={first.observations[0].id: first.observation_revisions[0]},
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=None,
    )

    stats = await engine.prepare_and_commit_projected_lifecycle(
        projection=second,
        doc_id="confluence-123",
        raw_memories=[],
        doc_type="design-doc",
        project_key="ENG",
        repo_identifier=None,
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
    *,
    primary_excerpt: str | None = "Decision: retain A7",
    primary_provenance: EvidenceContentProvenance | None = None,
    source_type: str = "jira",
    access_context_hash: str = "workspace-eng",
    required_excerpts: tuple[str, ...] = (),
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
        source_type,
        "Decision: retain A7",
        source_updated_at=None,
    )
    primary = first.observations[1]
    required = first.observations[0]
    revisions = {item.observation_id: item for item in first.observation_revisions}
    unit = EvidenceUnit(
        id="eu-jira-required",
        source_id="src-1",
        doc_id="confluence-123",
        doc_revision_id=first.source_unit_revisions[0].id,
        source_type=source_type,
        source_anchor=primary.id,
        source_lineage_id=first.source_units[0].id,
        project_key="ENG",
        visibility="workspace",
        owner_user_id=None,
        repo_identifier=None,
        content=revisions[primary.id].content,
        excerpt=primary_excerpt,
        evidence_provenance=(
            primary_provenance
            or (
                EvidenceContentProvenance.SOURCE_EXCERPT
                if primary_excerpt
                else EvidenceContentProvenance.NO_EXCERPT
            )
        ),
        access_context_hash=access_context_hash,
    )
    await db.upsert_evidence_unit(unit)
    required_references = (
        tuple(
            EvidenceReference(
                role=EvidenceRole.REQUIRED,
                anchor=SourceAnchor(
                    kind=AnchorKind.REVISION_RANGE,
                    observation_id=required.id,
                    observation_revision_id=revisions[required.id].id,
                    range_start=revisions[required.id].content.index(excerpt),
                    range_end=(
                        revisions[required.id].content.index(excerpt)
                        + len(excerpt)
                    ),
                ),
            )
            for excerpt in required_excerpts
        )
        if required_excerpts
        else (
            EvidenceReference(
                role=EvidenceRole.REQUIRED,
                anchor=SourceAnchor(
                    kind=AnchorKind.WHOLE_OBSERVATION,
                    observation_id=required.id,
                    observation_revision_id=revisions[required.id].id,
                ),
            ),
        )
    )
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
            *required_references,
        ),
    )
    for index, reference in enumerate(references):
        await db.upsert_memory_support_assertion(
            MemorySupportAssertion(
                id=f"support-jira-required-{index}",
                memory_id=incumbent.id,
                evidence_reference_id=reference.id or "",
                source_id="src-1",
                access_context_hash=access_context_hash,
            )
        )
    return incumbent


@pytest.mark.asyncio
async def test_partial_jira_projection_skips_llm_for_proven_disjoint_incumbent(
    db: Database,
    caplog: pytest.LogCaptureFixture,
) -> None:
    await _set_fixture_source_type(db, "jira")
    caplog.set_level("INFO", logger="memforge.memory.engine")
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
        prior_observations={revision.observation_id: revision for revision in first.observation_revisions},
    )
    assert second.coverage.value == "partial_projection"
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=_UnexpectedReconciliationClient(),
    )

    stats = await engine.prepare_and_commit_projected_lifecycle(
        projection=second,
        doc_id="confluence-123",
        raw_memories=[],
        doc_type="ticket",
        project_key="ENG",
        repo_identifier=None,
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
    sample = json.loads(
        next(
            record.getMessage()
            for record in caplog.records
            if '"event":"projected_lifecycle_reconciliation"' in record.getMessage()
        )
    )
    assert sample["reconciliation_incumbent_count"] == 1
    assert sample["reconciliation_model_incumbent_count"] == 0
    assert sample["reconciliation_disjoint_keep_count"] == 1
    assert sample["reconciliation_llm_call_count"] == 0
    assert await db.get_active_memory_support_reference_ids(incumbent.id) == old_support


@pytest.mark.asyncio
async def test_new_candidate_keeps_disjoint_incumbent_in_semantic_reconciliation(
    db: Database,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _set_fixture_source_type(db, "jira")
    caplog.set_level("INFO", logger="memforge.memory.engine")
    first = _jira_projection(
        run_id="projection-jira-new-candidate-1",
        description="Initial issue description.",
        comment_body="Decision: retain A7",
    )
    await db.record_source_projection(first)
    incumbent = await _seed_incumbent_support(
        db,
        projection=first,
        memory_id="mem-jira-new-candidate-incumbent",
        memory_content="Decision: retain A7",
        observation_index=1,
        source_type="jira",
    )
    await db.enable_lifecycle_gate("src-1")
    second = _jira_projection(
        run_id="projection-jira-new-candidate-2",
        description="Payroll validation requires approval before release.",
        comment_body="Decision: retain A7",
        prior=first.source_unit_revisions[0],
        prior_observations={revision.observation_id: revision for revision in first.observation_revisions},
    )
    description = second.observations[0]
    description_revision = next(
        revision for revision in second.observation_revisions if revision.observation_id == description.id
    )
    candidate = RawMemory(
        content="Payroll validation requires approval before release.",
        memory_type="procedure",
        evidence_quote="Payroll validation requires approval before release.",
        source_observation_id=description.id,
    )
    responses = _RecordingAddClient(incumbent.id)
    from memforge.llm.structured import LiteLlmStructuredClient, StructuredLlmConfig

    async def provider(**kwargs):
        prompt = kwargs["messages"][0]["content"]
        response = (
            await responses.classify_memory_relations(prompt)
            if "<memory_pair_groups>" in prompt
            else await responses.audit_incumbent_support(prompt)
        )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=response.model_dump_json()),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )

    monkeypatch.setattr("memforge.llm.structured.litellm.acompletion", provider)
    monkeypatch.setattr("memforge.llm.structured.litellm.supports_response_schema", lambda **_: False)
    client = LiteLlmStructuredClient(
        StructuredLlmConfig(
            model="anthropic/test",
            base_url=None,
            api_key=None,
            timeout_s=1,
            num_retries=0,
        )
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=client,
    )

    async def unexpected_impact_scan(**kwargs):
        del kwargs
        raise AssertionError("new-candidate reconciliation must not pre-scan impacts")

    monkeypatch.setattr(engine, "_projected_incumbent_impacts", unexpected_impact_scan)

    stats = await engine.prepare_and_commit_projected_lifecycle(
        projection=second,
        doc_id="confluence-123",
        raw_memories=[candidate],
        doc_type="ticket",
        project_key="ENG",
        repo_identifier=None,
        document_content=description_revision.content,
        update_mode="diff_guided",
        changed_hunks="Payroll validation requires approval before release.",
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 16, tzinfo=timezone.utc),
    )

    assert stats["added"] == 1
    sample = json.loads(
        next(
            record.getMessage()
            for record in caplog.records
            if '"event":"projected_lifecycle_reconciliation"' in record.getMessage()
        )
    )
    assert sample["reconciliation_incumbent_count"] == 1
    assert sample["reconciliation_model_incumbent_count"] == 1
    assert sample["reconciliation_disjoint_keep_count"] == 0
    assert sample["reconciliation_llm_batch_count"] == 2
    assert sample["reconciliation_llm_call_count"] == 2
    assert len(responses.prompts) == 1
    assert incumbent.content in responses.prompts[0]


@pytest.mark.asyncio
async def test_partial_jira_projection_admits_directly_affected_incumbent_delete(
    db: Database,
) -> None:
    await _set_fixture_source_type(db, "jira")
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
        prior_observations={revision.observation_id: revision for revision in first.observation_revisions},
    )
    assert second.coverage.value == "partial_projection"
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=_DeleteClient(incumbent.id),
    )

    stats = await engine.prepare_and_commit_projected_lifecycle(
        projection=second,
        doc_id="confluence-123",
        raw_memories=[],
        doc_type="ticket",
        project_key="ENG",
        repo_identifier=None,
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
    await _set_fixture_source_type(db, "jira")
    first = _jira_projection(
        run_id="projection-jira-required-1",
        description="A7 applies only to regular payroll.",
    )
    await db.record_source_projection(first)
    incumbent = await _seed_jira_required_incumbent(db, first)
    await db.enable_lifecycle_gate("src-1")
    old_support = await db.get_active_memory_support_reference_ids(incumbent.id)
    revisions = {item.observation_id: item for item in first.observation_revisions}
    second = _jira_projection(
        run_id="projection-jira-required-2",
        description="A7 remains limited to regular payroll runs.",
        prior=first.source_unit_revisions[0],
        prior_observations=revisions,
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=_SupportValidatingNoopClient(
            incumbent.id,
            supported=True,
        ),
    )

    stats = await engine.prepare_and_commit_projected_lifecycle(
        projection=second,
        doc_id="confluence-123",
        raw_memories=[],
        doc_type="ticket",
        project_key="ENG",
        repo_identifier=None,
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
    current_revisions = {item.observation_id: item.id for item in second.observation_revisions}
    assert all(
        item.anchor.observation_revision_id == current_revisions[item.anchor.observation_id] for item in evidence
    )
    current = await db.get_current_source_unit_revision(first.source_units[0].id)
    assert current is not None
    assert current.id == second.source_unit_revisions[0].id
    [plan_row] = await db.db.execute_fetchall(
        "SELECT payload_json FROM lifecycle_plans WHERE source_id = ?",
        ("src-1",),
    )
    plan_payload = json.loads(str(plan_row["payload_json"]))
    attach = next(mutation for mutation in plan_payload["mutations"] if mutation["mutation_type"] == "attach_support")
    assert attach["payload"]["support_validation"]["supported"] is True


@pytest.mark.asyncio
async def test_v2_noop_revalidates_revised_required_jira_description(
    db: Database,
) -> None:
    await _set_fixture_source_type(db, "jira")
    access_context_hash = lifecycle_access_context_hash(
        visibility="workspace",
        owner_user_id=None,
        project_key="ENG",
        repo_identifier=None,
    )
    first = _jira_projection(
        run_id="projection-v2-jira-required-1",
        description="A7 applies only to regular payroll.",
    )
    await db.record_source_projection(first)
    incumbent = await _seed_jira_required_incumbent(
        db,
        first,
        access_context_hash=access_context_hash,
    )
    cutover = await db.report_support_scope_cutover()
    await db.apply_support_scope_v2_cutover(
        expected_report_id=cutover.id,
        owner_id="test-v2-required-rebind",
    )
    await db.enable_lifecycle_gate("src-1")
    old_support = await db.get_active_memory_support_unit_ids(incumbent.id)
    second = _jira_projection(
        run_id="projection-v2-jira-required-2",
        description="A7 remains limited to regular payroll runs.",
        comments_truncated=True,
        prior=first.source_unit_revisions[0],
        prior_observations={
            item.observation_id: item
            for item in first.observation_revisions
        },
    )
    assert second.coverage is ProjectionCoverage.PARTIAL_PROJECTION
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=_SupportValidatingNoopClient(
            incumbent.id,
            supported=True,
            required_evidence_quote=(
                "A7 remains limited to regular payroll runs."
            ),
        ),
    )

    stats = await engine.prepare_and_commit_projected_lifecycle(
        projection=second,
        doc_id="confluence-123",
        raw_memories=[],
        doc_type="ticket",
        project_key="ENG",
        repo_identifier=None,
        document_content="PAY-12",
        update_mode="diff_guided",
        changed_hunks="wording clarified; scope remains regular payroll",
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 15, 10, 36, tzinfo=timezone.utc),
    )

    current_support = await db.get_active_memory_support_unit_ids(incumbent.id)
    assert stats["noop"] == 1
    assert current_support
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
        item.observation_id: item.id
        for item in second.observation_revisions
    }
    assert all(
        item.anchor.observation_revision_id
        == current_revisions[item.anchor.observation_id]
        for item in evidence
    )


@pytest.mark.asyncio
async def test_v2_noop_ambiguous_current_required_fragment_stages_review(
    db: Database,
) -> None:
    await _set_fixture_source_type(db, "jira")
    access_context_hash = lifecycle_access_context_hash(
        visibility="workspace",
        owner_user_id=None,
        project_key="ENG",
        repo_identifier=None,
    )
    first = _jira_projection(
        run_id="projection-v2-jira-ambiguous-required-1",
        description="A7 applies only to regular payroll.",
    )
    await db.record_source_projection(first)
    incumbent = await _seed_jira_required_incumbent(
        db,
        first,
        access_context_hash=access_context_hash,
    )
    cutover = await db.report_support_scope_cutover()
    await db.apply_support_scope_v2_cutover(
        expected_report_id=cutover.id,
        owner_id="test-v2-ambiguous-required",
    )
    await db.enable_lifecycle_gate("src-1")
    old_support = await db.get_active_memory_support_unit_ids(incumbent.id)
    second = _jira_projection(
        run_id="projection-v2-jira-ambiguous-required-2",
        description="Payroll",
        prior=first.source_unit_revisions[0],
        prior_observations={
            item.observation_id: item
            for item in first.observation_revisions
        },
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=_SupportValidatingNoopClient(
            incumbent.id,
            supported=True,
            required_evidence_quote="Payroll",
        ),
    )

    stats = await engine.prepare_and_commit_projected_lifecycle(
        projection=second,
        doc_id="confluence-123",
        raw_memories=[],
        doc_type="ticket",
        project_key="ENG",
        repo_identifier=None,
        document_content="PAY-12",
        update_mode="diff_guided",
        changed_hunks="description now duplicates the summary Fragment",
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 15, 10, 36, tzinfo=timezone.utc),
    )

    assert stats["pending_review"] == 1
    assert await db.get_active_memory_support_unit_ids(incumbent.id) == old_support
    [review] = await db.list_lifecycle_reviews("src-1")
    assert review.status is LifecycleReviewStatus.PENDING
    assert review.reason.endswith(": ambiguous")


@pytest.mark.asyncio
async def test_v2_noop_unpresentable_current_fragment_stages_review(
    db: Database,
) -> None:
    access_context_hash = lifecycle_access_context_hash(
        visibility="workspace",
        owner_user_id=None,
        project_key="ENG",
        repo_identifier=None,
    )
    first = _projection(
        run_id="projection-v2-unpresentable-1",
        body="A7 is removed.",
    )
    await db.record_source_projection(first)
    incumbent = await _seed_incumbent_support(
        db,
        projection=first,
        access_context_hash=access_context_hash,
    )
    cutover = await db.report_support_scope_cutover()
    await db.apply_support_scope_v2_cutover(
        expected_report_id=cutover.id,
        owner_id="test-v2-unpresentable",
    )
    await db.enable_lifecycle_gate("src-1")
    old_support = await db.get_active_memory_support_unit_ids(incumbent.id)
    second = _projection(
        run_id="projection-v2-unpresentable-2",
        body="<!-- no selectable current claim -->",
        prior=first.source_unit_revisions[0],
        prior_observations={
            first.observations[0].id: first.observation_revisions[0]
        },
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=_SupportValidatingNoopClient(
            incumbent.id,
            supported=True,
            evidence_quote="<!-- no selectable current claim -->",
        ),
    )

    stats = await engine.prepare_and_commit_projected_lifecycle(
        projection=second,
        doc_id="confluence-123",
        raw_memories=[],
        doc_type="design-doc",
        project_key="ENG",
        repo_identifier=None,
        document_content=second.observation_revisions[0].content,
        update_mode="diff_guided",
        changed_hunks="supporting claim removed from selectable content",
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 15, 10, 36, tzinfo=timezone.utc),
    )

    assert stats["pending_review"] == 1
    assert await db.get_active_memory_support_unit_ids(incumbent.id) == old_support
    [review] = await db.list_lifecycle_reviews("src-1")
    assert review.status is LifecycleReviewStatus.PENDING
    assert review.reason.endswith(": unpresentable")


@pytest.mark.asyncio
async def test_v2_pending_review_ignores_unrelated_stale_cross_unit_support(
    db: Database,
) -> None:
    scenario = await _seed_v2_stale_cross_unit_scenario(
        db,
        prefix="projection-v2-causal-review",
    )
    second = _projection(
        run_id="projection-v2-causal-review-2",
        body="<!-- no selectable current claim -->",
        prior=scenario.first.source_unit_revisions[0],
        prior_observations={
            scenario.first.observations[0].id:
                scenario.first.observation_revisions[0]
        },
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=_SupportValidatingNoopClient(
            scenario.incumbent.id,
            supported=True,
            evidence_quote="<!-- no selectable current claim -->",
        ),
    )

    stats = await engine.prepare_and_commit_projected_lifecycle(
        projection=second,
        doc_id="confluence-123",
        raw_memories=[],
        doc_type="design-doc",
        project_key="ENG",
        repo_identifier=None,
        document_content=second.observation_revisions[0].content,
        update_mode="diff_guided",
        changed_hunks="supporting claim removed from selectable content",
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 15, 10, 36, tzinfo=timezone.utc),
    )

    assert stats["pending_review"] == 1
    assert (
        scenario.alternative_unit_id
        in await db.get_active_memory_support_unit_ids(scenario.incumbent.id)
    )
    current = await db.get_current_source_unit_revision(
        scenario.first.source_units[0].id
    )
    assert current is not None
    assert current.id == second.source_unit_revisions[0].id
    [review] = await db.list_lifecycle_reviews("src-1")
    assert review.status is LifecycleReviewStatus.PENDING
    assert review.reason.endswith(": unpresentable")


@pytest.mark.asyncio
async def test_v2_noop_rebind_ignores_unrelated_stale_cross_unit_support(
    db: Database,
) -> None:
    scenario = await _seed_v2_stale_cross_unit_scenario(
        db,
        prefix="projection-v2-causal-rebind",
    )
    old_scope_support = set(
        await db.get_source_unit_support_unit_ids(
            scenario.first.source_units[0].id
        )
    )
    second = _projection(
        run_id="projection-v2-causal-rebind-2",
        body="A7 remains excluded.",
        prior=scenario.first.source_unit_revisions[0],
        prior_observations={
            scenario.first.observations[0].id:
                scenario.first.observation_revisions[0]
        },
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=_SupportValidatingNoopClient(
            scenario.incumbent.id,
            supported=True,
            evidence_quote="A7 remains excluded.",
        ),
    )

    stats = await engine.prepare_and_commit_projected_lifecycle(
        projection=second,
        doc_id="confluence-123",
        raw_memories=[],
        doc_type="design-doc",
        project_key="ENG",
        repo_identifier=None,
        document_content=second.observation_revisions[0].content,
        update_mode="diff_guided",
        changed_hunks="A7 is removed. -> A7 remains excluded.",
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 16, 10, 36, tzinfo=timezone.utc),
    )

    current_support = set(
        await db.get_active_memory_support_unit_ids(scenario.incumbent.id)
    )
    assert stats["noop"] == 1
    assert scenario.alternative_unit_id in current_support
    assert current_support.isdisjoint(old_scope_support)
    assert len(current_support) == 2


@pytest.mark.asyncio
async def test_v2_destructive_commit_defers_on_stale_cross_unit_support(
    db: Database,
) -> None:
    scenario = await _seed_v2_stale_cross_unit_scenario(
        db,
        prefix="projection-v2-causal-deferred",
    )
    old_support = await db.get_active_memory_support_unit_ids(
        scenario.incumbent.id
    )
    second = _projection(
        run_id="projection-v2-causal-deferred-2",
        body="",
        prior=scenario.first.source_unit_revisions[0],
        prior_observations={
            scenario.first.observations[0].id:
                scenario.first.observation_revisions[0]
        },
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=None,
    )

    with pytest.raises(ProjectedLifecycleDeferredError) as raised:
        await engine.prepare_and_commit_projected_lifecycle(
            projection=second,
            doc_id="confluence-123",
            raw_memories=[],
            doc_type="design-doc",
            project_key="ENG",
            repo_identifier=None,
            document_content="",
            update_mode="diff_guided",
            changed_hunks="A7 is removed. -> empty",
            update_plan_stats=None,
            source_updated_at=datetime(2026, 7, 16, 10, 36, tzinfo=timezone.utc),
        )

    assert raised.value.blocking_source_unit_ids == (
        scenario.alternative.source_units[0].id,
    )
    assert raised.value.blocking_evidence_unit_ids == (
        scenario.alternative_unit_id,
    )
    assert (
        await db.get_active_memory_support_unit_ids(scenario.incumbent.id)
        == old_support
    )
    current = await db.get_current_source_unit_revision(
        scenario.first.source_units[0].id
    )
    assert current is not None
    assert current.id == scenario.first.source_unit_revisions[0].id


@pytest.mark.asyncio
async def test_v2_deferred_plan_rolls_back_every_memory_in_source_unit(
    db: Database,
) -> None:
    access_context_hash = lifecycle_access_context_hash(
        visibility="workspace",
        owner_user_id=None,
        project_key="ENG",
        repo_identifier=None,
    )
    first = _projection(
        run_id="projection-v2-multi-memory-1",
        body="A7 is removed.\nB8 is removed.",
    )
    await db.record_source_projection(first)
    first_memory = await _seed_incumbent_support(
        db,
        projection=first,
        memory_id="mem-a7",
        memory_content="A7 is removed.",
        access_context_hash=access_context_hash,
    )
    second_memory = await _seed_incumbent_support(
        db,
        projection=first,
        memory_id="mem-b8",
        memory_content="B8 is removed.",
        access_context_hash=access_context_hash,
    )
    alternative = _projection(
        run_id="projection-v2-multi-memory-alternative-1",
        body="A7 is removed.\nB8 is removed.",
        item_id="confluence-456",
        page_id="456",
    )
    alternative_units = {
        await _add_independent_legacy_support_alternative(
            db,
            incumbent=memory,
            projection=alternative,
            doc_id="confluence-456",
            access_context_hash=access_context_hash,
        )
        for memory in (first_memory, second_memory)
    }
    cutover = await db.report_support_scope_cutover()
    await db.apply_support_scope_v2_cutover(
        expected_report_id=cutover.id,
        owner_id="test-v2-multi-memory",
    )
    await db.enable_lifecycle_gate("src-1")
    memory_ids = (first_memory.id, second_memory.id)
    old_support = {
        memory_id: await db.get_active_memory_support_unit_ids(memory_id)
        for memory_id in memory_ids
    }
    alternative_current = _projection(
        run_id="projection-v2-multi-memory-alternative-2",
        body="A7 and B8 are excluded in the current release.",
        item_id="confluence-456",
        page_id="456",
        prior=alternative.source_unit_revisions[0],
        prior_observations={
            alternative.observations[0].id: alternative.observation_revisions[0]
        },
    )
    await db.record_source_projection(alternative_current)
    target = _projection(
        run_id="projection-v2-multi-memory-2",
        body="",
        prior=first.source_unit_revisions[0],
        prior_observations={first.observations[0].id: first.observation_revisions[0]},
    )
    source_support = await db.get_source_unit_support_unit_ids(first.source_units[0].id)
    support_states = await db.get_active_memory_support_states(memory_ids)
    current_memories = {
        memory_id: await db.get_memory(memory_id)
        for memory_id in memory_ids
    }
    assert all(current_memories.values())
    plan = build_lifecycle_plan(
        plan_id=lifecycle_plan_id(
            ReconciliationScope(
                id=f"scope:{target.run_id}",
                source_id="src-1",
                source_unit_id=first.source_units[0].id,
                base_unit_revision_id=first.source_unit_revisions[0].id,
                target_unit_revision_id=target.source_unit_revisions[0].id,
            )
        ),
        scope=ReconciliationScope(
            id=f"scope:{target.run_id}",
            source_id="src-1",
            source_unit_id=first.source_units[0].id,
            base_unit_revision_id=first.source_unit_revisions[0].id,
            target_unit_revision_id=target.source_unit_revisions[0].id,
        ),
        gate_state=LifecycleGateState.ENABLED,
        operations=tuple(
            ReconcileOperation(
                action=ReconcileAction.DELETE,
                memory_id=memory_id,
                reason="current Source Unit is empty",
            )
            for memory_id in memory_ids
        ),
        incumbents={
            memory_id: current_memories[memory_id]  # type: ignore[dict-item]
            for memory_id in memory_ids
        },
        source_support_reference_ids=source_support,
        all_active_support_reference_ids={
            memory_id: support_states[memory_id].support_ids
            for memory_id in memory_ids
        },
        support_set_hashes={
            memory_id: support_states[memory_id].support_set_hash
            for memory_id in memory_ids
        },
        observation_revision_ids=(),
        new_evidence_reference_ids=(),
        support_scope_version=SupportScopeVersion.EVIDENCE_UNIT_SET_V2,
        source_support_unit_ids=source_support,
        all_active_support_unit_ids={
            memory_id: support_states[memory_id].support_ids
            for memory_id in memory_ids
        },
        defaults=NewMemoryDefaults(
            visibility="workspace",
            owner_user_id=None,
            project_key="ENG",
            repo_identifier=None,
            doc_id="confluence-123",
            source_type="confluence",
            access_context_hash=access_context_hash,
        ),
    )

    with pytest.raises(ProjectedLifecycleDeferredError) as raised:
        await db.apply_source_projection_lifecycle(target, plan)

    assert raised.value.blocking_source_unit_ids == (
        alternative.source_units[0].id,
    )
    assert set(raised.value.blocking_evidence_unit_ids) == alternative_units
    assert {
        memory_id: await db.get_active_memory_support_unit_ids(memory_id)
        for memory_id in memory_ids
    } == old_support
    current_rows = [await db.get_memory(memory_id) for memory_id in memory_ids]
    assert all(memory is not None and memory.status == "active" for memory in current_rows)


@pytest.mark.asyncio
async def test_v2_deferred_commit_rematerializes_without_semantic_replay(
    db: Database,
) -> None:
    scenario = await _seed_v2_stale_cross_unit_scenario(
        db,
        prefix="projection-v2-prepared",
    )
    second = _projection(
        run_id="projection-v2-prepared-2",
        body="",
        prior=scenario.first.source_unit_revisions[0],
        prior_observations={
            scenario.first.observations[0].id:
                scenario.first.observation_revisions[0]
        },
    )
    adapters = build_sqlite_adapters(db, object())
    client = _SupportValidatingNoopClient(
        scenario.incumbent.id,
        supported=True,
        evidence_quote="A7 is excluded in the current release.",
    )
    engine = MemoryEngine(
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=client,
    )

    with pytest.raises(SourceUnitLifecycleDeferred) as raised:
        await engine.prepare_and_commit_projected_lifecycle(
            projection=second,
            doc_id="confluence-123",
            raw_memories=[],
            doc_type="design-doc",
            project_key="ENG",
            repo_identifier=None,
            document_content="",
            update_mode="diff_guided",
            changed_hunks="A7 is removed. -> empty",
            update_plan_stats=None,
            source_updated_at=datetime(2026, 7, 16, 10, 36, tzinfo=timezone.utc),
            lifecycle_execution_owner_id="sync-prepared:lease-1",
        )
    assert isinstance(raised.value.handle, DeferredProjectedLifecycleHandle)
    assert not hasattr(raised.value, "prepared_commit")
    with pytest.raises(SourceUnitLifecycleExecutionError) as ineligible:
        await engine.retry_deferred_projected_lifecycle(
            raised.value.handle,
            eligible_same_run_source_unit_ids=set(),
        )
    assert ineligible.value.commit_attempted is False

    await db.db.execute(
        "UPDATE memories SET confidence = ? WHERE id = ?",
        (0.1, scenario.incumbent.id),
    )
    await db.db.commit()
    with pytest.raises(
        ProjectedSupportInvariantError,
        match="Memory changed before commit",
    ):
        await engine.retry_deferred_projected_lifecycle(
            raised.value.handle,
            eligible_same_run_source_unit_ids={scenario.alternative.source_units[0].id},
        )
    await db.db.execute(
        "UPDATE memories SET confidence = ? WHERE id = ?",
        (scenario.incumbent.confidence, scenario.incumbent.id),
    )
    await db.db.commit()
    await db.db.execute(
        "UPDATE source_lifecycle_gates SET state = 'gated' WHERE source_id = ?",
        ("src-1",),
    )
    await db.db.commit()
    with pytest.raises(
        ProjectedSupportInvariantError,
        match="gate changed before commit",
    ):
        await engine.retry_deferred_projected_lifecycle(
            raised.value.handle,
            eligible_same_run_source_unit_ids={scenario.alternative.source_units[0].id},
        )
    await db.db.execute(
        "UPDATE source_lifecycle_gates SET state = 'enabled' WHERE source_id = ?",
        ("src-1",),
    )
    await db.db.execute(
        "UPDATE sources SET access_policy = 'private' WHERE id = ?",
        ("src-1",),
    )
    await db.db.commit()
    with pytest.raises(
        ProjectedSupportInvariantError,
        match="access context changed before commit",
    ):
        await engine.retry_deferred_projected_lifecycle(
            raised.value.handle,
            eligible_same_run_source_unit_ids={scenario.alternative.source_units[0].id},
        )
    await db.db.execute(
        "UPDATE sources SET access_policy = 'workspace' WHERE id = ?",
        ("src-1",),
    )
    await db.db.commit()

    await engine.prepare_and_commit_projected_lifecycle(
        projection=scenario.alternative_current,
        doc_id="confluence-456",
        raw_memories=[],
        doc_type="design-doc",
        project_key="ENG",
        repo_identifier=None,
        document_content=(
            scenario.alternative_current.observation_revisions[0].content
        ),
        update_mode="diff_guided",
        changed_hunks="A7 is removed. -> A7 is excluded in the current release.",
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 16, 10, 37, tzinfo=timezone.utc),
    )
    semantic_calls = client.validation_calls

    stats = await engine.retry_deferred_projected_lifecycle(
        raised.value.handle,
        eligible_same_run_source_unit_ids={scenario.alternative.source_units[0].id},
    )

    assert client.validation_calls == semantic_calls
    assert stats["deleted"] == 0
    current_support = await db.get_active_memory_support_unit_ids(
        scenario.incumbent.id
    )
    assert scenario.alternative_unit_id not in current_support
    assert len(current_support) == 1
    current_memory = await db.get_memory(scenario.incumbent.id)
    assert current_memory is not None
    assert current_memory.status == "active"

    replayed = await engine.retry_deferred_projected_lifecycle(
        raised.value.handle,
        eligible_same_run_source_unit_ids={scenario.alternative.source_units[0].id},
    )
    assert replayed == stats
    assert client.validation_calls == semantic_calls
    assert (
        await db.get_active_memory_support_unit_ids(scenario.incumbent.id)
        == current_support
    )


@pytest.mark.asyncio
async def test_v2_prepared_commit_rejects_undeclared_support_drift(
    db: Database,
) -> None:
    scenario = await _seed_v2_stale_cross_unit_scenario(
        db,
        prefix="projection-v2-prepared-drift",
    )
    target = _projection(
        run_id="projection-v2-prepared-drift-2",
        body="",
        prior=scenario.first.source_unit_revisions[0],
        prior_observations={
            scenario.first.observations[0].id:
                scenario.first.observation_revisions[0]
        },
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=None,
    )
    with pytest.raises(SourceUnitLifecycleDeferred) as raised:
        await engine.prepare_and_commit_projected_lifecycle(
            projection=target,
            doc_id="confluence-123",
            raw_memories=[],
            doc_type="design-doc",
            project_key="ENG",
            repo_identifier=None,
            document_content="",
            update_mode="diff_guided",
            changed_hunks="A7 is removed. -> empty",
            update_plan_stats=None,
            source_updated_at=datetime(2026, 7, 16, 10, 36, tzinfo=timezone.utc),
            lifecycle_execution_owner_id="sync-prepared-drift:lease-1",
        )

    await db.db.execute(
        """UPDATE memory_unit_support_assertions
              SET active = 0, removed_at = ?
            WHERE memory_id = ? AND evidence_unit_id = ?""",
        (
            datetime(2026, 7, 16, 10, 37, tzinfo=timezone.utc).isoformat(),
            scenario.incumbent.id,
            f"eu-{scenario.incumbent.id}",
        ),
    )
    await db.db.commit()

    with pytest.raises(
        ProjectedSupportInvariantError,
        match="Support topology changed outside declared blockers",
    ):
        await engine.retry_deferred_projected_lifecycle(
            raised.value.handle,
            eligible_same_run_source_unit_ids={scenario.alternative.source_units[0].id},
        )


@pytest.mark.asyncio
async def test_v2_noop_preserves_multiple_required_parts_in_one_observation(
    db: Database,
) -> None:
    await _set_fixture_source_type(db, "jira")
    access_context_hash = lifecycle_access_context_hash(
        visibility="workspace",
        owner_user_id=None,
        project_key="ENG",
        repo_identifier=None,
    )
    first = _jira_projection(
        run_id="projection-v2-jira-multi-required-1",
        description="A7 applies only to regular payroll.",
    )
    await db.record_source_projection(first)
    incumbent = await _seed_jira_required_incumbent(
        db,
        first,
        access_context_hash=access_context_hash,
        required_excerpts=(
            "A7 applies only to regular payroll.",
            "Payroll",
        ),
    )
    cutover = await db.report_support_scope_cutover()
    await db.apply_support_scope_v2_cutover(
        expected_report_id=cutover.id,
        owner_id="test-v2-multi-required-rebind",
    )
    await db.enable_lifecycle_gate("src-1")
    old_support = await db.get_active_memory_support_unit_ids(incumbent.id)
    second = _jira_projection(
        run_id="projection-v2-jira-multi-required-2",
        description="A7 remains limited to regular payroll runs.",
        prior=first.source_unit_revisions[0],
        prior_observations={
            item.observation_id: item
            for item in first.observation_revisions
        },
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=_SupportValidatingNoopClient(
            incumbent.id,
            supported=True,
            required_evidence_quotes=(
                "A7 remains limited to regular payroll runs.",
                "Payroll",
            ),
        ),
    )

    stats = await engine.prepare_and_commit_projected_lifecycle(
        projection=second,
        doc_id="confluence-123",
        raw_memories=[],
        doc_type="ticket",
        project_key="ENG",
        repo_identifier=None,
        document_content="PAY-12",
        update_mode="diff_guided",
        changed_hunks="description wording clarified",
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 15, 10, 36, tzinfo=timezone.utc),
    )

    current_support = await db.get_active_memory_support_unit_ids(incumbent.id)
    assert stats["noop"] == 1
    assert set(current_support).isdisjoint(old_support)
    evidence = await db.get_active_memory_support_evidence(
        incumbent.id,
        source_id="src-1",
    )
    assert [item.role for item in evidence].count(EvidenceRole.PRIMARY) == 1
    required = [
        item for item in evidence if item.role is EvidenceRole.REQUIRED
    ]
    assert len(required) == 2
    assert len({item.reference_id for item in required}) == 2
    assert len({item.anchor.observation_id for item in required}) == 1


@pytest.mark.asyncio
async def test_v2_noop_resolves_decoded_canonical_quotes_to_raw_json_ranges(
    db: Database,
) -> None:
    await _set_fixture_source_type(db, "jira")
    access_context_hash = lifecycle_access_context_hash(
        visibility="workspace",
        owner_user_id=None,
        project_key="ENG",
        repo_identifier=None,
    )
    old_primary = 'Decision "A7"\n保留。'
    old_required = 'Policy "A7"\n适用于常规工资。'
    first = _jira_projection(
        run_id="projection-v2-jira-escaped-1",
        description=old_required,
        comment_body=old_primary,
    )
    await db.record_source_projection(first)
    incumbent = await _seed_jira_required_incumbent(
        db,
        first,
        primary_excerpt=old_primary,
        access_context_hash=access_context_hash,
    )
    cutover = await db.report_support_scope_cutover()
    await db.apply_support_scope_v2_cutover(
        expected_report_id=cutover.id,
        owner_id="test-v2-escaped-rebind",
    )
    await db.enable_lifecycle_gate("src-1")
    old_support = await db.get_active_memory_support_unit_ids(incumbent.id)
    new_primary = 'Decision "A7"\n仍然保留。'
    new_required = 'Policy "A7"\n仅适用于常规工资。'
    second = _jira_projection(
        run_id="projection-v2-jira-escaped-2",
        description=new_required,
        comment_body=new_primary,
        prior=first.source_unit_revisions[0],
        prior_observations={
            item.observation_id: item
            for item in first.observation_revisions
        },
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=_SupportValidatingNoopClient(
            incumbent.id,
            supported=True,
            evidence_quote=new_primary,
            required_evidence_quote=new_required,
        ),
    )

    stats = await engine.prepare_and_commit_projected_lifecycle(
        projection=second,
        doc_id="confluence-123",
        raw_memories=[],
        doc_type="ticket",
        project_key="ENG",
        repo_identifier=None,
        document_content="PAY-12",
        update_mode="diff_guided",
        changed_hunks="quoted multilingual Jira fields changed",
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 15, 10, 36, tzinfo=timezone.utc),
    )

    current_support = await db.get_active_memory_support_unit_ids(incumbent.id)
    assert stats["noop"] == 1
    assert set(current_support).isdisjoint(old_support)
    evidence = await db.get_active_memory_support_evidence(
        incumbent.id,
        source_id="src-1",
    )
    assert {item.excerpt for item in evidence} == {
        new_primary,
        new_required,
    }
    current_revisions = {
        item.observation_id: item
        for item in second.observation_revisions
    }
    for item in evidence:
        revision = current_revisions[item.anchor.observation_id]
        assert item.anchor.kind is AnchorKind.REVISION_RANGE
        raw_slice = revision.content[
            item.anchor.range_start : item.anchor.range_end
        ]
        assert "\\n" in raw_slice
        assert '\\"A7\\"' in raw_slice


@pytest.mark.asyncio
async def test_v2_noop_propagates_representation_compiler_contract_failure(
    db: Database,
) -> None:
    access_context_hash = lifecycle_access_context_hash(
        visibility="workspace",
        owner_user_id=None,
        project_key="ENG",
        repo_identifier=None,
    )
    first = _projection(
        run_id="projection-v2-compiler-contract-1",
        body="A7 is removed.\nOld deployment note.",
    )
    await db.record_source_projection(first)
    incumbent = await _seed_incumbent_support(
        db,
        projection=first,
        access_context_hash=access_context_hash,
    )
    cutover = await db.report_support_scope_cutover()
    await db.apply_support_scope_v2_cutover(
        expected_report_id=cutover.id,
        owner_id="test-v2-compiler-contract",
    )
    await db.enable_lifecycle_gate("src-1")
    second = _projection(
        run_id="projection-v2-compiler-contract-2",
        body="A7 is removed.\nNew deployment note.",
        prior=first.source_unit_revisions[0],
        prior_observations={
            first.observations[0].id: first.observation_revisions[0]
        },
    )
    second = replace(
        second,
        observation_revisions=tuple(
            replace(
                revision,
                evidence_profile=EvidenceRepresentationProfile(
                    name="unsupported-test-profile",
                    version=1,
                    coordinate_space=EvidenceCoordinateSpace.UNICODE_SCALAR,
                ),
            )
            for revision in second.observation_revisions
        ),
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=_SupportValidatingNoopClient(
            incumbent.id,
            supported=True,
            evidence_quote="A7 is removed.",
        ),
    )

    with pytest.raises(RuntimeError, match="revalidation compiler contract"):
        await engine.prepare_and_commit_projected_lifecycle(
            projection=second,
            doc_id="confluence-123",
            raw_memories=[],
            doc_type="design-doc",
            project_key="ENG",
            repo_identifier=None,
            document_content=second.observation_revisions[0].content,
            update_mode="diff_guided",
            changed_hunks="representation contract changed",
            update_plan_stats=None,
            source_updated_at=datetime(
                2026,
                7,
                15,
                10,
                36,
                tzinfo=timezone.utc,
            ),
        )

    assert await db.get_active_memory_support_unit_ids(incumbent.id)
    assert await db.list_lifecycle_reviews("src-1") == []


@pytest.mark.asyncio
async def test_noop_revalidates_revised_required_with_artifact_primary(
    db: Database,
) -> None:
    first = _projection_with_artifact(
        run_id="projection-artifact-primary-1",
        payload=b"stable-diagram",
        provider_revision="1",
        inference_eligible=True,
        body="# Page\nA7 applies only to regular payroll.",
    )
    await db.record_source_projection(first)
    incumbent = await _seed_jira_required_incumbent(
        db,
        first,
        primary_excerpt=None,
        primary_provenance=EvidenceContentProvenance.SOURCE_ARTIFACT,
        source_type="confluence",
    )
    await db.enable_lifecycle_gate("src-1")
    second = _projection_with_artifact(
        run_id="projection-artifact-primary-2",
        payload=b"stable-diagram",
        provider_revision="1",
        inference_eligible=True,
        body="# Page\nA7 remains limited to regular payroll runs.",
        prior=first.source_unit_revisions[0],
        prior_observations={
            revision.observation_id: revision
            for revision in first.observation_revisions
        },
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=_SupportValidatingNoopClient(
            incumbent.id,
            supported=True,
        ),
    )

    stats = await engine.prepare_and_commit_projected_lifecycle(
        projection=second,
        doc_id="confluence-123",
        raw_memories=[],
        doc_type="design-doc",
        project_key="ENG",
        repo_identifier=None,
        document_content="# Page\nA7 remains limited to regular payroll runs.",
        update_mode="diff_guided",
        changed_hunks="wording clarified; scope remains regular payroll",
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 15, 10, 36, tzinfo=timezone.utc),
    )

    assert stats["noop"] == 1
    evidence = await db.get_active_memory_support_evidence(
        incumbent.id,
        source_id="src-1",
    )
    [primary] = [item for item in evidence if item.role is EvidenceRole.PRIMARY]
    assert primary.excerpt is None
    assert primary.anchor.kind is AnchorKind.WHOLE_OBSERVATION
    unit = await db.get_evidence_unit(primary.evidence_unit_id)
    assert unit is not None
    assert unit.evidence_provenance is EvidenceContentProvenance.SOURCE_ARTIFACT


@pytest.mark.asyncio
async def test_noop_with_invalidated_required_evidence_creates_review(db: Database) -> None:
    await _set_fixture_source_type(db, "jira")
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
        prior_observations={revision.observation_id: revision for revision in first.observation_revisions},
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=_SupportValidatingNoopClient(
            incumbent.id,
            supported=False,
        ),
    )

    stats = await engine.prepare_and_commit_projected_lifecycle(
        projection=second,
        doc_id="confluence-123",
        raw_memories=[],
        doc_type="ticket",
        project_key="ENG",
        repo_identifier=None,
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
async def test_lifecycle_vector_retry_respects_durable_backoff_and_completion_is_idempotent(
    db: Database,
) -> None:
    projection = _projection(run_id="projection-vector-backoff", body="A7 is removed.")
    await db.record_source_projection(projection)
    await _seed_incumbent_support(db, projection=projection)
    await db.rebaseline_source_lifecycle("src-1")
    [task] = await db.list_lifecycle_vector_tasks(source_id="src-1")

    retry_at = "2026-07-30T10:01:00+00:00"
    await db.fail_lifecycle_vector_task(
        task.id,
        "temporary vector failure",
        next_attempt_at=retry_at,
    )

    assert (
        await db.list_ready_lifecycle_vector_tasks(
            source_id="src-1",
            max_attempts=5,
            now="2026-07-30T10:00:59+00:00",
        )
        == []
    )
    [ready] = await db.list_ready_lifecycle_vector_tasks(
        source_id="src-1",
        max_attempts=5,
        now=retry_at,
    )
    assert ready.id == task.id

    await db.complete_lifecycle_vector_task(task.id)
    await db.complete_lifecycle_vector_task(task.id)
    assert await db.list_lifecycle_vector_tasks(source_id="src-1") == []


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
        return LifecycleVectorDeliveryResult(state=LifecycleVectorDeliveryState.DELIVERED)

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
        support_set_hashes={incumbent.id: await db.get_memory_support_set_hash(incumbent.id)},
        observation_revision_ids=tuple(revision.id for revision in second.observation_revisions),
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
    client = _SemanticEquivalentClient()
    engine = MemoryEngine(
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_EquivalentMemoryStore(db, incumbent),
        structured_llm_client=client,
    )

    stats = await engine.prepare_and_commit_projected_lifecycle(
        projection=second,
        doc_id="confluence-456",
        raw_memories=[raw],
        doc_type="design-doc",
        project_key=None,
        repo_identifier=None,
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
    attach = next(mutation for mutation in plan_payload["mutations"] if mutation["mutation_type"] == "attach_support")
    assert attach["payload"]["equivalence_proof"] == {
        "candidate_content_hash": content_hash("A7 remains excluded."),
        "incumbent_content_hash": content_hash("A7 is removed."),
        "method": "structured_relation_classifier",
        "model": engine.llm_model,
        "reason": "Both claims state that A7 is excluded.",
    }
    support = await db.get_active_memory_support_evidence(
        incumbent.id,
        source_id="src-2",
    )
    assert len(support) == 1
    assert support[0].anchor.observation_revision_id == second.observation_revisions[0].id
    assert client.relation_calls == 1


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
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_EquivalentMemoryStore(db, incumbent),
        structured_llm_client=_SemanticEquivalentClient(),
    )

    stats = await engine.prepare_and_commit_projected_lifecycle(
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
        document_content="A7 remains excluded.",
        update_mode="full_document",
        changed_hunks=None,
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 15, 11, 0, tzinfo=timezone.utc),
    )

    assert stats["added"] == 0
    assert stats["corroborated"] == 1
    assert len(await db.list_memories(source="src-1", status="active")) == 1
    assert {(source.source_id, source.doc_id) for source in await db.get_memory_sources(incumbent.id)} == {
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
        cross_document_candidates=_candidate_retriever(adapters),
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
    first_stats = await engine.prepare_and_commit_projected_lifecycle(
        projection=first,
        doc_id="confluence-123",
        raw_memories=[first_raw],
        doc_type="design-doc",
        project_key="ENG",
        repo_identifier=None,
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

    second_stats = await engine.prepare_and_commit_projected_lifecycle(
        projection=second,
        doc_id="confluence-456",
        raw_memories=[second_raw],
        doc_type="design-doc",
        project_key="ENG",
        repo_identifier=None,
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
    attach = next(mutation for mutation in payload["mutations"] if mutation["mutation_type"] == "attach_support")
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
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=None,
    )

    stats = await engine.prepare_and_commit_projected_lifecycle(
        projection=second,
        doc_id="confluence-456",
        raw_memories=[raw],
        doc_type="design-doc",
        project_key="ENG",
        repo_identifier=None,
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
    assert {item.anchor.observation_revision_id for item in support} == {
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
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=None,
    )

    stats = await engine.prepare_and_commit_projected_lifecycle(
        projection=projection,
        doc_id="private-confluence-123",
        raw_memories=[raw],
        doc_type="design-doc",
        project_key="ENG",
        repo_identifier="repo-a",
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
            observation_revision_ids=tuple(revision.id for revision in projection.observation_revisions),
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
async def test_projected_memory_support_survives_relation_work_retry_and_empty_completion(
    db: Database,
) -> None:
    await _set_fixture_source_type(db, "jira")
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
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=_SemanticEquivalentClient(),
    )

    stats = await engine.prepare_and_commit_projected_lifecycle(
        projection=projection,
        doc_id="confluence-123",
        raw_memories=[raw],
        doc_type="ticket",
        project_key="ENG",
        repo_identifier=None,
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
    assert await db.has_ready_relation_discovery_work(max_attempts=5) is True
    [work] = await db.lease_relation_discovery_work(
        worker_id="relation-worker-a",
        limit=10,
        lease_seconds=60,
        max_attempts=3,
    )
    assert work.request.memory_id == memory.id
    assert work.request.expected_content_hash == memory.content_hash
    assert work.request.source_id == "src-1"
    assert work.request.source_unit_id == projection.source_units[0].id
    assert work.request.source_unit_revision_id == projection.source_unit_revisions[0].id
    assert work.attempts == 1
    assert work.lease_token
    assert await db.has_ready_relation_discovery_work(max_attempts=5) is False

    with pytest.raises(ValueError, match="lease was lost"):
        await db.obsolete_relation_discovery_work(
            work.request.id,
            worker_id="relation-worker-b",
            lease_token=work.lease_token or "",
            reason="wrong owner",
        )

    await db.fail_relation_discovery_work(
        work.request.id,
        worker_id="relation-worker-a",
        lease_token=work.lease_token or "",
        error="transient classifier failure",
        next_attempt_at="2999-01-01T00:00:00+00:00",
        exhausted=False,
    )
    assert await db.has_ready_relation_discovery_work(max_attempts=5) is False
    assert (
        await db.lease_relation_discovery_work(
            worker_id="relation-worker-c",
            limit=1,
            lease_seconds=60,
            max_attempts=3,
        )
        == []
    )
    await db.db.execute(
        "UPDATE relation_discovery_work SET next_attempt_at = ? WHERE id = ?",
        ("2000-01-01T00:00:00+00:00", work.request.id),
    )
    await db.db.commit()
    assert await db.has_ready_relation_discovery_work(max_attempts=5) is True
    [retried] = await db.lease_relation_discovery_work(
        worker_id="relation-worker-c",
        limit=1,
        lease_seconds=60,
        max_attempts=3,
    )
    assert retried.attempts == 2
    assert retried.error == "transient classifier failure"
    await db.db.execute(
        "UPDATE relation_discovery_work SET lease_until = ? WHERE id = ?",
        ("2000-01-01T00:00:00+00:00", retried.request.id),
    )
    await db.db.commit()
    with pytest.raises(ValueError, match="lease was lost"):
        await db.obsolete_relation_discovery_work(
            retried.request.id,
            worker_id="relation-worker-c",
            lease_token=retried.lease_token or "",
            reason="expired worker must not finish",
        )
    await db.db.rollback()

    [completion] = await db.lease_relation_discovery_work(
        worker_id="relation-worker-d",
        limit=1,
        lease_seconds=60,
        max_attempts=4,
    )
    evidence_unit = await db.get_current_relation_evidence_unit(
        memory.id,
        source_id=completion.request.source_id,
        source_unit_id=completion.request.source_unit_id,
    )
    assert evidence_unit is not None
    support_run = RelationRunRecord(
        id="relation-run-authoritative-support",
        evidence_unit_id=evidence_unit.id,
        access_context_hash=evidence_unit.access_context_hash,
        candidate_count=0,
        mandatory_candidate_count=0,
        checked_candidate_count=0,
        incomplete_mandatory_buckets=(),
        classifier_version="source-projection-test-v1",
        lifecycle_action=LifecycleAction.CREATE_MEMORY,
        review_case=None,
        status="applied",
        result_memory_id=memory.id,
        audit={"source": "source_projection"},
    )
    await db.record_relation_outcome_bundle(
        RelationOutcomeBundle(
            evidence_unit=evidence_unit,
            relation_run=support_run,
            relations=(
                EvidenceRelationRecord(
                    evidence_unit_id=evidence_unit.id,
                    memory_id=memory.id,
                    relation_type=RelationType.SUPPORTS,
                    authority_case=AuthorityCase.SAME_SOURCE_LINEAGE,
                    is_authoritative_support=True,
                    source_lineage_id=evidence_unit.source_lineage_id,
                    confidence=1.0,
                    classifier_version="source-projection-test-v1",
                    relation_run_id=support_run.id,
                ),
            ),
        )
    )
    authoritative_relations = await db.get_evidence_relations(evidence_unit.id)
    assert len(authoritative_relations) == 1
    assert authoritative_relations[0].is_authoritative_support is True

    completed_at = datetime(2026, 7, 15, 11, 1, tzinfo=timezone.utc).isoformat()
    await db.complete_relation_discovery_work(
        completion.request.id,
        worker_id="relation-worker-d",
        lease_token=completion.lease_token or "",
        relation_outcome=RelationOutcomeBundle(
            evidence_unit=evidence_unit,
            relation_run=RelationRunRecord(
                id="relation-run-empty-discovery",
                evidence_unit_id=evidence_unit.id,
                access_context_hash=evidence_unit.access_context_hash,
                candidate_count=0,
                mandatory_candidate_count=0,
                checked_candidate_count=0,
                incomplete_mandatory_buckets=(),
                classifier_version="relation-discovery-test-v1",
                lifecycle_action=LifecycleAction.NONE,
                review_case=None,
                status="checked",
                result_memory_id=memory.id,
                audit={"source": "relation_discovery"},
                started_at=completed_at,
                completed_at=completed_at,
            ),
        ),
    )

    assert await db.get_evidence_relations(evidence_unit.id) == authoritative_relations
    async with db.db.execute(
        "SELECT status, error FROM relation_discovery_work WHERE id = ?",
        (completion.request.id,),
    ) as cursor:
        completed_work = await cursor.fetchone()
    assert completed_work is not None
    assert completed_work["status"] == "completed"
    assert completed_work["error"] is None


class _DeterministicRefinementClassifier:
    def plan(self, pairs):
        return MemoryPairClassificationPlan(
            pair_count=len(pairs),
            llm_calls=1 if pairs else 0,
            prompt_chars=123 if pairs else 0,
        )

    async def classify(self, pairs):
        return MemoryPairClassification(
            decisions=tuple(
                MemoryPairDecision(
                    pair=pair,
                    relation_type=MemoryRelationType.REFINES,
                    direction=RelationDirection.CHALLENGER_TO_CANDIDATE,
                    reason="deterministic contract fixture",
                )
                for pair in pairs
            ),
            llm_calls=1 if pairs else 0,
            prompt_chars=123 if pairs else 0,
        )


class _DeterministicContradictionClassifier:
    def plan(self, pairs):
        return MemoryPairClassificationPlan(
            pair_count=len(pairs),
            llm_calls=1 if pairs else 0,
            prompt_chars=123 if pairs else 0,
        )

    async def classify(self, pairs):
        return MemoryPairClassification(
            decisions=tuple(
                MemoryPairDecision(
                    pair=pair,
                    relation_type=MemoryRelationType.CONTRADICTS,
                    direction=RelationDirection.SYMMETRIC,
                    reason="deterministic contract fixture",
                )
                for pair in pairs
            ),
            llm_calls=1 if pairs else 0,
            prompt_chars=123 if pairs else 0,
        )


class _DeterministicRelationCandidates:
    def __init__(self, candidate: Memory) -> None:
        self.candidate = candidate
        self.candidate_row = CandidateMemory(
            memory_id=candidate.id,
            source_id="src-other",
            doc_id="other-doc",
            source_lineage_id="other-doc",
            visibility=candidate.visibility,
            owner_user_id=candidate.owner_user_id,
            repo_identifier=candidate.repo_identifier,
        )

    async def retrieve(self, **_kwargs):
        return CrossDocumentCandidateSelection(
            discovery=(
                RetrievedRelationCandidate(
                    memory=self.candidate_row,
                    score=1.0,
                    channels=("lexical_bm25",),
                ),
            ),
            audit={"candidate_count_kind": "windowed"},
        )

    async def load_selected_memories(self, selection, **_kwargs):
        return selection, {self.candidate.id: self.candidate}

    async def ensure_selection_current(self, *_args, **_kwargs):
        return None


class _MutatingRelationCandidates(_DeterministicRelationCandidates):
    def __init__(self, candidate: Memory, after_current) -> None:
        super().__init__(candidate)
        self._after_current = after_current

    async def ensure_selection_current(self, *_args, **_kwargs):
        await self._after_current()


class _EmptyRelationCandidates:
    def __init__(self, after_current=None) -> None:
        self._after_current = after_current

    async def retrieve(self, **_kwargs):
        return CrossDocumentCandidateSelection(
            discovery=(),
            audit={"candidate_count_kind": "windowed"},
        )

    async def load_selected_memories(self, selection, **_kwargs):
        return selection, {}

    async def ensure_selection_current(self, *_args, **_kwargs):
        if self._after_current is not None:
            await self._after_current()


async def _create_relation_discovery_fixture(
    db: Database,
    *,
    run_id: str,
) -> tuple[SourceProjection, Memory]:
    projection = _projection(run_id=run_id, body="A7 applies only to regular payroll.")
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=_SemanticEquivalentClient(),
    )
    await engine.prepare_and_commit_projected_lifecycle(
        projection=projection,
        doc_id="confluence-123",
        raw_memories=[
            RawMemory(
                content="A7 applies only to regular payroll.",
                memory_type="decision",
                evidence_quote="A7 applies only to regular payroll.",
                extraction_context="A7 applies only to regular payroll.",
            )
        ],
        doc_type="design-doc",
        project_key="ENG",
        repo_identifier=None,
        document_content=projection.observation_revisions[0].content,
        update_mode="full_document",
        changed_hunks=None,
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 15, 11, 0, tzinfo=timezone.utc),
    )
    [memory] = await db.list_memories(source="src-1", status="active")
    return projection, memory


@pytest.mark.asyncio
async def test_relation_discovery_rejects_stale_source_unit_revision(
    db: Database,
) -> None:
    projection, _memory = await _create_relation_discovery_fixture(
        db,
        run_id="projection-stale-relation-revision",
    )
    await db.db.execute(
        "UPDATE source_units SET current_revision_id = NULL WHERE id = ?",
        (projection.source_units[0].id,),
    )
    await db.db.commit()

    result = await RelationDiscovery(
        store=db,
        candidate_retriever=_EmptyRelationCandidates(),
        pair_classifier=_DeterministicRefinementClassifier(),
    ).process_slice(worker_id="relation-worker")

    assert result.failed_work == 1
    row = await db.db.execute_fetchall("SELECT status, error FROM relation_discovery_work")
    assert row[0]["status"] == "failed"
    assert "Source Unit revision is stale" in row[0]["error"]
    assert await db.db.execute_fetchall("SELECT id FROM relation_runs") == []


@pytest.mark.asyncio
async def test_relation_discovery_does_not_overwrite_changed_evidence_access(
    db: Database,
) -> None:
    projection, memory = await _create_relation_discovery_fixture(
        db,
        run_id="projection-stale-relation-access",
    )
    unit = await db.get_current_relation_evidence_unit(
        memory.id,
        source_id="src-1",
        source_unit_id=projection.source_units[0].id,
    )
    assert unit is not None

    async def change_access() -> None:
        await db.db.execute(
            "UPDATE evidence_units SET access_context_hash = ? WHERE id = ?",
            ("new-access-context", unit.id),
        )
        await db.db.commit()

    result = await RelationDiscovery(
        store=db,
        candidate_retriever=_EmptyRelationCandidates(change_access),
        pair_classifier=_DeterministicRefinementClassifier(),
    ).process_slice(worker_id="relation-worker")

    assert result.failed_work == 1
    [row] = await db.db.execute_fetchall(
        "SELECT access_context_hash FROM evidence_units WHERE id = ?",
        (unit.id,),
    )
    assert row["access_context_hash"] == "new-access-context"
    assert await db.db.execute_fetchall("SELECT id FROM relation_runs") == []


@pytest.mark.asyncio
async def test_relation_discovery_rejects_primary_evidence_demotion_before_commit(
    db: Database,
) -> None:
    projection, memory = await _create_relation_discovery_fixture(
        db,
        run_id="projection-stale-primary-relation-evidence",
    )
    unit = await db.get_current_relation_evidence_unit(
        memory.id,
        source_id="src-1",
        source_unit_id=projection.source_units[0].id,
    )
    assert unit is not None

    async def demote_primary_evidence() -> None:
        await db.db.execute(
            "UPDATE evidence_references SET role = 'required' WHERE evidence_unit_id = ?",
            (unit.id,),
        )
        await db.db.commit()

    result = await RelationDiscovery(
        store=db,
        candidate_retriever=_EmptyRelationCandidates(demote_primary_evidence),
        pair_classifier=_DeterministicRefinementClassifier(),
    ).process_slice(worker_id="relation-worker")

    assert result.failed_work == 1
    [row] = await db.db.execute_fetchall("SELECT status, error FROM relation_discovery_work")
    assert row["status"] == "failed"
    assert "evidence is no longer current" in row["error"]
    assert await db.db.execute_fetchall("SELECT id FROM relation_runs") == []


@pytest.mark.asyncio
async def test_relation_discovery_rejects_candidate_provenance_removed_before_commit(
    db: Database,
) -> None:
    await _create_relation_discovery_fixture(
        db,
        run_id="projection-stale-candidate-provenance",
    )
    candidate = Memory(
        id="mem-stale-relation-candidate",
        memory_type="decision",
        content="A7 applies to payroll.",
        content_hash=content_hash("A7 applies to payroll."),
        project_key="ENG",
    )
    await db.insert_memory(candidate)
    now = datetime(2026, 7, 15, 11, 0, tzinfo=timezone.utc)
    await db.upsert_document(
        DocumentRecord(
            doc_id="other-doc",
            source="src-other",
            source_url="https://example.test/other-doc",
            title="Relation candidate",
            space_or_project="ENG",
            author=None,
            last_modified=now,
            labels=[],
            version="1",
            content_hash="candidate-doc-hash",
            token_count=4,
            raw_content_uri=None,
            raw_content_type=None,
            normalized_content_uri=None,
            pdf_content_uri=None,
            last_synced=now,
        )
    )
    await db.add_memory_source(
        candidate.id,
        "other-doc",
        "confluence",
        source_updated_at=now,
    )

    async def remove_provenance() -> None:
        await db.db.execute(
            "DELETE FROM memory_sources WHERE memory_id = ? AND doc_id = ?",
            (candidate.id, "other-doc"),
        )
        await db.db.commit()

    result = await RelationDiscovery(
        store=db,
        candidate_retriever=_MutatingRelationCandidates(candidate, remove_provenance),
        pair_classifier=_DeterministicRefinementClassifier(),
    ).process_slice(worker_id="relation-worker")

    assert result.failed_work == 1
    [row] = await db.db.execute_fetchall("SELECT status, error FROM relation_discovery_work")
    assert row["status"] == "failed"
    assert "candidate provenance is stale" in row["error"]
    assert await db.db.execute_fetchall("SELECT id FROM relation_runs") == []


@pytest.mark.asyncio
async def test_relation_discovery_rejects_candidate_support_change_before_commit(
    db: Database,
) -> None:
    projection, challenger = await _create_relation_discovery_fixture(
        db,
        run_id="projection-stale-candidate-support",
    )
    challenger_evidence = await db.get_active_memory_support_evidence(challenger.id)
    assert len(challenger_evidence) == 1
    unit = await db.get_current_relation_evidence_unit(
        challenger.id,
        source_id="src-1",
        source_unit_id=projection.source_units[0].id,
    )
    assert unit is not None
    candidate = Memory(
        id="mem-stale-relation-support",
        memory_type="decision",
        content="A7 applies to payroll.",
        content_hash=content_hash("A7 applies to payroll."),
        project_key="ENG",
    )
    await db.insert_memory(candidate)
    now = datetime(2026, 7, 15, 11, 0, tzinfo=timezone.utc)
    await db.upsert_document(
        DocumentRecord(
            doc_id="other-doc",
            source="src-other",
            source_url="https://example.test/other-doc",
            title="Relation candidate",
            space_or_project="ENG",
            author=None,
            last_modified=now,
            labels=[],
            version="1",
            content_hash="candidate-doc-hash",
            token_count=4,
            raw_content_uri=None,
            raw_content_type=None,
            normalized_content_uri=None,
            pdf_content_uri=None,
            last_synced=now,
        )
    )
    await db.add_memory_source(candidate.id, "other-doc", "confluence", source_updated_at=now)

    async def attach_new_support() -> None:
        await db.upsert_memory_support_assertion(
            MemorySupportAssertion(
                id="support-stale-relation-candidate",
                memory_id=candidate.id,
                evidence_reference_id=challenger_evidence[0].reference_id,
                source_id="src-1",
                access_context_hash=unit.access_context_hash or "",
            )
        )

    result = await RelationDiscovery(
        store=db,
        candidate_retriever=_MutatingRelationCandidates(candidate, attach_new_support),
        pair_classifier=_DeterministicRefinementClassifier(),
    ).process_slice(worker_id="relation-worker")

    assert result.failed_work == 1
    [row] = await db.db.execute_fetchall("SELECT status, error FROM relation_discovery_work")
    assert row["status"] == "failed"
    assert "candidate current Support is stale" in row["error"]
    assert await db.db.execute_fetchall("SELECT id FROM relation_runs") == []


@pytest.mark.asyncio
async def test_private_relation_completion_rechecks_access_as_current_owner(
    db: Database,
) -> None:
    _projection_row, challenger = await _create_relation_discovery_fixture(
        db,
        run_id="projection-private-relation-access",
    )
    await db.db.execute(
        "UPDATE memories SET visibility = 'private', owner_user_id = ?, repo_identifier = ? WHERE id = ?",
        ("owner-1", "repo-1", challenger.id),
    )
    await db.db.execute(
        "UPDATE evidence_units SET visibility = 'private', owner_user_id = ?, repo_identifier = ? WHERE source_id = ?",
        ("owner-1", "repo-1", "src-1"),
    )
    await db.db.execute("UPDATE relation_discovery_work SET actor_user_id = NULL")
    await db.db.commit()

    candidate = Memory(
        id="mem-private-relation-candidate",
        memory_type="decision",
        content="A7 applies to payroll.",
        content_hash=content_hash("A7 applies to payroll."),
        project_key="ENG",
        visibility="private",
        owner_user_id="owner-1",
        repo_identifier="repo-1",
    )
    await db.insert_memory(candidate)
    now = datetime(2026, 7, 15, 11, 0, tzinfo=timezone.utc)
    await db.upsert_source(
        id="src-other",
        type="confluence",
        name="Private relation candidate",
        config_json="{}",
        access_policy="private",
        owner_user_id="owner-1",
    )
    await db.upsert_document(
        DocumentRecord(
            doc_id="other-doc",
            source="src-other",
            source_url="https://example.test/other-doc",
            title="Relation candidate",
            space_or_project="ENG",
            author=None,
            last_modified=now,
            labels=[],
            version="1",
            content_hash="candidate-doc-hash",
            token_count=4,
            raw_content_uri=None,
            raw_content_type=None,
            normalized_content_uri=None,
            pdf_content_uri=None,
            last_synced=now,
        )
    )
    await db.add_memory_source(
        candidate.id,
        "other-doc",
        "confluence",
        source_updated_at=now,
    )

    async def disable_candidate_source() -> None:
        await db.set_source_subscription("src-other", "owner-1", False)

    result = await RelationDiscovery(
        store=db,
        candidate_retriever=_MutatingRelationCandidates(candidate, disable_candidate_source),
        pair_classifier=_DeterministicRefinementClassifier(),
    ).process_slice(worker_id="relation-worker")

    assert result.failed_work == 1
    [row] = await db.db.execute_fetchall("SELECT status, error FROM relation_discovery_work")
    assert row["status"] == "failed"
    assert "candidate source access is stale" in row["error"]
    assert await db.db.execute_fetchall("SELECT id FROM relation_runs") == []


@pytest.mark.asyncio
async def test_relation_discovery_persists_direction_after_lifecycle_commit(
    db: Database,
) -> None:
    projection = _projection(
        run_id="projection-relation-discovery",
        body="A7 applies only to regular payroll.",
    )
    raw = RawMemory(
        content="A7 applies only to regular payroll.",
        memory_type="decision",
        evidence_quote="A7 applies only to regular payroll.",
        extraction_context="A7 applies only to regular payroll.",
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=_SemanticEquivalentClient(),
    )
    await engine.prepare_and_commit_projected_lifecycle(
        projection=projection,
        doc_id="confluence-123",
        raw_memories=[raw],
        doc_type="design-doc",
        project_key="ENG",
        repo_identifier=None,
        document_content=projection.observation_revisions[0].content,
        update_mode="full_document",
        changed_hunks=None,
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 15, 11, 0, tzinfo=timezone.utc),
    )
    [challenger] = await db.list_memories(source="src-1", status="active")
    candidate = Memory(
        id="mem-relation-candidate",
        memory_type="decision",
        content="A7 applies to payroll.",
        content_hash=content_hash("A7 applies to payroll."),
        project_key="ENG",
    )
    await db.insert_memory(candidate)
    now = datetime(2026, 7, 15, 11, 0, tzinfo=timezone.utc)
    await db.upsert_document(
        DocumentRecord(
            doc_id="other-doc",
            source="src-other",
            source_url="https://example.test/other-doc",
            title="Relation candidate",
            space_or_project="ENG",
            author=None,
            last_modified=now,
            labels=[],
            version="1",
            content_hash="candidate-doc-hash",
            token_count=4,
            raw_content_uri=None,
            raw_content_type=None,
            normalized_content_uri=None,
            pdf_content_uri=None,
            last_synced=now,
        )
    )
    await db.add_memory_source(
        candidate.id,
        "other-doc",
        "confluence",
        source_updated_at=now,
    )

    result = await RelationDiscovery(
        store=db,
        candidate_retriever=_DeterministicRelationCandidates(candidate),
        pair_classifier=_DeterministicRefinementClassifier(),
    ).process_slice(worker_id="relation-worker")

    assert result.completed_work == 1
    assert result.checked_candidate_pairs == 1
    assert result.llm_calls == 1
    unit = await db.get_current_relation_evidence_unit(
        challenger.id,
        source_id="src-1",
        source_unit_id=projection.source_units[0].id,
    )
    assert unit is not None
    [relation] = await db.get_evidence_relations(unit.id)
    assert relation.memory_id == candidate.id
    assert relation.relation_type is RelationType.REFINES
    assert relation.direction is RelationDirection.CHALLENGER_TO_CANDIDATE
    [relation_run] = await db.db.execute_fetchall(
        "SELECT result_memory_id FROM relation_runs WHERE evidence_unit_id = ?",
        (unit.id,),
    )
    assert relation_run["result_memory_id"] == challenger.id


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
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_FailingOutboxDrainer(db),
        structured_llm_client=_SemanticEquivalentClient(),
    )

    stats = await engine.prepare_and_commit_projected_lifecycle(
        projection=projection,
        doc_id="confluence-123",
        raw_memories=[raw],
        doc_type="design-doc",
        project_key="ENG",
        repo_identifier=None,
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
    assert (
        await db.get_evidence_unit((await db.get_active_memory_support_evidence(incumbent.id))[0].evidence_unit_id)
        is not None
    )


@pytest.mark.asyncio
async def test_rebaseline_replay_reuses_memory_with_explicit_observation_support(
    db: Database,
) -> None:
    await _set_fixture_source_type(db, "jira")
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
        cross_document_candidates=_candidate_retriever(adapters),
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
        "document_content": "PAY-12",
        "update_mode": "full_document",
        "changed_hunks": None,
        "update_plan_stats": None,
        "source_updated_at": datetime(2026, 7, 15, 11, 0, tzinfo=timezone.utc),
    }

    first = await engine.prepare_and_commit_projected_lifecycle(projection=projection, **arguments)
    [memory] = await db.list_memories(source="src-1", status="active")
    assert first["added"] == 1
    assert await db.get_active_memory_support_evidence(memory.id, source_id="src-1")

    await db.rebaseline_source_lifecycle("src-1")
    replay = _jira_projection(
        run_id="projection-jira-after-rebaseline",
        description="A7 applies only to regular payroll.",
        comment_body="The rollout note is unrelated.",
    )
    second = await engine.prepare_and_commit_projected_lifecycle(projection=replay, **arguments)

    replayed = await db.get_memory(memory.id)
    support = await db.get_active_memory_support_evidence(memory.id, source_id="src-1")
    assert replayed is not None and replayed.status == "active"
    assert second["reactivated"] == 1
    assert second["corroborated"] == 1
    assert second["relation_discovery_enqueued"] == 1
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
async def test_rebaseline_reset_preserves_an_overlapping_source_projection(
    db: Database,
) -> None:
    memory = Memory(
        id="mem-overlapping-source-edge",
        memory_type="fact",
        content="A fact projected by two Configured Sources.",
        content_hash=content_hash("A fact projected by two Configured Sources."),
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
    await db.upsert_source(
        id="src-overlap",
        type="confluence",
        name="Overlapping source",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="owner-1",
    )
    await db.restore_memory_source_snapshot(
        MemorySource(
            memory_id=memory.id,
            doc_id="confluence-123",
            source_id="src-overlap",
            source_type="confluence",
            excerpt=memory.content,
            source_updated_at=None,
        )
    )
    overlap_projection = _projection(
        run_id="projection-overlapping-source-edge",
        body=memory.content,
        source_id="src-overlap",
    )
    await db.record_source_projection(overlap_projection)
    overlap_observation = overlap_projection.observations[0]
    overlap_revision = next(
        revision
        for revision in overlap_projection.observation_revisions
        if revision.observation_id == overlap_observation.id
    )
    overlap_unit = EvidenceUnit(
        id="eu-overlapping-source-edge",
        source_id="src-overlap",
        doc_id="confluence-123",
        doc_revision_id=overlap_projection.source_unit_revisions[0].id,
        source_type="confluence",
        source_anchor=overlap_observation.id,
        source_lineage_id=overlap_projection.source_units[0].id,
        project_key="ENG",
        visibility="workspace",
        owner_user_id=None,
        repo_identifier=None,
        content=overlap_revision.content,
        excerpt=memory.content,
        evidence_provenance=EvidenceContentProvenance.SOURCE_EXCERPT,
        access_context_hash="workspace-eng",
    )
    await db.upsert_evidence_unit(overlap_unit)
    overlap_reference = (
        await db.record_evidence_references(
            overlap_unit.id,
            (
                EvidenceReference(
                    role=EvidenceRole.PRIMARY,
                    anchor=SourceAnchor(
                        kind=AnchorKind.WHOLE_OBSERVATION,
                        observation_id=overlap_observation.id,
                        observation_revision_id=overlap_revision.id,
                    ),
                ),
            ),
        )
    )[0]
    await db.upsert_memory_support_assertion(
        MemorySupportAssertion(
            id="support-overlapping-source-edge",
            memory_id=memory.id,
            evidence_reference_id=overlap_reference.id or "",
            source_id="src-overlap",
            access_context_hash="workspace-eng",
        )
    )

    result = await db.rebaseline_source_lifecycle("src-1")

    persisted = await db.get_memory(memory.id)
    assert result.retired_memory_ids == ()
    assert persisted is not None and persisted.status == "active"
    remaining_sources = await db.get_memory_sources(memory.id)
    assert [(source.source_id, source.doc_id) for source in remaining_sources] == [("src-overlap", "confluence-123")]
    metadata = await db.db.execute_fetchall(
        """SELECT source_id FROM memory_search_metadata_trigram
             WHERE memory_id = ? AND doc_id = ?""",
        (memory.id, "confluence-123"),
    )
    assert [row["source_id"] for row in metadata] == ["src-overlap"]
    supports = await db.db.execute_fetchall(
        "SELECT source_id FROM memory_support_assertions WHERE memory_id = ? AND active = 1",
        (memory.id,),
    )
    assert [row["source_id"] for row in supports] == ["src-overlap"]


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
        )

    # Non-semantic metadata tuning does not invalidate source evidence.
    await db.update_memory_content(
        incumbent.id,
        incumbent.content,
        0.8,
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

    entity_id = await db.upsert_entity("A7", "A7")
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
        entity_refs=["A7"],
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
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=_ReplacementClient(incumbent.id),
    )
    stats = await engine.prepare_and_commit_projected_lifecycle(
        projection=second,
        doc_id="confluence-123",
        raw_memories=[raw],
        doc_type="design-doc",
        project_key="ENG",
        repo_identifier=None,
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
    assert stats["relation_discovery_enqueued"] == 1
    discovery = await RelationDiscovery(
        store=db,
        candidate_retriever=_candidate_retriever(adapters),
        pair_classifier=_DeterministicContradictionClassifier(),
    ).process_slice(worker_id="relation-worker-cross-source")
    assert discovery.completed_work == 1
    assert discovery.checked_candidate_pairs >= 1
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
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_OutboxDrainer(db),
    )

    stats = await engine.prepare_and_commit_projected_lifecycle(
        projection=projection,
        doc_id="confluence-123",
        raw_memories=[raw],
        doc_type="document",
        project_key="ENG",
        repo_identifier=None,
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
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_OutboxDrainer(db),
    )

    stats = await engine.prepare_and_commit_projected_lifecycle(
        projection=projection,
        doc_id="confluence-123",
        raw_memories=[raw],
        doc_type="document",
        project_key="ENG",
        repo_identifier=None,
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
        prior_observation_revisions={revision.observation_id: revision for revision in initial.observation_revisions},
        reason="not_returned_by_authoritative_snapshot",
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        cross_document_candidates=_candidate_retriever(adapters),
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
        prior_observation_revisions={revision.observation_id: revision for revision in initial.observation_revisions},
        reason="not_returned_by_authoritative_snapshot",
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        cross_document_candidates=_candidate_retriever(adapters),
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


@pytest.mark.asyncio
async def test_tombstone_retains_document_when_unmapped_provenance_remains(
    db: Database,
) -> None:
    initial = _projection(
        run_id="projection-before-unmapped-delete",
        body="A7 was previously corroborated.",
    )
    await db.record_source_projection(initial)
    memory = await _seed_incumbent_support(db, projection=initial)
    await db.db.execute(
        "UPDATE memory_sources SET support_kind = 'corroborated' WHERE memory_id = ? AND doc_id = ?",
        (memory.id, "confluence-123"),
    )
    await db.db.execute(
        "UPDATE memory_support_assertions SET active = 0, removed_at = ? WHERE memory_id = ?",
        (datetime.now(timezone.utc).isoformat(), memory.id),
    )
    await db.db.commit()
    tombstone = project_source_unit_tombstone(
        source_type="confluence",
        run_id="projection-unmapped-delete",
        source_unit=initial.source_units[0],
        prior_unit_revision=initial.source_unit_revisions[0],
        prior_observation_revisions={revision.observation_id: revision for revision in initial.observation_revisions},
        reason="not_returned_by_authoritative_snapshot",
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_OutboxDrainer(db),
    )

    result = await engine.apply_projected_tombstone(
        projection=tombstone,
        doc_id="confluence-123",
        reason="not_returned_by_authoritative_snapshot",
        lifecycle_cycle_id="unmapped-source-removal",
    )

    assert result == {
        "retired": 0,
        "pending_review": 0,
        "can_delete_document": False,
    }
    assert await db.get_document("confluence-123") is not None
    assert [source.doc_id for source in await db.get_memory_sources(memory.id)] == ["confluence-123"]

    retry = await engine.apply_projected_tombstone(
        projection=tombstone,
        doc_id="confluence-123",
        reason="not_returned_by_authoritative_snapshot",
        lifecycle_cycle_id="unmapped-source-removal",
    )

    assert retry == result
