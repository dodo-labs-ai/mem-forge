from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from memforge.models import (
    ContentItem,
    DocumentRecord,
    MemoryExtractionResult,
    NormalizedContent,
    RawContent,
)
from memforge.pipeline.extraction_contract import (
    DURABLE_MEMORY_QUALITY_RULES,
    PROJECTION_EXTRACTION_CONTRACT_VERSION,
)
from memforge.pipeline.memory_extractor import PROJECTION_FRAGMENT_EXTRACTION_PROMPT
from memforge.pipeline.source_projection_adapters import project_source_item
from memforge.pipeline.sync import _memory_extraction_error
from memforge.source_derivation import (
    SourceUnitDerivationContext,
    SourceUnitDerivationRequest,
    SourceUnitDeriver,
    _planning_failure_extraction,
    require_source_derivation_ready_for_commit,
    source_derivation_manifest,
)
from memforge.pipeline.extraction_requests import plan_extraction_requests
from memforge.pipeline.projection_context import (
    PROJECTION_AUTHORITY_SEGMENTATION_POLICY_VERSION,
    ExtractionAuthority,
    ExtractionPlan,
    ProjectionEvidencePlanningFailure,
    ProjectionEvidencePlanningFailureCode,
    plan_projection_evidence_work,
)
from memforge.pipeline.revision_assessment import REVISION_INPUT_POLICY, RevisionAssessmentContext
from memforge.storage.database import Database
from tests.llm_fixture import NoopMemoryExtractor


@pytest.fixture
async def db(tmp_path) -> Database:
    database = Database(str(tmp_path / "extraction-contract.db"))
    await database.connect()
    try:
        yield database
    finally:
        await database.close()



def _planned_requests(projection, authority):
    assert isinstance(authority, ExtractionAuthority)
    return plan_extraction_requests(
        RevisionAssessmentContext(projection=projection, base=None, access_context_hash="workspace"),
        authority,
        extractor=NoopMemoryExtractor(),
        source_type=projection.source_type,
        doc_type="document",
    ).requests

def test_extraction_contract_version_is_pinned_into_derivation_identity() -> None:
    # Stored derivations and batch ids hash this value; changing it supersedes
    # every stored derivation instead of resuming it.
    assert PROJECTION_EXTRACTION_CONTRACT_VERSION == "projection-extraction-v22"


@pytest.mark.parametrize("status,version,reason,ready", (
    ("completed", PROJECTION_EXTRACTION_CONTRACT_VERSION, None, True),
    ("completed", PROJECTION_EXTRACTION_CONTRACT_VERSION, "EXTRACTION_INPUT_CAPACITY_EXCEEDED", False),
    ("completed", "projection-extraction-v12", None, False),
    ("retryable_failure", PROJECTION_EXTRACTION_CONTRACT_VERSION, None, False),
    ("pending", PROJECTION_EXTRACTION_CONTRACT_VERSION, None, False),
    ("applied", "projection-extraction-v12", None, True),
))
def test_derivation_commit_readiness_distinguishes_failed_or_old_work_from_applied_history(status, version, reason, ready):
    if ready:
        require_source_derivation_ready_for_commit(status=status, extraction_contract_version=version, terminal_reason_code=reason)
    else:
        with pytest.raises(ValueError, match="not ready for lifecycle commit"):
            require_source_derivation_ready_for_commit(status=status, extraction_contract_version=version, terminal_reason_code=reason)
def test_extraction_prompt_carries_the_durable_memory_quality_contract() -> None:
    required_rules = (
        "Worth remembering (keep):",
        "When unsure, keep.",
        "ONE CLAIM, ONE MEMORY",
        "Fold a rejected alternative into the chosen decision",
        "NO META-MEMORIES",
        "OPERATIONAL DOES NOT MEAN TRANSIENT",
        "For each candidate, preserve the language of its owned source evidence.",
        "When that evidence is primarily Chinese, write memory.content in Chinese.",
        "Read-only context may resolve meaning but must not change the candidate's language.",
    )

    assert DURABLE_MEMORY_QUALITY_RULES in PROJECTION_FRAGMENT_EXTRACTION_PROMPT
    for rule in required_rules:
        assert rule in DURABLE_MEMORY_QUALITY_RULES
    from memforge.memory.candidate_admission import _ADMISSION_INSTRUCTIONS
    from memforge.pipeline.memory_value import MEMORY_VALUE_DEFINITION

    assert MEMORY_VALUE_DEFINITION in DURABLE_MEMORY_QUALITY_RULES
    assert MEMORY_VALUE_DEFINITION in _ADMISSION_INSTRUCTIONS
    assert all(obsolete not in DURABLE_MEMORY_QUALITY_RULES for obsolete in (
        "PREFER EMPTY", "CODE-RECOVERABLE", "six months", "in under a minute",
    ))


@pytest.mark.parametrize("failure", ("missing_result", "typed_failure"))
def test_unit_aggregation_refuses_incomplete_work_and_preserves_typed_failure(failure):
    from memforge.models import RawMemory
    from memforge.source_derivation import assemble_source_derivation_results, safe_derivation_error
    from tests.test_projection_fragments import _projection

    projection = _projection()
    [request] = _planned_requests(projection, ExtractionAuthority({"obs-primary": None}))
    plan = ExtractionPlan((request,))
    complete = MemoryExtractionResult(memories=[RawMemory(content="Use approval before release.", memory_type="fact")])
    results = () if failure == "missing_result" else (complete,)
    if failure == "typed_failure":
        results = (MemoryExtractionResult(
            error_type="structured_llm_error", error="Fixture provider error",
            metadata={"safe_error_code": "INPUT_CAPACITY_EXCEEDED",
                      "safe_validation_fields": [{"location": "memories.0.content", "type": "missing"}]},
        ),)
    result = assemble_source_derivation_results(projection=projection, plan=plan, results=results)
    assert result.error_type == "source_derivation_work_failure" and result.memories == []
    assert result.metadata["lifecycle_mutation_skipped"]
    if failure == "typed_failure":
        _, code, fields = safe_derivation_error(result)
        assert code == "INPUT_CAPACITY_EXCEEDED" and fields == (("memories.0.content", "missing"),)


@pytest.mark.parametrize("stage", ("planning", "execution"))
def test_accounted_local_skips_preserve_successful_extraction(stage):
    from memforge.models import RawMemory
    from memforge.source_derivation import assemble_source_derivation_results
    from tests.test_projection_fragments import _projection

    projection = _projection()
    [request] = _planned_requests(projection, ExtractionAuthority({"obs-primary": None}))
    plan = ExtractionPlan((request,), ("unread-group",) if stage == "planning" else ())
    memory = RawMemory(content="Use approval before release.", memory_type="fact")
    complete = MemoryExtractionResult(memories=[memory])
    if stage == "execution":
        complete.metadata["skipped_reading_group_count"] = 1
    result = assemble_source_derivation_results(projection=projection, plan=plan, results=(complete,))
    assert result.error_type is None and result.memories == [memory]
    assert result.metadata["skipped_reading_group_count"] == 1


def test_complete_empty_claim_response_is_valid_unit_success():
    from memforge.source_derivation import assemble_source_derivation_results
    from tests.test_projection_fragments import _projection

    projection = _projection()
    [request] = _planned_requests(projection, ExtractionAuthority({"obs-primary": None}))
    result = assemble_source_derivation_results(
        projection=projection, plan=ExtractionPlan((request,)), results=(MemoryExtractionResult(memories=[]),),
    )
    assert result.error_type is None and result.memories == []


def test_v9_derivation_identity_binds_access_and_inference_capability() -> None:
    now = datetime(2026, 9, 4, 0, 0, tzinfo=timezone.utc)
    body = "# Rule\n\nUse the current approval policy.\n"
    item = ContentItem(
        item_id="doc-work-identity",
        title="Work identity",
        source_url="https://example.test/repo/work.md",
        last_modified=now,
        content_type="text/markdown",
        version="1",
    )
    projection = project_source_item(
        source_id="source-work-identity",
        source_type="github_repo",
        run_id="run-work-identity",
        item=item,
        raw=RawContent(
            item=item,
            body=body.encode(),
            content_type="text/markdown",
        ),
        normalized=NormalizedContent(item=item, markdown_body=body),
        scope={},
        access_context={"visibility": "workspace"},
    )
    document = DocumentRecord(
        doc_id=item.item_id,
        source="source-work-identity",
        source_url=item.source_url,
        title=item.title,
        space_or_project="TEST",
        author=None,
        last_modified=now,
        labels=[],
        version="1",
        content_hash=hashlib.sha256(body.encode()).hexdigest(),
        token_count=12,
        last_synced=now,
    )
    context = SourceUnitDerivationContext(
        document=document,
        doc_type="document",
        project_key=None,
        repo_identifier=None,
        document_content=body,
        update_mode="full_document",
        changed_hunks=None,
        update_plan_stats=None,
        source_updated_at=now.isoformat(),
        user_id=None,
    )
    batches = _planned_requests(
        projection,
        plan_projection_evidence_work(projection, reprocess_all_current_observations=False),
    )

    first = source_derivation_manifest(
        projection,
        batches,
        context=context,
        evidence_work_identity_hash="a" * 64,
    )
    changed = source_derivation_manifest(
        projection,
        batches,
        context=context,
        evidence_work_identity_hash="b" * 64,
    )

    assert first.id != changed.id
    assert first.batches[0].input_payload_hash != (
        changed.batches[0].input_payload_hash
    )

    reprocess_batches = _planned_requests(
        projection,
        plan_projection_evidence_work(projection, reprocess_all_current_observations=True),
    )
    first_operation = source_derivation_manifest(
        projection,
        reprocess_batches,
        context=replace(
            context,
            reprocess_all_current_observations=True,
            reprocess_operation_id="sync-run-1",
        ),
        evidence_work_identity_hash="a" * 64,
    )
    next_operation = source_derivation_manifest(
        projection,
        reprocess_batches,
        context=replace(
            context,
            reprocess_all_current_observations=True,
            reprocess_operation_id="sync-run-2",
        ),
        evidence_work_identity_hash="a" * 64,
    )

    assert first_operation.id != next_operation.id


@pytest.mark.asyncio
async def test_v9_reprocess_without_operation_identity_fails_before_llm(
    db: Database,
) -> None:
    source_id = "source-reprocess-identity"
    await db.upsert_source(
        id=source_id,
        type="github_repo",
        name="Reprocess identity",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
    )
    now = datetime(2026, 9, 4, 0, 0, tzinfo=timezone.utc)
    body = "# Rule\n\nUse the approved calendar.\n"
    item = ContentItem(
        item_id="doc-reprocess-identity",
        title="Rule",
        source_url="https://example.test/repo/rule.md",
        last_modified=now,
        content_type="text/markdown",
        version="1",
    )
    projection = project_source_item(
        source_id=source_id,
        source_type="github_repo",
        run_id="run-reprocess-identity",
        item=item,
        raw=RawContent(item=item, body=body.encode(), content_type="text/markdown"),
        normalized=NormalizedContent(item=item, markdown_body=body),
    )
    document = DocumentRecord(
        doc_id=item.item_id,
        source=source_id,
        source_url=item.source_url,
        title=item.title,
        space_or_project="TEST",
        author=None,
        last_modified=now,
        labels=[],
        version="1",
        content_hash=hashlib.sha256(body.encode()).hexdigest(),
        token_count=8,
        last_synced=now,
    )
    extractor_called = False

    async def plan(_authority):
        nonlocal extractor_called
        extractor_called = True
        return ExtractionPlan(())

    async def extract(_batch):
        nonlocal extractor_called
        extractor_called = True
        return MemoryExtractionResult(memories=[])

    result = await SourceUnitDeriver(db).derive(
        SourceUnitDerivationRequest(
            projection=projection,
            context=SourceUnitDerivationContext(
                document=document,
                doc_type="document",
                project_key=None,
                repo_identifier=None,
                document_content=body,
                update_mode="full_document",
                changed_hunks=None,
                update_plan_stats=None,
                source_updated_at=now.isoformat(),
                user_id=None,
                reprocess_all_current_observations=True,
            ),
            plan_requests=plan,
            extract_request=extract,
            max_concurrent=1,
            access_context_hash="access-reprocess",
            inference_capability_hash="inference-reprocess",
        )
    )

    assert extractor_called is False
    assert result.derivation.terminal_reason_code == (
        "REPROCESS_AUTHORIZATION_MISSING"
    )


@pytest.mark.asyncio
async def test_missing_v9_authority_base_is_durable_and_skips_the_llm(
    db: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_id = "source-unmappable-authority"
    await db.upsert_source(
        id=source_id,
        type="github_repo",
        name="Unmappable authority",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
    )
    now = datetime(2026, 9, 4, 0, 0, tzinfo=timezone.utc)
    initial_body = "# Rule\n\nUse Monday.\n"
    target_body = "# Rule\n\nUse Tuesday.\n"
    item = ContentItem(
        item_id="doc-unmappable-authority",
        title="Rule",
        source_url="https://example.test/repo/rule.md",
        last_modified=now,
        content_type="text/markdown",
        version="1",
    )
    initial = project_source_item(
        source_id=source_id,
        source_type="github_repo",
        run_id="run-unmappable-base",
        item=item,
        raw=RawContent(item=item, body=initial_body.encode(), content_type="text/markdown"),
        normalized=NormalizedContent(item=item, markdown_body=initial_body),
    )
    target_item = ContentItem(
        item_id=item.item_id,
        title=item.title,
        source_url=item.source_url,
        last_modified=now,
        content_type="text/markdown",
        version="2",
    )
    target = project_source_item(
        source_id=source_id,
        source_type="github_repo",
        run_id="run-unmappable-target",
        item=target_item,
        raw=RawContent(
            item=target_item,
            body=target_body.encode(),
            content_type="text/markdown",
        ),
        normalized=NormalizedContent(item=target_item, markdown_body=target_body),
        prior_unit_revision=initial.source_unit_revisions[0],
        prior_observation_revisions={
            revision.observation_id: revision
            for revision in initial.observation_revisions
        },
    )
    document = DocumentRecord(
        doc_id=item.item_id,
        source=source_id,
        source_url=item.source_url,
        title=item.title,
        space_or_project="TEST",
        author=None,
        last_modified=now,
        labels=[],
        version="2",
        content_hash=hashlib.sha256(target_body.encode()).hexdigest(),
        token_count=5,
        last_synced=now,
    )
    extractor_called = False

    async def plan(_authority):
        nonlocal extractor_called
        extractor_called = True
        return ExtractionPlan(())

    async def extract(_batch):
        nonlocal extractor_called
        extractor_called = True
        return MemoryExtractionResult(memories=[])

    request = SourceUnitDerivationRequest(
        projection=target,
        context=SourceUnitDerivationContext(
            document=document,
            doc_type="document",
            project_key=None,
            repo_identifier=None,
            document_content="content that cannot map to the target Revision",
            update_mode="diff_guided",
            changed_hunks="@@ changed @@",
            update_plan_stats=None,
            source_updated_at=now.isoformat(),
            user_id=None,
            current_changed_ranges=((0, 7),),
        ),
        plan_requests=plan,
        extract_request=extract,
        max_concurrent=1,
        access_context_hash="access-unmappable-authority",
        inference_capability_hash="inference-unmappable-authority",
    )
    published_events: list[tuple] = []
    published_assessments: list[tuple] = []

    class RuntimeSink:
        def publish(self, events):
            published_events.append(events)

    class AssessmentSink:
        def publish(self, assessments, events):
            published_assessments.append((assessments, events))

    deriver = SourceUnitDeriver(
        db,
        runtime_event_trace_sink=RuntimeSink(),
        agent_assessment_sink=AssessmentSink(),
    )
    monkeypatch.setenv("MEMFORGE_DEPLOYMENT_REVISION", "deployment-a")

    result = await deriver.derive(request)

    monkeypatch.setenv("MEMFORGE_DEPLOYMENT_REVISION", "deployment-b")
    replay = await deriver.derive(request)

    assert extractor_called is False
    assert replay.derivation == result.derivation
    assert len(published_events) == 1
    assert published_events[0][0].deployment_revision == "deployment-a"
    assert len(published_assessments) == 1
    assert published_assessments[0][1] == published_events[0]
    assert result.extraction.error_type == "evidence_authority_planning_failed"
    assert result.extraction.error is not None
    assert result.extraction.error.startswith("INCREMENTAL_BASE_UNAVAILABLE")
    assert result.derivation.status == "completed"
    assert (
        result.derivation.terminal_reason_code
        == "INCREMENTAL_BASE_UNAVAILABLE"
    )
    assert result.derivation.authority_plan_identity == {
        "semantic_input_policy": REVISION_INPUT_POLICY,
        "access_context_hash": "access-unmappable-authority",
        "authority_policy_version": PROJECTION_AUTHORITY_SEGMENTATION_POLICY_VERSION,
        "base_unit_revision_id": initial.source_unit_revisions[0].id,
        "extraction_contract_version": PROJECTION_EXTRACTION_CONTRACT_VERSION,
        "inference_capability_hash": "inference-unmappable-authority",
        "representation_profiles": [
            {
                "coordinate_space": "unicode-scalar",
                "name": revision.evidence_profile.name,
                "observation_id": revision.observation_id,
                "schema_name": None,
                "schema_version": None,
                "version": 1,
            }
            for revision in sorted(target.observation_revisions, key=lambda item: (item.observation_id, item.id))
        ],
        "reprocess_operation_id": None,
        "target_unit_revision_id": target.source_unit_revisions[0].id,
        "transition": "incremental",
    }
    rows = await db.db.execute_fetchall(
        """SELECT event_name, outcome, reason_code,
                  base_unit_revision_id, operation,
                  model_call_count, mutation_count, deployment_revision
           FROM agent_runtime_events"""
    )
    assert [tuple(row) for row in rows] == [
        (
            "evidence_authority_planning",
            "failed",
            "INCREMENTAL_BASE_UNAVAILABLE",
            initial.source_unit_revisions[0].id,
            "incremental",
            0,
            0,
            "deployment-a",
        )
    ]
    assessment_rows = await db.db.execute_fetchall(
        "SELECT criterion, label, reason_code FROM agent_assessments"
    )
    assert [tuple(row) for row in assessment_rows] == [
        (
            "evidence_authority_planning",
            "fail",
            "INCREMENTAL_BASE_UNAVAILABLE",
        )
    ]


def test_planning_failure_error_names_its_code_and_identifiers_but_no_source_text() -> None:
    failure = ProjectionEvidencePlanningFailure(
        code=ProjectionEvidencePlanningFailureCode.INCREMENTAL_AUTHORITY_UNMAPPABLE,
        observation_id="obs-issue-core",
        observation_revision_id="rev-issue-core-2",
        representation_profile="jira.issue_core",
        changed_structure_count=3,
        authorized_structure_count=1,
    )

    error = _memory_extraction_error(doc_id="jira-SFPAY-1", extraction=_planning_failure_extraction(failure))

    assert str(error) == (
        "memory extraction failed for jira-SFPAY-1: evidence_authority_planning_failed: "
        "INCREMENTAL_AUTHORITY_UNMAPPABLE (observation=obs-issue-core, observation_revision=rev-issue-core-2, "
        "representation_profile=jira.issue_core, changed_structures=3, authorized_structures=1)"
    )
    assert error.error_code == "INCREMENTAL_AUTHORITY_UNMAPPABLE"


def test_planning_failure_without_identifiers_is_its_code() -> None:
    failure = ProjectionEvidencePlanningFailure(
        code=ProjectionEvidencePlanningFailureCode.REPROCESS_AUTHORIZATION_MISSING,
        observation_id=None,
        observation_revision_id=None,
        representation_profile=None,
    )

    assert _planning_failure_extraction(failure).error == "REPROCESS_AUTHORIZATION_MISSING"
