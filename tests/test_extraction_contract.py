from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from memforge.memory.evidence import SupportScopeVersion
from memforge.models import (
    ContentItem,
    DocumentRecord,
    MemoryExtractionResult,
    NormalizedContent,
    RawContent,
)
from memforge.pipeline import extraction_contract
from memforge.pipeline.extraction_contract import (
    PROJECTION_EXTRACTION_V8,
    PROJECTION_EXTRACTION_V9,
    ProjectionExtractionContract,
    active_projection_extraction_contract,
    projection_extraction_contract,
)
from memforge.pipeline.source_projection_adapters import project_source_item
from memforge.source_derivation import (
    SourceUnitDerivationContext,
    SourceUnitDerivationRequest,
    SourceUnitDeriver,
    source_derivation_manifest,
)
from memforge.pipeline.projection_context import plan_projection_evidence_work
from memforge.storage.database import Database


@pytest.fixture
async def db(tmp_path) -> Database:
    database = Database(str(tmp_path / "extraction-contract.db"))
    await database.connect()
    try:
        yield database
    finally:
        await database.close()


@pytest.mark.parametrize(
    ("support_scope_version", "expected_version", "uses_fragment_catalog"),
    (
        (SupportScopeVersion.REFERENCE_SET_V1, PROJECTION_EXTRACTION_V8, False),
        (SupportScopeVersion.EVIDENCE_UNIT_SET_V2, PROJECTION_EXTRACTION_V9, True),
    ),
)
def test_active_projection_extraction_contract_is_scope_driven(
    support_scope_version: SupportScopeVersion,
    expected_version: str,
    uses_fragment_catalog: bool,
) -> None:
    contract = active_projection_extraction_contract(support_scope_version)

    assert contract.version == expected_version
    assert contract.uses_fragment_catalog is uses_fragment_catalog
    assert projection_extraction_contract(expected_version) is contract


def test_future_fragment_contract_promotion_changes_only_the_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v10 = ProjectionExtractionContract(
        version="projection-extraction-v10",
        uses_fragment_catalog=True,
    )
    monkeypatch.setitem(
        extraction_contract._PROJECTION_EXTRACTION_CONTRACTS,
        v10.version,
        v10,
    )
    monkeypatch.setitem(
        extraction_contract._ACTIVE_CONTRACT_VERSION_BY_SUPPORT_SCOPE,
        SupportScopeVersion.EVIDENCE_UNIT_SET_V2,
        v10.version,
    )

    assert active_projection_extraction_contract(
        SupportScopeVersion.EVIDENCE_UNIT_SET_V2
    ) is v10
    assert projection_extraction_contract(v10.version) is v10


@pytest.mark.asyncio
async def test_future_fragment_contract_runs_through_the_deriver(
    db: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v10 = ProjectionExtractionContract(
        version="projection-extraction-v10",
        uses_fragment_catalog=True,
    )
    monkeypatch.setitem(
        extraction_contract._PROJECTION_EXTRACTION_CONTRACTS,
        v10.version,
        v10,
    )
    monkeypatch.setitem(
        extraction_contract._ACTIVE_CONTRACT_VERSION_BY_SUPPORT_SCOPE,
        SupportScopeVersion.EVIDENCE_UNIT_SET_V2,
        v10.version,
    )
    source_id = "source-v10-contract"
    await db.upsert_source(
        id=source_id,
        type="github_repo",
        name="V10 contract",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
    )
    now = datetime(2026, 9, 2, 0, 0, tzinfo=timezone.utc)
    body = "# Rule\n\nA future contract must keep compiler-backed planning.\n"
    item = ContentItem(
        item_id="doc-v10-contract",
        title="V10 contract",
        source_url="https://example.test/repo/v10.md",
        last_modified=now,
        content_type="text/markdown",
        version="1",
    )
    projection = project_source_item(
        source_id=source_id,
        source_type="github_repo",
        run_id="run-v10-contract",
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
        source=source_id,
        source_url=item.source_url,
        title=item.title,
        space_or_project="TEST",
        author=None,
        last_modified=now,
        labels=[],
        version="1",
        content_hash=hashlib.sha256(body.encode()).hexdigest(),
        token_count=12,
        raw_content_uri=None,
        raw_content_type="text/markdown",
        normalized_content_uri=None,
        pdf_content_uri=None,
        last_synced=now,
    )
    seen_batches = []

    async def extract(batch):
        seen_batches.append(batch)
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
                source_activity_epoch=None,
            ),
            extract_batch=extract,
            max_concurrent=1,
            extraction_contract_version=(
                active_projection_extraction_contract(
                    SupportScopeVersion.EVIDENCE_UNIT_SET_V2
                ).version
            ),
            access_context_hash="access-v10-contract",
            inference_capability_hash="inference-v10-contract",
        )
    )

    assert result.derivation.extraction_contract_version == v10.version
    assert seen_batches
    assert {batch.__class__.__name__ for batch in seen_batches} == {
        "ProjectionExtractionBatch"
    }


def test_unknown_projection_extraction_contract_fails_closed() -> None:
    with pytest.raises(
        ValueError,
        match="unknown projection extraction contract",
    ):
        projection_extraction_contract("projection-extraction-v999")


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
        raw_content_uri=None,
        raw_content_type="text/markdown",
        normalized_content_uri=None,
        pdf_content_uri=None,
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
        source_activity_epoch=1,
    )
    batches = plan_projection_evidence_work(
        projection,
        reprocess_all_current_observations=False,
        extraction_contract_version=PROJECTION_EXTRACTION_V9,
    )
    assert isinstance(batches, tuple)

    first = source_derivation_manifest(
        projection,
        batches,
        context=context,
        extraction_contract_version=PROJECTION_EXTRACTION_V9,
        evidence_work_identity_hash="a" * 64,
    )
    changed = source_derivation_manifest(
        projection,
        batches,
        context=context,
        extraction_contract_version=PROJECTION_EXTRACTION_V9,
        evidence_work_identity_hash="b" * 64,
    )

    assert first.id != changed.id
    assert first.batches[0].input_payload_hash != (
        changed.batches[0].input_payload_hash
    )

    reprocess_batches = plan_projection_evidence_work(
        projection,
        reprocess_all_current_observations=True,
        extraction_contract_version=PROJECTION_EXTRACTION_V9,
    )
    assert isinstance(reprocess_batches, tuple)
    first_operation = source_derivation_manifest(
        projection,
        reprocess_batches,
        context=replace(
            context,
            reprocess_all_current_observations=True,
            reprocess_operation_id="sync-run-1",
        ),
        extraction_contract_version=PROJECTION_EXTRACTION_V9,
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
        extraction_contract_version=PROJECTION_EXTRACTION_V9,
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
        raw_content_uri=None,
        raw_content_type="text/markdown",
        normalized_content_uri=None,
        pdf_content_uri=None,
        last_synced=now,
    )
    extractor_called = False

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
                source_activity_epoch=1,
                reprocess_all_current_observations=True,
            ),
            extract_batch=extract,
            max_concurrent=1,
            extraction_contract_version=PROJECTION_EXTRACTION_V9,
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
        raw_content_uri=None,
        raw_content_type="text/markdown",
        normalized_content_uri=None,
        pdf_content_uri=None,
        last_synced=now,
    )
    extractor_called = False

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
            source_activity_epoch=1,
            current_changed_ranges=((0, 7),),
        ),
        extract_batch=extract,
        max_concurrent=1,
        extraction_contract_version=PROJECTION_EXTRACTION_V9,
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
    assert result.derivation.status == "completed"
    assert (
        result.derivation.terminal_reason_code
        == "INCREMENTAL_BASE_UNAVAILABLE"
    )
    assert result.derivation.authority_plan_identity == {
        "access_context_hash": "access-unmappable-authority",
        "authority_policy_version": 5,
        "base_unit_revision_id": initial.source_unit_revisions[0].id,
        "extraction_contract_version": PROJECTION_EXTRACTION_V9,
        "inference_capability_hash": "inference-unmappable-authority",
        "representation_profiles": [
            {
                "coordinate_space": "unicode-scalar",
                "name": "markdown-structural",
                "observation_id": target.observations[0].id,
                "schema_name": None,
                "schema_version": None,
                "version": 1,
            }
        ],
        "reprocess_operation_id": None,
        "source_activity_epoch": 1,
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
