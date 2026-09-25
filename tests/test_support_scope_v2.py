from __future__ import annotations

from tests.revision_client_fixture import RevisionClientFixture

import hashlib
import json
import sqlite3
from datetime import datetime, timezone

import pytest
import pytest_asyncio

from memforge.memory.evidence import (
    EvidenceContentProvenance,
    EvidencePartKind,
    EvidenceReference,
    EvidenceRole,
    EvidenceUnit,
    SupportScopeVersion,
    evidence_part_set_digest,
    evidence_unit_id_v2,
    evidence_unit_revision_lineage_is_valid,
)
from memforge.memory.lifecycle_plan import ReconciliationScope
from memforge.memory.lifecycle_planner import NewMemoryDefaults, build_lifecycle_plan
from memforge.models import (
    ContentItem,
    DocumentRecord,
    Memory,
    MemoryExtractionResult,
    NormalizedContent,
    RawContent,
    ReconcileAction,
    ReconcileOperation,
    content_hash,
)
from memforge.llm.structured import (
    ProjectionFragmentMemoryCandidate,
    ProjectionFragmentMemoryExtractionResponse,
)
from memforge.pipeline.memory_extractor import MemoryExtractor
from memforge.pipeline.projection_context import plan_projection_extraction_batches
from memforge.pipeline.projection_evidence import build_projected_claim_evidence
from memforge.pipeline.projection_fragments import compile_projection_fragment_catalog
from memforge.pipeline.source_projection_adapters import project_source_item
from memforge.source_projection import AnchorKind, SourceAnchor
from memforge.source_access import source_is_discoverable
from memforge.storage.database import Database
from tests.unit_support_fixture import record_unit_support
from memforge.source_derivation import (
    SourceUnitDerivationContext,
    SourceUnitDerivationRequest,
    SourceUnitDeriver,
)


@pytest_asyncio.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "support-v2.db"))
    await database.connect()
    try:
        yield database
    finally:
        await database.close()


async def _seed_complete_unit_support(db: Database) -> tuple[str, str, str, str]:
    now = datetime(2026, 8, 27, 8, 0, tzinfo=timezone.utc).isoformat()
    source_id = "source-1"
    memory_id = "memory-1"
    unit_id = "unit-1"
    unit_revision_id = "unitrev-1"
    evidence_unit_id = "evidence-unit-1"
    access_hash = "access-1"
    await db.upsert_source(
        id=source_id,
        type="github_repo",
        name="Repository",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="owner-1",
    )
    await db.db.execute(
        """INSERT INTO documents (
               doc_id, source, source_url, title, space_or_project,
               last_modified, version, content_hash, last_synced
           ) VALUES ('doc-1', ?, 'https://example.test/repo', 'Repo', 'TEST',
                     ?, '1', 'doc-hash', ?)""",
        (source_id, now, now),
    )
    await db.insert_memory(
        Memory(
            id=memory_id,
            memory_type="convention",
            content="Release requires approval under the documented condition.",
            content_hash=content_hash(
                "Release requires approval under the documented condition."
            ),
        )
    )
    await db.db.execute(
        """INSERT INTO source_units (
               id, source_id, unit_type, provider_key, locator_json,
               current_revision_id, updated_at
           ) VALUES (?, ?, 'document', 'doc-1', '{}', ?, ?)""",
        (unit_id, source_id, unit_revision_id, now),
    )
    observation_specs = (
        ("obs-primary", "obsrev-primary", "Release requires approval."),
        ("obs-required", "obsrev-required", "Only after two reviewers agree."),
    )
    for observation_id, revision_id, text in observation_specs:
        await db.db.execute(
            """INSERT INTO source_observations (
                   id, source_id, source_unit_id, observation_type,
                   provider_key, locator_json, current_revision_id, updated_at
               ) VALUES (?, ?, ?, 'document_body', ?, '{}', ?, ?)""",
            (observation_id, source_id, unit_id, observation_id, revision_id, now),
        )
        await db.db.execute(
            """INSERT INTO source_observation_revisions (
                   id, observation_id, semantic_hash, content, metadata_json,
                   observed_at, profile_name, profile_version, coordinate_space,
                   created_at
               ) VALUES (?, ?, ?, ?, '{}', ?, 'markdown-structural', 1,
                         'unicode-scalar', ?)""",
            (
                revision_id,
                observation_id,
                hashlib.sha256(text.encode()).hexdigest(),
                text,
                now,
                now,
            ),
        )
    await db.db.execute(
        """INSERT INTO source_unit_revisions (
               id, source_unit_id, semantic_hash, access_hash,
               observation_revision_ids_json, observed_at, created_at
           ) VALUES (?, ?, 'unit-hash', ?, ?, ?, ?)""",
        (
            unit_revision_id,
            unit_id,
            access_hash,
            json.dumps([item[1] for item in observation_specs]),
            now,
            now,
        ),
    )
    await record_unit_support(
        db,
        memory_id=memory_id,
        unit=EvidenceUnit(
            id=evidence_unit_id,
            source_id=source_id,
            doc_id="doc-1",
            doc_revision_id=unit_revision_id,
            source_type="github_repo",
            source_anchor="obs-primary",
            source_lineage_id=unit_id,
            project_key=None,
            visibility="workspace",
            owner_user_id=None,
            repo_identifier=None,
            content="Release requires approval.",
            excerpt="Release requires approval.",
            evidence_provenance=EvidenceContentProvenance.SOURCE_EXCERPT,
            access_context_hash=access_hash,
        ),
        references=(
            EvidenceReference(
                id="eref-primary",
                role=EvidenceRole.PRIMARY,
                anchor=SourceAnchor(
                    kind=AnchorKind.WHOLE_OBSERVATION,
                    observation_id="obs-primary",
                    observation_revision_id="obsrev-primary",
                ),
            ),
            EvidenceReference(
                id="eref-required",
                role=EvidenceRole.REQUIRED,
                anchor=SourceAnchor(
                    kind=AnchorKind.WHOLE_OBSERVATION,
                    observation_id="obs-required",
                    observation_revision_id="obsrev-required",
                ),
            ),
        ),
    )
    await db.db.execute(
        """INSERT INTO memory_sources (
               memory_id, doc_id, source_id, source_type, excerpt,
               support_kind, added_at
           ) VALUES (?, 'doc-1', ?, 'github_repo',
                     'Release requires approval.', 'extracted', ?)""",
        (memory_id, source_id, now),
    )
    await db.db.commit()
    return memory_id, evidence_unit_id, source_id, access_hash


_REQUIRE_EVIDENCE_UNIT_SUPPORT_MIGRATION = 97


async def _support_scope_marker(db: Database) -> str:
    [row] = await db.db.execute_fetchall(
        "SELECT marker_value FROM system_contract_markers WHERE marker_key = 'support_scope_version'"
    )
    return str(row["marker_value"])


def _mark_reference_scoped_before_upgrade(path: str, *, reference_support_row: bool) -> None:
    """Rewind a closed workspace to the state an earlier version left behind."""

    with sqlite3.connect(path) as connection:
        connection.execute(
            """UPDATE system_contract_markers SET marker_value = 'reference-set-v1'
                WHERE marker_key = 'support_scope_version'"""
        )
        if reference_support_row:
            connection.execute(
                """INSERT INTO memory_support_assertions (
                       id, memory_id, evidence_reference_id, source_id,
                       access_context_hash, active, created_at
                   ) VALUES ('reference-support', 'memory-1', 'eref-primary',
                             'source-1', 'access-1', 1, '2026-08-27T08:00:00+00:00')"""
            )
        connection.execute(
            "DELETE FROM schema_migrations WHERE version = ?",
            (_REQUIRE_EVIDENCE_UNIT_SUPPORT_MIGRATION,),
        )


@pytest.mark.asyncio
async def test_new_workspace_starts_on_evidence_unit_support(db) -> None:
    assert await _support_scope_marker(db) == "evidence-unit-set-v2"


@pytest.mark.asyncio
async def test_workspace_without_reference_scoped_support_moves_to_evidence_unit_support(db) -> None:
    await _seed_complete_unit_support(db)
    await db.close()
    _mark_reference_scoped_before_upgrade(db.db_path, reference_support_row=False)

    await db.connect()

    assert await _support_scope_marker(db) == "evidence-unit-set-v2"
    assert await db.get_active_memory_support_unit_ids("memory-1") == ("evidence-unit-1",)


@pytest.mark.asyncio
async def test_workspace_with_reference_scoped_support_refuses_to_start(db) -> None:
    await _seed_complete_unit_support(db)
    await db.close()
    _mark_reference_scoped_before_upgrade(db.db_path, reference_support_row=True)

    with pytest.raises(RuntimeError, match="reference-scoped Support"):
        await db.connect()
    await db.close()

    with sqlite3.connect(db.db_path) as connection:
        [(marker,)] = connection.execute(
            "SELECT marker_value FROM system_contract_markers WHERE marker_key = 'support_scope_version'"
        ).fetchall()
        applied = connection.execute(
            "SELECT 1 FROM schema_migrations WHERE version = ?",
            (_REQUIRE_EVIDENCE_UNIT_SUPPORT_MIGRATION,),
        ).fetchall()
    assert marker == "reference-set-v1"
    assert applied == []


@pytest.mark.asyncio
async def test_unknown_support_scope_marker_refuses_to_start(db) -> None:
    await db.close()
    with sqlite3.connect(db.db_path) as connection:
        connection.execute(
            """UPDATE system_contract_markers SET marker_value = 'unknown-scope'
                WHERE marker_key = 'support_scope_version'"""
        )

    with pytest.raises(RuntimeError, match="support_scope_version marker is unknown"):
        await db.connect()
    await db.close()


def test_evidence_unit_revision_lineage_predicate_covers_identity_and_membership() -> None:
    valid = {
        "evidence_unit_source_lineage_id": "unit-1",
        "unit_revision_source_unit_id": "unit-1",
        "unit_revision_observation_revision_ids": ("rev-1", "rev-2"),
    }

    assert evidence_unit_revision_lineage_is_valid(
        **valid,
        reference_observation_revision_ids=("rev-1",),
    )
    assert not evidence_unit_revision_lineage_is_valid(
        **(valid | {"unit_revision_source_unit_id": "unit-2"}),
        reference_observation_revision_ids=("rev-1",),
    )
    assert not evidence_unit_revision_lineage_is_valid(
        **valid,
        reference_observation_revision_ids=("rev-3",),
    )


def test_v2_part_and_support_identity_exclude_presentation_only_changes() -> None:
    anchor = SourceAnchor(
        kind=AnchorKind.REVISION_RANGE,
        observation_id="obs-1",
        observation_revision_id="rev-1",
        range_start=0,
        range_end=4,
    )
    first = EvidenceReference(
        role=EvidenceRole.PRIMARY,
        anchor=anchor,
        kind=EvidencePartKind.TEXT,
        raw_content_sha256="a" * 64,
        presentation_sha256="b" * 64,
        excerpt="text",
    )
    second = EvidenceReference(
        role=EvidenceRole.PRIMARY,
        anchor=anchor,
        kind=EvidencePartKind.TEXT,
        raw_content_sha256="a" * 64,
        presentation_sha256="c" * 64,
        excerpt="rendered differently",
    )
    assert evidence_part_set_digest((first,)) == evidence_part_set_digest((second,))
    digest = evidence_part_set_digest((first,))
    assert evidence_unit_id_v2(
        source_unit_id="unit-1",
        claim_content="Claim",
        part_set_digest=digest,
        access_context_hash="access-1",
    ).startswith("eu-v2-")
    required_whole = EvidenceReference(
        role=EvidenceRole.REQUIRED,
        anchor=SourceAnchor(
            kind=AnchorKind.WHOLE_OBSERVATION,
            observation_id="obs-2",
            observation_revision_id="rev-2",
        ),
        kind=EvidencePartKind.TEXT,
        raw_content_sha256="d" * 64,
        presentation_sha256="e" * 64,
        excerpt="whole",
    )
    required_range = EvidenceReference(
        role=EvidenceRole.REQUIRED,
        anchor=SourceAnchor(
            kind=AnchorKind.REVISION_RANGE,
            observation_id="obs-2",
            observation_revision_id="rev-2",
            range_start=0,
            range_end=5,
        ),
        kind=EvidencePartKind.TEXT,
        raw_content_sha256="f" * 64,
        presentation_sha256="0" * 64,
        excerpt="range",
    )
    assert len(
        evidence_part_set_digest((first, required_whole, required_range))
    ) == 64


@pytest.mark.asyncio
async def test_v2_lifecycle_removes_one_complete_unit_then_retires_last_support(db) -> None:
    memory_id, unit_id, source_id, access_hash = await _seed_complete_unit_support(db)
    await db.enable_lifecycle_gate(source_id)
    memory = await db.get_memory(memory_id)
    assert memory is not None
    states = await db.get_active_memory_support_states((memory_id,))
    state = states[memory_id]
    assert state.support_scope_version is SupportScopeVersion.EVIDENCE_UNIT_SET_V2
    assert state.unit_ids == (unit_id,)
    scope = ReconciliationScope(
        id="scope-v2-delete",
        source_id=source_id,
        source_unit_id="unit-1",
        base_unit_revision_id="unitrev-1",
        target_unit_revision_id="unitrev-1",
    )
    plan = build_lifecycle_plan(
        plan_id="plan-v2-delete",
        scope=scope,
        gate_state=(await db.get_lifecycle_gate(source_id)).state,
        operations=(
            ReconcileOperation(
                action=ReconcileAction.DELETE,
                memory_id=memory_id,
                reason="authoritative source removed the claim",
            ),
        ),
        incumbents={memory_id: memory},
        source_support_reference_ids={},
        all_active_support_reference_ids={},
        support_set_hashes={memory_id: state.support_set_hash},
        observation_revision_ids=(),
        new_evidence_reference_ids=(),
        support_scope_version=SupportScopeVersion.EVIDENCE_UNIT_SET_V2,
        source_support_unit_ids={memory_id: (unit_id,)},
        all_active_support_unit_ids={memory_id: (unit_id,)},
        defaults=NewMemoryDefaults(
            visibility="workspace",
            owner_user_id=None,
            project_key=None,
            repo_identifier=None,
            doc_id="doc-1",
            source_type="github_repo",
            access_context_hash=access_hash,
        ),
    )
    await db.apply_lifecycle_plan(plan)
    retired = await db.get_memory(memory_id)
    assert retired is not None and retired.status == "retired"
    assert await db.get_active_memory_support_unit_ids(memory_id) == ()


@pytest.mark.asyncio
async def test_v2_source_removal_retires_last_support_without_deleting_history(db) -> None:
    memory_id, unit_id, source_id, _access_hash = await _seed_complete_unit_support(db)

    result = await db.delete_source_cascade(source_id)

    assert result.retired_memory_ids == (memory_id,)
    source = await db.get_source(source_id)
    assert source is not None
    assert source["status"] == "retired"
    assert source["sync_schedule"]["enabled"] is False
    assert source_is_discoverable(source, viewer_id="owner-1") is False
    memory = await db.get_memory(memory_id)
    assert memory is not None
    assert memory.status == "retired"
    assert memory.retirement_reason == "source_deleted"
    assert await db.get_memory_sources(memory_id) == []
    assert await db.get_evidence_unit(unit_id) is not None
    support_rows = await db.db.execute_fetchall(
        "SELECT active, removed_at FROM memory_unit_support_assertions WHERE memory_id = ?",
        (memory_id,),
    )
    assert [(row["active"], bool(row["removed_at"])) for row in support_rows] == [(0, True)]
    assert await db.db.execute_fetchall(
        "SELECT id FROM evidence_references WHERE evidence_unit_id = ?",
        (unit_id,),
    )
    assert await db.db.execute_fetchall(
        "SELECT id FROM source_units WHERE source_id = ?",
        (source_id,),
    )


@pytest.mark.asyncio
async def test_v2_source_removal_preserves_memory_with_independent_support(db) -> None:
    memory_id, unit_id, source_id, _access_hash = await _seed_complete_unit_support(db)
    now = datetime(2026, 8, 27, 9, 0, tzinfo=timezone.utc).isoformat()
    await db.upsert_source(
        id="source-2",
        type="jira",
        name="Independent source",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="owner-2",
    )
    await db.db.execute(
        """INSERT INTO evidence_units (
               id, source_id, doc_id, doc_revision_id, source_type,
               source_anchor, source_lineage_id, source_metadata_json,
               visibility, access_context_hash, content, excerpt,
               evidence_provenance, part_set_digest, created_at, updated_at
           ) VALUES ('evidence-unit-2', 'source-2', 'doc-1', NULL, 'jira',
                     'PAY-2-description', 'unit-2', '{}', 'workspace',
                     'access-2', 'Independent support', 'Independent support',
                     'source_excerpt', 'part-digest-2', ?, ?)""",
        (now, now),
    )
    await db.db.execute(
        """INSERT INTO memory_unit_support_assertions (
               id, memory_id, evidence_unit_id, source_id,
               access_context_hash, active, created_at
           ) VALUES ('support-v2-independent', ?, 'evidence-unit-2',
                     'source-2', 'access-2', 1, ?)""",
        (memory_id, now),
    )
    await db.db.execute(
        """INSERT INTO memory_sources (
               memory_id, doc_id, source_id, source_type, excerpt,
               support_kind, added_at
           ) VALUES (?, 'doc-1', 'source-2', 'jira', 'Independent support',
                     'corroborated', ?)""",
        (memory_id, now),
    )
    await db.db.commit()

    result = await db.delete_source_cascade(source_id)

    assert result.retired_memory_ids == ()
    memory = await db.get_memory(memory_id)
    assert memory is not None and memory.status == "active"
    assert await db.get_active_memory_support_unit_ids(memory_id) == (
        "evidence-unit-2",
    )
    assert [(item.source_id, item.doc_id) for item in await db.get_memory_sources(memory_id)] == [
        ("source-2", "doc-1"),
    ]
    document = await db.get_document("doc-1")
    assert document is not None and document.source == "source-2"
    assert await db.get_evidence_unit(unit_id) is not None


@pytest.mark.asyncio
async def test_v9_fragment_selection_commits_one_complete_unit_support(db) -> None:
    await _seed_complete_unit_support(db)
    await db.enable_lifecycle_gate("source-1")
    now = datetime(2026, 8, 27, 9, 0, tzinfo=timezone.utc)
    body = "# Deployment rule\n\nDeploy only after approval.\n"
    item = ContentItem(
        item_id="doc-2",
        title="Deployment",
        source_url="https://example.test/repo/deployment.md",
        last_modified=now,
        content_type="text/markdown",
        version="1",
    )
    projection = project_source_item(
        source_id="source-1",
        source_type="github_repo",
        run_id="run-v9",
        item=item,
        raw=RawContent(item=item, body=body.encode(), content_type="text/markdown"),
        normalized=NormalizedContent(item=item, markdown_body=body),
        scope={},
        access_context={"visibility": "workspace"},
    )
    await db.record_source_projection(projection)
    await db.upsert_document(
        DocumentRecord(
            doc_id="doc-2",
            source="source-1",
            source_url=item.source_url,
            title=item.title,
            space_or_project="TEST",
            author=None,
            last_modified=now,
            labels=[],
            version="1",
            content_hash=hashlib.sha256(body.encode()).hexdigest(),
            token_count=10,
            raw_content_uri=None,
            raw_content_type="text/markdown",
            normalized_content_uri=None,
            pdf_content_uri=None,
            last_synced=now,
        )
    )
    batch = plan_projection_extraction_batches(
        projection,
        extraction_contract_version="projection-extraction-v9",
    )[0]
    access_hash = hashlib.sha256("workspace\x1f\x1f".encode()).hexdigest()
    catalog = compile_projection_fragment_catalog(
        projection,
        batch,
        access_context_hash=access_hash,
    )
    primary = next(
        fragment
        for fragment in catalog.fragments
        if "approval" in fragment.presentation_text.lower()
        and fragment.primary_eligible
    )

    class Client(RevisionClientFixture):
        def request_fits(self, *args, **kwargs):
            return True

        async def extract_projection_fragment_memories(self, prompt: str, **kwargs):
            return ProjectionFragmentMemoryExtractionResponse(
                memories=[
                    ProjectionFragmentMemoryCandidate(
                        content="Deployment is allowed only after approval.",
                        memory_type="convention",
                        primary_ref=primary.reference,
                        required_refs=[],
                    )
                ]
            )

    extraction = await MemoryExtractor(
        structured_llm_client=Client()
    ).extract_projection_fragment_memories(
        catalog,
        source_type="github_repo",
    )
    assert extraction.error_type is None
    raw = extraction.memories[0]
    evidence = build_projected_claim_evidence(
        projection=projection,
        raw_memories=(raw,),
        doc_id="doc-2",
        source_type="github_repo",
        project_key=None,
        visibility="workspace",
        owner_user_id=None,
        repo_identifier=None,
        access_context_hash=access_hash,
        extractor_run_id="run-v9",
        observed_at=now.isoformat(),
        support_scope_version=SupportScopeVersion.EVIDENCE_UNIT_SET_V2,
    )
    claim_hash = content_hash(raw.content.strip())
    scope = ReconciliationScope(
        id="scope-v9-add",
        source_id="source-1",
        source_unit_id=projection.source_units[0].id,
        base_unit_revision_id=None,
        target_unit_revision_id=projection.source_unit_revisions[0].id,
    )
    plan = build_lifecycle_plan(
        plan_id="plan-v9-add",
        scope=scope,
        gate_state=(await db.get_lifecycle_gate("source-1")).state,
        operations=(
            ReconcileOperation(
                action=ReconcileAction.ADD,
                memory=evidence.canonical_memories_by_claim_hash[claim_hash],
                reason="new durable rule",
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
        support_scope_version=SupportScopeVersion.EVIDENCE_UNIT_SET_V2,
        evidence_unit_ids_by_claim_hash=evidence.evidence_unit_ids_by_claim_hash,
        evidence_units=evidence.units,
        evidence_references=evidence.references,
        defaults=NewMemoryDefaults(
            visibility="workspace",
            owner_user_id=None,
            project_key=None,
            repo_identifier=None,
            doc_id="doc-2",
            source_type="github_repo",
            access_context_hash=access_hash,
        ),
    )
    await db.apply_lifecycle_plan(plan)
    created = [
        memory
        for memory in await db.list_memories(status="active")
        if memory.content == "Deployment is allowed only after approval."
    ]
    assert len(created) == 1
    unit_ids = await db.get_active_memory_support_unit_ids(created[0].id)
    assert unit_ids == evidence.evidence_unit_ids_by_claim_hash[claim_hash]
    parts = await db.get_active_memory_support_evidence(created[0].id)
    assert len(parts) == 1 and parts[0].role is EvidenceRole.PRIMARY

    updated_body = "# Deployment rule\n\nDeploy only after two approvals.\n"
    updated_item = ContentItem(
        item_id="doc-2",
        title="Deployment",
        source_url=item.source_url,
        last_modified=now.replace(hour=10),
        content_type="text/markdown",
        version="2",
    )
    updated_projection = project_source_item(
        source_id="source-1",
        source_type="github_repo",
        run_id="run-v9-update",
        item=updated_item,
        raw=RawContent(
            item=updated_item,
            body=updated_body.encode(),
            content_type="text/markdown",
        ),
        normalized=NormalizedContent(
            item=updated_item,
            markdown_body=updated_body,
        ),
        scope={},
        access_context={"visibility": "workspace"},
        prior_unit_revision=projection.source_unit_revisions[0],
        prior_observation_revisions={
            revision.observation_id: revision
            for revision in projection.observation_revisions
        },
    )
    await db.record_source_projection(updated_projection)
    updated_batch = plan_projection_extraction_batches(
        updated_projection,
        extraction_contract_version="projection-extraction-v9",
    )[0]
    updated_catalog = compile_projection_fragment_catalog(
        updated_projection,
        updated_batch,
        access_context_hash=access_hash,
    )
    updated_primary = next(
        fragment
        for fragment in updated_catalog.fragments
        if "two approvals" in fragment.presentation_text.lower()
        and fragment.primary_eligible
    )

    class UpdatedClient(RevisionClientFixture):
        def request_fits(self, *args, **kwargs):
            return True

        async def extract_projection_fragment_memories(self, prompt: str, **kwargs):
            return ProjectionFragmentMemoryExtractionResponse(
                memories=[
                    ProjectionFragmentMemoryCandidate(
                        content="Deployment is allowed only after two approvals.",
                        memory_type="convention",
                        primary_ref=updated_primary.reference,
                        required_refs=[],
                    )
                ]
            )

    updated_extraction = await MemoryExtractor(
        structured_llm_client=UpdatedClient()
    ).extract_projection_fragment_memories(
        updated_catalog,
        source_type="github_repo",
    )
    updated_raw = updated_extraction.memories[0]
    updated_evidence = build_projected_claim_evidence(
        projection=updated_projection,
        raw_memories=(updated_raw,),
        doc_id="doc-2",
        source_type="github_repo",
        project_key=None,
        visibility="workspace",
        owner_user_id=None,
        repo_identifier=None,
        access_context_hash=access_hash,
        extractor_run_id="run-v9-update",
        observed_at=updated_item.last_modified.isoformat(),
        support_scope_version=SupportScopeVersion.EVIDENCE_UNIT_SET_V2,
    )
    old_memory = created[0]
    old_state = (
        await db.get_active_memory_support_states((old_memory.id,))
    )[old_memory.id]
    old_unit_ids = old_state.unit_ids
    updated_claim_hash = content_hash(updated_raw.content.strip())
    update_scope = ReconciliationScope(
        id="scope-v9-update",
        source_id="source-1",
        source_unit_id=updated_projection.source_units[0].id,
        base_unit_revision_id=projection.source_unit_revisions[0].id,
        target_unit_revision_id=updated_projection.source_unit_revisions[0].id,
    )
    update_plan = build_lifecycle_plan(
        plan_id="plan-v9-update",
        scope=update_scope,
        gate_state=(await db.get_lifecycle_gate("source-1")).state,
        operations=(
            ReconcileOperation(
                action=ReconcileAction.UPDATE,
                memory_id=old_memory.id,
                memory=updated_evidence.canonical_memories_by_claim_hash[
                    updated_claim_hash
                ],
                reason="approval rule was revised",
            ),
        ),
        incumbents={old_memory.id: old_memory},
        source_support_reference_ids={},
        all_active_support_reference_ids={},
        support_set_hashes={old_memory.id: old_state.support_set_hash},
        observation_revision_ids=tuple(
            revision.id for revision in updated_projection.observation_revisions
        ),
        new_evidence_reference_ids=(),
        support_scope_version=SupportScopeVersion.EVIDENCE_UNIT_SET_V2,
        source_support_unit_ids={old_memory.id: old_unit_ids},
        all_active_support_unit_ids={old_memory.id: old_unit_ids},
        evidence_unit_ids_by_claim_hash=(
            updated_evidence.evidence_unit_ids_by_claim_hash
        ),
        evidence_units=updated_evidence.units,
        evidence_references=updated_evidence.references,
        defaults=NewMemoryDefaults(
            visibility="workspace",
            owner_user_id=None,
            project_key=None,
            repo_identifier=None,
            doc_id="doc-2",
            source_type="github_repo",
            access_context_hash=access_hash,
        ),
    )
    await db.apply_lifecycle_plan(update_plan)
    superseded = await db.get_memory(old_memory.id)
    assert superseded is not None and superseded.status == "superseded"
    replacements = [
        memory
        for memory in await db.list_memories(status="active")
        if memory.content == "Deployment is allowed only after two approvals."
    ]
    assert len(replacements) == 1
    assert await db.get_active_memory_support_unit_ids(old_memory.id) == ()
    assert await db.get_active_memory_support_unit_ids(replacements[0].id) == (
        updated_evidence.evidence_unit_ids_by_claim_hash[updated_claim_hash]
    )


@pytest.mark.asyncio
async def test_context_replacement_does_not_change_unit_support_identity_or_hash(db) -> None:
    memory_id, unit_id, source_id, _access_hash = await _seed_complete_unit_support(db)
    before = (await db.get_active_memory_support_states((memory_id,)))[memory_id]
    now = datetime(2026, 8, 27, 10, 0, tzinfo=timezone.utc).isoformat()
    for suffix in ("one", "two"):
        observation_id = f"obs-context-{suffix}"
        revision_id = f"obsrev-context-{suffix}"
        text = f"Helpful context {suffix}."
        await db.db.execute(
            """INSERT INTO source_observations (
                   id, source_id, source_unit_id, observation_type,
                   provider_key, locator_json, current_revision_id, updated_at
               ) VALUES (?, ?, 'unit-1', 'document_context', ?, '{}', ?, ?)""",
            (observation_id, source_id, observation_id, revision_id, now),
        )
        await db.db.execute(
            """INSERT INTO source_observation_revisions (
                   id, observation_id, semantic_hash, content, metadata_json,
                   observed_at, profile_name, profile_version, coordinate_space,
                   created_at
               ) VALUES (?, ?, ?, ?, '{}', ?, 'plain-text', 1,
                         'unicode-scalar', ?)""",
            (
                revision_id,
                observation_id,
                hashlib.sha256(text.encode()).hexdigest(),
                text,
                now,
                now,
            ),
        )
        await db.db.commit()
        await db.replace_evidence_context_associations(
            unit_id,
            (
                EvidenceReference(
                    role=EvidenceRole.CONTEXT,
                    anchor=SourceAnchor(
                        kind=AnchorKind.WHOLE_OBSERVATION,
                        observation_id=observation_id,
                        observation_revision_id=revision_id,
                    ),
                ),
            ),
        )
    associations = await db.db.execute_fetchall(
        """SELECT active, removed_at FROM evidence_context_associations
           WHERE evidence_unit_id = ? ORDER BY created_at, id""",
        (unit_id,),
    )
    assert len(associations) == 2
    assert sorted(int(row["active"]) for row in associations) == [0, 1]
    assert next(row for row in associations if not row["active"])["removed_at"]
    after = (await db.get_active_memory_support_states((memory_id,)))[memory_id]
    assert after.unit_ids == before.unit_ids == (unit_id,)
    assert after.support_set_hash == before.support_set_hash
    [group] = await db.get_memory_evidence_units(memory_id)
    assert [item.role for item in group.items] == [
        EvidenceRole.PRIMARY,
        EvidenceRole.REQUIRED,
        EvidenceRole.CONTEXT,
    ]
    assert group.items[-1].grants_support is False
    assert group.items[-1].anchor.observation_id == "obs-context-two"
    await db.db.execute(
        """UPDATE evidence_references
              SET raw_content_sha256 = ?
            WHERE id = ?""",
        ("0" * 64, group.items[-1].reference_id),
    )
    await db.db.commit()
    [without_bad_context] = await db.get_memory_evidence_units(memory_id)
    assert [item.role for item in without_bad_context.items] == [
        EvidenceRole.PRIMARY,
        EvidenceRole.REQUIRED,
    ]


@pytest.mark.asyncio
async def test_invalid_supporting_part_omits_complete_evidence_unit(db) -> None:
    memory_id, _unit_id, _source_id, _access_hash = (
        await _seed_complete_unit_support(db)
    )
    await db.db.execute(
        """UPDATE evidence_references
              SET raw_content_sha256 = ?
            WHERE id = 'eref-required'""",
        ("0" * 64,),
    )
    await db.db.commit()
    assert await db.get_memory_evidence_units(memory_id) == ()


@pytest.mark.asyncio
async def test_v2_deriver_stages_projection_extraction_v9_without_ingestion_replay(db) -> None:
    await _seed_complete_unit_support(db)
    now = datetime(2026, 8, 27, 12, 0, tzinfo=timezone.utc)
    body = "# Durable rule\n\nAlways validate the complete Evidence Unit.\n"
    item = ContentItem(
        item_id="doc-v9-derivation",
        title="Evidence rule",
        source_url="https://example.test/repo/evidence.md",
        last_modified=now,
        content_type="text/markdown",
        version="1",
    )
    projection = project_source_item(
        source_id="source-1",
        source_type="github_repo",
        run_id="run-v9-derivation",
        item=item,
        raw=RawContent(item=item, body=body.encode(), content_type="text/markdown"),
        normalized=NormalizedContent(item=item, markdown_body=body),
        scope={},
        access_context={"visibility": "workspace"},
    )
    await db.record_source_projection(projection)
    document = DocumentRecord(
        doc_id=item.item_id,
        source="source-1",
        source_url=item.source_url,
        title=item.title,
        space_or_project="TEST",
        author=None,
        last_modified=now,
        labels=[],
        version="1",
        content_hash=hashlib.sha256(body.encode()).hexdigest(),
        token_count=10,
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
            extraction_contract_version="projection-extraction-v9",
            access_context_hash="access-support-v2",
            inference_capability_hash="inference-support-v2",
        )
    )
    assert len(seen_batches) == 1
    assert seen_batches[0].__class__.__name__ == "ProjectionExtractionBatch"
    assert result.derivation.extraction_contract_version == "projection-extraction-v9"
    assert result.derivation.target_unit_revision_id == projection.source_unit_revisions[0].id

