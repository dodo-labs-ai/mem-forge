from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import pytest
import pytest_asyncio

from memforge.memory.evidence import (
    EvidenceContentProvenance,
    EvidenceReference,
    EvidenceRole,
    EvidenceUnit,
    MemorySupportAssertion,
)
from memforge.memory.lifecycle_plan import (
    CutoverFindingReason,
    CutoverFindingStatus,
    LifecycleCutoverFinding,
    LifecycleVectorOperation,
    HistoricalProjectionFailureReason,
)
from memforge.memory.cutover import (
    HistoricalProjectionUnavailable,
    reconstruct_historical_source_projection,
)
from memforge.models import (
    ContentItem,
    DocumentRecord,
    Memory,
    NormalizedContent,
    RawContent,
    content_hash,
)
from memforge.pipeline.source_projection_adapters import project_source_item
from memforge.source_projection import AnchorKind, SourceAnchor
from memforge.storage.database import Database
from memforge.storage.document_store import LocalDocumentStore


NOW = datetime(2026, 7, 16, tzinfo=timezone.utc)


@pytest_asyncio.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "unprovable-cutover.db"))
    await database.connect()
    await database.upsert_source(
        id="src-agent",
        type="agent_session",
        name="Agent Session",
        config_json="{}",
        access_policy="private",
        owner_user_id="owner-1",
    )
    await _seed_document(database, source_id="src-agent", doc_id="doc-agent-1")
    try:
        yield database
    finally:
        await database.close()


async def _seed_document(db: Database, *, source_id: str, doc_id: str) -> None:
    await db.upsert_document(
        DocumentRecord(
            doc_id=doc_id,
            source=source_id,
            source_url=f"agent-session://codex/session/{doc_id}",
            title=doc_id,
            space_or_project="memforge",
            author="codex",
            last_modified=NOW,
            labels=[],
            version="1",
            content_hash=f"hash-{doc_id}",
            token_count=10,
            raw_content_uri=None,
            raw_content_type="application/json",
            normalized_content_uri=None,
            pdf_content_uri=None,
            last_synced=NOW,
            client="codex",
        )
    )


async def _seed_open_finding(
    db: Database,
    *,
    memory_id: str = "mem-unprovable",
    source_id: str = "src-agent",
    doc_id: str = "doc-agent-1",
) -> LifecycleCutoverFinding:
    memory = Memory(
        id=memory_id,
        memory_type="fact",
        content="Historical Agent Session fact",
        content_hash=content_hash("Historical Agent Session fact"),
        visibility="private",
        owner_user_id="owner-1",
        project_key="memforge",
        confidence=0.9,
        created_at=NOW,
        updated_at=NOW,
        status="active",
    )
    await db.insert_memory(memory)
    await db.add_memory_source(
        memory_id,
        doc_id,
        "agent_session",
        "Historical excerpt",
        source_updated_at=None,
    )
    finding = LifecycleCutoverFinding(
        id=f"finding-{memory_id}",
        source_id=source_id,
        memory_id=memory_id,
        reason=CutoverFindingReason.MISSING_SOURCE_PROVENANCE,
        status=CutoverFindingStatus.OPEN,
        available_provenance={
            "documents": [
                {
                    "doc_id": doc_id,
                    "source_type": "agent_session",
                    "excerpt": "Historical excerpt",
                }
            ]
        },
        mapping_attempt={
            "strategy": "exact_document_locator_then_excerpt",
            "attempts": [{"doc_id": doc_id, "result": "source_unit_not_found"}],
        },
    )
    await db.upsert_lifecycle_cutover_finding(finding)
    return finding


async def _retire(db: Database, finding: LifecycleCutoverFinding):
    return await db.retire_unprovable_lifecycle_cutover_finding(
        finding.id,
        source_id=finding.source_id,
        reconstruction_attempt_id="cutover-attempt-1",
        operator_id="agent-session-lifecycle-migration",
        unavailable_documents={"doc-agent-1": "exact_inputs_missing"},
    )


@pytest.mark.asyncio
async def test_unprovable_cutover_retirement_is_atomic_auditable_and_retryable(db: Database) -> None:
    finding = await _seed_open_finding(db)

    first = await _retire(db, finding)
    second = await _retire(db, finding)

    assert second == first
    stored_memory = await db.get_memory(finding.memory_id)
    assert stored_memory is not None
    assert stored_memory.status == "retired"
    assert stored_memory.retirement_reason == "unprovable_source_lineage"
    assert first.status is CutoverFindingStatus.RESOLVED
    assert first.observation_id is None
    assert first.source_unit_id is None
    assert first.mapping_attempt["resolution"] == {
        "kind": "unprovable_source_retired",
        "operator_id": "agent-session-lifecycle-migration",
        "reconstruction_attempt_id": "cutover-attempt-1",
        "unavailable_documents": {"doc-agent-1": "exact_inputs_missing"},
    }
    [task] = await db.list_lifecycle_vector_tasks(source_id="src-agent")
    assert task.memory_id == finding.memory_id
    assert task.operation is LifecycleVectorOperation.DELETE

    with pytest.raises(ValueError, match="idempotent retirement evidence mismatch"):
        await db.retire_unprovable_lifecycle_cutover_finding(
            finding.id,
            source_id=finding.source_id,
            reconstruction_attempt_id="different-attempt",
            operator_id="agent-session-lifecycle-migration",
            unavailable_documents={"doc-agent-1": "exact_inputs_missing"},
        )


@pytest.mark.asyncio
async def test_unprovable_cutover_retirement_rejects_another_source_edge(db: Database) -> None:
    finding = await _seed_open_finding(db)
    await db.upsert_source(
        id="src-other",
        type="confluence",
        name="Other",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="owner-1",
    )
    await _seed_document(db, source_id="src-other", doc_id="doc-other")
    await db.add_memory_source(
        finding.memory_id,
        "doc-other",
        "confluence",
        source_updated_at=None,
    )

    with pytest.raises(ValueError, match="exclusive source provenance"):
        await _retire(db, finding)

    assert (await db.get_memory(finding.memory_id)).status == "active"
    assert (await db.get_lifecycle_cutover_finding(finding.id)).status is CutoverFindingStatus.OPEN


@pytest.mark.asyncio
async def test_unprovable_cutover_retirement_rejects_active_support(db: Database) -> None:
    finding = await _seed_open_finding(db)
    item = ContentItem(
        item_id="doc-agent-1",
        title="Agent Session",
        source_url="agent-session://codex/session/doc-agent-1",
        last_modified=NOW,
        version="1",
    )
    native = '{"doc_id":"doc-agent-1","markdown":"Historical excerpt","receipt":{"client":"codex"}}'
    projection = project_source_item(
        source_id="src-agent",
        source_type="agent_session",
        run_id="projection-active-support",
        item=item,
        raw=RawContent(item=item, body=native.encode(), content_type="application/json"),
        normalized=NormalizedContent(item=item, markdown_body="Historical excerpt"),
    )
    await db.record_source_projection(projection)
    observation = projection.observations[0]
    revision = projection.observation_revisions[0]
    source_unit = projection.source_units[0]
    unit = EvidenceUnit(
        id="eu-active-support",
        source_id="src-agent",
        doc_id="doc-agent-1",
        doc_revision_id=revision.id,
        source_type="agent_session",
        source_anchor=observation.id,
        source_lineage_id=source_unit.id,
        project_key="memforge",
        repo_identifier=None,
        visibility="private",
        owner_user_id="owner-1",
        content="Historical excerpt",
        excerpt="Historical excerpt",
        evidence_provenance=EvidenceContentProvenance.SOURCE_EXCERPT,
        access_context_hash="access-1",
    )
    await db.upsert_evidence_unit(unit)
    [reference] = await db.record_evidence_references(
        unit.id,
        (
            EvidenceReference(
                id="ref-active-support",
                evidence_unit_id=unit.id,
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
            id="support-active",
            memory_id=finding.memory_id,
            evidence_reference_id=reference.id or "",
            source_id="src-agent",
            access_context_hash="access-1",
        )
    )

    with pytest.raises(ValueError, match="active support"):
        await _retire(db, finding)

    assert (await db.get_memory(finding.memory_id)).status == "active"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "corrupt",
    [
        "wrong_source_type",
        "malformed_document",
        "extra_ambiguous_attempt",
    ],
)
async def test_unprovable_cutover_retirement_rejects_any_contradictory_evidence_entry(
    db: Database,
    corrupt: str,
) -> None:
    finding = await _seed_open_finding(db)
    stored = await db.get_lifecycle_cutover_finding(finding.id)
    assert stored is not None
    available = dict(stored.available_provenance)
    documents = [dict(item) for item in available["documents"]]
    mapping_attempt = dict(stored.mapping_attempt)
    attempts = [dict(item) for item in mapping_attempt["attempts"]]
    if corrupt == "wrong_source_type":
        documents[0]["source_type"] = "confluence"
    elif corrupt == "malformed_document":
        documents.append({"source_type": "agent_session"})
    else:
        attempts.append({"doc_id": "doc-agent-1", "result": "ambiguous_observation"})
    await db.db.execute(
        """UPDATE lifecycle_cutover_findings
           SET available_provenance_json = ?, mapping_attempt_json = ? WHERE id = ?""",
        (
            json.dumps({**available, "documents": documents}),
            json.dumps({**mapping_attempt, "attempts": attempts}),
            finding.id,
        ),
    )
    await db.db.commit()

    with pytest.raises(ValueError, match="strict exact source provenance"):
        await _retire(db, finding)

    assert (await db.get_memory(finding.memory_id)).status == "active"


@pytest.mark.asyncio
async def test_unprovable_cutover_retirement_rolls_back_if_finding_resolution_fails(
    db: Database,
) -> None:
    finding = await _seed_open_finding(db)
    await db.db.execute(
        """CREATE TRIGGER fail_unprovable_resolution
           BEFORE UPDATE ON lifecycle_cutover_findings
           BEGIN SELECT RAISE(ABORT, 'forced resolution failure'); END"""
    )
    await db.db.commit()

    with pytest.raises(Exception, match="forced resolution failure"):
        await _retire(db, finding)

    assert (await db.get_memory(finding.memory_id)).status == "active"
    assert (await db.get_lifecycle_cutover_finding(finding.id)).status is CutoverFindingStatus.OPEN
    assert await db.list_lifecycle_vector_tasks(source_id="src-agent") == []


@pytest.mark.asyncio
async def test_unprovable_cutover_retirement_drains_rollback_under_repeated_cancellation(
    db: Database,
    monkeypatch,
) -> None:
    finding = await _seed_open_finding(db)
    mutation_entered = asyncio.Event()
    rollback_entered = asyncio.Event()
    release_mutation = asyncio.Event()
    release_rollback = asyncio.Event()
    original_rebuild = db._rebuild_memory_fts_unlocked
    original_rollback = db.db.rollback

    async def blocked_rebuild(*args, **kwargs):
        mutation_entered.set()
        await release_mutation.wait()
        await original_rebuild(*args, **kwargs)

    async def blocked_rollback():
        rollback_entered.set()
        await release_rollback.wait()
        await original_rollback()

    monkeypatch.setattr(db, "_rebuild_memory_fts_unlocked", blocked_rebuild)
    monkeypatch.setattr(db.db, "rollback", blocked_rollback)
    task = asyncio.create_task(_retire(db, finding))
    await mutation_entered.wait()
    task.cancel()
    release_mutation.set()
    await rollback_entered.wait()
    for _ in range(5):
        task.cancel()
        await asyncio.sleep(0)
    release_rollback.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert (await db.get_memory(finding.memory_id)).status == "active"
    assert (await db.get_lifecycle_cutover_finding(finding.id)).status is CutoverFindingStatus.OPEN
    assert await db.list_lifecycle_vector_tasks(source_id="src-agent") == []
    async with db.db.execute(
        "SELECT 1 FROM lifecycle_plans WHERE source_id = ?",
        (finding.source_id,),
    ) as cursor:
        assert await cursor.fetchone() is None


@pytest.mark.asyncio
async def test_agent_session_artifact_absence_is_terminal_only_after_concept_fallback() -> None:
    document = DocumentRecord(
        doc_id="doc-agent-missing-both",
        source="src-agent",
        source_url="agent-session://codex/session/doc-agent-missing-both",
        title="Missing historical input",
        space_or_project="memforge",
        author="codex",
        last_modified=NOW,
        labels=[],
        version="1",
        content_hash="hash-missing-both",
        token_count=10,
        raw_content_uri="raw/missing.json",
        raw_content_type="application/json",
        normalized_content_uri="normalized/missing.md",
        pdf_content_uri=None,
        last_synced=NOW,
        client="codex",
    )

    class FakeDatabase:
        concept_lookups = 0

        async def get_document(self, document_id: str):
            assert document_id == document.doc_id
            return document

        async def get_agent_concept(self, document_id: str):
            assert document_id == document.doc_id
            self.concept_lookups += 1
            return None

    class MissingArtifacts:
        def read_artifact(self, _uri: str):
            raise FileNotFoundError("missing raw artifact")

    database = FakeDatabase()
    with pytest.raises(HistoricalProjectionUnavailable) as error:
        await reconstruct_historical_source_projection(
            database,
            MissingArtifacts(),
            source_id="src-agent",
            source_type="agent_session",
            document_id=document.doc_id,
        )

    assert error.value.reason is HistoricalProjectionFailureReason.EXACT_INPUTS_MISSING
    assert database.concept_lookups == 1


@pytest.mark.asyncio
async def test_missing_agent_session_document_is_reconstructed_from_canonical_concept(
    db: Database,
    tmp_path,
) -> None:
    markdown = "# Durable concept\n\nHistorical exact content"
    await db.upsert_agent_concept(
        concept_id="concept-without-document",
        source_id="src-agent",
        owner_user_id="owner-1",
        workspace="memforge",
        repo_identifier="repo-1",
        concept_type="convention",
        concept_path="conventions/durable.md",
        title="Durable concept",
        markdown_body=markdown,
        frontmatter={"source_type": "agent_session"},
        observed_at=NOW,
    )
    assert await db.get_document("concept-without-document") is None

    projection = await reconstruct_historical_source_projection(
        db,
        LocalDocumentStore(str(tmp_path / "artifacts")),
        source_id="src-agent",
        source_type="agent_session",
        document_id="concept-without-document",
    )

    rebuilt = await db.get_document("concept-without-document")
    assert rebuilt is not None
    assert rebuilt.source == "src-agent"
    assert projection.source_id == "src-agent"
    assert "Historical exact content" in projection.observation_revisions[0].content
