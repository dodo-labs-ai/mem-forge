from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from memforge.agent_knowledge import (
    AgentKnowledgeBundleService,
    AgentKnowledgePatchProposal,
    render_agent_knowledge_patch_prompt,
)
from memforge.memory.evidence import CandidateBucket, LifecycleAction, RelationType
from memforge.memory.lifecycle_service import MemoryLifecycleService
from memforge.memory.store import MemoryStore
from memforge.models import Memory, Visibility, content_hash
from memforge.storage.adapters.sqlite import build_sqlite_adapters
from memforge.storage.database import Database
from memforge.source_access_transition import SourceAccessTransitionService


class RecordingCollection:
    def __init__(self) -> None:
        self.upserted: dict[str, dict] = {}
        self.deleted: list[str] = []
        self.query_ids: list[str] = []
        self.query_distances: list[float] = []

    def query(self, **kwargs):
        return {"ids": [self.query_ids], "distances": [self.query_distances]}

    def upsert(self, *, ids, embeddings=None, metadatas=None, documents=None):
        for index, record_id in enumerate(ids):
            self.upserted[record_id] = dict(metadatas[index] if metadatas else {})

    def delete(self, *, ids):
        self.deleted.extend(ids)
        for record_id in ids:
            self.upserted.pop(record_id, None)

    def get(self, *, ids=None, include=None):
        selected = [record_id for record_id in (ids or list(self.upserted)) if record_id in self.upserted]
        out = {"ids": selected}
        if include and "metadatas" in include:
            out["metadatas"] = [self.upserted[record_id] for record_id in selected]
        if include and "embeddings" in include:
            out["embeddings"] = [[0.1] for _ in selected]
        if include and "documents" in include:
            out["documents"] = [None for _ in selected]
        return out


@pytest.fixture
async def bundle_stack(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "agent-knowledge.db"))
    await db.connect()
    await db.upsert_source(
        id="src-agent-sessions-codex",
        type="agent_session",
        name="Codex Session",
        config_json='{"client":"codex"}',
        access_policy="private",
        owner_user_id="u-andrew",
        created_by_user_id="u-andrew",
        execution_owner_user_id="u-andrew",
    )
    await db.enable_lifecycle_gate("src-agent-sessions-codex")
    collection = RecordingCollection()
    adapters = build_sqlite_adapters(db, collection)
    store = MemoryStore(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
    )

    async def fake_embed(text: str):
        return [0.1]

    monkeypatch.setattr(store, "_embed", fake_embed)
    try:
        yield db, store, collection
    finally:
        await db.close()


def _durable(
    rule: str,
    *,
    scope: str = "Agent-session memory extraction.",
    rationale: str | None = None,
) -> dict:
    return {"rule": rule, "scope": scope, "rationale": rationale}


def _proposal(**overrides) -> AgentKnowledgePatchProposal:
    base = {
        "action": "create_new_concept",
        "concept_type": "debugging_takeaway",
        "title": "Source scheduler lifecycle",
        "claim_text": (
            "Workspace source schedulers must start during app startup so overdue "
            "source schedules run without UI traffic."
        ),
        "durable_claim": {
            "rule": "Workspace source schedulers must start during app startup.",
            "scope": "Workspace source scheduling in MemForge.",
            "rationale": "This lets overdue schedules run without waiting for UI traffic.",
        },
        "memory_type": "procedure",
        "reason": "The window confirms a durable scheduler invariant.",
        "confidence": 0.9,
        "citations": ["agent-window://codex/sess-1/sha256-window"],
    }
    base.update(overrides)
    return AgentKnowledgePatchProposal(**base)


async def _relation_runs_for_memory(db: Database, memory_id: str) -> list[dict]:
    rows: list[dict] = []
    async with db.db.execute(
        """SELECT rr.*
           FROM relation_runs rr
           JOIN evidence_relations er ON er.relation_run_id = rr.id
           WHERE er.memory_id = ?
           ORDER BY rr.started_at""",
        (memory_id,),
    ) as cursor:
        async for row in cursor:
            rows.append(dict(row))
    return rows


async def _support_evidence_unit_ids(db: Database, memory_id: str) -> set[str]:
    async with db.db.execute(
        """SELECT DISTINCT er.evidence_unit_id
             FROM memory_support_assertions msa
             JOIN evidence_references er ON er.id = msa.evidence_reference_id
            WHERE msa.memory_id = ? AND msa.active = 1""",
        (memory_id,),
    ) as cursor:
        return {str(row[0]) async for row in cursor}


async def _stale_support_reference_count(db: Database, memory_id: str) -> int:
    async with db.db.execute(
        """SELECT COUNT(*)
             FROM memory_support_assertions msa
             JOIN evidence_references er ON er.id = msa.evidence_reference_id
             JOIN source_observations so ON so.id = er.observation_id
            WHERE msa.memory_id = ?
              AND msa.active = 1
              AND er.observation_revision_id <> so.current_revision_id""",
        (memory_id,),
    ) as cursor:
        row = await cursor.fetchone()
    return int(row[0])


@pytest.mark.asyncio
async def test_upsert_agent_concept_rolls_back_after_cancellation(bundle_stack, monkeypatch):
    db, _store, _collection = bundle_stack
    observed_at = datetime(2026, 7, 19, tzinfo=timezone.utc)
    concept = {
        "concept_id": "concept-cancelled-upsert",
        "source_id": "src-agent-sessions-codex",
        "owner_user_id": "u-andrew",
        "workspace": "/workspace/memforge-cloud",
        "repo_identifier": "github.tools.sap/hcm/memforge-cloud",
        "concept_type": "decision",
        "concept_path": "decisions/cancellation.md",
        "frontmatter": {},
        "observed_at": observed_at,
    }
    await db.upsert_agent_concept(
        **concept,
        title="Original title",
        markdown_body="Original body",
    )
    original_commit = db.db.commit
    commit_started = asyncio.Event()

    async def block_commit() -> None:
        commit_started.set()
        await asyncio.Future()

    monkeypatch.setattr(db.db, "commit", block_commit)
    update = asyncio.create_task(
        db.upsert_agent_concept(
            **concept,
            title="Cancelled title",
            markdown_body="Cancelled body",
        )
    )
    await commit_started.wait()
    update.cancel()

    with pytest.raises(asyncio.CancelledError):
        await update

    monkeypatch.setattr(db.db, "commit", original_commit)
    stored = await db.get_agent_concept(concept["concept_id"])
    assert stored is not None
    assert stored["title"] == "Original title"
    assert stored["markdown_body"] == "Original body"

    await db.upsert_agent_concept(
        **concept,
        title="Committed title",
        markdown_body="Committed body",
    )
    committed = await db.get_agent_concept(concept["concept_id"])
    assert committed is not None
    assert committed["title"] == "Committed title"


@pytest.mark.asyncio
async def test_create_private_concept_claim_and_memory(bundle_stack):
    db, store, collection = bundle_stack
    service = AgentKnowledgeBundleService(db=db, memory_store=store)

    result = await service.apply_patch_proposal(
        proposal=_proposal(),
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-1",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
        submitted_at=datetime(2026, 6, 18, 12, 0, tzinfo=timezone.utc),
        source_updated_at=None,
    )

    assert result.outcome == "applied"
    assert result.concept_id
    assert result.claim_id
    assert result.memory_id

    concept = await db.get_agent_concept(result.concept_id)
    assert concept is not None
    assert concept["owner_user_id"] == "u-andrew"
    assert concept["visibility"] == Visibility.PRIVATE.value
    assert concept["repo_identifier"] == "github.tools.sap/hcm/memforge-cloud"

    claim = await db.get_agent_claim(result.claim_id)
    assert claim is not None
    assert claim["concept_id"] == result.concept_id
    assert claim["memory_id"] == result.memory_id

    memory = await db.get_memory(result.memory_id)
    assert memory is not None
    assert memory.visibility == Visibility.PRIVATE.value
    assert memory.owner_user_id == "u-andrew"
    assert memory.repo_identifier == "github.tools.sap/hcm/memforge-cloud"
    assert (
        memory.content == "Workspace source schedulers must start during app startup.\n"
        "Applies: Workspace source scheduling in MemForge.\n"
        "Why: This lets overdue schedules run without waiting for UI traffic."
    )
    assert "overdue source schedules" in (memory.extraction_context or "")
    assert collection.upserted[result.memory_id]["owner_user_id"] == "u-andrew"

    relation_runs = await _relation_runs_for_memory(db, result.memory_id)
    assert len(relation_runs) == 1
    assert relation_runs[0]["lifecycle_action"] == LifecycleAction.CREATE_MEMORY.value
    evidence_unit = await db.get_evidence_unit(relation_runs[0]["evidence_unit_id"])
    assert evidence_unit is not None
    assert evidence_unit.source_type == "agent_session"
    assert evidence_unit.client == "codex"
    assert evidence_unit.repo_identifier == "github.tools.sap/hcm/memforge-cloud"
    assert evidence_unit.source_metadata["claim_anchor"] == (
        f"u-andrew:github.tools.sap/hcm/memforge-cloud:{result.concept_id}:{result.claim_id}"
    )
    relations = await db.get_evidence_relations(evidence_unit.id)
    assert [(relation.memory_id, relation.relation_type) for relation in relations] == [
        (result.memory_id, RelationType.SUPPORTS)
    ]
    assert await _support_evidence_unit_ids(db, result.memory_id) == {evidence_unit.id}
    assert await _stale_support_reference_count(db, result.memory_id) == 0


@pytest.mark.asyncio
async def test_agent_claim_identity_does_not_merge_with_ordinary_exact_memory(bundle_stack):
    db, store, _collection = bundle_stack
    content = (
        "Workspace source schedulers must start during app startup.\n"
        "Applies: Workspace source scheduling in MemForge.\n"
        "Why: This lets overdue schedules run without waiting for UI traffic."
    )
    ordinary = Memory(
        id="mem-ordinary-exact-claim",
        memory_type="procedure",
        content=content,
        content_hash=content_hash(content),
        visibility=Visibility.PRIVATE.value,
        owner_user_id="u-andrew",
        project_key="UNSORTED",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
    )
    await db.insert_memory(ordinary)
    service = AgentKnowledgeBundleService(db=db, memory_store=store)

    result = await service.apply_patch_proposal(
        proposal=_proposal(),
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-agent-after-ordinary",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
        source_updated_at=None,
    )

    claim = await db.get_agent_claim(result.claim_id)
    exact_rows = await db.db.execute_fetchall(
        """SELECT id FROM memories
           WHERE content_hash = ? AND status = 'active'
           ORDER BY id""",
        (ordinary.content_hash,),
    )
    assert result.outcome == "applied"
    assert result.memory_id != ordinary.id
    assert claim is not None
    assert claim["memory_id"] == result.memory_id
    assert {row["id"] for row in exact_rows} == {
        ordinary.id,
        result.memory_id,
    }


@pytest.mark.asyncio
async def test_post_cutover_agent_claim_write_commits_source_projection_lineage(bundle_stack):
    db, store, _collection = bundle_stack
    source_id = "src-agent-sessions-codex"
    await db.enable_lifecycle_gate(source_id)
    service = AgentKnowledgeBundleService(db=db, memory_store=store)

    result = await service.apply_patch_proposal(
        proposal=_proposal(
            concept_id="akb_concept_post_cutover",
            claim_id="akb_claim_post_cutover",
        ),
        owner_user_id="u-andrew",
        source_id=source_id,
        client="codex",
        session_id="sess-post-cutover",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
        submitted_at=datetime(2026, 7, 15, 8, 0, tzinfo=timezone.utc),
        source_updated_at=datetime(2026, 7, 15, 8, 0, tzinfo=timezone.utc),
    )

    assert result.outcome == "applied"
    assert result.memory_id is not None
    assert result.concept_id is not None
    source_unit = await db.find_source_unit_by_document_id(source_id, result.concept_id)
    assert source_unit is not None
    support_reference_ids = await db.get_active_memory_support_reference_ids(result.memory_id)
    assert support_reference_ids
    support_by_memory = await db.get_source_unit_support_reference_ids(source_unit.id)
    assert support_by_memory[result.memory_id] == support_reference_ids


@pytest.mark.asyncio
async def test_shared_managed_source_exposes_concepts_to_subscribed_workspace_members(bundle_stack):
    db, store, _ = bundle_stack
    bundle = AgentKnowledgeBundleService(db=db, memory_store=store)
    result = await bundle.apply_patch_proposal(
        proposal=_proposal(),
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-share",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
        source_updated_at=None,
    )

    assert (
        await db.list_agent_concepts(
            viewer_user_id="u-bob",
            repo_identifier="github.tools.sap/hcm/memforge-cloud",
        )
        == []
    )

    access = SourceAccessTransitionService(db=db, memory_store=store)
    transition = await access.start(
        source_id="src-agent-sessions-codex",
        actor_user_id="u-andrew",
        target_policy="workspace",
        idempotency_key="share-codex-source",
    )
    await access.run(transition["operation_id"])

    shared = await db.list_agent_concepts(
        viewer_user_id="u-bob",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
    )
    assert [concept["id"] for concept in shared] == [result.concept_id]
    await db.set_source_subscription("src-agent-sessions-codex", "u-bob", False)
    assert (
        await db.list_agent_concepts(
            viewer_user_id="u-bob",
            repo_identifier="github.tools.sap/hcm/memforge-cloud",
        )
        == []
    )


@pytest.mark.asyncio
async def test_create_concept_does_not_commit_projection_before_memory_lifecycle(bundle_stack, monkeypatch):
    db, store, _ = bundle_stack

    async def fail_insert_memory(*args, **kwargs):
        raise RuntimeError("memory lifecycle commit failed")

    monkeypatch.setattr(store, "insert_agent_claim_memory", fail_insert_memory)
    service = AgentKnowledgeBundleService(db=db, memory_store=store)

    with pytest.raises(RuntimeError, match="memory lifecycle commit failed"):
        await service.apply_patch_proposal(
            proposal=_proposal(concept_id="akb_concept_fail", claim_id="akb_claim_fail"),
            owner_user_id="u-andrew",
            source_id="src-agent-sessions-codex",
            client="codex",
            session_id="sess-1",
            workspace="/workspace/memforge-cloud",
            repo_identifier="github.tools.sap/hcm/memforge-cloud",
            project_key="UNSORTED",
            submitted_at=datetime(2026, 6, 18, 12, 0, tzinfo=timezone.utc),
            source_updated_at=None,
        )

    assert await db.get_document("akb_concept_fail") is not None
    assert await db.get_agent_concept("akb_concept_fail") is None
    assert await db.get_agent_claim("akb_claim_fail") is None


@pytest.mark.asyncio
async def test_create_concept_does_not_leave_memory_when_claim_projection_fails(bundle_stack, monkeypatch):
    db, store, _ = bundle_stack
    service = AgentKnowledgeBundleService(db=db, memory_store=store)

    async def fail_upsert_claim(*args, **kwargs):
        raise RuntimeError("claim projection failed")

    monkeypatch.setattr(db, "_upsert_agent_claim_unlocked", fail_upsert_claim)

    with pytest.raises(RuntimeError, match="claim projection failed"):
        await service.apply_patch_proposal(
            proposal=_proposal(concept_id="akb_concept_projection_fail", claim_id="akb_claim_projection_fail"),
            owner_user_id="u-andrew",
            source_id="src-agent-sessions-codex",
            client="codex",
            session_id="sess-1",
            workspace="/workspace/memforge-cloud",
            repo_identifier="github.tools.sap/hcm/memforge-cloud",
            project_key="UNSORTED",
            submitted_at=datetime(2026, 6, 18, 12, 0, tzinfo=timezone.utc),
            source_updated_at=None,
        )

    assert await db.get_agent_claim("akb_claim_projection_fail") is None
    async with db.db.execute(
        """SELECT COUNT(*)
           FROM memories m
           JOIN memory_sources ms ON ms.memory_id = m.id
           WHERE ms.doc_id = ?""",
        ("akb_concept_projection_fail",),
    ) as cursor:
        assert (await cursor.fetchone())[0] == 0


@pytest.mark.asyncio
async def test_add_claim_does_not_commit_projection_before_memory_lifecycle(bundle_stack, monkeypatch):
    db, store, _ = bundle_stack
    service = AgentKnowledgeBundleService(db=db, memory_store=store)
    created = await service.apply_patch_proposal(
        proposal=_proposal(),
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-1",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
        source_updated_at=None,
    )

    async def fail_insert_memory(*args, **kwargs):
        raise RuntimeError("memory lifecycle commit failed")

    monkeypatch.setattr(store, "insert_agent_claim_memory", fail_insert_memory)

    with pytest.raises(RuntimeError, match="memory lifecycle commit failed"):
        await service.apply_patch_proposal(
            proposal=_proposal(
                action="add_new_claim",
                concept_id=created.concept_id,
                claim_id="akb_claim_add_fail",
                claim_text="A second claim should not be projected if memory creation fails.",
                durable_claim=_durable("A second claim should not be projected if memory creation fails."),
            ),
            owner_user_id="u-andrew",
            source_id="src-agent-sessions-codex",
            client="codex",
            session_id="sess-2",
            workspace="/workspace/memforge-cloud",
            repo_identifier="github.tools.sap/hcm/memforge-cloud",
            project_key="UNSORTED",
            source_updated_at=None,
        )

    assert await db.get_agent_claim("akb_claim_add_fail") is None
    claims = await db.list_agent_claims(created.concept_id)
    assert [claim["id"] for claim in claims] == [created.claim_id]


@pytest.mark.asyncio
async def test_add_claim_writes_citations_and_markdown_inside_lifecycle_contract(bundle_stack, monkeypatch):
    db, store, _ = bundle_stack
    service = AgentKnowledgeBundleService(db=db, memory_store=store)
    created = await service.apply_patch_proposal(
        proposal=_proposal(),
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-1",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
        source_updated_at=None,
    )

    async def fail_public_citation_write(*args, **kwargs):
        raise AssertionError("citation projection must be committed by the lifecycle store contract")

    async def fail_public_markdown_write(*args, **kwargs):
        raise AssertionError("concept markdown must be committed by the lifecycle store contract")

    monkeypatch.setattr(db, "add_agent_claim_citation", fail_public_citation_write)
    monkeypatch.setattr(db, "update_agent_concept_markdown", fail_public_markdown_write)

    added = await service.apply_patch_proposal(
        proposal=_proposal(
            action="add_new_claim",
            concept_id=created.concept_id,
            claim_id="akb_claim_add_atomic",
            claim_text="Scheduler claims are persisted with their concept markdown in one lifecycle commit.",
            durable_claim=_durable("Scheduler claims persist with their concept markdown in one lifecycle commit."),
            citations=["agent-window://codex/sess-2/sha256-window"],
        ),
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-2",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
        source_updated_at=None,
    )

    assert added.outcome == "applied"
    citations = await db.list_agent_claim_citations("akb_claim_add_atomic")
    assert [citation["citation_url"] for citation in citations] == ["agent-window://codex/sess-2/sha256-window"]
    concept = await db.get_agent_concept(created.concept_id)
    assert concept is not None
    assert "Workspace source schedulers must start during app startup" in concept["markdown_body"]
    assert "Scheduler claims are persisted with their concept markdown" in concept["markdown_body"]
    assert "agent-window://codex/sess-2/sha256-window" in concept["markdown_body"]


@pytest.mark.asyncio
async def test_update_claim_does_not_commit_projection_before_memory_lifecycle(bundle_stack, monkeypatch):
    db, store, _ = bundle_stack
    service = AgentKnowledgeBundleService(db=db, memory_store=store)
    created = await service.apply_patch_proposal(
        proposal=_proposal(),
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-1",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
        source_updated_at=None,
    )
    original_claim = await db.get_agent_claim(created.claim_id)

    async def fail_supersede_memory(*args, **kwargs):
        raise RuntimeError("memory lifecycle commit failed")

    monkeypatch.setattr(store, "supersede_agent_claim_memory", fail_supersede_memory)

    with pytest.raises(RuntimeError, match="memory lifecycle commit failed"):
        await service.apply_patch_proposal(
            proposal=_proposal(
                action="update_existing_claim",
                concept_id=created.concept_id,
                claim_id=created.claim_id,
                claim_text="This update must not replace the projection if memory supersession fails.",
                durable_claim=_durable("This update must not replace the projection if memory supersession fails."),
            ),
            owner_user_id="u-andrew",
            source_id="src-agent-sessions-codex",
            client="codex",
            session_id="sess-2",
            workspace="/workspace/memforge-cloud",
            repo_identifier="github.tools.sap/hcm/memforge-cloud",
            project_key="UNSORTED",
            source_updated_at=None,
        )

    claim = await db.get_agent_claim(created.claim_id)
    assert claim == original_claim
    memory = await db.get_memory(created.memory_id)
    assert memory is not None
    assert memory.status == "active"
    assert memory.superseded_by is None


@pytest.mark.asyncio
async def test_update_claim_rolls_back_when_atomic_citation_projection_fails(bundle_stack, monkeypatch):
    db, store, _ = bundle_stack
    service = AgentKnowledgeBundleService(db=db, memory_store=store)
    created = await service.apply_patch_proposal(
        proposal=_proposal(),
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-1",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
        source_updated_at=None,
    )
    original_claim = await db.get_agent_claim(created.claim_id)
    original_memory = await db.get_memory(created.memory_id)
    original_concept = await db.get_agent_concept(created.concept_id)
    original_citations = await db.list_agent_claim_citations(created.claim_id)
    original_add_citation = db._add_agent_claim_citation_unlocked

    async def fail_new_citation(*, claim_id: str, citation_url: str, observed: str):
        if citation_url == "agent-window://codex/sess-2/sha256-window":
            raise RuntimeError("atomic citation projection failed")
        await original_add_citation(claim_id=claim_id, citation_url=citation_url, observed=observed)

    monkeypatch.setattr(db, "_add_agent_claim_citation_unlocked", fail_new_citation)

    with pytest.raises(RuntimeError, match="atomic citation projection failed"):
        await service.apply_patch_proposal(
            proposal=_proposal(
                action="update_existing_claim",
                concept_id=created.concept_id,
                claim_id=created.claim_id,
                claim_text="Workspace source schedulers advance next_run_at after a successful claim.",
                durable_claim=_durable("Workspace source schedulers advance next_run_at after a successful claim."),
                reason="New evidence refines the scheduler lifecycle claim.",
                citations=["agent-window://codex/sess-2/sha256-window"],
            ),
            owner_user_id="u-andrew",
            source_id="src-agent-sessions-codex",
            client="codex",
            session_id="sess-2",
            workspace="/workspace/memforge-cloud",
            repo_identifier="github.tools.sap/hcm/memforge-cloud",
            project_key="UNSORTED",
            source_updated_at=None,
        )

    assert await db.get_agent_claim(created.claim_id) == original_claim
    assert await db.list_agent_claim_citations(created.claim_id) == original_citations
    assert await db.get_agent_concept(created.concept_id) == original_concept
    restored_memory = await db.get_memory(created.memory_id)
    assert original_memory is not None
    assert restored_memory is not None
    assert restored_memory.status == original_memory.status == "active"
    assert restored_memory.superseded_by is None


@pytest.mark.asyncio
async def test_update_existing_claim_rolls_back_if_claim_projection_commit_fails(bundle_stack, monkeypatch):
    db, store, _collection = bundle_stack
    service = AgentKnowledgeBundleService(db=db, memory_store=store)
    created = await service.apply_patch_proposal(
        proposal=_proposal(),
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-1",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
        source_updated_at=None,
    )
    update_proposal = _proposal(
        action="update_existing_claim",
        concept_id=created.concept_id,
        claim_id=created.claim_id,
        claim_text="Workspace source schedulers advance next_run_at after a successful claim.",
        durable_claim=_durable("Workspace source schedulers advance next_run_at after a successful claim."),
        reason="New evidence refines the scheduler lifecycle claim.",
    )
    original_claim = await db.get_agent_claim(created.claim_id)
    original_memory = await db.get_memory(created.memory_id)
    original_claim_projection = db._upsert_agent_claim_unlocked

    async def flaky_claim_projection(*args, **kwargs):
        await original_claim_projection(*args, **kwargs)
        raise RuntimeError("atomic claim replacement commit failed")

    monkeypatch.setattr(db, "_upsert_agent_claim_unlocked", flaky_claim_projection)
    with pytest.raises(RuntimeError, match="atomic claim replacement commit failed"):
        await service.apply_patch_proposal(
            proposal=update_proposal,
            owner_user_id="u-andrew",
            source_id="src-agent-sessions-codex",
            client="codex",
            session_id="sess-2",
            workspace="/workspace/memforge-cloud",
            repo_identifier="github.tools.sap/hcm/memforge-cloud",
            project_key="UNSORTED",
            source_updated_at=None,
        )

    restored_memory = await db.get_memory(created.memory_id)
    claim_after_failure = await db.get_agent_claim(created.claim_id)
    assert original_memory is not None
    assert restored_memory is not None
    assert restored_memory.status == original_memory.status == "active"
    assert restored_memory.superseded_by is None
    assert claim_after_failure is not None
    assert claim_after_failure == original_claim
    claims = await db.list_agent_claims(created.concept_id)
    assert [claim["id"] for claim in claims] == [created.claim_id]


@pytest.mark.asyncio
async def test_update_agent_concept_markdown_fails_if_projection_target_is_missing(bundle_stack):
    db, store, _collection = bundle_stack
    service = AgentKnowledgeBundleService(db=db, memory_store=store)
    created = await service.apply_patch_proposal(
        proposal=_proposal(),
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-1",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
        source_updated_at=None,
    )

    with pytest.raises(RuntimeError, match="agent concept projection target missing"):
        await db.update_agent_concept_markdown(
            concept_id="missing-concept",
            markdown_body="# Missing\n",
            observed_at=datetime(2026, 6, 22, 12, 0, tzinfo=timezone.utc),
        )

    existing_concept = await db.get_agent_concept(created.concept_id)
    assert existing_concept is not None
    assert "Workspace source schedulers must start during app startup" in existing_concept["markdown_body"]


@pytest.mark.asyncio
async def test_retired_claim_backed_memory_is_removed_from_active_claim_projection(bundle_stack):
    db, store, _collection = bundle_stack
    service = AgentKnowledgeBundleService(db=db, memory_store=store)
    created = await service.apply_patch_proposal(
        proposal=_proposal(),
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-1",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
        source_updated_at=None,
    )

    old = await db.get_memory(created.memory_id)
    assert old is not None
    await MemoryLifecycleService(db=db, memory_store=store).retire_memory(
        created.memory_id,
        reason="user_retired",
        expected_content_hash=old.content_hash,
    )

    active_claims = await db.list_agent_claims(created.concept_id)
    lineage_claim = await db.get_agent_claim(created.claim_id)
    concept = await db.get_agent_concept(created.concept_id)
    prompt = await render_agent_knowledge_patch_prompt(
        db=db,
        owner_user_id="u-andrew",
        client="codex",
        session_id="sess-2",
        trigger="stop",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        branch=None,
        history_window={"kind": "tail"},
        events=[
            {
                "evidence_id": "event-1",
                "kind": "user",
                "text": "Keep future scheduler work aligned with current active memories.",
            }
        ],
        transcript_markdown="user: Keep future scheduler work aligned with current active memories.",
    )

    assert lineage_claim is not None
    assert lineage_claim["memory_id"] == created.memory_id
    assert active_claims == []
    assert await db.get_active_memory_support_reference_ids(created.memory_id) == ()
    assert (await db.get_memory(created.memory_id)).status == "retired"
    assert concept is not None
    assert "Workspace source schedulers must start during app startup" not in concept["markdown_body"]
    assert created.claim_id not in prompt
    assert "Workspace source schedulers must start during app startup" not in prompt


@pytest.mark.asyncio
async def test_update_agent_concept_markdown_ignores_stale_projection_body(bundle_stack):
    db, store, _collection = bundle_stack
    service = AgentKnowledgeBundleService(db=db, memory_store=store)
    created = await service.apply_patch_proposal(
        proposal=_proposal(),
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-1",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
        source_updated_at=None,
    )

    newer_observed = datetime(2030, 6, 22, 12, 30, tzinfo=timezone.utc)
    await db.update_agent_concept_markdown(
        concept_id=created.concept_id,
        markdown_body="# Scheduler\n\nNewer projection.",
        observed_at=newer_observed,
    )
    older_observed = datetime(2030, 6, 22, 12, 0, tzinfo=timezone.utc)
    await db.update_agent_concept_markdown(
        concept_id=created.concept_id,
        markdown_body="# Scheduler\n\nStale projection.",
        observed_at=older_observed,
    )

    concept = await db.get_agent_concept(created.concept_id)
    assert concept is not None
    assert concept["markdown_body"] == "# Scheduler\n\nNewer projection."
    assert concept["updated_at"] == "2030-06-22T12:30:00+00:00"
    assert concept["last_observed_at"] == "2030-06-22T12:30:00+00:00"


@pytest.mark.asyncio
async def test_update_existing_claim_records_relation_inside_supersede_contract(bundle_stack, monkeypatch):
    db, store, _collection = bundle_stack
    service = AgentKnowledgeBundleService(db=db, memory_store=store)
    created = await service.apply_patch_proposal(
        proposal=_proposal(),
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-1",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
        source_updated_at=None,
    )
    update_proposal = _proposal(
        action="update_existing_claim",
        concept_id=created.concept_id,
        claim_id=created.claim_id,
        claim_text="Workspace source schedulers advance next_run_at after a successful claim.",
        durable_claim=_durable("Workspace source schedulers advance next_run_at after a successful claim."),
        reason="New evidence refines the scheduler lifecycle claim.",
    )
    original_claim = await db.get_agent_claim(created.claim_id)
    original_memory = await db.get_memory(created.memory_id)

    async def fail_public_relation_bundle(*args, **kwargs):
        raise AssertionError("agent claim supersede must record the relation inside the supersede contract")

    monkeypatch.setattr(db, "record_relation_outcome_bundle", fail_public_relation_bundle)
    await service.apply_patch_proposal(
        proposal=update_proposal,
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-2",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
        source_updated_at=None,
    )

    restored_memory = await db.get_memory(created.memory_id)
    claim_after_failure = await db.get_agent_claim(created.claim_id)
    assert original_memory is not None
    assert restored_memory is not None
    assert restored_memory.status == "superseded"
    assert restored_memory.superseded_by is not None
    assert claim_after_failure != original_claim
    claims = await db.list_agent_claims(created.concept_id)
    assert [claim["id"] for claim in claims] == [created.claim_id]


@pytest.mark.asyncio
async def test_update_existing_claim_supersedes_memory_projection(bundle_stack):
    db, store, collection = bundle_stack
    service = AgentKnowledgeBundleService(db=db, memory_store=store)
    created = await service.apply_patch_proposal(
        proposal=_proposal(),
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-1",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
        source_updated_at=None,
    )
    source_unit = await db.find_source_unit_by_document_id(
        "src-agent-sessions-codex",
        created.concept_id,
    )
    assert source_unit is not None
    initial_revision = await db.get_current_source_unit_revision(source_unit.id)
    assert initial_revision is not None
    await db.enable_lifecycle_gate("src-agent-sessions-codex")

    updated = await service.apply_patch_proposal(
        proposal=_proposal(
            action="update_existing_claim",
            concept_id=created.concept_id,
            claim_id=created.claim_id,
            claim_text=(
                "Workspace source schedulers must start during app startup, claim due "
                "source schedules, and advance next_run_at after a successful claim."
            ),
            durable_claim=_durable(
                "Source schedulers start on app startup, claim due schedules, and advance next_run_at after success."
            ),
            reason="New evidence refines the scheduler lifecycle claim.",
            citations=["agent-window://codex/sess-2/sha256-window"],
        ),
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-2",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
        source_updated_at=None,
    )

    assert updated.outcome == "applied"
    assert updated.concept_id == created.concept_id
    assert updated.claim_id == created.claim_id
    assert updated.memory_id != created.memory_id

    memory = await db.get_memory(updated.memory_id)
    assert memory is not None
    assert memory.content == (
        "Source schedulers start on app startup, claim due schedules, and advance next_run_at after success.\n"
        "Applies: Agent-session memory extraction."
    )
    assert "advance next_run_at" in (memory.extraction_context or "")
    assert collection.upserted[updated.memory_id]["content_hash"] == memory.content_hash
    assert created.memory_id in collection.deleted

    old_memory = await db.get_memory(created.memory_id)
    assert old_memory is not None
    assert old_memory.status == "superseded"
    assert old_memory.superseded_by == updated.memory_id
    assert old_memory.replacement_reason == "New evidence refines the scheduler lifecycle claim."
    assert old_memory.replacement_kind == "revision"
    assert await db.get_active_memory_support_reference_ids(created.memory_id) == ()

    updated_support = await db.get_active_memory_support_reference_ids(updated.memory_id)
    assert updated_support
    current_revision = await db.get_current_source_unit_revision(source_unit.id)
    assert current_revision is not None
    assert current_revision.id != initial_revision.id
    support_by_memory = await db.get_source_unit_support_reference_ids(source_unit.id)
    assert support_by_memory == {updated.memory_id: updated_support}

    claim = await db.get_agent_claim(created.claim_id)
    assert claim is not None
    assert claim["memory_id"] == updated.memory_id
    assert "advance next_run_at" in claim["claim_text"]

    citations = await db.list_agent_claim_citations(created.claim_id)
    assert [citation["citation_url"] for citation in citations] == [
        "agent-window://codex/sess-1/sha256-window",
        "agent-window://codex/sess-2/sha256-window",
    ]
    concept = await db.get_agent_concept(created.concept_id)
    assert concept is not None
    assert "Workspace source schedulers must start during app startup" in concept["markdown_body"]
    assert "agent-window://codex/sess-2/sha256-window" in concept["markdown_body"]

    relation_runs = await _relation_runs_for_memory(db, updated.memory_id)
    assert relation_runs[-1]["lifecycle_action"] == LifecycleAction.SUPERSEDE_MEMORY.value
    evidence_unit = await db.get_evidence_unit(relation_runs[-1]["evidence_unit_id"])
    assert evidence_unit is not None
    assert evidence_unit.source_metadata["source_patch_intent"] == "update_existing_claim"
    candidates = await db.get_relation_candidates(relation_runs[-1]["id"])
    assert [candidate.memory_id for candidate in candidates] == [created.memory_id]
    assert all(candidate.was_checked for candidate in candidates)
    relations = await db.get_evidence_relations(evidence_unit.id)
    assert [(relation.memory_id, relation.relation_type) for relation in relations] == [
        (updated.memory_id, RelationType.SUPPORTS)
    ]
    assert evidence_unit.client == "codex"
    assert await _support_evidence_unit_ids(db, updated.memory_id) == {evidence_unit.id}
    assert await _stale_support_reference_count(db, updated.memory_id) == 0


@pytest.mark.asyncio
async def test_update_without_model_claim_id_resolves_target_from_memory_candidate(bundle_stack):
    db, store, collection = bundle_stack
    service = AgentKnowledgeBundleService(db=db, memory_store=store)
    created = await service.apply_patch_proposal(
        proposal=_proposal(),
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-1",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
        source_updated_at=None,
    )
    collection.query_ids = [created.memory_id]
    collection.query_distances = [0.01]

    updated = await service.apply_patch_proposal(
        proposal=_proposal(
            action="update_existing_claim",
            concept_id=None,
            claim_id=None,
            claim_text=(
                "Workspace source schedulers are now owned by the cloud app bootstrap "
                "and advance next_run_at after a successful claim."
            ),
            durable_claim=_durable(
                "Source schedulers are owned by the cloud app bootstrap and advance next_run_at after success."
            ),
            reason="The new session outcome refines the older scheduler lifecycle memory.",
            citations=["agent-window://codex/sess-2/sha256-window"],
        ),
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-2",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
        source_updated_at=None,
    )

    assert updated.outcome == "applied"
    assert updated.concept_id == created.concept_id
    assert updated.claim_id == created.claim_id
    assert updated.memory_id != created.memory_id

    old_memory = await db.get_memory(created.memory_id)
    new_memory = await db.get_memory(updated.memory_id)
    claim = await db.get_agent_claim(created.claim_id)
    assert old_memory is not None
    assert new_memory is not None
    assert claim is not None
    assert old_memory.status == "superseded"
    assert old_memory.superseded_by == updated.memory_id
    assert claim["memory_id"] == updated.memory_id
    assert "cloud app bootstrap" in claim["claim_text"]


@pytest.mark.asyncio
async def test_create_action_reconciles_existing_memory_candidate_instead_of_duplicate(bundle_stack):
    db, store, collection = bundle_stack
    service = AgentKnowledgeBundleService(db=db, memory_store=store)
    created = await service.apply_patch_proposal(
        proposal=_proposal(),
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-1",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
        source_updated_at=None,
    )
    collection.query_ids = [created.memory_id]
    collection.query_distances = [0.01]

    reconciled = await service.apply_patch_proposal(
        proposal=_proposal(
            action="create_new_concept",
            concept_id=None,
            claim_id=None,
            title="Cloud bootstrap scheduler ownership",
            concept_type="debugging_takeaway",
            claim_text="Source schedulers are owned by the cloud app bootstrap.",
            durable_claim=_durable("Source schedulers are owned by the cloud app bootstrap."),
            reason="The model proposed a new concept, but the memory candidate is the current scheduler claim.",
        ),
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-2",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
        source_updated_at=None,
    )

    assert reconciled.outcome == "applied"
    assert reconciled.concept_id == created.concept_id
    assert reconciled.claim_id == created.claim_id
    assert reconciled.memory_id != created.memory_id
    claims = await db.list_agent_claims(created.concept_id)
    assert [claim["id"] for claim in claims] == [created.claim_id]
    old_memory = await db.get_memory(created.memory_id)
    assert old_memory is not None
    assert old_memory.status == "superseded"


@pytest.mark.asyncio
async def test_create_action_with_same_memory_content_moves_existing_claim_forward(bundle_stack):
    db, store, collection = bundle_stack
    service = AgentKnowledgeBundleService(db=db, memory_store=store)
    created = await service.apply_patch_proposal(
        proposal=_proposal(),
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-1",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
        source_updated_at=None,
    )
    collection.query_ids = [created.memory_id]
    collection.query_distances = [0.01]

    moved = await service.apply_patch_proposal(
        proposal=_proposal(
            action="create_new_concept",
            concept_id=None,
            claim_id=None,
        ),
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-2",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
        source_updated_at=None,
    )

    assert moved.outcome == "applied"
    assert moved.concept_id == created.concept_id
    assert moved.claim_id == created.claim_id
    assert moved.memory_id != created.memory_id
    claims = await db.list_agent_claims(created.concept_id)
    assert [claim["id"] for claim in claims] == [created.claim_id]
    old_memory = await db.get_memory(created.memory_id)
    new_memory = await db.get_memory(moved.memory_id)
    assert old_memory is not None
    assert new_memory is not None
    assert old_memory.status == "superseded"
    assert new_memory.status == "active"


@pytest.mark.asyncio
async def test_update_without_model_claim_id_does_not_create_duplicate_when_no_memory_target(bundle_stack):
    db, store, _collection = bundle_stack
    service = AgentKnowledgeBundleService(db=db, memory_store=store)
    created = await service.apply_patch_proposal(
        proposal=_proposal(),
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-1",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
        source_updated_at=None,
    )

    rejected = await service.apply_patch_proposal(
        proposal=_proposal(
            action="update_existing_claim",
            concept_id=None,
            claim_id=None,
            claim_text="This update has no safe memory target.",
            durable_claim=_durable("This update has no safe memory target."),
        ),
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-2",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
        source_updated_at=None,
    )

    assert rejected.outcome == "rejected_scope"
    claims = await db.list_agent_claims(created.concept_id)
    assert [claim["id"] for claim in claims] == [created.claim_id]
    memory = await db.get_memory(created.memory_id)
    assert memory is not None
    assert memory.status == "active"


@pytest.mark.asyncio
async def test_update_existing_claim_retry_is_idempotent_after_replacement(bundle_stack):
    db, store, collection = bundle_stack
    service = AgentKnowledgeBundleService(db=db, memory_store=store)
    created = await service.apply_patch_proposal(
        proposal=_proposal(),
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-1",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
        source_updated_at=None,
    )
    update = _proposal(
        action="update_existing_claim",
        concept_id=created.concept_id,
        claim_id=created.claim_id,
        claim_text=(
            "Workspace source schedulers must start during app startup, claim due "
            "source schedules, and advance next_run_at after a successful claim."
        ),
        durable_claim=_durable(
            "Source schedulers start on app startup, claim due schedules, and advance next_run_at after success."
        ),
        reason="New evidence refines the scheduler lifecycle claim.",
        citations=["agent-window://codex/sess-2/sha256-window"],
    )

    first_update = await service.apply_patch_proposal(
        proposal=update,
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-2",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
        source_updated_at=None,
    )
    async with db._write_lock:
        await db.db.execute(
            "DELETE FROM memory_sources WHERE memory_id = ? AND doc_id = ?",
            (first_update.memory_id, created.concept_id),
        )
        await db.db.execute("DELETE FROM memories_fts WHERE memory_id = ?", (first_update.memory_id,))
        await db.db.commit()
    collection.upserted.pop(first_update.memory_id, None)

    retry = await service.apply_patch_proposal(
        proposal=update,
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-2",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
        source_updated_at=None,
    )

    assert retry.outcome == "applied"
    assert retry.memory_id == first_update.memory_id
    claim = await db.get_agent_claim(created.claim_id)
    assert claim is not None
    assert claim["memory_id"] == first_update.memory_id
    old_memory = await db.get_memory(created.memory_id)
    current_memory = await db.get_memory(first_update.memory_id)
    assert old_memory is not None
    assert current_memory is not None
    assert old_memory.status == "superseded"
    assert old_memory.superseded_by == first_update.memory_id
    assert current_memory.status == "active"

    relation_runs = await _relation_runs_for_memory(db, first_update.memory_id)
    assert len(relation_runs) == 1
    assert relation_runs[0]["result_memory_id"] == first_update.memory_id
    async with db.db.execute("SELECT COUNT(*) FROM memories") as cursor:
        row = await cursor.fetchone()
    assert row[0] == 2
    async with db.db.execute(
        "SELECT COUNT(*) FROM memory_sources WHERE memory_id = ? AND doc_id = ?",
        (first_update.memory_id, created.concept_id),
    ) as cursor:
        row = await cursor.fetchone()
    assert row[0] == 1
    async with db.db.execute(
        "SELECT COUNT(*) FROM memories_fts WHERE memory_id = ?", (first_update.memory_id,)
    ) as cursor:
        row = await cursor.fetchone()
    assert row[0] == 1
    assert list(collection.upserted) == [first_update.memory_id]


@pytest.mark.asyncio
async def test_update_existing_claim_retry_rejects_changed_relation_payload(bundle_stack):
    db, store, _collection = bundle_stack
    service = AgentKnowledgeBundleService(db=db, memory_store=store)
    created = await service.apply_patch_proposal(
        proposal=_proposal(),
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-1",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
        source_updated_at=None,
    )
    update = _proposal(
        action="update_existing_claim",
        concept_id=created.concept_id,
        claim_id=created.claim_id,
        claim_text=(
            "Workspace source schedulers must start during app startup, claim due "
            "source schedules, and advance next_run_at after a successful claim."
        ),
        durable_claim=_durable(
            "Source schedulers start on app startup, claim due schedules, and advance next_run_at after success."
        ),
        reason="New evidence refines the scheduler lifecycle claim.",
    )
    first_update = await service.apply_patch_proposal(
        proposal=update,
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-2",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
        source_updated_at=None,
    )

    changed_retry = _proposal(
        action="update_existing_claim",
        concept_id=created.concept_id,
        claim_id=created.claim_id,
        claim_text=update.claim_text,
        durable_claim=update.durable_claim,
        reason="A different retry reason would change the relation payload.",
    )
    with pytest.raises(RuntimeError, match="relation_run_id collision"):
        await service.apply_patch_proposal(
            proposal=changed_retry,
            owner_user_id="u-andrew",
            source_id="src-agent-sessions-codex",
            client="codex",
            session_id="sess-2",
            workspace="/workspace/memforge-cloud",
            repo_identifier="github.tools.sap/hcm/memforge-cloud",
            project_key="UNSORTED",
            source_updated_at=None,
        )

    relation_runs = await _relation_runs_for_memory(db, first_update.memory_id)
    assert len(relation_runs) == 1


@pytest.mark.asyncio
async def test_update_existing_claim_retry_rejects_committed_candidate_snapshot_change(bundle_stack):
    db, store, _collection = bundle_stack
    service = AgentKnowledgeBundleService(db=db, memory_store=store)
    created = await service.apply_patch_proposal(
        proposal=_proposal(),
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-1",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
        source_updated_at=None,
    )
    update = _proposal(
        action="update_existing_claim",
        concept_id=created.concept_id,
        claim_id=created.claim_id,
        claim_text=(
            "Workspace source schedulers must start during app startup, claim due "
            "source schedules, and advance next_run_at after a successful claim."
        ),
        durable_claim=_durable(
            "Source schedulers start on app startup, claim due schedules, and advance next_run_at after success."
        ),
        reason="New evidence refines the scheduler lifecycle claim.",
    )
    first_update = await service.apply_patch_proposal(
        proposal=update,
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-2",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
        source_updated_at=None,
    )
    relation_runs = await _relation_runs_for_memory(db, first_update.memory_id)
    assert len(relation_runs) == 1

    async with db._write_lock:
        await db.db.execute(
            "UPDATE relation_candidates SET reason = ? WHERE relation_run_id = ?",
            ("tampered candidate snapshot", relation_runs[0]["id"]),
        )
        await db.db.commit()

    with pytest.raises(RuntimeError, match="relation_run_id collision"):
        await service.apply_patch_proposal(
            proposal=update,
            owner_user_id="u-andrew",
            source_id="src-agent-sessions-codex",
            client="codex",
            session_id="sess-2",
            workspace="/workspace/memforge-cloud",
            repo_identifier="github.tools.sap/hcm/memforge-cloud",
            project_key="UNSORTED",
            source_updated_at=None,
        )


@pytest.mark.asyncio
async def test_update_existing_claim_records_complete_mandatory_candidate_universe(bundle_stack):
    db, store, _collection = bundle_stack
    service = AgentKnowledgeBundleService(db=db, memory_store=store)
    created = await service.apply_patch_proposal(
        proposal=_proposal(),
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-1",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
        source_updated_at=None,
    )
    same_doc_memory = Memory(
        id="mem-other-claim",
        memory_type="procedure",
        content="A separate scheduler claim under the same concept remains active.",
        content_hash=content_hash("A separate scheduler claim under the same concept remains active."),
        confidence=0.9,
        visibility=Visibility.PRIVATE.value,
        owner_user_id="u-andrew",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
    )
    await db.insert_memory(same_doc_memory)
    await db.add_memory_source(
        same_doc_memory.id,
        created.concept_id,
        "agent_session",
        excerpt="A separate scheduler claim under the same concept remains active.",
        source_updated_at=None,
    )

    updated = await service.apply_patch_proposal(
        proposal=_proposal(
            action="update_existing_claim",
            concept_id=created.concept_id,
            claim_id=created.claim_id,
            claim_text="Workspace source schedulers advance next_run_at after a successful claim.",
            durable_claim=_durable("Workspace source schedulers advance next_run_at after a successful claim."),
            reason="New evidence refines the scheduler lifecycle claim.",
            citations=["agent-window://codex/sess-2/sha256-window"],
        ),
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-2",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
        source_updated_at=None,
    )

    relation_runs = await _relation_runs_for_memory(db, updated.memory_id)
    candidates = await db.get_relation_candidates(relation_runs[-1]["id"])
    assert [candidate.memory_id for candidate in candidates] == [
        created.memory_id,
        same_doc_memory.id,
    ]
    assert [candidate.bucket for candidate in candidates] == [
        CandidateBucket.EXACT_SOURCE_ANCHOR,
        CandidateBucket.SAME_DOC_LINEAGE,
    ]
    assert relation_runs[-1]["mandatory_candidate_count"] == 2
    assert relation_runs[-1]["candidate_count"] == 2


@pytest.mark.asyncio
async def test_supersede_existing_claim_records_memory_lifecycle(bundle_stack):
    db, store, _collection = bundle_stack
    service = AgentKnowledgeBundleService(db=db, memory_store=store)
    created = await service.apply_patch_proposal(
        proposal=_proposal(),
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-1",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
        source_updated_at=None,
    )

    superseded = await service.apply_patch_proposal(
        proposal=_proposal(
            action="supersede_existing_claim",
            concept_id=created.concept_id,
            claim_id=created.claim_id,
            claim_text=(
                "The source scheduler startup claim is obsolete: the scheduler is "
                "now started by the cloud app bootstrap during lifespan startup."
            ),
            durable_claim=_durable("Source scheduler startup is owned by the cloud app bootstrap lifespan."),
            reason="New implementation replaced the older scheduler startup claim.",
        ),
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-2",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
        source_updated_at=None,
    )

    assert superseded.outcome == "applied"
    assert superseded.claim_id == created.claim_id
    new_memory = await db.get_memory(superseded.memory_id)
    old_memory = await db.get_memory(created.memory_id)
    assert new_memory is not None
    assert old_memory is not None
    assert new_memory.status == "active"
    assert old_memory.status == "superseded"
    assert old_memory.superseded_by == new_memory.id
    assert old_memory.replacement_reason == "New implementation replaced the older scheduler startup claim."
    assert old_memory.replacement_kind == "supersession"


@pytest.mark.asyncio
async def test_agent_claim_requires_structured_memory_projection(bundle_stack):
    db, store, _collection = bundle_stack
    service = AgentKnowledgeBundleService(db=db, memory_store=store)

    result = await service.apply_patch_proposal(
        proposal=_proposal(durable_claim=None),
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-1",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
        source_updated_at=None,
    )

    assert result.outcome == "parse_failed"
    assert result.reason == "durable_claim is required"


@pytest.mark.parametrize(
    "durable_claim",
    [
        {"rule": "   ", "scope": "Agent-session memory extraction.", "rationale": None},
        {"rule": "Durable rule.", "scope": "   ", "rationale": None},
    ],
)
def test_agent_claim_rejects_blank_durable_claim_fields(durable_claim):
    with pytest.raises(ValueError, match="must not be blank"):
        _proposal(durable_claim=durable_claim)


def test_agent_claim_ignores_blank_optional_rationale():
    proposal = _proposal(
        durable_claim={
            "rule": "Durable rule.",
            "scope": "Agent-session memory extraction.",
            "rationale": "   ",
        }
    )

    assert proposal.durable_claim is not None
    assert proposal.durable_claim.rationale is None


def test_agent_patch_rejects_stale_memory_content_field():
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        AgentKnowledgePatchProposal(
            action="create_new_concept",
            concept_type="debugging_takeaway",
            title="Stale schema",
            claim_text="The model emitted the previous patch schema.",
            memory_content="This stale field must fail loudly.",
            memory_type="fact",
            reason="stale schema",
        )


@pytest.mark.asyncio
async def test_agent_patch_result_carries_explicit_result_bucket(bundle_stack):
    db, store, _collection = bundle_stack
    service = AgentKnowledgeBundleService(db=db, memory_store=store)

    parsed_failure = await service.apply_patch_proposal(
        proposal=_proposal(durable_claim=None),
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-1",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
        source_updated_at=None,
    )
    no_output = await service.apply_patch_proposal(
        proposal=_proposal(action="no_output", claim_text="", durable_claim=None),
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-1",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
        source_updated_at=None,
    )

    assert parsed_failure.result_bucket == "failed"
    assert no_output.result_bucket == "no_output"


@pytest.mark.asyncio
async def test_no_output_validates_covered_concept_and_claim_ids(bundle_stack):
    db, store, _collection = bundle_stack
    service = AgentKnowledgeBundleService(db=db, memory_store=store)

    created = await service.apply_patch_proposal(
        proposal=_proposal(),
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-1",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
        source_updated_at=None,
    )

    covered = await service.apply_patch_proposal(
        proposal=_proposal(
            action="no_output",
            claim_text="",
            durable_claim=None,
            covered_concept_id=created.concept_id,
            covered_claim_id=created.claim_id,
        ),
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-1",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
        source_updated_at=None,
    )
    wrong_repo = await service.apply_patch_proposal(
        proposal=_proposal(
            action="no_output",
            claim_text="",
            durable_claim=None,
            covered_concept_id=created.concept_id,
            covered_claim_id=created.claim_id,
        ),
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-1",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/other",
        project_key="UNSORTED",
        source_updated_at=None,
    )
    wrong_claim = await service.apply_patch_proposal(
        proposal=_proposal(
            action="no_output",
            claim_text="",
            durable_claim=None,
            covered_concept_id=created.concept_id,
            covered_claim_id="claim-does-not-exist",
        ),
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-1",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
        source_updated_at=None,
    )

    assert covered.result_bucket == "no_output"
    assert covered.covered_concept_id == created.concept_id
    assert covered.covered_claim_id == created.claim_id
    assert wrong_repo.covered_concept_id is None
    assert wrong_repo.covered_claim_id is None
    assert wrong_claim.covered_concept_id == created.concept_id
    assert wrong_claim.covered_claim_id is None


@pytest.mark.asyncio
async def test_agent_claim_accepts_detailed_memory_projection(bundle_stack):
    db, store, _collection = bundle_stack
    service = AgentKnowledgeBundleService(db=db, memory_store=store)
    detailed_projection = " ".join(["Detailed projection remains valid when the flow needs context."] * 40)

    result = await service.apply_patch_proposal(
        proposal=_proposal(durable_claim=_durable(detailed_projection)),
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-1",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
        source_updated_at=None,
    )

    assert result.outcome == "applied"


@pytest.mark.asyncio
async def test_agent_evidence_unit_retry_is_idempotent(bundle_stack):
    db, store, collection = bundle_stack
    service = AgentKnowledgeBundleService(db=db, memory_store=store)
    proposal = _proposal(
        concept_id="akb_concept_retry",
        claim_id="akb_claim_retry",
    )
    first = await service.apply_patch_proposal(
        proposal=proposal,
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-retry",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
        source_updated_at=None,
    )
    second = await service.apply_patch_proposal(
        proposal=proposal,
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-retry",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
        source_updated_at=None,
    )

    assert first.outcome == "applied"
    assert second.outcome == "applied"
    assert second.memory_id == first.memory_id
    assert list(collection.upserted) == [first.memory_id]

    async with db.db.execute("SELECT COUNT(*) FROM memories") as cursor:
        row = await cursor.fetchone()
    assert row[0] == 1


@pytest.mark.asyncio
async def test_agent_evidence_unit_retry_is_idempotent_without_model_supplied_ids(bundle_stack):
    db, _store, collection = bundle_stack
    service = AgentKnowledgeBundleService(db=db, memory_store=_store)
    proposal = _proposal()

    first = await service.apply_patch_proposal(
        proposal=proposal,
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-retry-no-ids",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
        source_updated_at=None,
    )
    second = await service.apply_patch_proposal(
        proposal=proposal,
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-retry-no-ids",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
        source_updated_at=None,
    )

    assert first.outcome == "applied"
    assert second.outcome == "applied"
    assert second.concept_id == first.concept_id
    assert second.claim_id == first.claim_id
    assert second.memory_id == first.memory_id
    assert list(collection.upserted) == [first.memory_id]

    async with db.db.execute("SELECT COUNT(*) FROM agent_concepts") as cursor:
        concept_count = await cursor.fetchone()
    async with db.db.execute("SELECT COUNT(*) FROM agent_claims") as cursor:
        claim_count = await cursor.fetchone()
    async with db.db.execute("SELECT COUNT(*) FROM memories") as cursor:
        memory_count = await cursor.fetchone()
    assert concept_count[0] == 1
    assert claim_count[0] == 1
    assert memory_count[0] == 1


@pytest.mark.asyncio
async def test_agent_memory_prompt_uses_primary_evidence_as_authorization_boundary(bundle_stack):
    db, _store, _collection = bundle_stack

    prompt = await render_agent_knowledge_patch_prompt(
        db=db,
        owner_user_id="u-andrew",
        client="claude-code",
        session_id="sess-1",
        trigger="stop",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        branch="codex/example-branch",
        history_window={"kind": "transcript_window"},
        events=[
            {
                "evidence_id": "E1",
                "evidence_role": "supporting",
                "kind": "assistant_message",
                "text": "I verified the prompt tests and deployed the change.",
            },
            {
                "evidence_id": "E2",
                "evidence_role": "primary",
                "kind": "user_message",
                "text": "Yes, agent-session memories should be user-approved, not agent self-reasoning.",
            },
        ],
        transcript_markdown="",
    )

    assert "<primary_evidence>" in prompt
    assert "[E2:user_message] Yes, agent-session memories should be user-approved" in prompt
    assert "<supporting_evidence>" in prompt
    assert "[E1:assistant_message] I verified the prompt tests" in prompt
    assert "<candidate_evidence>" not in prompt


@pytest.mark.asyncio
async def test_agent_patch_missing_claim_text_is_failed_not_no_output(bundle_stack):
    db, store, _collection = bundle_stack
    service = AgentKnowledgeBundleService(db=db, memory_store=store)

    result = await service.apply_patch_proposal(
        proposal=_proposal(claim_text=""),
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-1",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
        source_updated_at=None,
    )

    assert result.outcome == "parse_failed"
    assert result.result_bucket == "failed"
    assert result.reason == "claim_text is required"


@pytest.mark.asyncio
async def test_private_concept_rejects_other_user_update(bundle_stack):
    db, store, _collection = bundle_stack
    service = AgentKnowledgeBundleService(db=db, memory_store=store)
    created = await service.apply_patch_proposal(
        proposal=_proposal(),
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-1",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
        source_updated_at=None,
    )

    rejected = await service.apply_patch_proposal(
        proposal=_proposal(
            action="update_existing_claim",
            concept_id=created.concept_id,
            claim_id=created.claim_id,
            claim_text="A different user must not be able to patch this private claim.",
        ),
        owner_user_id="u-test001",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-evil",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
        source_updated_at=None,
    )

    assert rejected.outcome == "rejected_scope"
    memory = await db.get_memory(created.memory_id)
    assert memory is not None
    assert "different user" not in memory.content
