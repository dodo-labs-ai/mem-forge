from __future__ import annotations

from datetime import datetime, timezone

import pytest

from memforge.agent_knowledge import (
    AgentKnowledgeBundleService,
    AgentKnowledgePatchProposal,
    render_agent_knowledge_patch_prompt,
)
from memforge.memory.store import MemoryStore
from memforge.models import Visibility
from memforge.storage.adapters.sqlite import build_sqlite_adapters
from memforge.storage.database import Database


class RecordingCollection:
    def __init__(self) -> None:
        self.upserted: dict[str, dict] = {}
        self.deleted: list[str] = []

    def query(self, **kwargs):
        return {"ids": [[]], "distances": [[]]}

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


def _proposal(**overrides) -> AgentKnowledgePatchProposal:
    base = {
        "action": "create_new_concept",
        "concept_type": "debugging_takeaway",
        "title": "Source scheduler lifecycle",
        "claim_text": (
            "Workspace source schedulers must start during app startup so overdue "
            "source schedules run without UI traffic."
        ),
        "memory_content": "Workspace source schedulers must start during app startup so overdue schedules run without UI traffic.",
        "memory_type": "procedure",
        "tags": ["scheduler", "source-sync"],
        "reason": "The window confirms a durable scheduler invariant.",
        "confidence": 0.9,
        "citations": ["agent-window://codex/sess-1/sha256-window"],
    }
    base.update(overrides)
    return AgentKnowledgePatchProposal(**base)


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
    assert memory.content == "Workspace source schedulers must start during app startup so overdue schedules run without UI traffic."
    assert "overdue source schedules" in (memory.extraction_context or "")
    assert collection.upserted[result.memory_id]["owner_user_id"] == "u-andrew"


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
    )

    updated = await service.apply_patch_proposal(
        proposal=_proposal(
            action="update_existing_claim",
            concept_id=created.concept_id,
            claim_id=created.claim_id,
            claim_text=(
                "Workspace source schedulers must start during app startup, claim due "
                "source schedules, and advance next_run_at after a successful claim."
            ),
            memory_content="Source schedulers start on app startup, claim due schedules, and advance next_run_at after success.",
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
    )

    assert updated.outcome == "applied"
    assert updated.concept_id == created.concept_id
    assert updated.claim_id == created.claim_id
    assert updated.memory_id != created.memory_id

    memory = await db.get_memory(updated.memory_id)
    assert memory is not None
    assert memory.content == "Source schedulers start on app startup, claim due schedules, and advance next_run_at after success."
    assert "advance next_run_at" in (memory.extraction_context or "")
    assert collection.upserted[updated.memory_id]["content_hash"] == memory.content_hash
    assert created.memory_id in collection.deleted

    old_memory = await db.get_memory(created.memory_id)
    assert old_memory is not None
    assert old_memory.status == "superseded"
    assert old_memory.superseded_by == updated.memory_id
    assert old_memory.replacement_reason == "New evidence refines the scheduler lifecycle claim."
    assert old_memory.replacement_kind == "revision"

    claim = await db.get_agent_claim(created.claim_id)
    assert claim is not None
    assert claim["memory_id"] == updated.memory_id
    assert "advance next_run_at" in claim["claim_text"]

    citations = await db.list_agent_claim_citations(created.claim_id)
    assert [citation["citation_url"] for citation in citations] == [
        "agent-window://codex/sess-1/sha256-window",
        "agent-window://codex/sess-2/sha256-window",
    ]


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
            memory_content="Source scheduler startup is owned by the cloud app bootstrap lifespan.",
            reason="New implementation replaced the older scheduler startup claim.",
        ),
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-2",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
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
        proposal=_proposal(memory_content=None),
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-1",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
    )

    assert result.outcome == "parse_failed"
    assert result.reason == "memory_content is required"


@pytest.mark.asyncio
async def test_agent_patch_result_carries_explicit_result_bucket(bundle_stack):
    db, store, _collection = bundle_stack
    service = AgentKnowledgeBundleService(db=db, memory_store=store)

    parsed_failure = await service.apply_patch_proposal(
        proposal=_proposal(memory_content=None),
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-1",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
    )
    no_output = await service.apply_patch_proposal(
        proposal=_proposal(action="no_output", claim_text="", memory_content=None),
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-1",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
    )

    assert parsed_failure.result_bucket == "failed"
    assert no_output.result_bucket == "no_output"


@pytest.mark.asyncio
async def test_agent_claim_accepts_detailed_memory_projection(bundle_stack):
    db, store, _collection = bundle_stack
    service = AgentKnowledgeBundleService(db=db, memory_store=store)
    detailed_projection = " ".join(["Detailed projection remains valid when the flow needs context."] * 40)

    result = await service.apply_patch_proposal(
        proposal=_proposal(memory_content=detailed_projection),
        owner_user_id="u-andrew",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="sess-1",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
    )

    assert result.outcome == "applied"


@pytest.mark.asyncio
async def test_agent_memory_prompt_describes_durable_memory_not_retrieval_projection(bundle_stack):
    db, _store, _collection = bundle_stack

    prompt = await render_agent_knowledge_patch_prompt(
        db=db,
        owner_user_id="u-andrew",
        client="codex",
        session_id="sess-1",
        trigger="stop",
        workspace="/workspace/memforge-cloud",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        branch="main",
        history_window={"kind": "transcript_window"},
        events=[{"kind": "decision", "text": "Use immutable memory revisions."}],
        transcript_markdown="",
    )

    assert "durable memory record" in prompt
    assert "retrieval-ready" not in prompt
    assert "memory projection" not in prompt


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
    )

    assert rejected.outcome == "rejected_scope"
    memory = await db.get_memory(created.memory_id)
    assert memory is not None
    assert "different user" not in memory.content
