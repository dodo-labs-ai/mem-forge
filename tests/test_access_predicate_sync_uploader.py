"""Two uploaders submit agent-session documents; the gene -> sync chain must
carry the uploader through normalization into the persistence pipeline.

This guards the production write path: receipt metadata alone is not enough,
because the gene re-reads the package and feeds the sync pipeline. The
normalized source_semantics has to expose the uploader so the sync pipeline can
forward it to the memory engine, instead of falling back to LOCAL_DEV_USER_ID.

The orchestrator-driven test below is the regression gate: it runs the real
GeneSyncOrchestrator against a real AgentSessionGene and watches what the
orchestrator forwards to the memory engine. Removing the ``user_id=...`` kwarg
from either branch of ``_process_item`` (the new-document ``process_memories``
call or the existing-document ``reconcile_and_persist`` call) makes the
spy-engine assertions fail.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from memforge.agent_sessions import submit_agent_session_document
from memforge.config import AppConfig
from memforge.genes.agent_session_gene import AgentSessionGene
from memforge.models import (
    EnrichmentResult,
    MemoryExtractionResult,
    RawMemory,
)
from memforge.pipeline.sync import GeneSyncOrchestrator
from memforge.storage.adapters.context import AccessScope
from memforge.storage.database import Database


U1_USER = "u-1"
U2_USER = "u-2"


def _config(tmp_path: Path) -> AppConfig:
    cfg = AppConfig(base_dir=tmp_path / "mem")
    cfg.llm.enrichment_api_key = ""
    cfg.llm.embedding_api_key = ""
    return cfg


def _personalized_scope(user_id: str) -> AccessScope:
    return AccessScope(
        user_id=user_id,
        include_private=True,
        allowed_statuses=("active",),
        active_project=None,
        scope_mode="project-first",
    )


@pytest.fixture
async def database_fixture(tmp_path):
    database = Database(str(tmp_path / "sync_uploader.db"))
    await database.connect()
    yield database
    await database.close()


# ---------------------------------------------------------------------------
# Test doubles for the orchestrator-driven test
# ---------------------------------------------------------------------------


class _StubDocumentStore:
    def store_raw(self, *, source_name, title, content, content_type, extension=None):
        suffix = extension or ".raw"
        return f"file:///tmp/{source_name}/{title}{suffix}"

    def store_normalized(self, *, source_name, title, markdown):
        return f"file:///tmp/{source_name}/{title}.md"

    def delete_document_files(self, *, source_name, title):
        return None


class _StubEnricher:
    """Returns an empty enrichment so the orchestrator advances to memory extraction."""

    async def enrich_document(self, *, doc_id, content, source_type):
        return EnrichmentResult(
            summary="agent session summary",
            tags=[],
            entities=[],
            relationships=[],
            doc_type="agent_session_summary",
            complexity="low",
        )


class _SingleMemoryExtractor:
    """Yields one RawMemory per call so the orchestrator reaches process_memories."""

    async def extract_memories(self, **kwargs):
        return MemoryExtractionResult(
            memories=[
                RawMemory(
                    memory_type="fact",
                    content="durable design fact",
                    entity_refs=[],
                    tags=[],
                    confidence=0.9,
                )
            ],
        )

    async def extract_memory_changes(self, **kwargs):
        return MemoryExtractionResult(
            memories=[
                RawMemory(
                    memory_type="fact",
                    content="durable design fact",
                    entity_refs=[],
                    tags=[],
                    confidence=0.9,
                )
            ],
        )

    async def extract_unit_memories(self, context, **kwargs):
        return MemoryExtractionResult(
            memories=[
                RawMemory(
                    memory_type="fact",
                    content="durable design fact",
                    entity_refs=[],
                    tags=[],
                    confidence=0.9,
                )
            ],
        )


class _SpyMemoryEngine:
    """Records every kwarg the orchestrator forwards on memory persistence calls.

    The assertions read from these recordings, never from values the test passed
    in itself. Deleting ``user_id=uploader_user_id`` from either branch of
    ``GeneSyncOrchestrator._process_item`` flips ``user_id`` to ``None`` here
    and the test fails.
    """

    def __init__(self) -> None:
        self.process_memories_calls: list[dict] = []
        self.reconcile_calls: list[dict] = []

    async def process_enrichment(self, *, doc_id, enrichment, doc_context=None):
        return []

    async def process_memories(self, **kwargs):
        self.process_memories_calls.append(kwargs)
        return {"inserted": len(kwargs.get("raw_memories") or []), "corroborated": 0, "skipped": 0}

    async def reconcile_and_persist(self, **kwargs):
        self.reconcile_calls.append(kwargs)
        return {"added": 1, "updated": 0, "superseded": 0, "deleted": 0, "noop": 0}


class _SpySourceSupportDetector:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def detect_and_persist(self, **kwargs):
        self.calls.append(kwargs)
        return {"added": 0, "updated": 0, "removed_stale": 0}


async def _submit_and_normalize(
    *,
    db: Database,
    cfg: AppConfig,
    client: str,
    session_id: str,
    user_id: str,
    fact: str,
    repo: str = "mem-forge",
    submitted_at: str | None = None,
    source_observed_at: str | None = None,
):
    """Run the production write path and return (submitted, item, normalized)."""
    submitted = await submit_agent_session_document(
        db=db,
        config=cfg,
        client=client,
        session_id=session_id,
        trigger="Stop",
        document_markdown=f"## Durable Findings\n- {fact}\n",
        workspace="/workspace/mem-forge",
        repo=repo,
        branch="main",
        commit_sha="abc",
        history_window_kind="session",
        user_id=user_id,
        submitted_at=submitted_at,
        source_observed_at=source_observed_at,
    )
    source = await db.get_source(submitted["source_id"])
    gene = AgentSessionGene(config=source["config"], source_id=source["id"])

    items = [item async for item in gene.discover()]
    item = next(it for it in items if it.item_id == submitted["doc_id"])
    raw = await gene.fetch(item)
    normalized = await gene.normalize(raw)
    return submitted, item, normalized


@pytest.mark.asyncio
async def test_agent_session_gene_exposes_uploader_on_normalize(database_fixture, tmp_path):
    """Gene boundary: ``normalize()`` must surface the uploader hint.

    Without this hint on ``source_semantics``, the sync pipeline has nothing to
    forward downstream and the memory silently falls back to LOCAL_DEV_USER_ID.
    """
    database = database_fixture
    cfg = _config(tmp_path)

    _, _, normalized_u1 = await _submit_and_normalize(
        db=database,
        cfg=cfg,
        client="codex",
        session_id="sess-u1",
        user_id=U1_USER,
        fact="u1 deploys via argo",
    )
    _, _, normalized_u2 = await _submit_and_normalize(
        db=database,
        cfg=cfg,
        client="claude-code",
        session_id="sess-u2",
        user_id=U2_USER,
        fact="u2 deploys via flux",
    )

    assert normalized_u1.source_semantics.get("uploader_user_id") == U1_USER
    assert normalized_u2.source_semantics.get("uploader_user_id") == U2_USER


@pytest.mark.asyncio
async def test_agent_session_gene_exposes_normalized_repo_identifier(database_fixture, tmp_path):
    """Gene boundary: repo identity must survive the package read path.

    Curator grouping depends on repo_identifier at memory persistence time. A
    receipt-only value is insufficient because sync rereads the generated
    package through the gene before extraction.
    """
    database = database_fixture
    cfg = _config(tmp_path)

    _, _, normalized = await _submit_and_normalize(
        db=database,
        cfg=cfg,
        client="codex",
        session_id="sess-repo",
        user_id=U1_USER,
        fact="agent-session memories are repo-first",
        repo="git@github.com:shno-labs/mem-forge.git",
    )

    assert normalized.source_semantics["repo_identifier"] == "github.com/shno-labs/mem-forge"


@pytest.mark.asyncio
async def test_agent_session_gene_exposes_explicit_source_observed_at(database_fixture, tmp_path):
    """Gene boundary: explicit source observation time survives package reads.

    Submission time is lifecycle metadata. It must not be copied into source
    provenance when the uploader did not provide an absolute source timestamp.
    """
    database = database_fixture
    cfg = _config(tmp_path)

    _, _, normalized = await _submit_and_normalize(
        db=database,
        cfg=cfg,
        client="codex",
        session_id="sess-source-observed",
        user_id=U1_USER,
        fact="agent-session source timestamps are explicit provenance",
        source_observed_at="2026-06-20T04:23:51Z",
    )

    assert normalized.source_semantics["source_observed_at"] == "2026-06-20T04:23:51+00:00"


@pytest.mark.asyncio
async def test_orchestrator_forwards_uploader_user_id_on_new_documents(database_fixture, tmp_path):
    """First sync (new docs): the orchestrator MUST forward each uploader's id
    on its ``process_memories`` call.

    The spy engine records the kwargs the orchestrator passed; the assertion
    reads them back. If ``user_id=uploader_user_id`` is removed from the
    new-document branch (sync.py around line 1048), the recorded ``user_id``
    becomes ``None`` and this test fails.
    """
    database = database_fixture
    cfg = _config(tmp_path)

    submitted_u1, item_u1, _ = await _submit_and_normalize(
        db=database,
        cfg=cfg,
        client="codex",
        session_id="sess-u1",
        user_id=U1_USER,
        fact="u1 deploys via argo",
    )
    submitted_u2, item_u2, _ = await _submit_and_normalize(
        db=database,
        cfg=cfg,
        client="claude-code",
        session_id="sess-u2",
        user_id=U2_USER,
        fact="u2 deploys via flux",
    )

    # Two clients map to two per-client sources; sync each through the real
    # orchestrator with a spy engine on the persistence boundary.
    spy = _SpyMemoryEngine()
    orchestrator = GeneSyncOrchestrator(
        db=database,
        doc_store=_StubDocumentStore(),
        enricher=_StubEnricher(),
        memory_extractor=_SingleMemoryExtractor(),
        memory_engine=spy,
        memory_store=None,
        max_concurrent=1,
    )

    src_u1 = await database.get_source(submitted_u1["source_id"])
    src_u2 = await database.get_source(submitted_u2["source_id"])
    gene_u1 = AgentSessionGene(config=src_u1["config"], source_id=src_u1["id"])
    gene_u2 = AgentSessionGene(config=src_u2["config"], source_id=src_u2["id"])

    state_u1 = await orchestrator.sync_gene(
        gene=gene_u1,
        source_name=src_u1["name"],
        source_id=src_u1["id"],
    )
    state_u2 = await orchestrator.sync_gene(
        gene=gene_u2,
        source_name=src_u2["name"],
        source_id=src_u2["id"],
    )

    assert state_u1.last_sync_status == "success"
    assert state_u2.last_sync_status == "success"

    # New documents -> the orchestrator goes through process_memories, not
    # reconcile_and_persist. Both calls must carry the uploader's user_id.
    by_doc = {call["doc_id"]: call for call in spy.process_memories_calls}
    assert item_u1.item_id in by_doc
    assert item_u2.item_id in by_doc
    assert by_doc[item_u1.item_id]["user_id"] == U1_USER
    assert by_doc[item_u2.item_id]["user_id"] == U2_USER
    assert by_doc[item_u1.item_id]["repo_identifier"] == "mem-forge"
    assert by_doc[item_u2.item_id]["repo_identifier"] == "mem-forge"
    assert spy.reconcile_calls == []


@pytest.mark.asyncio
async def test_orchestrator_forwards_source_observed_at_on_new_documents(database_fixture, tmp_path):
    """First sync (new docs): source provenance timestamp reaches the memory engine."""
    database = database_fixture
    cfg = _config(tmp_path)

    submitted, item, _ = await _submit_and_normalize(
        db=database,
        cfg=cfg,
        client="codex",
        session_id="sess-source-observed-new",
        user_id=U1_USER,
        fact="new document carries source observed timestamp",
        source_observed_at="2026-06-20T04:23:51Z",
    )

    spy = _SpyMemoryEngine()
    orchestrator = GeneSyncOrchestrator(
        db=database,
        doc_store=_StubDocumentStore(),
        enricher=_StubEnricher(),
        memory_extractor=_SingleMemoryExtractor(),
        memory_engine=spy,
        memory_store=None,
        max_concurrent=1,
    )

    src = await database.get_source(submitted["source_id"])
    gene = AgentSessionGene(config=src["config"], source_id=src["id"])
    state = await orchestrator.sync_gene(gene=gene, source_name=src["name"], source_id=src["id"])

    assert state.last_sync_status == "success"
    by_doc = {call["doc_id"]: call for call in spy.process_memories_calls}
    assert by_doc[item.item_id]["source_observed_at"].isoformat() == "2026-06-20T04:23:51+00:00"
    assert spy.reconcile_calls == []


@pytest.mark.asyncio
async def test_orchestrator_forwards_source_observed_at_to_source_support_detector(database_fixture, tmp_path):
    """Source-support detection must receive the same explicit provenance timestamp."""
    database = database_fixture
    cfg = _config(tmp_path)

    submitted, _, _ = await _submit_and_normalize(
        db=database,
        cfg=cfg,
        client="codex",
        session_id="sess-source-observed-support",
        user_id=U1_USER,
        fact="source support detector carries source observed timestamp",
        source_observed_at="2026-06-20T04:23:51Z",
    )

    support_detector = _SpySourceSupportDetector()
    orchestrator = GeneSyncOrchestrator(
        db=database,
        doc_store=_StubDocumentStore(),
        enricher=_StubEnricher(),
        memory_extractor=_SingleMemoryExtractor(),
        memory_engine=_SpyMemoryEngine(),
        memory_store=None,
        source_support_detector=support_detector,
        max_concurrent=1,
    )

    src = await database.get_source(submitted["source_id"])
    state = await orchestrator.sync_gene(
        gene=AgentSessionGene(config=src["config"], source_id=src["id"]),
        source_name=src["name"],
        source_id=src["id"],
    )

    assert state.last_sync_status == "success"
    assert support_detector.calls[0]["source_observed_at"].isoformat() == "2026-06-20T04:23:51+00:00"


@pytest.mark.asyncio
async def test_orchestrator_forwards_uploader_user_id_on_document_updates(database_fixture, tmp_path):
    """Second sync (existing doc, new content): the orchestrator MUST forward
    the uploader's id on its ``reconcile_and_persist`` call.

    If ``user_id=uploader_user_id`` is removed from the update branch (sync.py
    around line 1036), the recorded ``user_id`` becomes ``None`` and this test
    fails.
    """
    database = database_fixture
    cfg = _config(tmp_path)

    # First submission: seeds the document so the second sync hits the update path.
    submitted_first, item_first, _ = await _submit_and_normalize(
        db=database,
        cfg=cfg,
        client="codex",
        session_id="sess-update",
        user_id=U1_USER,
        fact="initial deploy via argo",
        submitted_at="2026-01-01T00:00:00+00:00",
    )

    src = await database.get_source(submitted_first["source_id"])
    initial_spy = _SpyMemoryEngine()
    initial_orchestrator = GeneSyncOrchestrator(
        db=database,
        doc_store=_StubDocumentStore(),
        enricher=_StubEnricher(),
        memory_extractor=_SingleMemoryExtractor(),
        memory_engine=initial_spy,
        memory_store=None,
        max_concurrent=1,
    )
    initial_state = await initial_orchestrator.sync_gene(
        gene=AgentSessionGene(config=src["config"], source_id=src["id"]),
        source_name=src["name"],
        source_id=src["id"],
    )
    assert initial_state.last_sync_status == "success"
    assert initial_spy.process_memories_calls
    assert initial_spy.reconcile_calls == []

    # Second submission: same client + session + trigger -> same doc_id, with
    # different markdown (and therefore a different content_hash). The next
    # sync must take the update branch (reconcile_and_persist).
    submitted_second, item_second, _ = await _submit_and_normalize(
        db=database,
        cfg=cfg,
        client="codex",
        session_id="sess-update",
        user_id=U1_USER,
        fact="updated deploy via flux",
        submitted_at="2026-01-02T00:00:00+00:00",
    )
    assert submitted_second["doc_id"] == submitted_first["doc_id"]
    assert item_second.item_id == item_first.item_id

    update_spy = _SpyMemoryEngine()
    update_orchestrator = GeneSyncOrchestrator(
        db=database,
        doc_store=_StubDocumentStore(),
        enricher=_StubEnricher(),
        memory_extractor=_SingleMemoryExtractor(),
        memory_engine=update_spy,
        memory_store=None,
        max_concurrent=1,
    )
    update_state = await update_orchestrator.sync_gene(
        gene=AgentSessionGene(config=src["config"], source_id=src["id"]),
        source_name=src["name"],
        source_id=src["id"],
        force_full_sync=True,
    )

    assert update_state.last_sync_status == "success"
    # Existing document with changed content -> reconcile_and_persist branch.
    assert update_spy.reconcile_calls, (
        "expected the orchestrator to take the reconcile_and_persist branch "
        "for an existing document with changed content"
    )
    forwarded = update_spy.reconcile_calls[0]
    assert forwarded["doc_id"] == item_second.item_id
    assert forwarded["user_id"] == U1_USER
    assert forwarded["repo_identifier"] == "mem-forge"


@pytest.mark.asyncio
async def test_orchestrator_forwards_source_observed_at_on_document_updates(database_fixture, tmp_path):
    """Second sync (existing doc, new content): provenance timestamp reaches reconciliation."""
    database = database_fixture
    cfg = _config(tmp_path)

    submitted_first, item_first, _ = await _submit_and_normalize(
        db=database,
        cfg=cfg,
        client="codex",
        session_id="sess-source-observed-update",
        user_id=U1_USER,
        fact="initial source observed value",
        source_observed_at="2026-06-20T04:23:51Z",
        submitted_at="2026-01-01T00:00:00+00:00",
    )

    src = await database.get_source(submitted_first["source_id"])
    initial_spy = _SpyMemoryEngine()
    initial_orchestrator = GeneSyncOrchestrator(
        db=database,
        doc_store=_StubDocumentStore(),
        enricher=_StubEnricher(),
        memory_extractor=_SingleMemoryExtractor(),
        memory_engine=initial_spy,
        memory_store=None,
        max_concurrent=1,
    )
    initial_state = await initial_orchestrator.sync_gene(
        gene=AgentSessionGene(config=src["config"], source_id=src["id"]),
        source_name=src["name"],
        source_id=src["id"],
    )
    assert initial_state.last_sync_status == "success"
    assert initial_spy.process_memories_calls

    submitted_second, item_second, _ = await _submit_and_normalize(
        db=database,
        cfg=cfg,
        client="codex",
        session_id="sess-source-observed-update",
        user_id=U1_USER,
        fact="updated source observed value",
        source_observed_at="2026-06-21T05:00:00Z",
        submitted_at="2026-01-02T00:00:00+00:00",
    )
    assert submitted_second["doc_id"] == submitted_first["doc_id"]
    assert item_second.item_id == item_first.item_id

    update_spy = _SpyMemoryEngine()
    update_orchestrator = GeneSyncOrchestrator(
        db=database,
        doc_store=_StubDocumentStore(),
        enricher=_StubEnricher(),
        memory_extractor=_SingleMemoryExtractor(),
        memory_engine=update_spy,
        memory_store=None,
        max_concurrent=1,
    )
    update_state = await update_orchestrator.sync_gene(
        gene=AgentSessionGene(config=src["config"], source_id=src["id"]),
        source_name=src["name"],
        source_id=src["id"],
        force_full_sync=True,
    )

    assert update_state.last_sync_status == "success"
    by_doc = {call["doc_id"]: call for call in update_spy.reconcile_calls}
    assert by_doc[item_second.item_id]["source_observed_at"].isoformat() == "2026-06-21T05:00:00+00:00"
    assert update_spy.process_memories_calls == []
