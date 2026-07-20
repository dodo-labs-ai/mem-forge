"""E2E tests: contradiction detection with real LLM calls.

Tests the full pipeline from memory insertion through LLM classification to
DB persistence. Uses a self-contained test DB (no prod dependency) and real
LLM calls through the local proxy when available.

Run:  .venv/bin/python -m pytest tests/test_contradiction_e2e.py -v -s
"""

from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from memforge.llm.structured import (
    ContradictionDecision,
    ContradictionResponse,
    LiteLlmStructuredClient,
    StructuredLlmConfig,
)
from memforge.memory.evidence import (
    EvidenceContentProvenance,
    EvidenceReference,
    EvidenceRole,
    EvidenceUnit,
    MemorySupportAssertion,
)
from memforge.memory.store import MemoryStore
from memforge.models import (
    ContentItem,
    Memory,
    NormalizedContent,
    RawContent,
    content_hash,
)
from memforge.pipeline.contradiction_detector import detect_cross_doc_contradictions
from memforge.pipeline.source_projection_adapters import project_source_item
from memforge.source_projection import AnchorKind, SourceAnchor
from memforge.storage.database import Database
from memforge.storage.adapters.sqlite import build_sqlite_adapters

LLM_BASE_URL = os.environ.get(
    "MEMFORGE_E2E_ANTHROPIC_BASE_URL",
    os.environ.get("MEMFORGE_ENRICHMENT_BASE_URL", "http://localhost:6655/anthropic"),
)


def _read_local_env_key() -> str:
    env_file = Path(__file__).resolve().parents[1] / ".env.local"
    if not env_file.exists():
        return ""

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() in {
            "MEMFORGE_E2E_ANTHROPIC_API_KEY",
            "MEMFORGE_ENRICHMENT_API_KEY",
            "ANTHROPIC_API_KEY",
        }:
            return value.strip().strip("\"'")
    return ""


def _llm_api_key() -> str:
    return (
        os.environ.get("MEMFORGE_E2E_ANTHROPIC_API_KEY")
        or os.environ.get("MEMFORGE_ENRICHMENT_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or _read_local_env_key()
    )


def _can_reach_llm() -> bool:
    """Check if the local LLM proxy is reachable."""
    if not _llm_api_key():
        return False
    try:
        import httpx

        r = httpx.get(LLM_BASE_URL, follow_redirects=True, timeout=3)
        return r.status_code < 500
    except Exception:
        return False


skip_no_llm = pytest.mark.skipif(
    not _can_reach_llm(),
    reason="Local LLM proxy not reachable or API key not configured",
)


def _make_memory(mem_id: str, content: str, mem_type: str = "fact") -> Memory:
    now = datetime.now(timezone.utc)
    return Memory(
        id=mem_id,
        memory_type=mem_type,
        content=content,
        content_hash=content_hash(content),
        project_key="TEST",
        entity_refs=[],
        tags=[],
        confidence=0.9,
        corroboration_count=1,
        contradiction_count=0,
        valid_from=None,
        valid_until=None,
        created_at=now,
        updated_at=now,
        status="active",
        extraction_context=None,
    )


def _project_document(
    *,
    source_id: str,
    doc_id: str,
    body: str,
):
    item = ContentItem(
        item_id=doc_id,
        title=doc_id,
        source_url=f"http://test/{doc_id}",
        last_modified=datetime.now(timezone.utc),
        version="1",
        extra={"page_id": doc_id, "space_key": "TEST"},
    )
    return project_source_item(
        source_id=source_id,
        source_type="confluence",
        run_id=f"projection-{doc_id}",
        item=item,
        raw=RawContent(
            item=item,
            body=body.encode(),
            content_type="text/plain",
        ),
        normalized=NormalizedContent(item=item, markdown_body=body),
    )


async def _seed_support(
    db: Database,
    *,
    memory: Memory,
    source_id: str,
    doc_id: str,
    projection,
) -> None:
    observation = projection.observations[0]
    revision = next(
        item
        for item in projection.observation_revisions
        if item.observation_id == observation.id
    )
    unit = EvidenceUnit(
        id=f"eu-{memory.id}",
        source_id=source_id,
        doc_id=doc_id,
        doc_revision_id=projection.source_unit_revisions[0].id,
        source_type="confluence",
        source_anchor=observation.id,
        source_lineage_id=projection.source_units[0].id,
        project_key="TEST",
        visibility="workspace",
        owner_user_id=None,
        repo_identifier=None,
        content=revision.content,
        excerpt=memory.content,
        evidence_provenance=EvidenceContentProvenance.SOURCE_EXCERPT,
        access_context_hash="workspace-test",
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
            id=f"support-{memory.id}",
            memory_id=memory.id,
            evidence_reference_id=reference.id or "",
            source_id=source_id,
            access_context_hash="workspace-test",
        )
    )


async def _persist_projected_memory(
    db: Database,
    *,
    memory: Memory,
    entity_id: int,
    source_id: str,
    doc_id: str,
) -> None:
    """Persist a claim whose current Observation contains the claim itself."""
    await db.upsert_source(
        id=source_id,
        type="confluence",
        name=source_id,
        config_json="{}",
        access_policy="workspace",
        owner_user_id="e2e-owner",
    )
    now = datetime.now(timezone.utc).isoformat()
    await db.db.execute(
        """INSERT INTO documents
           (doc_id, source, source_url, title, space_or_project,
            last_modified, version, content_hash, last_synced)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            doc_id,
            source_id,
            f"http://test/{doc_id}",
            doc_id,
            "TEST",
            now,
            "1",
            f"hash-{doc_id}",
            now,
        ),
    )
    await db.insert_memory(memory)
    await db.add_memory_source(
        memory.id,
        doc_id,
        "confluence",
        memory.content,
        source_updated_at=None,
    )
    await db.link_memory_entity(memory.id, entity_id)
    projection = _project_document(
        source_id=source_id,
        doc_id=doc_id,
        body=memory.content,
    )
    await db.record_source_projection(projection)
    await _seed_support(
        db,
        memory=memory,
        source_id=source_id,
        doc_id=doc_id,
        projection=projection,
    )


class StubChromaCollection:
    def delete(self, *, ids) -> None:
        pass

    def query(self, **kwargs):
        return {"ids": [[]], "distances": [[]]}

    def upsert(self, **kwargs) -> None:
        pass


def _test_memory_store(db: Database) -> MemoryStore:
    adapters = build_sqlite_adapters(db, StubChromaCollection())
    return MemoryStore(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
    )


# ---------------------------------------------------------------------------
# Self-contained DB fixture (no prod dependency)
# ---------------------------------------------------------------------------


@pytest.fixture
async def db(tmp_path):
    """Create a fresh test database with schema."""
    db_path = str(tmp_path / "test_e2e.db")
    database = Database(db_path)
    await database.connect()
    yield database
    await database.close()


@pytest.fixture
async def seeded_db(db):
    """Database seeded with two documents, shared entities, and memories.

    Layout:
      doc-arch (Architecture Doc)  ──┐
        mem-arch-pg: "pay-api uses PostgreSQL 14 on port 5432"  ── entity: postgresql
        mem-arch-kafka: "Kafka retention is 7 days"               ── entity: kafka
      doc-runbook (Runbook)  ────────┘
        mem-run-pg: "pay-api migrated to MySQL 8 in Q1 2026"   ── entity: postgresql
    """
    now = datetime.now(timezone.utc).isoformat()

    # Two independent sources and their documents.
    for source_id in ("src-arch", "src-runbook"):
        await db.upsert_source(
            id=source_id,
            type="confluence",
            name=source_id,
            config_json="{}",
            access_policy="workspace",
            owner_user_id="e2e-owner",
        )
    for doc_id, title, source_id in [
        ("doc-arch", "Architecture Doc", "src-arch"),
        ("doc-runbook", "Runbook", "src-runbook"),
    ]:
        await db.db.execute(
            """INSERT INTO documents
               (doc_id, source, source_url, title, space_or_project, last_modified, version, content_hash, last_synced)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (doc_id, source_id, f"http://test/{doc_id}", title, "TEST", now, "1", f"hash-{doc_id}", now),
        )

    # Shared entity: postgresql
    pg_id = await db.upsert_entity("postgresql", display_name="PostgreSQL", tags=["technology"])

    # Separate entity: kafka
    kafka_id = await db.upsert_entity("kafka", display_name="Kafka", tags=["technology"])

    # Memory from doc-arch: PostgreSQL fact
    mem_arch_pg = _make_memory("mem-arch-pg01", "pay-api uses PostgreSQL 14 on port 5432")
    await db.insert_memory(mem_arch_pg)
    await db.add_memory_source(
        mem_arch_pg.id,
        "doc-arch",
        "confluence",
        mem_arch_pg.content,
        source_updated_at=None,
    )
    await db.db.execute(
        "INSERT INTO memory_entities (memory_id, entity_id) VALUES (?, ?)",
        (mem_arch_pg.id, pg_id),
    )

    # Memory from doc-arch: Kafka fact (no cross-doc overlap)
    mem_arch_kafka = _make_memory("mem-arch-kfk1", "Kafka retention is set to 7 days")
    await db.insert_memory(mem_arch_kafka)
    await db.add_memory_source(
        mem_arch_kafka.id,
        "doc-arch",
        "confluence",
        mem_arch_kafka.content,
        source_updated_at=None,
    )
    await db.db.execute(
        "INSERT INTO memory_entities (memory_id, entity_id) VALUES (?, ?)",
        (mem_arch_kafka.id, kafka_id),
    )

    # Memory from doc-runbook: contradicts PostgreSQL fact
    mem_run_pg = _make_memory("mem-run-pg001", "pay-api migrated to MySQL 8 in Q1 2026")
    await db.insert_memory(mem_run_pg)
    await db.add_memory_source(
        mem_run_pg.id,
        "doc-runbook",
        "confluence",
        mem_run_pg.content,
        source_updated_at=None,
    )
    await db.db.execute(
        "INSERT INTO memory_entities (memory_id, entity_id) VALUES (?, ?)",
        (mem_run_pg.id, pg_id),
    )

    await db.db.commit()
    arch_projection = _project_document(
        source_id="src-arch",
        doc_id="doc-arch",
        body=f"{mem_arch_pg.content}\n\n{mem_arch_kafka.content}",
    )
    run_projection = _project_document(
        source_id="src-runbook",
        doc_id="doc-runbook",
        body=mem_run_pg.content,
    )
    await db.record_source_projection(arch_projection)
    await db.record_source_projection(run_projection)
    for memory in (mem_arch_pg, mem_arch_kafka):
        await _seed_support(
            db,
            memory=memory,
            source_id="src-arch",
            doc_id="doc-arch",
            projection=arch_projection,
        )
    await _seed_support(
        db,
        memory=mem_run_pg,
        source_id="src-runbook",
        doc_id="doc-runbook",
        projection=run_projection,
    )

    return {
        "db": db,
        "pg_id": pg_id,
        "kafka_id": kafka_id,
        "mem_arch_pg": mem_arch_pg,
        "mem_arch_kafka": mem_arch_kafka,
        "mem_run_pg": mem_run_pg,
    }


@pytest.fixture
def structured_llm_client():
    """Real structured LiteLLM client connecting to the local proxy."""
    api_key = _llm_api_key()
    if not api_key:
        pytest.skip("Local LLM proxy API key not configured")
    return LiteLlmStructuredClient(
        StructuredLlmConfig(
            model="claude-sonnet-4-20250514",
            base_url=LLM_BASE_URL,
            api_key=api_key,
            timeout_s=300.0,
        )
    )


async def _get_llm_model(db: Database) -> str:
    """Read the LLM model from DB config, or fall back to default."""
    try:
        async with db.db.execute("SELECT enrichment_model FROM llm_config WHERE id=1") as cur:
            row = await cur.fetchone()
        if row:
            return row["enrichment_model"]
    except Exception:
        pass
    return "claude-sonnet-4-20250514"


# ===========================================================================
# E2E: Real LLM contradiction detection
# ===========================================================================


@skip_no_llm
class TestContradictionE2E:
    """End-to-end tests using real LLM calls through the local proxy."""

    @pytest.mark.asyncio
    async def test_detects_real_contradiction(self, seeded_db, structured_llm_client):
        """Two memories about the same entity with conflicting facts should be
        detected as CONTRADICTION or TEMPORAL by the real LLM."""
        s = seeded_db
        db = s["db"]

        stats = await detect_cross_doc_contradictions(
            new_memory_ids=[s["mem_run_pg"].id],
            doc_id="doc-runbook",
            db=db,
            memory_store=_test_memory_store(db),
            structured_llm_client=structured_llm_client,
        )

        print(f"\n  Memory A: {s['mem_arch_pg'].content}")
        print(f"  Memory B: {s['mem_run_pg'].content}")
        print(f"  Stats: {stats}")

        assert stats["checked"] >= 1, "Should have checked at least one pair"
        assert stats["contradictions"] + stats["temporal"] >= 1, (
            f"LLM should detect a conflict between PostgreSQL 14 and MySQL 8, got: {stats}"
        )

        # Verify DB state
        if stats["contradictions"] > 0:
            mem = await db.get_memory(s["mem_run_pg"].id)
            assert mem.contradiction_count >= 1

            async with db.db.execute(
                "SELECT * FROM memory_contradictions WHERE memory_id_a = ? OR memory_id_b = ?",
                (s["mem_run_pg"].id, s["mem_run_pg"].id),
            ) as cur:
                row = await cur.fetchone()
            assert row is not None, "Contradiction row should exist"
            assert row["classification"] in ("contradiction", "temporal")
            assert row["reason"], "LLM should provide a reason"
            print(f"  Classification: {row['classification']}")
            print(f"  Reason: {row['reason']}")

    @pytest.mark.asyncio
    async def test_unrelated_memories_not_flagged(self, seeded_db, structured_llm_client):
        """Two memories sharing an entity but about different topics should not
        be classified as contradictions."""
        s = seeded_db
        db = s["db"]

        # Insert an unrelated memory in doc-runbook sharing the postgresql entity
        mem_unrelated = _make_memory(
            "mem-run-unrel1",
            "The PostgreSQL team meets every Wednesday at 3pm for backlog grooming.",
        )
        await _persist_projected_memory(
            db,
            memory=mem_unrelated,
            entity_id=s["pg_id"],
            source_id="src-run-unrelated",
            doc_id="doc-run-unrelated",
        )

        stats = await detect_cross_doc_contradictions(
            new_memory_ids=[mem_unrelated.id],
            doc_id="doc-run-unrelated",
            db=db,
            memory_store=_test_memory_store(db),
            structured_llm_client=structured_llm_client,
        )

        print(f"\n  Memory A (arch): {s['mem_arch_pg'].content}")
        print(f"  Memory B (unrel): {mem_unrelated.content}")
        print(f"  Stats: {stats}")

        assert stats["contradictions"] == 0, f"Meeting schedule vs DB port should not be a contradiction, got: {stats}"

    @pytest.mark.asyncio
    async def test_temporal_update_detected(self, seeded_db, structured_llm_client):
        """A memory that updates a time-sensitive fact should be classified as
        TEMPORAL or CONTRADICTION (either is acceptable — both are non-UNRELATED)."""
        s = seeded_db
        db = s["db"]

        mem_temporal = _make_memory(
            "mem-run-temp1",
            "pay-api PostgreSQL was upgraded from version 14 to version 16 in March 2026.",
        )
        await _persist_projected_memory(
            db,
            memory=mem_temporal,
            entity_id=s["pg_id"],
            source_id="src-run-temporal",
            doc_id="doc-run-temporal",
        )

        stats = await detect_cross_doc_contradictions(
            new_memory_ids=[mem_temporal.id],
            doc_id="doc-run-temporal",
            db=db,
            memory_store=_test_memory_store(db),
            structured_llm_client=structured_llm_client,
        )

        print(f"\n  Memory A (arch): {s['mem_arch_pg'].content}")
        print(f"  Memory B (temporal): {mem_temporal.content}")
        print(f"  Stats: {stats}")

        # Either TEMPORAL or CONTRADICTION is acceptable for a version upgrade
        assert stats["checked"] >= 1
        assert stats["contradictions"] + stats["temporal"] >= 1, (
            f"Version upgrade should be detected as conflict or temporal update, got: {stats}"
        )

    @pytest.mark.asyncio
    async def test_multiple_contradictions_in_batch(self, seeded_db, structured_llm_client):
        """Multiple new memories from one doc, each contradicting a different
        existing memory, should all be detected in a single batch."""
        s = seeded_db
        db = s["db"]

        # Add another existing memory in doc-arch about kafka
        # (mem_arch_kafka already exists: "Kafka retention is set to 7 days")

        # New source/doc with two contradicting memories
        await db.upsert_source(
            id="src-ops",
            type="confluence",
            name="src-ops",
            config_json="{}",
            access_policy="workspace",
            owner_user_id="e2e-owner",
        )
        await db.db.execute(
            """INSERT INTO documents
               (doc_id, source, source_url, title, space_or_project, last_modified, version, content_hash, last_synced)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "doc-ops",
                "src-ops",
                "http://test/doc-ops",
                "Ops Runbook",
                "TEST",
                datetime.now(timezone.utc).isoformat(),
                "1",
                "hash-ops",
                datetime.now(timezone.utc).isoformat(),
            ),
        )

        # Contradicts PostgreSQL fact
        mem_ops_pg = _make_memory(
            "mem-ops-pg001",
            "pay-api no longer uses PostgreSQL — it was fully replaced by CockroachDB in Q4 2025.",
        )
        await db.insert_memory(mem_ops_pg)
        await db.add_memory_source(
            mem_ops_pg.id,
            "doc-ops",
            "confluence",
            mem_ops_pg.content,
            source_updated_at=None,
        )
        await db.db.execute(
            "INSERT INTO memory_entities (memory_id, entity_id) VALUES (?, ?)",
            (mem_ops_pg.id, s["pg_id"]),
        )

        # Contradicts Kafka retention fact
        mem_ops_kafka = _make_memory(
            "mem-ops-kfk01",
            "Kafka retention was increased from 7 days to 30 days to meet compliance requirements.",
        )
        await db.insert_memory(mem_ops_kafka)
        await db.add_memory_source(
            mem_ops_kafka.id,
            "doc-ops",
            "confluence",
            mem_ops_kafka.content,
            source_updated_at=None,
        )
        await db.db.execute(
            "INSERT INTO memory_entities (memory_id, entity_id) VALUES (?, ?)",
            (mem_ops_kafka.id, s["kafka_id"]),
        )
        await db.db.commit()
        ops_projection = _project_document(
            source_id="src-ops",
            doc_id="doc-ops",
            body=f"{mem_ops_pg.content}\n\n{mem_ops_kafka.content}",
        )
        await db.record_source_projection(ops_projection)
        for memory in (mem_ops_pg, mem_ops_kafka):
            await _seed_support(
                db,
                memory=memory,
                source_id="src-ops",
                doc_id="doc-ops",
                projection=ops_projection,
            )

        stats = await detect_cross_doc_contradictions(
            new_memory_ids=[mem_ops_pg.id, mem_ops_kafka.id],
            doc_id="doc-ops",
            db=db,
            memory_store=_test_memory_store(db),
            structured_llm_client=structured_llm_client,
        )

        print(f"\n  Batch stats: {stats}")
        print(f"  PG conflict:    {s['mem_arch_pg'].content}  vs  {mem_ops_pg.content}")
        print(f"  Kafka conflict:  {s['mem_arch_kafka'].content}  vs  {mem_ops_kafka.content}")

        # Both pairs should be checked
        assert stats["checked"] >= 2, f"Should check at least 2 pairs, got {stats['checked']}"
        # At least one conflict detected overall
        assert stats["contradictions"] + stats["temporal"] >= 1, (
            f"Should detect at least one conflict across the batch, got: {stats}"
        )


# ===========================================================================
# Integration: error resilience with mocked LLM
# ===========================================================================


class TestContradictionErrorResilience:
    """Tests malformed structured decisions without needing a live LLM."""

    @pytest.mark.asyncio
    async def test_unexpected_pair_index_fails_closed(self, seeded_db):
        """An out-of-range pair index cannot be mixed with durable decisions."""
        s = seeded_db
        db = s["db"]

        mock_client = AsyncMock()
        mock_client.detect_contradictions = AsyncMock(
            return_value=ContradictionResponse(
                decisions=[
                    ContradictionDecision(pair_index=999, classification="contradiction", reason="bogus"),
                    ContradictionDecision(pair_index=0, classification="contradiction", reason="real conflict"),
                ]
            )
        )

        stats = await detect_cross_doc_contradictions(
            new_memory_ids=[s["mem_run_pg"].id],
            doc_id="doc-runbook",
            db=db,
            memory_store=_test_memory_store(db),
            structured_llm_client=mock_client,
        )

        assert stats["contradictions"] == 0
        assert stats["checked"] == 1
        assert await db.get_pending_review_for_challenger(s["mem_run_pg"].id) is None
        async with db.db.execute(
            "SELECT COUNT(*) FROM relation_runs WHERE evidence_unit_id = ?",
            (f"eu-{s['mem_run_pg'].id}",),
        ) as cursor:
            assert (await cursor.fetchone())[0] == 0
        async with db.db.execute("SELECT COUNT(*) FROM memory_contradictions") as cursor:
            assert (await cursor.fetchone())[0] == 0

    @pytest.mark.asyncio
    async def test_empty_llm_response_fails_closed(self, seeded_db):
        """An empty response cannot silently classify prompted pairs as unrelated."""
        s = seeded_db
        db = s["db"]

        mock_client = AsyncMock()
        mock_client.detect_contradictions = AsyncMock(return_value=ContradictionResponse(decisions=[]))

        stats = await detect_cross_doc_contradictions(
            new_memory_ids=[s["mem_run_pg"].id],
            doc_id="doc-runbook",
            db=db,
            memory_store=_test_memory_store(db),
            structured_llm_client=mock_client,
        )

        assert stats["contradictions"] == 0
        assert stats["temporal"] == 0
        assert stats["checked"] == 1
        assert await db.get_pending_review_for_challenger(s["mem_run_pg"].id) is None
        async with db.db.execute(
            "SELECT COUNT(*) FROM relation_runs WHERE evidence_unit_id = ?",
            (f"eu-{s['mem_run_pg'].id}",),
        ) as cursor:
            assert (await cursor.fetchone())[0] == 0
        async with db.db.execute("SELECT COUNT(*) FROM memory_contradictions") as cursor:
            assert (await cursor.fetchone())[0] == 0

    @pytest.mark.asyncio
    async def test_superseded_memory_excluded_from_candidates(self, seeded_db):
        """Superseded memories should not appear as contradiction candidates."""
        s = seeded_db
        db = s["db"]

        # Supersede the existing postgresql memory
        await db.db.execute(
            "UPDATE memories SET status = 'superseded' WHERE id = ?",
            (s["mem_arch_pg"].id,),
        )
        await db.db.commit()

        mock_client = AsyncMock()

        stats = await detect_cross_doc_contradictions(
            new_memory_ids=[s["mem_run_pg"].id],
            doc_id="doc-runbook",
            db=db,
            memory_store=_test_memory_store(db),
            structured_llm_client=mock_client,
        )

        # No active candidates remain, so LLM should not be called
        assert stats["checked"] == 0
        mock_client.detect_contradictions.assert_not_called()

    @pytest.mark.asyncio
    async def test_contradiction_count_not_double_incremented(self, seeded_db):
        """Recording the same contradiction pair twice should not double-increment counts."""
        s = seeded_db
        db = s["db"]

        # Record same pair twice
        await db.record_contradiction(s["mem_arch_pg"].id, s["mem_run_pg"].id, "contradiction", "first")
        await db.record_contradiction(s["mem_arch_pg"].id, s["mem_run_pg"].id, "contradiction", "second")

        mem = await db.get_memory(s["mem_arch_pg"].id)

        async with db.db.execute(
            "SELECT COUNT(*) FROM memory_contradictions WHERE memory_id_a = ? AND memory_id_b = ?",
            (s["mem_arch_pg"].id, s["mem_run_pg"].id),
        ) as cur:
            row_count = (await cur.fetchone())[0]
        assert row_count == 1, "Should have exactly one contradiction row (INSERT OR IGNORE)"

        assert mem.contradiction_count == 1


# ===========================================================================
# E2E: Full DB state verification
# ===========================================================================


@skip_no_llm
class TestContradictionDBState:
    """Verify complete database state after contradiction detection."""

    @pytest.mark.asyncio
    async def test_full_db_state_after_contradiction(self, seeded_db, structured_llm_client):
        """After detecting a contradiction, verify all related DB tables are correct."""
        s = seeded_db
        db = s["db"]

        stats = await detect_cross_doc_contradictions(
            new_memory_ids=[s["mem_run_pg"].id],
            doc_id="doc-runbook",
            db=db,
            memory_store=_test_memory_store(db),
            structured_llm_client=structured_llm_client,
        )

        if stats["contradictions"] > 0:
            # 1. memory_contradictions row exists
            async with db.db.execute(
                """SELECT * FROM memory_contradictions
                   WHERE (memory_id_a = ? AND memory_id_b = ?)
                      OR (memory_id_a = ? AND memory_id_b = ?)""",
                (s["mem_arch_pg"].id, s["mem_run_pg"].id, s["mem_run_pg"].id, s["mem_arch_pg"].id),
            ) as cur:
                row = await cur.fetchone()

            assert row is not None
            assert row["classification"] == "contradiction"
            assert row["resolution"] == "pending"
            assert row["detected_at"] is not None
            assert row["resolved_at"] is None
            assert len(row["reason"]) > 0

            # 2. contradiction_count incremented on both memories
            mem_a = await db.get_memory(s["mem_arch_pg"].id)
            mem_b = await db.get_memory(s["mem_run_pg"].id)
            assert mem_a.contradiction_count >= 1
            assert mem_b.contradiction_count >= 1

            # 3. Other memories unaffected
            mem_kafka = await db.get_memory(s["mem_arch_kafka"].id)
            assert mem_kafka.contradiction_count == 0

        elif stats["temporal"] > 0:
            # Temporal: row exists but contradiction_count not incremented
            async with db.db.execute(
                """SELECT * FROM memory_contradictions
                   WHERE (memory_id_a = ? AND memory_id_b = ?)
                      OR (memory_id_a = ? AND memory_id_b = ?)""",
                (s["mem_arch_pg"].id, s["mem_run_pg"].id, s["mem_run_pg"].id, s["mem_arch_pg"].id),
            ) as cur:
                row = await cur.fetchone()

            assert row is not None
            assert row["classification"] == "temporal"

            mem_a = await db.get_memory(s["mem_arch_pg"].id)
            mem_b = await db.get_memory(s["mem_run_pg"].id)
            assert mem_a.contradiction_count == 0, "Temporal should not increment count"
            assert mem_b.contradiction_count == 0, "Temporal should not increment count"
