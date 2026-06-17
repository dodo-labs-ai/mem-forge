from __future__ import annotations

from datetime import datetime, timezone

import pytest

from memforge.models import (
    Memory,
    MemoryCurationRun,
    MemoryLevel,
    content_hash,
)
from memforge.storage.adapters.sqlite import build_sqlite_adapters
from memforge.storage.database import Database


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "curation.db"))
    await database.connect()
    yield database
    await database.close()


@pytest.mark.asyncio
async def test_memory_curation_fields_roundtrip(db: Database):
    memory = Memory(
        id="mem-consolidated",
        memory_type="procedure",
        content="Use the repo-scoped deployment checklist before CF deploys.",
        content_hash=content_hash("Use the repo-scoped deployment checklist before CF deploys."),
        memory_level=MemoryLevel.CONSOLIDATED.value,
        curation_cluster_id="agent-session|repo:github.tools.sap/hcm/memforge-cloud|deploy",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        tags=["deployment"],
    )

    await db.insert_memory(memory)

    loaded = await db.get_memory("mem-consolidated")

    assert loaded is not None
    assert loaded.memory_level == MemoryLevel.CONSOLIDATED.value
    assert loaded.curation_cluster_id == (
        "agent-session|repo:github.tools.sap/hcm/memforge-cloud|deploy"
    )
    assert loaded.repo_identifier == "github.tools.sap/hcm/memforge-cloud"


@pytest.mark.asyncio
async def test_memory_derivation_lineage_roundtrip(db: Database):
    for memory_id in ("mem-parent", "mem-child-1", "mem-child-2"):
        await db.insert_memory(
            Memory(
                id=memory_id,
                memory_type="fact",
                content=f"Memory {memory_id}",
                content_hash=content_hash(f"Memory {memory_id}"),
                repo_identifier="github.tools.sap/hcm/memforge-cloud",
            )
        )

    await db.add_memory_derivation("mem-parent", "mem-child-1", relation="summarizes")
    await db.add_memory_derivation("mem-parent", "mem-child-2", relation="summarizes")

    children = await db.get_memory_derivation_children("mem-parent")

    assert [child.child_memory_id for child in children] == [
        "mem-child-1",
        "mem-child-2",
    ]
    assert {child.parent_memory_id for child in children} == {"mem-parent"}
    assert {child.relation for child in children} == {"summarizes"}


@pytest.mark.asyncio
async def test_memory_curation_run_roundtrip(db: Database):
    run = MemoryCurationRun(
        id="cur-run-1",
        policy_id="agent_session.codex.v1",
        source_type="agent_session",
        client="codex",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
        project_key="UNSORTED",
        candidate_count=23,
        created_memory_count=2,
        skipped_reason=None,
        error=None,
        started_at=datetime(2026, 6, 17, 8, 0, tzinfo=timezone.utc),
        completed_at=datetime(2026, 6, 17, 8, 1, tzinfo=timezone.utc),
    )

    await db.record_memory_curation_run(run)

    loaded = await db.get_memory_curation_run("cur-run-1")

    assert loaded == run


@pytest.mark.asyncio
async def test_sqlite_ranking_metadata_includes_curation_fields(db: Database):
    await db.insert_memory(
        Memory(
            id="mem-ranked",
            memory_type="fact",
            content="Repo-scoped fact",
            content_hash=content_hash("Repo-scoped fact"),
            project_key="UNSORTED",
            memory_level=MemoryLevel.CONSOLIDATED.value,
            curation_cluster_id="cluster-1",
            repo_identifier="github.tools.sap/hcm/memforge-cloud",
        )
    )
    adapters = build_sqlite_adapters(db, memory_collection=None)

    metadata = await adapters.relational.fetch_ranking_metadata(["mem-ranked"])

    assert metadata["mem-ranked"]["project_key"] == "UNSORTED"
    assert metadata["mem-ranked"]["memory_level"] == MemoryLevel.CONSOLIDATED.value
    assert metadata["mem-ranked"]["curation_cluster_id"] == "cluster-1"
    assert metadata["mem-ranked"]["repo_identifier"] == (
        "github.tools.sap/hcm/memforge-cloud"
    )
