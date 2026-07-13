from __future__ import annotations

import pytest

from memforge.memory.curator import (
    AgentSessionCuratorPolicy,
    CuratedMemoryDraft,
    CuratorCandidate,
    MemoryCuratorRunner,
)
from memforge.models import Memory, MemoryLevel, Visibility, content_hash
from memforge.storage.adapters.sqlite import build_sqlite_adapters
from memforge.storage.database import Database


@pytest.fixture
async def store(tmp_path):
    db = Database(str(tmp_path / "curator.db"))
    await db.connect()
    try:
        adapters = build_sqlite_adapters(db, memory_collection=None)
        yield db, adapters.relational
    finally:
        await db.close()


def _memory(
    memory_id: str,
    *,
    repo: str,
    project: str = "UNSORTED",
    visibility: str = Visibility.WORKSPACE.value,
    owner: str | None = None,
    tags: list[str] | None = None,
) -> Memory:
    content = f"Memory {memory_id} for {repo}"
    return Memory(
        id=memory_id,
        memory_type="fact",
        content=content,
        content_hash=content_hash(content),
        visibility=visibility,
        owner_user_id=owner,
        project_key=project,
        repo_identifier=repo,
        tags=tags or ["deployment"],
    )


async def _summarize(cluster):
    contents = ", ".join(candidate.memory.id for candidate in cluster.candidates)
    return CuratedMemoryDraft(
        memory_type="procedure",
        content=f"Consolidated from {contents}",
        tags=["curated", *cluster.topic_tags],
    )


def test_agent_session_policy_only_accepts_known_agent_clients():
    policy = AgentSessionCuratorPolicy()
    memory = _memory("mem-1", repo="github.tools.sap/hcm/memforge-cloud")

    assert policy.applies_to(CuratorCandidate(memory, "agent_session", "codex"))
    assert policy.applies_to(CuratorCandidate(memory, "agent_session", "claude-code"))
    assert not policy.applies_to(CuratorCandidate(memory, "agent_session", "unknown"))
    assert not policy.applies_to(CuratorCandidate(memory, "jira", None))


def test_agent_session_policy_clusters_by_repo_before_project():
    policy = AgentSessionCuratorPolicy()
    left = CuratorCandidate(
        _memory("mem-1", repo="github.tools.sap/hcm/memforge-cloud", project="SFPAY"),
        "agent_session",
        "codex",
    )
    right = CuratorCandidate(
        _memory("mem-2", repo="github.tools.sap/hcm/other", project="SFPAY"),
        "agent_session",
        "codex",
    )

    assert policy.cluster_key(left) != policy.cluster_key(right)


def test_agent_session_policy_does_not_merge_private_owners():
    policy = AgentSessionCuratorPolicy()
    left = CuratorCandidate(
        _memory(
            "mem-1",
            repo="github.tools.sap/hcm/memforge-cloud",
            visibility=Visibility.PRIVATE.value,
            owner="andrew",
        ),
        "agent_session",
        "codex",
    )
    right = CuratorCandidate(
        _memory(
            "mem-2",
            repo="github.tools.sap/hcm/memforge-cloud",
            visibility=Visibility.PRIVATE.value,
            owner="test001",
        ),
        "agent_session",
        "codex",
    )

    assert policy.cluster_key(left) != policy.cluster_key(right)


@pytest.mark.asyncio
async def test_curator_runner_creates_consolidated_memory_and_lineage(store):
    db, relational = store
    source_memories = [
        _memory("mem-1", repo="github.tools.sap/hcm/memforge-cloud"),
        _memory("mem-2", repo="github.tools.sap/hcm/memforge-cloud"),
        _memory("mem-3", repo="github.tools.sap/hcm/other"),
    ]
    for memory in source_memories:
        await db.insert_memory(memory)

    runner = MemoryCuratorRunner(
        store=relational,
        policy=AgentSessionCuratorPolicy(),
        summarize=_summarize,
        min_cluster_size=2,
    )

    result = await runner.curate(
        [
            CuratorCandidate(source_memories[0], "agent_session", "codex"),
            CuratorCandidate(source_memories[1], "agent_session", "claude-code"),
            CuratorCandidate(source_memories[2], "agent_session", "codex"),
        ],
        run_id="cur-run-agent-1",
    )

    assert result.created_memory_count == 1
    consolidated = await db.get_memory(result.created_memory_ids[0])
    assert consolidated is not None
    assert consolidated.memory_level == MemoryLevel.CONSOLIDATED.value
    assert consolidated.repo_identifier == "github.tools.sap/hcm/memforge-cloud"
    assert consolidated.project_key == "UNSORTED"

    children = await db.get_memory_derivation_children(consolidated.id)
    assert [child.child_memory_id for child in children] == ["mem-1", "mem-2"]

    untouched = await db.get_memory("mem-1")
    assert untouched is not None
    assert untouched.memory_level == MemoryLevel.ATOMIC.value
    assert untouched.status == "active"

    run = await db.get_memory_curation_run("cur-run-agent-1")
    assert run is not None
    assert run.policy_id == "agent_session.coding.v1"
    assert run.client is None
    assert run.repo_identifier == "github.tools.sap/hcm/memforge-cloud"
    assert run.project_key == "UNSORTED"
    assert run.candidate_count == 3
    assert run.created_memory_count == 1
