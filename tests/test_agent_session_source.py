from __future__ import annotations

from pathlib import Path

import pytest

from memforge.agent_sessions import agent_session_source_id
from memforge.genes import GENE_REGISTRY, create_gene, source_type_supports_sync
from memforge.genes.agent_session_gene import AgentSessionGene
from memforge.storage.database import Database
from tests.agent_session_fixture import seed_agent_session_source


@pytest.fixture
async def db(tmp_path: Path):
    database = Database(str(tmp_path / "agent_sessions.db"))
    await database.connect()
    yield database
    await database.close()


@pytest.mark.asyncio
async def test_agent_session_sources_are_private_and_partitioned_by_owner(db: Database):
    alice = await seed_agent_session_source(db, client="codex", owner_user_id="alice")
    bob = await seed_agent_session_source(db, client="codex", owner_user_id="bob")

    assert alice["id"] == agent_session_source_id("codex", "alice")
    assert bob["id"] == agent_session_source_id("codex", "bob")
    assert alice["id"] != bob["id"]
    assert alice["owner_user_id"] == "alice"
    assert bob["owner_user_id"] == "bob"
    assert alice["access_policy"] == bob["access_policy"] == "private"
    assert alice["config"] == bob["config"] == {"client": "codex"}


def test_agent_session_source_id_requires_client_and_owner():
    with pytest.raises(ValueError, match="client is required"):
        agent_session_source_id(" ", "alice")
    with pytest.raises(ValueError, match="owner_user_id is required"):
        agent_session_source_id("codex", "")


@pytest.mark.asyncio
async def test_agent_session_gene_is_registered_without_sync_or_discovery():
    gene = create_gene("agent_session", {"client": "codex"}, agent_session_source_id("codex", "alice"))

    assert GENE_REGISTRY["agent_session"] is AgentSessionGene
    assert AgentSessionGene.metadata().execution_kinds == ()
    assert not source_type_supports_sync("agent_session")
    assert [item async for item in gene.discover()] == []
