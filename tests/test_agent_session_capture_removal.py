"""Removing agent-session capture storage (migration 131) and reading what it leaves.

An installation upgraded from a release that kept agent-session capture holds
agent-session Sources, Agent Knowledge claims and receipts. The upgrade takes
its recovery point, retires every remaining agent-session Source as Source
deletion does, queues the vector removal of each retired Memory and drops the
five capture tables. The retired Source, its Documents and its Memories stay
as history and stay readable although ``agent_session`` is no longer a
registered source type.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from memforge.config import AppConfig
from memforge.memory.evidence import EvidenceContentProvenance, EvidenceUnit
from memforge.memory.lifecycle_service import MemoryLifecycleService
from memforge.memory.store import MemoryStore
from memforge.models import (
    ContentItem,
    DocumentRecord,
    Memory,
    NormalizedContent,
    RawContent,
    Visibility,
    content_hash,
)
from memforge.pipeline.source_projection_adapters import project_source_item
from memforge.source_projection import AnchorKind, SourceAnchor
from memforge.storage.adapters.sqlite import build_sqlite_adapters
from memforge.storage.database import Database, MigrationFailed, latest_schema_version
from memforge.storage.recovery_points import PRE_UPGRADE, RecoveryPoints
from tests.embedding_fixture import stand_in_embedding
from tests.legacy_capture_tables import CAPTURE_TABLES, CAPTURE_TABLES_SCHEMA
from tests.unit_support_fixture import primary_reference, record_unit_support

REMOVAL = 131
OWNER = "owner-1"
AGENT_SOURCE = "src-agent-sessions-codex-0123456789abcdef"
WIKI_SOURCE = "src-wiki"
CONCEPT_DOC = "concept-release-checks"
WIKI_DOC = "doc-release-wiki"
CLAIM_MEMORY = "mem-claim"
SHARED_MEMORY = "mem-shared"
CLAIM_UNIT = "eu-claim"
OBSERVED = datetime(2026, 9, 1, tzinfo=timezone.utc)


class RecordingCollection:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}
        self.deleted: list[str] = []

    def upsert(self, *, ids, embeddings=None, metadatas=None, documents=None) -> None:
        for index, record_id in enumerate(ids):
            self.records[record_id] = {"metadata": metadatas[index] if metadatas else {}}

    def delete(self, *, ids) -> None:
        self.deleted.extend(ids)
        for record_id in ids:
            self.records.pop(record_id, None)

    def query(self, **_params):
        return {"ids": [[]], "distances": [[]]}

    def get(self, *, ids=None, include=None):
        selected = [record_id for record_id in (ids or self.records) if record_id in self.records]
        result: dict[str, Any] = {"ids": selected}
        if "metadatas" in (include or []):
            result["metadatas"] = [self.records[record_id]["metadata"] for record_id in selected]
        return result


def _store(db: Database, collection: RecordingCollection) -> MemoryStore:
    adapters = build_sqlite_adapters(db, collection)
    store = MemoryStore(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embedding=stand_in_embedding(),
    )

    async def fake_embed(_text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    store._embed = fake_embed  # type: ignore[method-assign]
    return store


def _memory(memory_id: str, content: str) -> Memory:
    return Memory(
        id=memory_id,
        memory_type="fact",
        content=content,
        content_hash=content_hash(content),
        visibility=Visibility.PRIVATE.value,
        owner_user_id=OWNER,
        created_at=OBSERVED,
        updated_at=OBSERVED,
    )


async def _document(db: Database, *, doc_id: str, source_id: str, url: str, client: str | None = None) -> None:
    await db.upsert_document(
        DocumentRecord(
            doc_id=doc_id,
            source=source_id,
            source_url=url,
            title=doc_id,
            space_or_project="ENG",
            author=client,
            last_modified=OBSERVED,
            labels=[],
            version="1",
            content_hash=f"hash-{doc_id}",
            token_count=8,
            last_synced=OBSERVED,
            client=client,
        )
    )


async def _support(db: Database, *, memory: Memory, unit_id: str, source_id: str, doc_id: str, content: str) -> None:
    """Record the Memory's Support by one Evidence Unit of the Document's projection."""

    item = ContentItem(
        item_id=doc_id,
        title=doc_id,
        source_url=f"https://example.test/{doc_id}",
        last_modified=OBSERVED,
        version="1",
        extra={"page_id": doc_id, "space_key": "ENG"},
    )
    projection = project_source_item(
        source_id=source_id,
        source_type="confluence",
        run_id=f"projection-{unit_id}",
        item=item,
        raw=RawContent(item=item, body=content.encode(), content_type="text/html"),
        normalized=NormalizedContent(item=item, markdown_body=content),
    )
    await db.record_source_projection(projection)
    observation = projection.observations[0]
    unit = EvidenceUnit(
        id=unit_id,
        source_id=source_id,
        doc_id=doc_id,
        doc_revision_id=projection.source_unit_revisions[0].id,
        source_type="confluence",
        source_anchor=observation.id,
        source_lineage_id=projection.source_units[0].id,
        project_key="ENG",
        visibility=Visibility.PRIVATE.value,
        owner_user_id=OWNER,
        repo_identifier=None,
        content=content,
        excerpt=content,
        evidence_provenance=EvidenceContentProvenance.SOURCE_EXCERPT,
        access_context_hash=f"private-{OWNER}",
    )
    await record_unit_support(
        db,
        memory_id=memory.id,
        unit=unit,
        references=(
            primary_reference(
                SourceAnchor(
                    kind=AnchorKind.WHOLE_OBSERVATION,
                    observation_id=observation.id,
                    observation_revision_id=projection.observation_revisions[0].id,
                )
            ),
        ),
    )
    source = await db.get_source(source_id)
    await db.add_memory_source(memory.id, doc_id, str(source["type"]), content, source_updated_at=OBSERVED)


async def _seed_current_data(path: Path) -> str:
    """Store the data both releases understand, and return the id of a Memory ``create_memory`` wrote."""

    db = Database(str(path))
    await db.connect()
    try:
        # Projected as an ordinary Source; its rows then take the types earlier releases stored.
        await db.upsert_source(
            id=AGENT_SOURCE,
            type="confluence",
            name="Codex Session",
            config_json='{"client": "codex"}',
            access_policy="private",
            owner_user_id=OWNER,
            created_by_user_id=OWNER,
        )
        await db.upsert_source(
            id=WIKI_SOURCE,
            type="confluence",
            name="Release wiki",
            config_json="{}",
            access_policy="private",
            owner_user_id=OWNER,
            created_by_user_id=OWNER,
        )
        await _document(
            db,
            doc_id=CONCEPT_DOC,
            source_id=AGENT_SOURCE,
            url=f"agent-knowledge://{OWNER}/{CONCEPT_DOC}",
            client="codex",
        )
        await _document(db, doc_id=WIKI_DOC, source_id=WIKI_SOURCE, url=f"https://wiki.example.test/{WIKI_DOC}")
        claim = _memory(CLAIM_MEMORY, "Release candidates are validated through the marketplace install path.")
        shared = _memory(SHARED_MEMORY, "Every release records its smoke evidence.")
        for memory in (claim, shared):
            await db.insert_memory(memory)
        await _support(
            db, memory=claim, unit_id=CLAIM_UNIT, source_id=AGENT_SOURCE, doc_id=CONCEPT_DOC, content=claim.content
        )
        await _support(
            db, memory=shared, unit_id="eu-shared-agent", source_id=AGENT_SOURCE, doc_id=CONCEPT_DOC,
            content=shared.content,
        )
        await _support(
            db, memory=shared, unit_id="eu-shared-wiki", source_id=WIKI_SOURCE, doc_id=WIKI_DOC,
            content=shared.content,
        )
        for sql in (
            "UPDATE sources SET type = 'agent_session' WHERE id = ?",
            "UPDATE evidence_units SET source_type = 'agent_session' WHERE source_id = ?",
            "UPDATE memory_sources SET source_type = 'agent_session' WHERE source_id = ?",
            "UPDATE source_projection_runs SET source_type = 'agent_session' WHERE source_id = ?",
            "UPDATE source_units SET unit_type = 'agent_concept' WHERE source_id = ?",
            "UPDATE source_observations SET observation_type = 'agent_concept' WHERE source_id = ?",
        ):
            await db.db.execute(sql, (AGENT_SOURCE,))
        await db.db.execute(
            """INSERT INTO evidence_relations (
                   evidence_unit_id, memory_id, relation_type, authority_case,
                   is_authoritative_support, classifier_version, relation_run_id
               ) VALUES (?, ?, 'supports', 'same_agent_claim', 1, 'agent-session-intent', 'run-claim')""",
            (CLAIM_UNIT, CLAIM_MEMORY),
        )
        await db.db.commit()
        created = await MemoryLifecycleService(db=db, memory_store=_store(db, RecordingCollection())).create_memory(
            content="The release owner signs off the changelog.",
            provenance="The user asked to remember who signs off the changelog.",
            owner_user_id=OWNER,
            client="codex",
        )
        return created.memory_id
    finally:
        await db.close()


def _make_prior_release(path: Path) -> None:
    """Make the database look as a release that still kept agent-session capture left it."""

    with closing(sqlite3.connect(path)) as connection:
        connection.executescript(CAPTURE_TABLES_SCHEMA)
        now = OBSERVED.isoformat()
        connection.execute(
            """INSERT INTO agent_concepts (
                   id, source_id, owner_user_id, visibility, workspace, concept_type, concept_path,
                   title, markdown_body, last_observed_at
               ) VALUES (?, ?, ?, 'private', '/workspace', 'workflow', 'release-checks.md',
                         'Release checks', '# Release checks', ?)""",
            (CONCEPT_DOC, AGENT_SOURCE, OWNER, now),
        )
        for claim_id, memory_id in (("claim-1", CLAIM_MEMORY), ("claim-2", SHARED_MEMORY)):
            connection.execute(
                """INSERT INTO agent_claims (
                       id, concept_id, display_anchor, claim_text, memory_type, memory_id, last_observed_at
                   ) VALUES (?, ?, ?, 'claim', 'fact', ?, ?)""",
                (claim_id, CONCEPT_DOC, claim_id, memory_id, now),
            )
        connection.execute(
            "INSERT INTO agent_claim_citations (claim_id, citation_url, observed_at) VALUES ('claim-1', ?, ?)",
            (f"agent-knowledge://{OWNER}/{CONCEPT_DOC}", now),
        )
        connection.execute(
            """INSERT INTO agent_session_receipts (
                   doc_id, source_id, client, session_id, trigger, workspace, history_window_kind,
                   submitted_at, document_hash, source_kind, document_uri, updated_at
               ) VALUES ('window-1', ?, 'codex', 'session-1', 'Stop', '/workspace', 'summary',
                         ?, 'hash', 'generated_agent_window_summary', 'memforge://window-1', ?)""",
            (AGENT_SOURCE, now, now),
        )
        connection.execute(
            """INSERT INTO agent_hook_receipts (
                   receipt_id, client, session_id, hook, workspace, submitted_at, updated_at
               ) VALUES ('hook-1', 'codex', 'session-1', 'Stop', '/workspace', ?, ?)""",
            (now, now),
        )
        connection.execute("DELETE FROM schema_migrations WHERE version = ?", (REMOVAL,))
        connection.commit()


def _schema(path: Path) -> int:
    with closing(sqlite3.connect(path)) as connection:
        return int(connection.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0])


def _capture_objects(path: Path) -> set[str]:
    with closing(sqlite3.connect(path)) as connection:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE name IN ({}) OR name LIKE 'idx_agent_session_receipts%' "
            "OR name LIKE 'idx_agent_hook_receipts%' OR name LIKE 'idx_agent_concepts%' "
            "OR name LIKE 'idx_agent_claims%'".format(",".join("?" for _ in CAPTURE_TABLES)),
            CAPTURE_TABLES,
        ).fetchall()
    return {str(row[0]) for row in rows}


def _rows(path: Path, sql: str, params: tuple = ()) -> list[tuple]:
    with closing(sqlite3.connect(path)) as connection:
        return connection.execute(sql, params).fetchall()


@pytest.fixture
def base_dir(tmp_path) -> Path:
    return tmp_path / "mem"


@pytest.fixture
def db_path(base_dir) -> Path:
    path = base_dir / "db" / "memforge.db"
    path.parent.mkdir(parents=True)
    return path


@pytest.fixture
def points(base_dir, db_path) -> RecoveryPoints:
    return RecoveryPoints(base_dir / "backups", db_path=db_path, chroma_path=base_dir / "vectors" / "chroma")


def _config(tmp_path: Path) -> AppConfig:
    config = AppConfig(base_dir=tmp_path / "memforge")
    config.server.jwt_secret = "test-secret"
    config.llm.enrichment_api_key = ""
    config.llm.embedding_api_key = ""
    config.sync.worker_enabled = False
    return config


def _app(tmp_path: Path, db: Database):
    from memforge.server.admin_api import create_admin_app

    def principal(request: Request) -> str:
        return request.headers.get("x-test-user", OWNER)

    return create_admin_app(
        db=db,
        config=_config(tmp_path),
        principal_resolver=principal,
        workspace_role_resolver=lambda _request: "member",
    )


async def _upgraded(db_path: Path, points: RecoveryPoints) -> str:
    created_memory_id = await _seed_current_data(db_path)
    _make_prior_release(db_path)
    db = Database(str(db_path), recovery_points=points)
    await db.connect()
    await db.close()
    return created_memory_id


# ---------------------------------------------------------------------------
# The upgrade
# ---------------------------------------------------------------------------


async def test_upgrade_retires_agent_session_sources_and_drops_the_capture_tables(db_path, points, tmp_path) -> None:
    created_memory_id = await _upgraded(db_path, points)

    [point] = points.list()
    assert point.reason == PRE_UPGRADE
    assert point.schema_version == REMOVAL - 1
    assert _schema(db_path) == latest_schema_version()
    assert _capture_objects(db_path) == set()
    assert _rows(db_path, "SELECT status FROM sources WHERE id = ?", (AGENT_SOURCE,)) == [("retired",)]
    assert _rows(db_path, "SELECT status FROM sources WHERE id = ?", (WIKI_SOURCE,)) == [("active",)]
    assert _rows(
        db_path, "SELECT id, status, retirement_reason FROM memories WHERE id IN (?, ?) ORDER BY id",
        (CLAIM_MEMORY, SHARED_MEMORY),
    ) == [(CLAIM_MEMORY, "retired", "source_deleted"), (SHARED_MEMORY, "active", None)]
    assert _rows(
        db_path, "SELECT source_id, memory_id, status FROM source_deletion_vector_outbox"
    ) == [(AGENT_SOURCE, CLAIM_MEMORY, "pending")]
    # History stays: the Source row, its Document and its Evidence Units.
    assert _rows(db_path, "SELECT source FROM documents WHERE doc_id = ?", (CONCEPT_DOC,)) == [(AGENT_SOURCE,)]
    assert _rows(db_path, "SELECT COUNT(*) FROM evidence_units WHERE source_id = ?", (AGENT_SOURCE,)) == [(2,)]

    db = Database(str(db_path))
    await db.connect()
    try:
        with TestClient(_app(tmp_path, db)) as client:
            created = client.get(f"/api/v1/memories/{created_memory_id}")
        assert created.status_code == 200, created.text
        assert created.json()["origin_client"] == "codex"
    finally:
        await db.close()


async def test_reopening_an_upgraded_database_changes_nothing(db_path, points) -> None:
    await _upgraded(db_path, points)
    before = _rows(db_path, "SELECT id, status, updated_at FROM memories ORDER BY id")

    db = Database(str(db_path), recovery_points=points)
    await db.connect()
    await db.close()

    assert len(points.list()) == 1
    assert _rows(db_path, "SELECT id, status, updated_at FROM memories ORDER BY id") == before
    assert len(_rows(db_path, "SELECT id FROM source_deletion_vector_outbox")) == 1


async def test_the_vector_worker_removes_the_memories_the_upgrade_retired(db_path, points) -> None:
    await _upgraded(db_path, points)
    collection = RecordingCollection()
    collection.records[CLAIM_MEMORY] = {"metadata": {}}

    db = Database(str(db_path))
    await db.connect()
    try:
        await _store(db, collection).attempt_lifecycle_vector_delivery(source_id=AGENT_SOURCE)
        remaining = await db.list_lifecycle_vector_tasks(source_id=AGENT_SOURCE)
    finally:
        await db.close()

    assert CLAIM_MEMORY in collection.deleted
    assert CLAIM_MEMORY not in collection.records
    assert remaining == []


async def test_restoring_the_recovery_point_brings_the_capture_data_back(db_path, points) -> None:
    await _upgraded(db_path, points)
    [point] = points.list()

    points.restore(point.id, readable_schema_version=latest_schema_version())

    assert _schema(db_path) == REMOVAL - 1
    assert set(CAPTURE_TABLES) <= _capture_objects(db_path)
    assert _rows(db_path, "SELECT status FROM sources WHERE id = ?", (AGENT_SOURCE,)) == [("active",)]
    assert _rows(db_path, "SELECT status FROM memories WHERE id = ?", (CLAIM_MEMORY,)) == [("active",)]


async def test_an_agent_session_source_with_a_failed_access_change_stops_the_upgrade(db_path, points) -> None:
    await _seed_current_data(db_path)
    db = Database(str(db_path))
    await db.connect()
    try:
        await db.create_source_access_transition(
            operation_id="op-failed",
            source_id=AGENT_SOURCE,
            idempotency_key="key-failed",
            actor_user_id=OWNER,
            target_policy="workspace",
        )
        await db.db.execute("UPDATE source_access_transitions SET status = 'failed' WHERE operation_id = 'op-failed'")
        await db.db.commit()
    finally:
        await db.close()
    _make_prior_release(db_path)
    source_status = _rows(db_path, "SELECT status FROM sources WHERE id = ?", (AGENT_SOURCE,))

    upgrading = Database(str(db_path), recovery_points=points)
    with pytest.raises(MigrationFailed) as failed:
        await upgrading.connect()
    await upgrading.close()

    message = str(failed.value)
    [point] = points.list()
    assert f"Migration {REMOVAL}" in message
    assert AGENT_SOURCE in message
    assert "op-failed" in message
    assert "delete the Source in the admin UI" in message
    assert f"recovery point {point.id}" in message
    assert _schema(db_path) == REMOVAL - 1
    assert set(CAPTURE_TABLES) <= _capture_objects(db_path)
    assert _rows(db_path, "SELECT status FROM sources WHERE id = ?", (AGENT_SOURCE,)) == source_status
    assert source_status != [("retired",)]
    assert _rows(db_path, "SELECT status FROM memories WHERE id = ?", (CLAIM_MEMORY,)) == [("active",)]
    assert _rows(db_path, "SELECT COUNT(*) FROM source_deletion_vector_outbox") == [(0,)]


async def test_a_fresh_database_has_no_capture_tables(db_path, points) -> None:
    db = Database(str(db_path), recovery_points=points)
    await db.connect()
    await db.close()

    assert _schema(db_path) == latest_schema_version()
    assert _capture_objects(db_path) == set()
    assert points.list() == []


# ---------------------------------------------------------------------------
# The API after the removal
# ---------------------------------------------------------------------------


async def test_capture_routes_are_gone_and_search_refuses_the_clients_facet(tmp_path) -> None:
    db = Database(str(tmp_path / "api.db"))
    await db.connect()
    try:
        with TestClient(_app(tmp_path, db)) as client:
            window = client.post("/api/v1/agent-sessions/windows", json={})
            receipt = client.post("/api/v1/hooks/receipts", json={})
            search = client.post(
                "/api/v1/memories/search",
                json={"query": "release checks", "source_filter": {"clients": ["codex"]}},
            )
            openapi = client.get("/openapi.json").json()
    finally:
        await db.close()

    assert window.status_code == 404
    assert receipt.status_code == 404
    assert search.status_code == 422
    assert any("clients" in str(error.get("loc")) for error in search.json()["detail"])
    assert "/api/v1/agent-sessions/windows" not in openapi["paths"]
    assert "/api/v1/hooks/receipts" not in openapi["paths"]
    assert "clients" not in openapi["components"]["schemas"]["SourceFacetFilterRequest"]["properties"]


async def test_readers_of_a_retired_agent_session_source_handle_its_unregistered_type(
    db_path, points, tmp_path
) -> None:
    await _upgraded(db_path, points)

    db = Database(str(db_path))
    await db.connect()
    try:
        claim = await db.get_memory(CLAIM_MEMORY)
        assert claim is not None
        with TestClient(_app(tmp_path, db)) as client:
            sources = client.get("/api/v1/sources")
            searchable = client.get("/api/v1/sources/searchable")
            retired = client.get("/api/v1/memories", params={"status": "retired", "include_private": "true"})
            detail = client.get(f"/api/v1/memories/{CLAIM_MEMORY}")
            evidence = client.get(f"/api/v1/memories/{CLAIM_MEMORY}/evidence/{CLAIM_UNIT}/resource")
            document = client.get(f"/api/v1/documents/{CONCEPT_DOC}/content")
            changes = client.get("/api/v1/recent-changes", params={"since": "2026-01-01T00:00:00+00:00"})
            lifecycle = client.get(f"/api/v1/sources/{AGENT_SOURCE}/memory-lifecycle")
            retire = client.post(
                f"/api/v1/memories/{CLAIM_MEMORY}/retire",
                json={"reason": "No longer wanted.", "expected_content_hash": claim.content_hash},
            )
            correct = client.post(
                f"/api/v1/memories/{CLAIM_MEMORY}/corrections/propose",
                json={
                    "replacement_content": "Release candidates are validated by hand.",
                    "provenance": "The user corrected the claim.",
                    "reason": "Out of date.",
                    "expected_content_hash": claim.content_hash,
                },
            )
            shared = client.get(f"/api/v1/memories/{SHARED_MEMORY}")
    finally:
        await db.close()

    assert sources.status_code == 200, sources.text
    assert AGENT_SOURCE not in {row["id"] for row in sources.json()["data"]}
    assert searchable.status_code == 200, searchable.text
    assert retired.status_code == 200, retired.text
    assert CLAIM_MEMORY in {row["id"] for row in retired.json()["data"]}
    assert detail.status_code == 200, detail.text
    assert (detail.json()["status"], detail.json()["retirement_reason"]) == ("retired", "source_deleted")
    assert evidence.status_code == 404, evidence.text
    assert document.status_code == 404, document.text
    assert changes.status_code == 200, changes.text
    assert lifecycle.status_code == 404, lifecycle.text
    # Correction and retirement act on active Memories only; a retired one is refused as not found.
    assert retire.status_code == 404, retire.text
    assert correct.status_code == 404, correct.text
    assert shared.status_code == 200, shared.text
    assert [source["source_id"] for source in shared.json()["sources"]] == [WIKI_SOURCE]
