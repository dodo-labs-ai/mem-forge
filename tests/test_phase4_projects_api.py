"""HTTP coverage for the `/api/projects` CRUD surface.

The wire model exposes `kind: 'normal' | 'shared'` over the storage
`is_shared` column. Reserved keys (SHARED, UNSORTED) refuse to delete.
A real project's delete rebuckets its memories to UNSORTED across both
the relational row and the vector metadata before removing the row.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import re

from fastapi.testclient import TestClient

from memforge.config import AppConfig
from memforge.models import (
    Memory,
    SHARED_PROJECT_KEY,
    UNSORTED_PROJECT_KEY,
    Visibility,
    content_hash,
)
from memforge.storage.database import Database


def _config(tmp_path: Path) -> AppConfig:
    cfg = AppConfig(base_dir=tmp_path / "mem")
    cfg.server.jwt_secret = "test-secret"
    cfg.llm.enrichment_api_key = ""
    cfg.llm.embedding_api_key = ""
    return cfg


def _make_app(tmp_path: Path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = Database(str(tmp_path / "api.db"))

    async def _setup():
        await database.connect()

    asyncio.run(_setup())
    app = create_admin_app(db=database, config=cfg)
    return app, database


def test_create_list_update_round_trip(tmp_path):
    app, database = _make_app(tmp_path)
    try:
        with TestClient(app) as client:
            create = client.post(
                "/api/projects",
                json={"name": "Payroll", "kind": "normal"},
            )
            assert create.status_code == 201, create.text
            created = create.json()
            assert created["key"] == "PAYROLL"
            assert created["kind"] == "normal"
            assert created["name"] == "Payroll"

            listed = client.get("/api/projects").json()
            keys = {p["key"] for p in listed}
            assert {SHARED_PROJECT_KEY, UNSORTED_PROJECT_KEY, "PAYROLL"} <= keys

            patch = client.patch(
                f"/api/projects/{created['id']}",
                json={"name": "Pay", "kind": "shared"},
            )
            assert patch.status_code == 200, patch.text
            updated = patch.json()
            assert updated["name"] == "Pay"
            assert updated["kind"] == "shared"
    finally:
        asyncio.run(database.close())


def test_create_with_explicit_key(tmp_path):
    app, database = _make_app(tmp_path)
    try:
        with TestClient(app) as client:
            create = client.post(
                "/api/projects",
                json={"name": "Risk Engine", "key": "RISK"},
            )
            assert create.status_code == 201, create.text
            assert create.json()["key"] == "RISK"
    finally:
        asyncio.run(database.close())


def test_project_created_at_is_timezone_qualified(tmp_path):
    app, database = _make_app(tmp_path)
    try:
        with TestClient(app) as client:
            create = client.post(
                "/api/projects",
                json={"name": "Risk Engine", "key": "RISK"},
            )
            assert create.status_code == 201, create.text
            created_at = create.json()["created_at"]
            assert re.search(r"(Z|[+-]\d{2}:\d{2})$", created_at), created_at
    finally:
        asyncio.run(database.close())


def test_duplicate_key_returns_conflict(tmp_path):
    app, database = _make_app(tmp_path)
    try:
        with TestClient(app) as client:
            first = client.post("/api/projects", json={"name": "Pay"})
            assert first.status_code == 201
            second = client.post("/api/projects", json={"name": "Pay"})
            assert second.status_code == 409
            assert "already exists" in second.json()["detail"]
    finally:
        asyncio.run(database.close())


def test_delete_reserved_keys_refused(tmp_path):
    app, database = _make_app(tmp_path)
    try:
        with TestClient(app) as client:
            listed = client.get("/api/projects").json()
            by_key = {p["key"]: p for p in listed}
            for reserved in (SHARED_PROJECT_KEY, UNSORTED_PROJECT_KEY):
                resp = client.delete(f"/api/projects/{by_key[reserved]['id']}")
                assert resp.status_code == 400
                assert "reserved" in resp.json()["detail"]
    finally:
        asyncio.run(database.close())


def test_delete_real_project_rebuckets_to_unsorted(tmp_path):
    app, database = _make_app(tmp_path)
    try:
        # Seed a memory under PAY so the delete has something to rebucket.
        async def _seed():
            await database.insert_memory(
                Memory(
                    id="m-pay",
                    memory_type="fact",
                    content="payroll fact",
                    content_hash=content_hash("payroll fact"),
                    visibility=Visibility.WORKSPACE.value,
                    owner_user_id=None,
                    project_key="PAY",
                    tags=[],
                )
            )

        asyncio.run(_seed())

        with TestClient(app) as client:
            create = client.post(
                "/api/projects",
                json={"name": "Pay", "key": "PAY"},
            )
            assert create.status_code == 201, create.text
            project_id = create.json()["id"]

            delete = client.delete(f"/api/projects/{project_id}")
            assert delete.status_code == 200, delete.text
            body = delete.json()
            assert body["id"] == project_id
            assert body["rebucketed_count"] == 1
            assert body["rebucketed_memory_ids"] == ["m-pay"]

            # The project row is gone.
            assert client.get("/api/projects").status_code == 200
            keys = {p["key"] for p in client.get("/api/projects").json()}
            assert "PAY" not in keys

        async def _verify_rebucket():
            stored = await database.get_memory("m-pay")
            assert stored is not None
            assert stored.project_key == UNSORTED_PROJECT_KEY

        asyncio.run(_verify_rebucket())
    finally:
        asyncio.run(database.close())


def test_delete_unknown_project_returns_404(tmp_path):
    app, database = _make_app(tmp_path)
    try:
        with TestClient(app) as client:
            resp = client.delete("/api/projects/proj-does-not-exist")
            assert resp.status_code == 404
    finally:
        asyncio.run(database.close())


def test_derive_project_key_collapses_punctuation():
    from memforge.server.admin_api import _derive_project_key

    assert _derive_project_key("Pay & Risk") == "PAY_RISK"
    assert _derive_project_key("   ") == "PROJECT"
    assert _derive_project_key("a" * 100) == "A" * 32


def test_create_source_round_trips_project_binding(tmp_path):
    """A binding sent on POST /api/sources is persisted and surfaces on GET."""
    app, database = _make_app(tmp_path)
    try:
        with TestClient(app) as client:
            binding = {
                "mode": "by_field",
                "field": "repo",
                "map": {"my-app": "APP"},
                "default": UNSORTED_PROJECT_KEY,
            }
            resp = client.post(
                "/api/sources",
                json={
                    "type": "agent_session",
                    "name": "codex sessions",
                    "config": {"client": "codex", "documents_dir": str(tmp_path / "in")},
                    "project_binding": binding,
                },
            )
            assert resp.status_code == 200, resp.text
            source_id = resp.json()["id"]

        async def _read():
            return await database.get_source(source_id)

        stored = asyncio.run(_read())
        assert stored is not None
        assert stored["project_binding"] == binding
    finally:
        asyncio.run(database.close())


def test_create_source_requires_project_binding(tmp_path):
    app, database = _make_app(tmp_path)
    try:
        with TestClient(app) as client:
            resp = client.post(
                "/api/sources",
                json={
                    "type": "agent_session",
                    "name": "codex sessions",
                    "config": {"client": "codex", "documents_dir": str(tmp_path / "in")},
                },
            )
            assert resp.status_code == 400
            assert resp.json()["detail"] == "project binding is required"
    finally:
        asyncio.run(database.close())


def test_update_source_replaces_and_preserves_project_binding(tmp_path):
    """PUT /api/sources/{id} accepts a new binding; PUT without one preserves it."""
    app, database = _make_app(tmp_path)
    try:
        with TestClient(app) as client:
            initial = {"mode": "fixed", "project_key": "PAY"}
            create = client.post(
                "/api/sources",
                json={
                    "type": "agent_session",
                    "name": "codex sessions",
                    "config": {"client": "codex", "documents_dir": str(tmp_path / "in")},
                    "project_binding": initial,
                },
            )
            source_id = create.json()["id"]

            replacement = {"mode": "fixed", "project_key": "RISK"}
            put = client.put(
                f"/api/sources/{source_id}",
                json={"project_binding": replacement},
            )
            assert put.status_code == 200, put.text

        async def _read():
            return await database.get_source(source_id)

        after_replace = asyncio.run(_read())
        assert after_replace["project_binding"] == replacement

        with TestClient(app) as client:
            # A PUT that does not mention the binding must not clear it.
            put_noop = client.put(
                f"/api/sources/{source_id}",
                json={"name": "renamed"},
            )
            assert put_noop.status_code == 200, put_noop.text

        after_noop = asyncio.run(_read())
        assert after_noop["project_binding"] == replacement
    finally:
        asyncio.run(database.close())


def test_update_source_rejects_clearing_project_binding(tmp_path):
    app, database = _make_app(tmp_path)
    try:
        with TestClient(app) as client:
            create = client.post(
                "/api/sources",
                json={
                    "type": "agent_session",
                    "name": "codex sessions",
                    "config": {"client": "codex", "documents_dir": str(tmp_path / "in")},
                    "project_binding": {"mode": "fixed", "project_key": "PAY"},
                },
            )
            source_id = create.json()["id"]

            resp = client.put(
                f"/api/sources/{source_id}",
                json={"project_binding": None},
            )
            assert resp.status_code == 400
            assert resp.json()["detail"] == "project binding is required"

        async def _read():
            return await database.get_source(source_id)

        stored = asyncio.run(_read())
        assert stored["project_binding"] == {"mode": "fixed", "project_key": "PAY"}
    finally:
        asyncio.run(database.close())


def test_rebucket_partial_vector_failure_rolls_back_already_applied():
    """If a per-id vector upsert fails partway through the batch, every
    record this call already moved must be restored to its original
    metadata. The relational rebucket has not run yet, so a clean
    rollback returns the system to exactly the pre-call state."""
    from memforge.memory.store import MemoryStore

    captured_upserts: list[tuple[str, str]] = []

    class _FlakyVector:
        def __init__(self) -> None:
            self.records: dict[str, dict] = {
                "m1": {
                    "id": "m1",
                    "embedding": [0.1, 0.2],
                    "metadata": {"project_key": "PAY", "visibility": "workspace"},
                },
                "m2": {
                    "id": "m2",
                    "embedding": [0.3, 0.4],
                    "metadata": {"project_key": "PAY", "visibility": "workspace"},
                },
            }
            self._calls = 0

        async def get_record(self, memory_id: str):
            return self.records.get(memory_id)

        async def upsert(self, *, ids, embeddings, metadatas):
            assert len(ids) == 1
            self._calls += 1
            captured_upserts.append((ids[0], metadatas[0]["project_key"]))
            if ids[0] == "m2" and self._calls == 2:
                raise RuntimeError("vector store failed mid-batch")
            self.records[ids[0]] = {
                "id": ids[0],
                "embedding": embeddings[0],
                "metadata": dict(metadatas[0]),
            }

    store = MemoryStore.__new__(MemoryStore)
    store.vector = _FlakyVector()  # type: ignore[attr-defined]

    with __import__("pytest").raises(RuntimeError, match="vector store failed mid-batch"):
        asyncio.run(store.rebucket_project_memories(["m1", "m2"], UNSORTED_PROJECT_KEY))

    # m1 was upserted to UNSORTED then rolled back to PAY; m2 never moved.
    assert store.vector.records["m1"]["metadata"]["project_key"] == "PAY"  # type: ignore[attr-defined]
    assert store.vector.records["m2"]["metadata"]["project_key"] == "PAY"  # type: ignore[attr-defined]
    # The captured sequence proves rollback ran: forward(m1) -> forward(m2 fails) -> rollback(m1).
    assert captured_upserts == [
        ("m1", UNSORTED_PROJECT_KEY),
        ("m2", UNSORTED_PROJECT_KEY),
        ("m1", "PAY"),
    ]


def test_delete_orders_vector_before_relational_commit(tmp_path):
    """If the vector channel raises during rebucket, the project row and
    the relational rebucket must NOT be applied. SQLite and Chroma both
    keep pointing at the original project until the operation can be
    re-run."""
    app, database = _make_app(tmp_path)

    async def _seed():
        await database.insert_memory(
            Memory(
                id="m-fail",
                memory_type="fact",
                content="will not move",
                content_hash=content_hash("will not move"),
                visibility=Visibility.WORKSPACE.value,
                owner_user_id=None,
                project_key="PAY",
                tags=[],
            )
        )

    asyncio.run(_seed())

    try:
        # raise_server_exceptions=False mirrors a real client: the
        # uncaught error becomes a 500 instead of bubbling to pytest.
        with TestClient(app, raise_server_exceptions=False) as client:
            create = client.post(
                "/api/projects",
                json={"name": "Pay", "key": "PAY"},
            )
            project_id = create.json()["id"]

            from memforge.memory.store import MemoryStore

            original = MemoryStore.rebucket_project_memories

            async def _explode(self, *args, **kwargs):
                raise RuntimeError("vector store offline")

            MemoryStore.rebucket_project_memories = _explode  # type: ignore[assignment]
            try:
                resp = client.delete(f"/api/projects/{project_id}")
                assert resp.status_code >= 500
            finally:
                MemoryStore.rebucket_project_memories = original  # type: ignore[assignment]

            # The project row still exists; the memory still points to PAY.
            keys = {p["key"] for p in client.get("/api/projects").json()}
            assert "PAY" in keys

        async def _verify_unmoved():
            stored = await database.get_memory("m-fail")
            assert stored is not None
            assert stored.project_key == "PAY"

        asyncio.run(_verify_unmoved())
    finally:
        asyncio.run(database.close())


def test_resolved_projects_endpoint_groups_memories_by_resolved_key(tmp_path):
    """GET /api/sources/{id}/projects/resolved reports the resolver's
    verdict on memories from this source, distinct from the raw
    `documents.space_or_project` view served by /projects."""
    from datetime import datetime, timezone

    from memforge.models import DocumentRecord

    app, database = _make_app(tmp_path)

    async def _seed():
        await database.upsert_source(
            id="src-doc",
            type="agent_session",
            name="codex sessions",
            config_json="{}",
        )
        ts = datetime.now(tz=timezone.utc)
        for doc_id in ("doc-1", "doc-2"):
            await database.upsert_document(
                DocumentRecord(
                    doc_id=doc_id,
                    source="src-doc",
                    source_url="",
                    title=doc_id,
                    space_or_project="ignored-raw-value",
                    author=None,
                    last_modified=ts,
                    labels=[],
                    version="1",
                    content_hash=f"h-{doc_id}",
                    token_count=None,
                    raw_content_uri=None,
                    raw_content_type=None,
                    normalized_content_uri=None,
                    pdf_content_uri=None,
                    last_synced=ts,
                )
            )
        for mid, doc, key in (
            ("m-pay", "doc-1", "PAY"),
            ("m-other", "doc-2", "RISK"),
        ):
            await database.insert_memory(
                Memory(
                    id=mid,
                    memory_type="fact",
                    content=mid,
                    content_hash=content_hash(mid),
                    visibility=Visibility.WORKSPACE.value,
                    owner_user_id=None,
                    project_key=key,
                    tags=[],
                )
            )
            await database.add_memory_source(
                memory_id=mid,
                doc_id=doc,
                source_type="agent_session",
                excerpt=None,
            )

    asyncio.run(_seed())

    try:
        with TestClient(app) as client:
            resp = client.get("/api/sources/src-doc/projects/resolved")
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["source_id"] == "src-doc"
            by_key = {p["project_key"]: p["memory_count"] for p in body["projects"]}
            assert by_key == {"PAY": 1, "RISK": 1}
    finally:
        asyncio.run(database.close())
