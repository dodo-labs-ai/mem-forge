from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from memforge.agent_knowledge import AgentKnowledgePatchProposal
from memforge.config import AppConfig
from memforge.models import DocumentRecord, Memory, content_hash
from memforge.storage.database import Database


@pytest.fixture(autouse=True)
def _stub_memory_embedding(monkeypatch):
    async def _fake_embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr("memforge.memory.store.MemoryStore._embed", _fake_embed)


def _config(tmp_path: Path) -> AppConfig:
    cfg = AppConfig(base_dir=tmp_path / "mem")
    cfg.server.jwt_secret = "test-secret"
    cfg.llm.enrichment_api_key = ""
    cfg.llm.embedding_api_key = ""
    return cfg


def _knowledge_patch(**overrides) -> AgentKnowledgePatchProposal:
    claim_text = overrides.get(
        "claim_text",
        "The agent session window recorded a durable implementation rule.",
    )
    data = {
        "action": "create_new_concept",
        "concept_type": "debugging_takeaway",
        "title": "Agent session durable rule",
        "claim_text": claim_text,
        "memory_content": claim_text,
        "memory_type": "procedure",
        "tags": ["agent-session"],
        "confidence": 0.9,
        "reason": "durable implementation behavior",
    }
    data.update(overrides)
    return AgentKnowledgePatchProposal(**data)


async def _seed_source_project(
    db: Database,
    *,
    doc_id: str,
    project: str,
    last_modified: datetime,
    memory_ids: list[str],
    source_id: str = "src-agent-sessions-codex",
    client: str = "codex",
) -> None:
    await db.upsert_document(
        DocumentRecord(
            doc_id=doc_id,
            source=source_id,
            source_url=f"agent-session://codex/sess/{doc_id}",
            title=f"Agent Session {doc_id}",
            space_or_project=project,
            author=client,
            last_modified=last_modified,
            labels=[],
            version=f"version-{doc_id}",
            content_hash=f"hash-{doc_id}",
            token_count=100,
            raw_content_uri=None,
            raw_content_type="application/json",
            normalized_content_uri=None,
            pdf_content_uri=None,
            last_synced=last_modified,
            client=client,
        )
    )
    for memory_id in memory_ids:
        memory = Memory(
            id=memory_id,
            memory_type="fact",
            content=f"Memory {memory_id}",
            content_hash=content_hash(f"Memory {memory_id}"),
            project_key=project,
            tags=["agent-session"],
            confidence=0.9,
            created_at=last_modified,
            updated_at=last_modified,
            status="active",
        )
        await db.insert_memory(memory)
        await db.add_memory_source(memory.id, doc_id, "agent_session")


def test_agent_session_document_submit_api_records_generated_source(tmp_path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = Database(str(tmp_path / "api.db"))

    async def _setup():
        await database.connect()

    import asyncio

    asyncio.run(_setup())
    try:
        app = create_admin_app(db=database, config=cfg)
        with TestClient(app) as client:
            response = client.post(
                "/api/agent-sessions/documents",
                json={
                    "client": "codex",
                    "session_id": "sess-api",
                    "trigger": "Stop",
                    "workspace": "/workspace/mem-forge",
                    "repo": "mem-forge",
                    "branch": "main",
                    "commit_sha": "abc123",
                    "history_window_kind": "session",
                    "history_window_start": "2026-05-21T10:00:00+00:00",
                    "history_window_end": "2026-05-21T11:00:00+00:00",
                    "document_markdown": "# Summary\n\n## Outcome\nAPI accepted a generated session document.",
                    "process_now": False,
                },
            )

        assert response.status_code == 200
        body = response.json()
        assert body["source_id"] == "src-agent-sessions-codex"
        assert body["sync_started"] is False
        assert body["receipt"]["client"] == "codex"
        assert Path(body["document_uri"]).exists()
    finally:
        asyncio.run(database.close())


def test_agent_session_document_submit_uses_server_principal(tmp_path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = Database(str(tmp_path / "api.db"))

    async def _setup():
        await database.connect()

    import asyncio

    asyncio.run(_setup())
    try:
        app = create_admin_app(
            db=database,
            config=cfg,
            principal_resolver=lambda request: "u-authorized",
        )
        with TestClient(app) as client:
            response = client.post(
                "/api/agent-sessions/documents",
                json={
                    "client": "codex",
                    "session_id": "sess-principal",
                    "trigger": "Stop",
                    "workspace": "/workspace/mem-forge",
                    "repo": "mem-forge",
                    "document_markdown": "# Summary\n\nThe route pins owner identity.",
                    "process_now": False,
                    "user_id": "u-spoofed",
                },
            )

        assert response.status_code == 200, response.text
        receipt = asyncio.run(
            database.get_agent_session_receipt(response.json()["doc_id"])
        )
        assert receipt is not None
        assert receipt["metadata"]["user_id"] == "u-authorized"
    finally:
        asyncio.run(database.close())


def test_agent_session_window_submit_uses_server_principal(tmp_path):
    from memforge.server.admin_api import create_admin_app

    class PackageClient:
        async def generate_agent_knowledge_patch(self, prompt: str, **kwargs):
            return _knowledge_patch(
                title="Principal patch",
                claim_text="The window route pins owner identity to the server principal.",
            )

    cfg = _config(tmp_path)
    database = Database(str(tmp_path / "api.db"))

    async def _setup():
        await database.connect()

    import asyncio

    asyncio.run(_setup())
    try:
        app = create_admin_app(
            db=database,
            config=cfg,
            principal_resolver=lambda request: "u-authorized",
        )
        app.state.agent_session_window_client = PackageClient()
        with TestClient(app) as client:
            response = client.post(
                "/api/agent-sessions/windows",
                json={
                    "schema_version": "agent-session-window/v1",
                    "client": "codex",
                    "session_id": "sess-window-principal",
                    "trigger": "PreCompact",
                    "workspace": "/workspace/mem-forge",
                    "repo": "mem-forge",
                    "events": [
                        {
                            "kind": "assistant_message",
                            "actor": "assistant",
                            "text": "Implemented the durable rule.",
                        }
                    ],
                    "retention": "none",
                    "process_now": False,
                    "user_id": "u-spoofed",
                },
            )

        assert response.status_code == 200, response.text
        body = response.json()
        memory = asyncio.run(database.get_memory(body["memory_id"]))
        assert memory is not None
        assert memory.owner_user_id == "u-authorized"
    finally:
        asyncio.run(database.close())


def test_agent_session_window_uses_bounded_completion_budget(tmp_path):
    from memforge.server.admin_api import create_admin_app

    class RecordingWindowClient:
        def __init__(self):
            self.calls = []

        async def generate_agent_knowledge_patch(self, prompt: str, **kwargs):
            self.calls.append(kwargs)
            return _knowledge_patch(
                title="Bounded patch",
                claim_text="Agent-session patch generation uses a bounded completion budget.",
            )

    cfg = _config(tmp_path)
    cfg.llm.enrichment_max_tokens = 64000
    database = Database(str(tmp_path / "api.db"))
    window_client = RecordingWindowClient()

    import asyncio

    asyncio.run(database.connect())
    try:
        app = create_admin_app(db=database, config=cfg)
        app.state.agent_session_window_client = window_client
        with TestClient(app) as client:
            response = client.post(
                "/api/agent-sessions/windows",
                json={
                    "client": "codex",
                    "session_id": "sess-budget",
                    "trigger": "Stop",
                    "workspace": "/workspace/mem-forge",
                    "events": [
                        {
                            "kind": "assistant_message",
                            "actor": "assistant",
                            "text": "Captured a durable design decision.",
                        }
                    ],
                    "transcript_markdown": "Durable design decision worth keeping.",
                },
            )

        assert response.status_code == 200, response.text
        assert window_client.calls
        assert window_client.calls[0]["max_tokens"] == 8192
    finally:
        asyncio.run(database.close())


def test_source_projects_endpoint_groups_agent_session_memory_by_project(tmp_path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = Database(str(tmp_path / "api.db"))
    base_time = datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc)

    async def _setup():
        await database.connect()
        await database.upsert_source(
            "src-agent-sessions-codex",
            "agent_session",
            "Codex Session",
            json.dumps({}),
        )
        await _seed_source_project(
            database,
            doc_id="agent-doc-mem-inception-1",
            project="mem-inception",
            last_modified=base_time,
            memory_ids=["mem-agent-1"],
        )
        await _seed_source_project(
            database,
            doc_id="agent-doc-mem-inception-2",
            project="mem-inception",
            last_modified=base_time + timedelta(minutes=5),
            memory_ids=["mem-agent-2"],
        )
        await _seed_source_project(
            database,
            doc_id="agent-doc-payroll-1",
            project="payroll-processing-service",
            last_modified=base_time - timedelta(days=1),
            memory_ids=["mem-agent-3"],
        )

    import asyncio

    asyncio.run(_setup())
    try:
        app = create_admin_app(db=database, config=cfg)
        with TestClient(app) as client:
            response = client.get("/api/sources/src-agent-sessions-codex/projects")

        assert response.status_code == 200
        body = response.json()
        assert body == {
            "source_id": "src-agent-sessions-codex",
            "projects": [
                {
                    "project": "mem-inception",
                    "document_count": 2,
                    "memory_count": 2,
                    "last_observed_at": "2026-06-01T10:05:00+00:00",
                },
                {
                    "project": "payroll-processing-service",
                    "document_count": 1,
                    "memory_count": 1,
                    "last_observed_at": "2026-05-31T10:00:00+00:00",
                },
            ],
        }
    finally:
        asyncio.run(database.close())


def test_agent_session_window_api_generates_package_and_discards_raw_window(tmp_path):
    from memforge.server.admin_api import create_admin_app

    class FakeWindowClient:
        async def generate_agent_knowledge_patch(self, prompt: str, **kwargs):
            assert "Canonical evidence" in prompt
            assert "api_key: [REDACTED]" in prompt
            assert "token: secret-value" not in prompt
            assert "raw-api-secret" not in prompt
            assert "history-secret" not in prompt
            return _knowledge_patch(
                title="Agent Session: useful implementation window",
                claim_text=(
                    "The agent updated the window upload endpoint and unit tests "
                    "covered the knowledge patch path."
                ),
            )

    cfg = _config(tmp_path)
    database = Database(str(tmp_path / "api.db"))

    async def _setup():
        await database.connect()

    import asyncio

    asyncio.run(_setup())
    try:
        app = create_admin_app(db=database, config=cfg)
        app.state.agent_session_window_client = FakeWindowClient()
        with TestClient(app) as client:
            response = client.post(
                "/api/agent-sessions/windows",
                json={
                    "client": "codex",
                    "session_id": "sess-window",
                    "trigger": "PreCompact",
                    "workspace": "/workspace/mem-forge",
                    "repo": "mem-forge",
                    "branch": "main",
                    "commit_sha": "abc123",
                    "history_window": {
                        "kind": "boundary",
                        "start_event_id": "evt-1",
                        "end_event_id": "evt-3",
                        "text": "api_key: history-secret",
                    },
                    "events": [
                        {"role": "user", "text": "Implement the window upload endpoint."},
                        {
                            "role": "tool",
                            "name": "apply_patch",
                            "summary": "Edited admin_api.py with api_key: raw-api-secret.",
                        },
                    ],
                    "transcript_markdown": "token: secret-value",
                    "receipt": {"hook": "PreCompact", "has_transcript_path": True},
                    "retention": "none",
                },
            )

        assert response.status_code == 200
        body = response.json()
        assert body["result"] == "knowledge_patched"
        assert body["patch_outcome"] == "applied"
        assert body["source_id"] == "src-agent-sessions-codex"
        assert body["sync_started"] is False
        assert body["window_hash"].startswith("sha256:")
        memory = asyncio.run(database.get_memory(body["memory_id"]))
        assert memory is not None
        assert "window upload endpoint" in memory.content
        assert "secret-value" not in memory.content
        receipt = asyncio.run(database.get_agent_session_receipt(body["concept_id"]))
        assert receipt is None
    finally:
        asyncio.run(database.close())


def test_agent_session_window_api_canonicalizes_evidence_before_packaging(tmp_path):
    from memforge.server.admin_api import create_admin_app

    class FakeWindowClient:
        async def generate_agent_knowledge_patch(self, prompt: str, **kwargs):
            assert "Canonical evidence" in prompt
            assert "apply_patch" in prompt
            assert "Edited src/memforge/hook_adapter.py" in prompt
            assert "service-json-secret" not in prompt
            assert "session_meta" not in prompt
            assert "private developer bootstrap" not in prompt
            assert "raw JSONL prefix" not in prompt
            return _knowledge_patch(
                title="Agent Session: canonical evidence",
                claim_text="Canonical evidence is filtered before generating agent knowledge patches.",
            )

    cfg = _config(tmp_path)
    database = Database(str(tmp_path / "api.db"))

    async def _setup():
        await database.connect()

    import asyncio

    asyncio.run(_setup())
    try:
        app = create_admin_app(db=database, config=cfg)
        app.state.agent_session_window_client = FakeWindowClient()
        with TestClient(app) as client:
            response = client.post(
                "/api/agent-sessions/windows",
                json={
                    "client": "codex",
                    "session_id": "sess-canonical",
                    "trigger": "REQUIRED_CAPTURE",
                    "workspace": "/workspace/mem-forge",
                    "events": [
                        {
                            "type": "session_meta",
                            "preview": "private developer bootstrap",
                        },
                        {
                            "kind": "tool_call",
                            "actor": "assistant",
                            "name": "apply_patch",
                            "text": "Edited src/memforge/hook_adapter.py",
                            "input": {"api_key": "service-json-secret"},
                        },
                    ],
                    "transcript_markdown": (
                        '{"type":"session_meta","payload":"raw JSONL prefix"}\n'
                        '{"type":"turn_context","payload":"private developer bootstrap"}'
                    ),
                    "history_window": {
                        "kind": "transcript_window",
                        "start": "0",
                        "end": "2",
                    },
                    "retention": "none",
                },
            )

        assert response.status_code == 200
        assert response.json()["result"] == "knowledge_patched"
    finally:
        asyncio.run(database.close())


def test_agent_session_window_api_queues_service_owned_sync(tmp_path):
    from memforge.server.admin_api import create_admin_app

    class FakeWindowClient:
        async def generate_agent_knowledge_patch(self, prompt: str, **kwargs):
            return _knowledge_patch(
                title="Agent Session: queued service sync",
                claim_text="Agent session windows write knowledge directly without queuing source sync.",
            )

    class FakeSyncService:
        def __init__(self):
            self.queued: list[str] = []

        async def request_source_sync(self, source_id: str) -> bool:
            self.queued.append(source_id)
            return True

        async def start_source(self, source_id: str, *, force_full_sync: bool = False):
            raise AssertionError("plugin-style immediate sync should not be used")

        async def shutdown(self):
            return None

    cfg = _config(tmp_path)
    database = Database(str(tmp_path / "api.db"))

    async def _setup():
        await database.connect()

    import asyncio

    asyncio.run(_setup())
    try:
        app = create_admin_app(db=database, config=cfg)
        app.state.agent_session_window_client = FakeWindowClient()
        fake_sync = FakeSyncService()
        with TestClient(app) as client:
            client.app.state.sync_service = fake_sync
            response = client.post(
                "/api/agent-sessions/windows",
                json={
                    "client": "codex",
                    "session_id": "sess-window-sync",
                    "trigger": "PreCompact",
                    "workspace": "/workspace/mem-forge",
                    "events": [{"role": "tool", "name": "apply_patch", "summary": "Edited code."}],
                    "retention": "none",
                    "process_now": False,
                },
            )

        assert response.status_code == 200
        body = response.json()
        assert body["result"] == "knowledge_patched"
        assert body["sync_started"] is False
        assert body["sync_queued"] is False
        assert fake_sync.queued == []
    finally:
        asyncio.run(database.close())


def test_agent_session_window_api_rejects_unknown_schema_version(tmp_path):
    from memforge.server.admin_api import create_admin_app

    class UnexpectedWindowClient:
        async def generate_agent_knowledge_patch(self, prompt: str, **kwargs):
            raise AssertionError("unsupported schema should fail before LLM")

    cfg = _config(tmp_path)
    database = Database(str(tmp_path / "api.db"))

    async def _setup():
        await database.connect()

    import asyncio

    asyncio.run(_setup())
    try:
        app = create_admin_app(db=database, config=cfg)
        app.state.agent_session_window_client = UnexpectedWindowClient()
        with TestClient(app) as client:
            response = client.post(
                "/api/agent-sessions/windows",
                json={
                    "schema_version": "agent-session-window/v999",
                    "client": "codex",
                    "session_id": "sess-window-schema",
                    "trigger": "PreCompact",
                    "workspace": "/workspace/mem-forge",
                    "events": [{"role": "tool", "name": "apply_patch", "summary": "Edited code."}],
                    "retention": "none",
                    "process_now": False,
                },
            )

        assert response.status_code == 400
        assert "unsupported schema_version" in response.json()["detail"]
    finally:
        asyncio.run(database.close())


def test_agent_session_window_api_accepts_no_output_without_creating_source(tmp_path):
    from memforge.server.admin_api import create_admin_app

    class NoOutputClient:
        async def generate_agent_knowledge_patch(self, prompt: str, **kwargs):
            return _knowledge_patch(
                action="no_output",
                title=None,
                concept_type=None,
                claim_text="",
                reason="trivial explanation",
            )

    cfg = _config(tmp_path)
    database = Database(str(tmp_path / "api.db"))

    async def _setup():
        await database.connect()

    import asyncio

    asyncio.run(_setup())
    try:
        app = create_admin_app(db=database, config=cfg)
        app.state.agent_session_window_client = NoOutputClient()
        with TestClient(app) as client:
            response = client.post(
                "/api/agent-sessions/windows",
                json={
                    "client": "codex",
                    "session_id": "sess-window-trivial",
                    "trigger": "Stop",
                    "workspace": "/workspace/mem-forge",
                    "events": [{"role": "assistant", "text": "Sure, that function formats text."}],
                    "process_now": False,
                },
            )

        assert response.status_code == 200
        body = response.json()
        assert body["result"] == "no_output"
        assert body["reason"] == "trivial explanation"

        async def _assert_no_source():
            assert await database.get_source("src-agent-sessions-codex") is None

        asyncio.run(_assert_no_source())
    finally:
        asyncio.run(database.close())


def test_agent_session_window_api_reports_missing_llm(tmp_path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = Database(str(tmp_path / "api.db"))

    async def _setup():
        await database.connect()

    import asyncio

    asyncio.run(_setup())
    try:
        app = create_admin_app(db=database, config=cfg)
        with TestClient(app) as client:
            response = client.post(
                "/api/agent-sessions/windows",
                json={
                    "client": "codex",
                    "session_id": "sess-window-no-llm",
                    "trigger": "PreCompact",
                    "workspace": "/workspace/mem-forge",
                    "events": [{"role": "tool", "name": "apply_patch", "summary": "Edited code."}],
                    "process_now": False,
                },
            )

        assert response.status_code == 503
        assert "LLM unavailable" in response.json()["detail"]
    finally:
        asyncio.run(database.close())


def test_agent_session_window_api_records_failed_outcome_for_invalid_patch(tmp_path):
    from memforge.server.admin_api import create_admin_app

    class InvalidPatchClient:
        async def generate_agent_knowledge_patch(self, prompt: str, **kwargs):
            return {"action": "unsupported_action", "claim_text": "Invalid action."}

    cfg = _config(tmp_path)
    database = Database(str(tmp_path / "api.db"))

    async def _setup():
        await database.connect()

    import asyncio

    asyncio.run(_setup())
    try:
        app = create_admin_app(db=database, config=cfg)
        app.state.agent_session_window_client = InvalidPatchClient()
        with TestClient(app) as client:
            response = client.post(
                "/api/agent-sessions/windows",
                json={
                    "client": "codex",
                    "session_id": "sess-window-bad-patch",
                    "trigger": "PreCompact",
                    "workspace": "/workspace/mem-forge",
                    "events": [{"role": "tool", "name": "apply_patch", "summary": "Edited code."}],
                    "process_now": False,
                },
            )

        assert response.status_code == 400
        assert "agent knowledge patch validation failed" in response.json()["detail"]

        async def _assert_failed_receipt():
            summary = await database.summarize_agent_session_outcomes(
                session_id="sess-window-bad-patch"
            )
            assert summary["counts"]["failed"] == 1
            assert summary["latest_failure"]["reason"].startswith("ValidationError:")

        asyncio.run(_assert_failed_receipt())
    finally:
        asyncio.run(database.close())


def test_agent_session_window_api_records_failed_outcome_for_parse_failed_patch(tmp_path):
    from memforge.server.admin_api import create_admin_app

    class ParseFailedPatchClient:
        async def generate_agent_knowledge_patch(self, prompt: str, **kwargs):
            return _knowledge_patch(memory_content=None)

    cfg = _config(tmp_path)
    database = Database(str(tmp_path / "api.db"))

    import asyncio

    asyncio.run(database.connect())
    try:
        app = create_admin_app(db=database, config=cfg)
        app.state.agent_session_window_client = ParseFailedPatchClient()
        with TestClient(app) as client:
            response = client.post(
                "/api/agent-sessions/windows",
                json={
                    "client": "codex",
                    "session_id": "sess-window-parse-failed",
                    "trigger": "Stop",
                    "workspace": "/workspace/mem-forge",
                    "events": [{"role": "tool", "name": "apply_patch", "summary": "Edited code."}],
                },
            )

        assert response.status_code == 200
        body = response.json()
        assert body["result"] == "failed"
        assert body["patch_outcome"] == "parse_failed"
        assert body["reason"] == "memory_content is required"

        async def _assert_failed_receipt():
            summary = await database.summarize_agent_session_outcomes(
                session_id="sess-window-parse-failed"
            )
            assert summary["counts"]["failed"] == 1
            assert summary["latest_failure"]["reason"] == "memory_content is required"

        asyncio.run(_assert_failed_receipt())
    finally:
        asyncio.run(database.close())


def test_agent_session_window_api_keeps_windows_distinct_and_idempotent(tmp_path):
    """Windows of one session/trigger get distinct, idempotent memory patches."""
    from memforge.server.admin_api import create_admin_app

    class EchoWindowClient:
        async def generate_agent_knowledge_patch(self, prompt: str, **kwargs):
            return _knowledge_patch(
                title="Agent Session Window",
                claim_text="Window content recorded as a durable private agent memory.",
            )

    cfg = _config(tmp_path)
    database = Database(str(tmp_path / "api.db"))

    async def _setup():
        await database.connect()

    import asyncio

    asyncio.run(_setup())

    def _window(events):
        return {
            "client": "codex",
            "session_id": "sess-distinct",
            "trigger": "Stop",
            "workspace": "/workspace/mem-forge",
            "repo": "mem-forge",
            "events": events,
            "history_window": {"kind": "transcript_window"},
            "retention": "none",
        }

    try:
        app = create_admin_app(db=database, config=cfg)
        app.state.agent_session_window_client = EchoWindowClient()
        with TestClient(app) as client:
            first = client.post(
                "/api/agent-sessions/windows",
                json=_window([{"role": "tool", "name": "apply_patch", "summary": "Edited module A."}]),
            ).json()
            second = client.post(
                "/api/agent-sessions/windows",
                json=_window([{"role": "tool", "name": "apply_patch", "summary": "Edited module B."}]),
            ).json()
            repeat_first = client.post(
                "/api/agent-sessions/windows",
                json=_window([{"role": "tool", "name": "apply_patch", "summary": "Edited module A."}]),
            ).json()

        # Different window content -> different memory patch.
        assert first["window_hash"] != second["window_hash"]
        assert first["memory_id"] != second["memory_id"]

        # Identical window content -> same memory patch (idempotent retry).
        assert repeat_first["window_hash"] == first["window_hash"]
        assert repeat_first["memory_id"] == first["memory_id"]
        assert repeat_first["idempotent"] is True
    finally:
        asyncio.run(database.close())


def test_agent_session_window_api_same_range_different_content_is_distinct(tmp_path):
    """A reused explicit event range with different content must not overwrite."""
    from memforge.server.admin_api import create_admin_app

    class EchoWindowClient:
        async def generate_agent_knowledge_patch(self, prompt: str, **kwargs):
            return _knowledge_patch(
                title="Agent Session Window",
                claim_text="A reused event range with different content creates a distinct memory patch.",
            )

    cfg = _config(tmp_path)
    database = Database(str(tmp_path / "api.db"))

    import asyncio

    asyncio.run(database.connect())

    def _window(events):
        return {
            "client": "codex",
            "session_id": "sess-range",
            "trigger": "PreCompact",
            "workspace": "/workspace/mem-forge",
            "events": events,
            # Same explicit event-id boundary on every call.
            "history_window": {"kind": "boundary", "start_event_id": "evt-1", "end_event_id": "evt-9"},
            "retention": "none",
        }

    try:
        app = create_admin_app(db=database, config=cfg)
        app.state.agent_session_window_client = EchoWindowClient()
        with TestClient(app) as client:
            a = client.post(
                "/api/agent-sessions/windows",
                json=_window([{"role": "tool", "name": "apply_patch", "summary": "edited A"}]),
            ).json()
            b = client.post(
                "/api/agent-sessions/windows",
                json=_window([{"role": "tool", "name": "apply_patch", "summary": "edited B"}]),
            ).json()
            repeat_a = client.post(
                "/api/agent-sessions/windows",
                json=_window([{"role": "tool", "name": "apply_patch", "summary": "edited A"}]),
            ).json()

        # Same [evt-1, evt-9] range, different content -> distinct memory patches.
        assert a["window_hash"] != b["window_hash"]
        assert a["memory_id"] != b["memory_id"]
        # Identical content -> same memory patch (idempotent).
        assert repeat_a["memory_id"] == a["memory_id"]
    finally:
        asyncio.run(database.close())


def test_agent_session_window_retry_identity_ignores_receipt_and_submission_date(tmp_path):
    """The same range/content retry keeps one doc even if receipt metadata changes."""
    from memforge.server.admin_api import create_admin_app

    class EchoWindowClient:
        async def generate_agent_knowledge_patch(self, prompt: str, **kwargs):
            return _knowledge_patch(
                title="Agent Session Window",
                claim_text="The same range and content retry reuses the existing memory patch.",
            )

    cfg = _config(tmp_path)
    database = Database(str(tmp_path / "api.db"))

    import asyncio

    asyncio.run(database.connect())

    def _window(*, receipt, submitted_at):
        return {
            "client": "codex",
            "session_id": "sess-retry",
            "trigger": "Stop",
            "workspace": "/workspace/mem-forge",
            "events": [{"role": "tool", "name": "apply_patch", "summary": "same edit"}],
            "history_window": {"kind": "transcript_window", "start": "10", "end": "20"},
            "receipt": receipt,
            "submitted_at": submitted_at,
            "retention": "none",
        }

    try:
        app = create_admin_app(db=database, config=cfg)
        app.state.agent_session_window_client = EchoWindowClient()
        with TestClient(app) as client:
            first = client.post(
                "/api/agent-sessions/windows",
                json=_window(receipt={"attempt": 1}, submitted_at="2026-05-30T23:59:00+00:00"),
            ).json()
            retry = client.post(
                "/api/agent-sessions/windows",
                json=_window(receipt={"attempt": 2}, submitted_at="2026-05-31T00:01:00+00:00"),
            ).json()

        assert retry["window_hash"] == first["window_hash"]
        assert retry["memory_id"] == first["memory_id"]
        assert retry["idempotent"] is True
    finally:
        asyncio.run(database.close())


def test_agent_session_window_can_patch_existing_private_claim(tmp_path):
    from memforge.server.admin_api import create_admin_app

    class PatchClient:
        def __init__(self):
            self.created_concept_id: str | None = None
            self.created_claim_id: str | None = None

        async def generate_agent_knowledge_patch(self, prompt: str, **kwargs):
            if self.created_concept_id is None:
                return _knowledge_patch(
                    title="Scheduler lifecycle",
                    claim_text="Workspace source schedulers must start during app startup.",
                    memory_content="Workspace source schedulers must start during app startup.",
                )
            assert f"concept_id={self.created_concept_id}" in prompt
            assert f"claim_id={self.created_claim_id}" in prompt
            return _knowledge_patch(
                action="update_existing_claim",
                concept_id=self.created_concept_id,
                claim_id=self.created_claim_id,
                claim_text=(
                    "Workspace source schedulers must start during app startup "
                    "and advance next_run_at after claiming overdue schedules."
                ),
                memory_content="Workspace source schedulers start during app startup and advance next_run_at after claiming overdue schedules.",
            )

    cfg = _config(tmp_path)
    database = Database(str(tmp_path / "api.db"))
    import asyncio

    asyncio.run(database.connect())

    def _window(summary: str):
        return {
            "client": "codex",
            "session_id": "sess-patch-existing",
            "trigger": "Stop",
            "workspace": "/workspace/memforge-cloud",
            "repo": "github.tools.sap/hcm/memforge-cloud",
            "events": [{"role": "tool", "name": "apply_patch", "summary": summary}],
            "history_window": {"kind": "transcript_window", "start": summary, "end": summary},
            "retention": "none",
        }

    try:
        app = create_admin_app(db=database, config=cfg, principal_resolver=lambda request: "u-authorized")
        fake_client = PatchClient()
        app.state.agent_session_window_client = fake_client
        with TestClient(app) as client:
            first = client.post(
                "/api/agent-sessions/windows",
                json=_window("scheduler startup rule"),
            ).json()
            fake_client.created_concept_id = first["concept_id"]
            fake_client.created_claim_id = first["claim_id"]
            second = client.post(
                "/api/agent-sessions/windows",
                json=_window("scheduler next_run_at rule"),
            ).json()

        assert second["concept_id"] == first["concept_id"]
        assert second["claim_id"] == first["claim_id"]
        assert second["memory_id"] != first["memory_id"]
        old_memory = asyncio.run(database.get_memory(first["memory_id"]))
        assert old_memory is not None
        assert old_memory.status == "superseded"
        assert old_memory.superseded_by == second["memory_id"]
        memory = asyncio.run(database.get_memory(second["memory_id"]))
        assert memory is not None
        assert "advance next_run_at" in memory.content
    finally:
        asyncio.run(database.close())


def test_agent_session_window_api_records_no_output_receipt(tmp_path):
    """A no_output window still leaves a traceable receipt, not silence."""
    from memforge.server.admin_api import create_admin_app

    class NoOutputClient:
        async def generate_agent_knowledge_patch(self, prompt: str, **kwargs):
            return _knowledge_patch(
                action="no_output",
                title=None,
                concept_type=None,
                claim_text="",
                reason="trivial chat",
            )

    cfg = _config(tmp_path)
    database = Database(str(tmp_path / "api.db"))

    import asyncio

    asyncio.run(database.connect())
    try:
        app = create_admin_app(db=database, config=cfg)
        app.state.agent_session_window_client = NoOutputClient()
        with TestClient(app) as client:
            response = client.post(
                "/api/agent-sessions/windows",
                json={
                    "client": "codex",
                    "session_id": "sess-noout-receipt",
                    "trigger": "Stop",
                    "workspace": "/workspace/mem-forge",
                    "events": [{"role": "assistant", "text": "Sure, that formats text."}],
                },
            )

        assert response.status_code == 200
        assert response.json()["result"] == "no_output"

        async def _check():
            receipts = await database.list_agent_session_receipts(session_id="sess-noout-receipt")
            assert len(receipts) == 1
            metadata = receipts[0]["metadata"]
            assert metadata["outcome"] == "no_output"
            assert metadata["reason"] == "trivial chat"

        asyncio.run(_check())
    finally:
        asyncio.run(database.close())


def test_agent_session_window_api_records_failed_receipt(tmp_path):
    """A Stage-1 failure leaves a `failed` receipt so the loss is recorded."""
    from memforge.server.admin_api import create_admin_app

    class FailingClient:
        async def generate_agent_knowledge_patch(self, prompt: str, **kwargs):
            raise RuntimeError("llm exploded")

    cfg = _config(tmp_path)
    database = Database(str(tmp_path / "api.db"))

    import asyncio

    asyncio.run(database.connect())
    try:
        app = create_admin_app(db=database, config=cfg)
        app.state.agent_session_window_client = FailingClient()
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                "/api/agent-sessions/windows",
                json={
                    "client": "codex",
                    "session_id": "sess-failed-receipt",
                    "trigger": "Stop",
                    "workspace": "/workspace/mem-forge",
                    "events": [{"role": "tool", "name": "apply_patch", "summary": "edit"}],
                },
            )

        assert response.status_code == 500

        async def _check():
            receipts = await database.list_agent_session_receipts(session_id="sess-failed-receipt")
            assert len(receipts) == 1
            metadata = receipts[0]["metadata"]
            assert metadata["outcome"] == "failed"
            assert "llm exploded" in metadata["reason"]

        asyncio.run(_check())
    finally:
        asyncio.run(database.close())


def test_hook_receipt_api_records_lineage_without_source_document(tmp_path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = Database(str(tmp_path / "api.db"))

    async def _setup():
        await database.connect()

    import asyncio

    asyncio.run(_setup())
    try:
        app = create_admin_app(db=database, config=cfg)
        with TestClient(app) as client:
            response = client.post(
                "/api/hooks/receipts",
                json={
                    "client": "codex",
                    "session_id": "sess-hook",
                    "hook": "Stop",
                    "workspace": "/workspace/mem-forge",
                    "repo": "mem-forge",
                    "branch": "main",
                    "commit_sha": "abc123",
                    "metadata": {"has_transcript_path": True},
                },
            )

        assert response.status_code == 200
        body = response.json()
        assert body["receipt"]["client"] == "codex"
        assert body["receipt"]["hook"] == "Stop"
        assert body["receipt"]["metadata"] == {"has_transcript_path": True}

        async def _assert_storage():
            receipts = await database.list_agent_hook_receipts(session_id="sess-hook")
            source = await database.get_source("src-agent-sessions-codex")
            assert len(receipts) == 1
            assert receipts[0]["receipt_id"] == body["receipt_id"]
            assert source is None

        asyncio.run(_assert_storage())
    finally:
        asyncio.run(database.close())


def _make_receipt(
    *,
    doc_id,
    session_id,
    outcome,
    source_id="src-agent-sessions-codex",
    source_kind="generated_agent_window_summary",
    reason=None,
    updated_at=None,
):
    from memforge.models import AgentSessionReceipt

    metadata: dict = {"outcome": outcome}
    if reason is not None:
        metadata["reason"] = reason

    return AgentSessionReceipt(
        doc_id=doc_id,
        source_id=source_id,
        client="codex",
        session_id=session_id,
        trigger="Stop",
        workspace="/workspace/mem-forge",
        repo=None,
        branch=None,
        commit_sha=None,
        history_window_kind="transcript_window",
        history_window_start=None,
        history_window_end=None,
        submitted_at="2026-05-30T00:00:00+00:00",
        document_hash="sha256:deadbeef",
        source_kind=source_kind,
        document_uri="",
        metadata=metadata,
        updated_at=updated_at,
    )


def test_summarize_agent_session_outcomes_counts_and_fraction(tmp_path):
    database = Database(str(tmp_path / "api.db"))
    import asyncio

    async def _run():
        await database.connect()
        try:
            outcomes = ["knowledge_patched", "package_created", "no_output", "failed"]
            for index, outcome in enumerate(outcomes):
                await database.upsert_agent_session_receipt(
                    _make_receipt(doc_id=f"doc-{index}", session_id="sess-sum", outcome=outcome)
                )
            summary = await database.summarize_agent_session_outcomes(session_id="sess-sum")
            assert summary["session_id"] == "sess-sum"
            assert summary["total"] == 4
            assert summary["processed_total"] == 3
            assert summary["counts"] == {"knowledge_patched": 2, "no_output": 1, "failed": 1}
            assert summary["no_output_fraction"] == 1 / 3
        finally:
            await database.close()

    asyncio.run(_run())


def test_summarize_empty_session_returns_zero_fraction(tmp_path):
    database = Database(str(tmp_path / "api.db"))
    import asyncio

    async def _run():
        await database.connect()
        try:
            summary = await database.summarize_agent_session_outcomes(session_id="nobody")
            assert summary["total"] == 0
            assert summary["processed_total"] == 0
            assert summary["counts"] == {"knowledge_patched": 0, "no_output": 0, "failed": 0}
            assert summary["no_output_fraction"] == 0.0
            assert summary["latest_failure"] is None
        finally:
            await database.close()

    asyncio.run(_run())


def test_summarize_agent_session_outcomes_includes_latest_failure(tmp_path):
    database = Database(str(tmp_path / "api.db"))
    import asyncio

    async def _run():
        await database.connect()
        try:
            await database.upsert_agent_session_receipt(
                _make_receipt(
                    doc_id="fail-1",
                    session_id="sess-fail",
                    outcome="failed",
                    reason="LLM timeout",
                    updated_at="2026-05-29T00:00:00+00:00",
                )
            )
            await database.upsert_agent_session_receipt(
                _make_receipt(
                    doc_id="fail-2",
                    session_id="sess-fail",
                    outcome="failed",
                    reason="schema validation error",
                    updated_at="2026-05-30T12:00:00+00:00",
                )
            )
            await database.upsert_agent_session_receipt(
                _make_receipt(
                    doc_id="ok-1",
                    session_id="sess-fail",
                    outcome="knowledge_patched",
                )
            )
            summary = await database.summarize_agent_session_outcomes(session_id="sess-fail")
            assert summary["counts"]["failed"] == 2
            assert summary["latest_failure"] == {
                "count": 2,
                "reason": "schema validation error",
                "last_seen_at": "2026-05-30T12:00:00+00:00",
            }
        finally:
            await database.close()

    asyncio.run(_run())


def test_agent_session_completeness_endpoint(tmp_path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = Database(str(tmp_path / "api.db"))
    import asyncio

    asyncio.run(database.connect())
    try:
        async def _seed():
            for index, outcome in enumerate(["knowledge_patched", "no_output"]):
                await database.upsert_agent_session_receipt(
                    _make_receipt(doc_id=f"ep-{index}", session_id="sess-ep", outcome=outcome)
                )

        asyncio.run(_seed())
        app = create_admin_app(db=database, config=cfg)
        with TestClient(app) as client:
            response = client.get(
                "/api/agent-sessions/completeness", params={"session_id": "sess-ep"}
            )

        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 2
        assert body["processed_total"] == 2
        assert body["counts"] == {"knowledge_patched": 1, "no_output": 1, "failed": 0}
        assert body["no_output_fraction"] == 0.5
        assert body["latest_failure"] is None
    finally:
        asyncio.run(database.close())


def test_summarize_agent_session_outcomes_ignores_explicit_document_metadata(tmp_path):
    database = Database(str(tmp_path / "api.db"))
    import asyncio

    async def _run():
        await database.connect()
        try:
            await database.upsert_agent_session_receipt(
                _make_receipt(
                    doc_id="window-doc",
                    session_id="sess-filter",
                    outcome="knowledge_patched",
                )
            )
            await database.upsert_agent_session_receipt(
                _make_receipt(
                    doc_id="explicit-doc",
                    session_id="sess-filter",
                    outcome="no_output",
                    source_kind="generated_agent_summary",
                )
            )
            summary = await database.summarize_agent_session_outcomes(session_id="sess-filter")
            assert summary["total"] == 1
            assert summary["processed_total"] == 1
            assert summary["counts"] == {"knowledge_patched": 1, "no_output": 0, "failed": 0}
            assert summary["no_output_fraction"] == 0.0
        finally:
            await database.close()

    asyncio.run(_run())


def test_agent_window_patch_writes_memory_without_package_file(tmp_path):
    from memforge.server.admin_api import create_admin_app

    class FakeWindowClient:
        async def generate_agent_knowledge_patch(self, prompt: str, **kwargs):
            return _knowledge_patch(
                title="Agent Session: atomic write",
                claim_text="The agent-session window was written directly as a private memory patch.",
            )

    cfg = _config(tmp_path)
    database = Database(str(tmp_path / "api.db"))
    import asyncio

    asyncio.run(database.connect())
    try:
        app = create_admin_app(db=database, config=cfg)
        app.state.agent_session_window_client = FakeWindowClient()
        with TestClient(app) as client:
            response = client.post(
                "/api/agent-sessions/windows",
                json={
                    "client": "codex",
                    "session_id": "sess-atomic",
                    "trigger": "Stop",
                    "workspace": "/workspace/mem-forge",
                    "events": [{"role": "tool", "name": "apply_patch", "summary": "edit"}],
                    "transcript_markdown": "did some real work worth keeping",
                },
            )

        assert response.status_code == 200
        body = response.json()
        assert "document_uri" not in body
        memory = asyncio.run(database.get_memory(body["memory_id"]))
        assert memory is not None
        assert "private memory patch" in memory.content
    finally:
        asyncio.run(database.close())


def test_per_client_source_split_creates_two_distinct_source_rows(tmp_path):
    """Submitting from codex and claude-code creates two separate source rows."""
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = Database(str(tmp_path / "api.db"))
    import asyncio

    asyncio.run(database.connect())
    try:
        app = create_admin_app(db=database, config=cfg)
        with TestClient(app) as client:
            codex_response = client.post(
                "/api/agent-sessions/documents",
                json={
                    "client": "codex",
                    "session_id": "sess-codex",
                    "trigger": "Stop",
                    "workspace": "/workspace/mem-forge",
                    "document_markdown": "## Outcome\nCodex session summary.",
                    "process_now": False,
                },
            )
            assert codex_response.status_code == 200

            claude_response = client.post(
                "/api/agent-sessions/documents",
                json={
                    "client": "claude-code",
                    "session_id": "sess-claude-code",
                    "trigger": "Stop",
                    "workspace": "/workspace/mem-forge",
                    "document_markdown": "## Outcome\nClaude Code session summary.",
                    "process_now": False,
                },
            )
            assert claude_response.status_code == 200

        codex_body = codex_response.json()
        claude_body = claude_response.json()

        assert codex_body["source_id"] == "src-agent-sessions-codex"
        assert claude_body["source_id"] == "src-agent-sessions-claude-code"
        assert codex_body["receipt"]["client"] == "codex"
        assert claude_body["receipt"]["client"] == "claude-code"

        async def _check_sources():
            sources = await database.list_sources()
            source_ids = {s["id"] for s in sources}
            assert "src-agent-sessions-codex" in source_ids
            assert "src-agent-sessions-claude-code" in source_ids
            # The legacy singleton must not be created for known clients.
            assert "src-agent-sessions" not in source_ids

            codex_src = await database.get_source("src-agent-sessions-codex")
            claude_src = await database.get_source("src-agent-sessions-claude-code")
            assert codex_src is not None
            assert codex_src["name"] == "Codex Session"
            assert codex_src["type"] == "agent_session"
            assert claude_src is not None
            assert claude_src["name"] == "Claude Code Session"
            assert claude_src["type"] == "agent_session"

        asyncio.run(_check_sources())
    finally:
        asyncio.run(database.close())


def test_db_migration_splits_singleton_documents_to_per_client_sources(tmp_path):
    """Migration 12 re-points documents from the old singleton to per-client sources."""
    import asyncio
    import aiosqlite

    async def _run():
        db_path = str(tmp_path / "migration.db")
        # Open a raw connection and apply only the base schema and migrations
        # up through 10, so we can seed singleton data before 11/12 run.
        conn = await aiosqlite.connect(db_path)
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA foreign_keys = ON")

        # Apply just enough schema for our test.
        from memforge.storage.database import SCHEMA, MIGRATIONS
        await conn.executescript(SCHEMA)
        now_ts = "2026-06-01T10:00:00+00:00"
        # Record migrations 1-10 as applied without executing (schema already created them).
        for version, description, _ in MIGRATIONS:
            if version > 10:
                break
            await conn.execute(
                "INSERT OR IGNORE INTO schema_migrations (version, description, applied_at) VALUES (?, ?, ?)",
                (version, description, now_ts),
            )
        await conn.commit()

        # Seed singleton source and documents.
        await conn.execute(
            "INSERT INTO sources (id, type, name, config) VALUES (?, ?, ?, ?)",
            ("src-agent-sessions", "agent_session", "Agent Session Summaries", "{}"),
        )
        for client_name, doc_id in [("codex", "doc-codex-m"), ("claude-code", "doc-cc-m")]:
            await conn.execute(
                """INSERT INTO documents
                   (doc_id, source, source_url, title, space_or_project, author,
                    last_modified, labels, version, content_hash, last_synced)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (doc_id, "src-agent-sessions",
                 f"agent-session://{client_name}/sess/{doc_id}",
                 f"{client_name} doc", "workspace", client_name,
                 now_ts, "[]", "v1", f"hash-{doc_id}", now_ts),
            )
            await conn.execute(
                """INSERT INTO agent_session_receipts
                   (doc_id, source_id, client, session_id, trigger, workspace,
                    history_window_kind, submitted_at, document_hash, source_kind,
                    document_uri, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (doc_id, "src-agent-sessions", client_name, "sess-x", "Stop",
                 "/workspace", "session", now_ts, f"hash-{doc_id}",
                 "generated_agent_summary", "", now_ts),
            )
        await conn.commit()
        await conn.close()

        # Now open via Database, which will run migrations 11 and 12.
        database = Database(db_path)
        await database.connect()
        try:
            codex_src = await database.get_source("src-agent-sessions-codex")
            cc_src = await database.get_source("src-agent-sessions-claude-code")
            singleton_src = await database.get_source("src-agent-sessions")
            codex_doc = await database.get_document("doc-codex-m")
            cc_doc = await database.get_document("doc-cc-m")
        finally:
            await database.close()

        assert codex_src is not None, "codex source must exist after migration"
        assert codex_src["name"] == "Codex Session"
        assert codex_src["type"] == "agent_session"
        assert cc_src is not None, "claude-code source must exist after migration"
        assert cc_src["name"] == "Claude Code Session"
        # Both known-client docs were re-pointed, so singleton has zero docs.
        assert singleton_src is None, "singleton must be removed after all docs are re-pointed"
        assert codex_doc is not None
        assert codex_doc.source == "src-agent-sessions-codex"
        assert codex_doc.client == "codex"
        assert cc_doc is not None
        assert cc_doc.source == "src-agent-sessions-claude-code"
        assert cc_doc.client == "claude-code"

    asyncio.run(_run())


def test_memories_endpoint_exposes_origin_client_for_agent_session_memories(tmp_path):
    """origin_client is 'codex' or 'claude-code' for agent-session memories and None for jira."""
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = Database(str(tmp_path / "api.db"))
    base_time = datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc)

    async def _setup():
        await database.connect()
        # Seed a jira source and document.
        await database.upsert_source("src-jira", "jira", "Jira", json.dumps({}))
        jira_doc_id = "doc-jira-1"
        await database.upsert_document(
            DocumentRecord(
                doc_id=jira_doc_id,
                source="src-jira",
                source_url="https://jira.example.com/PAY-1",
                title="PAY-1",
                space_or_project="payroll",
                author=None,
                last_modified=base_time,
                labels=[],
                version="v1",
                content_hash="hash-jira",
                token_count=50,
                raw_content_uri=None,
                raw_content_type="text/plain",
                normalized_content_uri=None,
                pdf_content_uri=None,
                last_synced=base_time,
            )
        )
        # Seed codex and claude-code agent session documents with the client column set.
        await _seed_source_project(
            database,
            doc_id="doc-codex-origin",
            project="mem-inception",
            last_modified=base_time,
            memory_ids=["mem-codex-origin"],
            source_id="src-agent-sessions-codex",
            client="codex",
        )
        await _seed_source_project(
            database,
            doc_id="doc-cc-origin",
            project="mem-inception",
            last_modified=base_time,
            memory_ids=["mem-cc-origin"],
            source_id="src-agent-sessions-claude-code",
            client="claude-code",
        )
        # Jira memory without a client column on its document.
        jira_memory = Memory(
            id="mem-jira-origin",
            memory_type="fact",
            content="Jira memory",
            content_hash=content_hash("Jira memory"),
            project_key="payroll",
            tags=[],
            confidence=0.9,
            created_at=base_time,
            updated_at=base_time,
            status="active",
        )
        await database.insert_memory(jira_memory)
        await database.add_memory_source(jira_memory.id, jira_doc_id, "jira")

    import asyncio

    asyncio.run(_setup())
    try:
        app = create_admin_app(db=database, config=cfg)
        with TestClient(app) as client:
            response = client.get("/api/memories")

        assert response.status_code == 200
        data = {m["id"]: m for m in response.json()["data"]}

        assert data["mem-codex-origin"]["origin_client"] == "codex"
        assert data["mem-codex-origin"]["origin_source_type"] == "agent_session"

        assert data["mem-cc-origin"]["origin_client"] == "claude-code"
        assert data["mem-cc-origin"]["origin_source_type"] == "agent_session"

        assert data["mem-jira-origin"]["origin_client"] is None
        assert data["mem-jira-origin"]["origin_source_type"] == "jira"
    finally:
        asyncio.run(database.close())
