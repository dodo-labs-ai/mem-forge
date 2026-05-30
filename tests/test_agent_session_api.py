from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from meminception.config import AppConfig
from meminception.storage.database import Database


def _config(tmp_path: Path) -> AppConfig:
    cfg = AppConfig(base_dir=tmp_path / "mem")
    cfg.server.jwt_secret = "test-secret"
    cfg.llm.enrichment_api_key = ""
    cfg.llm.embedding_api_key = ""
    return cfg


def test_agent_session_window_prompt_uses_memory_quality_gate():
    from meminception.agent_sessions import render_agent_session_window_prompt

    prompt = render_agent_session_window_prompt(
        client="codex",
        session_id="sess-prompt",
        trigger="REQUIRED_CAPTURE",
        workspace="/workspace/mem-inception",
        repo="mem-inception",
        branch="main",
        history_window={"from": 0, "to": 2},
        events=[
            {"kind": "user_message", "actor": "user", "text": "Please keep this clean and light."},
            {"kind": "tool_result", "actor": "tool", "text": "uv run pytest -q passed."},
        ],
        transcript_markdown=None,
    )

    assert "Will a future agent plausibly act better because this package exists?" in prompt
    assert "Prefer evidence in this order:" in prompt
    assert "User-confirmed decisions, constraints, corrections, and accepted direction" in prompt
    assert "Tool-verified facts: files changed, tests run, errors observed, service responses" in prompt
    assert "Assistant summaries only when backed by user or tool evidence" in prompt
    assert "Do not preserve tentative proposals, rejected paths, or brainstorming as durable facts" in prompt


def test_agent_session_document_submit_api_records_generated_source(tmp_path):
    from meminception.server.admin_api import create_admin_app

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
                    "workspace": "/workspace/mem-inception",
                    "repo": "mem-inception",
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
        assert body["source_id"] == "src-agent-sessions"
        assert body["sync_started"] is False
        assert body["receipt"]["client"] == "codex"
        assert Path(body["document_uri"]).exists()
    finally:
        asyncio.run(database.close())


def test_agent_session_window_api_generates_package_and_discards_raw_window(tmp_path):
    from meminception.server.admin_api import create_admin_app

    class FakeWindowClient:
        async def generate_agent_session_package(self, prompt: str, **kwargs):
            assert "Canonical evidence" in prompt
            assert "api_key: [REDACTED]" in prompt
            assert "token: secret-value" not in prompt
            assert "raw-api-secret" not in prompt
            assert "history-secret" not in prompt
            return SimpleNamespace(
                result="package_created",
                title="Agent Session: useful implementation window",
                summary_markdown=(
                    "## Tool-Verified Implementation Facts\n"
                    "- The agent updated the window upload endpoint.\n\n"
                    "## Verification Evidence\n"
                    "- Unit tests covered the generated package path."
                ),
                reason=None,
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
                    "workspace": "/workspace/mem-inception",
                    "repo": "mem-inception",
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
        assert body["result"] == "package_created"
        assert body["source_id"] == "src-agent-sessions"
        assert body["sync_started"] is False
        assert body["window_hash"].startswith("sha256:")

        package = json.loads(Path(body["document_uri"]).read_text(encoding="utf-8"))
        assert "window upload endpoint" in package["markdown"]
        assert "secret-value" not in package["markdown"]
        assert package["receipt"]["source_kind"] == "generated_agent_window_summary"
        assert package["receipt"]["metadata"]["window_retention"] == "none"
        assert package["receipt"]["metadata"]["window_hash"] == body["window_hash"]
    finally:
        asyncio.run(database.close())


def test_agent_session_window_api_canonicalizes_evidence_before_packaging(tmp_path):
    from meminception.server.admin_api import create_admin_app

    class FakeWindowClient:
        async def generate_agent_session_package(self, prompt: str, **kwargs):
            assert "Canonical evidence" in prompt
            assert "apply_patch" in prompt
            assert "Edited src/meminception/hook_adapter.py" in prompt
            assert "service-json-secret" not in prompt
            assert "session_meta" not in prompt
            assert "private developer bootstrap" not in prompt
            assert "raw JSONL prefix" not in prompt
            return SimpleNamespace(
                result="package_created",
                title="Agent Session: canonical evidence",
                summary_markdown="## Tool-Verified Implementation Facts\n- Canonical evidence was packaged.",
                reason=None,
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
                    "workspace": "/workspace/mem-inception",
                    "events": [
                        {
                            "type": "session_meta",
                            "preview": "private developer bootstrap",
                        },
                        {
                            "kind": "tool_call",
                            "actor": "assistant",
                            "name": "apply_patch",
                            "text": "Edited src/meminception/hook_adapter.py",
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
        assert response.json()["result"] == "package_created"
    finally:
        asyncio.run(database.close())


def test_agent_session_window_api_queues_service_owned_sync(tmp_path):
    from meminception.server.admin_api import create_admin_app

    class FakeWindowClient:
        async def generate_agent_session_package(self, prompt: str, **kwargs):
            return SimpleNamespace(
                result="package_created",
                title="Agent Session: queued service sync",
                summary_markdown="## Tool-Verified Implementation Facts\n- Service queued sync.",
                reason=None,
            )

    class FakeSyncService:
        def __init__(self):
            self.queued: list[str] = []

        def request_source_sync(self, source_id: str) -> bool:
            self.queued.append(source_id)
            return True

        def start_source(self, source_id: str, *, force_full_sync: bool = False):
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
                    "workspace": "/workspace/mem-inception",
                    "events": [{"role": "tool", "name": "apply_patch", "summary": "Edited code."}],
                    "retention": "none",
                    "process_now": False,
                },
            )

        assert response.status_code == 200
        body = response.json()
        assert body["result"] == "package_created"
        assert body["sync_started"] is False
        assert body["sync_queued"] is True
        assert fake_sync.queued == ["src-agent-sessions"]
    finally:
        asyncio.run(database.close())


def test_agent_session_window_api_rejects_unknown_schema_version(tmp_path):
    from meminception.server.admin_api import create_admin_app

    class UnexpectedWindowClient:
        async def generate_agent_session_package(self, prompt: str, **kwargs):
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
                    "workspace": "/workspace/mem-inception",
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
    from meminception.server.admin_api import create_admin_app

    class NoOutputClient:
        async def generate_agent_session_package(self, prompt: str, **kwargs):
            return SimpleNamespace(
                result="no_output",
                title=None,
                summary_markdown="",
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
                    "workspace": "/workspace/mem-inception",
                    "events": [{"role": "assistant", "text": "Sure, that function formats text."}],
                    "process_now": False,
                },
            )

        assert response.status_code == 200
        body = response.json()
        assert body["result"] == "no_output"
        assert body["reason"] == "trivial explanation"

        async def _assert_no_source():
            assert await database.get_source("src-agent-sessions") is None

        asyncio.run(_assert_no_source())
    finally:
        asyncio.run(database.close())


def test_agent_session_window_api_reports_missing_llm(tmp_path):
    from meminception.server.admin_api import create_admin_app

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
                    "workspace": "/workspace/mem-inception",
                    "events": [{"role": "tool", "name": "apply_patch", "summary": "Edited code."}],
                    "process_now": False,
                },
            )

        assert response.status_code == 503
        assert "LLM unavailable" in response.json()["detail"]
    finally:
        asyncio.run(database.close())


def test_agent_session_window_api_keeps_windows_distinct_and_idempotent(tmp_path):
    """Windows of one session/trigger get distinct, idempotent doc_ids (no overwrite)."""
    from meminception.server.admin_api import create_admin_app

    class EchoWindowClient:
        async def generate_agent_session_package(self, prompt: str, **kwargs):
            return SimpleNamespace(
                result="package_created",
                title="Agent Session Window",
                summary_markdown="## Tool-Verified Implementation Facts\n- Window content recorded.",
                reason=None,
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
            "workspace": "/workspace/mem-inception",
            "repo": "mem-inception",
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

        # Different window content -> different doc_id and a separate file (no overwrite).
        assert first["doc_id"] != second["doc_id"]
        assert first["document_uri"] != second["document_uri"]
        assert Path(first["document_uri"]).exists()
        assert Path(second["document_uri"]).exists()

        # Identical window content -> same doc_id (idempotent retry).
        assert repeat_first["doc_id"] == first["doc_id"]
        assert repeat_first["document_uri"] == first["document_uri"]
    finally:
        asyncio.run(database.close())


def test_agent_session_window_api_same_range_different_content_is_distinct(tmp_path):
    """A reused explicit event range with different content must not overwrite."""
    from meminception.server.admin_api import create_admin_app

    class EchoWindowClient:
        async def generate_agent_session_package(self, prompt: str, **kwargs):
            return SimpleNamespace(
                result="package_created",
                title="Agent Session Window",
                summary_markdown="## Tool-Verified Implementation Facts\n- recorded.",
                reason=None,
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
            "workspace": "/workspace/mem-inception",
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

        # Same [evt-1, evt-9] range, different content -> distinct doc_id, both files kept.
        assert a["doc_id"] != b["doc_id"]
        assert Path(a["document_uri"]).exists()
        assert Path(b["document_uri"]).exists()
        # Identical content -> same doc_id (idempotent).
        assert repeat_a["doc_id"] == a["doc_id"]
    finally:
        asyncio.run(database.close())


def test_agent_session_window_retry_identity_ignores_receipt_and_submission_date(tmp_path):
    """The same range/content retry keeps one doc even if receipt metadata changes."""
    from meminception.server.admin_api import create_admin_app

    class EchoWindowClient:
        async def generate_agent_session_package(self, prompt: str, **kwargs):
            return SimpleNamespace(
                result="package_created",
                title="Agent Session Window",
                summary_markdown="## Tool-Verified Implementation Facts\n- retry recorded.",
                reason=None,
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
            "workspace": "/workspace/mem-inception",
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

        assert retry["doc_id"] == first["doc_id"]
        assert retry["document_uri"] == first["document_uri"]
    finally:
        asyncio.run(database.close())


def test_agent_session_window_api_records_no_output_receipt(tmp_path):
    """A no_output window still leaves a traceable receipt, not silence."""
    from meminception.server.admin_api import create_admin_app

    class NoOutputClient:
        async def generate_agent_session_package(self, prompt: str, **kwargs):
            return SimpleNamespace(
                result="no_output", title=None, summary_markdown="", reason="trivial chat"
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
                    "workspace": "/workspace/mem-inception",
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
    from meminception.server.admin_api import create_admin_app

    class FailingClient:
        async def generate_agent_session_package(self, prompt: str, **kwargs):
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
                    "workspace": "/workspace/mem-inception",
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
    from meminception.server.admin_api import create_admin_app

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
                    "workspace": "/workspace/mem-inception",
                    "repo": "mem-inception",
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
            source = await database.get_source("src-agent-sessions")
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
    source_id="src-agent-sessions",
    source_kind="generated_agent_window_summary",
):
    from meminception.models import AgentSessionReceipt

    return AgentSessionReceipt(
        doc_id=doc_id,
        source_id=source_id,
        client="codex",
        session_id=session_id,
        trigger="Stop",
        workspace="/workspace/mem-inception",
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
        metadata={"outcome": outcome},
    )


def test_summarize_agent_session_outcomes_counts_and_fraction(tmp_path):
    database = Database(str(tmp_path / "api.db"))
    import asyncio

    async def _run():
        await database.connect()
        try:
            outcomes = ["package_created", "package_created", "no_output", "failed"]
            for index, outcome in enumerate(outcomes):
                await database.upsert_agent_session_receipt(
                    _make_receipt(doc_id=f"doc-{index}", session_id="sess-sum", outcome=outcome)
                )
            summary = await database.summarize_agent_session_outcomes(session_id="sess-sum")
            assert summary["session_id"] == "sess-sum"
            assert summary["total"] == 4
            assert summary["processed_total"] == 3
            assert summary["counts"] == {"package_created": 2, "no_output": 1, "failed": 1}
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
            assert summary["counts"] == {"package_created": 0, "no_output": 0, "failed": 0}
            assert summary["no_output_fraction"] == 0.0
        finally:
            await database.close()

    asyncio.run(_run())


def test_agent_session_completeness_endpoint(tmp_path):
    from meminception.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = Database(str(tmp_path / "api.db"))
    import asyncio

    asyncio.run(database.connect())
    try:
        async def _seed():
            for index, outcome in enumerate(["package_created", "no_output"]):
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
        assert body["counts"] == {"package_created": 1, "no_output": 1, "failed": 0}
        assert body["no_output_fraction"] == 0.5
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
                    outcome="package_created",
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
            assert summary["counts"] == {"package_created": 1, "no_output": 0, "failed": 0}
            assert summary["no_output_fraction"] == 0.0
        finally:
            await database.close()

    asyncio.run(_run())


def test_package_write_is_atomic_no_temp_left(tmp_path):
    from meminception.server.admin_api import create_admin_app

    class FakeWindowClient:
        async def generate_agent_session_package(self, prompt: str, **kwargs):
            return SimpleNamespace(
                result="package_created",
                title="Agent Session: atomic write",
                summary_markdown=(
                    "## Tool-Verified Implementation Facts\n"
                    "- The package was written atomically."
                ),
                reason=None,
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
                    "workspace": "/workspace/mem-inception",
                    "events": [{"role": "tool", "name": "apply_patch", "summary": "edit"}],
                    "transcript_markdown": "did some real work worth keeping",
                },
            )

        assert response.status_code == 200
        package_path = Path(response.json()["document_uri"])
        assert package_path.exists()
        json.loads(package_path.read_text(encoding="utf-8"))  # complete, valid JSON
        assert list(package_path.parent.glob("*.tmp")) == []  # no leftover temp file
    finally:
        asyncio.run(database.close())
