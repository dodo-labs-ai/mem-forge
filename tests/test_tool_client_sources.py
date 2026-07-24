"""Contract tests for the source management methods on ToolClient.

These verify the HTTP method, path, and body the client sends, since those are
the parts most likely to drift from the admin API. The transport itself
(`_http_json`) is exercised by the agent-tool read paths elsewhere.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any
from urllib.error import URLError

import pytest

import memforge.tool_client as tool_client
from memforge.api_target import build_host_target, build_target
from memforge.tool_client import ToolClient


class _RecordingClient(ToolClient):
    """ToolClient that captures the _http_json call instead of making a request."""

    def __init__(self, response: dict[str, Any]) -> None:
        super().__init__(
            target=build_target(
                origin="https://self.example.test",
                workspace_id=None,
            ),
            api_token="tok",
        )
        self._response = response
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def _http_json(self, method, url, body):  # type: ignore[override]
        self.calls.append((method, url.removeprefix(self.target.origin), body))
        return self._response


def test_create_source_posts_type_name_and_config():
    client = _RecordingClient({"id": "src-abcd1234", "name": "Notes", "type": "local_markdown"})

    result = client.create_source(
        source_type="local_markdown",
        name="Notes",
        config={"vault_id": "engineering", "display_label": "Engineering notes"},
    )

    assert result["id"] == "src-abcd1234"
    assert client.calls == [
        (
            "POST",
            "/api/sources",
            {
                "type": "local_markdown",
                "name": "Notes",
                "config": {"vault_id": "engineering", "display_label": "Engineering notes"},
            },
        )
    ]


def test_list_sources_gets_sources_collection():
    client = _RecordingClient({"data": [{"id": "src-1", "type": "local_markdown"}]})

    result = client.list_sources()

    assert result["data"][0]["id"] == "src-1"
    assert client.calls == [("GET", "/api/sources", None)]


def test_get_source_schedule_uses_source_schedule_endpoint():
    client = _RecordingClient({"enabled": True, "interval_minutes": 60})

    result = client.get_source_schedule("src-1")

    assert result["enabled"] is True
    assert client.calls == [("GET", "/api/sources/src-1/schedule", None)]


def test_update_source_schedule_puts_source_schedule_payload():
    client = _RecordingClient({"ok": True})

    result = client.update_source_schedule(
        source_id="src-1",
        enabled=True,
        interval_minutes=60,
    )

    assert result["ok"] is True
    assert client.calls == [
        (
            "PUT",
            "/api/sources/src-1/schedule",
            {"enabled": True, "interval_minutes": 60},
        )
    ]


def test_push_github_repo_document_posts_adapter_payload():
    client = _RecordingClient({"package_id": "github_repo_document:hash"})

    result = client.push_github_repo_document(
        source_id="src-github",
        repo_url="https://github.tools.sap/org/repo",
        repo_ref="main",
        relative_path="docs/design.md",
        markdown_body="# Design",
        title="Design",
        raw_hash="raw-1",
        blob_sha="blob-1",
        sync_snapshot_id="laj-sync-1",
        submitted_by="codex",
        submitted_at="2026-07-07T08:00:00Z",
    )

    assert result["package_id"] == "github_repo_document:hash"
    assert client.calls == [
        (
            "POST",
            "/api/sources/src-github/adapter/packages",
            {
                "repo_url": "https://github.tools.sap/org/repo",
                "repo_ref": "main",
                "relative_path": "docs/design.md",
                "markdown_body": "# Design",
                "content_type": "text/markdown",
                "title": "Design",
                "raw_hash": "raw-1",
                "blob_sha": "blob-1",
                "sync_snapshot_id": "laj-sync-1",
                "submitted_by": "codex",
                "submitted_at": "2026-07-07T08:00:00Z",
            },
        )
    ]


def test_prepare_local_source_snapshot_posts_complete_fenced_manifest():
    client = _RecordingClient({"required_doc_ids": []})

    result = client.prepare_local_source_snapshot(
        source_id="src-github",
        items=[{"doc_id": "doc-1", "revision": "blob-1", "change_kind": "upsert"}],
        coverage="complete_snapshot",
        sync_snapshot_id="job-1:attempt:2",
        local_agent_job_id="job-1",
        local_agent_attempt_count=2,
    )

    assert result == {"required_doc_ids": []}
    assert client.calls == [
        (
            "POST",
            "/api/sources/src-github/adapter/manifest",
            {
                "items": [{"doc_id": "doc-1", "revision": "blob-1", "change_kind": "upsert"}],
                "coverage": "complete_snapshot",
                "sync_snapshot_id": "job-1:attempt:2",
                "local_agent_job_id": "job-1",
                "local_agent_attempt_count": 2,
            },
        )
    ]


def test_push_jira_package_posts_raw_payload_without_markdown_body():
    client = _RecordingClient({"doc_id": "jira-doc"})

    result = client.push_jira_package(
        source_id="src-jira",
        base_url="https://jira.example.test",
        issue_key="PAY-1",
        source_url="https://jira.example.test/browse/PAY-1",
        raw_payload={"key": "PAY-1", "fields": {"summary": "Payroll task"}},
        title="Payroll task",
        raw_hash="raw-1",
        provider_revision="2026-07-07T07:59:00Z",
        sync_snapshot_id="laj-sync-2",
        submitted_by="codex",
        submitted_at="2026-07-07T08:00:00Z",
    )

    assert result["doc_id"] == "jira-doc"
    assert client.calls == [
        (
            "POST",
            "/api/sources/src-jira/adapter/packages",
            {
                "base_url": "https://jira.example.test",
                "issue_key": "PAY-1",
                "source_url": "https://jira.example.test/browse/PAY-1",
                "raw_payload": {"key": "PAY-1", "fields": {"summary": "Payroll task"}},
                "title": "Payroll task",
                "raw_hash": "raw-1",
                "provider_revision": "2026-07-07T07:59:00Z",
                "sync_snapshot_id": "laj-sync-2",
                "submitted_by": "codex",
                "submitted_at": "2026-07-07T08:00:00Z",
            },
        )
    ]


def test_push_teams_window_package_posts_raw_payload_without_markdown_body():
    client = _RecordingClient({"doc_id": "teams-doc"})

    result = client.push_teams_window_package(
        source_id="src-teams",
        conversation_id="19:channel@example.test",
        window_id="teams-thread:src-teams:conv:root",
        revision_hash="rev-1",
        raw_payload={"messages": [{"id": "root-1", "content": "Decision captured."}]},
        title="Teams decision",
        root_message_id="root-1",
        window_type="thread",
        source_url="teams-window://src-teams/conv/window/rev-1",
        raw_hash="raw-1",
        sync_snapshot_id="laj-teams:attempt:1",
        submitted_by="codex",
        submitted_at="2026-07-08T08:00:00Z",
    )

    assert result["doc_id"] == "teams-doc"
    assert client.calls == [
        (
            "POST",
            "/api/sources/src-teams/adapter/packages",
            {
                "conversation_id": "19:channel@example.test",
                "window_id": "teams-thread:src-teams:conv:root",
                "revision_hash": "rev-1",
                "raw_payload": {"messages": [{"id": "root-1", "content": "Decision captured."}]},
                "title": "Teams decision",
                "root_message_id": "root-1",
                "window_type": "thread",
                "source_url": "teams-window://src-teams/conv/window/rev-1",
                "raw_hash": "raw-1",
                "sync_snapshot_id": "laj-teams:attempt:1",
                "submitted_by": "codex",
                "submitted_at": "2026-07-08T08:00:00Z",
            },
        )
    ]


def test_start_source_sync_posts_source_sync_payload():
    client = _RecordingClient({"ok": True})

    result = client.start_source_sync("src-jira")

    assert result["ok"] is True
    assert client.calls == [
        (
            "POST",
            "/api/sources/src-jira/sync",
            {"force_full_sync": False},
        )
    ]


def test_start_source_processing_posts_snapshot_identity():
    client = _RecordingClient({"ok": True})

    result = client.start_source_processing(
        source_id="src-local",
        force_full_sync=True,
        sync_snapshot_id="laj-sync-3",
        local_agent_job_id="laj-sync",
        local_agent_attempt_count=3,
    )

    assert result["ok"] is True
    assert client.calls == [
        (
            "POST",
            "/api/sources/src-local/process",
            {
                "force_full_sync": True,
                "sync_snapshot_id": "laj-sync-3",
                "local_agent_job_id": "laj-sync",
                "local_agent_attempt_count": 3,
            },
        )
    ]


def test_start_source_processing_resolves_the_workspace_resource_url(monkeypatch):
    client = ToolClient(
        target=build_target(
            origin="https://memforge-dev.cfapps.eu12.hana.ondemand.com",
            workspace_id="mount_tai",
        ),
        api_token="tok",
    )
    calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def record_http_json(method, url, body):
        calls.append((method, url, body))
        return {"ok": True}

    monkeypatch.setattr(client, "_http_json", record_http_json)

    result = client.start_source_processing(source_id="src-local")

    assert result == {"ok": True}
    assert calls == [
        (
            "POST",
            "https://memforge-dev.cfapps.eu12.hana.ondemand.com/api/workspaces/mount_tai/api/sources/src-local/process",
            {"force_full_sync": False},
        )
    ]


def test_get_source_projection_inventory_uses_source_resource_url():
    client = _RecordingClient({"units": []})

    result = client.get_source_projection_inventory("src teams")

    assert result == {"units": []}
    assert client.calls == [
        (
            "GET",
            "/api/sources/src%20teams/projection-inventory?limit=200",
            None,
        )
    ]


def test_local_agent_job_methods_use_cloud_local_agent_contract():
    client = _RecordingClient({"jobs": []})

    lease = client.lease_local_agent_jobs(limit=3, lease_seconds=120, wait_seconds=25)
    heartbeat = client.heartbeat_local_agent_job("laj-1", attempt_count=2, lease_seconds=120)
    complete = client.complete_local_agent_job(
        "laj-1",
        attempt_count=2,
        status="succeeded",
        result={"count": 1},
    )

    assert lease == {"jobs": []}
    assert heartbeat == {"jobs": []}
    assert complete == {"jobs": []}
    assert client.calls == [
        (
            "POST",
            "/api/cloud/local-agent/jobs/lease",
            {"limit": 3, "lease_seconds": 120, "wait_seconds": 25},
        ),
        (
            "POST",
            "/api/cloud/local-agent/jobs/laj-1/heartbeat",
            {"attempt_count": 2, "lease_seconds": 120},
        ),
        (
            "POST",
            "/api/cloud/local-agent/jobs/laj-1/complete",
            {"status": "succeeded", "attempt_count": 2, "result": {"count": 1}},
        ),
    ]


def test_local_agent_job_lease_default_matches_ui_sync_wait_window():
    client = _RecordingClient({"jobs": []})

    client.lease_local_agent_jobs()

    assert client.calls == [
        (
            "POST",
            "/api/cloud/local-agent/jobs/lease",
            {"limit": 5, "lease_seconds": 60, "wait_seconds": 0},
        )
    ]


def test_local_agent_job_heartbeat_sends_user_progress():
    client = _RecordingClient({"ok": True})

    result = client.heartbeat_local_agent_job(
        "laj-1",
        attempt_count=2,
        lease_seconds=120,
        progress={
            "schema_version": 1,
            "phase": "uploading",
            "progress": {"completed": 7, "total": 16, "unit": "message"},
        },
    )

    assert result == {"ok": True}
    assert client.calls == [
        (
            "POST",
            "/api/cloud/local-agent/jobs/laj-1/heartbeat",
            {
                "attempt_count": 2,
                "lease_seconds": 120,
                "progress": {
                    "schema_version": 1,
                    "phase": "uploading",
                    "progress": {"completed": 7, "total": 16, "unit": "message"},
                },
            },
        )
    ]


def test_tool_client_uses_target_for_workspace_resource_calls():
    client = ToolClient(
        target=build_target(
            origin="https://memforge.example.hana.ondemand.com",
            workspace_id="ws-a",
        ),
        api_token="tok",
    )

    assert (
        client._resource_url("/sources/src-github/adapter/packages")
        == "https://memforge.example.hana.ondemand.com/api/workspaces/ws-a/api/sources/src-github/adapter/packages"
    )


def test_tool_client_can_scope_server_level_client_to_job_workspace():
    server_client = ToolClient(
        target=build_host_target(origin="https://memforge.example.hana.ondemand.com"),
        api_token="token",
    )

    scoped = server_client.for_workspace("mount_tai")

    assert scoped.target.workspace_id == "mount_tai"
    assert scoped.target.workspace_api_base == "https://memforge.example.hana.ondemand.com/api/workspaces/mount_tai/api"
    assert scoped.api_token == "token"


def test_control_plane_daemon_jobs_use_host_origin_not_workspace_resource():
    client = ToolClient(
        target=build_target(
            origin="https://memforge.example.hana.ondemand.com",
            workspace_id="ws-a",
        ),
        api_token="tok",
    )

    assert (
        client._host_url("/api/cloud/local-agent/jobs/lease")
        == "https://memforge.example.hana.ondemand.com/api/cloud/local-agent/jobs/lease"
    )
    with pytest.raises(ValueError, match="host_path_must_start_with_api"):
        client._host_url("/cloud/local-agent/jobs/lease")
    with pytest.raises(ValueError, match="resource_path_must_be_relative_to_api_base"):
        client._resource_url("/api/cloud/local-agent/jobs/lease")


def test_control_plane_lease_unavailable_reports_attempted_host_url(monkeypatch):
    class UnavailableOpener:
        def open(self, request, timeout):
            raise URLError("host unavailable")

    monkeypatch.setattr(tool_client, "build_opener", lambda *_handlers: UnavailableOpener())
    client = ToolClient(
        target=build_target(
            origin="https://memforge.example.hana.ondemand.com",
            workspace_id="ws-a",
        ),
        api_token="tok",
    )

    result = client.lease_local_agent_jobs()

    assert result["error"] == "MemForge API unavailable"
    assert result["api_url"] == "https://memforge.example.hana.ondemand.com/api/cloud/local-agent/jobs/lease"


def test_tool_client_forwards_search_to_hosted_workspace(monkeypatch):
    captured = {}

    class FakeResponse:
        headers = {"content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return b'{"results":[]}'

    class FakeOpener:
        def open(self, request, timeout):
            captured["url"] = request.full_url
            captured["authorization"] = request.get_header("Authorization")
            captured["body"] = request.data
            return FakeResponse()

    monkeypatch.setattr(tool_client, "build_opener", lambda *_handlers: FakeOpener())
    client = ToolClient(
        target=build_target(
            origin="https://memforge.example.hana.ondemand.com",
            workspace_id="mount_tai",
        ),
        api_token="tok",
    )

    result = client.search(query="artifact cache")

    assert result == {"results": []}
    assert captured["url"] == "https://memforge.example.hana.ondemand.com/api/workspaces/mount_tai/api/memories/search"
    assert captured["authorization"] == "Bearer tok"
    assert json.loads(captured["body"].decode())["query"] == "artifact cache"


def test_tool_client_forwards_structured_memory_search_facets():
    client = _RecordingClient({"results": []})

    client.search(
        query="scheduler fixes last week",
        source_filter={
            "source_ids": ["src-codex-session"],
            "clients": ["codex"],
            "repo_identifiers": ["github.tools.sap/hcm/memforge-cloud"],
        },
        include_private=True,
        active_repo_identifier="github.tools.sap/hcm/memforge-cloud",
        status="active",
    )

    assert client.calls == [
        (
            "POST",
            "/api/memories/search",
            {
                "query": "scheduler fixes last week",
                "top_k": 10,
                "include_superseded": False,
                "source_filter": {
                    "source_ids": ["src-codex-session"],
                    "clients": ["codex"],
                    "repo_identifiers": ["github.tools.sap/hcm/memforge-cloud"],
                },
                "include_private": True,
                "active_repo_identifier": "github.tools.sap/hcm/memforge-cloud",
                "status": "active",
            },
        )
    ]


def test_tool_client_get_memory_uses_personalized_detail_route():
    client = _RecordingClient({"id": "mem-private"})

    result = client.get_memory("mem-private")

    assert result["id"] == "mem-private"
    assert client.calls == [
        ("GET", "/api/memories/mem-private?include_private=true", None),
    ]


def test_tool_client_create_memory_posts_user_memory_payload():
    client = _RecordingClient({"memory_id": "mem-new", "status": "inserted"})

    result = client.create_memory(
        content="Use readable confirmation previews before memory mutations.",
        provenance="User asked to remember this after reviewing the MemForge MCP UX.",
        memory_type="convention",
        client="codex",
        repo_identifier="github.com/shno-labs/mem-forge",
    )

    assert result["memory_id"] == "mem-new"
    assert client.calls == [
        (
            "POST",
            "/api/memories/create",
            {
                "content": "Use readable confirmation previews before memory mutations.",
                "provenance": "User asked to remember this after reviewing the MemForge MCP UX.",
                "memory_type": "convention",
                "client": "codex",
                "repo_identifier": "github.com/shno-labs/mem-forge",
            },
        )
    ]


def test_tool_client_retire_memory_posts_lifecycle_guard():
    client = _RecordingClient({"memory_id": "mem-1", "status": "retired"})

    result = client.retire_memory(
        "mem-1",
        reason="User says this is stale",
        expected_content_hash="hash-1",
    )

    assert result["status"] == "retired"
    assert client.calls == [
        (
            "POST",
            "/api/memories/mem-1/retire",
            {"reason": "User says this is stale", "expected_content_hash": "hash-1"},
        )
    ]


def test_tool_client_replace_memory_posts_lifecycle_guard():
    client = _RecordingClient({"memory_id": "mem-1", "replacement_memory_id": "mem-2"})

    result = client.replace_memory(
        "mem-1",
        replacement_content="Corrected memory",
        provenance="User supplied the corrected value in chat.",
        reason="User corrected it",
        expected_content_hash="hash-1",
        replacement_kind="revision",
    )

    assert result["replacement_memory_id"] == "mem-2"
    assert client.calls == [
        (
            "POST",
            "/api/memories/mem-1/replace",
            {
                "replacement_content": "Corrected memory",
                "provenance": "User supplied the corrected value in chat.",
                "reason": "User corrected it",
                "expected_content_hash": "hash-1",
                "replacement_kind": "revision",
            },
        )
    ]


def test_tool_client_memory_review_methods_use_review_endpoints():
    client = _RecordingClient({"ok": True})

    client.list_memory_reviews(status="open", limit=5, offset=2)
    client.get_memory_review("rev-1")
    client.resolve_memory_review("rev-1", decision="reject", note="Not durable", reviewer="alice")

    assert client.calls == [
        ("GET", "/api/memory-reviews?status=open&limit=5&offset=2", None),
        ("GET", "/api/memory-reviews/rev-1", None),
        ("POST", "/api/memory-reviews/rev-1/reject", {"note": "Not durable", "reviewer": "alice"}),
    ]


def test_tool_client_fetches_resource_through_hosted_workspace(monkeypatch):
    captured = {}

    class FakeResponse:
        headers = {"content-type": "text/markdown"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return b"# Source"

    class FakeOpener:
        def open(self, request, timeout):
            captured["url"] = request.full_url
            captured["authorization"] = request.get_header("Authorization")
            return FakeResponse()

    monkeypatch.setattr(tool_client, "build_opener", lambda *_handlers: FakeOpener())
    client = ToolClient(
        target=build_target(
            origin="https://memforge.example.hana.ondemand.com",
            workspace_id="mount_tai",
        ),
        api_token="tok",
    )

    result = client.get_resource(url="/api/documents/doc-1/content")

    assert result["text"] == "# Source"
    assert (
        captured["url"]
        == "https://memforge.example.hana.ondemand.com/api/workspaces/mount_tai/api/documents/doc-1/content"
    )
    assert captured["authorization"] == "Bearer tok"


def test_tool_client_fetches_source_artifact_through_hosted_workspace(monkeypatch):
    captured = {}
    body = b"\x89PNG\r\n\x1a\n"

    class FakeResponse:
        headers = {
            "content-type": "image/png",
            "content-disposition": 'inline; filename="diagram.png"',
            "content-length": str(len(body)),
            "x-content-sha256": "4c4b6a3be1314ab86138bef4314dde022e600960d8689a2c8f8631802d20dab6",
        }

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return body

    class FakeOpener:
        def open(self, request, timeout):
            captured["url"] = request.full_url
            captured["authorization"] = request.get_header("Authorization")
            return FakeResponse()

    monkeypatch.setattr(tool_client, "build_opener", lambda *_handlers: FakeOpener())
    client = ToolClient(
        target=build_target(
            origin="https://memforge.example.hana.ondemand.com",
            workspace_id="mount_tai",
        ),
        api_token="tok",
    )

    result = client.get_resource(
        url="/api/source-artifacts/obsrev-1",
        mode="base64",
    )

    assert result["observation_revision_id"] == "obsrev-1"
    assert "doc_id" not in result
    assert result["content_type"] == "image/png"
    assert result["sha256"] == "4c4b6a3be1314ab86138bef4314dde022e600960d8689a2c8f8631802d20dab6"
    assert (
        captured["url"]
        == "https://memforge.example.hana.ondemand.com"
        "/api/workspaces/mount_tai/api/source-artifacts/obsrev-1"
    )
    assert captured["authorization"] == "Bearer tok"


@pytest.mark.parametrize(
    ("size_delta", "sha256", "expected_error"),
    [
        (0, "actual", None),
        (1, "actual", "resource byte count does not match Content-Length"),
        (0, "wrong", "resource SHA-256 does not match X-Content-SHA256"),
    ],
)
def test_tool_client_file_mode_verifies_resource_integrity(
    monkeypatch,
    tmp_path,
    size_delta: int,
    sha256: str,
    expected_error: str | None,
) -> None:
    body = b"streamed-source-artifact"
    actual_sha256 = hashlib.sha256(body).hexdigest()

    class FakeResponse:
        headers = {
            "content-type": "image/png",
            "content-disposition": 'inline; filename="diagram.png"',
            "content-length": str(len(body) + size_delta),
            "x-content-sha256": actual_sha256 if sha256 == "actual" else "0" * 64,
        }

        def __init__(self):
            self._offset = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, size=-1):
            if size is None or size < 0:
                size = len(body) - self._offset
            chunk = body[self._offset : self._offset + size]
            self._offset += len(chunk)
            return chunk

    class FakeOpener:
        def open(self, request, timeout):
            return FakeResponse()

    monkeypatch.setenv("MEMFORGE_ARTIFACT_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(tool_client, "build_opener", lambda *_handlers: FakeOpener())
    client = ToolClient(
        target=build_target(
            origin="https://memforge.example.hana.ondemand.com",
            workspace_id="mount_tai",
        ),
        api_token="tok",
    )

    result = client.get_resource(
        url="/api/source-artifacts/obsrev-1",
        mode="file",
    )

    if expected_error is None:
        assert result["sha256"] == actual_sha256
        assert result["size_bytes"] == len(body)
    else:
        assert result["error"] == "resource fetch failed"
        assert result["detail"] == expected_error
