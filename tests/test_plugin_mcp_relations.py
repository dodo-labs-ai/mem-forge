from __future__ import annotations

import json
from pathlib import Path

from memforge import plugin_mcp_proxy as proxy
from memforge.plugin_mcp_proxy import _compact_memory_response, _compact_search_result

ROOT = Path(__file__).resolve().parents[1]


def _relation() -> dict[str, object]:
    return {
        "label": "updates",
        "role": "older",
        "counterpart": {
            "memory_id": "mem-newer",
            "summary": "Payroll closes on the 25th.",
            "content_hash": "hash-newer",
            "sources": [{"source_id": "src-jira", "source_type": "jira", "name": "Payroll Jira"}],
            "revision_at": "2026-08-01T00:00:00+00:00",
        },
        "reason": "The later ticket moves the closing day.",
        "decided_by": "classifier",
    }


def test_search_compaction_keeps_relations_and_notice() -> None:
    result = _compact_search_result(
        {
            "memory_id": "mem-older",
            "summary": "Payroll closes on the 20th.",
            "relation_notice": "Updated by the newer Memory mem-newer from Payroll Jira (revised 2026-08-01).",
            "relations": [_relation()],
            "corroborated_by": 2,
        }
    )

    assert result == {
        "memory_id": "mem-older",
        "summary": "Payroll closes on the 20th.",
        "relation_notice": "Updated by the newer Memory mem-newer from Payroll Jira (revised 2026-08-01).",
        "relations": [_relation()],
    }


def test_memory_detail_compaction_keeps_relations_and_notice() -> None:
    result = _compact_memory_response(
        {
            "id": "mem-older",
            "content_hash": "hash-older",
            "relation_notice": "Updated by the newer Memory mem-newer.",
            "relations": [_relation()],
            "dismissed_relations": [_dismissed()],
        }
    )

    assert result["relations"] == [_relation()]
    assert result["relation_notice"] == "Updated by the newer Memory mem-newer."
    # An agent sees what restore_memory_relation would undo.
    assert result["dismissed_relations"] == [_dismissed()]


def _dismissed() -> dict[str, object]:
    return {
        "counterpart": {"memory_id": "mem-other", "summary": "Payroll closes on the 25th.", "content_hash": "h"},
        "labels": ["contradicts"],
        "dismissed_by": "reader@example.test",
        "dismissed_at": "2026-09-01T00:00:00+00:00",
        "note": None,
    }


def _capture_requests(monkeypatch, tmp_path) -> list[dict[str, object]]:
    captured: list[dict[str, object]] = []

    class FakeResponse:
        headers = {"content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return b"{}"

    class FakeOpener:
        def open(self, request, timeout):
            captured.append(
                {
                    "method": request.get_method(),
                    "url": request.full_url,
                    "body": json.loads(request.data.decode()) if request.data else None,
                }
            )
            return FakeResponse()

    # The edition is explicit so the proxy performs no capability discovery.
    monkeypatch.setenv("MEMFORGE_EDITION", "cloud")
    monkeypatch.setenv("MEMFORGE_WORKSPACE_BINDINGS_FILE", str(tmp_path / "bindings.json"))
    monkeypatch.setenv("MEMFORGE_API_URL", "https://memforge.example.hana.ondemand.com")
    monkeypatch.setenv("MEMFORGE_API_TOKEN", "token-123")
    monkeypatch.setenv("MEMFORGE_WORKSPACE_ID", "mount_tai")
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())
    return captured


def test_dismiss_and_restore_forward_to_the_relation_dismissal_route(monkeypatch, tmp_path) -> None:
    captured = _capture_requests(monkeypatch, tmp_path)

    proxy._call_tool(
        "dismiss_memory_relation",
        {
            "memory_id": "mem-older",
            "counterpart_memory_id": "mem-newer",
            "label": "updates",
            "expected_content_hash": "hash-older",
            "counterpart_expected_content_hash": "hash-newer",
            "note": "Different payroll areas.",
        },
    )
    proxy._call_tool("restore_memory_relation", {"memory_id": "mem-older", "counterpart_memory_id": "mem-newer"})

    route = "https://memforge.example.hana.ondemand.com/api/v1/memories/mem-older/relations/mem-newer/dismissal"
    assert captured == [
        {
            "method": "POST",
            "url": route,
            "body": {
                "label": "updates",
                "expected_content_hash": "hash-older",
                "counterpart_expected_content_hash": "hash-newer",
                "note": "Different payroll areas.",
            },
        },
        {"method": "DELETE", "url": route, "body": None},
    ]


def test_dismiss_requires_both_content_hashes() -> None:
    result = proxy._call_tool(
        "dismiss_memory_relation",
        {"memory_id": "mem-older", "counterpart_memory_id": "mem-newer", "label": "updates"},
    )

    assert result == {"error": "expected_content_hash is required"}


def test_packaged_mcp_proxies_match_the_canonical_source() -> None:
    canonical = (ROOT / "src" / "memforge" / "plugin_mcp_proxy.py").read_text()

    for client in ("codex", "claude-code"):
        packaged = ROOT / "integrations" / client / "memforge-memory" / "scripts" / "memforge_mcp.py"
        assert packaged.read_text() == canonical
