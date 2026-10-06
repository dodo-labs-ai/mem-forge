"""Citation identity and complete resource transport at the actual tool boundary."""
from __future__ import annotations

import hashlib
import json

import pytest

from memforge import plugin_mcp_proxy as proxy
from memforge import tool_client
from memforge.api_target import build_target


@pytest.fixture(params=["proxy", "client"])
def resource_reader(request, monkeypatch):
    target = build_target(origin="https://memforge.example.test", edition="cloud")
    if request.param == "proxy":
        return proxy, lambda args: proxy._handle_get_resource(args, target=target, workspace_id="test-space")
    client = tool_client.ToolClient(target=target, workspace_id="test-space", api_token="test-token")
    return tool_client, lambda args: client.get_resource(**args)


@pytest.mark.parametrize("budget,corrupt", [(1000, False), (5, False), (1000, True)])
@pytest.mark.parametrize("selected_ref", [None, "eref-1"])
def test_pinned_json_is_complete_verified_and_workspace_routed(resource_reader, monkeypatch, budget, corrupt, selected_ref):
    module, read = resource_reader
    locator = "/api/v1/memories/mem-1/evidence/eu-1"
    locator += f"/references/{selected_ref}/resource" if selected_ref else "/resource"
    identity_key = "evidence_reference_id" if selected_ref else "evidence_unit_id"
    identity = selected_ref or "eu-1"
    body = json.dumps({"evidence": {"items": [{"excerpt": "原文条件", "evidence_reference_id": "eref-1"}]}}).encode()
    headers = {
        "content-type": "application/json", "content-length": str(len(body)),
        "x-content-sha256": "0" * 64 if corrupt else hashlib.sha256(body).hexdigest(),
    }

    class Response:
        def __init__(self):
            self.headers = headers

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, size=-1):
            return body if size == -1 else body[:size]

    class Opener:
        def open(self, request, timeout):
            assert request.full_url == "https://memforge.example.test" + locator + "?workspace_id=test-space"
            return Response()

    monkeypatch.setattr(module, "build_opener", lambda *args: Opener())
    result = read({"url": locator, "max_chars": budget})
    if corrupt:
        assert result["error"] == "resource fetch failed"
        assert "SHA-256" in result["detail"]
        assert "text" not in result
    elif budget == 5:
        assert result["truncated"] is True
        assert "text" not in result
        assert result[identity_key] == identity
        assert "mode=file" in result["hint"]
    else:
        assert result[identity_key] == identity
        assert result["kind"] == "evidence_unit"
        assert result["truncated"] is False
        assert json.loads(result["text"]) == json.loads(body)


@pytest.mark.parametrize("url", [
    "https://other.example.test/api/v1/memories/mem-1/evidence/eu-1/resource",
    "/api/v1/memories/mem-1/evidence/%2e%2e/resource",
    "/api/v1/memories/mem-1/evidence/eu%2fother/resource",
    "/api/v1/memories/mem-1/evidence/eu-1/resource?untrusted=value",
    "/api/v1/memories/mem-1/evidence/eu-1/resource#fragment",
])
def test_pinned_resource_rejects_untrusted_locator(resource_reader, monkeypatch, url):
    module, read = resource_reader

    def no_request(*args):
        raise AssertionError("Invalid locator must not make a request")

    monkeypatch.setattr(module, "build_opener", no_request)
    assert read({"url": url})["error"] == "unsupported resource URL"


def test_compaction_preserves_every_ref_and_its_pinned_identity():
    items = [{
        "authority": "revision_pinned", "role": "primary" if index == 0 else "required",
        "kind": "text", "support_contribution": True,
        "evidence_reference_id": f"eref-{index}", "observation_id": "obs-1",
        "observation_revision_id": "obsrev-1", "anchor_kind": "revision_range",
        "fragment_id": None, "range_start": index, "range_end": index + 1,
        "raw_content_sha256": f"raw-{index}", "presentation_sha256": f"display-{index}",
        "resource_url": "/api/v1/memories/mem-1/evidence/eu-1/resource",
        "excerpt": "Exact readable evidence", "current": True,
        "text_view": {"kind": "paragraph", "version": 1, "origins": []},
    } for index in range(14)]
    unit = {
        "kind": "evidence_unit", "evidence_unit_id": "eu-1", "support_ids": ["support-1"],
        "source_id": "src-1", "source_type": "github_repo", "source_unit_id": "unit-1",
        "source_unit_revision_id": "unitrev-1", "doc_id": "doc-1", "current": True,
        "resource_url": items[0]["resource_url"], "items": items,
    }
    result = proxy._compact_memory_response({"id": "mem-1", "evidence": [unit]})
    assert result["evidence"] == [unit]


def test_proxy_relative_pinned_resource_keeps_workspace_locator():
    target = build_target(origin="https://memforge.example.test", edition="cloud")
    locator = "/api/v1/memories/mem-1/evidence/eu-1/resource?workspace_id=test-space"
    parsed = proxy._parse_resource_url(locator, target)
    assert parsed.relative_url == locator
    assert parsed.request_url == "https://memforge.example.test" + locator
    assert proxy._parse_resource_url(locator, target, workspace_id="another-space") is None
