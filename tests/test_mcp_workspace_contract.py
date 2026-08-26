from __future__ import annotations

import json

from memforge import plugin_mcp_proxy as proxy


def test_mcp_exposes_workspace_directory_and_optional_workspace_selector() -> None:
    tools = {tool["name"]: tool for tool in proxy.TOOLS}

    assert len(tools) == 14
    assert "list_workspaces" in tools
    assert "set_default_workspace" not in tools
    assert "validate_memory_review_decisions" in tools
    assert "apply_memory_review_decisions" in tools
    assert "refresh_memory_review" not in tools
    assert tools["list_workspaces"]["inputSchema"]["properties"] == {}
    for name, tool in tools.items():
        if name == "list_workspaces":
            continue
        workspace = tool["inputSchema"]["properties"]["workspace_id"]
        assert workspace["type"] == "string"
        assert "workspace_id" not in tool["inputSchema"].get("required", [])


def test_mcp_passes_explicit_workspace_to_the_http_request(monkeypatch) -> None:
    captured: dict[str, object] = {}

    target = type("Target", (), {"origin": "https://cloud.example"})()
    monkeypatch.setattr(proxy, "configured_target", lambda *args: target)

    def fake_http_json(method, path, body, *, target=None, workspace_id=None):
        captured.update(
            method=method,
            path=path,
            body=body,
            target=target,
            workspace_id=workspace_id,
        )
        return {"results": []}

    monkeypatch.setattr(proxy, "_http_json", fake_http_json)

    result = proxy._call_tool(
        "search",
        {"query": "cross-workspace memory", "workspace_id": "ws-payroll"},
    )

    assert result == {"results": []}
    assert captured["workspace_id"] == "ws-payroll"


def test_mcp_resolves_an_ordinary_directory_binding_when_selector_is_omitted(
    monkeypatch,
    tmp_path,
) -> None:
    working_directory = tmp_path / "ordinary"
    working_directory.mkdir()
    bindings = tmp_path / "workspace-bindings.json"
    bindings.write_text(
        json.dumps(
            {
                "version": 1,
                "targets": {
                    "https://cloud.example.hana.ondemand.com": {
                        "directory_bindings": {
                            str(working_directory): "payroll_agent",
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_http_json(method, path, body, *, target=None, workspace_id=None):
        captured.update(workspace_id=workspace_id, body=body)
        return {"results": []}

    monkeypatch.setenv(
        "MEMFORGE_API_URL",
        "https://cloud.example.hana.ondemand.com",
    )
    monkeypatch.setenv("MEMFORGE_EDITION", "cloud")
    monkeypatch.setenv("MEMFORGE_WORKSPACE_BINDINGS_FILE", str(bindings))
    monkeypatch.setattr(proxy, "_http_json", fake_http_json)

    result = proxy._call_tool(
        "search",
        {
            "query": "directory routing",
            "repository_context": {"working_directory": str(working_directory)},
        },
    )

    assert result == {"results": []}
    assert captured["workspace_id"] == "payroll_agent"
    assert "active_repo_identifier" not in captured["body"]


def test_explicit_workspace_overrides_local_directory_binding(
    monkeypatch,
    tmp_path,
) -> None:
    working_directory = tmp_path / "ordinary"
    working_directory.mkdir()
    bindings = tmp_path / "workspace-bindings.json"
    bindings.write_text(
        json.dumps(
            {
                "version": 1,
                "targets": {
                    "https://cloud.example.hana.ondemand.com": {
                        "directory_bindings": {
                            str(working_directory): "bound_workspace",
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_http_json(method, path, body, *, target=None, workspace_id=None):
        captured["workspace_id"] = workspace_id
        return {"results": []}

    monkeypatch.setenv(
        "MEMFORGE_API_URL",
        "https://cloud.example.hana.ondemand.com",
    )
    monkeypatch.setenv("MEMFORGE_EDITION", "cloud")
    monkeypatch.setenv("MEMFORGE_WORKSPACE_BINDINGS_FILE", str(bindings))
    monkeypatch.setattr(proxy, "_http_json", fake_http_json)

    proxy._call_tool(
        "search",
        {
            "query": "cross-workspace",
            "workspace_id": "explicit_workspace",
            "repository_context": {"working_directory": str(working_directory)},
        },
    )

    assert captured["workspace_id"] == "explicit_workspace"


def test_hook_default_is_not_an_interactive_tool_fallback(
    monkeypatch,
    tmp_path,
) -> None:
    working_directory = tmp_path / "ordinary"
    working_directory.mkdir()
    bindings = tmp_path / "workspace-bindings.json"
    bindings.write_text(
        json.dumps(
            {
                "version": 1,
                "targets": {
                    "https://cloud.example.hana.ondemand.com": {
                        "hook_workspace_id": "capture_workspace",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_http_json(method, path, body, *, target=None, workspace_id=None):
        captured["workspace_id"] = workspace_id
        return {"results": []}

    monkeypatch.setenv(
        "MEMFORGE_API_URL",
        "https://cloud.example.hana.ondemand.com",
    )
    monkeypatch.setenv("MEMFORGE_EDITION", "cloud")
    monkeypatch.setenv("MEMFORGE_WORKSPACE_BINDINGS_FILE", str(bindings))
    monkeypatch.setattr(proxy, "_http_json", fake_http_json)

    proxy._call_tool(
        "search",
        {
            "query": "must stay ambiguous",
            "repository_context": {"working_directory": str(working_directory)},
        },
    )

    assert captured["workspace_id"] is None


def test_list_workspaces_is_discovery_not_a_workspace_scoped_call(monkeypatch) -> None:
    captured: dict[str, object] = {}
    target = object()
    monkeypatch.setattr(proxy, "configured_target", lambda *args: target)
    monkeypatch.setattr(
        proxy,
        "_tool_call_context",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("workspace discovery must not resolve repository context")
        ),
    )

    def fake_http_json(method, path, body, *, target=None, workspace_id=None):
        captured.update(
            method=method,
            path=path,
            target=target,
            workspace_id=workspace_id,
        )
        return {"workspaces": []}

    monkeypatch.setattr(proxy, "_http_json", fake_http_json)

    assert proxy._call_tool("list_workspaces", {}) == {"workspaces": []}
    assert captured == {
        "method": "GET",
        "path": "/workspaces",
        "target": target,
        "workspace_id": None,
    }
