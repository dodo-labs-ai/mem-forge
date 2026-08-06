from __future__ import annotations

from memforge import plugin_mcp_proxy as proxy


def test_mcp_exposes_workspace_directory_and_optional_workspace_selector() -> None:
    tools = {tool["name"]: tool for tool in proxy.TOOLS}

    assert "list_workspaces" in tools
    assert tools["list_workspaces"]["inputSchema"]["properties"] == {}
    for name, tool in tools.items():
        if name == "list_workspaces":
            continue
        workspace = tool["inputSchema"]["properties"]["workspace_id"]
        assert workspace["type"] == "string"
        assert "workspace_id" not in tool["inputSchema"].get("required", [])


def test_mcp_passes_explicit_workspace_to_the_http_request(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(proxy, "configured_target", lambda *args: object())

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
