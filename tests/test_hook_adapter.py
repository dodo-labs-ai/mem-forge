from __future__ import annotations

import ast
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import io
import json
import os
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest


def _unused_loopback_url() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    return f"http://127.0.0.1:{port}"


@pytest.fixture(autouse=True)
def _isolate_memforge_plugin_config(monkeypatch, tmp_path):
    # Nothing on this machine may answer the plugin: a MemForge server running
    # on the default local port, or a client configured for a real service,
    # would turn every "the service is unreachable" case into a real call.
    # Tests that need a reachable service patch the transport themselves.
    monkeypatch.setenv("MEMFORGE_API_URL", _unused_loopback_url())
    # Remote endpoint behavior is not under test in this module; keep its
    # edition explicit so only capability-discovery tests perform I/O.
    monkeypatch.setenv("MEMFORGE_EDITION", "cloud")
    monkeypatch.delenv("MEMFORGE_API_TOKEN", raising=False)
    monkeypatch.delenv("MEMFORGE_WORKSPACE_ID", raising=False)
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setenv("MEMFORGE_WORKSPACE_BINDINGS_FILE", str(tmp_path / "missing-workspace-bindings.json"))
    monkeypatch.setenv("MEMFORGE_CODEX_CONFIG", str(tmp_path / "missing-codex-config.toml"))
    try:
        from memforge import plugin_config

        monkeypatch.setattr(plugin_config, "_CONFIG_CACHE", None)
        plugin_config._target_for_configuration.cache_clear()
    except Exception:
        pass


def _normalized_plugin_config_ast(source: str) -> str:
    module = ast.parse(source)
    expected_import_switches = [
        ast.parse(
            """
if __package__:
    from .api_target import MemForgeTarget, build_target
    from .capability_discovery import discover_target
else:
    from memforge.api_target import MemForgeTarget, build_target
    from memforge.capability_discovery import discover_target
"""
        ).body[0],
        ast.parse(
            """
if __package__:
    from .memforge_api_target import MemForgeTarget, build_target
    from .memforge_capability_discovery import discover_target
else:
    from memforge_api_target import MemForgeTarget, build_target
    from memforge_capability_discovery import discover_target
"""
        ).body[0],
    ]
    expected_import_switch_asts = {ast.dump(node, include_attributes=False) for node in expected_import_switches}
    normalized_body: list[ast.stmt] = []
    for node in module.body:
        contains_target_import = any(
            isinstance(child, ast.ImportFrom)
            and child.module
            in {
                "api_target",
                "memforge.api_target",
                "memforge_api_target",
                "capability_discovery",
                "memforge.capability_discovery",
                "memforge_capability_discovery",
            }
            for child in ast.walk(node)
        )
        if contains_target_import:
            assert ast.dump(node, include_attributes=False) in expected_import_switch_asts, (
                "target import compatibility block must be exact"
            )
            continue
        normalized_body.append(node)
    module.body = normalized_body
    return ast.dump(module, include_attributes=False)


def _init_git_repo_with_origin(path: Path, origin_url: str) -> None:
    subprocess.run(["git", "init"], cwd=path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(
        ["git", "remote", "add", "origin", origin_url],
        cwd=path,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _forbid_network(monkeypatch) -> None:
    def refuse(*_args, **_kwargs):
        raise AssertionError("the SessionStart hook must not open a network connection")

    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(urllib.request, "urlopen", refuse)


def test_session_start_emits_usage_guidance_without_network_or_local_state(monkeypatch, tmp_path, capsys):
    from memforge import hook_adapter

    monkeypatch.setenv("HOME", str(tmp_path))
    _forbid_network(monkeypatch)
    monkeypatch.setattr(
        hook_adapter.sys,
        "stdin",
        _Stdin(
            {
                "hook_event_name": "SessionStart",
                "session_id": "sess-start",
                "cwd": "/tmp/mem-forge",
                "transcript_path": str(tmp_path / "transcript.jsonl"),
            }
        ),
    )

    exit_code = hook_adapter.main(["context", "--client", "claude-code"])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["continue"] is True
    assert "systemMessage" not in output
    assert output["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    context = output["hookSpecificOutput"]["additionalContext"]
    assert context == hook_adapter.SESSION_START_USAGE_GUIDANCE
    assert "MemForge Usage Guidance" in context
    assert "MCP search" in context
    assert "list_sources" in context
    assert "source_ids" in context
    assert "source_types" not in context
    assert "`clients`" not in context
    assert "get_memory" in context
    assert "get_resource" in context
    assert "confirmed content must be the durable memory only" in context
    assert "why-the-tool-was-called out of content" in context
    assert list(tmp_path.iterdir()) == []


def test_session_start_guidance_explains_agentic_memforge_usage(monkeypatch, capsys):
    from memforge import hook_adapter

    monkeypatch.setattr(hook_adapter.sys, "stdin", _Stdin({"hook_event_name": "SessionStart"}))

    hook_adapter.main(["context"])

    guidance = json.loads(capsys.readouterr().out)["hookSpecificOutput"]["additionalContext"]
    assert "Use it agentically" in guidance
    assert "Treat memory as context, not current truth" in guidance
    assert "Never mutate memory silently" in guidance


@pytest.mark.parametrize(
    "argv",
    [
        ["submit-session", "--client", "claude-code"],
        ["worker-drain", "--timeout", "not-a-number", "--owner-fd"],
        ["--client"],
        [],
    ],
)
def test_unknown_hook_mode_lets_the_client_continue(monkeypatch, capsys, argv):
    from memforge import hook_adapter

    _forbid_network(monkeypatch)
    monkeypatch.setattr(hook_adapter.sys, "stdin", _Stdin({"hook_event_name": "Stop"}))

    exit_code = hook_adapter.main(argv)

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {"continue": True}


def test_unknown_hook_mode_reads_the_whole_hook_input(monkeypatch, capsys):
    from memforge import hook_adapter

    stdin = io.StringIO(json.dumps({"hook_event_name": "Stop", "transcript_path": "/tmp/t.jsonl"}))
    monkeypatch.setattr(hook_adapter.sys, "stdin", stdin)

    assert hook_adapter.main(["submit-session", "--client", "codex"]) == 0
    assert json.loads(capsys.readouterr().out) == {"continue": True}
    assert stdin.read() == ""


@pytest.mark.parametrize("client", ["codex", "claude-code"])
@pytest.mark.parametrize("mode", ["submit-session", "worker-drain"])
def test_plugin_wrapper_accepts_a_large_payload_for_an_old_hook_mode(client, mode):
    # An old hooks.json still sends Stop and PreCompact payloads. The payload is
    # larger than a pipe buffer, so the client's write fails with a broken pipe
    # unless the hook reads it to the end.
    root = Path(__file__).resolve().parents[1]
    wrapper = root / "integrations" / client / "memforge-memory" / "scripts" / "memforge_hook.py"
    payload = json.dumps({"hook_event_name": "Stop", "last_assistant_message": "x" * 1_000_000})

    process = subprocess.Popen(
        [sys.executable, str(wrapper), mode, "--client", client],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        process.stdin.write(payload)
        process.stdin.close()
        stdout = process.stdout.read()
        stderr = process.stderr.read()
        assert process.wait(timeout=5) == 0
    finally:
        process.kill()
        process.wait()

    assert stderr == ""
    assert json.loads(stdout) == {"continue": True}


@pytest.mark.parametrize("event_name", ["Stop", "PreCompact", "SubagentStop", "UserPromptSubmit"])
def test_context_mode_adds_nothing_outside_session_start(monkeypatch, capsys, event_name):
    from memforge import hook_adapter

    _forbid_network(monkeypatch)
    monkeypatch.setattr(hook_adapter.sys, "stdin", _Stdin({"hook_event_name": event_name}))

    assert hook_adapter.main(["context"]) == 0
    assert json.loads(capsys.readouterr().out) == {"continue": True}


@pytest.mark.parametrize("raw_input", ["not json", "[1, 2]", ""])
def test_unreadable_hook_input_lets_the_client_continue(monkeypatch, capsys, raw_input):
    from memforge import hook_adapter

    monkeypatch.setattr(hook_adapter.sys, "stdin", io.StringIO(raw_input))

    assert hook_adapter.main(["context"]) == 0
    assert json.loads(capsys.readouterr().out) == {"continue": True}


def test_session_start_ignores_malformed_memforge_configuration(monkeypatch, tmp_path, capsys):
    from memforge import hook_adapter

    malformed = tmp_path / "codex-config.toml"
    malformed.write_text("[memforge\nMEMFORGE_API_URL = ", encoding="utf-8")
    monkeypatch.setenv("MEMFORGE_CODEX_CONFIG", str(malformed))
    monkeypatch.setenv("MEMFORGE_API_URL", "not a url")
    monkeypatch.setenv("MEMFORGE_HOOK_TIMEOUT_SECONDS", "not-a-number")
    monkeypatch.setattr(hook_adapter.sys, "stdin", _Stdin({"hook_event_name": "SessionStart"}))

    assert hook_adapter.main(["context"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["continue"] is True
    assert output["hookSpecificOutput"]["additionalContext"] == hook_adapter.SESSION_START_USAGE_GUIDANCE


def test_codex_and_claude_plugins_include_hooks_and_adapter_wrappers():
    root = Path(__file__).resolve().parents[1]

    codex_root = root / "integrations" / "codex" / "memforge-memory"
    claude_root = root / "integrations" / "claude-code" / "memforge-memory"

    for plugin_root in (codex_root, claude_root):
        assert (plugin_root / "hooks" / "hooks.json").exists()
        assert (plugin_root / "scripts" / "memforge_hook.py").exists()
        assert (plugin_root / "scripts" / "memforge_hook_adapter.py").exists()
        assert (plugin_root / "scripts" / "memforge_mcp.py").exists()
        assert (plugin_root / "scripts" / "memforge_repo_identity.py").exists()
        assert (plugin_root / "scripts" / "memforge_repository_context.py").exists()
        assert (plugin_root / ".mcp.json").exists()
    assert (codex_root / ".codex-plugin" / "plugin.json").exists()
    assert (claude_root / ".claude-plugin" / "plugin.json").exists()

    codex_manifest = json.loads((codex_root / ".codex-plugin" / "plugin.json").read_text())
    assert codex_manifest["hooks"] == "./hooks/hooks.json"
    claude_manifest = json.loads((claude_root / ".claude-plugin" / "plugin.json").read_text())
    assert codex_manifest["version"] == claude_manifest["version"]

    for plugin_root, client in ((codex_root, "codex"), (claude_root, "claude-code")):
        hooks = json.loads((plugin_root / "hooks" / "hooks.json").read_text())
        assert list(hooks["hooks"]) == ["SessionStart"]
        [entry] = hooks["hooks"]["SessionStart"]
        assert entry["matcher"] == "startup|resume|compact"
        [hook] = entry["hooks"]
        command = hook["command"]
        assert "scripts/memforge_hook.py" in command
        assert command.endswith(f'"$script" context --client {client}\'')
        assert "${PLUGIN_ROOT:-}" in command
        assert "plugins/cache/memforge/memory" not in command
        assert "version=" not in command
        assert " -nt " not in command
    codex_mcp = json.loads((codex_root / ".mcp.json").read_text())
    claude_mcp = json.loads((claude_root / ".mcp.json").read_text())

    codex_memforge = codex_mcp["mcpServers"]["memforge"]
    assert codex_memforge["command"] == "python3"
    assert codex_memforge["args"] == ["scripts/memforge_mcp.py"]
    assert codex_memforge["cwd"] == "."
    assert "env_vars" not in codex_memforge

    claude_memforge = claude_mcp["mcpServers"]["memforge"]
    assert claude_memforge["command"] == "sh"
    assert claude_memforge["args"] == [
        "-lc",
        (
            "root=$CLAUDE_PLUGIN_ROOT; "
            'if [ -z "$root" ]; then root=$PLUGIN_ROOT; fi; '
            'if [ -z "$root" ]; then root=.; fi; '
            'cd "$root" && exec python3 scripts/memforge_mcp.py'
        ),
    ]
    assert "cwd" not in claude_memforge


def test_plugin_manifests_describe_guidance_and_mcp_tools_without_capture():
    root = Path(__file__).resolve().parents[1]
    codex_manifest = json.loads((root / "integrations/codex/memforge-memory/.codex-plugin/plugin.json").read_text())
    claude_manifest = json.loads(
        (root / "integrations/claude-code/memforge-memory/.claude-plugin/plugin.json").read_text()
    )
    marketplace = json.loads((root / ".claude-plugin" / "marketplace.json").read_text())

    texts = [
        codex_manifest["description"],
        codex_manifest["interface"]["shortDescription"],
        codex_manifest["interface"]["longDescription"],
        *codex_manifest["interface"]["defaultPrompt"],
        claude_manifest["description"],
        *(plugin["description"] for plugin in marketplace["plugins"]),
    ]
    for text in texts:
        lowered = text.lower()
        assert "window" not in lowered
        assert "session summary" not in lowered
        assert "agent-session" not in lowered
    assert "hooks" not in codex_manifest["keywords"]
    assert "Hooks" in codex_manifest["interface"]["capabilities"]
    assert "hook" not in marketplace["plugins"][0]["description"].lower()


def test_mcp_data_tools_accept_repository_context_but_workspace_directory_does_not():
    from memforge import plugin_mcp_proxy

    assert plugin_mcp_proxy.TOOLS
    for tool in plugin_mcp_proxy.TOOLS:
        if tool["name"] == "list_workspaces":
            assert tool["inputSchema"]["properties"] == {}
            continue
        assert "repository_context" in tool["inputSchema"]["properties"], tool["name"]


def test_packaged_plugin_version_is_consistent():
    import tomllib

    from packaging.version import Version

    root = Path(__file__).resolve().parents[1]
    # The plugins and their runtime copies spell the version as semver; the
    # package spells the same version as PEP 440.
    version = "0.2.0-beta.2"
    package = tomllib.loads((root / "pyproject.toml").read_text())
    canonical_mcp = (root / "src" / "memforge" / "plugin_mcp_proxy.py").read_text()
    canonical_hook = (root / "src" / "memforge" / "hook_adapter.py").read_text()

    assert Version(package["project"]["version"]) == Version(version)
    assert f'SERVER_VERSION = "{version}"' in canonical_mcp
    assert f'PLUGIN_VERSION = "{version}"' in canonical_hook

    for client, manifest_dir in (
        ("codex", ".codex-plugin"),
        ("claude-code", ".claude-plugin"),
    ):
        plugin_root = root / "integrations" / client / "memforge-memory"
        manifest = json.loads((plugin_root / manifest_dir / "plugin.json").read_text())
        assert manifest["version"] == version
        assert f"version is `{version}`" in (plugin_root / "README.md").read_text()
        assert f'SERVER_VERSION = "{version}"' in (plugin_root / "scripts" / "memforge_mcp.py").read_text()
        assert f'PLUGIN_VERSION = "{version}"' in (plugin_root / "scripts" / "memforge_hook_adapter.py").read_text()


def test_plugin_readmes_key_the_upgrade_note_on_the_earlier_plugin_message():
    root = Path(__file__).resolve().parents[1]
    for client in ("codex", "claude-code"):
        readme = (root / "integrations" / client / "memforge-memory" / "README.md").read_text()
        assert "`MemForge hook skipped: /hooks/receipts returned HTTP 410`" in readme
        assert "create_memory" in readme
        assert "agent-sessions/windows" not in readme
        assert "queue.sqlite" in readme


@pytest.mark.parametrize(
    ("plugin_path", "home_cache", "env_root"),
    [
        (
            "integrations/codex/memforge-memory/hooks/hooks.json",
            ".codex/plugins/cache/memforge/memory",
            "CODEX_PLUGIN_ROOT",
        ),
        (
            "integrations/claude-code/memforge-memory/hooks/hooks.json",
            ".claude/plugins/cache/memforge/memory",
            "CLAUDE_PLUGIN_ROOT",
        ),
    ],
)
def test_plugin_hook_commands_skip_when_registered_root_is_stale(tmp_path, plugin_path, home_cache, env_root):
    root = Path(__file__).resolve().parents[1]
    hooks = json.loads((root / plugin_path).read_text())
    command = hooks["hooks"]["SessionStart"][0]["hooks"][0]["command"]
    marker = tmp_path / "argv.txt"
    cache_script = tmp_path / home_cache / "0.1.20" / "scripts" / "memforge_hook.py"
    cache_script.parent.mkdir(parents=True)
    cache_script.write_text(
        "import os, pathlib, sys\npathlib.Path(os.environ['MEMFORGE_TEST_MARKER']).write_text(' '.join(sys.argv[1:]))\n"
    )
    env = os.environ.copy()
    env["HOME"] = str(tmp_path)
    env[env_root] = str(tmp_path / "deleted-plugin-root")
    env["PLUGIN_ROOT"] = str(tmp_path / "also-missing")
    env["MEMFORGE_TEST_MARKER"] = str(marker)

    result = subprocess.run(
        command,
        shell=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 0
    assert "MemForge hook script not found" in result.stderr
    assert not marker.exists()


def _run_session_start_wrapper(wrapper: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(wrapper), "context"],
        input=json.dumps({"hook_event_name": "SessionStart"}),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=5,
        check=False,
    )


def test_plugin_wrapper_runs_from_repo_checkout_without_pythonpath():
    root = Path(__file__).resolve().parents[1]
    wrapper = root / "integrations" / "codex" / "memforge-memory" / "scripts" / "memforge_hook.py"

    result = _run_session_start_wrapper(wrapper)

    assert result.returncode == 0
    assert result.stderr == ""
    output = json.loads(result.stdout)
    assert output["continue"] is True
    assert "MemForge Usage Guidance" in output["hookSpecificOutput"]["additionalContext"]


def test_plugin_wrapper_runs_from_copied_package_without_pythonpath(tmp_path):
    import shutil

    root = Path(__file__).resolve().parents[1]
    for client in ("codex", "claude-code"):
        source_plugin = root / "integrations" / client / "memforge-memory"
        copied_plugin = tmp_path / client / "memforge-memory"
        shutil.copytree(source_plugin, copied_plugin)

        result = _run_session_start_wrapper(copied_plugin / "scripts" / "memforge_hook.py")

        assert result.returncode == 0
        assert result.stderr == ""
        output = json.loads(result.stdout)
        assert output["continue"] is True
        assert "MemForge Usage Guidance" in output["hookSpecificOutput"]["additionalContext"]


def test_plugin_adapters_match_canonical_adapter():
    root = Path(__file__).resolve().parents[1]
    canonical = (root / "src" / "memforge" / "hook_adapter.py").read_text()

    for adapter in (
        root / "integrations" / "codex" / "memforge-memory" / "scripts" / "memforge_hook_adapter.py",
        root / "integrations" / "claude-code" / "memforge-memory" / "scripts" / "memforge_hook_adapter.py",
    ):
        assert adapter.read_text() == canonical


def test_packaged_repo_identity_matches_canonical_helper():
    root = Path(__file__).resolve().parents[1]
    canonical = (root / "src" / "memforge" / "repo_identity.py").read_text()

    for helper in (
        root / "integrations" / "codex" / "memforge-memory" / "scripts" / "memforge_repo_identity.py",
        root / "integrations" / "claude-code" / "memforge-memory" / "scripts" / "memforge_repo_identity.py",
    ):
        assert helper.read_text() == canonical


def test_packaged_repository_context_matches_canonical_module():
    root = Path(__file__).resolve().parents[1]
    canonical = (root / "src" / "memforge" / "repository_context.py").read_text()

    for helper in (
        root / "integrations" / "codex" / "memforge-memory" / "scripts" / "memforge_repository_context.py",
        root / "integrations" / "claude-code" / "memforge-memory" / "scripts" / "memforge_repository_context.py",
    ):
        assert helper.read_text() == canonical


def test_packaged_plugin_config_matches_canonical_target_and_helpers():
    root = Path(__file__).resolve().parents[1]
    canonical = (root / "src" / "memforge" / "plugin_config.py").read_text()

    for helper in (
        root / "integrations" / "codex" / "memforge-memory" / "scripts" / "memforge_plugin_config.py",
        root / "integrations" / "claude-code" / "memforge-memory" / "scripts" / "memforge_plugin_config.py",
    ):
        assert _normalized_plugin_config_ast(helper.read_text()) == _normalized_plugin_config_ast(canonical)


def test_mcp_resource_url_targets_cloud_api(monkeypatch):
    from memforge import plugin_mcp_proxy

    monkeypatch.setenv("MEMFORGE_API_URL", "https://cloud.example.com")
    monkeypatch.setenv("MEMFORGE_EDITION", "cloud")
    monkeypatch.setenv("MEMFORGE_WORKSPACE_ID", "team_alpha")

    assert plugin_mcp_proxy._resource_url("/sources") == ("https://cloud.example.com/api/v1/sources")


def test_mcp_roots_keep_the_user_target_despite_repository_workspace_override(monkeypatch, tmp_path):
    from memforge import plugin_config, plugin_mcp_proxy

    monkeypatch.delenv("MEMFORGE_API_URL")

    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / ".git").mkdir()
    local_config = repository / ".memforge" / "config.toml"
    local_config.parent.mkdir()
    local_config.write_text('[memforge]\nworkspace_id = "repository_workspace"\n', encoding="utf-8")
    user_config = tmp_path / "codex-config.toml"
    user_config.write_text(
        """
[memforge]
MEMFORGE_API_URL = "https://cloud.example.com"
MEMFORGE_EDITION = "cloud"
MEMFORGE_WORKSPACE_ID = "user_workspace"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("MEMFORGE_CODEX_CONFIG", str(user_config))
    monkeypatch.setenv("MEMFORGE_WORKSPACE_ID", "process_workspace")
    monkeypatch.setattr(plugin_config, "_CONFIG_CACHE", None)
    _provide_mcp_roots(plugin_mcp_proxy, repository)

    assert plugin_mcp_proxy._resource_url("/sources") == ("https://cloud.example.com/api/v1/sources")


def test_mcp_tool_call_waits_for_pending_roots_before_selecting_workspace(monkeypatch, tmp_path):
    proxy = _load_plugin_mcp_proxy()
    repository = tmp_path / "repository"
    repository.mkdir()
    _init_git_repo_with_origin(repository, "https://github.com/example/repository.git")
    local_config = repository / ".memforge" / "config.toml"
    local_config.parent.mkdir()
    local_config.write_text('[memforge]\nworkspace_id = "repository_workspace"\n', encoding="utf-8")
    captured_urls: list[str] = []

    class FakeResponse:
        headers = {"content-type": "application/json"}

        def __init__(self, payload: dict) -> None:
            self.payload = json.dumps(payload).encode()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return self.payload

    class FakeOpener:
        def open(self, request, timeout):
            captured_urls.append(request.full_url)
            if request.full_url.endswith("/sources/searchable"):
                return FakeResponse({"data": []})
            return FakeResponse({"results": []})

    monkeypatch.setenv("MEMFORGE_API_URL", "https://cloud.example.com")
    monkeypatch.setenv("MEMFORGE_API_TOKEN", "token-123")
    monkeypatch.setenv("MEMFORGE_WORKSPACE_ID", "process_workspace")
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())

    initialize = proxy._handle_rpc_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {"roots": {"listChanged": True}},
                "clientInfo": {"name": "test"},
            },
        }
    )
    assert initialize["result"]["serverInfo"]["name"] == "memforge"
    roots_request = proxy._handle_rpc_message({"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert roots_request["method"] == "roots/list"

    deferred = proxy._handle_rpc_message(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "list_sources", "arguments": {}},
        }
    )

    assert deferred is None
    assert captured_urls == []

    resumed = proxy._handle_rpc_message(
        {
            "jsonrpc": "2.0",
            "id": roots_request["id"],
            "result": {"roots": [{"uri": repository.as_uri(), "name": repository.name}]},
        }
    )
    search_response = proxy._handle_rpc_message(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "search",
                "arguments": {
                    "query": "workspace routing",
                    "repository_context": {"working_directory": str(repository)},
                },
            },
        }
    )

    assert isinstance(resumed, list)
    assert [response["id"] for response in resumed] == [2]
    assert "result" in resumed[0]
    assert "result" in search_response
    base = "https://cloud.example.com/api/v1"
    assert captured_urls == [
        f"{base}/sources/searchable",
        f"{base}/memories/search",
    ]


def test_mcp_tool_calls_use_explicit_repository_context_for_workspace_without_roots(monkeypatch, tmp_path):
    proxy = _load_plugin_mcp_proxy()
    repository = tmp_path / "repository"
    repository.mkdir()
    _init_git_repo_with_origin(repository, "https://github.com/example/repository.git")
    local_config = repository / ".memforge" / "config.toml"
    local_config.parent.mkdir()
    local_config.write_text('[memforge]\nworkspace_id = "repository_workspace"\n', encoding="utf-8")
    captured_requests: list[tuple[str, bytes | None]] = []

    class FakeResponse:
        headers = {"content-type": "application/json"}

        def __init__(self, payload: dict) -> None:
            self.payload = json.dumps(payload).encode()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return self.payload

    class FakeOpener:
        def open(self, request, timeout):
            captured_requests.append((request.full_url, request.data))
            if request.full_url.endswith("/sources/searchable"):
                return FakeResponse({"data": []})
            return FakeResponse({"results": []})

    monkeypatch.setenv("MEMFORGE_API_URL", "https://cloud.example.com")
    monkeypatch.setenv("MEMFORGE_API_TOKEN", "token-123")
    monkeypatch.setenv("MEMFORGE_WORKSPACE_ID", "process_workspace")
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())

    initialize = proxy._handle_rpc_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "test"},
            },
        }
    )
    assert initialize["result"]["serverInfo"]["name"] == "memforge"

    repository_context = {"working_directory": str(repository)}
    list_sources_response = proxy._handle_rpc_message(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "list_sources",
                "arguments": {"repository_context": repository_context},
            },
        }
    )
    search_response = proxy._handle_rpc_message(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "search",
                "arguments": {
                    "query": "workspace routing",
                    "repository_context": repository_context,
                },
            },
        }
    )

    assert "result" in list_sources_response
    assert "result" in search_response
    base = "https://cloud.example.com/api/v1"
    assert [url for url, _body in captured_requests] == [
        f"{base}/sources/searchable",
        f"{base}/memories/search",
    ]
    search_body = json.loads(captured_requests[1][1].decode())
    assert search_body["active_repo_identifier"] == "github.com/example/repository"
    assert "repository_context" not in search_body


def test_codex_sandbox_cwd_routes_each_tool_call_to_repository_workspace(monkeypatch, tmp_path):
    proxy = _load_plugin_mcp_proxy()
    repository = tmp_path / "repository"
    repository.mkdir()
    _init_git_repo_with_origin(repository, "https://github.com/example/repository.git")
    local_config = repository / ".memforge" / "config.toml"
    local_config.parent.mkdir()
    local_config.write_text('[memforge]\nworkspace_id = "repository_workspace"\n', encoding="utf-8")
    captured: list[tuple[str, str | None]] = []

    def fake_http_json(method, path, body, *, target=None, workspace_id=None):
        captured.append((target.resource_url(path), (body or {}).get("active_repo_identifier")))
        return {"results": []}

    monkeypatch.setenv("MEMFORGE_API_URL", "https://cloud.example.com")
    monkeypatch.setenv("MEMFORGE_WORKSPACE_ID", "process_workspace")
    monkeypatch.setattr(proxy, "_http_json", fake_http_json)

    initialize = proxy._handle_rpc_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "codex"},
            },
        }
    )
    response = proxy._handle_rpc_message(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "search",
                "arguments": {"query": "workspace routing"},
                "_meta": {
                    "codex/sandbox-state-meta": {
                        "sandboxCwd": repository.as_uri(),
                    }
                },
            },
        }
    )

    assert "result" in response
    assert captured == [
        (
            "https://cloud.example.com/api/v1/memories/search",
            "github.com/example/repository",
        )
    ]
    assert initialize["result"]["capabilities"]["experimental"] == {"codex/sandbox-state-meta": {}}


def test_explicit_repository_context_beats_codex_sandbox_cwd(monkeypatch, tmp_path):
    proxy = _load_plugin_mcp_proxy()
    explicit_repository = tmp_path / "explicit"
    sandbox_repository = tmp_path / "sandbox"
    for repository, workspace_id in (
        (explicit_repository, "explicit_workspace"),
        (sandbox_repository, "sandbox_workspace"),
    ):
        repository.mkdir()
        _init_git_repo_with_origin(repository, f"https://github.com/example/{repository.name}.git")
        local_config = repository / ".memforge" / "config.toml"
        local_config.parent.mkdir()
        local_config.write_text(f'[memforge]\nworkspace_id = "{workspace_id}"\n', encoding="utf-8")
    captured_urls: list[str] = []

    def fake_http_json(method, path, body, *, target=None, workspace_id=None):
        captured_urls.append(target.resource_url(path))
        return {"data": []}

    monkeypatch.setenv("MEMFORGE_API_URL", "https://cloud.example.com")
    monkeypatch.setenv("MEMFORGE_WORKSPACE_ID", "process_workspace")
    monkeypatch.setattr(proxy, "_http_json", fake_http_json)

    response = proxy._handle_rpc_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "list_sources",
                "arguments": {
                    "repository_context": {
                        "working_directory": str(explicit_repository),
                    }
                },
                "_meta": {
                    "codex/sandbox-state-meta": {
                        "sandboxCwd": sandbox_repository.as_uri(),
                    }
                },
            },
        }
    )

    assert "result" in response
    assert captured_urls == ["https://cloud.example.com/api/v1/sources/searchable"]


def test_codex_sandbox_cwd_beats_settled_roots(monkeypatch, tmp_path):
    proxy = _load_plugin_mcp_proxy()
    root_repository = tmp_path / "root"
    sandbox_repository = tmp_path / "sandbox"
    for repository, workspace_id in (
        (root_repository, "root_workspace"),
        (sandbox_repository, "sandbox_workspace"),
    ):
        repository.mkdir()
        _init_git_repo_with_origin(repository, f"https://github.com/example/{repository.name}.git")
        local_config = repository / ".memforge" / "config.toml"
        local_config.parent.mkdir()
        local_config.write_text(f'[memforge]\nworkspace_id = "{workspace_id}"\n', encoding="utf-8")
    captured_urls: list[str] = []

    def fake_http_json(method, path, body, *, target=None, workspace_id=None):
        captured_urls.append(target.resource_url(path))
        return {"data": []}

    monkeypatch.setenv("MEMFORGE_API_URL", "https://cloud.example.com")
    monkeypatch.setenv("MEMFORGE_WORKSPACE_ID", "process_workspace")
    monkeypatch.setattr(proxy, "_http_json", fake_http_json)
    _provide_mcp_roots(proxy, root_repository)

    result = proxy._call_tool(
        "list_sources",
        {},
        request_meta={"codex/sandbox-state-meta": {"sandboxCwd": sandbox_repository.as_uri()}},
    )

    assert result == {"data": []}
    assert captured_urls == ["https://cloud.example.com/api/v1/sources/searchable"]


def test_codex_sandbox_cwd_bypasses_pending_roots(monkeypatch, tmp_path):
    proxy = _load_plugin_mcp_proxy()
    repository = tmp_path / "repository"
    repository.mkdir()
    _init_git_repo_with_origin(repository, "https://github.com/example/repository.git")
    local_config = repository / ".memforge" / "config.toml"
    local_config.parent.mkdir()
    local_config.write_text('[memforge]\nworkspace_id = "repository_workspace"\n', encoding="utf-8")
    captured_urls: list[str] = []

    def fake_http_json(method, path, body, *, target=None, workspace_id=None):
        captured_urls.append(target.resource_url(path))
        return {"data": []}

    monkeypatch.setenv("MEMFORGE_API_URL", "https://cloud.example.com")
    monkeypatch.setenv("MEMFORGE_WORKSPACE_ID", "process_workspace")
    monkeypatch.setattr(proxy, "_http_json", fake_http_json)
    proxy._handle_rpc_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {"roots": {"listChanged": True}},
                "clientInfo": {"name": "codex"},
            },
        }
    )
    roots_request = proxy._handle_rpc_message({"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert roots_request["method"] == "roots/list"

    response = proxy._handle_rpc_message(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "list_sources",
                "arguments": {},
                "_meta": {
                    "codex/sandbox-state-meta": {
                        "sandboxCwd": repository.as_uri(),
                    }
                },
            },
        }
    )

    assert "result" in response
    assert captured_urls == ["https://cloud.example.com/api/v1/sources/searchable"]


@pytest.mark.parametrize(
    "sandbox_state",
    [
        None,
        "not-an-object",
        {},
        {"sandboxCwd": "relative/path"},
        {"sandboxCwd": "/not/a/git/repository"},
    ],
)
def test_invalid_codex_sandbox_cwd_fails_closed(monkeypatch, sandbox_state):
    proxy = _load_plugin_mcp_proxy()

    def fake_http_json(*_args, **_kwargs):
        raise AssertionError("invalid negotiated context must fail before HTTP")

    monkeypatch.setenv("MEMFORGE_WORKSPACE_ID", "process_workspace")
    monkeypatch.setattr(proxy, "_http_json", fake_http_json)

    response = proxy._handle_rpc_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "list_sources",
                "arguments": {},
                "_meta": {"codex/sandbox-state-meta": sandbox_state},
            },
        }
    )

    payload = json.loads(response["result"]["content"][0]["text"])
    assert "error" in payload
    assert "sandboxCwd" in payload["error"]


def test_explicit_repository_context_bypasses_pending_roots(monkeypatch, tmp_path):
    proxy = _load_plugin_mcp_proxy()
    repository = tmp_path / "repository"
    repository.mkdir()
    _init_git_repo_with_origin(repository, "https://github.com/example/repository.git")
    local_config = repository / ".memforge" / "config.toml"
    local_config.parent.mkdir()
    local_config.write_text('[memforge]\nworkspace_id = "repository_workspace"\n', encoding="utf-8")
    captured_urls: list[str] = []

    class FakeResponse:
        headers = {"content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return b'{"data":[]}'

    class FakeOpener:
        def open(self, request, timeout):
            captured_urls.append(request.full_url)
            return FakeResponse()

    monkeypatch.setenv("MEMFORGE_API_URL", "https://cloud.example.com")
    monkeypatch.setenv("MEMFORGE_API_TOKEN", "token-123")
    monkeypatch.setenv("MEMFORGE_WORKSPACE_ID", "process_workspace")
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())
    proxy._handle_rpc_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {"roots": {"listChanged": True}},
                "clientInfo": {"name": "test"},
            },
        }
    )
    roots_request = proxy._handle_rpc_message({"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert roots_request["method"] == "roots/list"

    response = proxy._handle_rpc_message(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "list_sources",
                "arguments": {"repository_context": {"working_directory": str(repository)}},
            },
        }
    )

    assert "result" in response
    assert captured_urls == ["https://cloud.example.com/api/v1/sources/searchable"]


def test_explicit_repository_context_beats_roots_and_uses_normal_fallback_without_override(monkeypatch, tmp_path):
    proxy = _load_plugin_mcp_proxy()
    root_repository = tmp_path / "root-repository"
    explicit_repository = tmp_path / "explicit-repository"
    fallback_repository = tmp_path / "fallback-repository"
    for repository, remote in (
        (root_repository, "https://github.com/example/root.git"),
        (explicit_repository, "https://github.com/example/explicit.git"),
        (fallback_repository, "https://github.com/example/fallback.git"),
    ):
        repository.mkdir()
        _init_git_repo_with_origin(repository, remote)
    for repository, workspace_id in (
        (root_repository, "root_workspace"),
        (explicit_repository, "explicit_workspace"),
    ):
        local_config = repository / ".memforge" / "config.toml"
        local_config.parent.mkdir()
        local_config.write_text(f'[memforge]\nworkspace_id = "{workspace_id}"\n', encoding="utf-8")
    captured_urls: list[str] = []

    class FakeResponse:
        headers = {"content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return b'{"data":[]}'

    class FakeOpener:
        def open(self, request, timeout):
            captured_urls.append(request.full_url)
            return FakeResponse()

    monkeypatch.setenv("MEMFORGE_API_URL", "https://cloud.example.com")
    monkeypatch.setenv("MEMFORGE_API_TOKEN", "token-123")
    monkeypatch.setenv("MEMFORGE_WORKSPACE_ID", "process_workspace")
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())
    _provide_mcp_roots(proxy, root_repository)

    proxy._call_tool(
        "list_sources",
        {"repository_context": {"working_directory": str(explicit_repository)}},
    )
    proxy._call_tool(
        "list_sources",
        {"repository_context": {"working_directory": str(fallback_repository)}},
    )

    assert captured_urls == [
        "https://cloud.example.com/api/v1/sources/searchable",
        "https://cloud.example.com/api/v1/sources/searchable",
    ]


@pytest.mark.parametrize(
    "repository_context",
    [
        "not-an-object",
        {},
        {"working_directory": "relative/path"},
        {"working_directory": "/not/a/git/repository"},
        {"working_directory": "/tmp", "unexpected": True},
    ],
)
def test_invalid_explicit_repository_context_fails_closed(monkeypatch, repository_context):
    proxy = _load_plugin_mcp_proxy()

    class FakeOpener:
        def open(self, *_args, **_kwargs):
            raise AssertionError("invalid explicit context must fail before HTTP")

    monkeypatch.setenv("MEMFORGE_WORKSPACE_ID", "process_workspace")
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())

    result = proxy._call_tool(
        "list_sources",
        {"repository_context": repository_context},
    )

    assert "error" in result
    assert "repository_context" in result["error"]


def test_concurrent_repository_contexts_keep_workspace_and_attribution_request_local(monkeypatch, tmp_path):
    proxy = _load_plugin_mcp_proxy()
    repositories: list[tuple[Path, str, str]] = []
    for name, workspace_id in (("alpha", "alpha_workspace"), ("beta", "beta_workspace")):
        repository = tmp_path / name
        repository.mkdir()
        _init_git_repo_with_origin(repository, f"https://github.com/example/{name}.git")
        local_config = repository / ".memforge" / "config.toml"
        local_config.parent.mkdir()
        local_config.write_text(f'[memforge]\nworkspace_id = "{workspace_id}"\n', encoding="utf-8")
        repositories.append((repository, name, workspace_id))
    captured: dict[str, tuple[str, str | None]] = {}

    def fake_http_json(method, path, body, *, target=None, workspace_id=None):
        captured[body["query"]] = (
            target.resource_url(path),
            body.get("active_repo_identifier"),
        )
        return {"results": []}

    monkeypatch.setenv("MEMFORGE_API_URL", "https://cloud.example.com")
    monkeypatch.setenv("MEMFORGE_WORKSPACE_ID", "process_workspace")
    monkeypatch.setattr(proxy, "_http_json", fake_http_json)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                proxy._call_tool,
                "search",
                {
                    "query": name,
                    "repository_context": {"working_directory": str(repository)},
                },
            )
            for repository, name, _workspace_id in repositories
        ]
        assert [future.result() for future in futures] == [{"results": []}, {"results": []}]

    for _repository, name, _workspace_id in repositories:
        url, repo_identifier = captured[name]
        assert url == "https://cloud.example.com/api/v1/memories/search"
        assert repo_identifier == f"github.com/example/{name}"


def test_concurrent_codex_sandbox_cwds_keep_workspace_and_attribution_request_local(monkeypatch, tmp_path):
    proxy = _load_plugin_mcp_proxy()
    repositories: list[tuple[Path, str, str]] = []
    for name, workspace_id in (("alpha", "alpha_workspace"), ("beta", "beta_workspace")):
        repository = tmp_path / name
        repository.mkdir()
        _init_git_repo_with_origin(repository, f"https://github.com/example/{name}.git")
        local_config = repository / ".memforge" / "config.toml"
        local_config.parent.mkdir()
        local_config.write_text(f'[memforge]\nworkspace_id = "{workspace_id}"\n', encoding="utf-8")
        repositories.append((repository, name, workspace_id))
    captured: dict[str, tuple[str, str | None]] = {}

    def fake_http_json(method, path, body, *, target=None, workspace_id=None):
        captured[body["query"]] = (
            target.resource_url(path),
            body.get("active_repo_identifier"),
        )
        return {"results": []}

    monkeypatch.setenv("MEMFORGE_API_URL", "https://cloud.example.com")
    monkeypatch.setenv("MEMFORGE_WORKSPACE_ID", "process_workspace")
    monkeypatch.setattr(proxy, "_http_json", fake_http_json)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                proxy._call_tool,
                "search",
                {"query": name},
                request_meta={"codex/sandbox-state-meta": {"sandboxCwd": repository.as_uri()}},
            )
            for repository, name, _workspace_id in repositories
        ]
        assert [future.result() for future in futures] == [{"results": []}, {"results": []}]

    for _repository, name, _workspace_id in repositories:
        url, repo_identifier = captured[name]
        assert url == "https://cloud.example.com/api/v1/memories/search"
        assert repo_identifier == f"github.com/example/{name}"


def test_plugin_config_parity_normalizer_rejects_mixed_target_import_block():
    source = """
if __package__:
    from .api_target import MemForgeTarget, build_target
    unrelated_statement = True
else:
    from memforge.api_target import MemForgeTarget, build_target
"""

    with pytest.raises(AssertionError, match="target import compatibility block must be exact"):
        _normalized_plugin_config_ast(source)


def test_plugin_mcp_launchers_match_each_other():
    root = Path(__file__).resolve().parents[1]
    canonical = root / "src" / "memforge" / "plugin_mcp_proxy.py"
    codex_launcher = root / "integrations" / "codex" / "memforge-memory" / "scripts" / "memforge_mcp.py"
    claude_launcher = root / "integrations" / "claude-code" / "memforge-memory" / "scripts" / "memforge_mcp.py"

    assert codex_launcher.read_text() == canonical.read_text()
    assert codex_launcher.read_text() == claude_launcher.read_text()


def test_packaged_mcp_proxy_starts_in_isolated_python_without_memforge_package():
    root = Path(__file__).resolve().parents[1]
    proxy = root / "integrations" / "codex" / "memforge-memory" / "scripts" / "memforge_mcp.py"
    manifest = json.loads(
        (root / "integrations" / "codex" / "memforge-memory" / ".codex-plugin" / "plugin.json").read_text()
    )
    request = _mcp_initialize_request()
    frame = b"Content-Length: " + str(len(request)).encode() + b"\r\n\r\n" + request
    launcher = "import runpy, sys; sys.path.insert(0, sys.argv[1]); runpy.run_path(sys.argv[2], run_name='__main__')"

    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            launcher,
            str(proxy.parent),
            str(proxy),
        ],
        input=frame,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=5,
        check=False,
    )

    assert result.returncode == 0
    assert result.stderr == b""
    _, payload = result.stdout.split(b"\r\n\r\n", 1)
    response = json.loads(payload)
    assert response["result"]["serverInfo"]["name"] == "memforge"
    assert response["result"]["serverInfo"]["version"] == manifest["version"]
    assert response["result"]["capabilities"]["tools"]["listChanged"] is False


def test_packaged_mcp_emits_deferred_tool_response_after_roots_response(tmp_path):
    root = Path(__file__).resolve().parents[1]
    proxy = root / "integrations" / "codex" / "memforge-memory" / "scripts" / "memforge_mcp.py"
    messages = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {"roots": {"listChanged": True}},
                "clientInfo": {"name": "test"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "unknown_tool", "arguments": {}},
        },
        {
            "jsonrpc": "2.0",
            "id": "memforge-roots-list-1",
            "result": {"roots": [{"uri": tmp_path.as_uri(), "name": "test"}]},
        },
    ]
    frames = b"".join(
        b"Content-Length: " + str(len(payload)).encode() + b"\r\n\r\n" + payload
        for message in messages
        if (payload := json.dumps(message).encode())
    )

    result = subprocess.run(
        [sys.executable, str(proxy)],
        input=frames,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=5,
        check=False,
    )

    assert result.returncode == 0
    assert result.stderr == b""
    responses = _mcp_content_length_messages(result.stdout)
    assert [(response.get("method"), response.get("id")) for response in responses] == [
        (None, 1),
        ("roots/list", "memforge-roots-list-1"),
        (None, 2),
    ]
    tool_payload = json.loads(responses[-1]["result"]["content"][0]["text"])
    assert tool_payload == {"error": "Unknown tool: unknown_tool"}


@pytest.mark.parametrize("client", ["codex", "claude-code"])
def test_packaged_mcp_proxy_starts_when_tomllib_is_unavailable(client):
    root = Path(__file__).resolve().parents[1]
    proxy = root / "integrations" / client / "memforge-memory" / "scripts" / "memforge_mcp.py"
    request = _mcp_initialize_request()
    frame = b"Content-Length: " + str(len(request)).encode() + b"\r\n\r\n" + request
    launcher = """
import builtins
import os
import runpy
import sys

original_import = builtins.__import__


def import_without_tomllib(name, *args, **kwargs):
    if name == "tomllib":
        raise ModuleNotFoundError("No module named 'tomllib'")
    return original_import(name, *args, **kwargs)


builtins.__import__ = import_without_tomllib
sys.path.insert(0, os.path.dirname(sys.argv[1]))
runpy.run_path(sys.argv[1], run_name="__main__")
"""

    result = subprocess.run(
        [sys.executable, "-c", launcher, str(proxy)],
        input=frame,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=5,
        check=False,
    )

    assert result.returncode == 0
    assert result.stderr == b""
    _, payload = result.stdout.split(b"\r\n\r\n", 1)
    response = json.loads(payload)
    assert response["result"]["serverInfo"]["name"] == "memforge"


def test_mcp_proxy_supports_json_line_stdio():
    root = Path(__file__).resolve().parents[1]
    proxy = root / "integrations" / "codex" / "memforge-memory" / "scripts" / "memforge_mcp.py"
    request = _mcp_initialize_request() + b"\n"

    result = subprocess.run(
        [sys.executable, str(proxy)],
        input=request,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=5,
        check=False,
    )

    assert result.returncode == 0
    assert result.stderr == b""
    response = json.loads(result.stdout)
    assert response["result"]["serverInfo"]["name"] == "memforge"
    assert response["result"]["capabilities"]["tools"]["listChanged"] is False
    assert response["result"]["capabilities"]["experimental"] == {"codex/sandbox-state-meta": {}}


def test_mcp_proxy_instructions_request_per_call_repository_context_without_guessing():
    proxy = _load_plugin_mcp_proxy()

    response = proxy._handle_rpc_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "portable-agent-client"},
            },
        }
    )

    instructions = response["result"]["instructions"]
    assert "repository_context" in instructions
    assert "current working directory" in instructions
    assert "Never guess" in instructions
    assert "Omit it" in instructions
    assert "hook" not in instructions.lower()


@pytest.mark.parametrize("client", ["codex", "claude-code"])
def test_mcp_proxy_launchers_request_roots_when_client_supports_roots(client):
    root = Path(__file__).resolve().parents[1]
    proxy = root / "integrations" / client / "memforge-memory" / "scripts" / "memforge_mcp.py"
    initialize = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {"roots": {"listChanged": True}},
                "clientInfo": {"name": "test"},
            },
        }
    ).encode()
    initialized = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}).encode()
    frame = (
        b"Content-Length: "
        + str(len(initialize)).encode()
        + b"\r\n\r\n"
        + initialize
        + b"Content-Length: "
        + str(len(initialized)).encode()
        + b"\r\n\r\n"
        + initialized
    )

    result = subprocess.run(
        [sys.executable, str(proxy)],
        input=frame,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=5,
        check=False,
    )

    assert result.returncode == 0
    assert result.stderr == b""
    messages = _mcp_content_length_messages(result.stdout)
    assert messages[0]["result"]["serverInfo"]["name"] == "memforge"
    assert messages[1]["method"] == "roots/list"


def test_mcp_proxy_retries_roots_request_after_list_changed_while_previous_request_is_pending():
    proxy = _load_plugin_mcp_proxy()
    initialize = proxy._handle_rpc_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {"roots": {"listChanged": True}},
                "clientInfo": {"name": "test"},
            },
        }
    )
    assert initialize["result"]["serverInfo"]["name"] == "memforge"
    first_roots_request = proxy._handle_rpc_message({"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert first_roots_request["method"] == "roots/list"

    retry_roots_request = proxy._handle_rpc_message({"jsonrpc": "2.0", "method": "notifications/roots/list_changed"})

    assert retry_roots_request["method"] == "roots/list"
    assert retry_roots_request["id"] == first_roots_request["id"]


def _mcp_initialize_request() -> bytes:
    return json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "test"}},
        }
    ).encode()


def _mcp_content_length_messages(stream: bytes) -> list[dict]:
    messages = []
    remaining = stream
    while remaining:
        header, payload_start = remaining.split(b"\r\n\r\n", 1)
        length = int(header.removeprefix(b"Content-Length: ").strip())
        payload = payload_start[:length]
        messages.append(json.loads(payload))
        remaining = payload_start[length:]
    return messages


def test_mcp_proxy_forwards_search_to_service_with_token(monkeypatch):
    proxy = _load_plugin_mcp_proxy()
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
            captured["timeout"] = timeout
            captured["authorization"] = request.get_header("Authorization")
            captured["content_type"] = request.get_header("Content-type")
            captured["body"] = request.data
            return FakeResponse()

    monkeypatch.setenv("MEMFORGE_API_URL", "https://self.example")
    monkeypatch.setenv("MEMFORGE_API_TOKEN", "token-123")
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())
    monkeypatch.setattr(proxy, "_active_repo_identifier", lambda: None)

    result = proxy._call_tool("search", {"query": "artifact cache"})

    assert result == {"results": []}
    assert captured["url"] == "https://self.example/api/v1/memories/search"
    assert captured["authorization"] == "Bearer token-123"
    assert captured["content_type"] == "application/json"
    assert json.loads(captured["body"].decode()) == {
        "query": "artifact cache",
        "include_private": True,
        "include_superseded": False,
    }


def test_mcp_proxy_forwards_explicit_search_entities(monkeypatch):
    proxy = _load_plugin_mcp_proxy()
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
            captured["body"] = request.data
            return FakeResponse()

    monkeypatch.setenv("MEMFORGE_API_URL", "https://self.example")
    monkeypatch.setenv("MEMFORGE_API_TOKEN", "token-123")
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())
    monkeypatch.setattr(proxy, "_active_repo_identifier", lambda: None)

    result = proxy._call_tool(
        "search",
        {
            "query": "deployment owner",
            "entities": ["payroll control center", "deployment owner"],
        },
    )

    assert result == {"results": []}
    assert json.loads(captured["body"].decode()) == {
        "query": "deployment owner",
        "entities": ["payroll control center", "deployment owner"],
        "include_private": True,
        "include_superseded": False,
    }


def test_mcp_proxy_trims_and_dedupes_explicit_search_entities(monkeypatch):
    proxy = _load_plugin_mcp_proxy()
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
            captured["body"] = request.data
            return FakeResponse()

    monkeypatch.setenv("MEMFORGE_API_URL", "https://self.example")
    monkeypatch.setenv("MEMFORGE_API_TOKEN", "token-123")
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())
    monkeypatch.setattr(proxy, "_active_repo_identifier", lambda: None)

    result = proxy._call_tool(
        "search",
        {
            "query": "deployment owner",
            "entities": [" Payroll Control Center ", "payroll control center", "Deployment Owner"],
        },
    )

    assert result == {"results": []}
    assert json.loads(captured["body"].decode())["entities"] == [
        "Payroll Control Center",
        "Deployment Owner",
    ]


def test_mcp_proxy_omits_empty_explicit_search_entities(monkeypatch):
    proxy = _load_plugin_mcp_proxy()
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
            captured["body"] = request.data
            return FakeResponse()

    monkeypatch.setenv("MEMFORGE_API_URL", "https://self.example")
    monkeypatch.setenv("MEMFORGE_API_TOKEN", "token-123")
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())
    monkeypatch.setattr(proxy, "_active_repo_identifier", lambda: None)

    result = proxy._call_tool("search", {"query": "deployment owner", "entities": []})

    assert result == {"results": []}
    assert json.loads(captured["body"].decode()) == {
        "query": "deployment owner",
        "include_private": True,
        "include_superseded": False,
    }


@pytest.mark.parametrize(
    "entities",
    [
        "payroll control center",
        ["payroll control center", 3],
        ["payroll control center", ""],
        ["payroll control center", "   "],
        ["e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", "e9"],
        ["x" * 129],
    ],
)
def test_mcp_proxy_rejects_invalid_explicit_search_entities(monkeypatch, entities):
    proxy = _load_plugin_mcp_proxy()
    monkeypatch.setattr(proxy, "_active_repo_identifier", lambda: None)

    result = proxy._call_tool("search", {"query": "deployment owner", "entities": entities})

    assert result == {"error": "entities must be an array of 1-8 strings, each 1-128 characters after trimming"}


def test_mcp_proxy_forwards_list_sources_to_searchable_sources(monkeypatch):
    proxy = _load_plugin_mcp_proxy()
    captured = {}

    class FakeResponse:
        headers = {"content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return (
                b'{"data":[{"source_id":"src-alpha","name":"TeamAlpha Defects",'
                b'"type":"jira","status":"active","doc_count":68,"memory_count":199,'
                b'"last_synced_at":"2026-06-27T00:00:00Z"}]}'
            )

    class FakeOpener:
        def open(self, request, timeout):
            captured["url"] = request.full_url
            captured["authorization"] = request.get_header("Authorization")
            return FakeResponse()

    monkeypatch.setenv("MEMFORGE_API_URL", "https://memforge.example.com")
    monkeypatch.setenv("MEMFORGE_API_TOKEN", "token-123")
    monkeypatch.setenv("MEMFORGE_WORKSPACE_ID", "team_alpha")
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())

    result = proxy._call_tool("list_sources", {})

    assert result["data"][0]["source_id"] == "src-alpha"
    assert captured["url"] == "https://memforge.example.com/api/v1/sources/searchable"
    assert captured["authorization"] == "Bearer token-123"


def test_mcp_proxy_adds_client_root_git_remote_as_ranking_hint_only(monkeypatch, tmp_path):
    proxy = _load_plugin_mcp_proxy()
    captured = {}
    repo_root = tmp_path / "memforge-cloud"
    repo_root.mkdir()
    _init_git_repo_with_origin(repo_root, "https://github.com/dodoman-sun/memforge-cloud.git")

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
            captured["body"] = request.data
            return FakeResponse()

    _provide_mcp_roots(proxy, repo_root)
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())

    proxy._call_tool("search", {"query": "scheduler fix"})

    body = json.loads(captured["body"].decode())
    assert body["active_repo_identifier"] == "github.com/dodoman-sun/memforge-cloud"
    assert "source_filter" not in body


def test_mcp_proxy_uses_explicit_repository_context_as_ranking_hint(monkeypatch, tmp_path):
    proxy = _load_plugin_mcp_proxy()
    captured = {}
    repo_root = tmp_path / "memforge-cloud"
    repo_root.mkdir()
    _init_git_repo_with_origin(repo_root, "https://github.com/dodoman-sun/memforge-cloud.git")

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
            captured["body"] = request.data
            return FakeResponse()

    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())

    proxy._call_tool(
        "search",
        {
            "query": "scheduler fix",
            "repository_context": {
                "working_directory": str(repo_root),
            },
        },
    )

    body = json.loads(captured["body"].decode())
    assert body["active_repo_identifier"] == "github.com/dodoman-sun/memforge-cloud"
    assert "repository_context" not in body


def test_mcp_proxy_explicit_repository_context_selects_one_repo_from_multiple_roots(monkeypatch, tmp_path):
    proxy = _load_plugin_mcp_proxy()
    captured = {}
    active_root = tmp_path / "memforge-cloud"
    other_root = tmp_path / "mem-forge"
    active_root.mkdir()
    other_root.mkdir()
    _init_git_repo_with_origin(active_root, "https://github.com/dodoman-sun/memforge-cloud.git")
    _init_git_repo_with_origin(other_root, "https://github.com/dodo-labs-ai/mem-forge.git")
    _provide_mcp_roots(proxy, active_root, other_root)

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
            captured["body"] = request.data
            return FakeResponse()

    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())

    proxy._call_tool(
        "search",
        {
            "query": "scheduler fix",
            "repository_context": {
                "working_directory": active_root.as_uri(),
            },
        },
    )

    body = json.loads(captured["body"].decode())
    assert body["active_repo_identifier"] == "github.com/dodoman-sun/memforge-cloud"
    assert "repository_context" not in body


def test_mcp_proxy_rejects_current_repo_filter_when_repo_context_is_unavailable(monkeypatch):
    proxy = _load_plugin_mcp_proxy()
    monkeypatch.setenv("MEMFORGE_ACTIVE_REPO_IDENTIFIER", "github.example.com/acme/memforge-cloud")

    class FakeOpener:
        def open(self, *_args, **_kwargs):
            raise AssertionError("current_repo_only should fail before posting")

    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())

    result = proxy._call_tool(
        "search",
        {
            "query": "scheduler fix",
            "source_filter": {
                "current_repo_only": True,
            },
        },
    )

    assert result == {
        "error": (
            "current_repo_only is not exposed by this MCP search tool. Omit the filter to search all visible memories."
        )
    }


def test_mcp_proxy_rejects_current_repo_filter_even_when_client_roots_are_available(monkeypatch, tmp_path):
    proxy = _load_plugin_mcp_proxy()
    repo_root = tmp_path / "memforge-cloud"
    repo_root.mkdir()
    _init_git_repo_with_origin(repo_root, "https://github.com/dodoman-sun/memforge-cloud.git")
    plugin_root = tmp_path / "plugin"
    plugin_root.mkdir()
    monkeypatch.chdir(plugin_root)

    class FakeOpener:
        def open(self, *_args, **_kwargs):
            raise AssertionError("current_repo_only should fail before posting")

    initialize = proxy._handle_rpc_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {"roots": {"listChanged": True}},
                "clientInfo": {"name": "test"},
            },
        }
    )
    assert initialize["result"]["serverInfo"]["name"] == "memforge"
    roots_request = proxy._handle_rpc_message(
        {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        }
    )
    assert roots_request["method"] == "roots/list"

    assert (
        proxy._handle_rpc_message(
            {
                "jsonrpc": "2.0",
                "id": roots_request["id"],
                "result": {"roots": [{"uri": repo_root.as_uri(), "name": "memforge-cloud"}]},
            }
        )
        is None
    )
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())

    result = proxy._call_tool(
        "search",
        {
            "query": "scheduler fix",
            "source_filter": {
                "current_repo_only": True,
            },
        },
    )

    assert result == {
        "error": (
            "current_repo_only is not exposed by this MCP search tool. Omit the filter to search all visible memories."
        )
    }


def test_mcp_proxy_rejects_current_repo_filter_when_cwd_is_git_repo_but_roots_are_missing(monkeypatch, tmp_path):
    proxy = _load_plugin_mcp_proxy()
    repo_root = tmp_path / "memforge-cloud"
    repo_root.mkdir()
    _init_git_repo_with_origin(repo_root, "https://github.com/dodoman-sun/memforge-cloud.git")
    monkeypatch.chdir(repo_root)

    class FakeOpener:
        def open(self, *_args, **_kwargs):
            raise AssertionError("current_repo_only should fail before posting")

    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())

    result = proxy._call_tool(
        "search",
        {
            "query": "scheduler fix",
            "source_filter": {
                "current_repo_only": True,
            },
        },
    )

    assert result == {
        "error": (
            "current_repo_only is not exposed by this MCP search tool. Omit the filter to search all visible memories."
        )
    }


def test_mcp_proxy_ignores_codex_workspace_root_when_roots_are_not_advertised(monkeypatch, tmp_path):
    proxy = _load_plugin_mcp_proxy()
    captured = {}
    repo_root = tmp_path / "memforge-cloud"
    repo_root.mkdir()
    _init_git_repo_with_origin(repo_root, "https://github.com/dodoman-sun/memforge-cloud.git")
    plugin_root = tmp_path / "plugin-cache"
    plugin_root.mkdir()
    monkeypatch.chdir(plugin_root)

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
            captured["body"] = request.data
            return FakeResponse()

    monkeypatch.setenv("CODEX_WORKSPACE_ROOT", str(repo_root))
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())

    proxy._call_tool(
        "search",
        {
            "query": "scheduler fix",
        },
    )

    body = json.loads(captured["body"].decode())
    assert "active_repo_identifier" not in body
    assert "source_filter" not in body


def test_mcp_proxy_rejects_current_repo_filter_when_root_has_no_git_remote(monkeypatch, tmp_path):
    proxy = _load_plugin_mcp_proxy()
    repo_root = tmp_path / "local-only"
    repo_root.mkdir()
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _provide_mcp_roots(proxy, repo_root)

    class FakeOpener:
        def open(self, *_args, **_kwargs):
            raise AssertionError("current_repo_only should fail before posting")

    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())

    result = proxy._call_tool(
        "search",
        {
            "query": "scheduler fix",
            "source_filter": {
                "current_repo_only": True,
            },
        },
    )

    assert result == {
        "error": (
            "current_repo_only is not exposed by this MCP search tool. Omit the filter to search all visible memories."
        )
    }


def test_mcp_proxy_rejects_current_repo_filter_when_roots_have_multiple_git_remotes(monkeypatch, tmp_path):
    proxy = _load_plugin_mcp_proxy()
    first_root = tmp_path / "memforge-cloud"
    second_root = tmp_path / "mem-forge"
    first_root.mkdir()
    second_root.mkdir()
    _init_git_repo_with_origin(first_root, "https://github.com/dodoman-sun/memforge-cloud.git")
    _init_git_repo_with_origin(second_root, "https://github.com/dodo-labs-ai/mem-forge.git")
    _provide_mcp_roots(proxy, first_root, second_root)

    class FakeOpener:
        def open(self, *_args, **_kwargs):
            raise AssertionError("current_repo_only should fail before posting")

    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())

    result = proxy._call_tool(
        "search",
        {
            "query": "scheduler fix",
            "source_filter": {
                "current_repo_only": True,
            },
        },
    )

    assert result == {
        "error": (
            "current_repo_only is not exposed by this MCP search tool. Omit the filter to search all visible memories."
        )
    }


def test_mcp_proxy_rejects_unadvertised_search_source_ids(monkeypatch):
    proxy = _load_plugin_mcp_proxy()
    monkeypatch.setattr(proxy, "_active_repo_identifier", lambda: None)

    result = proxy._call_tool(
        "search",
        {"query": "scheduler fix", "sources": ["src-hidden"]},
    )

    assert result == {"error": ("Unsupported search parameter(s): sources. Omit unknown filters instead of guessing.")}


def test_mcp_proxy_rejects_explicit_repo_hint(monkeypatch):
    proxy = _load_plugin_mcp_proxy()
    monkeypatch.setattr(proxy, "_active_repo_identifier", lambda: None)

    result = proxy._call_tool(
        "search",
        {
            "query": "scheduler fix",
            "active_repo_identifier": "github.example.com/acme/other",
        },
    )

    assert result == {
        "error": ("Unsupported search parameter(s): active_repo_identifier. Omit unknown filters instead of guessing.")
    }


def test_mcp_proxy_search_schema_exposes_validated_facets_not_recent_changes():
    proxy = _load_plugin_mcp_proxy()

    tools = {tool["name"]: tool for tool in proxy.TOOLS}

    assert "search" in tools
    assert "list_sources" in tools
    assert "list_recent_changes" not in tools
    assert "suggest_memory_replacement" not in tools
    assert "create_memory" in tools
    assert "retire_memory" in tools
    assert "propose_memory_correction" in tools
    assert "replace_memory" not in tools
    assert "list_memory_reviews" in tools
    assert "get_memory_review" in tools
    assert "refresh_memory_review" not in tools
    assert "resolve_memory_review" in tools
    assert "validate_memory_review_decisions" in tools
    assert "apply_memory_review_decisions" in tools
    validate_tool = next(tool for tool in proxy.TOOLS if tool["name"] == "validate_memory_review_decisions")
    apply_tool = next(tool for tool in proxy.TOOLS if tool["name"] == "apply_memory_review_decisions")
    assert validate_tool["outputSchema"] == apply_tool["outputSchema"]
    assert apply_tool["annotations"]["destructiveHint"] is True
    assert "validation_receipt" in apply_tool["inputSchema"]["required"]
    assert tools["resolve_memory_review"]["annotations"] == {
        "readOnlyHint": False,
        "destructiveHint": True,
    }
    result_schema = apply_tool["outputSchema"]["properties"]["results"]["items"]
    assert result_schema["additionalProperties"] is False
    assert set(result_schema["required"]) == {
        "review_id",
        "decision",
        "outcome",
        "status",
        "message",
        "decision_label",
        "consequence",
    }

    search_schema = tools["search"]["inputSchema"]
    properties = search_schema["properties"]
    assert "source_types" not in properties["source_filter"]["properties"]
    assert properties["source_filter"]["properties"]["source_ids"]["items"]["type"] == "string"
    assert "list_sources" in properties["source_filter"]["properties"]["source_ids"]["description"]
    assert set(properties["source_filter"]["properties"]) == {"source_ids"}
    assert "current_repo_only" not in properties["source_filter"]["properties"]
    assert "repo_identifiers" not in properties["source_filter"]["properties"]
    assert "source_instance_ids" not in properties["source_filter"]["properties"]
    assert "sources" not in properties
    assert set(properties) == {
        "query",
        "intent",
        "repository_context",
        "source_filter",
        "time_range",
        "top_k",
        "offset",
        "entities",
        "workspace_id",
    }
    repository_context = properties["repository_context"]
    assert repository_context["additionalProperties"] is False
    assert set(repository_context["properties"]) == {"working_directory"}
    assert repository_context["properties"]["working_directory"]["type"] == "string"
    assert "current working directory" in repository_context["properties"]["working_directory"]["description"]
    assert "Do not guess" in repository_context["properties"]["working_directory"]["description"]
    assert properties["entities"]["type"] == "array"
    assert properties["entities"]["maxItems"] == 8
    assert properties["entities"]["items"]["type"] == "string"
    assert properties["entities"]["items"]["minLength"] == 1
    assert properties["entities"]["items"]["maxLength"] == 128
    assert "agent-selected entity hints" in properties["entities"]["description"]
    assert "keeping query unchanged" in properties["entities"]["description"]
    assert "not filters or authority" in properties["entities"]["description"]
    assert "backlog" not in tools["search"]["description"].lower()
    assert "bounded ranked window" in properties["top_k"]["description"]
    assert "up to 50" in properties["top_k"]["description"]
    assert "does not make ranked retrieval exhaustive" in properties["top_k"]["description"]
    assert "list_recent_memories" in properties["top_k"]["description"]
    assert "backlog" not in properties["top_k"]["description"].lower()
    assert properties["top_k"]["default"] == 20
    assert properties["top_k"]["minimum"] == 1
    assert properties["top_k"]["maximum"] == 50
    assert properties["offset"]["default"] == 0
    assert properties["offset"]["minimum"] == 0
    assert "ranked window" in properties["offset"]["description"]
    assert search_schema["required"] == ["query"]
    time_range_schema = properties["time_range"]
    assert "Omit time_range" in time_range_schema["description"]
    assert "start_date and end_date are individually optional" in time_range_schema["description"]
    assert time_range_schema["anyOf"] == [{"required": ["start_date"]}, {"required": ["end_date"]}]
    assert set(time_range_schema["properties"]) == {"date_type", "start_date", "end_date"}
    assert "include_private" not in properties
    assert "include_superseded" not in properties
    assert "status" not in properties
    assert "memory_types" not in properties
    assert "active_repo_identifier" not in properties
    assert "search -> get_memory -> get_resource" in tools["get_resource"]["description"]
    get_resource_url = tools["get_resource"]["inputSchema"]["properties"]["url"]["description"]
    assert "get_memory.evidence[].document" in get_resource_url
    assert "/api/v1/source-artifacts/{observation_revision_id}" in get_resource_url

    create_schema = tools["create_memory"]["inputSchema"]
    assert create_schema["required"] == ["content", "provenance"]
    assert create_schema["properties"]["memory_type"]["enum"] == ["fact", "decision", "convention", "procedure"]
    assert "provenance" in create_schema["properties"]
    assert "reason" not in create_schema["properties"]
    assert "client" not in create_schema["properties"]
    assert "repo_identifier" not in create_schema["properties"]
    assert create_schema["properties"]["repository_context"] == repository_context
    assert "readable preview" in tools["create_memory"]["description"]
    assert "request_user_input" in tools["create_memory"]["description"]
    assert "durable memory content" in tools["create_memory"]["description"]
    assert "provenance" in tools["create_memory"]["description"]
    assert "one self-contained" in create_schema["properties"]["content"]["description"].lower()
    assert "get_memory.evidence[].items[].text" in create_schema["properties"]["provenance"]["description"]
    assert "canonical content" in tools["get_memory"]["description"]
    assert "grouped Evidence" in tools["get_memory"]["description"]
    assert "sources[]" not in tools["get_memory"]["description"]
    assert "Do not put confirmation details" in create_schema["properties"]["content"]["description"]

    retire_schema = tools["retire_memory"]["inputSchema"]
    assert retire_schema["required"] == ["memory_id", "reason", "expected_content_hash"]
    assert "status" not in retire_schema["properties"]

    correction_schema = tools["propose_memory_correction"]["inputSchema"]
    assert correction_schema["required"] == [
        "memory_id",
        "replacement_content",
        "provenance",
        "reason",
        "expected_content_hash",
    ]
    assert "provenance" in correction_schema["properties"]
    assert (
        "Do not put confirmation details"
        in correction_schema["properties"]["replacement_content"]["description"]
    )
    assert "provenance" in tools["propose_memory_correction"]["description"]
    assert "direct-apply-or-Review consequence" in tools["propose_memory_correction"]["description"]
    assert tools["propose_memory_correction"]["annotations"] == {
        "readOnlyHint": False,
        "destructiveHint": True,
    }
    assert correction_schema["properties"]["replacement_kind"]["enum"] == ["revision", "supersession"]
    assert "status" not in correction_schema["properties"]

    resolve_schema = tools["resolve_memory_review"]["inputSchema"]
    assert resolve_schema["properties"]["decision"]["enum"] == ["approve", "reject"]
    assert "reviewer" not in resolve_schema["properties"]
    assert "expected_fingerprint" in resolve_schema["required"]
    assert "requires_note=true" in resolve_schema["properties"]["note"]["description"]


def test_mcp_proxy_source_selection_descriptions_guide_scoped_and_global_search():
    proxy = _load_plugin_mcp_proxy()
    tools = {tool["name"]: tool for tool in proxy.TOOLS}

    search_description = tools["search"]["description"]
    list_sources_description = tools["list_sources"]["description"]

    assert "call list_sources first" in search_description
    assert "exact source_ids" in search_description
    assert "omit source_filter" in search_description
    assert "time_range only when explicitly requested" in search_description
    assert "deterministic source/time listings" in search_description
    assert "total_candidates and offset" in search_description
    assert "ranked queries are not exhaustive" in search_description
    assert "Call get_memory for provenance" in search_description
    assert "self-contained query in the user's language" in search_description
    assert "preserve identifiers and domain terms" in search_description
    assert "translation or keyword stuffing" in search_description
    assert len(search_description) <= 700

    search_schema = tools["search"]["inputSchema"]
    assert search_schema["required"] == ["query"]
    assert search_schema["properties"]["query"]["minLength"] == 1
    assert search_schema["properties"]["intent"]["enum"] == [
        "general_hybrid",
        "known_item",
        "relationship",
    ]
    intent_description = search_schema["properties"]["intent"]["description"]
    assert "conversation context" in intent_description
    assert "quote the exact title, name, or path" in intent_description
    assert "self-contained" in intent_description

    assert "Use before source-specific search" in list_sources_description
    assert "exact source_ids" in list_sources_description
    assert "skip for broad or cross-source requests" in list_sources_description
    assert "Returns source_id, name, type, status, counts, and last_synced_at" in list_sources_description
    assert len(list_sources_description) <= 260


def test_mcp_proxy_exposes_recent_current_memory_listing_without_a_query():
    proxy = _load_plugin_mcp_proxy()
    tools = {tool["name"]: tool for tool in proxy.TOOLS}

    tool = tools["list_recent_memories"]
    schema = tool["inputSchema"]

    assert "current Memories" in tool["description"]
    assert "not a source changelog" in tool["description"]
    assert "Do not supply or invent a query" in tool["description"]
    assert "query" not in schema["properties"]
    assert schema["required"] == ["time_range"]
    assert schema["properties"]["time_range"]["required"] == ["start_at", "end_at"]
    assert schema["properties"]["time_range"]["properties"]["date_type"]["enum"] == [
        "source_updated_at",
        "memory_updated_at",
    ]
    assert schema["properties"]["memory_types"]["items"]["enum"] == [
        "fact",
        "decision",
        "convention",
        "procedure",
    ]
    assert schema["properties"]["cursor"]["description"].startswith("Opaque continuation cursor")


def test_mcp_proxy_forwards_recent_memory_listing_and_preserves_pagination_contract(monkeypatch):
    proxy = _load_plugin_mcp_proxy()
    captured = {}

    class FakeResponse:
        headers = {"content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return json.dumps(
                {
                    "results": [
                        {
                            "memory_id": "mem-1",
                            "memory_type": "decision",
                            "summary": "Use stable recent-memory cursors.",
                            "matched_at": "2026-07-25T00:00:00+00:00",
                            "relevance_score": 42.0,
                            "status": "active",
                            "retrieval_evidence": {"large": "internal"},
                        }
                    ],
                    "result_kind": "current_memories",
                    "is_changelog": False,
                    "time_field": "source_updated_at",
                    "resolved_window": {
                        "start_at": "2026-07-20T16:00:00+00:00",
                        "end_at": "2026-07-27T16:00:00+00:00",
                    },
                    "listing_watermark": "2026-07-31T04:00:00+00:00",
                    "cursor_kind": "keyset",
                    "consistency": "request_bound_watermark_not_mvcc_snapshot",
                    "total_candidates": 2,
                    "candidate_count_kind": "exact",
                    "count_scope": "current_page_read",
                    "limit": 1,
                    "has_more": True,
                    "next_cursor": "opaque-next-page",
                }
            ).encode()

    class FakeOpener:
        def open(self, request, timeout):
            captured["url"] = request.full_url
            captured["body"] = request.data
            return FakeResponse()

    monkeypatch.setenv("MEMFORGE_API_URL", "https://memforge.example.com")
    monkeypatch.setenv("MEMFORGE_API_TOKEN", "token-123")
    monkeypatch.setenv("MEMFORGE_WORKSPACE_ID", "team_alpha")
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())

    result = proxy._call_tool(
        "list_recent_memories",
        {
            "source_ids": ["src-jira"],
            "time_range": {
                "date_type": "source_updated_at",
                "start_at": "2026-07-21T00:00:00+08:00",
                "end_at": "2026-07-28T00:00:00+08:00",
            },
            "memory_types": ["decision"],
            "page_size": 1,
        },
    )

    assert captured["url"].endswith("/api/v1/memories/recent")
    assert json.loads(captured["body"].decode()) == {
        "source_ids": ["src-jira"],
        "time_range": {
            "date_type": "source_updated_at",
            "start_at": "2026-07-21T00:00:00+08:00",
            "end_at": "2026-07-28T00:00:00+08:00",
        },
        "memory_types": ["decision"],
        "page_size": 1,
        "include_private": True,
    }
    assert result == {
        "results": [
            {
                "memory_id": "mem-1",
                "memory_type": "decision",
                "summary": "Use stable recent-memory cursors.",
                "matched_at": "2026-07-25T00:00:00+00:00",
                "status": "active",
            }
        ],
        "result_kind": "current_memories",
        "is_changelog": False,
        "time_field": "source_updated_at",
        "resolved_window": {
            "start_at": "2026-07-20T16:00:00+00:00",
            "end_at": "2026-07-27T16:00:00+00:00",
        },
        "listing_watermark": "2026-07-31T04:00:00+00:00",
        "cursor_kind": "keyset",
        "consistency": "request_bound_watermark_not_mvcc_snapshot",
        "total_candidates": 2,
        "candidate_count_kind": "exact",
        "count_scope": "current_page_read",
        "limit": 1,
        "has_more": True,
        "next_cursor": "opaque-next-page",
    }


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            {
                "query": "updates",
                "time_range": {
                    "start_at": "2026-07-21T00:00:00+08:00",
                    "end_at": "2026-07-28T00:00:00+08:00",
                },
            },
            "Unsupported list_recent_memories parameter(s): query",
        ),
        (
            {
                "time_range": {
                    "start_at": "2026-07-21T00:00:00",
                    "end_at": "2026-07-28T00:00:00+08:00",
                },
            },
            "time_range.start_at must include an explicit UTC offset",
        ),
    ],
)
def test_mcp_proxy_rejects_recent_memory_filler_query_or_ambiguous_timezone(arguments, message):
    proxy = _load_plugin_mcp_proxy()

    assert proxy._call_tool("list_recent_memories", arguments) == {"error": message}


def test_mcp_proxy_forwards_search_to_hosted_workspace(monkeypatch):
    proxy = _load_plugin_mcp_proxy()
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

    monkeypatch.setenv("MEMFORGE_API_URL", "https://memforge.example.com")
    monkeypatch.setenv("MEMFORGE_API_TOKEN", "token-123")
    monkeypatch.setenv("MEMFORGE_WORKSPACE_ID", "team_alpha")
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())
    monkeypatch.setattr(proxy, "_active_repo_identifier", lambda: None)

    result = proxy._call_tool(
        "search",
        {"query": "artifact cache", "intent": "known_item"},
    )

    assert result == {"results": []}
    assert captured["url"] == "https://memforge.example.com/api/v1/memories/search"
    assert captured["authorization"] == "Bearer token-123"
    assert json.loads(captured["body"].decode()) == {
        "query": "artifact cache",
        "intent": "known_item",
        "include_private": True,
        "include_superseded": False,
    }


def test_mcp_proxy_forwards_retire_memory_to_lifecycle_endpoint(monkeypatch):
    proxy = _load_plugin_mcp_proxy()
    captured = {}

    class FakeResponse:
        headers = {"content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return b'{"status":"retired","memory_id":"mem-123"}'

    class FakeOpener:
        def open(self, request, timeout):
            captured["url"] = request.full_url
            captured["method"] = request.get_method()
            captured["body"] = request.data
            return FakeResponse()

    monkeypatch.setenv("MEMFORGE_API_URL", "https://memforge.example.com")
    monkeypatch.setenv("MEMFORGE_API_TOKEN", "token-123")
    monkeypatch.setenv("MEMFORGE_WORKSPACE_ID", "team_alpha")
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())

    result = proxy._call_tool(
        "retire_memory",
        {
            "memory_id": "mem-123",
            "reason": "User confirmed this memory is obsolete.",
            "expected_content_hash": "hash-old",
        },
    )

    assert result == {"status": "retired", "memory_id": "mem-123"}
    assert captured["method"] == "POST"
    assert captured["url"] == "https://memforge.example.com/api/v1/memories/mem-123/retire"
    assert json.loads(captured["body"].decode()) == {
        "reason": "User confirmed this memory is obsolete.",
        "expected_content_hash": "hash-old",
    }


def test_mcp_proxy_forwards_create_memory_with_plugin_client_context(monkeypatch):
    proxy = _load_plugin_mcp_proxy()
    captured = {}

    class FakeResponse:
        headers = {"content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return b'{"status":"inserted","memory_id":"mem-new"}'

    class FakeOpener:
        def open(self, request, timeout):
            captured["url"] = request.full_url
            captured["method"] = request.get_method()
            captured["body"] = request.data
            return FakeResponse()

    monkeypatch.setenv("MEMFORGE_API_URL", "https://memforge.example.com")
    monkeypatch.setenv("MEMFORGE_WORKSPACE_ID", "team_alpha")
    monkeypatch.setenv("MEMFORGE_MCP_CLIENT", "claude-code")
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())
    monkeypatch.setattr(proxy, "_active_repo_identifier", lambda: "github.com/dodo-labs-ai/mem-forge")

    result = proxy._call_tool(
        "create_memory",
        {
            "content": "Use readable confirmation previews before memory mutations.",
            "provenance": "User asked to remember this after reviewing the MemForge MCP UX.",
            "memory_type": "convention",
        },
    )

    assert result == {"status": "inserted", "memory_id": "mem-new"}
    assert captured["method"] == "POST"
    assert captured["url"] == "https://memforge.example.com/api/v1/memories/create"
    assert json.loads(captured["body"].decode()) == {
        "content": "Use readable confirmation previews before memory mutations.",
        "provenance": "User asked to remember this after reviewing the MemForge MCP UX.",
        "memory_type": "convention",
        "client": "claude-code",
        "repo_identifier": "github.com/dodo-labs-ai/mem-forge",
    }


def test_mcp_proxy_create_memory_uses_explicit_repository_context(monkeypatch, tmp_path):
    proxy = _load_plugin_mcp_proxy()
    captured = {}
    repo_root = tmp_path / "mem-forge"
    repo_root.mkdir()
    _init_git_repo_with_origin(repo_root, "https://github.com/dodo-labs-ai/mem-forge.git")
    plugin_root = tmp_path / "plugin-cache"
    plugin_root.mkdir()
    monkeypatch.chdir(plugin_root)

    class FakeResponse:
        headers = {"content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return b'{"status":"inserted","memory_id":"mem-new"}'

    class FakeOpener:
        def open(self, request, timeout):
            captured["body"] = request.data
            return FakeResponse()

    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())

    result = proxy._call_tool(
        "create_memory",
        {
            "content": "Use readable confirmation previews before memory mutations.",
            "provenance": "User confirmed this convention after reviewing the MemForge MCP UX.",
            "memory_type": "convention",
            "repository_context": {
                "working_directory": str(repo_root),
            },
        },
    )

    assert result == {"status": "inserted", "memory_id": "mem-new"}
    assert json.loads(captured["body"].decode()) == {
        "content": "Use readable confirmation previews before memory mutations.",
        "provenance": "User confirmed this convention after reviewing the MemForge MCP UX.",
        "memory_type": "convention",
        "client": "codex",
        "repo_identifier": "github.com/dodo-labs-ai/mem-forge",
    }


@pytest.mark.parametrize("client", ["codex", "claude-code"])
def test_packaged_mcp_proxy_forwards_create_memory_without_repo_roots(monkeypatch, client):
    proxy = _load_packaged_mcp_proxy(client, monkeypatch)
    captured = {}

    class FakeResponse:
        headers = {"content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return b'{"status":"inserted","memory_id":"mem-new"}'

    class FakeOpener:
        def open(self, request, timeout):
            captured["body"] = request.data
            return FakeResponse()

    monkeypatch.setenv("MEMFORGE_MCP_CLIENT", client)
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())

    initialize = proxy._handle_rpc_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": client},
            },
        }
    )
    assert initialize["result"]["serverInfo"]["name"] == "memforge"
    tools = proxy._handle_rpc_message({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    create_tool = next(tool for tool in tools["result"]["tools"] if tool["name"] == "create_memory")
    assert "repo_root" not in create_tool["inputSchema"]["properties"]
    assert set(create_tool["inputSchema"]["properties"]["repository_context"]["properties"]) == {"working_directory"}

    response = proxy._handle_rpc_message(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "create_memory",
                "arguments": {
                    "content": "Use readable confirmation previews before memory mutations.",
                    "provenance": "User confirmed this convention after reviewing the MemForge MCP UX.",
                    "memory_type": "convention",
                },
            },
        }
    )

    assert json.loads(response["result"]["content"][0]["text"]) == {
        "status": "inserted",
        "memory_id": "mem-new",
    }
    assert json.loads(captured["body"].decode()) == {
        "content": "Use readable confirmation previews before memory mutations.",
        "provenance": "User confirmed this convention after reviewing the MemForge MCP UX.",
        "memory_type": "convention",
        "client": client,
    }


@pytest.mark.parametrize("client", ["codex", "claude-code"])
def test_packaged_mcp_proxy_resolves_explicit_repository_context_locally(monkeypatch, tmp_path, client):
    proxy = _load_packaged_mcp_proxy(client, monkeypatch)
    captured = {}
    repo_root = tmp_path / "mem-forge"
    repo_root.mkdir()
    _init_git_repo_with_origin(repo_root, "https://github.com/dodo-labs-ai/mem-forge.git")

    class FakeResponse:
        headers = {"content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return b'{"status":"inserted","memory_id":"mem-new"}'

    class FakeOpener:
        def open(self, request, timeout):
            captured["body"] = request.data
            return FakeResponse()

    monkeypatch.setenv("MEMFORGE_MCP_CLIENT", client)
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())

    response = proxy._handle_rpc_message(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "create_memory",
                "arguments": {
                    "content": "Use one portable repository context contract.",
                    "provenance": "User confirmed the cross-client MCP design.",
                    "memory_type": "decision",
                    "repository_context": {
                        "working_directory": repo_root.as_uri(),
                    },
                },
            },
        }
    )

    assert json.loads(response["result"]["content"][0]["text"]) == {
        "status": "inserted",
        "memory_id": "mem-new",
    }
    assert json.loads(captured["body"].decode()) == {
        "content": "Use one portable repository context contract.",
        "provenance": "User confirmed the cross-client MCP design.",
        "memory_type": "decision",
        "client": client,
        "repo_identifier": "github.com/dodo-labs-ai/mem-forge",
    }


def test_mcp_proxy_forwards_create_memory_when_client_roots_request_fails(monkeypatch):
    proxy = _load_plugin_mcp_proxy()
    captured = {}
    initialize = proxy._handle_rpc_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {"roots": {"listChanged": True}},
                "clientInfo": {"name": "test"},
            },
        }
    )
    assert initialize["result"]["serverInfo"]["name"] == "memforge"
    roots_request = proxy._handle_rpc_message({"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert roots_request["method"] == "roots/list"
    assert (
        proxy._handle_rpc_message(
            {
                "jsonrpc": "2.0",
                "id": roots_request["id"],
                "error": {"code": -32601, "message": "roots/list rejected by host"},
            }
        )
        is None
    )

    class FakeResponse:
        headers = {"content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return b'{"status":"inserted","memory_id":"mem-new"}'

    class FakeOpener:
        def open(self, request, timeout):
            captured["body"] = request.data
            return FakeResponse()

    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())

    result = proxy._call_tool(
        "create_memory",
        {
            "content": "Use readable confirmation previews before memory mutations.",
            "provenance": "User confirmed this convention after reviewing the MemForge MCP UX.",
            "memory_type": "convention",
        },
    )

    assert result == {"status": "inserted", "memory_id": "mem-new"}
    assert json.loads(captured["body"].decode()) == {
        "content": "Use readable confirmation previews before memory mutations.",
        "provenance": "User confirmed this convention after reviewing the MemForge MCP UX.",
        "memory_type": "convention",
        "client": "codex",
    }


def test_mcp_proxy_rejects_create_memory_without_provenance(monkeypatch):
    proxy = _load_plugin_mcp_proxy()

    class FailOpener:
        def open(self, *_args, **_kwargs):
            raise AssertionError("create_memory should fail before posting without provenance")

    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FailOpener())
    monkeypatch.setattr(proxy, "_active_repo_identifier", lambda: "github.com/dodo-labs-ai/mem-forge")

    result = proxy._call_tool(
        "create_memory",
        {
            "content": "Use readable confirmation previews before memory mutations.",
            "memory_type": "convention",
        },
    )

    assert result == {"error": "provenance is required"}


def test_mcp_proxy_forwards_propose_memory_correction_to_lifecycle_endpoint(monkeypatch):
    proxy = _load_plugin_mcp_proxy()
    captured = {}

    class FakeResponse:
        headers = {"content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return b'{"status":"superseded","memory_id":"mem-old","replacement_memory_id":"mem-new"}'

    class FakeOpener:
        def open(self, request, timeout):
            captured["url"] = request.full_url
            captured["method"] = request.get_method()
            captured["body"] = request.data
            return FakeResponse()

    monkeypatch.setenv("MEMFORGE_API_URL", "https://self.example")
    monkeypatch.setenv("MEMFORGE_API_TOKEN", "token-123")
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())

    result = proxy._call_tool(
        "propose_memory_correction",
        {
            "memory_id": "mem-old",
            "replacement_content": "Use the new deployment route.",
            "provenance": "User corrected this while reviewing the deployment guide.",
            "reason": "User corrected the stale route.",
            "expected_content_hash": "hash-old",
            "replacement_kind": "revision",
        },
    )

    assert result["replacement_memory_id"] == "mem-new"
    assert captured["method"] == "POST"
    assert captured["url"] == "https://self.example/api/v1/memories/mem-old/corrections/propose"
    assert json.loads(captured["body"].decode()) == {
        "replacement_content": "Use the new deployment route.",
        "provenance": "User corrected this while reviewing the deployment guide.",
        "reason": "User corrected the stale route.",
        "expected_content_hash": "hash-old",
        "replacement_kind": "revision",
    }


def test_mcp_proxy_rejects_propose_memory_correction_without_provenance(monkeypatch):
    proxy = _load_plugin_mcp_proxy()

    class FailOpener:
        def open(self, *_args, **_kwargs):
            raise AssertionError("propose_memory_correction should fail before posting without provenance")

    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FailOpener())

    result = proxy._call_tool(
        "propose_memory_correction",
        {
            "memory_id": "mem-old",
            "replacement_content": "Use the new deployment route.",
            "reason": "User corrected the stale route.",
            "expected_content_hash": "hash-old",
            "replacement_kind": "revision",
        },
    )

    assert result == {"error": "provenance is required"}


def test_mcp_proxy_forwards_memory_review_tools(monkeypatch):
    proxy = _load_plugin_mcp_proxy()
    calls = []

    class FakeResponse:
        headers = {"content-type": "application/json"}

        def __init__(self, body: bytes = b'{"ok":true}'):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return self.body

    class FakeOpener:
        def open(self, request, timeout):
            calls.append(
                {
                    "url": request.full_url,
                    "method": request.get_method(),
                    "body": request.data,
                }
            )
            return FakeResponse()

    monkeypatch.setenv("MEMFORGE_API_URL", "https://self.example")
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())

    assert proxy._call_tool("list_memory_reviews", {"status": "open", "limit": 5}) == {"ok": True}
    assert proxy._call_tool("get_memory_review", {"review_id": "rev-1"}) == {"ok": True}
    assert proxy._call_tool(
        "refresh_memory_review",
        {
            "review_id": "review-stale-1",
            "expected_fingerprint": "review-decision-v1:stale",
        },
    ) == {"error": "Unknown tool: refresh_memory_review"}
    assert proxy._call_tool(
        "resolve_memory_review",
        {
            "review_id": "rev-1",
            "decision": "reject",
            "expected_fingerprint": "review-decision-v1:abc",
            "note": "Not durable enough.",
        },
    ) == {"ok": True}

    assert calls[0]["method"] == "GET"
    assert calls[0]["url"] == "https://self.example/api/v1/memory-reviews?status=open&limit=5&offset=0"
    assert calls[1]["method"] == "GET"
    assert calls[1]["url"] == "https://self.example/api/v1/memory-reviews/rev-1"
    assert calls[2]["method"] == "POST"
    assert calls[2]["url"] == "https://self.example/api/v1/memory-reviews/rev-1/reject"
    assert json.loads(calls[2]["body"].decode()) == {
        "expected_fingerprint": "review-decision-v1:abc",
        "note": "Not durable enough.",
    }


def test_mcp_proxy_requires_note_before_rejecting_memory_review(monkeypatch):
    proxy = _load_plugin_mcp_proxy()

    class FakeOpener:
        def open(self, request, timeout):
            raise AssertionError("reject without note should fail before HTTP")

    monkeypatch.setenv("MEMFORGE_API_URL", "https://self.example")
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())

    assert proxy._call_tool(
        "resolve_memory_review",
        {
            "review_id": "rev-1",
            "decision": "reject",
            "expected_fingerprint": "review-decision-v1:abc",
        },
    ) == {"error": "note is required when decision is reject"}


def test_mcp_proxy_validates_and_applies_bounded_review_manifests(monkeypatch):
    proxy = _load_plugin_mcp_proxy()
    calls = []

    class FakeResponse:
        headers = {"content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return b'{"mode":"validate","results":[]}'

    class FakeOpener:
        def open(self, request, timeout):
            calls.append((request.get_method(), request.full_url, json.loads(request.data.decode())))
            return FakeResponse()

    monkeypatch.setenv("MEMFORGE_API_URL", "https://self.example")
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())
    decisions = [
        {
            "review_id": "rev-1",
            "decision": "approve",
            "expected_fingerprint": "review-decision-v1:abc",
            "rationale": "Same subject and deployment.",
            "confidence": 0.92,
            "risk": "low",
        }
    ]

    assert proxy._call_tool("validate_memory_review_decisions", {"decisions": decisions})["mode"] == "validate"
    assert (
        proxy._call_tool(
            "apply_memory_review_decisions",
            {"decisions": decisions, "validation_receipt": "review-manifest-v1:receipt"},
        )["mode"]
        == "validate"
    )
    assert calls == [
        (
            "POST",
            "https://self.example/api/v1/memory-reviews/decisions/validate",
            {"decisions": decisions},
        ),
        (
            "POST",
            "https://self.example/api/v1/memory-reviews/decisions/apply",
            {"decisions": decisions, "validation_receipt": "review-manifest-v1:receipt"},
        ),
    ]

    duplicate = {"decisions": [decisions[0], decisions[0]]}
    assert proxy._call_tool("validate_memory_review_decisions", duplicate) == {"error": "duplicate review_id: rev-1"}
    assert proxy._call_tool("apply_memory_review_decisions", {"decisions": decisions}) == {
        "error": "validation_receipt is required before apply"
    }


def test_mcp_proxy_returns_review_manifest_as_structured_content(monkeypatch):
    proxy = _load_plugin_mcp_proxy()
    payload = {
        "mode": "validate",
        "results": [
            {
                "review_id": "rev-1",
                "decision": "approve",
                "outcome": "ready",
                "status": "pending",
                "message": None,
                "decision_label": "Conflict",
                "consequence": "Keep both memories active.",
            }
        ],
        "ready": 1,
        "applied": 0,
        "already_applied": 0,
        "stale": 0,
        "forbidden": 0,
        "failed": 0,
        "validation_receipt": "review-manifest-v1:receipt",
    }
    monkeypatch.setattr(proxy, "_call_tool", lambda *_args, **_kwargs: payload)

    response = proxy._tool_call_response(
        7,
        "validate_memory_review_decisions",
        {"decisions": []},
    )

    assert response["result"]["structuredContent"] == payload
    assert json.loads(response["result"]["content"][0]["text"]) == payload


def test_mcp_proxy_rejects_invalid_memory_review_pagination():
    proxy = _load_plugin_mcp_proxy()

    assert proxy._call_tool("list_memory_reviews", {"limit": "ten"}) == {"error": "limit must be an integer"}
    assert proxy._call_tool("list_memory_reviews", {"offset": "soon"}) == {"error": "offset must be an integer"}


def test_mcp_proxy_rejects_queryless_source_id_time_range(monkeypatch):
    proxy = _load_plugin_mcp_proxy()
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
            captured["body"] = request.data
            return FakeResponse()

    monkeypatch.setenv("MEMFORGE_API_URL", "https://memforge.example.com")
    monkeypatch.setenv("MEMFORGE_API_TOKEN", "token-123")
    monkeypatch.setenv("MEMFORGE_WORKSPACE_ID", "team_alpha")
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())
    monkeypatch.setattr(proxy, "_active_repo_identifier", lambda: None)

    result = proxy._call_tool(
        "search",
        {
            "source_filter": {"source_ids": ["src-alpha"]},
            "time_range": {
                "date_type": "source_updated_at",
                "start_date": "2026-06-20",
                "end_date": "2026-06-26",
            },
        },
    )

    assert result == {
        "error": "search.query is required; use list_recent_memories for deterministic listings",
    }
    assert "body" not in captured


def test_mcp_proxy_rejects_queryless_offset_compatibility(monkeypatch):
    proxy = _load_plugin_mcp_proxy()
    captured = {}

    class FakeResponse:
        headers = {"content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return b'{"results":[],"total_candidates":52,"candidate_count_kind":"exact","limit":10,"offset":10}'

    class FakeOpener:
        def open(self, request, timeout):
            captured["body"] = request.data
            return FakeResponse()

    monkeypatch.setenv("MEMFORGE_API_URL", "https://memforge.example.com")
    monkeypatch.setenv("MEMFORGE_API_TOKEN", "token-123")
    monkeypatch.setenv("MEMFORGE_WORKSPACE_ID", "team_alpha")
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())
    monkeypatch.setattr(proxy, "_active_repo_identifier", lambda: None)

    result = proxy._call_tool(
        "search",
        {
            "source_filter": {"source_ids": ["src-backlog"]},
            "time_range": {"date_type": "source_updated_at", "start_date": "2026-06-29"},
            "top_k": 10,
            "offset": 10,
        },
    )

    assert result == {
        "error": "search.query is required; use list_recent_memories for deterministic listings",
    }
    assert "body" not in captured


def test_mcp_proxy_does_not_guess_has_more_for_windowed_legacy_response(monkeypatch):
    proxy = _load_plugin_mcp_proxy()

    class FakeResponse:
        headers = {"content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return b'{"results":[],"total_candidates":61,"candidate_count_kind":"windowed","limit":10,"offset":0}'

    class FakeOpener:
        def open(self, request, timeout):
            return FakeResponse()

    monkeypatch.setenv("MEMFORGE_API_URL", "https://self.example")
    monkeypatch.setenv("MEMFORGE_API_TOKEN", "token-123")
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())
    monkeypatch.setattr(proxy, "_active_repo_identifier", lambda: None)

    result = proxy._call_tool("search", {"query": "create blocker hint"})

    assert result == {
        "results": [],
        "total_candidates": 61,
        "candidate_count_kind": "windowed",
        "limit": 10,
        "offset": 0,
    }


def test_mcp_proxy_compacts_search_response_for_agent_context(monkeypatch):
    proxy = _load_plugin_mcp_proxy()

    class FakeResponse:
        headers = {"content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return json.dumps(
                {
                    "query_analysis": {
                        "detected_entities": ["periodcutoffblockerhint"],
                        "entity_linking": [{"entity_id": 95, "matched_alias": "blocker hint"}],
                        "strategies_used": ["vector", "bm25_metadata_tokens", "graph"],
                    },
                    "retrieval_intent": {
                        "requested_intent": "relationship",
                        "resolved_intent": "general_hybrid",
                        "intent_source": "fallback",
                        "fallback_reason": "no_traversable_entity",
                        "internal": "omit",
                    },
                    "results": [
                        {
                            "memory_id": "mem-1",
                            "memory_type": "fact",
                            "summary": "Create blocker hint task details",
                            "relevance_score": 0.99,
                            "freshness": "current",
                            "status": "active",
                            "follow_up": {"suggested_tool": "get_memory"},
                            "retrieval_evidence": {"metadata_lexical": {"matched_text": ["large debug text"]}},
                            "repo_identifier": "repo",
                        }
                    ],
                    "total_candidates": 61,
                    "candidate_count_kind": "windowed",
                    "ranking_window_size": 50,
                    "limit": 10,
                    "offset": 0,
                    "has_more": True,
                    "retrieval_time_ms": 123,
                }
            ).encode()

    class FakeOpener:
        def open(self, request, timeout):
            return FakeResponse()

    monkeypatch.setenv("MEMFORGE_API_URL", "https://self.example")
    monkeypatch.setenv("MEMFORGE_API_TOKEN", "token-123")
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())
    monkeypatch.setattr(proxy, "_active_repo_identifier", lambda: None)

    result = proxy._call_tool("search", {"query": "create blocker hint"})

    assert result == {
        "results": [
            {
                "memory_id": "mem-1",
                "memory_type": "fact",
                "summary": "Create blocker hint task details",
                "relevance_score": 0.99,
                "freshness": "current",
                "status": "active",
                "follow_up": {"suggested_tool": "get_memory"},
            }
        ],
        "retrieval_intent": {
            "requested_intent": "relationship",
            "resolved_intent": "general_hybrid",
            "intent_source": "fallback",
            "fallback_reason": "no_traversable_entity",
        },
        "total_candidates": 61,
        "candidate_count_kind": "windowed",
        "ranking_window_size": 50,
        "limit": 10,
        "offset": 0,
        "has_more": True,
    }


def test_mcp_proxy_compacts_get_memory_response_for_agent_context(monkeypatch):
    proxy = _load_plugin_mcp_proxy()

    class FakeResponse:
        headers = {"content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return json.dumps(
                {
                    "id": "mem-1",
                    "memory_type": "procedure",
                    "content": "Persist blocker hints.",
                    "content_hash": "sha256-secret",
                    "visibility": "workspace",
                    "owner_user_id": "user-1",
                    "project_key": "PAY",
                    "corroboration_count": 1,
                    "status": "active",
                    "retirement_reason": None,
                    "replacement_reason": None,
                    "entity_refs": ["periodcutoffblockerhint"],
                    "evidence": [
                        {
                            "kind": "document",
                            "evidence_unit_id": None,
                            "support_ids": [],
                            "source_id": "src-jira",
                            "source_type": "jira",
                            "doc_id": "jira-PAYR-179397",
                            "current": True,
                            "document": {
                                "doc_id": "jira-PAYR-179397",
                                "title": "PAYR-179397: Create Blocker Hint",
                                "source_url": "https://jira.example/browse/PAYR-179397",
                                "content_url": "/api/v1/documents/jira-PAYR-179397/content",
                                "pdf_url": None,
                                "source_updated_at": "2026-07-02T03:50:32+00:00",
                            },
                            "items": [
                                {
                                    "authority": "application_document",
                                    "role": "primary",
                                    "kind": "text",
                                    "support_contribution": True,
                                    "excerpt": "large excerpt",
                                    "current": True,
                                }
                            ],
                        }
                    ],
                }
            ).encode()

    class FakeOpener:
        def open(self, request, timeout):
            return FakeResponse()

    monkeypatch.setenv("MEMFORGE_API_URL", "https://self.example")
    monkeypatch.setenv("MEMFORGE_API_TOKEN", "token-123")
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())

    result = proxy._call_tool("get_memory", {"memory_id": "mem-1"})

    assert result == {
        "id": "mem-1",
        "memory_type": "procedure",
        "content": "Persist blocker hints.",
        "content_hash": "sha256-secret",
        "status": "active",
        "entity_refs": ["periodcutoffblockerhint"],
        "evidence": [
            {
                "source_id": "src-jira",
                "source_type": "jira",
                "current": True,
                "document": {
                    "title": "PAYR-179397: Create Blocker Hint",
                    "source_url": "https://jira.example/browse/PAYR-179397",
                    "content_url": "/api/v1/documents/jira-PAYR-179397/content",
                    "pdf_url": None,
                    "source_updated_at": "2026-07-02T03:50:32+00:00",
                },
                "items": [{"role": "primary", "kind": "text", "current": True, "text": "large excerpt"}],
            }
        ],
    }


@pytest.mark.parametrize("offset", [-1, "10", True])
def test_mcp_proxy_rejects_invalid_search_offset(monkeypatch, offset):
    proxy = _load_plugin_mcp_proxy()
    monkeypatch.setattr(proxy, "_active_repo_identifier", lambda: None)

    result = proxy._call_tool(
        "search",
        {
            "source_filter": {"source_ids": ["src-backlog"]},
            "offset": offset,
        },
    )

    assert result == {"error": "offset must be a non-negative integer"}


@pytest.mark.parametrize("top_k", [0, 51, "10", True])
def test_mcp_proxy_rejects_invalid_search_top_k(monkeypatch, top_k):
    proxy = _load_plugin_mcp_proxy()
    monkeypatch.setattr(proxy, "_active_repo_identifier", lambda: None)

    result = proxy._call_tool(
        "search",
        {
            "query": "payroll",
            "top_k": top_k,
        },
    )

    assert result == {"error": "top_k must be an integer from 1 to 50"}


def test_mcp_proxy_rejects_queryless_search_even_with_deterministic_filter(monkeypatch):
    proxy = _load_plugin_mcp_proxy()
    monkeypatch.setattr(proxy, "_active_repo_identifier", lambda: None)

    missing = proxy._call_tool("search", {})
    filtered = proxy._call_tool(
        "search",
        {"source_filter": {"source_ids": ["src-backlog"]}},
    )

    assert missing == {"error": "search.query is required"}
    assert filtered == {
        "error": "search.query is required; use list_recent_memories for deterministic listings",
    }


def test_mcp_proxy_rejects_unknown_search_intent_before_http(monkeypatch):
    proxy = _load_plugin_mcp_proxy()
    monkeypatch.setattr(proxy, "_active_repo_identifier", lambda: None)
    monkeypatch.setattr(
        proxy,
        "_http_json",
        lambda *args, **kwargs: pytest.fail("HTTP must not run"),
    )

    result = proxy._call_tool(
        "search",
        {"query": "payroll lock", "intent": "semantic_lookup"},
    )

    assert result == {
        "error": "intent must be general_hybrid, known_item, or relationship",
    }


def test_mcp_proxy_forwards_explicit_optional_date_bounds(monkeypatch):
    proxy = _load_plugin_mcp_proxy()
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
            captured["body"] = request.data
            return FakeResponse()

    monkeypatch.setenv("MEMFORGE_API_URL", "https://memforge.example.com")
    monkeypatch.setenv("MEMFORGE_API_TOKEN", "token-123")
    monkeypatch.setenv("MEMFORGE_WORKSPACE_ID", "team_alpha")
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())
    monkeypatch.setattr(proxy, "_active_repo_identifier", lambda: None)

    result = proxy._call_tool(
        "search",
        {
            "query": "jira memories",
            "time_range": {
                "date_type": "source_updated_at",
                "start_date": "2026-06-19",
            },
        },
    )

    assert result == {"results": []}
    assert json.loads(captured["body"].decode())["time_range"] == {
        "date_type": "source_updated_at",
        "start_date": "2026-06-19",
    }


@pytest.mark.parametrize(
    "time_range",
    [
        {},
        {"start_date": "2026-06-20T00:00:00Z"},
        {"after": "2026-06-20"},
        {"date_type": "created_at", "start_date": "2026-06-20"},
        {"start_date": "2026-06-21", "end_date": "2026-06-20"},
    ],
)
def test_mcp_proxy_rejects_invalid_time_range_shapes(monkeypatch, time_range):
    proxy = _load_plugin_mcp_proxy()
    monkeypatch.setattr(proxy, "_active_repo_identifier", lambda: None)

    result = proxy._call_tool("search", {"query": "jira memories", "time_range": time_range})

    assert "error" in result
    assert "time_range" in result["error"]


def test_mcp_proxy_rejects_empty_source_ids(monkeypatch):
    proxy = _load_plugin_mcp_proxy()
    monkeypatch.setattr(proxy, "_active_repo_identifier", lambda: None)

    result = proxy._call_tool(
        "search",
        {
            "source_filter": {"source_ids": []},
            "time_range": {"start_date": "2026-06-20"},
        },
    )

    assert result == {"error": "source_filter.source_ids must be a non-empty array of source IDs from list_sources"}


def test_mcp_proxy_rejects_removed_backend_search_knobs(monkeypatch):
    proxy = _load_plugin_mcp_proxy()
    monkeypatch.setattr(proxy, "_active_repo_identifier", lambda: None)

    result = proxy._call_tool(
        "search",
        {
            "query": "private memories",
            "include_private": True,
            "status": "active",
            "memory_types": ["decision"],
        },
    )

    assert result == {
        "error": (
            "Unsupported search parameter(s): include_private, memory_types, status. "
            "Omit unknown filters instead of guessing."
        )
    }


def test_mcp_proxy_rejects_unadvertised_source_filter_facets(monkeypatch):
    proxy = _load_plugin_mcp_proxy()
    monkeypatch.setattr(proxy, "_active_repo_identifier", lambda: None)

    result = proxy._call_tool(
        "search",
        {
            "query": "scheduler fix",
            "source_filter": {
                "source_types": ["agent_session"],
                "repo_identifiers": ["github.example.com/acme/other"],
            },
        },
    )

    assert result == {
        "error": (
            "Unsupported source_filter parameter(s): repo_identifiers, source_types. "
            "Omit repo-scoped facets instead of guessing."
        )
    }


def test_mcp_proxy_fetches_resource_through_hosted_workspace(monkeypatch):
    proxy = _load_plugin_mcp_proxy()
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

    monkeypatch.setenv("MEMFORGE_API_URL", "https://memforge.example.com")
    monkeypatch.setenv("MEMFORGE_API_TOKEN", "token-123")
    monkeypatch.setenv("MEMFORGE_WORKSPACE_ID", "team_alpha")
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())

    result = proxy._call_tool(
        "get_resource",
        {"url": "/api/v1/documents/doc-1/content", "mode": "text"},
    )

    assert result["text"] == "# Source"
    assert result["url"] == "/api/v1/documents/doc-1/content"
    assert captured["url"] == "https://memforge.example.com/api/v1/documents/doc-1/content"
    assert captured["authorization"] == "Bearer token-123"


def test_mcp_proxy_keeps_what_an_agent_reads_of_artifact_evidence_on_get_memory(monkeypatch):
    proxy = _load_plugin_mcp_proxy()
    artifact = {
        "artifact_id": "artifact-1",
        "parent_observation_id": "obs-parent",
        "filename": "diagram.png",
        "content_type": "image/png",
        "size_bytes": 3,
        "url": "/api/v1/source-artifacts/obsrev-1",
    }
    monkeypatch.setattr(
        proxy,
        "_http_json",
        lambda *_args, **_kwargs: {
            "id": "mem-1",
            "memory_type": "fact",
            "content": "A visual claim",
            "evidence": [
                {
                    "kind": "evidence_unit",
                    "evidence_unit_id": "evidence-1",
                    "source_id": "src-github",
                    "source_type": "github",
                    "doc_id": "doc-1",
                    "current": True,
                    "document": {
                        "doc_id": "doc-1",
                        "title": "Architecture",
                        "content_url": "/api/v1/documents/doc-1/content",
                    },
                    "items": [
                        {
                            "authority": "revision_pinned",
                            "role": "primary",
                            "kind": "artifact",
                            "support_contribution": True,
                            "current": True,
                            "artifact": artifact,
                        }
                    ],
                }
            ],
        },
    )

    result = proxy._call_tool("get_memory", {"memory_id": "mem-1"})

    [group] = result["evidence"]
    assert group["document"]["content_url"] == "/api/v1/documents/doc-1/content"
    assert group["items"][0] == {
        "role": "primary",
        "kind": "artifact",
        "current": True,
        "artifact": {
            "filename": "diagram.png",
            "content_type": "image/png",
            "size_bytes": 3,
            "url": "/api/v1/source-artifacts/obsrev-1",
        },
    }


def test_mcp_proxy_returns_source_artifact_as_native_image_content(monkeypatch):
    proxy = _load_plugin_mcp_proxy()
    body = b"\x89PNG"

    class FakeResponse:
        headers = {
            "content-type": "image/png",
            "content-disposition": 'inline; filename="diagram.png"',
            "content-length": str(len(body)),
            "x-content-sha256": "0f4636c78f65d3639ece5a064b5ae753e3408614a14fb18ab4d7540d2c248543",
        }

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return body

    class FakeOpener:
        def open(self, request, timeout):
            assert request.full_url == "http://memforge.test/api/v1/source-artifacts/obsrev-1"
            return FakeResponse()

    monkeypatch.setenv("MEMFORGE_API_URL", "http://memforge.test")
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())

    response = proxy._handle_rpc_message(
        {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {
                "name": "get_resource",
                "arguments": {
                    "url": "/api/v1/source-artifacts/obsrev-1",
                    "mode": "base64",
                },
            },
        }
    )

    assert response["result"]["content"][1] == {
        "type": "image",
        "data": base64.b64encode(body).decode("ascii"),
        "mimeType": "image/png",
    }
    metadata = json.loads(response["result"]["content"][0]["text"])
    assert metadata["url"] == "/api/v1/source-artifacts/obsrev-1"
    assert metadata["sha256"] == "0f4636c78f65d3639ece5a064b5ae753e3408614a14fb18ab4d7540d2c248543"
    assert "data_base64" not in metadata


def test_mcp_proxy_downloads_resource_to_local_cache(monkeypatch, tmp_path):
    proxy = _load_plugin_mcp_proxy()
    captured = {}
    body = b"%PDF-1.4\n%local-cache\n"

    class FakeResponse:
        headers = {
            "content-type": "application/pdf",
            "content-disposition": 'attachment; filename="source.pdf"',
            "content-length": str(len(body)),
            "x-content-sha256": hashlib.sha256(body).hexdigest(),
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
            captured["url"] = request.full_url
            captured["timeout"] = timeout
            return FakeResponse()

    monkeypatch.setenv("MEMFORGE_API_URL", "http://memforge.test")
    monkeypatch.setenv("MEMFORGE_ARTIFACT_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())

    result = proxy._handle_get_resource(
        {
            "url": "/api/v1/documents/doc-123/pdf",
            "mode": "file",
        }
    )

    local_path = Path(result["local_path"])
    assert captured["url"] == "http://memforge.test/api/v1/documents/doc-123/pdf"
    assert result["mode"] == "file"
    assert result["content_type"] == "application/pdf"
    assert result["size_bytes"] == len(body)
    assert local_path.parent == tmp_path
    assert local_path.read_bytes() == body


def test_mcp_proxy_file_mode_rejects_truncated_resource(monkeypatch, tmp_path):
    proxy = _load_plugin_mcp_proxy()
    body = b"truncated"

    class FakeResponse:
        headers = {
            "content-type": "image/png",
            "content-length": str(len(body) + 1),
            "x-content-sha256": hashlib.sha256(body).hexdigest(),
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

    monkeypatch.setenv("MEMFORGE_API_URL", "http://memforge.test")
    monkeypatch.setenv("MEMFORGE_ARTIFACT_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())

    result = proxy._handle_get_resource(
        {
            "url": "/api/v1/source-artifacts/obsrev-1",
            "mode": "file",
        }
    )

    assert result["error"] == "resource fetch failed"
    assert result["detail"] == "resource byte count does not match Content-Length"
    assert list(tmp_path.iterdir()) == []


def test_mcp_proxy_rejects_foreign_and_ambiguous_resource_urls(monkeypatch):
    proxy = _load_plugin_mcp_proxy()
    monkeypatch.setenv("MEMFORGE_API_URL", "https://self.example")

    foreign = proxy._handle_get_resource(
        {
            "url": "https://evil.example/api/documents/doc-123/pdf",
            "mode": "file",
        }
    )
    encoded_slash = proxy._handle_get_resource(
        {
            "url": "/api/v1/documents/doc%2F123/pdf",
            "mode": "file",
        }
    )

    assert foreign["error"] == "unsupported resource URL"
    assert encoded_slash["error"] == "unsupported resource URL"


def _load_plugin_mcp_proxy():
    root = Path(__file__).resolve().parents[1]
    proxy_path = root / "src" / "memforge" / "plugin_mcp_proxy.py"
    spec = importlib.util.spec_from_file_location("memforge_mcp_test", proxy_path)
    assert spec is not None
    assert spec.loader is not None
    proxy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(proxy)
    return proxy


def _load_packaged_mcp_proxy(client: str, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    proxy_path = root / "integrations" / client / "memforge-memory" / "scripts" / "memforge_mcp.py"
    for name in (
        "memforge_api_target",
        "memforge_capability_discovery",
        "memforge_plugin_config",
        "memforge_repo_identity",
    ):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.syspath_prepend(str(proxy_path.parent))
    module_name = f"memforge_mcp_{client.replace('-', '_')}_test"
    spec = importlib.util.spec_from_file_location(module_name, proxy_path)
    assert spec is not None
    assert spec.loader is not None
    proxy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(proxy)
    return proxy


def _provide_mcp_roots(proxy, *roots: Path) -> None:
    initialize = proxy._handle_rpc_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {"roots": {"listChanged": True}},
                "clientInfo": {"name": "test"},
            },
        }
    )
    assert initialize["result"]["serverInfo"]["name"] == "memforge"
    roots_request = proxy._handle_rpc_message(
        {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        }
    )
    assert roots_request["method"] == "roots/list"
    assert (
        proxy._handle_rpc_message(
            {
                "jsonrpc": "2.0",
                "id": roots_request["id"],
                "result": {"roots": [{"uri": root.as_uri(), "name": root.name} for root in roots]},
            }
        )
        is None
    )


class _Stdin:
    def __init__(self, payload: dict) -> None:
        self._text = json.dumps(payload)

    def read(self) -> str:
        return self._text

