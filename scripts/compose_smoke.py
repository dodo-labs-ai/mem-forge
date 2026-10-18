"""HTTP, MCP and Docker checks for ``scripts/compose_smoke.sh``.

The shell script owns the Compose project: it builds, starts, restarts and
removes the stack. This helper makes the checks against the running stack, one
subcommand per check, and prints one line saying what it saw. That line goes
into the smoke's report; the exit status is 0 when the check passed.

It uses the standard library only and runs on the ``python3`` the plugins'
MCP proxy runs on (3.9 or newer), so it needs no checkout environment. Every
request goes straight to the smoke stack on loopback, never through a proxy
set for the containers. API keys are read from the environment only and are
sent in request bodies, never printed.

What one run learns, such as the ids of the memory and the source it created,
is kept in a JSON state file so the checks after the restart can look for them.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import select
import socket
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable, Iterable, Sequence
from typing import Any

# The memory tools an agent needs from the packaged MCP proxy.
EXPECTED_MCP_TOOLS = frozenset(
    {
        "list_workspaces",
        "search",
        "list_recent_memories",
        "list_sources",
        "get_memory",
        "get_resource",
        "create_memory",
        "propose_memory_correction",
        "retire_memory",
    }
)
# Environment variables that would send the MCP proxy's loopback requests to a proxy.
PROXY_VARIABLES = ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy")
LOOPBACK = "127.0.0.1"
# Token limits for a generation model whose listing gives none, as the
# quickstart suggests for a local model.
DEFAULT_CONTEXT_WINDOW_TOKENS = 32768
DEFAULT_MAX_OUTPUT_TOKENS = 4096
SOURCE_NAME = "Compose smoke notes"
SOURCE_QUERY = "Which storage does the Kestrel job queue use?"
FINISHED_SYNC_STATES = frozenset({"success", "partial"})


class CheckFailed(Exception):
    """A check saw something other than what it expects."""


# -- State --------------------------------------------------------------------------


class State:
    """What earlier checks of this run recorded, kept in a JSON file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}

    def get(self, key: str) -> Any:
        return self.data.get(key)

    def set(self, **values: Any) -> None:
        self.data.update(values)
        self.path.write_text(json.dumps(self.data, indent=2, sort_keys=True))


# -- HTTP ---------------------------------------------------------------------------

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


@dataclass(frozen=True)
class Response:
    status: int
    content_type: str
    body: bytes

    def json(self) -> Any:
        return json.loads(self.body.decode("utf-8"))


def request(method: str, url: str, payload: Any = None, *, timeout: float = 30.0) -> Response:
    """Send one request without any proxy; an HTTP error status is returned, not raised."""
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Accept", "application/json, text/html;q=0.9")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with _OPENER.open(req, timeout=timeout) as response:
            return Response(response.status, response.headers.get("Content-Type", ""), response.read())
    except urllib.error.HTTPError as exc:
        return Response(exc.code, exc.headers.get("Content-Type", ""), exc.read())
    except (urllib.error.URLError, OSError) as exc:
        raise CheckFailed(f"{method} {url} did not answer: {exc}") from exc


def api_json(api: str, method: str, path: str, payload: Any = None, *, expect: Iterable[int] = (200,)) -> Any:
    response = request(method, f"{api}{path}", payload)
    if response.status not in tuple(expect):
        raise CheckFailed(f"{method} {path} answered {response.status}: {_excerpt(response.body)}")
    return response.json()


def _excerpt(body: bytes, limit: int = 300) -> str:
    text = body.decode("utf-8", errors="replace").strip().replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "..."


def wait_until(check: Callable[[], str | None], *, timeout: float, interval: float, what: str) -> str:
    """Call ``check`` until it returns a description; it raises CheckFailed to give up early."""
    deadline = time.monotonic() + timeout
    while True:
        outcome = check()
        if outcome is not None:
            return outcome
        if time.monotonic() >= deadline:
            raise CheckFailed(f"{what} did not happen within {timeout:g}s")
        time.sleep(interval)


# -- Ports and bindings ---------------------------------------------------------------


def free_ports(count: int) -> list[int]:
    """Loopback ports nothing listens on now, for the smoke stack to publish."""
    sockets = []
    try:
        for _ in range(count):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind((LOOPBACK, 0))
            sockets.append(sock)
        return [sock.getsockname()[1] for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


def published_bindings(inspected: Sequence[dict[str, Any]]) -> dict[str, list[str]]:
    """Map each Compose service to the ``host_ip:host_port`` addresses it publishes."""
    bindings: dict[str, list[str]] = {}
    for container in inspected:
        service = container.get("Config", {}).get("Labels", {}).get("com.docker.compose.service", "?")
        ports = container.get("NetworkSettings", {}).get("Ports") or {}
        for published in ports.values():
            for binding in published or []:
                bindings.setdefault(service, []).append(f"{binding.get('HostIp', '')}:{binding.get('HostPort', '')}")
    return bindings


def check_bindings(inspected: Sequence[dict[str, Any]], expected: dict[str, int]) -> str:
    """Every published port binds to loopback only, and each service publishes the port it was given."""
    bindings = published_bindings(inspected)
    exposed = [
        f"{service} {address}"
        for service, addresses in bindings.items()
        for address in addresses
        if not address.startswith(f"{LOOPBACK}:")
    ]
    if exposed:
        raise CheckFailed("published beyond loopback: " + ", ".join(sorted(exposed)))
    missing = [
        f"{service} {LOOPBACK}:{port}"
        for service, port in expected.items()
        if f"{LOOPBACK}:{port}" not in bindings.get(service, [])
    ]
    if missing:
        raise CheckFailed("not published: " + ", ".join(missing))
    return "; ".join(f"{service} {', '.join(sorted(bindings[service]))}" for service in sorted(bindings))


# -- Stack ----------------------------------------------------------------------------


def probe(api: str, ui: str) -> str:
    health = api_json(api, "GET", "/api/v1/health")
    database = (health.get("database") or {}).get("status")
    if database != "ok":
        raise CheckFailed(f"health reports the database as {database!r}")
    pages = []
    for path in ("/", "/v2/"):
        response = request("GET", f"{ui}{path}")
        if response.status != 200 or "text/html" not in response.content_type:
            raise CheckFailed(f"admin UI {path} answered {response.status} {response.content_type!r}")
        if b"<html" not in response.body.lower():
            raise CheckFailed(f"admin UI {path} is not an HTML page")
        pages.append(path)
    proxied = api_json(ui, "GET", "/api/v1/health")
    if (proxied.get("database") or {}).get("status") != "ok":
        raise CheckFailed("the admin UI does not proxy /api to the API")
    return f"health {health.get('status')}; admin UI serves {' and '.join(pages)} and proxies /api"


# -- Packaged MCP proxy ----------------------------------------------------------------


class McpSession:
    """The plugin's packaged MCP proxy, started as the plugin starts it, over newline-delimited stdio."""

    def __init__(self, plugin_dir: Path, api: str, home: Path, *, timeout: float = 60.0) -> None:
        env = {key: value for key, value in os.environ.items() if key not in PROXY_VARIABLES}
        env.update(
            {
                "HOME": str(home),
                "CLAUDE_PLUGIN_ROOT": str(plugin_dir),
                "MEMFORGE_API_URL": api,
                "MEMFORGE_MCP_CLIENT": "claude-code",
                "MEMFORGE_ARTIFACT_CACHE_DIR": str(home / "artifacts"),
                "NO_PROXY": "*",
            }
        )
        env.pop("MEMFORGE_API_TOKEN", None)
        home.mkdir(parents=True, exist_ok=True)
        self._timeout = timeout
        self._next_id = 0
        self._process = subprocess.Popen(
            [sys.executable, "scripts/memforge_mcp.py"],
            cwd=plugin_dir,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def __enter__(self) -> McpSession:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._process.poll() is None:
            assert self._process.stdin is not None
            self._process.stdin.close()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()

    def _send(self, message: dict[str, Any]) -> None:
        assert self._process.stdin is not None
        self._process.stdin.write(json.dumps(message).encode("utf-8") + b"\n")
        self._process.stdin.flush()

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, **({"params": params} if params else {})})

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._next_id += 1
        request_id = self._next_id
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}})
        stdout = self._process.stdout
        assert stdout is not None
        deadline = time.monotonic() + self._timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([stdout], [], [], remaining)[0]:
                raise CheckFailed(f"MCP {method} got no answer within {self._timeout:g}s")
            line = stdout.readline()
            if not line:
                stderr = self._process.stderr.read().decode("utf-8", errors="replace") if self._process.stderr else ""
                raise CheckFailed(f"MCP proxy exited during {method}: {stderr.strip()[-300:]}")
            message = json.loads(line)
            if message.get("id") != request_id:
                continue
            if "error" in message:
                raise CheckFailed(f"MCP {method} failed: {message['error']}")
            return message["result"]

    def tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        result = self.call("tools/call", {"name": name, "arguments": arguments or {}})
        payload = json.loads(result["content"][0]["text"])
        if result.get("isError"):
            raise CheckFailed(f"MCP {name} returned an error: {payload.get('error', payload)}")
        return payload

    def start(self) -> dict[str, Any]:
        initialized = self.call(
            "initialize",
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "memforge-compose-smoke", "version": "1"},
            },
        )
        self.notify("notifications/initialized")
        return initialized


def _open_mcp(session: McpSession) -> str:
    """Initialize the proxy, check its tools and reach the API through it."""
    server = session.start().get("serverInfo", {})
    tools = {tool["name"] for tool in session.call("tools/list").get("tools", [])}
    missing = sorted(EXPECTED_MCP_TOOLS - tools)
    if missing:
        raise CheckFailed("tools/list lacks " + ", ".join(missing))
    workspaces = session.tool("list_workspaces")
    workspace_ids = [str(item.get("workspace_id")) for item in workspaces.get("workspaces", [])]
    if not workspace_ids:
        raise CheckFailed(f"list_workspaces returned no workspace: {workspaces}")
    return f"{server.get('name')} {server.get('version')}: {len(tools)} tools, workspace {', '.join(workspace_ids)}"


def _write_memory(session: McpSession, state: State) -> str:
    marker = secrets.token_hex(4)
    created = session.tool(
        "create_memory",
        {
            "content": f"The compose smoke record {marker} confirms that memories persist across restarts.",
            "provenance": "Written by scripts/compose_smoke.sh through the packaged MCP proxy.",
            "memory_type": "fact",
            "idempotency_key": f"compose-smoke-{marker}",
        },
    )
    state.set(marker=marker, mcp_memory_id=created["memory_id"])
    return created["memory_id"]


def _read_memory(session: McpSession, state: State) -> str:
    marker, memory_id = state.get("marker"), state.get("mcp_memory_id")
    memory = session.tool("get_memory", {"memory_id": memory_id})
    if marker not in str(memory.get("content", "")):
        raise CheckFailed(f"get_memory {memory_id} does not hold the smoke record")
    found = session.tool("search", {"query": f"compose smoke record {marker}"})
    if memory_id not in [result.get("memory_id") for result in found.get("results", [])]:
        raise CheckFailed(f"search does not find the smoke record {memory_id}")
    return memory_id


def _find_source_memory(session: McpSession) -> str:
    found = session.tool("search", {"query": SOURCE_QUERY})
    results = found.get("results", [])
    if not results:
        raise CheckFailed(f"MCP search for {SOURCE_QUERY!r} returned no memory")
    return str(results[0].get("memory_id"))


def _list_source(session: McpSession, source_id: str) -> None:
    listed = session.tool("list_sources").get("data", [])
    if source_id not in [str(row.get("id") or row.get("source_id")) for row in listed]:
        raise CheckFailed(f"list_sources does not list source {source_id}")


def mcp_check(plugin_dir: Path, api: str, state: State, home: Path, *, phase: str) -> str:
    """Drive the packaged proxy as an agent does.

    ``tools`` initializes it, lists its tools and reaches the API through it.
    ``memories`` also finds a synced memory and writes a memory, then reads it
    back. ``verify``, after the restart, finds the source and that memory again.
    """
    with McpSession(plugin_dir, api, home) as session:
        notes = [_open_mcp(session)]
        if phase == "memories":
            notes.append(f"search finds synced memory {_find_source_memory(session)}")
            _write_memory(session, state)
            notes.append(f"memory {_read_memory(session, state)} written, read back and found by search")
        elif phase == "verify":
            if state.get("source_id"):
                _list_source(session, state.get("source_id"))
                notes.append(f"source {state.get('source_id')} listed")
            if state.get("mcp_memory_id"):
                notes.append(f"memory {_read_memory(session, state)} read back and found by search")
    return "; ".join(notes)


# -- Models -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelPlan:
    provider: str
    endpoint: str | None
    auth_strategy: str
    api_key: str | None
    models: dict[str, str]


def _listed_model(api: str, kind: str, plan: ModelPlan) -> dict[str, Any]:
    discovery = api_json(
        api,
        "POST",
        f"/api/v1/model-settings/{kind}/discover-models",
        _with_key({"provider": plan.provider, "endpoint": plan.endpoint, "auth_strategy": plan.auth_strategy}, plan),
    )
    if not discovery.get("ok"):
        raise CheckFailed(f"listing the {kind} models failed: {discovery.get('message')}")
    # Ollama lists a model pulled without a tag under its ``latest`` tag.
    wanted = {plan.models[kind], f"{plan.models[kind]}:latest"}
    for option in discovery.get("models", []):
        if option.get("id") in wanted:
            return option
    listed = ", ".join(option.get("id", "?") for option in discovery.get("models", [])[:10])
    raise CheckFailed(f"{plan.provider} does not list the {kind} model {plan.models[kind]} (it lists {listed})")


def _with_key(body: dict[str, Any], plan: ModelPlan) -> dict[str, Any]:
    if plan.api_key:
        body["api_key"] = plan.api_key
    return body


def _candidate(kind: str, plan: ModelPlan, listed: dict[str, Any]) -> dict[str, Any]:
    body: dict[str, Any] = {
        "provider": plan.provider,
        "model_id": plan.models[kind],
        "endpoint": plan.endpoint,
        "auth_strategy": plan.auth_strategy,
    }
    if kind == "generation":
        body["token_limits"] = {
            "context_window_tokens": listed.get("context_window_tokens") or DEFAULT_CONTEXT_WINDOW_TOKENS,
            "max_output_tokens": listed.get("max_output_tokens") or DEFAULT_MAX_OUTPUT_TOKENS,
        }
        if listed.get("native_schema") is not None:
            body["native_schema"] = listed["native_schema"]
    return _with_key(body, plan)


def _reindex(api: str, timeout: float) -> str:
    api_json(api, "POST", "/api/v1/model-settings/vector-index/reindex", expect=(202,))

    def finished() -> str | None:
        index = api_json(api, "GET", "/api/v1/model-settings/vector-index")
        rebuild = index.get("reindex") or {}
        if rebuild.get("state") == "failed":
            raise CheckFailed(f"the vector index rebuild failed: {rebuild.get('error_code')}")
        if rebuild.get("state") == "succeeded" and index.get("serves_search"):
            return f"index rebuilt ({rebuild.get('memories_indexed')} memories)"
        return None

    return wait_until(finished, timeout=timeout, interval=2, what="the vector index rebuild")


def configure_models(api: str, plan: ModelPlan, state: State, *, timeout: float) -> str:
    """Save and validate both models through the Settings API, as Settings does."""
    notes = []
    for kind in ("generation", "embedding"):
        listed = _listed_model(api, kind, plan)
        api_json(api, "PUT", f"/api/v1/model-settings/{kind}/candidate", _candidate(kind, plan, listed))
        outcome = api_json(api, "POST", f"/api/v1/model-settings/{kind}/candidate/validate")
        if not outcome.get("passed"):
            raise CheckFailed(f"{kind} model {plan.models[kind]} failed validation: {outcome.get('error_code')}")
        latency = (outcome.get("validation") or {}).get("facts", {}).get("latency_ms")
        note = f"{kind} {plan.models[kind]} validated in {latency} ms"
        if outcome.get("reindex_required"):
            note += ", " + _reindex(api, timeout)
        elif not outcome.get("promoted"):
            raise CheckFailed(f"{kind} model passed validation but was not put to use")
        notes.append(note)
    settings = api_json(api, "GET", "/api/v1/model-settings")
    for kind in ("generation", "embedding"):
        active = settings[kind].get("active") or {}
        if active.get("model_id") != plan.models[kind] or active.get("state") != "ready":
            raise CheckFailed(f"the {kind} model in use is {active.get('model_id')} ({active.get('state')})")
    state.set(models={"provider": plan.provider, **plan.models})
    return f"{plan.provider}: " + "; ".join(notes)


# -- Source sync --------------------------------------------------------------------------


def create_source(api: str, root: str, state: State) -> str:
    """Add a Local Repository source over the folder mounted into the API container."""
    created = api_json(
        api,
        "POST",
        "/api/v1/sources",
        {"type": "local_markdown", "name": SOURCE_NAME, "access_policy": "private", "config": {"root": root}},
    )
    state.set(source_id=created["id"])
    _source(api, created["id"])
    return f"source {created['id']} ({created['type']}) over {root}"


def request_sync(api: str, state: State) -> str:
    """Ask for the source's sync; the local daemon collects it."""
    receipt = api_json(api, "POST", f"/api/v1/sources/{state.get('source_id')}/sync", expect=(202,))
    return f"collection job {receipt.get('job_id') or receipt.get('run_id')} {receipt.get('status')}"


def _source(api: str, source_id: str) -> dict[str, Any]:
    for source in api_json(api, "GET", "/api/v1/sources").get("data", []):
        if source.get("id") == source_id:
            return source
    raise CheckFailed(f"source {source_id} is not listed")


def wait_for_sync(api: str, state: State, *, timeout: float) -> str:
    """Wait until the source's sync has finished; its documents have to have given memories."""
    source_id = state.get("source_id")

    def synced() -> str | None:
        source = _source(api, source_id)
        sync = source.get("sync") or {}
        if sync.get("status") == "failed":
            raise CheckFailed(f"the sync failed: {sync.get('error_message')}")
        if sync.get("status") not in FINISHED_SYNC_STATES:
            return None
        if source.get("memory_count", 0) < 1:
            raise CheckFailed(
                f"sync {sync['status']} with {source['doc_count']} documents, but the generation model "
                "wrote no memory from them"
            )
        return f"sync {sync['status']}: {source['doc_count']} documents, {source['memory_count']} memories"

    outcome = wait_until(synced, timeout=timeout, interval=5, what="the source sync")
    source = _source(api, source_id)
    state.set(source_docs=source["doc_count"], source_memories=source["memory_count"])
    return outcome


def search_source(api: str, state: State) -> str:
    """A search limited to the source returns at least one of its memories."""
    found = api_json(
        api,
        "POST",
        "/api/v1/memories/search",
        {"query": SOURCE_QUERY, "top_k": 5, "source_filter": {"source_ids": [state.get("source_id")]}},
    )
    results = found.get("results", [])
    if not results:
        raise CheckFailed(f"searching the source for {SOURCE_QUERY!r} returned no memory")
    return f"{len(results)} memories for {SOURCE_QUERY!r}; first: {_excerpt(str(results[0].get('summary') or results[0].get('content') or '').encode(), 90)}"


# -- After the restart -------------------------------------------------------------------------


def check_persisted(api: str, state: State) -> str:
    """The models and the source recorded before the restart are still there."""
    notes = []
    health = api_json(api, "GET", "/api/v1/health")
    if (health.get("database") or {}).get("status") != "ok":
        raise CheckFailed("health reports the database as not ok after the restart")
    models = state.get("models")
    if models:
        settings = api_json(api, "GET", "/api/v1/model-settings")
        for kind in ("generation", "embedding"):
            active = settings[kind].get("active") or {}
            if active.get("provider") != models["provider"] or active.get("model_id") != models[kind]:
                raise CheckFailed(
                    f"the {kind} model in use changed to {active.get('provider')}/{active.get('model_id')}"
                )
            if active.get("state") == "needs_revalidation":
                raise CheckFailed(f"the {kind} model needs revalidation after the restart")
        vector_store = (health.get("vector_store") or {}).get("status")
        notes.append(f"models kept ({models['generation']}, {models['embedding']}); vector store {vector_store}")
    source_id = state.get("source_id")
    if source_id:
        source = _source(api, source_id)
        docs_before, memories_before = state.get("source_docs") or 0, state.get("source_memories") or 0
        if source["doc_count"] < docs_before or source["memory_count"] < memories_before:
            raise CheckFailed(
                f"source {source_id} holds {source['doc_count']} documents and {source['memory_count']} memories, "
                f"{docs_before} and {memories_before} before the restart"
            )
        notes.append(f"source {source_id}: {source['doc_count']} documents, {source['memory_count']} memories")
        if memories_before:
            notes.append(search_source(api, state).split(";")[0])
    return "; ".join(notes) if notes else f"health {health.get('status')}"


# -- Command line ------------------------------------------------------------------------------


def _model_plan(args: argparse.Namespace) -> ModelPlan:
    if args.provider == "ollama":
        return ModelPlan(
            "ollama",
            args.endpoint,
            "none",
            None,
            {"generation": args.generation_model, "embedding": args.embedding_model},
        )
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise CheckFailed("OPENROUTER_API_KEY is not set")
    return ModelPlan(
        "openrouter", None, "api_key", api_key, {"generation": args.generation_model, "embedding": args.embedding_model}
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    ports = commands.add_parser("free-ports", help="print loopback ports nothing listens on")
    ports.add_argument("count", type=int)

    bindings = commands.add_parser("bindings", help="check `docker inspect` output read from stdin")
    bindings.add_argument("--expect", action="append", default=[], metavar="SERVICE=PORT")

    def api_command(name: str, help_text: str) -> argparse.ArgumentParser:
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--api", required=True, help="the smoke API, e.g. http://127.0.0.1:18765")
        command.add_argument("--state", type=Path, required=True)
        return command

    probe_command = api_command("probe", "check API health and the admin UI")
    probe_command.add_argument("--ui", required=True)

    mcp = api_command("mcp", "drive the packaged MCP proxy")
    mcp.add_argument("--plugin", type=Path, required=True)
    mcp.add_argument("--home", type=Path, required=True, help="the HOME the proxy runs with")
    mcp.add_argument("--phase", choices=("tools", "memories", "verify"), required=True)

    models = api_command("models", "configure and validate the generation and embedding models")
    models.add_argument("--provider", choices=("ollama", "openrouter"), required=True)
    models.add_argument("--endpoint", default=None)
    models.add_argument("--generation-model", required=True)
    models.add_argument("--embedding-model", required=True)
    models.add_argument("--timeout", type=float, default=600)

    source = api_command("source", "add the local source")
    source.add_argument("--root", required=True, help="the folder as the API container sees it")

    api_command("sync", "ask for the source's sync")

    sync = api_command("wait-sync", "wait for the source's sync")
    sync.add_argument("--timeout", type=float, default=900)

    api_command("search", "search the source's memories")
    api_command("persisted", "check what was recorded before the restart")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> str:
    if args.command == "free-ports":
        return " ".join(str(port) for port in free_ports(args.count))
    if args.command == "bindings":
        expected = {}
        for item in args.expect:
            service, _, port = item.partition("=")
            expected[service] = int(port)
        return check_bindings(json.load(sys.stdin), expected)
    state = State(args.state)
    if args.command == "probe":
        return probe(args.api, args.ui)
    if args.command == "mcp":
        return mcp_check(args.plugin, args.api, state, args.home, phase=args.phase)
    if args.command == "models":
        return configure_models(args.api, _model_plan(args), state, timeout=args.timeout)
    if args.command == "source":
        return create_source(args.api, args.root, state)
    if args.command == "sync":
        return request_sync(args.api, state)
    if args.command == "wait-sync":
        return wait_for_sync(args.api, state, timeout=args.timeout)
    if args.command == "search":
        return search_source(args.api, state)
    return check_persisted(args.api, state)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        print(run(args))
    except CheckFailed as exc:
        print(str(exc))
        return 1
    except Exception as exc:  # an unexpected answer still ends as one report line
        traceback.print_exc()
        print(f"{args.command} stopped on {type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
