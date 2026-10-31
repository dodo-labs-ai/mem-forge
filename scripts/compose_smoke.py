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
import hashlib
import json
import os
import re
import secrets
import select
import shlex
import socket
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
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


class _KeepRedirect(urllib.request.HTTPRedirectHandler):
    """Answers a redirect as an HTTP error, so its status and Location can be read."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


_NO_REDIRECT_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), _KeepRedirect())


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


def redirect(url: str, *, timeout: float = 30.0) -> tuple[int, str]:
    """The status and Location of one GET, without following a redirect."""
    try:
        with _NO_REDIRECT_OPENER.open(urllib.request.Request(url), timeout=timeout) as response:
            return response.status, ""
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers.get("Location", "")
    except (urllib.error.URLError, OSError) as exc:
        raise CheckFailed(f"GET {url} did not answer: {exc}") from exc


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
    # Deprecated (admin-ui-v1): /v1/ serves V1 and /v2/ redirects to / for the
    # release that moves the admin UI to /. Remove both checks with V1.
    for path in ("/", "/v1/"):
        response = request("GET", f"{ui}{path}")
        if response.status != 200 or "text/html" not in response.content_type:
            raise CheckFailed(f"admin UI {path} answered {response.status} {response.content_type!r}")
        if b"<html" not in response.body.lower():
            raise CheckFailed(f"admin UI {path} is not an HTML page")
        pages.append(path)
    status, location = redirect(f"{ui}/v2/settings?from=smoke")
    if status != 301 or location != "/settings?from=smoke":
        raise CheckFailed(f"admin UI /v2/settings?from=smoke answered {status} to {location!r}, not 301 to /settings?from=smoke")
    proxied = api_json(ui, "GET", "/api/v1/health")
    if (proxied.get("database") or {}).get("status") != "ok":
        raise CheckFailed("the admin UI does not proxy /api to the API")
    return f"health {health.get('status')}; admin UI serves {' and '.join(pages)}, redirects /v2/ to / and proxies /api"


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

MODEL_KINDS = ("generation", "embedding")
DEFAULT_OLLAMA_URL = "http://host.docker.internal:11434"
# The model a kind uses when only its provider is named: the default
# generation model, Claude Haiku 5.5, wherever the provider serves it; the same
# small embedding model on the hosted providers that serve it; and for Ollama
# the small local pair the quickstart suggests. ``scripts/provider_smoke.py``
# states its own defaults in ``PROVIDERS``; this helper runs without the
# checkout environment and cannot import them.
DEFAULT_MODELS: dict[str, dict[str, str]] = {
    "openrouter": {"generation": "anthropic/claude-haiku-5.5", "embedding": "openai/text-embedding-3-small"},
    "anthropic": {"generation": "claude-haiku-5-5"},
    "openai": {"embedding": "text-embedding-3-small"},
    "ollama": {"generation": "qwen3:0.6b", "embedding": "nomic-embed-text"},
}


_VARIABLE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class PlanError(Exception):
    """The model options name no complete model for each kind."""


@dataclass(frozen=True)
class KindPlan:
    """How one model kind is reached: a provider Settings offers, its model and its credential."""

    provider: str
    model: str
    endpoint: str | None
    # The environment variable named for the API key; ``None`` reads the
    # provider's own ``<PROVIDER>_API_KEY`` when its descriptor asks for a key.
    key_variable: str | None = None

    @property
    def label(self) -> str:
        return f"{self.provider}: {self.model}"


@dataclass(frozen=True)
class ModelPlan:
    kinds: dict[str, KindPlan]

    @property
    def label(self) -> str:
        return "; ".join(f"{kind} {self.kinds[kind].label}" for kind in MODEL_KINDS)


def provider_key_variable(provider: str) -> str:
    """The variable a provider's API key is read from when no other is named."""
    return re.sub(r"[^A-Z0-9]+", "_", provider.upper()) + "_API_KEY"


def plan_models(options: argparse.Namespace, environ: dict[str, str]) -> ModelPlan | None:
    """Each kind's provider, model, endpoint and key variable, or ``None`` when no kind has a provider.

    A kind whose provider is not named takes Ollama with ``--ollama``, otherwise
    OpenRouter when ``OPENROUTER_API_KEY`` is set. The plan holds no key: the
    key is read from its variable when the model is configured.
    """
    default_provider = None
    if options.ollama:
        default_provider = "ollama"
    elif environ.get("OPENROUTER_API_KEY", "").strip():
        default_provider = "openrouter"
    providers = {kind: getattr(options, f"{kind}_provider") or default_provider for kind in MODEL_KINDS}
    if not any(providers.values()):
        return None
    kinds = {}
    for kind, provider in providers.items():
        if provider is None:
            raise PlanError(
                f"no {kind} provider: pass --{kind}-provider, or --ollama, or export OPENROUTER_API_KEY"
            )
        model = getattr(options, f"{kind}_model") or DEFAULT_MODELS.get(provider, {}).get(kind)
        if not model:
            raise PlanError(f"--{kind}-model is required with the {provider} provider")
        key_variable = getattr(options, f"{kind}_key_env")
        if key_variable is not None:
            if not _VARIABLE_NAME.fullmatch(key_variable):
                # The value may be the key itself, so it is not repeated.
                raise PlanError(f"--{kind}-key-env takes the name of an environment variable, not a key")
            if not environ.get(key_variable, "").strip():
                raise PlanError(f"{key_variable}, named by --{kind}-key-env, is not set")
        endpoint = getattr(options, f"{kind}_endpoint") or (options.ollama_url if provider == "ollama" else None)
        kinds[kind] = KindPlan(provider, model, endpoint.rstrip("/") if endpoint else None, key_variable)
    return ModelPlan(kinds)


def plan_judge(options: argparse.Namespace, environ: dict[str, str]) -> KindPlan | None:
    """The model that judges the quality cases against their expectations, or ``None`` without one.

    It is named apart from the generation model, so a run of any candidate is
    judged by the same model: by default Claude Haiku 5.5 on OpenRouter when
    ``OPENROUTER_API_KEY`` is set. Its provider is one Settings offers for
    generation, and its key is read from ``--judge-key-env`` or the provider's
    own variable.
    """
    provider = options.judge_provider or (
        "openrouter" if environ.get("OPENROUTER_API_KEY", "").strip() else None
    )
    if provider is None:
        return None
    model = options.judge_model or DEFAULT_MODELS.get(provider, {}).get("generation")
    if not model:
        raise PlanError(f"--judge-model is required with the {provider} provider")
    key_variable = options.judge_key_env
    if key_variable is not None and not _VARIABLE_NAME.fullmatch(key_variable):
        # The value may be the key itself, so it is not repeated.
        raise PlanError("--judge-key-env takes the name of an environment variable, not a key")
    if key_variable is not None and not environ.get(key_variable, "").strip():
        raise PlanError(f"{key_variable}, named by --judge-key-env, is not set")
    endpoint = options.judge_endpoint or (options.ollama_url if provider == "ollama" else None)
    return KindPlan(provider, model, endpoint.rstrip("/") if endpoint else None, key_variable)


def judge_env(plan: KindPlan | None, environ: dict[str, str]) -> str:
    """The API's environment lines that configure the judge; none without a judge.

    Each value is single-quoted, so Compose reads it as written and never
    interpolates a ``$`` in a key.
    """
    if plan is None:
        return ""
    key_variable = plan.key_variable or provider_key_variable(plan.provider)
    values = {
        "PROVIDER": plan.provider,
        "MODEL": plan.model,
        "ENDPOINT": plan.endpoint or "",
        "API_KEY": environ.get(key_variable, "").strip(),
    }
    for name, value in values.items():
        if "'" in value or "\n" in value:
            # The value may be the key itself, so it is not repeated.
            raise PlanError(f"the judge's {name.lower()} holds a quote or a line break, which the env file cannot carry")
    return "\n".join(f"MEMFORGE_AGENT_EVALUATION_JUDGE_{name}='{value}'" for name, value in values.items())


def shell_plan(plan: ModelPlan | None, judge: KindPlan | None = None) -> str:
    """The plan as shell assignments: the labels the report prints and the Ollama servers to reach."""
    values = {"models_label": "", "judge_label": "", "ollama_endpoints": ""}
    if plan is not None:
        values = {
            "models_label": plan.label,
            "judge_label": judge.label if judge is not None else "",
            "ollama_endpoints": " ".join(
                sorted({
                    kind.endpoint
                    for kind in (*plan.kinds.values(), *([judge] if judge is not None else []))
                    if kind.provider == "ollama" and kind.endpoint
                })
            ),
        }
    return "\n".join(f"{name}={shlex.quote(value)}" for name, value in values.items())


@dataclass(frozen=True)
class Binding:
    """What the Settings API is sent for one kind: the planned model with the descriptor's auth strategy."""

    kind: str
    plan: KindPlan
    auth_strategy: str
    api_key: str | None = field(default=None, repr=False)

    def with_key(self, body: dict[str, Any]) -> dict[str, Any]:
        if self.api_key:
            body["api_key"] = self.api_key
        return body


def _binding(kind: str, plan: KindPlan, descriptors: dict[str, dict[str, Any]], environ: dict[str, str]) -> Binding:
    """Check the planned model against the provider descriptor Settings offers, as Settings would."""
    descriptor = descriptors.get(plan.provider)
    if descriptor is None:
        raise CheckFailed(f"{plan.provider} is not a provider Settings offers ({', '.join(sorted(descriptors))})")
    if kind not in descriptor.get("kinds", []):
        raise CheckFailed(f"{descriptor.get('label')} serves no {kind} models")
    if descriptor.get("endpoint_required") and not plan.endpoint:
        raise CheckFailed(f"{descriptor.get('label')} needs an endpoint: pass --{kind}-endpoint")
    strategies = descriptor.get("auth_strategies", [])
    # A provider that proposes no key, such as a local Ollama, is sent one only when it is named.
    key_variable = plan.key_variable
    if key_variable is None and strategies[:1] == ["api_key"]:
        key_variable = provider_key_variable(plan.provider)
    api_key = (environ.get(key_variable, "").strip() or None) if key_variable else None
    auth_strategy = "api_key" if api_key else "none"
    if auth_strategy not in strategies:
        raise CheckFailed(
            f"{descriptor.get('label')} needs an API key: set {key_variable} or name one with --{kind}-key-env"
        )
    return Binding(kind, plan, auth_strategy, api_key)


def _listed_model(api: str, binding: Binding) -> dict[str, Any]:
    """The listing's entry for the planned model; an endpoint that lists no models gives none.

    Where nothing is listed, the model is entered by hand, as Settings then asks.
    """
    kind, plan = binding.kind, binding.plan
    discovery = api_json(
        api,
        "POST",
        f"/api/v1/model-settings/{kind}/discover-models",
        binding.with_key(
            {"provider": plan.provider, "endpoint": plan.endpoint, "auth_strategy": binding.auth_strategy}
        ),
    )
    if not discovery.get("ok"):
        raise CheckFailed(f"listing the {kind} models of {plan.provider} failed: {discovery.get('message')}")
    if discovery.get("models_supported") is False:
        return {}
    # Ollama lists a model pulled without a tag under its ``latest`` tag.
    wanted = {plan.model, f"{plan.model}:latest"}
    for option in discovery.get("models", []):
        if option.get("id") in wanted:
            return option
    listed = ", ".join(option.get("id", "?") for option in discovery.get("models", [])[:10])
    raise CheckFailed(f"{plan.provider} does not list the {kind} model {plan.model} (it lists {listed})")


def _candidate(binding: Binding, listed: dict[str, Any]) -> dict[str, Any]:
    plan = binding.plan
    body: dict[str, Any] = {
        "provider": plan.provider,
        "model_id": plan.model,
        "endpoint": plan.endpoint,
        "auth_strategy": binding.auth_strategy,
    }
    if binding.kind == "generation":
        # Validation needs the model's token limits; a model whose listing
        # gives none, or that is not listed, takes the ones the quickstart
        # suggests for a local model.
        body["token_limits"] = {
            "context_window_tokens": listed.get("context_window_tokens") or DEFAULT_CONTEXT_WINDOW_TOKENS,
            "max_output_tokens": listed.get("max_output_tokens") or DEFAULT_MAX_OUTPUT_TOKENS,
        }
        if listed.get("native_schema") is not None:
            body["native_schema"] = listed["native_schema"]
    return binding.with_key(body)


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


def configure_models(api: str, plan: ModelPlan, environ: dict[str, str], state: State, *, timeout: float) -> str:
    """Save and validate each kind's model through the Settings API, as Settings does."""
    descriptors = {
        descriptor["id"]: descriptor
        for descriptor in api_json(api, "GET", "/api/v1/model-settings/providers").get("providers", [])
    }
    bindings = [_binding(kind, plan.kinds[kind], descriptors, environ) for kind in MODEL_KINDS]
    notes = []
    for binding in bindings:
        kind, label = binding.kind, binding.plan.label
        listed = _listed_model(api, binding)
        api_json(api, "PUT", f"/api/v1/model-settings/{kind}/candidate", _candidate(binding, listed))
        outcome = api_json(api, "POST", f"/api/v1/model-settings/{kind}/candidate/validate")
        if not outcome.get("passed"):
            raise CheckFailed(f"{kind} model {label} failed validation: {outcome.get('error_code')}")
        latency = (outcome.get("validation") or {}).get("facts", {}).get("latency_ms")
        note = f"{kind} {label} validated in {latency} ms"
        if outcome.get("reindex_required"):
            note += ", " + _reindex(api, timeout)
        elif not outcome.get("promoted"):
            raise CheckFailed(f"{kind} model passed validation but was not put to use")
        notes.append(note)
    settings = api_json(api, "GET", "/api/v1/model-settings")
    for kind in MODEL_KINDS:
        active = settings[kind].get("active") or {}
        if active.get("model_id") != plan.kinds[kind].model or active.get("state") != "ready":
            raise CheckFailed(f"the {kind} model in use is {active.get('model_id')} ({active.get('state')})")
    state.set(
        models={kind: {"provider": plan.kinds[kind].provider, "model": plan.kinds[kind].model} for kind in MODEL_KINDS}
    )
    return "; ".join(notes)


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


def wait_for_sync(api: str, state: State, *, timeout: float, new_run: bool = False) -> str:
    """Wait until the source's sync has finished; its documents have to have given memories.

    With ``new_run``, the sync that finished before must not count: a later run
    has to finish.
    """
    source_id = state.get("source_id")
    earlier_run = state.get("sync_run_id") if new_run else None
    if new_run and earlier_run is None:
        raise CheckFailed("no earlier sync run is recorded to wait past")

    def synced() -> str | None:
        source = _source(api, source_id)
        sync = source.get("sync") or {}
        if earlier_run is not None and sync.get("run_id") == earlier_run:
            return None
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
    state.set(
        source_docs=source["doc_count"], source_memories=source["memory_count"],
        sync_run_id=(source.get("sync") or {}).get("run_id"),
    )
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


# -- Quality snapshots --------------------------------------------------------------------------

# The Source configuration field that names the snapshot set a Source replays.
SNAPSHOT_CONFIG_FIELD = "evaluation_snapshot"
SNAPSHOT_SOURCE_FILE = "source.json"
SNAPSHOT_MESSAGES_FILE = "messages.json"
# A replayed conversation's newest message lands at least this long before replay began.
REPLAY_LEAD = timedelta(hours=1)
_REVISION_NAME = re.compile(r"[1-9][0-9]*")
_TITLE_TIME = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2})$")


def _snapshot_sets(snapshots: Path) -> list[Path]:
    """Each snapshot set, one per source type: a directory holding the Source configuration to create."""
    return sorted(path for path in snapshots.iterdir() if (path / SNAPSHOT_SOURCE_FILE).is_file())


def _snapshot_config(set_dir: Path, revision: int, replayed_at: str) -> dict[str, Any]:
    spec = json.loads((set_dir / SNAPSHOT_SOURCE_FILE).read_text(encoding="utf-8"))
    return {
        **spec["config"],
        SNAPSHOT_CONFIG_FIELD: {"set": set_dir.name, "revision": revision, "replayed_at": replayed_at},
    }


def create_snapshot_sources(api: str, snapshots: Path, state: State) -> str:
    """Add one Source per snapshot set, replaying its revision 1; all of them begin their replay now."""
    replayed_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    sources = {}
    for set_dir in _snapshot_sets(snapshots):
        spec = json.loads((set_dir / SNAPSHOT_SOURCE_FILE).read_text(encoding="utf-8"))
        created = api_json(
            api,
            "POST",
            "/api/v1/sources",
            {
                "type": set_dir.name,
                "name": spec["name"],
                "access_policy": "private",
                "config": _snapshot_config(set_dir, 1, replayed_at),
            },
        )
        sources[set_dir.name] = {"id": created["id"], "replayed_at": replayed_at}
    if not sources:
        raise CheckFailed(f"{snapshots} holds no snapshot set")
    state.set(snapshot_sources=sources)
    return f"{len(sources)} snapshot sources: {', '.join(sorted(sources))}"


def request_snapshot_syncs(api: str, snapshots: Path, state: State, revision: int) -> str:
    """Move every snapshot Source to ``revision`` and ask for its sync; the run that finished last is waited past."""
    sources = state.get("snapshot_sources") or {}
    if not sources:
        raise CheckFailed("no snapshot source was added")
    for source_type, recorded in sources.items():
        if revision > 1:
            config = _snapshot_config(snapshots / source_type, revision, recorded["replayed_at"])
            api_json(api, "PUT", f"/api/v1/sources/{recorded['id']}", {"config": config})
        recorded["waits_past"] = (_source(api, recorded["id"]).get("sync") or {}).get("run_id")
        api_json(api, "POST", f"/api/v1/sources/{recorded['id']}/sync", expect=(202,))
    state.set(snapshot_sources=sources)
    return f"revision {revision} requested for {len(sources)} snapshot sources"


def wait_for_snapshot_syncs(api: str, state: State, *, timeout: float) -> str:
    """Wait until every snapshot Source has finished a sync after the one it last finished, with memories."""
    sources = state.get("snapshot_sources") or {}
    deadline = time.monotonic() + timeout
    outcomes = []
    for source_type, recorded in sorted(sources.items()):

        def synced() -> str | None:
            source = _source(api, recorded["id"])
            sync = source.get("sync") or {}
            if sync.get("run_id") in (None, recorded.get("waits_past")):
                return None
            if sync.get("status") == "failed":
                raise CheckFailed(f"the {source_type} snapshot sync failed: {sync.get('error_message')}")
            if sync.get("status") not in FINISHED_SYNC_STATES:
                return None
            if source.get("memory_count", 0) < 1:
                raise CheckFailed(f"the {source_type} snapshot synced {source['doc_count']} documents into no memory")
            return f"{source_type} {sync['status']} ({source['doc_count']} documents, {source['memory_count']} memories)"

        remaining = max(deadline - time.monotonic(), 1.0)
        outcomes.append(wait_until(synced, timeout=remaining, interval=5, what=f"the {source_type} snapshot sync"))
    return "; ".join(outcomes)


def _parse_time(value: str) -> datetime:
    """An ISO 8601 time, with ``Z`` read as UTC on every Python this helper runs on."""
    return datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)


def _snapshot_message_files(set_dir: Path) -> list[Path]:
    """Each conversation revision's messages, in the item and revision directories the server reads."""
    return [
        path
        for path in set_dir.glob(f"*/*/{SNAPSHOT_MESSAGES_FILE}")
        if not path.parent.parent.name.startswith(".") and _REVISION_NAME.fullmatch(path.parent.name)
    ]


def _replayed_title(title: str, set_dir: Path, replayed_at: str) -> str:
    """A conversation Unit's title as replay names it: the time it ends with moves by the set's replay offset.

    A conversation snapshot's times move forward by whole days, so that its
    newest message across every revision lands in the day before
    ``REPLAY_LEAD`` ahead of the replay's start. A title names the time of
    the Unit's first message, so only its date changes.
    """
    if not _snapshot_message_files(set_dir):
        return title
    match = _TITLE_TIME.search(title)
    if match is None:
        raise CheckFailed(f"the conversation case {title!r} does not end with its first message's time")
    newest = max(
        _parse_time(message["created_at"])
        for messages in _snapshot_message_files(set_dir)
        for message in json.loads(messages.read_text(encoding="utf-8"))
    )
    offset = timedelta(days=(_parse_time(replayed_at) - REPLAY_LEAD - newest) // timedelta(days=1))
    moved = datetime.fromisoformat(match.group(1)) + offset
    return title[: match.start()] + moved.isoformat()[:16]


def quality_expectations(api: str, run_id: str) -> str:
    """Each expectation the judge did not find met, with its sentence and the output items it rests on.

    The run report names an expectation by its case and number only; this
    reads the case outputs, so it is for the synthetic cohort, whose content
    may appear in a pull request.
    """
    payload = api_json(api, "GET", f"/api/v1/agent-evaluations/runs/{run_id}/case-outputs")
    rows = []
    for case in payload.get("cases") or []:
        expectations = (case.get("rubric") or {}).get("expectations") or []
        items = _case_items(case)
        title = _case_title(case)
        for assessment in case.get("checks") or []:
            if assessment.get("criterion") != "expectation":
                continue
            code, _, cited = str(assessment.get("reason_code") or "").partition("/")
            expectation, _, detail = code.partition(":")
            if assessment.get("status") == "completed" and assessment.get("label") == "pass":
                continue
            index = int(expectation[1:]) if expectation[1:].isdigit() else 0
            outcome = (
                {"fail": "not met"}.get(str(assessment.get("label")), "unclear")
                if assessment.get("status") == "completed"
                else f"not judged: {detail}"
            )
            sentence = expectations[index - 1] if 0 < index <= len(expectations) else "?"
            shown = "<br>".join(
                f"{item}: {_cell(items.get(item, '?'))}" for item in cited.split("/") if item
            )
            rows.append((str(title), index, _cell(sentence), outcome, shown))
    if not rows:
        return "Every expectation the judge read was met."
    lines = ["| Case | Expectation | Outcome | Items |", "| --- | --- | --- | --- |"]
    lines += [f"| {_cell(title)} | E{index}: {sentence} | {outcome} | {shown} |" for title, index, sentence, outcome, shown in sorted(rows)]
    return "\n".join(lines)


def _case_title(case: dict[str, Any]) -> str:
    """The case's Source Unit as its projection titles it, by the values of its title fields; else its id."""
    title = ((case.get("manifest") or {}).get("projection") or {}).get("unit_title")
    if isinstance(title, dict):
        values = [str(field[1]) for field in title.get("fields") or [] if isinstance(field, list) and len(field) == 2]
        return " / ".join(values) or str(case.get("case_id"))
    return str(title or case.get("case_id"))


def _case_items(case: dict[str, Any]) -> dict[str, str]:
    """The text of each output item, by the ID an expectation decision cites."""
    manifest = case.get("manifest") or {}
    output = (case.get("result") or {}).get("output") or {}
    if case.get("case_kind") == "source_unit_derivation_v1":
        memories = (output.get("extraction") or {}).get("memories") or []
        return {f"M{index}": str(memory.get("content") or "") for index, memory in enumerate(memories, start=1)}
    if case.get("case_kind") == "source_unit_support_v1":
        claims = {
            str(work["memory"].get("id")): str(work["memory"].get("content") or "")
            for work in manifest.get("work_items") or []
        }
        results = {str(row.get("memory_id")): str(row.get("result")) for row in output.get("supports") or []}
        return {memory_id: f"{claim} ({results.get(memory_id, '?')})" for memory_id, claim in claims.items()}
    outcomes = output.get("outcomes") or {}
    rechecked = {str(item.get("memory_id")) for item in output.get("rechecks") or []}
    items = {
        str(item.get("id")): f"{item.get('content')} ({outcomes.get(str(item.get('id')), '?')}"
        + (", re-checked first in production" if str(item.get("id")) in rechecked else "") + ")"
        for item in manifest.get("incumbents") or []
    }
    items.update({
        f"N{index}": str(item.get("content") or "")
        for index, item in enumerate(manifest.get("new_extractions") or [], start=1)
    })
    return items


def _cell(text: str, limit: int = 200) -> str:
    text = " ".join(str(text).split()).replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def quality_outputs(api: str, run_id: str) -> str:
    """The case outputs of one quality run: each case's pinned input, its rubric and the model's output."""
    return json.dumps(
        api_json(api, "GET", f"/api/v1/agent-evaluations/runs/{run_id}/case-outputs"), indent=2, ensure_ascii=False
    )


# -- Model usage --------------------------------------------------------------------------------

# Every structured model call logs one line with its operation, model and token usage.
_MODEL_CALL = re.compile(r"structured_llm_call (\{.*\})\s*$")


@dataclass
class _ModelUsage:
    calls: int = 0
    unreported: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    operations: dict[str, int] = field(default_factory=dict)

    def describe(self, model: str) -> str:
        unreported = f", {self.unreported} without reported usage" if self.unreported else ""
        operations = ", ".join(f"{name} {count}" for name, count in sorted(self.operations.items()))
        noun = "call" if self.calls == 1 else "calls"
        return (
            f"{self.calls} {noun} to {model}, {self.prompt_tokens} input and {self.completion_tokens} output tokens"
            f"{unreported} ({operations})"
        )


def model_usage(log_lines: Iterable[str]) -> str:
    """The structured model calls the API logged, per model, and their input and output token totals.

    Each model the calls name gets its own entry, by the route the API logs,
    such as ``openai/gpt-6-luna``, most calls first; a judge on another model
    than generation is counted apart, and each entry's operations say which
    calls were the judge's. Embedding requests are not structured calls and are
    not counted. A call whose provider reported no usage is counted apart.
    """
    usage: dict[str, _ModelUsage] = {}
    for line in log_lines:
        match = _MODEL_CALL.search(line)
        if match is None:
            continue
        try:
            call = json.loads(match.group(1))
        except ValueError:
            continue
        entry = usage.setdefault(str(call.get("model") or "an unnamed model"), _ModelUsage())
        entry.calls += 1
        operation = str(call.get("operation") or "unnamed")
        entry.operations[operation] = entry.operations.get(operation, 0) + 1
        if call.get("prompt_tokens") is None or call.get("completion_tokens") is None:
            entry.unreported += 1
            continue
        entry.prompt_tokens += int(call["prompt_tokens"])
        entry.completion_tokens += int(call["completion_tokens"])
    if not usage:
        raise CheckFailed("the API logged no model call")
    return "; ".join(
        entry.describe(model) for model, entry in sorted(usage.items(), key=lambda item: -item[1].calls)
    )


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
        for kind in MODEL_KINDS:
            active = settings[kind].get("active") or {}
            if active.get("provider") != models[kind]["provider"] or active.get("model_id") != models[kind]["model"]:
                raise CheckFailed(
                    f"the {kind} model in use changed to {active.get('provider')}: {active.get('model_id')}"
                )
            if active.get("state") == "needs_revalidation":
                raise CheckFailed(f"the {kind} model needs revalidation after the restart")
        vector_store = (health.get("vector_store") or {}).get("status")
        kept = ", ".join(f"{models[kind]['provider']}: {models[kind]['model']}" for kind in MODEL_KINDS)
        notes.append(f"models kept ({kept}); vector store {vector_store}")
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


# -- Quality checks ----------------------------------------------------------------

# The modules the extraction prompt is built from; a run names their content hash.
_EXTRACTION_PROMPT_MODULES = tuple(
    Path(__file__).resolve().parents[1] / "src" / "memforge" / "pipeline" / name
    for name in ("memory_extractor.py", "extraction_contract.py", "memory_value.py")
)


def _extraction_prompt_hash() -> str:
    digest = hashlib.sha256()
    for module in _EXTRACTION_PROMPT_MODULES:
        digest.update(module.read_bytes())
    return digest.hexdigest()



# The cohort file's list of each case kind, and the CLI command that pins it.
QUALITY_CASE_KINDS = {
    "derivation": ("cases", "seed-derivation-cases"),
    "support": ("support_cases", "seed-support-cases"),
    "reconciliation": ("reconciliation_cases", "seed-reconciliation-cases"),
}


def quality_seed_spec(cohort_file: Path, state: State, kind: str) -> str:
    """The seed request of one case kind of the cohort file.

    Each case names its Document by its current title within the smoke
    source, or, with ``snapshot``, within the Source replaying that snapshot
    set, where a conversation Unit's title names its replayed time. A
    reconciliation case reads its hand-written situation from the file it
    names, next to the cohort file.
    """
    cohort = json.loads(cohort_file.read_text(encoding="utf-8"))
    source_id = state.get("source_id")
    if not source_id:
        raise CheckFailed("no source was added before the quality checks")
    snapshot_sources = state.get("snapshot_sources") or {}
    cases = []
    for case in cohort[QUALITY_CASE_KINDS[kind][0]]:
        if kind == "reconciliation":
            situation = json.loads((cohort_file.parent / case["file"]).read_text(encoding="utf-8"))
            case = {"title": case["title"], "input": situation["input"], "rubric": situation["rubric"]}
        snapshot = case.pop("snapshot", None)
        if snapshot is None:
            cases.append({**case, "source_id": source_id})
            continue
        if snapshot not in snapshot_sources:
            raise CheckFailed(f"the case {case['title']!r} names the snapshot set {snapshot!r}, which no source replays")
        replayed = snapshot_sources[snapshot]
        title = _replayed_title(case["title"], cohort_file.parent / "snapshots" / snapshot, replayed["replayed_at"])
        cases.append({**case, "title": title, "source_id": replayed["id"]})
    return json.dumps(
        {
            "selection_policy_version": f"{cohort['selection_policy_version']}-{kind}",
            "role": cohort["role"],
            "cases": cases,
        }
    )


def quality_run_spec(cohort_file: Path, seeded: dict[str, Any]) -> str:
    """The run spec for the cohort the seed request froze; the server pins its code and model."""
    skipped = {reason: names for reason, names in (seeded.get("skipped") or {}).items() if names}
    if seeded.get("error") or not seeded.get("cohort_id") or skipped:
        raise CheckFailed(f"the quality cohort was not pinned completely: {seeded.get('error') or skipped}")
    run = json.loads(cohort_file.read_text(encoding="utf-8"))["run"]
    prompt_hash = _extraction_prompt_hash()
    return json.dumps(
        {
            **run,
            "cohort_id": seeded["cohort_id"],
            "candidate_manifest": {**run["candidate_manifest"], "prompt_hash": prompt_hash},
        }
    )


# -- Command line ------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    ports = commands.add_parser("free-ports", help="print loopback ports nothing listens on")
    ports.add_argument("count", type=int)

    bindings = commands.add_parser("bindings", help="check `docker inspect` output read from stdin")
    bindings.add_argument("--expect", action="append", default=[], metavar="SERVICE=PORT")

    commands.add_parser("model-usage", help="total the model calls per model in API logs read from stdin")

    def model_options(command: argparse.ArgumentParser) -> None:
        command.add_argument("--ollama", action="store_true", help="a kind without a provider uses Ollama")
        command.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help="Ollama as the API container reaches it")
        for kind in (*MODEL_KINDS, "judge"):
            command.add_argument(f"--{kind}-provider", help=f"the provider Settings offers for the {kind} model")
            command.add_argument(f"--{kind}-model", help=f"the {kind} model id")
            command.add_argument(f"--{kind}-endpoint", help=f"the {kind} provider's base URL")
            command.add_argument(f"--{kind}-key-env", help=f"the environment variable holding the {kind} API key")

    model_options(commands.add_parser("model-plan", help="print each kind's planned model as shell assignments"))
    model_options(commands.add_parser("judge-env", help="print the API environment lines that configure the judge"))

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
    model_options(models)
    models.add_argument("--timeout", type=float, default=600)

    source = api_command("source", "add the local source")
    source.add_argument("--root", required=True, help="the folder as the API container sees it")

    api_command("sync", "ask for the source's sync")

    sync = api_command("wait-sync", "wait for the source's sync")
    sync.add_argument("--timeout", type=float, default=900)
    sync.add_argument("--new-run", action="store_true", help="wait for a run other than the one that last finished")

    snapshot_sources = api_command("snapshot-sources", "add one source per quality snapshot set")
    snapshot_sources.add_argument("--snapshots", type=Path, required=True)
    snapshot_sync = api_command("snapshot-sync", "move the snapshot sources to a revision and ask for their sync")
    snapshot_sync.add_argument("--snapshots", type=Path, required=True)
    snapshot_sync.add_argument("--revision", type=int, required=True)
    snapshot_wait = api_command("snapshot-wait", "wait for every snapshot source's sync")
    snapshot_wait.add_argument("--timeout", type=float, default=900)

    api_command("search", "search the source's memories")
    seed_spec = api_command("quality-seed-spec", "print the quality cohort's seed request")
    seed_spec.add_argument("--cohort", type=Path, required=True)
    seed_spec.add_argument("--kind", choices=tuple(QUALITY_CASE_KINDS), required=True)
    run_spec = api_command("quality-run-spec", "print the quality run spec for a seed answer read from stdin")
    run_spec.add_argument("--cohort", type=Path, required=True)
    expectations = api_command("quality-expectations", "list the expectations of a run the judge did not find met")
    expectations.add_argument("--run", required=True)
    outputs = api_command("quality-outputs", "print each case's pinned input, rubric and model output of a run")
    outputs.add_argument("--run", required=True)
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
    if args.command == "model-usage":
        return model_usage(sys.stdin)
    if args.command == "model-plan":
        return shell_plan(plan_models(args, dict(os.environ)), plan_judge(args, dict(os.environ)))
    if args.command == "judge-env":
        return judge_env(plan_judge(args, dict(os.environ)), dict(os.environ))
    state = State(args.state)
    if args.command == "probe":
        return probe(args.api, args.ui)
    if args.command == "mcp":
        return mcp_check(args.plugin, args.api, state, args.home, phase=args.phase)
    if args.command == "models":
        plan = plan_models(args, dict(os.environ))
        if plan is None:
            raise CheckFailed("no model provider is named")
        return configure_models(args.api, plan, dict(os.environ), state, timeout=args.timeout)
    if args.command == "source":
        return create_source(args.api, args.root, state)
    if args.command == "sync":
        return request_sync(args.api, state)
    if args.command == "wait-sync":
        return wait_for_sync(args.api, state, timeout=args.timeout, new_run=args.new_run)
    if args.command == "snapshot-sources":
        return create_snapshot_sources(args.api, args.snapshots, state)
    if args.command == "snapshot-sync":
        return request_snapshot_syncs(args.api, args.snapshots, state, args.revision)
    if args.command == "snapshot-wait":
        return wait_for_snapshot_syncs(args.api, state, timeout=args.timeout)
    if args.command == "search":
        return search_source(args.api, state)
    if args.command == "quality-seed-spec":
        return quality_seed_spec(args.cohort, state, args.kind)
    if args.command == "quality-run-spec":
        return quality_run_spec(args.cohort, json.load(sys.stdin))
    if args.command == "quality-expectations":
        return quality_expectations(args.api, args.run)
    if args.command == "quality-outputs":
        return quality_outputs(args.api, args.run)
    return check_persisted(args.api, state)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    # The shell evaluates what model-plan prints, so its reasons go to stderr.
    reasons = sys.stderr if args.command in {"model-plan", "judge-env"} else sys.stdout
    try:
        print(run(args))
    except PlanError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except CheckFailed as exc:
        print(str(exc), file=reasons)
        return 1
    except Exception as exc:  # an unexpected answer still ends as one report line
        traceback.print_exc()
        print(f"{args.command} stopped on {type(exc).__name__}: {exc}", file=reasons)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
