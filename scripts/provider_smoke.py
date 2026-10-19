"""Check that real model providers accept what MemForge sends them.

``tests/test_provider_request_contract.py``, the provider request contract
matrix, states the request MemForge writes for every provider and answers it in
memory. This script sends the same requests to the providers themselves. Run it
before a release and whenever LiteLLM, the model transport or a provider
changes, then paste the table it prints into the pull request or the release
notes.

Each provider is tried only when it is available:

* OpenRouter, when ``OPENROUTER_API_KEY`` is set;
* OpenAI, when ``OPENAI_API_KEY`` is set;
* Anthropic (generation only), when ``ANTHROPIC_API_KEY`` is set;
* Ollama, when a server answers at ``--ollama-endpoint``;
* one LiteLLM route named with ``--litellm-route``, which takes its credential
  from the provider's own environment variables, as a deployment configures it.

A provider that is not available is reported as skipped, so the table shows
what was verified and what was not. API keys are read from the environment,
or from ``.env.test`` at the repository root for a variable the environment
leaves unset (see ``scripts/smoke_env.py``), and never written anywhere.

Every check takes the path "Save and validate" in Settings takes. The model is
bound through its provider descriptor; when the descriptor's endpoint lists its
models, the listing is asked once per kind whether it holds the model (a model
it does not hold is reported, not failed); a generation model is then bound
with the token limits and the response schema support the listing gives for it,
as Settings binds a model chosen from the list, and answers the structured
capability check, directly and without reasoning; an embedding model
embeds two short texts, twice, and has to answer one vector per text in one
dimension. Each check ends at ``--timeout`` seconds.

The exit status is 0 only when every check that ran passed; a skipped provider
fails nothing.

Usage::

    uv run python scripts/provider_smoke.py
    OPENROUTER_API_KEY=... uv run python scripts/provider_smoke.py \\
        --openrouter-generation-model openai/gpt-5-mini
    uv run python scripts/provider_smoke.py --ollama-endpoint http://gpu-box:11434
    uv run python scripts/provider_smoke.py --litellm-route voyage/voyage-3 --kind embedding
    uv run python scripts/provider_smoke.py --litellm-route sap/gpt-5-mini --endpoint https://...
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Protocol, TypeVar

import httpx

from memforge.config import AppConfig
from memforge.llm.bindings import ModelBinding, build_binding, compile_binding, installed_litellm_version
from memforge.llm.capability import CapabilityCheck
from memforge.llm.providers import (
    MODEL_KINDS,
    AuthStrategy,
    ModelConfigError,
    ModelKind,
    ResolvedInvocation,
    provider_descriptor,
)
from memforge.model_settings import check_binding
from memforge.release import installed_version
from memforge.runtime import DefaultRuntimeProvider
from memforge.server.model_settings_api import ModelDiscoveryResponse, ModelOption, list_endpoint_models

Result = Literal["pass", "fail", "skipped"]
CheckName = Literal["model listing", "structured answer", "embedding"]

DEFAULT_TIMEOUT_S = 60.0
DEFAULT_OLLAMA_ENDPOINT = "http://127.0.0.1:11434"
# How long an Ollama server has to answer before it counts as not running.
_OLLAMA_PROBE_TIMEOUT_S = 2.0
_LITELLM_ROUTE = "litellm"
_CAPABILITY_CHECKS: Mapping[ModelKind, CheckName] = {
    "generation": "structured answer",
    "embedding": "embedding",
}


@dataclass(frozen=True)
class Provider:
    """A provider the smoke tries on its own, with the models it checks by default."""

    descriptor: str
    # The environment variable its API key is read from; ``None`` for a
    # provider that needs no key.
    key_variable: str | None
    models: Mapping[ModelKind, str]


PROVIDERS: tuple[Provider, ...] = (
    Provider(
        "openrouter",
        "OPENROUTER_API_KEY",
        {"generation": "anthropic/claude-haiku-4.5", "embedding": "openai/text-embedding-3-small"},
    ),
    Provider("openai", "OPENAI_API_KEY", {"generation": "gpt-5-mini", "embedding": "text-embedding-3-small"}),
    Provider("anthropic", "ANTHROPIC_API_KEY", {"generation": "claude-haiku-4-5"}),
    Provider("ollama", None, {"generation": "qwen3:0.6b", "embedding": "nomic-embed-text"}),
)


@dataclass(frozen=True)
class Target:
    """One available provider and the models to check it with."""

    descriptor: str
    endpoint: str | None
    auth_strategy: AuthStrategy
    key_variable: str | None
    models: Mapping[ModelKind, str]


@dataclass(frozen=True)
class Row:
    """One line of the report: one check of one provider, or a provider that was skipped."""

    provider: str
    result: Result
    kind: ModelKind | None = None
    check: CheckName | None = None
    model: str | None = None
    latency_ms: int | None = None
    detail: str = ""


@dataclass(frozen=True)
class Report:
    date: str
    memforge_version: str
    litellm_version: str
    rows: Sequence[Row] = field(default_factory=tuple)


class Checks(Protocol):
    """The product calls each check is made through."""

    async def listing(self, binding: ModelBinding, api_key: str | None) -> ModelDiscoveryResponse: ...

    async def capability(self, binding: ModelBinding, invocation: ResolvedInvocation) -> CapabilityCheck: ...


class ProductChecks:
    """The model listing and the capability check Settings validates a model with."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self._config = config or AppConfig()
        self._runtime_provider = DefaultRuntimeProvider()

    async def listing(self, binding: ModelBinding, api_key: str | None) -> ModelDiscoveryResponse:
        descriptor = provider_descriptor(binding.provider)
        assert descriptor.listing is not None
        return await list_endpoint_models(
            descriptor,
            descriptor.listing,
            binding.kind,
            endpoint=binding.endpoint or "",
            api_key=api_key,
            strategy=binding.auth_strategy,
        )

    async def capability(self, binding: ModelBinding, invocation: ResolvedInvocation) -> CapabilityCheck:
        return await check_binding(self._config, self._runtime_provider, binding, invocation)


# -- Which providers to try ---------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check that real model providers accept what MemForge sends them.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help=f"seconds one check may take (default {DEFAULT_TIMEOUT_S:g})",
    )
    parser.add_argument(
        "--ollama-endpoint",
        default=DEFAULT_OLLAMA_ENDPOINT,
        help=f"the Ollama server to try (default {DEFAULT_OLLAMA_ENDPOINT})",
    )
    for provider in PROVIDERS:
        for kind, model in provider.models.items():
            parser.add_argument(
                f"--{provider.descriptor}-{kind}-model",
                default=model,
                help=f"the {kind} model to check {provider.descriptor} with (default {model})",
            )
    parser.add_argument(
        "--litellm-route",
        help="a LiteLLM provider/model route to check, with the provider's own environment variables",
    )
    parser.add_argument("--endpoint", help="the base URL the --litellm-route is sent to, when it needs one")
    parser.add_argument(
        "--kind",
        choices=MODEL_KINDS,
        default="generation",
        help="the kind of model --litellm-route names (default generation)",
    )
    return parser.parse_args(argv)


def plan_targets(
    options: argparse.Namespace,
    environ: Mapping[str, str],
    *,
    ollama_answers: bool,
) -> tuple[list[Target], list[Row]]:
    """Return the providers to try, and a skipped row for each one that is not available."""
    targets: list[Target] = []
    skipped: list[Row] = []
    for provider in PROVIDERS:
        models = {kind: getattr(options, f"{provider.descriptor}_{kind}_model") for kind in provider.models}
        if provider.key_variable is not None:
            if not environ.get(provider.key_variable, "").strip():
                skipped.append(Row(provider.descriptor, "skipped", detail=f"no key ({provider.key_variable} not set)"))
                continue
            targets.append(Target(provider.descriptor, None, "api_key", provider.key_variable, models))
        elif ollama_answers:
            targets.append(Target(provider.descriptor, options.ollama_endpoint, "none", None, models))
        else:
            skipped.append(Row(provider.descriptor, "skipped", detail=f"not reachable ({options.ollama_endpoint})"))
    if options.litellm_route:
        targets.append(
            Target(
                _LITELLM_ROUTE,
                options.endpoint,
                "environment",
                None,
                {options.kind: options.litellm_route},
            )
        )
    else:
        skipped.append(Row(_LITELLM_ROUTE, "skipped", detail="no --litellm-route given"))
    return targets, skipped


async def ollama_answers(endpoint: str, *, transport: httpx.AsyncBaseTransport | None = None) -> bool:
    """Whether an Ollama server answers at ``endpoint``."""
    try:
        async with httpx.AsyncClient(transport=transport, timeout=_OLLAMA_PROBE_TIMEOUT_S) as client:
            response = await client.get(f"{endpoint.rstrip('/')}/api/version")
    except httpx.HTTPError:
        return False
    return response.is_success


# -- Running the checks -------------------------------------------------------------

T = TypeVar("T")


class _DeadlineExceeded(Exception):
    pass


async def _within(timeout_s: float, call: Callable[[], Awaitable[T]]) -> tuple[T, int]:
    """Await ``call()`` within ``timeout_s``; return its result and how long it took."""
    started = time.perf_counter()
    try:
        async with asyncio.timeout(timeout_s):
            result = await call()
    except TimeoutError as exc:
        raise _DeadlineExceeded from exc
    return result, _elapsed_ms(started)


async def check_target(
    target: Target,
    environ: Mapping[str, str],
    checks: Checks,
    *,
    timeout_s: float,
) -> list[Row]:
    """Check every model of ``target``, each kind on its own."""
    api_key = environ[target.key_variable].strip() if target.key_variable else None
    rows: list[Row] = []
    for kind, model in target.models.items():
        rows.extend(await _check_model(target, kind, model, api_key, checks, timeout_s=timeout_s))
    return rows


async def _check_model(
    target: Target,
    kind: ModelKind,
    model: str,
    api_key: str | None,
    checks: Checks,
    *,
    timeout_s: float,
) -> list[Row]:
    capability = _CAPABILITY_CHECKS[kind]

    def row(check: CheckName, result: Result, latency_ms: int | None = None, detail: str = "") -> Row:
        return Row(target.descriptor, result, kind, check, model, latency_ms, detail)

    try:
        binding = _binding(target, kind, model)
    except ModelConfigError as exc:
        return [row(capability, "fail", detail=exc.code)]

    rows: list[Row] = []
    listed: ModelOption | None = None
    if provider_descriptor(target.descriptor).lists_models:
        try:
            listing, latency_ms = await _within(timeout_s, lambda: checks.listing(binding, api_key))
        except _DeadlineExceeded:
            rows.append(row("model listing", "fail", detail="deadline_exceeded"))
        except Exception as exc:
            rows.append(row("model listing", "fail", detail=type(exc).__name__))
        else:
            result, detail = _listing_result(target, model, listing)
            rows.append(row("model listing", result, latency_ms, detail))
            if kind == "generation" and listing.ok:
                listed = _listed(target, model, listing)

    try:
        checked = _binding(target, kind, model, listed=listed)
        invocation = compile_binding(checked, api_key)
        outcome, latency_ms = await _within(timeout_s, lambda: checks.capability(checked, invocation))
    except ModelConfigError as exc:
        rows.append(row(capability, "fail", detail=exc.code))
    except _DeadlineExceeded:
        rows.append(row(capability, "fail", detail="deadline_exceeded"))
    except Exception as exc:
        rows.append(row(capability, "fail", detail=type(exc).__name__))
    else:
        result, detail = _capability_result(outcome)
        rows.append(row(capability, result, latency_ms, detail))
    return rows


def _binding(
    target: Target,
    kind: ModelKind,
    model: str,
    *,
    listed: ModelOption | None = None,
) -> ModelBinding:
    """Bind the model as Settings saves it, with what the listing gives for a generation model.

    Settings takes the token limits and the response schema support the model
    listing gives for the model chosen from it.
    """
    token_limits: dict[str, int] = {}
    if listed is not None:
        limits = {
            "context_window_tokens": listed.context_window_tokens,
            "max_output_tokens": listed.max_output_tokens,
        }
        token_limits = {name: value for name, value in limits.items() if value is not None}
    return build_binding(
        kind=kind,
        provider=target.descriptor,
        model_id=model,
        endpoint=target.endpoint,
        auth_strategy=target.auth_strategy,
        token_limits=token_limits,
        native_schema=None if listed is None else listed.native_schema,
        config_source="settings",
    )


def _listed(target: Target, model: str, listing: ModelDiscoveryResponse) -> ModelOption | None:
    names = {model}
    if target.descriptor == "ollama" and ":" not in model:
        # Ollama lists a model pulled without a tag under its ``latest`` tag.
        names.add(f"{model}:latest")
    return next((option for option in listing.models if option.id in names), None)


def _listing_result(target: Target, model: str, listing: ModelDiscoveryResponse) -> tuple[Result, str]:
    if not listing.ok:
        return "fail", _failure(listing.error_code or "provider_error", listing.status)
    if not listing.models_supported:
        return "pass", "the endpoint lists no models"
    if _listed(target, model, listing) is None:
        return "pass", f"not among the {len(listing.models)} listed models"
    return "pass", ""


def _capability_result(outcome: CapabilityCheck) -> tuple[Result, str]:
    facts = outcome.facts
    # A model whose token limits nobody knows has still answered; only its
    # binding lacks what the runtime sizes requests with, which Settings asks
    # the operator for.
    limits_unknown = outcome.error_code == "token_limits_required"
    if not outcome.passed and not limits_unknown:
        return "fail", _failure(outcome.error_code or "provider_error", facts.get("status_code"))
    notes: list[str] = []
    if "dimension" in facts:
        notes.append(f"dimension {facts['dimension']}")
    mode = facts.get("structured_output_mode")
    if mode is not None:
        notes.append(f"{mode}, after fallback" if facts.get("schema_fallback") else str(mode))
    if limits_unknown:
        notes.append("token limits unknown to LiteLLM")
    return "pass", "; ".join(notes)


def _failure(error_code: str, status_code: object) -> str:
    return f"{error_code} (HTTP {status_code})" if isinstance(status_code, int) else error_code


def _elapsed_ms(started: float) -> int:
    return max(0, round((time.perf_counter() - started) * 1000))


async def run(
    options: argparse.Namespace,
    environ: Mapping[str, str],
    *,
    checks: Checks | None = None,
    probe_transport: httpx.AsyncBaseTransport | None = None,
) -> Report:
    """Try every available provider and report each check."""
    answers = await ollama_answers(options.ollama_endpoint, transport=probe_transport)
    targets, skipped = plan_targets(options, environ, ollama_answers=answers)
    checks = checks or ProductChecks()
    rows: list[Row] = []
    for target in targets:
        rows.extend(await check_target(target, environ, checks, timeout_s=options.timeout))
    return Report(
        date=datetime.now(timezone.utc).date().isoformat(),
        memforge_version=installed_version(),
        litellm_version=installed_litellm_version(),
        rows=(*rows, *skipped),
    )


# -- The report ---------------------------------------------------------------------


def render_report(report: Report) -> str:
    """Write the report as Markdown: a summary line, then one table row per check."""
    lines = [
        f"**Provider smoke** {report.date} · MemForge {report.memforge_version} · LiteLLM {report.litellm_version}",
        "",
        "| Provider | Kind | Check | Model | Result | Latency | Detail |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report.rows:
        cells = (
            row.provider,
            row.kind or "",
            row.check or "",
            f"`{row.model}`" if row.model else "",
            row.result,
            _latency(row.latency_ms),
            row.detail,
        )
        lines.append("| " + " | ".join(_cell(cell) for cell in cells) + " |")
    return "\n".join(lines) + "\n"


def exit_code(report: Report) -> int:
    """0 when every check that ran passed; a skipped provider fails nothing."""
    return 1 if any(row.result == "fail" for row in report.rows) else 0


def _latency(latency_ms: int | None) -> str:
    if latency_ms is None:
        return ""
    return f"{latency_ms} ms" if latency_ms < 1000 else f"{latency_ms / 1000:.1f} s"


def _cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


async def _main(options: argparse.Namespace) -> int:
    from litellm.llms.custom_httpx.async_client_cleanup import close_litellm_async_clients

    try:
        report = await run(options, os.environ)
    finally:
        await close_litellm_async_clients()
    sys.stdout.write(render_report(report))
    return exit_code(report)


def main(argv: Sequence[str] | None = None) -> int:
    from smoke_env import load_smoke_env

    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    options = parse_args(argv)
    # LiteLLM routes read their provider's variables from the process
    # environment, so the keys of .env.test are loaded into it.
    load_smoke_env()
    return asyncio.run(_main(options))


if __name__ == "__main__":
    raise SystemExit(main())
