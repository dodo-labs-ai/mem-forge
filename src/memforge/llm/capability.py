"""Capability checks: one real request that shows a configured model can do its job.

A binding is put to use only after the model it names has answered through the
same client the runtime uses. The generation check asks for one small nested
object under the runtime's structured-output policy and validates the reply.
The embedding check embeds two different texts, twice, and validates the
vectors.

A generation model passes only when the runtime can also size requests to it:
its context window and its largest output have to be known, from LiteLLM's
model registry, from the binding or from the ``MEMFORGE_LLM_*`` limits.

A check reports what it observed (latency, the vector dimension, the model id
the provider reported, how the structured reply was obtained, whether the
model's token limits are known) or one stable error code. It keeps no prompt
and no response.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from pydantic import ConfigDict

from memforge.llm.embedding import (
    EmbeddingDeadlineExceeded,
    EmbeddingError,
    EmbeddingProviderError,
    EmbeddingRuntime,
    EmbeddingSettings,
    EmbeddingTransport,
    InvalidEmbeddingResponse,
)
from memforge.llm.providers import ResolvedInvocation
from memforge.llm.structured import (
    LiteLlmStructuredClient,
    StructuredLlmError,
    StructuredLlmMetricsCollector,
    StructuredResponseModel,
)

logger = logging.getLogger(__name__)

__all__ = [
    "GENERATION_CHECK_TIMEOUT_S",
    "CapabilityCheck",
    "check_embedding",
    "check_generation",
    "embedding_error_code",
]

# The longest a generation check may take, queueing behind other model calls included.
GENERATION_CHECK_TIMEOUT_S = 60.0
_EMBEDDING_CHECK_TIMEOUT_S = 30.0
_GENERATION_CHECK_MAX_TOKENS = 256

_PROBE_LABEL = "memforge"
_PROBE_ITEMS = ["alpha", "beta", "gamma"]
_GENERATION_PROMPT = (
    "Reply with one JSON object and nothing else. Its \"probe\" field is an object "
    f"whose \"label\" is the text \"{_PROBE_LABEL}\", whose \"confirmed\" is true, and "
    f"whose \"items\" is the list of the texts {', '.join(_PROBE_ITEMS)} in that order."
)
_EMBEDDING_PROBE_TEXTS = (
    "The deployment runs every weekday at nine in the morning.",
    "Purple elephants prefer jazz to classical music.",
)
_STATUS_ERROR_CODES: Mapping[int, str] = {
    400: "request_rejected",
    401: "authentication_failed",
    403: "authentication_failed",
    404: "model_not_found",
    408: "timeout",
    422: "request_rejected",
    429: "rate_limited",
}


class CapabilityProbe(StructuredResponseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    confirmed: bool
    items: list[str]


class CapabilityCheckResponse(StructuredResponseModel):
    """The nested object a generation model must return to pass its check."""

    model_config = ConfigDict(extra="forbid")

    probe: CapabilityProbe


@dataclass(frozen=True)
class CapabilityCheck:
    """What one capability check observed, or the code of why it failed."""

    facts: Mapping[str, Any] = field(default_factory=dict)
    error_code: str | None = None

    @property
    def passed(self) -> bool:
        return self.error_code is None


async def check_generation(client: LiteLlmStructuredClient) -> CapabilityCheck:
    """Ask the generation model for one nested object and validate the reply.

    ``client`` is the client the runtime builds for the model, so the check
    takes the same route to structured output as every other call: the
    provider's native schema support when it has it, the schema as text
    otherwise, and one fallback from the first to the second.
    """
    collector = StructuredLlmMetricsCollector()
    started = perf_counter()
    error_code: str | None = None
    try:
        with client.metrics_scope(collector):
            async with asyncio.timeout(GENERATION_CHECK_TIMEOUT_S):
                reply = await client.answer_schema(
                    _GENERATION_PROMPT,
                    response_format=CapabilityCheckResponse,
                    max_tokens=_GENERATION_CHECK_MAX_TOKENS,
                )
        if not _is_expected_reply(reply):
            error_code = "structured_output_invalid"
    except TimeoutError:
        error_code = "timeout"
    except StructuredLlmError as exc:
        error_code = _generation_error_code(exc)
    except Exception as exc:
        logger.warning("Generation capability check failed (%s)", type(exc).__name__)
        error_code = "provider_error"

    facts: dict[str, Any] = {
        "latency_ms": _elapsed_ms(started),
        "token_limits_known": _token_limits_known(client),
    }
    if error_code is None and not facts["token_limits_known"]:
        error_code = "token_limits_required"
    calls = collector.calls()
    if calls:
        call = calls[-1]
        facts["structured_output_mode"] = call.final_mode
        facts["schema_fallback"] = call.fallback_count > 0
        if call.provider_model:
            facts["provider_model"] = call.provider_model
    return CapabilityCheck(facts=facts, error_code=error_code)


async def check_embedding(
    invocation: ResolvedInvocation,
    *,
    transport: EmbeddingTransport | None = None,
) -> CapabilityCheck:
    """Embed two different texts twice and validate what comes back.

    Each request goes through an Embedding Runtime, which accepts a response
    only when it holds one vector per input in input order, finite numbers and
    one dimension. The second request, from a runtime with no memory of the
    first, has to produce the same dimension, and both have to produce the
    dimension the invocation asks for when it asks for one.
    """
    started = perf_counter()
    facts: dict[str, Any] = {}
    error_code: str | None = None
    try:
        first = await _embed_probe(invocation, transport)
        facts["dimension"] = first.dimension
        if first.provider_model:
            facts["provider_model"] = first.provider_model
        requested = invocation.extra_params.get("dimensions")
        if requested is not None and requested != first.dimension:
            error_code = "embedding_dimension_mismatch"
        elif (await _embed_probe(invocation, transport)).dimension != first.dimension:
            error_code = "embedding_dimension_unstable"
    except EmbeddingError as exc:
        error_code = embedding_error_code(exc)
    except Exception as exc:
        logger.warning("Embedding capability check failed (%s)", type(exc).__name__)
        error_code = "provider_error"
    facts["latency_ms"] = _elapsed_ms(started)
    return CapabilityCheck(facts=facts, error_code=error_code)


@dataclass(frozen=True)
class _EmbeddingObservation:
    dimension: int
    provider_model: str | None


async def _embed_probe(
    invocation: ResolvedInvocation,
    transport: EmbeddingTransport | None,
) -> _EmbeddingObservation:
    runtime = EmbeddingRuntime(
        EmbeddingSettings(
            invocation=invocation,
            timeout_s=_EMBEDDING_CHECK_TIMEOUT_S,
            num_retries=1,
            cache_size=0,
        ),
        transport=transport,
    )
    vectors = await runtime.embed_documents(_EMBEDDING_PROBE_TEXTS)
    if len(vectors) != len(_EMBEDDING_PROBE_TEXTS) or runtime.dimension is None:
        raise InvalidEmbeddingResponse("Embedding response does not hold one vector per input")
    return _EmbeddingObservation(
        dimension=runtime.dimension,
        provider_model=_safe_model_label(runtime.provider_model),
    )


def _token_limits_known(client: LiteLlmStructuredClient) -> bool:
    """Whether the runtime can size requests to the model.

    Work that fills a request to the model's capacity needs its token limits,
    from LiteLLM's model registry, from the binding or from the MEMFORGE_LLM_*
    limits.
    """
    try:
        client.request_budget()
    except ValueError:
        return False
    return True


def _is_expected_reply(reply: object) -> bool:
    if not isinstance(reply, CapabilityCheckResponse):
        return False
    probe = reply.probe
    return probe.label == _PROBE_LABEL and probe.confirmed is True and probe.items == _PROBE_ITEMS


def _generation_error_code(exc: StructuredLlmError) -> str:
    if exc.terminal_category == "deadline_exceeded":
        return "timeout"
    if exc.terminal_category == "invalid_response":
        return "structured_output_invalid"
    # A connection error the client could name a transport cause for never got an answer.
    unreachable = exc.unreachable or exc.error_code.startswith("APIConnectionError.")
    return _provider_error_code(status_code=exc.status_code, unreachable=unreachable)


def embedding_error_code(exc: EmbeddingError) -> str:
    """Name a failed embedding operation by a stable code."""
    if isinstance(exc, EmbeddingDeadlineExceeded):
        return "timeout"
    if isinstance(exc, InvalidEmbeddingResponse):
        return "embedding_response_invalid"
    if isinstance(exc, EmbeddingProviderError):
        if exc.parameter_refused:
            return "parameter_not_supported_by_route"
        return _provider_error_code(status_code=exc.status_code, unreachable=exc.unreachable)
    return "provider_error"


def _provider_error_code(*, status_code: int | None, unreachable: bool) -> str:
    """Name a failed provider request by whether it arrived and how it was answered."""
    if unreachable:
        return "provider_unreachable"
    return _STATUS_ERROR_CODES.get(status_code or 0, "provider_error")


def _safe_model_label(value: str | None) -> str | None:
    if not value or len(value) > 128 or not re.fullmatch(r"[A-Za-z0-9_.:/@-]+", value):
        return None
    return value


def _elapsed_ms(started: float) -> int:
    return max(0, round((perf_counter() - started) * 1000))
