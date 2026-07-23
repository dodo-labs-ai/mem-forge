"""Structured LLM calls with LiteLLM response schemas."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import Lock
from time import perf_counter
from typing import Any, Callable, Iterator, Literal, Mapping, Protocol, get_args, get_origin

import litellm
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from memforge.llm.providers import litellm_optional_kwargs

logger = logging.getLogger(__name__)

type StructuredLlmTerminalCategory = Literal[
    "success",
    "deadline_exceeded",
    "provider_error",
    "invalid_response",
]


def _expects_container(annotation: object) -> bool:
    """True when a field annotation resolves to a list/tuple/set or nested model."""
    origin = get_origin(annotation)
    if origin in (list, tuple, set, frozenset):
        return True
    if origin is not None:  # Optional[...] / Union[...]
        return any(_expects_container(arg) for arg in get_args(annotation))
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


class StructuredResponseModel(BaseModel):
    """Base for LLM structured-output schemas.

    Some gateway/tool-use responses encode list or nested-object fields as JSON
    strings, for example ``{"memories": "[...]"}``. Normalize those containers
    before field validation so the declared schema still owns correctness.
    """

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def _decode_stringified_containers(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        decoded: dict[str, object] | None = None
        for name, field in cls.model_fields.items():
            key = field.alias if field.alias and field.alias in data else name
            value = data.get(key)
            if not isinstance(value, str) or not _expects_container(field.annotation):
                continue
            try:
                parsed = json.loads(value)
            except (TypeError, ValueError):
                continue
            if decoded is None:
                decoded = dict(data)
            decoded[key] = parsed
        return decoded if decoded is not None else data


class SourceSupportDecision(StructuredResponseModel):
    """One verifier decision for an existing memory candidate."""

    model_config = ConfigDict(extra="ignore")

    memory_id: str = Field(min_length=1)
    supported: bool
    excerpt: str | None = None
    reason: str | None = None


class SourceSupportResponse(StructuredResponseModel):
    """Schema returned by the source-support verifier."""

    model_config = ConfigDict(extra="ignore")

    decisions: list[SourceSupportDecision]


class AgentSessionAuthorityDecision(StructuredResponseModel):
    """One semantic authority decision for a candidate agent-session user event."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(min_length=1)
    is_authoritative: bool
    authority_kind: Literal[
        "durable_user_intent",
        "future_memory_intent",
        "durable_preference",
        "design_decision",
        "rule_or_convention",
        "approval_of_durable_direction",
        "not_authoritative",
    ]
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def _authority_kind_matches_decision(self):
        if self.is_authoritative and self.authority_kind == "not_authoritative":
            raise ValueError("authoritative decisions require an authoritative authority_kind")
        if not self.is_authoritative and self.authority_kind != "not_authoritative":
            raise ValueError("non-authoritative decisions require authority_kind='not_authoritative'")
        return self


class AgentSessionAuthorityResponse(StructuredResponseModel):
    """Schema returned by agent-session authority classification."""

    model_config = ConfigDict(extra="forbid")

    decisions: list[AgentSessionAuthorityDecision]


class MemoryCandidate(StructuredResponseModel):
    """One memory candidate extracted from a source document."""

    model_config = ConfigDict(extra="ignore")

    content: str = Field(min_length=1)
    memory_type: Literal["fact", "decision", "convention", "procedure"]
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    entity_refs: list[str] = Field(default_factory=list)
    valid_from: str | None = None
    valid_until: str | None = None
    extraction_context: str | None = None
    evidence_quote: str | None = None
    evidence_anchor: Literal["unit", "glossary", "preamble", "outline", "document", "unknown"] = "unknown"
    source_observation_id: str | None = None
    required_source_observation_ids: list[str] = Field(default_factory=list)


class MemoryExtractionResponse(StructuredResponseModel):
    """Schema returned by memory extraction."""

    model_config = ConfigDict(extra="ignore")

    memories: list[MemoryCandidate]


class CandidateLedgerDecision(StructuredResponseModel):
    """One uniqueness decision for a transient extracted candidate."""

    model_config = ConfigDict(extra="ignore")

    index: int = Field(ge=0)
    action: Literal["KEEP", "DROP_REDUNDANT"]
    canonical_index: int | None = Field(default=None, ge=0)
    reason: str = Field(default="", max_length=1000)


class CandidateLedgerResponse(StructuredResponseModel):
    """Complete within-revision uniqueness ledger for extracted candidates."""

    model_config = ConfigDict(extra="ignore")

    decisions: list[CandidateLedgerDecision]


class ReconciliationDecision(StructuredResponseModel):
    """One same-document memory reconciliation decision."""

    model_config = ConfigDict(extra="ignore")

    action: Literal["ADD", "UPDATE", "SUPERSEDE", "DELETE", "NOOP"]
    index: int | None = None
    memory_id: str | None = None
    updated_content: str | None = None
    reason: str | None = None
    flag_for_review: bool = False


class ReconciliationResponse(StructuredResponseModel):
    """Schema returned by same-document memory reconciliation."""

    model_config = ConfigDict(extra="ignore")

    decisions: list[ReconciliationDecision]


class MemoryRelationDecision(StructuredResponseModel):
    """One exact pair classification with explicit refinement direction."""

    model_config = ConfigDict(extra="ignore")

    pair_index: int = Field(ge=0)
    classification: Literal["equivalent", "refines", "contradicts", "unrelated"]
    direction: Literal[
        "symmetric",
        "challenger_to_candidate",
        "candidate_to_challenger",
    ]
    reason: str = Field(default="", max_length=1000)

    @model_validator(mode="after")
    def _validate_direction(self) -> MemoryRelationDecision:
        directional = self.classification == "refines"
        if directional == (self.direction == "symmetric"):
            raise ValueError("REFINES must be directional and other relations symmetric")
        return self


class MemoryRelationResponse(StructuredResponseModel):
    """Schema for a complete batch of exact Memory-pair decisions."""

    model_config = ConfigDict(extra="ignore")

    decisions: list[MemoryRelationDecision]


class MemorySupportValidationResponse(StructuredResponseModel):
    """Schema proving whether revised dependencies still support a claim."""

    model_config = ConfigDict(extra="ignore")

    supported: bool
    reason: str = Field(default="", max_length=1000)
    evidence_quote: str = Field(default="", max_length=4000)


class EntityValidationResponse(StructuredResponseModel):
    """Schema returned by entity-match validation."""

    model_config = ConfigDict(extra="ignore")

    same_entity: bool = False
    matched_id: int | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str | None = None


class EntityBatchValidationDecision(StructuredResponseModel):
    """One attributable decision for a bounded entity-mention candidate set."""

    model_config = ConfigDict(extra="forbid")

    mention: str = Field(min_length=1)
    matched_id: int | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = Field(default="", max_length=1000)


class EntityBatchValidationResponse(StructuredResponseModel):
    """Schema returned by one batched entity ambiguity adjudication call."""

    model_config = ConfigDict(extra="forbid")

    decisions: list[EntityBatchValidationDecision]


class QueryEntityDetectionResponse(StructuredResponseModel):
    """Schema returned by query entity detection."""

    model_config = ConfigDict(extra="ignore")

    entity_ids: list[int] = Field(default_factory=list)


class RerankResponse(StructuredResponseModel):
    """Schema returned by memory reranking."""

    model_config = ConfigDict(extra="ignore")

    ranking: list[int] = Field(default_factory=list)


@dataclass(frozen=True)
class StructuredLlmConfig:
    model: str
    base_url: str | None
    api_key: str | None
    timeout_s: float
    # One logical-call-wide budget for transient 408/409/429/5xx or connection
    # failures. The adapter owns these retries so fallback shares the same
    # deadline and attempt telemetry remains exact.
    num_retries: int = 2


@dataclass(frozen=True)
class StructuredLlmCallTelemetry:
    """Content-free outcome for one complete logical structured call."""

    operation: str
    attempt_count: int
    retry_count: int
    fallback_count: int
    final_mode: Literal["native_schema", "json_text"]
    elapsed_ms: int
    terminal_category: StructuredLlmTerminalCategory
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


@dataclass(frozen=True)
class StructuredLlmMetricsSummary:
    """Content-free aggregate for one bounded Source Unit lifecycle."""

    logical_calls: int
    provider_attempts: int
    retries: int
    schema_fallbacks: int
    reported_input_tokens: int
    reported_output_tokens: int
    reported_total_tokens: int
    usage_known_calls: int
    usage_unknown_calls: int
    llm_elapsed_ms: int
    source_unit_elapsed_ms: int
    terminal_category_counts: Mapping[str, int]
    operation_counts: Mapping[str, int]

    def to_payload(self) -> dict[str, object]:
        return {
            "logical_calls": self.logical_calls,
            "provider_attempts": self.provider_attempts,
            "retries": self.retries,
            "schema_fallbacks": self.schema_fallbacks,
            "reported_input_tokens": self.reported_input_tokens,
            "reported_output_tokens": self.reported_output_tokens,
            "reported_total_tokens": self.reported_total_tokens,
            "usage_known_calls": self.usage_known_calls,
            "usage_unknown_calls": self.usage_unknown_calls,
            "llm_elapsed_ms": self.llm_elapsed_ms,
            "source_unit_elapsed_ms": self.source_unit_elapsed_ms,
            "terminal_category_counts": dict(self.terminal_category_counts),
            "operation_counts": dict(self.operation_counts),
        }


class StructuredLlmMetricsCollector:
    """Collect logical-call outcomes for one request-local lifecycle scope."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._calls: list[StructuredLlmCallTelemetry] = []

    def record(self, telemetry: StructuredLlmCallTelemetry) -> None:
        with self._lock:
            self._calls.append(telemetry)

    def summary(self, *, source_unit_elapsed_ms: int) -> StructuredLlmMetricsSummary:
        with self._lock:
            calls = tuple(self._calls)

        terminal_category_counts: dict[str, int] = {}
        operation_counts: dict[str, int] = {}
        reported_input_tokens = 0
        reported_output_tokens = 0
        reported_total_tokens = 0
        usage_known_calls = 0
        for call in calls:
            terminal_category_counts[call.terminal_category] = (
                terminal_category_counts.get(call.terminal_category, 0) + 1
            )
            operation_counts[call.operation] = operation_counts.get(call.operation, 0) + 1
            if call.prompt_tokens is not None and call.completion_tokens is not None and call.total_tokens is not None:
                usage_known_calls += 1
                reported_input_tokens += call.prompt_tokens
                reported_output_tokens += call.completion_tokens
                reported_total_tokens += call.total_tokens

        return StructuredLlmMetricsSummary(
            logical_calls=len(calls),
            provider_attempts=sum(call.attempt_count for call in calls),
            retries=sum(call.retry_count for call in calls),
            schema_fallbacks=sum(call.fallback_count for call in calls),
            reported_input_tokens=reported_input_tokens,
            reported_output_tokens=reported_output_tokens,
            reported_total_tokens=reported_total_tokens,
            usage_known_calls=usage_known_calls,
            usage_unknown_calls=len(calls) - usage_known_calls,
            llm_elapsed_ms=sum(call.elapsed_ms for call in calls),
            source_unit_elapsed_ms=max(0, int(source_unit_elapsed_ms)),
            terminal_category_counts=dict(sorted(terminal_category_counts.items())),
            operation_counts=dict(sorted(operation_counts.items())),
        )


@dataclass
class _StructuredCallState:
    operation: str
    retry_budget: int
    attempt_count: int = 0
    retry_count: int = 0
    fallback_count: int = 0
    final_mode: Literal["native_schema", "json_text"] = "native_schema"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    usage_complete: bool = True
    usage_seen: bool = False

    def record_response(self, response: object) -> None:
        usage = _response_usage(response)
        if usage is None:
            self.usage_complete = False
            return
        self.usage_seen = True
        self.prompt_tokens += usage[0]
        self.completion_tokens += usage[1]
        self.total_tokens += usage[2]

    def record_failed_attempt(self) -> None:
        # A provider may have consumed tokens before surfacing an error. Without
        # an explicit usage object the logical total is unknown, never estimated.
        self.usage_complete = False

    def telemetry(
        self,
        *,
        elapsed_ms: int,
        terminal_category: StructuredLlmTerminalCategory,
    ) -> StructuredLlmCallTelemetry:
        usage_known = self.usage_complete and self.usage_seen
        return StructuredLlmCallTelemetry(
            operation=self.operation,
            attempt_count=self.attempt_count,
            retry_count=self.retry_count,
            fallback_count=self.fallback_count,
            final_mode=self.final_mode,
            elapsed_ms=elapsed_ms,
            terminal_category=terminal_category,
            prompt_tokens=self.prompt_tokens if usage_known else None,
            completion_tokens=self.completion_tokens if usage_known else None,
            total_tokens=self.total_tokens if usage_known else None,
        )


class SourceSupportStructuredClient(Protocol):
    async def verify_source_support(
        self,
        prompt: str,
        *,
        model: str | None = None,
    ) -> SourceSupportResponse:
        """Return schema-validated source-support decisions."""

    async def extract_memories(
        self,
        prompt: str,
        *,
        max_tokens: int,
        model: str | None = None,
    ) -> MemoryExtractionResponse:
        """Return schema-validated extracted memory candidates."""

    async def select_memory_candidates(
        self,
        prompt: str,
        *,
        max_tokens: int = 8192,
        model: str | None = None,
    ) -> CandidateLedgerResponse:
        """Return a complete within-revision candidate uniqueness ledger."""

    async def reconcile_memories(
        self,
        prompt: str,
        *,
        max_tokens: int = 4096,
        model: str | None = None,
    ) -> ReconciliationResponse:
        """Return schema-validated same-document reconciliation decisions."""

    async def classify_memory_relations(
        self,
        prompt: str,
        *,
        max_tokens: int = 32_768,
        model: str | None = None,
    ) -> MemoryRelationResponse:
        """Return exact, directed relationship decisions for Memory pairs."""

    async def validate_memory_support(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        model: str | None = None,
    ) -> MemorySupportValidationResponse:
        """Prove whether current Primary and Required evidence support a claim."""

    async def validate_entity_match(
        self,
        prompt: str,
        *,
        max_tokens: int = 200,
        model: str | None = None,
    ) -> EntityValidationResponse:
        """Return schema-validated entity validation."""

    async def validate_entity_batch(
        self,
        prompt: str,
        *,
        max_tokens: int = 2048,
        model: str | None = None,
    ) -> EntityBatchValidationResponse:
        """Return attributable decisions for bounded entity candidate sets."""

    async def detect_query_entities(
        self,
        prompt: str,
        *,
        max_tokens: int = 64,
        model: str | None = None,
    ) -> QueryEntityDetectionResponse:
        """Return schema-validated query entity ids."""

    async def rerank_memories(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        model: str | None = None,
    ) -> RerankResponse:
        """Return schema-validated reranking indices."""

    async def generate_agent_knowledge_patch(
        self,
        prompt: str,
        *,
        max_tokens: int = 2048,
        model: str | None = None,
    ):
        """Return a private agent-knowledge patch proposal."""

    async def classify_agent_session_evidence_authority(
        self,
        prompt: str,
        *,
        max_tokens: int = 4096,
        model: str | None = None,
    ) -> AgentSessionAuthorityResponse:
        """Return semantic authority decisions for candidate user evidence."""


class StructuredLlmError(RuntimeError):
    """Raised when a required structured LLM call cannot produce valid schema output."""

    def __init__(
        self,
        message: str,
        *,
        terminal_category: StructuredLlmTerminalCategory = "invalid_response",
    ) -> None:
        super().__init__(message)
        self.terminal_category = terminal_category


def _message_content(response) -> object:
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise StructuredLlmError(f"missing structured response content: {exc}") from exc
    if content is None:
        raise StructuredLlmError("missing structured response content")
    return content


def _response_usage(response: object) -> tuple[int, int, int] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None

    def value(name: str) -> int | None:
        raw = usage.get(name) if isinstance(usage, Mapping) else getattr(usage, name, None)
        if raw is None:
            return None
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return None

    prompt_tokens = value("prompt_tokens")
    completion_tokens = value("completion_tokens")
    total_tokens = value("total_tokens")
    if prompt_tokens is None or completion_tokens is None or total_tokens is None:
        return None
    return prompt_tokens, completion_tokens, total_tokens


def _schema_operation_name(response_format: type[BaseModel]) -> str:
    name = response_format.__name__.removesuffix("Response")
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _provider_status_code(exc: BaseException) -> int | None:
    raw = getattr(exc, "status_code", None)
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _is_retryable_provider_error(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    status_code = _provider_status_code(exc)
    return status_code in {408, 409, 429} or bool(status_code is not None and status_code >= 500)


def _is_non_fallback_provider_error(exc: BaseException) -> bool:
    if isinstance(exc, StructuredLlmError):
        return exc.terminal_category in {"deadline_exceeded", "provider_error"}
    if _is_retryable_provider_error(exc):
        return True
    status_code = _provider_status_code(exc)
    return status_code in {401, 403, 404}


def litellm_model_name(model: str) -> str:
    """Map existing model names into LiteLLM provider/model notation."""
    if "/" in model:
        return model
    return f"anthropic/{model}"


def _json_text_prompt(prompt: str, response_format: type[BaseModel]) -> str:
    """Append the schema as a text instruction for the no-tool JSON path."""
    schema = json.dumps(response_format.model_json_schema(), ensure_ascii=False)
    return (
        f"{prompt}\n\nReturn ONLY a single JSON object that matches this JSON Schema, "
        f"with no markdown fences and no commentary:\n{schema}"
    )


def _supports_native_response_schema(model_name: str) -> bool:
    if _is_sap_anthropic_model(model_name):
        return True
    try:
        return bool(litellm.supports_response_schema(model=model_name))
    except Exception:
        logger.debug(
            "Unable to determine native response_schema support for model %s",
            model_name,
            exc_info=True,
        )
        return False


def _is_sap_anthropic_model(model_name: str) -> bool:
    """Return true for SAP GenAI Hub Anthropic aliases that accept response_format."""
    return model_name.lower().startswith("sap/anthropic--")


def _strip_json_fences(text: str) -> str:
    """Drop a leading ```/```json fence and trailing ``` if the model adds them."""
    stripped = text.strip()
    if stripped.startswith("```"):
        newline = stripped.find("\n")
        stripped = stripped[newline + 1 :] if newline != -1 else stripped[3:]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[: -len("```")]
    return stripped.strip()


_INVALID_JSON_ESCAPE_RE = re.compile(r'\\(?!["\\/bfnrtu])')


def _escape_invalid_json_backslashes(text: str) -> str:
    """Preserve literal backslashes that models sometimes emit inside JSON strings."""
    return _INVALID_JSON_ESCAPE_RE.sub(r"\\\\", text)


def _validate_structured_json_text(text: str, response_format: type[BaseModel]):
    stripped = _strip_json_fences(text)
    try:
        return response_format.model_validate_json(stripped)
    except ValidationError as exc:
        repaired = _escape_invalid_json_backslashes(stripped)
        if repaired != stripped and "Invalid JSON" in str(exc):
            try:
                return response_format.model_validate_json(repaired)
            except ValidationError:
                pass

        valid_objects = []
        decoder = json.JSONDecoder()
        cursor = 0
        while (start := stripped.find("{", cursor)) != -1:
            try:
                candidate, end = decoder.raw_decode(stripped, start)
            except json.JSONDecodeError:
                cursor = start + 1
                continue
            cursor = end
            if not isinstance(candidate, dict):
                continue
            try:
                valid_objects.append(response_format.model_validate(candidate))
            except ValidationError:
                continue
        if len(valid_objects) == 1:
            logger.warning(
                "Structured LLM output for schema %s contained non-JSON framing; "
                "recovered exactly one schema-valid JSON object",
                response_format.__name__,
            )
            return valid_objects[0]
        if len(valid_objects) > 1:
            raise ValueError("ambiguous structured JSON objects") from exc
        raise


class LiteLlmStructuredClient:
    """LiteLLM-backed structured client.

    Native response schemas are the preferred path because gateway aliases can
    enforce them even when LiteLLM's model registry does not recognize the
    alias. If a gateway rejects schema output, the client falls back once to a
    plain JSON prompt with the same schema. Both paths validate against the same
    pydantic model before returning to callers.
    """

    def __init__(
        self,
        config: StructuredLlmConfig,
        *,
        telemetry_sink: Callable[[StructuredLlmCallTelemetry], None] | None = None,
    ) -> None:
        self.config = config
        self._telemetry_sink = telemetry_sink
        self._scoped_metrics_collector: ContextVar[StructuredLlmMetricsCollector | None] = ContextVar(
            f"memforge_structured_llm_metrics_collector_{id(self)}",
            default=None,
        )

    @contextmanager
    def metrics_scope(
        self,
        collector: StructuredLlmMetricsCollector,
    ) -> Iterator[StructuredLlmMetricsCollector]:
        """Route calls in the current async context to one request-local collector."""

        token = self._scoped_metrics_collector.set(collector)
        try:
            yield collector
        finally:
            self._scoped_metrics_collector.reset(token)

    async def verify_source_support(
        self,
        prompt: str,
        *,
        model: str | None = None,
    ) -> SourceSupportResponse:
        return await self._call_schema(
            prompt=prompt,
            response_format=SourceSupportResponse,
            max_tokens=4096,
            model=model,
        )

    async def extract_memories(
        self,
        prompt: str,
        *,
        max_tokens: int,
        model: str | None = None,
    ) -> MemoryExtractionResponse:
        return await self._call_schema(
            prompt=prompt,
            response_format=MemoryExtractionResponse,
            max_tokens=max_tokens,
            model=model,
        )

    async def select_memory_candidates(
        self,
        prompt: str,
        *,
        max_tokens: int = 8192,
        model: str | None = None,
    ) -> CandidateLedgerResponse:
        return await self._call_schema(
            prompt=prompt,
            response_format=CandidateLedgerResponse,
            max_tokens=max_tokens,
            model=model,
        )

    async def reconcile_memories(
        self,
        prompt: str,
        *,
        max_tokens: int = 4096,
        model: str | None = None,
    ) -> ReconciliationResponse:
        return await self._call_schema(
            prompt=prompt,
            response_format=ReconciliationResponse,
            max_tokens=max_tokens,
            model=model,
        )

    async def classify_memory_relations(
        self,
        prompt: str,
        *,
        max_tokens: int = 32_768,
        model: str | None = None,
    ) -> MemoryRelationResponse:
        return await self._call_schema(
            prompt=prompt,
            response_format=MemoryRelationResponse,
            max_tokens=max_tokens,
            model=model,
        )

    async def validate_memory_support(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        model: str | None = None,
    ) -> MemorySupportValidationResponse:
        return await self._call_schema(
            prompt=prompt,
            response_format=MemorySupportValidationResponse,
            max_tokens=max_tokens,
            model=model,
        )

    async def validate_entity_match(
        self,
        prompt: str,
        *,
        max_tokens: int = 200,
        model: str | None = None,
    ) -> EntityValidationResponse:
        return await self._call_schema(
            prompt=prompt,
            response_format=EntityValidationResponse,
            max_tokens=max_tokens,
            model=model,
        )

    async def validate_entity_batch(
        self,
        prompt: str,
        *,
        max_tokens: int = 2048,
        model: str | None = None,
    ) -> EntityBatchValidationResponse:
        return await self._call_schema(
            prompt=prompt,
            response_format=EntityBatchValidationResponse,
            max_tokens=max_tokens,
            model=model,
        )

    async def detect_query_entities(
        self,
        prompt: str,
        *,
        max_tokens: int = 64,
        model: str | None = None,
    ) -> QueryEntityDetectionResponse:
        return await self._call_schema(
            prompt=prompt,
            response_format=QueryEntityDetectionResponse,
            max_tokens=max_tokens,
            model=model,
        )

    async def rerank_memories(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        model: str | None = None,
    ) -> RerankResponse:
        return await self._call_schema(
            prompt=prompt,
            response_format=RerankResponse,
            max_tokens=max_tokens,
            model=model,
        )

    async def generate_agent_knowledge_patch(
        self,
        prompt: str,
        *,
        max_tokens: int = 2048,
        model: str | None = None,
    ):
        from memforge.agent_knowledge import AgentKnowledgePatchProposal

        return await self._call_schema(
            prompt=prompt,
            response_format=AgentKnowledgePatchProposal,
            max_tokens=max_tokens,
            model=model,
        )

    async def classify_agent_session_evidence_authority(
        self,
        prompt: str,
        *,
        max_tokens: int = 4096,
        model: str | None = None,
    ) -> AgentSessionAuthorityResponse:
        return await self._call_schema(
            prompt=prompt,
            response_format=AgentSessionAuthorityResponse,
            max_tokens=max_tokens,
            model=model,
        )

    async def _call_schema(
        self,
        *,
        prompt: str,
        response_format: type[BaseModel],
        max_tokens: int,
        model: str | None = None,
        retry_with_json_text: bool = True,
    ):
        model_name = litellm_model_name(model or self.config.model)
        started = perf_counter()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.0, self.config.timeout_s)
        state = _StructuredCallState(
            operation=_schema_operation_name(response_format),
            retry_budget=max(0, self.config.num_retries),
        )
        try:
            async with asyncio.timeout_at(deadline):
                result = await self._call_schema_with_deadline(
                    prompt=prompt,
                    response_format=response_format,
                    max_tokens=max_tokens,
                    model_name=model_name,
                    retry_with_json_text=retry_with_json_text,
                    deadline=deadline,
                    state=state,
                )
        except TimeoutError as exc:
            self._emit_telemetry(
                state.telemetry(
                    elapsed_ms=max(0, round((perf_counter() - started) * 1000)),
                    terminal_category="deadline_exceeded",
                )
            )
            raise StructuredLlmError(
                f"structured LLM logical deadline exceeded after {self.config.timeout_s:g}s",
                terminal_category="deadline_exceeded",
            ) from exc
        except StructuredLlmError as exc:
            self._emit_telemetry(
                state.telemetry(
                    elapsed_ms=max(0, round((perf_counter() - started) * 1000)),
                    terminal_category=exc.terminal_category,
                )
            )
            raise
        except Exception as exc:
            category = "provider_error" if _is_non_fallback_provider_error(exc) else "invalid_response"
            self._emit_telemetry(
                state.telemetry(
                    elapsed_ms=max(0, round((perf_counter() - started) * 1000)),
                    terminal_category=category,
                )
            )
            raise StructuredLlmError(str(exc), terminal_category=category) from exc

        self._emit_telemetry(
            state.telemetry(
                elapsed_ms=max(0, round((perf_counter() - started) * 1000)),
                terminal_category="success",
            )
        )
        return result

    async def _call_schema_with_deadline(
        self,
        *,
        prompt: str,
        response_format: type[BaseModel],
        max_tokens: int,
        model_name: str,
        retry_with_json_text: bool,
        deadline: float,
        state: _StructuredCallState,
    ):
        if not _supports_native_response_schema(model_name):
            state.final_mode = "json_text"
            logger.debug(
                "Structured LLM model %s does not advertise native response_schema support; "
                "using JSON-text schema for %s",
                model_name,
                response_format.__name__,
            )
            try:
                return await self._attempt_schema(
                    prompt=prompt,
                    response_format=response_format,
                    model_name=model_name,
                    max_tokens=max_tokens,
                    native_schema=False,
                    deadline=deadline,
                    state=state,
                )
            except Exception as exc:
                category = "provider_error" if _is_non_fallback_provider_error(exc) else "invalid_response"
                raise StructuredLlmError(str(exc), terminal_category=category) from exc

        state.final_mode = "native_schema"
        try:
            return await self._attempt_schema(
                prompt=prompt,
                response_format=response_format,
                model_name=model_name,
                max_tokens=max_tokens,
                native_schema=True,
                deadline=deadline,
                state=state,
            )
        except Exception as schema_exc:
            category = "provider_error" if _is_non_fallback_provider_error(schema_exc) else "invalid_response"
            if not retry_with_json_text or category == "provider_error":
                raise StructuredLlmError(
                    str(schema_exc),
                    terminal_category=category,
                ) from schema_exc
            state.fallback_count += 1
            state.final_mode = "json_text"
            logger.warning(
                "Structured LLM response_schema attempt failed for model %s and schema %s; "
                "retrying with JSON-text schema (error_type=%s, category=%s)",
                model_name,
                response_format.__name__,
                type(schema_exc).__name__,
                category,
            )
            try:
                return await self._attempt_schema(
                    prompt=prompt,
                    response_format=response_format,
                    model_name=model_name,
                    max_tokens=max_tokens,
                    native_schema=False,
                    deadline=deadline,
                    state=state,
                )
            except Exception as exc:
                category = "provider_error" if _is_non_fallback_provider_error(exc) else "invalid_response"
                raise StructuredLlmError(
                    f"{exc} (response_schema attempt failed first: {schema_exc})",
                    terminal_category=category,
                ) from exc

    async def _attempt_schema(
        self,
        *,
        prompt: str,
        response_format: type[BaseModel],
        model_name: str,
        max_tokens: int,
        native_schema: bool,
        deadline: float,
        state: _StructuredCallState,
    ):
        request_prompt = prompt if native_schema else _json_text_prompt(prompt, response_format)
        messages = [{"role": "user", "content": request_prompt}]
        provider_kwargs: dict[str, Any] = {}
        if model_name.startswith("sap/"):
            # SAP AI Core treats every chat message as a prompt template. Source
            # documents can legitimately contain examples such as {{?input}};
            # pass the complete MemForge prompt as one placeholder value so
            # nested template syntax remains source data rather than SAP input.
            messages = [{"role": "user", "content": "{{?memforge_prompt}}"}]
            provider_kwargs["placeholder_values"] = {"memforge_prompt": request_prompt}
        response = await self._completion_with_retries(
            model_name=model_name,
            messages=messages,
            max_tokens=max_tokens,
            provider_kwargs=provider_kwargs,
            response_format=response_format if native_schema else None,
            deadline=deadline,
            state=state,
        )
        raw_content = _message_content(response)
        if isinstance(raw_content, response_format):
            return raw_content
        if isinstance(raw_content, dict):
            return response_format.model_validate(raw_content)
        return _validate_structured_json_text(str(raw_content), response_format)

    async def _completion_with_retries(
        self,
        *,
        model_name: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        provider_kwargs: dict[str, Any],
        response_format: type[BaseModel] | None,
        deadline: float,
        state: _StructuredCallState,
    ):
        loop = asyncio.get_running_loop()
        while True:
            remaining_s = max(0.001, deadline - loop.time())
            state.attempt_count += 1
            try:
                response = await litellm.acompletion(
                    model=model_name,
                    messages=messages,
                    timeout=remaining_s,
                    max_tokens=max_tokens,
                    # The adapter owns the logical retry budget so attempt
                    # telemetry is exact and fallback cannot multiply it.
                    num_retries=0,
                    **litellm_optional_kwargs(
                        api_base=self.config.base_url,
                        api_key=self.config.api_key,
                    ),
                    **provider_kwargs,
                    **({"response_format": response_format} if response_format else {}),
                )
            except Exception as exc:
                state.record_failed_attempt()
                if not _is_retryable_provider_error(exc) or state.retry_budget <= 0:
                    if isinstance(exc, TimeoutError):
                        raise StructuredLlmError(
                            str(exc),
                            terminal_category="provider_error",
                        ) from exc
                    raise
                state.retry_budget -= 1
                state.retry_count += 1
                backoff_s = min(0.25 * (2 ** (state.retry_count - 1)), 1.0)
                await asyncio.sleep(min(backoff_s, max(0.0, deadline - loop.time())))
                continue
            state.record_response(response)
            return response

    def _emit_telemetry(self, telemetry: StructuredLlmCallTelemetry) -> None:
        payload = {
            "event": "structured_llm_call",
            "operation": telemetry.operation,
            "attempt_count": telemetry.attempt_count,
            "retry_count": telemetry.retry_count,
            "fallback_count": telemetry.fallback_count,
            "final_mode": telemetry.final_mode,
            "elapsed_ms": telemetry.elapsed_ms,
            "terminal_category": telemetry.terminal_category,
            "prompt_tokens": telemetry.prompt_tokens,
            "completion_tokens": telemetry.completion_tokens,
            "total_tokens": telemetry.total_tokens,
        }
        logger.info("structured_llm_call %s", json.dumps(payload, sort_keys=True, separators=(",", ":")))
        scoped_collector = self._scoped_metrics_collector.get()
        if scoped_collector is not None:
            scoped_collector.record(telemetry)
        if self._telemetry_sink is not None:
            try:
                self._telemetry_sink(telemetry)
            except Exception:
                logger.warning("Structured LLM telemetry sink failed", exc_info=True)
