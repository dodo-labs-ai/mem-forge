"""Structured LLM calls with LiteLLM response schemas."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field as dataclass_field
from threading import Lock
from time import perf_counter
from typing import Annotated, Any, Callable, Iterator, Literal, Mapping, Protocol, get_args, get_origin
from weakref import WeakKeyDictionary

import litellm
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from memforge.llm.providers import litellm_optional_kwargs
from memforge.llm.structured_images import (
    StructuredLlmImage,
    StructuredLlmImageError,
    prepare_structured_llm_images as _prepare_structured_llm_images,
)

logger = logging.getLogger(__name__)

type StructuredLlmTerminalCategory = Literal[
    "success",
    "deadline_exceeded",
    "provider_error",
    "invalid_response",
]
type NativeSchemaTransport = Literal[
    "auto",
    "json_schema_response_format",
]
type TransientEvidenceBlockId = Annotated[
    str,
    Field(min_length=1, pattern=r"^EB-\d{3,}$"),
]


@dataclass(frozen=True, slots=True)
class _StructuredLlmAdmission:
    max_concurrent: int
    semaphore: asyncio.Semaphore


_STRUCTURED_LLM_ADMISSION_LOCK = Lock()
_STRUCTURED_LLM_ADMISSIONS: WeakKeyDictionary[
    asyncio.AbstractEventLoop,
    _StructuredLlmAdmission,
] = WeakKeyDictionary()


def _process_structured_llm_admission(max_concurrent: int) -> _StructuredLlmAdmission:
    """Return the shared structured-call admission for the process event loop."""

    loop = asyncio.get_running_loop()
    requested_limit = max(1, int(max_concurrent))
    with _STRUCTURED_LLM_ADMISSION_LOCK:
        admission = _STRUCTURED_LLM_ADMISSIONS.get(loop)
        if admission is None:
            admission = _StructuredLlmAdmission(
                max_concurrent=requested_limit,
                semaphore=asyncio.Semaphore(requested_limit),
            )
            _STRUCTURED_LLM_ADMISSIONS[loop] = admission
            return admission
        if admission.max_concurrent != requested_limit:
            logger.warning(
                "Ignoring structured LLM concurrency limit %d because event-loop limit %d is already active",
                requested_limit,
                admission.max_concurrent,
            )
        return admission


def structured_llm_max_concurrent(client: object) -> int:
    """Return a client's safe phase fan-out; unknown test/provider clients are serial."""

    try:
        return max(1, int(getattr(client, "max_concurrent", 1)))
    except (TypeError, ValueError):
        return 1


def _structured_user_content(
    prompt: str,
    images: tuple[StructuredLlmImage, ...],
) -> str | list[dict[str, object]]:
    if not images:
        return prompt
    content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
    for image in images:
        encoded = base64.b64encode(image.body).decode("ascii")
        content.extend(
            (
                {
                    "type": "text",
                    "text": (f"Image evidence for Source Observation {image.source_observation_id}:"),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image.media_type};base64,{encoded}",
                    },
                },
            )
        )
    return content


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
    evidence_block_id: TransientEvidenceBlockId
    evidence_anchor: Literal["unit", "glossary", "preamble", "outline", "document", "unknown"] = "unknown"
    required_source_observation_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _reject_model_owned_observation_authority(cls, value: object):
        if isinstance(value, Mapping) and "source_observation_id" in value:
            raise ValueError(
                "textual candidates must select evidence_block_id; "
                "source_observation_id is application-owned"
            )
        return value

    @property
    def source_observation_id(self) -> None:
        """Text authority maps to an Observation only after Block resolution."""

        return None


class ArtifactSelectionSummary(StructuredResponseModel):
    """Untrusted optional selection hint validated independently from Memories."""

    model_config = ConfigDict(extra="ignore")

    # Keep the provider schema simple and typed. The pre-validator isolates
    # malformed optional values as empty entries so they cannot invalidate
    # schema-valid Memory judgments in the same response.
    source_observation_id: str = ""
    summary: str = ""

    @model_validator(mode="before")
    @classmethod
    def _isolate_invalid_entry(cls, value: object) -> Mapping[str, object]:
        if not isinstance(value, Mapping):
            return {}
        return {
            "source_observation_id": (
                value.get("source_observation_id")
                if isinstance(value.get("source_observation_id"), str)
                else ""
            ),
            "summary": (
                value.get("summary")
                if isinstance(value.get("summary"), str)
                else ""
            ),
        }

    @model_validator(mode="after")
    def _normalize(self) -> ArtifactSelectionSummary:
        self.source_observation_id = self.source_observation_id.strip()
        self.summary = " ".join(self.summary.split())
        return self


class MemoryExtractionResponse(StructuredResponseModel):
    """Schema returned by memory extraction."""

    model_config = ConfigDict(extra="ignore")

    memories: list[MemoryCandidate]
    artifact_summaries: list[ArtifactSelectionSummary] = Field(default_factory=list)


class _ProjectionMemoryCandidateBase(StructuredResponseModel):
    """Shared model judgments; evidence authority is selected by a concrete variant."""

    model_config = ConfigDict(extra="ignore")

    content: str = Field(min_length=1)
    memory_type: Literal["fact", "decision", "convention", "procedure"]
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    entity_refs: list[str] = Field(default_factory=list)
    valid_from: str | None = None
    valid_until: str | None = None
    required_source_observation_ids: list[str] = Field(default_factory=list)

    @property
    def extraction_context(self) -> None:
        return None

    @property
    def evidence_anchor(self) -> Literal["unknown"]:
        return "unknown"


class ProjectionTextMemoryCandidate(_ProjectionMemoryCandidateBase):
    """One textual claim bound to an application-owned transient Evidence Block."""

    evidence_quote: str | None = None
    evidence_block_id: TransientEvidenceBlockId

    @model_validator(mode="before")
    @classmethod
    def _reject_model_owned_observation_authority(cls, value: object):
        if isinstance(value, Mapping) and "source_observation_id" in value:
            raise ValueError(
                "textual candidates must select evidence_block_id; "
                "source_observation_id is application-owned"
            )
        return value

    @property
    def source_observation_id(self) -> None:
        """Text authority maps to an Observation only after Block resolution."""

        return None


class ProjectionArtifactMemoryCandidate(_ProjectionMemoryCandidateBase):
    """One visual claim bound to the exact supplied binary Artifact Observation."""

    source_observation_id: str = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _reject_text_authority(cls, value: object):
        if isinstance(value, Mapping):
            forbidden = {
                field
                for field in ("evidence_block_id", "evidence_quote")
                if field in value
            }
            if forbidden:
                raise ValueError(
                    "Artifact candidates cannot provide textual authority: "
                    + ", ".join(sorted(forbidden))
                )
        return value

    @property
    def evidence_quote(self) -> None:
        """Binary Artifact evidence has no textual quote."""

        return None

    @property
    def evidence_block_id(self) -> None:
        """Binary Artifact evidence is addressed by its supplied Observation."""

        return None


type ProjectionMemoryCandidate = (
    ProjectionTextMemoryCandidate | ProjectionArtifactMemoryCandidate
)


class ProjectionMemoryExtractionResponse(StructuredResponseModel):
    """Projection extraction schema containing model judgments only."""

    model_config = ConfigDict(extra="ignore")

    memories: list[ProjectionMemoryCandidate]
    artifact_summaries: list[ArtifactSelectionSummary] = Field(default_factory=list)


class CandidateLedgerDecision(StructuredResponseModel):
    """One ordered uniqueness judgment for a transient extracted candidate."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["KEEP", "DROP_REDUNDANT", "DROP_LOW_VALUE"]
    canonical_index: int | None = Field(default=None, ge=0)
    reason: str = Field(default="", max_length=1000)


class CandidateLedgerResponse(StructuredResponseModel):
    """Ordered decisions for one bounded candidate-ledger batch."""

    model_config = ConfigDict(extra="forbid")

    decisions: list[CandidateLedgerDecision]


class IncumbentSupportAuditDecision(StructuredResponseModel):
    """One factual Source Unit support judgment, without lifecycle semantics."""

    model_config = ConfigDict(extra="forbid")

    supported: bool
    reason: str = Field(default="", max_length=1000)


class IncumbentSupportAuditResponse(StructuredResponseModel):
    """Ordered incumbent side of a composed reconciliation ledger."""

    model_config = ConfigDict(extra="forbid")

    decisions: list[IncumbentSupportAuditDecision]


class RevisionCompositionDecision(StructuredResponseModel):
    """One transient proof that a REFINES pair is eligible for revision."""

    model_config = ConfigDict(extra="forbid")

    pair_index: int = Field(ge=0)
    same_memory_identity: bool
    preserves_incumbent_truth: bool
    candidate_is_canonical_composite: bool
    current_evidence_entails_candidate: bool
    reason: str = Field(default="", max_length=1000)


class RevisionCompositionResponse(StructuredResponseModel):
    """Ordered revision proofs for exact REFINES pairs."""

    model_config = ConfigDict(extra="forbid")

    decisions: list[RevisionCompositionDecision]


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
    same_subject_and_scope: bool
    incompatible_assertions: str = Field(max_length=1000)
    reason: str = Field(default="", max_length=1000)

    @model_validator(mode="after")
    def _validate_direction(self) -> MemoryRelationDecision:
        directional = self.classification == "refines"
        if directional == (self.direction == "symmetric"):
            raise ValueError("REFINES must be directional and other relations symmetric")
        incompatible = self.incompatible_assertions.strip()
        if self.classification == "contradicts":
            if not self.same_subject_and_scope:
                raise ValueError("CONTRADICTS requires the same subject and scope")
            if not incompatible:
                raise ValueError("CONTRADICTS requires the incompatible assertions")
        elif incompatible:
            raise ValueError("only CONTRADICTS may provide incompatible assertions")
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
    """One semantic judgment bound to a datastore-owned response slot."""

    model_config = ConfigDict(extra="forbid")

    matched_id: int | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = Field(default="", max_length=1000)


class EntityBatchValidationResponse(StructuredResponseModel):
    """Ordered decisions for one bounded entity ambiguity adjudication call."""

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


class OfflineSemanticJudgeResponse(StructuredResponseModel):
    """One bounded, content-free decision for an offline semantic criterion."""

    model_config = ConfigDict(extra="forbid")

    verdict: Literal[
        "criterion_satisfied",
        "criterion_not_satisfied",
        "insufficient_evidence",
    ]
    confidence: Literal["low", "medium", "high"]


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
    # Every client in the worker event loop shares this logical-call admission.
    # Callers that do not opt in remain conservatively serial.
    max_concurrent: int = 1
    # ``auto`` follows LiteLLM's provider capability registry. Integrations may
    # instead select an explicit standard wire contract when the registry lags
    # a deployed model, without teaching this provider-neutral client a gateway
    # or model alias.
    native_schema_transport: NativeSchemaTransport = "auto"
    # Some LiteLLM gateways interpret message text as a prompt template. When
    # configured by the deployment adapter, carry the complete prompt as one
    # placeholder value so template-like source text remains data.
    prompt_template_variable: str | None = None


@dataclass(frozen=True)
class StructuredLlmAttemptTelemetry:
    """Content-free diagnostic for one non-conformant provider attempt."""

    attempt_index: int
    structured_mode: Literal["native_schema", "json_text"]
    schema_transport: str
    requested_max_tokens: int
    terminal_category: StructuredLlmTerminalCategory
    error_code: str
    finish_reason: str | None
    stop_reason: str | None
    provider_request_id: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    response_chars: int | None
    response_hash: str | None
    validation_location: str | None
    validation_rule: str | None
    json_error_line: int | None
    json_error_column: int | None


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
    error_code: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    diagnostic_attempts: tuple[StructuredLlmAttemptTelemetry, ...] = ()


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
    error_code_counts: Mapping[str, int]

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
            "error_code_counts": dict(self.error_code_counts),
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
        error_code_counts: dict[str, int] = {}
        reported_input_tokens = 0
        reported_output_tokens = 0
        reported_total_tokens = 0
        usage_known_calls = 0
        for call in calls:
            terminal_category_counts[call.terminal_category] = (
                terminal_category_counts.get(call.terminal_category, 0) + 1
            )
            operation_counts[call.operation] = operation_counts.get(call.operation, 0) + 1
            if call.error_code is not None:
                error_code_counts[call.error_code] = error_code_counts.get(call.error_code, 0) + 1
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
            error_code_counts=dict(sorted(error_code_counts.items())),
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
    diagnostic_attempts: list[StructuredLlmAttemptTelemetry] = dataclass_field(
        default_factory=list
    )

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

    def record_provider_failure_attempt(
        self,
        *,
        attempt_index: int,
        structured_mode: Literal["native_schema", "json_text"],
        schema_transport: str,
        requested_max_tokens: int,
        failure: _StructuredLlmFailure,
    ) -> None:
        self.diagnostic_attempts.append(
            StructuredLlmAttemptTelemetry(
                attempt_index=attempt_index,
                structured_mode=structured_mode,
                schema_transport=schema_transport,
                requested_max_tokens=requested_max_tokens,
                terminal_category=failure.terminal_category,
                error_code=failure.error_code,
                finish_reason=None,
                stop_reason=None,
                provider_request_id=None,
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                response_chars=None,
                response_hash=None,
                validation_location=(failure.validation_fields[0][0] if failure.validation_fields else None),
                validation_rule=(failure.validation_fields[0][1] if failure.validation_fields else None),
                json_error_line=None,
                json_error_column=None,
            )
        )

    def record_invalid_response_attempt(
        self,
        response: object,
        *,
        attempt_index: int,
        structured_mode: Literal["native_schema", "json_text"],
        schema_transport: str,
        requested_max_tokens: int,
        exc: BaseException,
    ) -> None:
        failure = _structured_failure(exc, terminal_category="invalid_response")
        usage = _response_usage(response)
        response_chars, response_hash = _response_content_fingerprint(response)
        json_error_line, json_error_column = _safe_json_error_position(exc)
        self.diagnostic_attempts.append(
            StructuredLlmAttemptTelemetry(
                attempt_index=attempt_index,
                structured_mode=structured_mode,
                schema_transport=schema_transport,
                requested_max_tokens=requested_max_tokens,
                terminal_category=failure.terminal_category,
                error_code=failure.error_code,
                finish_reason=_response_finish_reason(response),
                stop_reason=_response_stop_reason(response),
                provider_request_id=_response_provider_request_id(response),
                prompt_tokens=usage[0] if usage is not None else None,
                completion_tokens=usage[1] if usage is not None else None,
                total_tokens=usage[2] if usage is not None else None,
                response_chars=response_chars,
                response_hash=response_hash,
                validation_location=(failure.validation_fields[0][0] if failure.validation_fields else None),
                validation_rule=(failure.validation_fields[0][1] if failure.validation_fields else None),
                json_error_line=json_error_line,
                json_error_column=json_error_column,
            )
        )

    def telemetry(
        self,
        *,
        elapsed_ms: int,
        terminal_category: StructuredLlmTerminalCategory,
        error_code: str | None = None,
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
            error_code=error_code,
            prompt_tokens=self.prompt_tokens if usage_known else None,
            completion_tokens=self.completion_tokens if usage_known else None,
            total_tokens=self.total_tokens if usage_known else None,
            diagnostic_attempts=tuple(self.diagnostic_attempts),
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
        images: tuple[StructuredLlmImage, ...] = (),
    ) -> MemoryExtractionResponse:
        """Return schema-validated extracted memory candidates."""

    async def extract_projection_memories(
        self,
        prompt: str,
        *,
        max_tokens: int,
        model: str | None = None,
        images: tuple[StructuredLlmImage, ...] = (),
    ) -> ProjectionMemoryExtractionResponse:
        """Return projection judgments without datastore-owned anchor fields."""

    async def select_memory_candidates(
        self,
        prompt: str,
        *,
        max_tokens: int = 8192,
        model: str | None = None,
    ) -> CandidateLedgerResponse:
        """Return one bounded candidate-admission ledger batch."""

    async def audit_incumbent_support(
        self,
        prompt: str,
        *,
        max_tokens: int = 4096,
        model: str | None = None,
    ) -> IncumbentSupportAuditResponse:
        """Return one support disposition for every incumbent in an audit batch."""

    async def prove_revision_compositions(
        self,
        prompt: str,
        *,
        max_tokens: int = 4096,
        model: str | None = None,
    ) -> RevisionCompositionResponse:
        """Return transient revision-eligibility proofs for REFINES pairs."""

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
        error_code: str = "structured_llm_error",
        validation_fields: tuple[tuple[str, str], ...] = (),
    ) -> None:
        super().__init__(message)
        self.terminal_category = terminal_category
        self.error_code = error_code
        self.validation_fields = validation_fields


@dataclass(frozen=True, slots=True)
class _StructuredLlmFailure:
    """Content-free failure value that can outlive provider call frames."""

    terminal_category: StructuredLlmTerminalCategory
    error_code: str
    validation_fields: tuple[tuple[str, str], ...] = ()

    def to_error(self, *, timeout_s: float | None = None) -> StructuredLlmError:
        if self.terminal_category == "deadline_exceeded":
            message = (
                f"structured LLM logical deadline exceeded after {timeout_s:g}s"
                if timeout_s is not None
                else "structured LLM logical deadline exceeded"
            )
        elif self.terminal_category == "provider_error":
            message = "structured LLM provider request failed"
        else:
            message = "structured LLM returned an invalid response"
        message = f"{message} (code={self.error_code})"
        return StructuredLlmError(
            message,
            terminal_category=self.terminal_category,
            error_code=self.error_code,
            validation_fields=self.validation_fields,
        )


_SAFE_PROVIDER_ERROR_PATTERNS = (
    (
        "payload_too_large",
        re.compile(
            r"status(?:_code)?[=: ]+413|HTTP/\S+ 413|request entity too large|"
            r"payload too large|body too large",
            re.IGNORECASE,
        ),
    ),
    (
        "remote_disconnect",
        re.compile(
            r"RemoteProtocolError|server disconnected|peer closed|connection reset|"
            r"connection closed|unexpected EOF",
            re.IGNORECASE,
        ),
    ),
    (
        "connect_timeout",
        re.compile(r"ConnectTimeout|connect timeout", re.IGNORECASE),
    ),
    (
        "read_timeout",
        re.compile(r"ReadTimeout|read timeout|timed out while reading", re.IGNORECASE),
    ),
    (
        "tls_error",
        re.compile(r"SSLError|certificate verify failed|TLSV1_ALERT", re.IGNORECASE),
    ),
    (
        "dns_error",
        re.compile(
            r"gaierror|name resolution|name or service not known|"
            r"nodename nor servname",
            re.IGNORECASE,
        ),
    ),
)


def _safe_provider_error_code(exc: BaseException) -> str:
    """Return a bounded content-free provider failure code."""

    outer_code = type(exc).__name__
    if outer_code != "APIConnectionError":
        return outer_code

    current: BaseException | None = exc
    seen: set[int] = set()
    for _ in range(4):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        type_name = type(current).__name__
        for detail, pattern in _SAFE_PROVIDER_ERROR_PATTERNS:
            if pattern.search(type_name):
                return f"{outer_code}.{detail}"
        current = current.__cause__ or current.__context__

    # LiteLLM 1.86 flattens many transport exceptions into the outer message
    # without preserving a cause. Inspect only a bounded prefix and persist
    # only the matched category, never the provider text itself.
    message_prefix = str(exc)[:2048]
    for detail, pattern in _SAFE_PROVIDER_ERROR_PATTERNS:
        if pattern.search(message_prefix):
            return f"{outer_code}.{detail}"
    return outer_code


def _safe_llm_provider(model: str) -> str | None:
    """Derive only the bounded provider prefix exposed by LiteLLM model IDs."""

    if "/" not in model:
        return None
    provider = model.split("/", 1)[0]
    if not provider or len(provider) > 128 or not re.fullmatch(r"[A-Za-z0-9_.:-]+", provider):
        return None
    return provider


def _structured_failure(
    exc: BaseException,
    *,
    terminal_category: StructuredLlmTerminalCategory | None = None,
) -> _StructuredLlmFailure:
    if isinstance(exc, StructuredLlmError):
        return _StructuredLlmFailure(
            terminal_category=terminal_category or exc.terminal_category,
            error_code=exc.error_code,
            validation_fields=exc.validation_fields,
        )
    category = terminal_category
    if category is None:
        category = "provider_error" if _is_non_fallback_provider_error(exc) else "invalid_response"
    return _StructuredLlmFailure(
        terminal_category=category,
        error_code=_safe_provider_error_code(exc),
        validation_fields=_safe_validation_fields(exc),
    )


def _safe_validation_fields(
    exc: BaseException,
) -> tuple[tuple[str, str], ...]:
    """Return only schema field paths and rule types from Pydantic failures."""

    if not isinstance(exc, ValidationError):
        return ()
    fields: list[tuple[str, str]] = []
    for error in exc.errors(include_url=False, include_context=False, include_input=False):
        location_parts = [
            str(part)
            for part in error.get("loc", ())
            if isinstance(part, (str, int))
        ]
        location = ".".join(location_parts) or "$"
        rule_type = str(error.get("type") or "").strip()
        if rule_type:
            fields.append((location, rule_type))
    return tuple(fields)


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


def _object_value(value: object, name: str) -> object | None:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _safe_response_label(value: object, *, max_length: int = 128) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized or len(normalized) > max_length:
        return None
    if not re.fullmatch(r"[A-Za-z0-9_.:/-]+", normalized):
        return None
    return normalized


def _first_response_choice(response: object) -> object | None:
    choices = _object_value(response, "choices")
    if not isinstance(choices, (list, tuple)) or not choices:
        return None
    return choices[0]


def _response_finish_reason(response: object) -> str | None:
    choice = _first_response_choice(response)
    return _safe_response_label(_object_value(choice, "finish_reason"))


def _response_stop_reason(response: object) -> str | None:
    choice = _first_response_choice(response)
    candidates = [
        _object_value(choice, "stop_reason"),
        _object_value(_object_value(choice, "provider_specific_fields"), "stop_reason"),
        _object_value(
            _object_value(_object_value(choice, "message"), "provider_specific_fields"),
            "stop_reason",
        ),
        _object_value(_object_value(response, "provider_specific_fields"), "stop_reason"),
        _object_value(
            _object_value(_object_value(response, "_hidden_params"), "original_response"),
            "stop_reason",
        ),
    ]
    for candidate in candidates:
        if normalized := _safe_response_label(candidate):
            return normalized
    return None


_PROVIDER_REQUEST_HEADER_NAMES = (
    "x-request-id",
    "request-id",
    "apim-request-id",
    "x-amzn-requestid",
    "x-correlation-id",
)


def _response_provider_request_id(response: object) -> str | None:
    hidden = _object_value(response, "_hidden_params")
    headers = _object_value(hidden, "additional_headers")
    if isinstance(headers, Mapping):
        normalized_headers = {str(key).lower(): value for key, value in headers.items()}
        for name in _PROVIDER_REQUEST_HEADER_NAMES:
            if request_id := _safe_response_label(
                normalized_headers.get(name),
                max_length=255,
            ):
                return request_id
    return _safe_response_label(_object_value(response, "id"), max_length=255)


def _response_content_fingerprint(response: object) -> tuple[int | None, str | None]:
    try:
        content = _message_content(response)
    except StructuredLlmError:
        return None, None
    if isinstance(content, BaseModel):
        serialized = content.model_dump_json()
    elif isinstance(content, (dict, list, tuple)):
        serialized = json.dumps(
            content,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    else:
        serialized = str(content)
    return len(serialized), hashlib.sha256(serialized.encode("utf-8")).hexdigest()


_JSON_ERROR_POSITION_RE = re.compile(
    r"\bline\s+(\d+)\s+column\s+(\d+)\b",
    re.IGNORECASE,
)


def _safe_json_error_position(exc: BaseException) -> tuple[int | None, int | None]:
    if not isinstance(exc, ValidationError):
        return None, None
    for error in exc.errors(include_url=False, include_input=False):
        context = error.get("ctx")
        if not isinstance(context, Mapping):
            continue
        match = _JSON_ERROR_POSITION_RE.search(str(context.get("error") or "")[:512])
        if match is not None:
            return int(match.group(1)), int(match.group(2))
    return None, None


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
    try:
        return bool(litellm.supports_response_schema(model=model_name))
    except Exception:
        logger.debug(
            "Unable to determine native response_schema support for model %s",
            model_name,
            exc_info=True,
        )
        return False


def _native_schema_request_kwargs(
    response_format: type[BaseModel] | None,
    transport: NativeSchemaTransport,
) -> dict[str, object]:
    """Build one provider wire contract while retaining local Pydantic authority."""

    if response_format is None:
        return {}
    if transport == "json_schema_response_format":
        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.__name__,
                    "strict": True,
                    "schema": response_format.model_json_schema(),
                },
            }
        }
    return {"response_format": response_format}

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
_MAX_UNESCAPED_JSON_QUOTE_PAIR_REPAIRS = 8
_MAX_UNESCAPED_JSON_QUOTED_TOKEN_CHARS = 128


def _escape_invalid_json_backslashes(text: str) -> str:
    """Preserve literal backslashes that models sometimes emit inside JSON strings."""
    return _INVALID_JSON_ESCAPE_RE.sub(r"\\\\", text)


def _is_unescaped_json_quote(text: str, index: int) -> bool:
    if index < 0 or index >= len(text) or text[index] != '"':
        return False
    backslashes = 0
    cursor = index - 1
    while cursor >= 0 and text[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return backslashes % 2 == 0


def _next_non_whitespace_index(text: str, start: int) -> int | None:
    for index in range(start, len(text)):
        if not text[index].isspace():
            return index
    return None


def _unambiguous_unescaped_json_quote_pair(
    text: str,
    error: json.JSONDecodeError,
) -> tuple[int, int] | None:
    """Locate one clearly internal quoted token after premature string closure.

    The repair is deliberately narrower than a general "JSON repair" parser.
    It accepts only the provider pattern observed in structured string values:
    an unescaped quote immediately before the parser's unexpected token, a
    short non-whitespace/non-structural token, and a second quote followed by
    more string content. A quote before a valid delimiter remains ambiguous and
    is never changed.
    """

    if error.msg != "Expecting ',' delimiter" or error.pos <= 0 or error.pos >= len(text):
        return None
    quote_start = error.pos - 1
    if not _is_unescaped_json_quote(text, quote_start):
        return None
    quote_end = next(
        (index for index in range(error.pos, len(text)) if _is_unescaped_json_quote(text, index)),
        None,
    )
    if quote_end is None:
        return None
    token = text[error.pos : quote_end]
    if (
        not token
        or len(token) > _MAX_UNESCAPED_JSON_QUOTED_TOKEN_CHARS
        or any(character.isspace() or character in '{}[],:\\"' for character in token)
        or any(ord(character) < 32 or ord(character) == 127 for character in token)
    ):
        return None
    following = _next_non_whitespace_index(text, quote_end + 1)
    if following is None or text[following] in ",}]":
        return None
    return quote_start, quote_end


def _repair_unambiguous_unescaped_json_quotes(text: str) -> tuple[str, int] | None:
    candidate = text
    repaired_pairs = 0
    while repaired_pairs < _MAX_UNESCAPED_JSON_QUOTE_PAIR_REPAIRS:
        try:
            json.loads(candidate)
        except json.JSONDecodeError as error:
            pair = _unambiguous_unescaped_json_quote_pair(candidate, error)
            if pair is None:
                return None
            quote_start, quote_end = pair
            candidate = (
                candidate[:quote_start]
                + "\\"
                + candidate[quote_start:quote_end]
                + "\\"
                + candidate[quote_end:]
            )
            repaired_pairs += 1
        else:
            return (candidate, repaired_pairs) if repaired_pairs else None
    return None


def _validate_structured_json_text(text: str, response_format: type[BaseModel]):
    stripped = _strip_json_fences(text)
    try:
        return response_format.model_validate_json(stripped)
    except ValidationError as exc:
        repaired_schema_error: ValidationError | None = None
        repaired = _escape_invalid_json_backslashes(stripped)
        if repaired != stripped and "Invalid JSON" in str(exc):
            try:
                return response_format.model_validate_json(repaired)
            except ValidationError:
                pass

        quote_repair = _repair_unambiguous_unescaped_json_quotes(repaired)
        if quote_repair is not None and "Invalid JSON" in str(exc):
            quote_repaired, repaired_pairs = quote_repair
            try:
                recovered = response_format.model_validate_json(quote_repaired)
            except ValidationError as repair_exc:
                repaired_schema_error = repair_exc
            else:
                logger.warning(
                    "structured_json_recovery %s",
                    json.dumps(
                        {
                            "event": "structured_json_recovery",
                            "recovery_kind": "unescaped_json_string_quotes",
                            "repaired_pairs": repaired_pairs,
                            "schema": response_format.__name__,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                )
                return recovered

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
        if repaired_schema_error is not None:
            raise repaired_schema_error
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

    @property
    def max_concurrent(self) -> int:
        """Maximum phase fan-out; final admission is shared across all clients."""

        return max(1, int(self.config.max_concurrent))

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
        images: tuple[StructuredLlmImage, ...] = (),
    ) -> MemoryExtractionResponse:
        return await self._call_schema(
            prompt=prompt,
            response_format=MemoryExtractionResponse,
            max_tokens=max_tokens,
            model=model,
            images=images,
        )

    async def extract_projection_memories(
        self,
        prompt: str,
        *,
        max_tokens: int,
        model: str | None = None,
        images: tuple[StructuredLlmImage, ...] = (),
    ) -> ProjectionMemoryExtractionResponse:
        return await self._call_schema(
            prompt=prompt,
            response_format=ProjectionMemoryExtractionResponse,
            max_tokens=max_tokens,
            model=model,
            images=images,
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

    async def audit_incumbent_support(
        self,
        prompt: str,
        *,
        max_tokens: int = 4096,
        model: str | None = None,
    ) -> IncumbentSupportAuditResponse:
        return await self._call_schema(
            prompt=prompt,
            response_format=IncumbentSupportAuditResponse,
            max_tokens=max_tokens,
            model=model,
        )

    async def prove_revision_compositions(
        self,
        prompt: str,
        *,
        max_tokens: int = 4096,
        model: str | None = None,
    ) -> RevisionCompositionResponse:
        return await self._call_schema(
            prompt=prompt,
            response_format=RevisionCompositionResponse,
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

    async def judge_offline_semantics(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        model: str | None = None,
    ) -> OfflineSemanticJudgeResponse:
        return await self._call_schema(
            prompt=prompt,
            response_format=OfflineSemanticJudgeResponse,
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
        images: tuple[StructuredLlmImage, ...] = (),
    ):
        admission = _process_structured_llm_admission(self.config.max_concurrent)
        async with admission.semaphore:
            return await self._call_schema_admitted(
                prompt=prompt,
                response_format=response_format,
                max_tokens=max_tokens,
                model=model,
                retry_with_json_text=retry_with_json_text,
                images=images,
            )

    async def _call_schema_admitted(
        self,
        *,
        prompt: str,
        response_format: type[BaseModel],
        max_tokens: int,
        model: str | None,
        retry_with_json_text: bool,
        images: tuple[StructuredLlmImage, ...],
    ):
        model_name = litellm_model_name(model or self.config.model)
        started = perf_counter()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.0, self.config.timeout_s)
        state = _StructuredCallState(
            operation=_schema_operation_name(response_format),
            retry_budget=max(0, self.config.num_retries),
        )
        failure: _StructuredLlmFailure | None = None
        try:
            async with asyncio.timeout_at(deadline):
                prepared_images = await asyncio.to_thread(
                    _prepare_structured_llm_images,
                    images,
                )
                if prepared_images.images:
                    logger.info(
                        "structured_llm_images %s",
                        json.dumps(
                            {
                                "image_count": len(prepared_images.images),
                                "normalized_count": prepared_images.normalized_count,
                                "original_bytes": prepared_images.original_bytes,
                                "transport_bytes": prepared_images.transport_bytes,
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    )
                result = await self._call_schema_with_deadline(
                    prompt=prompt,
                    response_format=response_format,
                    max_tokens=max_tokens,
                    model_name=model_name,
                    retry_with_json_text=retry_with_json_text,
                    deadline=deadline,
                    state=state,
                    images=prepared_images.images,
                )
        except TimeoutError:
            failure = _StructuredLlmFailure(
                terminal_category="deadline_exceeded",
                error_code="logical_deadline_exceeded",
            )
        except StructuredLlmError as exc:
            failure = _structured_failure(exc)
        except StructuredLlmImageError as exc:
            failure = _StructuredLlmFailure(
                terminal_category="invalid_response",
                error_code=exc.error_code,
            )
        except Exception as exc:
            failure = _structured_failure(exc)

        if failure is not None:
            self._emit_telemetry(
                state.telemetry(
                    elapsed_ms=max(0, round((perf_counter() - started) * 1000)),
                    terminal_category=failure.terminal_category,
                    error_code=failure.error_code,
                )
            )
            raise failure.to_error(timeout_s=self.config.timeout_s)

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
        images: tuple[StructuredLlmImage, ...],
    ):
        native_schema_transport = self.config.native_schema_transport
        if (
            native_schema_transport == "auto"
            and not _supports_native_response_schema(model_name)
        ):
            state.final_mode = "json_text"
            logger.debug(
                "Structured LLM model %s does not advertise native response_schema support; "
                "using JSON-text schema for %s",
                model_name,
                response_format.__name__,
            )
            failure: _StructuredLlmFailure | None = None
            try:
                result = await self._attempt_schema(
                    prompt=prompt,
                    response_format=response_format,
                    model_name=model_name,
                    max_tokens=max_tokens,
                    native_schema=False,
                    native_schema_transport=native_schema_transport,
                    deadline=deadline,
                    state=state,
                    images=images,
                )
            except Exception as exc:
                failure = _structured_failure(exc)
            if failure is not None:
                raise failure.to_error()
            return result

        state.final_mode = "native_schema"
        schema_failure: _StructuredLlmFailure | None = None
        try:
            result = await self._attempt_schema(
                prompt=prompt,
                response_format=response_format,
                model_name=model_name,
                max_tokens=max_tokens,
                native_schema=True,
                native_schema_transport=native_schema_transport,
                deadline=deadline,
                state=state,
                images=images,
            )
        except Exception as exc:
            schema_failure = _structured_failure(exc)
        if schema_failure is None:
            return result
        if not retry_with_json_text or schema_failure.terminal_category == "provider_error":
            raise schema_failure.to_error()

        state.fallback_count += 1
        state.final_mode = "json_text"
        logger.warning(
            "Structured LLM response_schema attempt failed for model %s and schema %s; "
            "retrying with JSON-text schema (error_code=%s, category=%s)",
            model_name,
            response_format.__name__,
            schema_failure.error_code,
            schema_failure.terminal_category,
        )
        fallback_failure: _StructuredLlmFailure | None = None
        try:
            result = await self._attempt_schema(
                prompt=prompt,
                response_format=response_format,
                model_name=model_name,
                max_tokens=max_tokens,
                native_schema=False,
                native_schema_transport=native_schema_transport,
                deadline=deadline,
                state=state,
                images=images,
            )
        except Exception as exc:
            fallback_failure = _structured_failure(exc)
        if (
            fallback_failure is not None
            and fallback_failure.terminal_category == "invalid_response"
            and fallback_failure.error_code == "ValidationError"
            and state.retry_budget > 0
        ):
            state.retry_budget -= 1
            state.retry_count += 1
            logger.warning(
                "Structured LLM JSON-text response was invalid for model %s and schema %s; "
                "retrying once within the logical deadline (error_code=%s)",
                model_name,
                response_format.__name__,
                fallback_failure.error_code,
            )
            try:
                result = await self._attempt_schema(
                    prompt=prompt,
                    response_format=response_format,
                    model_name=model_name,
                    max_tokens=max_tokens,
                    native_schema=False,
                    native_schema_transport=native_schema_transport,
                    deadline=deadline,
                    state=state,
                    images=images,
                )
            except Exception as exc:
                fallback_failure = _structured_failure(exc)
            else:
                fallback_failure = None
        if fallback_failure is not None:
            raise fallback_failure.to_error()
        return result

    async def _attempt_schema(
        self,
        *,
        prompt: str,
        response_format: type[BaseModel],
        model_name: str,
        max_tokens: int,
        native_schema: bool,
        native_schema_transport: NativeSchemaTransport,
        deadline: float,
        state: _StructuredCallState,
        images: tuple[StructuredLlmImage, ...],
    ):
        request_prompt = prompt if native_schema else _json_text_prompt(prompt, response_format)
        messages = [{"role": "user", "content": _structured_user_content(request_prompt, images)}]
        provider_kwargs: dict[str, Any] = {}
        prompt_template_variable = self.config.prompt_template_variable
        if prompt_template_variable:
            messages = [
                {
                    "role": "user",
                    "content": _structured_user_content(
                        f"{{{{?{prompt_template_variable}}}}}", images
                    ),
                }
            ]
            provider_kwargs["placeholder_values"] = {
                prompt_template_variable: request_prompt
            }
        response, attempt_index = await self._completion_with_retries(
            model_name=model_name,
            messages=messages,
            max_tokens=max_tokens,
            provider_kwargs=provider_kwargs,
            response_format=response_format if native_schema else None,
            native_schema_transport=native_schema_transport,
            deadline=deadline,
            state=state,
        )
        structured_mode: Literal["native_schema", "json_text"] = (
            "native_schema" if native_schema else "json_text"
        )
        schema_transport = native_schema_transport if native_schema else "json_text"
        try:
            raw_content = _message_content(response)
            if isinstance(raw_content, response_format):
                return raw_content
            if isinstance(raw_content, dict):
                return response_format.model_validate(raw_content)
            return _validate_structured_json_text(str(raw_content), response_format)
        except Exception as exc:
            state.record_invalid_response_attempt(
                response,
                attempt_index=attempt_index,
                structured_mode=structured_mode,
                schema_transport=schema_transport,
                requested_max_tokens=max_tokens,
                exc=exc,
            )
            raise

    async def _completion_with_retries(
        self,
        *,
        model_name: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        provider_kwargs: dict[str, Any],
        response_format: type[BaseModel] | None,
        native_schema_transport: NativeSchemaTransport,
        deadline: float,
        state: _StructuredCallState,
    ):
        loop = asyncio.get_running_loop()
        schema_kwargs = _native_schema_request_kwargs(
            response_format,
            native_schema_transport,
        )
        while True:
            remaining_s = max(0.001, deadline - loop.time())
            state.attempt_count += 1
            attempt_index = state.attempt_count
            failure: _StructuredLlmFailure | None = None
            retry = False
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
                    **schema_kwargs,
                )
            except Exception as exc:
                state.record_failed_attempt()
                retry = _is_retryable_provider_error(exc) and state.retry_budget > 0
                failure = _structured_failure(
                    exc,
                    terminal_category=(
                        "provider_error" if _is_non_fallback_provider_error(exc) else "invalid_response"
                    ),
                )
                state.record_provider_failure_attempt(
                    attempt_index=attempt_index,
                    structured_mode=(
                        "native_schema" if response_format is not None else "json_text"
                    ),
                    schema_transport=(
                        native_schema_transport if response_format is not None else "json_text"
                    ),
                    requested_max_tokens=max_tokens,
                    failure=failure,
                )
            if failure is not None and not retry:
                raise failure.to_error()
            if retry:
                state.retry_budget -= 1
                state.retry_count += 1
                backoff_s = min(0.25 * (2 ** (state.retry_count - 1)), 1.0)
                await asyncio.sleep(min(backoff_s, max(0.0, deadline - loop.time())))
                continue
            state.record_response(response)
            return response, attempt_index

    def _emit_telemetry(self, telemetry: StructuredLlmCallTelemetry) -> None:
        from memforge.evals.agent_evaluation import QualitySignal, record_quality_signal

        payload = {
            "event": "structured_llm_call",
            "operation": telemetry.operation,
            "attempt_count": telemetry.attempt_count,
            "retry_count": telemetry.retry_count,
            "fallback_count": telemetry.fallback_count,
            "final_mode": telemetry.final_mode,
            "elapsed_ms": telemetry.elapsed_ms,
            "terminal_category": telemetry.terminal_category,
            "error_code": telemetry.error_code,
            "prompt_tokens": telemetry.prompt_tokens,
            "completion_tokens": telemetry.completion_tokens,
            "total_tokens": telemetry.total_tokens,
            "diagnostic_attempt_count": len(telemetry.diagnostic_attempts),
        }
        logger.info("structured_llm_call %s", json.dumps(payload, sort_keys=True, separators=(",", ":")))
        recovered_with_fallback = telemetry.terminal_category == "success" and telemetry.fallback_count > 0
        if telemetry.terminal_category == "success":
            outcome = "degraded" if recovered_with_fallback else "expected"
            reason_code = "schema_fallback_recovered" if recovered_with_fallback else "schema_conformant"
        else:
            outcome = "rejected" if telemetry.terminal_category == "invalid_response" else "failed"
            reason_code = (
                "schema_validation_failed"
                if telemetry.terminal_category == "invalid_response"
                else telemetry.terminal_category
            )
        record_quality_signal(
            QualitySignal(
                event_name="structured_output_outcome",
                outcome=outcome,
                reason_code=reason_code,
                operation=telemetry.operation,
                provider=_safe_llm_provider(self.config.model),
                model=self.config.model,
                attempt_count=telemetry.attempt_count,
                retry_count=telemetry.retry_count,
                fallback_count=telemetry.fallback_count,
                structured_mode=telemetry.final_mode,
                terminal_category=telemetry.terminal_category,
                error_code=telemetry.error_code,
                prompt_tokens=telemetry.prompt_tokens,
                completion_tokens=telemetry.completion_tokens,
                total_tokens=telemetry.total_tokens,
            )
        )
        for attempt in telemetry.diagnostic_attempts:
            attempt_outcome = (
                "rejected" if attempt.terminal_category == "invalid_response" else "failed"
            )
            attempt_reason = (
                "schema_validation_failed"
                if attempt.terminal_category == "invalid_response"
                else attempt.terminal_category
            )
            record_quality_signal(
                QualitySignal(
                    event_name="structured_llm_attempt_outcome",
                    outcome=attempt_outcome,
                    reason_code=attempt_reason,
                    operation=telemetry.operation,
                    provider=_safe_llm_provider(self.config.model),
                    model=self.config.model,
                    attempt_index=attempt.attempt_index,
                    structured_mode=attempt.structured_mode,
                    schema_transport=attempt.schema_transport,
                    requested_max_tokens=attempt.requested_max_tokens,
                    terminal_category=attempt.terminal_category,
                    error_code=attempt.error_code,
                    finish_reason=attempt.finish_reason,
                    stop_reason=attempt.stop_reason,
                    provider_request_id=attempt.provider_request_id,
                    prompt_tokens=attempt.prompt_tokens,
                    completion_tokens=attempt.completion_tokens,
                    total_tokens=attempt.total_tokens,
                    response_chars=attempt.response_chars,
                    response_hash=attempt.response_hash,
                    validation_location=attempt.validation_location,
                    validation_rule=attempt.validation_rule,
                    json_error_line=attempt.json_error_line,
                    json_error_column=attempt.json_error_column,
                )
            )
        scoped_collector = self._scoped_metrics_collector.get()
        if scoped_collector is not None:
            scoped_collector.record(telemetry)
        if self._telemetry_sink is not None:
            try:
                self._telemetry_sink(telemetry)
            except Exception:
                logger.warning("Structured LLM telemetry sink failed", exc_info=True)
