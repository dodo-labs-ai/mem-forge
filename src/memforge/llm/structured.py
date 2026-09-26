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
from typing import Annotated, Any, Callable, Iterator, Literal, Mapping, get_args, get_origin
from weakref import WeakKeyDictionary

import litellm
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from memforge.llm.failure_trace import (
    FailureTraceSink, capture_call, current_capture, local_failure_trace_sink_from_env,
)
from memforge.llm.providers import litellm_optional_kwargs
from memforge.llm.structured_images import (
    StructuredLlmImage,
    StructuredLlmImageError,
    prepare_structured_llm_images as _prepare_structured_llm_images,
)

logger = logging.getLogger(__name__)

# ``invalid_response``: a model response was received and failed validation.
# ``provider_error``: a transient, authorization or capacity failure the provider reported.
# ``request_error``: the call failed without a response to validate for any other reason,
# such as a provider rejection of the request (400) or an unexpected exception.
type StructuredLlmTerminalCategory = Literal[
    "success",
    "cancelled",
    "deadline_exceeded",
    "provider_error",
    "invalid_response",
    "request_error",
]
type NativeSchemaTransport = Literal[
    "auto",
    "json_schema_response_format",
]
# A model call that ended in one of these may succeed when it is sent again.
TRANSIENT_TERMINAL_CATEGORIES: frozenset[str] = frozenset({"provider_error", "deadline_exceeded"})


def failure_retryable(error: BaseException) -> bool:
    """Whether the sync that hit ``error`` retries the work at once; one rule for every stage.

    A failed model call is retried only when it was transient: a provider error
    (rate limit and 5xx included) or a timeout. A request error (a 400, an
    unexpected exception) or an invalid response fails the same way again, so
    it is not retried within the run: the revision stays uncommitted and the
    next sync processes it again. Any other failure states it with its
    ``retryable`` attribute and is retried by default.
    """

    category = getattr(error, "terminal_category", None)
    if category is not None:
        return category in TRANSIENT_TERMINAL_CATEGORIES
    return bool(getattr(error, "retryable", True))


# Request-size failures: a smaller request can succeed where this one cannot,
# so callers split the work instead of resending it unchanged.
INPUT_CAPACITY_EXCEEDED = "input_capacity_exceeded"
PAYLOAD_TOO_LARGE = "payload_too_large"
OUTPUT_TRUNCATED = "output_truncated"
_HTTP_PAYLOAD_TOO_LARGE = 413
_SCHEMA_REPAIR_MAX_VALIDATION_FIELDS = 8
_SCHEMA_REPAIR_LOCATION_CHAR_CAP = 256
_SCHEMA_REPAIR_RULE_CHAR_CAP = 128
_SCHEMA_REPAIR_MESSAGE_CHAR_CAP = 256


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

    def row_error(self) -> str | None:
        """This row's meaning rule; None when it holds.

        Response models validate only the JSON shape of a row. A rule about one
        row's meaning is ``row_error``, which the task checks on that row alone,
        so a row that breaks it is rejected by itself.
        """

        if self.is_authoritative and self.authority_kind == "not_authoritative":
            return "an authoritative decision requires an authoritative authority_kind"
        if not self.is_authoritative and self.authority_kind != "not_authoritative":
            return "a non-authoritative decision requires authority_kind='not_authoritative'"
        return None


class AgentSessionAuthorityResponse(StructuredResponseModel):
    """Schema returned by agent-session authority classification."""

    model_config = ConfigDict(extra="forbid")

    decisions: list[AgentSessionAuthorityDecision]


class ProjectionFragmentMemoryCandidate(StructuredResponseModel):
    """v9 model judgment with catalog-local selectors and no authority fields."""

    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1)
    memory_type: Literal["fact", "decision", "convention", "procedure"]
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    entity_refs: list[str] = Field(default_factory=list)
    valid_from: str | None = None
    valid_until: str | None = None
    # Keep the transport schema structural.  Membership and role are
    # catalog-local facts, so malformed or stale string selectors are rejected
    # candidate-by-candidate by the catalog resolver instead of failing every
    # other valid candidate in the LLM response.
    primary_ref: str = Field(
        description=(
            "Exactly one reference copied unchanged from primary_candidates; "
            "never select from required_only_candidates."
        )
    )
    required_refs: list[str] = Field(
        default_factory=list,
        description=(
            "A duplicate-free list of references copied unchanged from "
            "primary_candidates or required_only_candidates; do not repeat primary_ref."
        ),
    )

class ProjectionFragmentMemoryExtractionResponse(StructuredResponseModel):
    """projection-extraction-v9 response containing model judgments only."""

    model_config = ConfigDict(extra="forbid")

    memories: list[ProjectionFragmentMemoryCandidate]


class ProjectionFragmentSelectorCorrection(StructuredResponseModel):
    """Selector-only proposal for one fixed extraction candidate."""

    model_config = ConfigDict(extra="forbid")

    candidate_index: int
    primary_ref: str = Field(
        description=(
            "Exactly one reference copied unchanged from primary_candidates; "
            "never select from required_only_candidates."
        )
    )
    required_refs: list[str] = Field(
        default_factory=list,
        description=(
            "A duplicate-free list of references copied unchanged from "
            "primary_candidates or required_only_candidates; do not repeat primary_ref."
        ),
    )


class ProjectionFragmentSelectorCorrectionResponse(StructuredResponseModel):
    """One optional correction pass; omitted candidates remain rejected."""

    model_config = ConfigDict(extra="forbid")

    corrections: list[ProjectionFragmentSelectorCorrection]


CANDIDATE_REF_PATTERN = r"^CND-[0-9]{4}$"
# Bounds the model's explanation of one admission decision.
CANDIDATE_ADMISSION_REASON_MAX_CHARS = 1000


class CandidateAdmissionDecision(StructuredResponseModel):
    """One admission judgment for the Candidate it names."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(pattern=CANDIDATE_REF_PATTERN)
    verdict: Literal["ADMITTED", "REJECTED"]
    reject_reason: Literal["evidence_incomplete", "low_value"] | None = Field(default=None, description=(
        "Required for REJECTED and null for ADMITTED: evidence_incomplete when the selected "
        "Evidence does not completely support the claim, low_value when the claim preserves "
        "no reusable knowledge."))
    duplicate_of: list[Annotated[str, Field(pattern=CANDIDATE_REF_PATTERN)]] = Field(
        default_factory=list, description=(
            "IDs from round_claims, other than this Candidate, that state the same knowledge."))
    reason: str = Field(default="", max_length=CANDIDATE_ADMISSION_REASON_MAX_CHARS)

    def row_error(self) -> str | None:
        """This decision's meaning rule; None when it holds."""

        if (self.verdict == "REJECTED") != (self.reject_reason is not None):
            return "REJECTED requires reject_reason and ADMITTED must not have one"
        if self.candidate_id in self.duplicate_of or len(set(self.duplicate_of)) != len(self.duplicate_of):
            return "duplicate_of must name other Candidates once each"
        return None


class CandidateAdmissionResponse(StructuredResponseModel):
    """Exactly one decision for every Candidate in an admission request."""

    model_config = ConfigDict(extra="forbid")

    decisions: list[CandidateAdmissionDecision]


class MemoryRelationAssessment(StructuredResponseModel):
    """A semantic relationship with its applicable direction and conflict proof."""

    model_config = ConfigDict(extra="ignore")

    classification: Literal["equivalent", "refines", "contradicts", "unrelated"]
    direction: Literal[
        "symmetric",
        "challenger_to_candidate",
        "candidate_to_challenger",
    ]
    same_subject_and_scope: bool
    incompatible_assertions: str = Field(max_length=1000)
    reason: str = Field(default="", max_length=1000)

    def row_error(self) -> str | None:
        """This relationship's meaning rule, its direction and its conflict proof; None when it holds."""

        directional = self.classification == "refines"
        if directional == (self.direction == "symmetric"):
            return "REFINES must be directional and other relations symmetric"
        incompatible = self.incompatible_assertions.strip()
        if self.classification == "contradicts":
            if not self.same_subject_and_scope:
                return "CONTRADICTS requires the same subject and scope"
            if not incompatible:
                return "CONTRADICTS requires the incompatible assertions"
        elif incompatible:
            return "only CONTRADICTS may provide incompatible assertions"
        return None


class MemoryRelationDecision(MemoryRelationAssessment):
    """Bind a relationship to one application-issued pair slot."""

    pair_index: int = Field(ge=0)


class MemoryRelationResponse(StructuredResponseModel):
    """Schema for a complete batch of exact Memory-pair decisions."""

    model_config = ConfigDict(extra="ignore")

    decisions: list[MemoryRelationDecision]


# One sentence of reasoning per pair keeps the output small and auditable.
CROSS_DOCUMENT_RELATION_REASON_MAX_CHARS = 500


class CrossDocumentRelationDecision(StructuredResponseModel):
    """One closed relation label for one application-issued pair slot."""

    model_config = ConfigDict(extra="forbid")

    pair_index: int = Field(ge=0)
    label: Literal["none", "equivalent", "updates", "contradicts"]
    reason: str = Field(max_length=CROSS_DOCUMENT_RELATION_REASON_MAX_CHARS)


class CrossDocumentRelationResponse(StructuredResponseModel):
    """Exactly one decision for every cross-document pair in the request."""

    model_config = ConfigDict(extra="forbid")

    decisions: list[CrossDocumentRelationDecision]


class RevisionAssessment(StructuredResponseModel):
    """Conditions for revising the incumbent with the current challenger."""

    same_knowledge_item: bool = Field(description=(
        "The challenger continues the same independently maintained fact, rule or decision "
        "about the same subject and concern. This is continuity through revision, not "
        "semantic equivalence: a compatible additional requirement can keep this identity."
    ))
    preserves_incumbent_truth: bool = Field(description=(
        "The challenger entails every assertion of the incumbent, including its population, "
        "time, modality and explicit sufficiency or exclusivity."
    ))
    challenger_is_complete_current_claim: bool = Field(description=(
        "The NEW challenger itself states the entire current proposition, including the "
        "preserved old requirement and added detail, without synthesizing missing text. "
        "Adding a requirement does not by itself make a current claim incomplete."
    ))


class ClaimRevisionDecision(StructuredResponseModel):
    """One fixed pair slot, including explicit unresolved and inapplicable results."""

    pair_index: int = Field(ge=0)
    status: Literal["resolved", "insufficient"]
    relation: MemoryRelationAssessment | None = None
    revision_assessment: RevisionAssessment | None = None
    reason: str = Field(default="", max_length=1000)


class ClaimContradiction(StructuredResponseModel):
    """Applicable proof only for a contradictory pair."""

    same_subject_and_scope: bool
    incompatible_assertions: str = Field(min_length=1, max_length=1000)


class ClaimRevisionWireDecision(StructuredResponseModel):
    """An explicitly discovered relationship; omission is not UNRELATED."""

    existing_id: str = Field(pattern=r"^MEM-[0-9]{4}$")
    relation: Literal[
        "equivalent", "refines_challenger_to_candidate", "refines_candidate_to_challenger",
        "contradicts",
    ]
    reason: str = Field(default="", max_length=1000)
    contradiction: ClaimContradiction | None = Field(default=None, description=(
        "Provide a proof only for relation=contradicts. For equivalent and both refinement "
        "directions this field must be null; do not fill unrelated proof fields."))
    revision_assessment: RevisionAssessment | None = Field(default=None, description=(
        "Assess the NEW challenger replacing the OLD incumbent only for "
        "relation=refines_challenger_to_candidate. For equivalent, contradicts and "
        "refines_candidate_to_challenger this field must be null, even if its conditions "
        "could all be true. Field presence is not a request to fill it."))

    def row_error(self) -> str | None:
        """This relationship's meaning rule: only its relation carries its proof; None when it holds."""

        if self.relation == "contradicts":
            if (self.contradiction is None or not self.contradiction.same_subject_and_scope
                    or not self.contradiction.incompatible_assertions.strip()):
                return "CONTRADICTS requires overlapping scope and incompatible assertions"
        elif self.contradiction is not None:
            return "only CONTRADICTS may provide a contradiction proof"
        if self.revision_assessment is not None and self.relation != "refines_challenger_to_candidate":
            return "a revision proof applies only to challenger-to-candidate refinement"
        return None

    def decision(self) -> ClaimRevisionDecision:
        refinement = self.relation.startswith("refines_")
        return ClaimRevisionDecision(
            pair_index=0, status="resolved", reason=self.reason,
            revision_assessment=self.revision_assessment,
            relation=MemoryRelationAssessment(
                classification="refines" if refinement else self.relation,
                direction=self.relation.removeprefix("refines_") if refinement else "symmetric",
                same_subject_and_scope=self.contradiction is not None,
                incompatible_assertions=self.contradiction.incompatible_assertions if self.contradiction else "",
                reason=self.reason,
            ),
        )


class ClaimCandidateResult(StructuredResponseModel):
    """Every requested candidate has one result, even when no edges were found."""

    candidate_id: str = Field(pattern=r"^NEW-[0-9]{4}$")
    relations: list[ClaimRevisionWireDecision]
    uncertain_existing_ids: list[Annotated[str, Field(pattern=r"^MEM-[0-9]{4}$")]]


class ClaimRevisionWireResponse(StructuredResponseModel):
    results: list[ClaimCandidateResult]


class EntityBatchValidationDecision(StructuredResponseModel):
    """One semantic judgment for the mention it names."""

    model_config = ConfigDict(extra="forbid")

    mention: str
    matched_id: int | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = Field(default="", max_length=1000)


class EntityBatchValidationResponse(StructuredResponseModel):
    """One decision per mention in an entity ambiguity adjudication request."""

    model_config = ConfigDict(extra="forbid")

    decisions: list[EntityBatchValidationDecision]


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
    # Gateway aliases may not exist in the model registry. These explicit,
    # limits let deployments lower known capacity or configure unknown aliases.
    max_input_tokens: int | None = None
    context_window_tokens: int | None = None
    max_output_tokens: int | None = None
    input_budget_fraction: float = 0.8


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
    """Collect logical-call outcomes for one request-local lifecycle scope.

    A collector with a parent also reports every call to it, so one concurrent
    line counts its own calls while its enclosing scope still counts them all.
    """

    def __init__(self, parent: StructuredLlmMetricsCollector | None = None) -> None:
        self._lock = Lock()
        self._calls: list[StructuredLlmCallTelemetry] = []
        self._parent = parent

    def record(self, telemetry: StructuredLlmCallTelemetry) -> None:
        with self._lock:
            self._calls.append(telemetry)
        if self._parent is not None:
            self._parent.record(telemetry)

    def checkpoint(self) -> int:
        """Mark the start of a stage without replacing its Unit collector."""
        with self._lock:
            return len(self._calls)

    def summary(self, *, source_unit_elapsed_ms: int, since: int = 0) -> StructuredLlmMetricsSummary:
        with self._lock:
            calls = tuple(self._calls[since:])

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


_scoped_metrics_collector: ContextVar[StructuredLlmMetricsCollector | None] = ContextVar(
    "memforge_structured_llm_metrics_collector",
    default=None,
)


@contextmanager
def structured_llm_metrics_scope(
    collector: StructuredLlmMetricsCollector | None = None,
) -> Iterator[StructuredLlmMetricsCollector]:
    """Reuse the execution collector for nested stages, or start an explicit Unit."""
    selected = collector if collector is not None else _scoped_metrics_collector.get()
    if selected is None:
        selected = StructuredLlmMetricsCollector()
    token = _scoped_metrics_collector.set(selected)
    try:
        yield selected
    finally:
        _scoped_metrics_collector.reset(token)


@contextmanager
def structured_llm_line_scope() -> Iterator[StructuredLlmMetricsCollector]:
    """Count the calls of one line that runs concurrently with others, apart from theirs."""
    with structured_llm_metrics_scope(StructuredLlmMetricsCollector(parent=_scoped_metrics_collector.get())) as line:
        yield line


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


class SupportWitnessDelta(BaseModel):
    """Current refs this request read that support or oppose the claim."""

    model_config = ConfigDict(extra="forbid")
    support_witness_refs: list[str]
    opposing_witness_refs: list[str]


class ContinueReadingWireResult(BaseModel):
    """The claim is not yet completely supported; keep reading."""

    model_config = ConfigDict(extra="forbid")
    work_id: str
    status: Literal["continue"]
    witness_delta: SupportWitnessDelta


class SupportedWireResult(BaseModel):
    """One complete current Evidence Unit supports the claim, without generated prose."""

    model_config = ConfigDict(extra="forbid")
    work_id: str
    status: Literal["supported"]
    primary_ref: str
    required_refs: list[str]


class UnsupportedWireResult(BaseModel):
    """The complete revision was read and no complete current Evidence Unit supports the claim."""

    model_config = ConfigDict(extra="forbid")
    work_id: str
    status: Literal["unsupported"]


class SupportAssessmentWireResponse(BaseModel):
    results: list[ContinueReadingWireResult | SupportedWireResult | UnsupportedWireResult]


class ChangeImpactWireResult(BaseModel):
    """Whether one revision's changes can affect one fixed claim, without generated prose."""

    model_config = ConfigDict(extra="forbid")
    work_id: str
    impact: Literal["affected", "unaffected"]


class ChangeImpactWireResponse(BaseModel):
    results: list[ChangeImpactWireResult]


class StructuredLlmError(RuntimeError):
    """Raised when a required structured LLM call cannot produce valid schema output."""

    def __init__(
        self,
        message: str,
        *,
        terminal_category: StructuredLlmTerminalCategory = "invalid_response",
        error_code: str = "structured_llm_error",
        validation_fields: tuple[tuple[str, str], ...] = (),
        diagnostic: Any = None,
    ) -> None:
        super().__init__(message)
        self.terminal_category = terminal_category
        self.error_code = error_code
        self.validation_fields = validation_fields
        self.diagnostic = diagnostic


class _InvalidModelOutput(Exception):
    """A received model response failed schema validation; its cause is the validation error."""


@dataclass(frozen=True, slots=True)
class _StructuredLlmFailure:
    """Content-free failure value that can outlive provider call frames."""

    terminal_category: StructuredLlmTerminalCategory
    error_code: str
    validation_fields: tuple[tuple[str, str], ...] = ()
    # Whole-object rules have no field path to point at, so the schema repair
    # prompt names them by the validator's own message.
    model_level_messages: tuple[str, ...] = ()

    def to_error(self, *, timeout_s: float | None = None) -> StructuredLlmError:
        if self.terminal_category == "deadline_exceeded":
            message = (
                f"structured LLM logical deadline exceeded after {timeout_s:g}s"
                if timeout_s is not None
                else "structured LLM logical deadline exceeded"
            )
        elif self.terminal_category == "provider_error":
            message = "structured LLM provider request failed"
        elif self.terminal_category == "request_error":
            message = "structured LLM request failed without a response to validate"
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
        PAYLOAD_TOO_LARGE,
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
    """Return a bounded content-free provider failure code.

    Request-size rejections get one shared code whatever transport shape the
    provider used, so callers can split the work without reading provider text.
    """

    if isinstance(exc, litellm.ContextWindowExceededError):
        return INPUT_CAPACITY_EXCEEDED
    if _provider_status_code(exc) == _HTTP_PAYLOAD_TOO_LARGE:
        return PAYLOAD_TOO_LARGE
    outer_code = type(exc).__name__
    if outer_code != "APIConnectionError":
        return outer_code
    detail = _connection_error_detail(exc)
    if detail is None:
        return outer_code
    return PAYLOAD_TOO_LARGE if detail == PAYLOAD_TOO_LARGE else f"{outer_code}.{detail}"


def _connection_error_detail(exc: BaseException) -> str | None:
    current: BaseException | None = exc
    seen: set[int] = set()
    for _ in range(4):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        type_name = type(current).__name__
        for detail, pattern in _SAFE_PROVIDER_ERROR_PATTERNS:
            if pattern.search(type_name):
                return detail
        current = current.__cause__ or current.__context__

    # LiteLLM 1.86 flattens many transport exceptions into the outer message
    # without preserving a cause. Inspect only a bounded prefix and persist
    # only the matched category, never the provider text itself.
    message_prefix = str(exc)[:2048]
    for detail, pattern in _SAFE_PROVIDER_ERROR_PATTERNS:
        if pattern.search(message_prefix):
            return detail
    return None


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
    if isinstance(exc, _InvalidModelOutput) and exc.__cause__ is not None:
        exc, terminal_category = exc.__cause__, "invalid_response"
    category = terminal_category
    if category is None:
        category = "provider_error" if _is_non_fallback_provider_error(exc) else "request_error"
    return _StructuredLlmFailure(
        terminal_category=category,
        error_code=_safe_provider_error_code(exc),
        validation_fields=_safe_validation_fields(exc),
        model_level_messages=_model_level_validation_messages(exc),
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
        from memforge.diagnostics import diagnostic_path

        location = diagnostic_path(location_parts)
        rule_type = str(error.get("type") or "").strip()
        if rule_type:
            fields.append((location, rule_type))
    return tuple(fields)


def _model_level_validation_messages(exc: BaseException) -> tuple[str, ...]:
    """Return validator messages for errors about the whole object, not one field."""

    if not isinstance(exc, ValidationError):
        return ()
    return tuple(
        str(error.get("msg") or "")
        for error in exc.errors(include_url=False, include_context=False, include_input=False)
        if not error.get("loc") and error.get("msg")
    )


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
    if response_format is ClaimRevisionWireResponse:
        return "claim_revision"
    if response_format is SupportAssessmentWireResponse:
        return "support_assessment"
    if response_format is ChangeImpactWireResponse:
        return "change_impact"
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
    if _provider_status_code(exc) in {401, 403, 404}:
        return True
    # The JSON-text transport is longer, so it cannot fit where this request did not.
    return _safe_provider_error_code(exc) in {INPUT_CAPACITY_EXCEEDED, PAYLOAD_TOO_LARGE}


def litellm_model_name(model: str) -> str:
    """Map existing model names into LiteLLM provider/model notation."""
    if "/" in model:
        return model
    return f"anthropic/{model}"


def _json_text_prompt(
    prompt: str,
    response_format: type[BaseModel],
    *,
    validation_failure: _StructuredLlmFailure | None = None,
    validation_source: Literal["native_schema", "json_text"] | None = None,
) -> str:
    """Append the schema as a text instruction for the no-tool JSON path."""
    schema = json.dumps(response_format.model_json_schema(), ensure_ascii=False)
    request_prompt = (
        f"{prompt}\n\nReturn ONLY a single JSON object that matches this JSON Schema, "
        f"with no markdown fences and no commentary:\n{schema}"
    )
    if validation_failure is None:
        return request_prompt

    validation_fields = [
        {
            "location": location[:_SCHEMA_REPAIR_LOCATION_CHAR_CAP],
            "rule": rule[:_SCHEMA_REPAIR_RULE_CHAR_CAP],
        }
        for location, rule in validation_failure.validation_fields[
            :_SCHEMA_REPAIR_MAX_VALIDATION_FIELDS
        ]
    ]
    diagnostic_fields: dict[str, object] = {
        "error_code": validation_failure.error_code[:_SCHEMA_REPAIR_RULE_CHAR_CAP],
        "validation_errors": validation_fields,
    }
    if validation_failure.model_level_messages:
        diagnostic_fields["model_level_errors"] = [
            message[:_SCHEMA_REPAIR_MESSAGE_CHAR_CAP]
            for message in validation_failure.model_level_messages[
                :_SCHEMA_REPAIR_MAX_VALIDATION_FIELDS
            ]
        ]
    diagnostic = json.dumps(diagnostic_fields, ensure_ascii=False, separators=(",", ":"))
    previous_attempt = (
        "The preceding native-schema response"
        if validation_source == "native_schema"
        else "The preceding JSON-text response"
    )
    return (
        f"{request_prompt}\n\n{previous_attempt} failed local schema validation. "
        "Return the complete JSON object again after correcting these diagnostics. "
        "The prior response body and invalid values are intentionally not repeated:\n"
        f"{diagnostic}"
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


_TRUNCATED_FINISH_REASONS = frozenset({"length", "max_tokens"})
_REFUSAL_FINISH_REASONS = frozenset({"content_filter", "refusal"})
# Judgments whose empty-but-valid output would read as "nothing found"; a
# refused reply must fail instead.
_REFUSAL_ERROR_CODES: dict[type[BaseModel], str] = {
    ClaimRevisionWireResponse: "claim_response_incomplete",
    CrossDocumentRelationResponse: "cross_document_relation_response_incomplete",
    SupportAssessmentWireResponse: "support_response_incomplete",
    ChangeImpactWireResponse: "change_impact_response_incomplete",
}


def _raise_for_unfinished_response(response: object, response_format: type[BaseModel]) -> None:
    """Reject replies the provider did not finish, even when they parse.

    Truncated output is a request-size failure: resending the same request at
    the same output limit cannot complete it, so no JSON-text fallback follows.
    """

    finish = _response_finish_reason(response)
    stop = _response_stop_reason(response)
    if finish in _TRUNCATED_FINISH_REASONS or stop == "max_tokens":
        raise StructuredLlmError("structured response reached its output limit", error_code=OUTPUT_TRUNCATED)
    refusal_code = _REFUSAL_ERROR_CODES.get(response_format)
    message = _object_value(_first_response_choice(response), "message")
    if refusal_code is not None and (
        finish in _REFUSAL_FINISH_REASONS or stop == "refusal" or _object_value(message, "refusal")
    ):
        raise StructuredLlmError("assessment response did not complete", error_code=refusal_code)


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
        failure_trace_sink: FailureTraceSink | None = None,
    ) -> None:
        self.config = config
        self._telemetry_sink = telemetry_sink
        self._failure_trace_sink = failure_trace_sink if failure_trace_sink is not None else local_failure_trace_sink_from_env()
        self._request_budgets = {}

    @property
    def max_concurrent(self) -> int:
        """Maximum phase fan-out; final admission is shared across all clients."""

        return max(1, int(self.config.max_concurrent))

    def request_budget(self, model: str | None = None):
        from memforge.llm.request_budget import RequestBudget

        name = litellm_model_name(model or self.config.model)
        if name not in self._request_budgets:
            self._request_budgets[name] = RequestBudget.resolve(name, self.config)
        return self._request_budgets[name]

    @property
    def input_policy_identity(self) -> str:
        return self.input_policy_identity_for()

    def input_policy_identity_for(self, model: str | None = None) -> str:
        return self.request_budget(model).identity

    def request_tokens(
        self, prompt: str, *, response_format: type[BaseModel],
        model: str | None = None, images: tuple[StructuredLlmImage, ...] = (),
    ) -> int:
        """Count the schema fallback transport, including its real instructions."""
        name = litellm_model_name(model or self.config.model)
        material = _json_text_prompt(prompt, response_format)
        messages = [{"role": "user", "content": _structured_user_content(material, images)}]
        from memforge.llm.request_budget import metadata_model

        return litellm.token_counter(model=metadata_model(name), messages=messages)

    def request_fits(
        self, prompt: str, *, response_format: type[BaseModel],
        max_tokens: int, model: str | None = None,
        images: tuple[StructuredLlmImage, ...] = (), reserve_correction: bool = True,
    ) -> bool:
        budget = self.request_budget(model)
        if budget.available_input(max_tokens, reserve_correction=reserve_correction) < 0:
            return False
        return budget.fits(self.request_tokens(
            prompt, response_format=response_format, model=model, images=images,
        ), max_tokens, reserve_correction=reserve_correction)

    @contextmanager
    def metrics_scope(
        self,
        collector: StructuredLlmMetricsCollector,
    ) -> Iterator[StructuredLlmMetricsCollector]:
        """Route calls in the current async context to one request-local collector."""

        with structured_llm_metrics_scope(collector) as selected:
            yield selected


    async def evaluate_revision_work(self, prompt, *, response_format, max_tokens, model=None, images=()):
        return await self._call_schema(prompt=prompt, response_format=response_format,
                                       max_tokens=max_tokens, model=model, images=images)


    async def assess_claim_revisions(
        self, prompt: str, *, max_tokens: int = 32_768,
        model: str | None = None, images: tuple[StructuredLlmImage, ...] = (),
    ) -> ClaimRevisionWireResponse:
        return await self._call_schema(
            prompt=prompt, response_format=ClaimRevisionWireResponse,
            max_tokens=max_tokens, model=model, images=images,
        )

    async def extract_projection_fragment_memories(
        self,
        prompt: str,
        *,
        max_tokens: int,
        model: str | None = None,
        images: tuple[StructuredLlmImage, ...] = (),
    ) -> ProjectionFragmentMemoryExtractionResponse:
        return await self._call_schema(
            prompt=prompt,
            response_format=ProjectionFragmentMemoryExtractionResponse,
            max_tokens=max_tokens,
            model=model,
            images=images,
        )

    async def correct_projection_fragment_selectors(
        self, prompt: str, *, max_tokens: int, model: str | None = None,
        images: tuple[StructuredLlmImage, ...] = (),
    ) -> ProjectionFragmentSelectorCorrectionResponse:
        return await self._call_schema(
            prompt=prompt,
            response_format=ProjectionFragmentSelectorCorrectionResponse,
            max_tokens=max_tokens,
            model=model,
            images=images,
        )

    async def admit_candidates(
        self, prompt: str, *, max_tokens: int, model: str | None = None,
        images: tuple[StructuredLlmImage, ...] = (),
    ) -> CandidateAdmissionResponse:
        return await self._call_schema(
            prompt=prompt, response_format=CandidateAdmissionResponse,
            max_tokens=max_tokens, model=model, images=images,
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

    async def classify_cross_document_relations(
        self, prompt: str, *, max_tokens: int, model: str | None = None,
    ) -> CrossDocumentRelationResponse:
        return await self._call_schema(
            prompt=prompt, response_format=CrossDocumentRelationResponse,
            max_tokens=max_tokens, model=model,
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
        from memforge.agent_knowledge import AgentKnowledgePatchModelResponse

        return await self._call_schema(
            prompt=prompt,
            response_format=AgentKnowledgePatchModelResponse,
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
        async with capture_call(self._failure_trace_sink, prompt=prompt,
                schema=response_format, operation=_schema_operation_name(response_format)) as capture:
            async with admission.semaphore:
                result = await self._call_schema_admitted(
                    prompt=prompt, response_format=response_format, max_tokens=max_tokens,
                    model=model, retry_with_json_text=retry_with_json_text, images=images,
                )
            if capture is not None:
                object.__setattr__(result, "_llm_failure_capture", capture)
            return result

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
        except asyncio.CancelledError:
            self._emit_telemetry(
                state.telemetry(
                    elapsed_ms=max(0, round((perf_counter() - started) * 1000)),
                    terminal_category="cancelled",
                    error_code="call_cancelled",
                )
            )
            raise
        except TimeoutError:
            failure = _StructuredLlmFailure(
                terminal_category="deadline_exceeded",
                error_code="logical_deadline_exceeded",
            )
        except StructuredLlmError as exc:
            failure = _structured_failure(exc)
        except StructuredLlmImageError as exc:
            failure = _StructuredLlmFailure(
                terminal_category="request_error",
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
            from dataclasses import asdict
            from memforge.evals.agent_evaluation import QualitySignal

            error = failure.to_error(timeout_s=self.config.timeout_s)
            attempt = next((item for item in reversed(state.diagnostic_attempts)
                            if item.error_code == failure.error_code), None)
            details = asdict(attempt) if attempt is not None else {
                "terminal_category": failure.terminal_category, "error_code": failure.error_code,
                "structured_mode": state.final_mode, "requested_max_tokens": max_tokens,
            }
            try:
                error.diagnostic = QualitySignal(
                    event_name="structured_llm_attempt_outcome", outcome="failed",
                    reason_code=failure.terminal_category, operation=state.operation,
                    provider=_safe_llm_provider(model or self.config.model), model=model or self.config.model,
                    **details,
                )
            except Exception:
                logger.warning("Structured LLM failure diagnostic construction failed")
            raise error

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
            return await self._attempt_json_text_with_repair(
                prompt=prompt,
                response_format=response_format,
                model_name=model_name,
                max_tokens=max_tokens,
                native_schema_transport=native_schema_transport,
                deadline=deadline,
                state=state,
                images=images,
            )

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
        if (
            not retry_with_json_text
            or schema_failure.terminal_category == "provider_error"
            or schema_failure.error_code == OUTPUT_TRUNCATED
        ):
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
        return await self._attempt_json_text_with_repair(
            prompt=prompt,
            response_format=response_format,
            model_name=model_name,
            max_tokens=max_tokens,
            native_schema_transport=native_schema_transport,
            deadline=deadline,
            state=state,
            images=images,
            # A request the gateway rejected has no response whose validation could be repaired.
            initial_validation_failure=schema_failure if schema_failure.terminal_category == "invalid_response" else None,
        )

    async def _attempt_json_text_with_repair(
        self,
        *,
        prompt: str,
        response_format: type[BaseModel],
        model_name: str,
        max_tokens: int,
        native_schema_transport: NativeSchemaTransport,
        deadline: float,
        state: _StructuredCallState,
        images: tuple[StructuredLlmImage, ...],
        initial_validation_failure: _StructuredLlmFailure | None = None,
    ):
        """Attempt JSON text and repair one invalid response under the shared budget."""

        failure: _StructuredLlmFailure | None = None
        repairable = False
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
                validation_failure=initial_validation_failure,
                validation_source=(
                    "native_schema"
                    if initial_validation_failure is not None
                    else None
                ),
            )
        except _InvalidModelOutput as exc:
            # Malformed or schema-invalid output gets its one repair; truncation and refusal do not.
            failure, repairable = _structured_failure(exc), True
        except Exception as exc:
            failure = _structured_failure(exc)
        if failure is not None and repairable and state.retry_budget > 0:
            state.retry_budget -= 1
            state.retry_count += 1
            logger.warning(
                "Structured LLM JSON-text response was invalid for model %s and schema %s; "
                "retrying once within the logical deadline (error_code=%s)",
                model_name,
                response_format.__name__,
                failure.error_code,
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
                    validation_failure=failure,
                    validation_source="json_text",
                )
            except Exception as exc:
                failure = _structured_failure(exc)
            else:
                failure = None
        if failure is not None:
            raise failure.to_error()
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
        validation_failure: _StructuredLlmFailure | None = None,
        validation_source: Literal["native_schema", "json_text"] | None = None,
    ):
        request_prompt = (
            prompt
            if native_schema
            else _json_text_prompt(
                prompt,
                response_format,
                validation_failure=validation_failure,
                validation_source=validation_source,
            )
        )
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
            _raise_for_unfinished_response(response, response_format)
            raw_content = _message_content(response)
            if isinstance(raw_content, response_format):
                return raw_content
            if isinstance(raw_content, dict):
                return response_format.model_validate(raw_content)
            return _validate_structured_json_text(str(raw_content), response_format)
        except (StructuredLlmError, ValueError) as exc:
            # Only a received response that fails validation is an invalid response.
            capture = current_capture()
            if capture is not None:
                capture.failed(exc, stage="schema_validation")
            state.record_invalid_response_attempt(
                response,
                attempt_index=attempt_index,
                structured_mode=structured_mode,
                schema_transport=schema_transport,
                requested_max_tokens=max_tokens,
                exc=exc,
            )
            if isinstance(exc, StructuredLlmError):
                raise
            raise _InvalidModelOutput(type(exc).__name__) from exc

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
            capture = current_capture()
            if capture is not None:
                capture.begin_attempt(dict(model=model_name, messages=messages,
                    max_tokens=max_tokens, timeout=remaining_s, num_retries=0,
                    **provider_kwargs, **schema_kwargs))
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
            except BaseException as exc:
                if capture is not None:
                    capture.failed(exc, stage="provider")
                if not isinstance(exc, Exception):
                    raise
                state.record_failed_attempt()
                retry = _is_retryable_provider_error(exc) and state.retry_budget > 0
                failure = _structured_failure(exc)
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
            if capture is not None:
                capture.response(response)
            state.record_response(response)
            return response, attempt_index

    def _emit_telemetry(self, telemetry: StructuredLlmCallTelemetry) -> None:
        try:
            self._emit_call_diagnostics(telemetry)
        except Exception:
            # Observability must not replace a successful response, cancellation,
            # or the original provider error. Never log raw diagnostic exceptions.
            logger.warning("Structured LLM diagnostic reporting failed")

    def _emit_call_diagnostics(self, telemetry: StructuredLlmCallTelemetry) -> None:
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
        if telemetry.terminal_category == "cancelled":
            outcome = "expected"
            reason_code = "call_cancelled"
        elif telemetry.terminal_category == "success":
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
        scoped_collector = _scoped_metrics_collector.get()
        if scoped_collector is not None:
            scoped_collector.record(telemetry)
        if self._telemetry_sink is not None:
            try:
                self._telemetry_sink(telemetry)
            except Exception:
                logger.warning("Structured LLM telemetry sink failed", exc_info=True)
