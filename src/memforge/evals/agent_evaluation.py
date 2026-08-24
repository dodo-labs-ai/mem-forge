"""Privacy-safe online quality events for agent runtime.

Producers emit small, content-free ``QualitySignal`` values.  The Source
Derivation seam binds them to stable source lineage before persistence, so a
signal remains actionable even when no Memory was created.
"""

from __future__ import annotations

import atexit
import hashlib
import importlib
import json
import logging
import os
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from functools import lru_cache
from threading import Lock
from typing import Callable, ContextManager, Iterator, Literal, Mapping, Protocol, Sequence


AGENT_RUNTIME_EVENT_SCHEMA_VERSION = "agent-runtime-event-v3"
AGENT_ASSESSMENT_SCHEMA_VERSION = "agent-assessment-v5"
SOURCE_UNIT_LIFECYCLE_CONTRACT_VERSION = "source-unit-lifecycle-v1"
DETERMINISTIC_RUNTIME_EVALUATOR_NAME = "memforge.deterministic.runtime_contract"
DETERMINISTIC_RUNTIME_EVALUATOR_VERSION = "1"
ONLINE_EVALUATION_COVERAGE_POLICY = "semantic_evaluator_v1"
AgentRuntimeOutcome = Literal["expected", "degraded", "rejected", "failed"]
AgentAssessmentStatus = Literal["completed", "failed"]
AgentAssessmentLabel = Literal["pass", "fail", "needs_review"]
AgentAssessmentAnnotatorKind = Literal["code", "llm", "human"]
AgentAssessmentConfidence = Literal["low", "medium", "high"]
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RuntimeTraceContext:
    """Optional OTel correlation; never the identity of a durable fact."""

    trace_id: str
    span_id: str
    trace_flags: int | None = None


@dataclass(frozen=True, slots=True)
class QualitySignal:
    """One content-free occurrence before durable lineage is attached."""

    event_name: str
    outcome: AgentRuntimeOutcome
    reason_code: str
    operation: str | None = None
    provider: str | None = None
    model: str | None = None
    prompt_hash: str | None = None
    candidate_hash: str | None = None
    observation_id: str | None = None
    observation_revision_id: str | None = None
    range_start: int | None = None
    range_end: int | None = None
    block_hash: str | None = None
    quote_hash: str | None = None
    quote_chars: int | None = None
    localization_mode: str | None = None
    attempt_count: int | None = None
    retry_count: int | None = None
    fallback_count: int | None = None
    attempt_index: int | None = None
    structured_mode: str | None = None
    schema_transport: str | None = None
    requested_max_tokens: int | None = None
    terminal_category: str | None = None
    error_code: str | None = None
    finish_reason: str | None = None
    stop_reason: str | None = None
    provider_request_id: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    response_chars: int | None = None
    response_hash: str | None = None
    validation_location: str | None = None
    validation_rule: str | None = None
    json_error_line: int | None = None
    json_error_column: int | None = None
    candidate_count: int | None = None
    rejected_count: int | None = None
    occurrence_count: int = 1

    def __post_init__(self) -> None:
        for name in (
            "event_name",
            "reason_code",
            "operation",
            "provider",
            "localization_mode",
            "structured_mode",
            "schema_transport",
            "terminal_category",
            "error_code",
            "finish_reason",
            "stop_reason",
            "validation_rule",
        ):
            _require_safe_label(name, getattr(self, name))
        _require_model(self.model)
        _require_safe_identifier("provider_request_id", self.provider_request_id)
        _require_safe_diagnostic_path("validation_location", self.validation_location)
        for name in (
            "prompt_hash",
            "candidate_hash",
            "block_hash",
            "quote_hash",
            "response_hash",
        ):
            _require_hash(name, getattr(self, name))
        if (self.range_start is None) != (self.range_end is None):
            raise ValueError("evaluation evidence range must be complete or absent")
        if self.range_start is not None and (
            self.range_start < 0 or self.range_end is None or self.range_end <= self.range_start
        ):
            raise ValueError("evaluation evidence range must be positive and ordered")
        for name in (
            "quote_chars",
            "attempt_count",
            "retry_count",
            "fallback_count",
            "attempt_index",
            "requested_max_tokens",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "response_chars",
            "json_error_line",
            "json_error_column",
            "candidate_count",
            "rejected_count",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.occurrence_count < 1:
            raise ValueError("occurrence_count must be positive")


@dataclass(frozen=True, slots=True)
class AgentRuntimeEvent:
    """One append-only, lineage-bound execution fact."""

    event_id: str
    event_name: str
    outcome: AgentRuntimeOutcome
    reason_code: str
    occurred_at: datetime
    source_id: str
    source_type: str
    doc_id: str
    source_unit_id: str
    target_unit_revision_id: str
    projection_run_id: str
    schema_version: str = AGENT_RUNTIME_EVENT_SCHEMA_VERSION
    operation_id: str | None = None
    execution_id: str | None = None
    contract_version: str | None = None
    payload_hash: str | None = None
    operation_input_hash: str | None = None
    execution_owner_id: str | None = None
    base_unit_revision_id: str | None = None
    duration_ms: int | None = None
    recovered: bool | None = None
    incumbent_count: int | None = None
    relation_pair_count: int | None = None
    mutation_count: int | None = None
    review_count: int | None = None
    model_call_count: int | None = None
    derivation_id: str | None = None
    batch_id: str | None = None
    batch_attempt: int | None = None
    extraction_contract_version: str | None = None
    deployment_revision: str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    trace_flags: int | None = None
    operation: str | None = None
    provider: str | None = None
    model: str | None = None
    prompt_hash: str | None = None
    candidate_hash: str | None = None
    observation_id: str | None = None
    observation_revision_id: str | None = None
    range_start: int | None = None
    range_end: int | None = None
    block_hash: str | None = None
    quote_hash: str | None = None
    quote_chars: int | None = None
    localization_mode: str | None = None
    attempt_count: int | None = None
    retry_count: int | None = None
    fallback_count: int | None = None
    attempt_index: int | None = None
    structured_mode: str | None = None
    schema_transport: str | None = None
    requested_max_tokens: int | None = None
    terminal_category: str | None = None
    error_code: str | None = None
    finish_reason: str | None = None
    stop_reason: str | None = None
    provider_request_id: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    response_chars: int | None = None
    response_hash: str | None = None
    validation_location: str | None = None
    validation_rule: str | None = None
    json_error_line: int | None = None
    json_error_column: int | None = None
    candidate_count: int | None = None
    rejected_count: int | None = None
    occurrence_count: int = 1


@dataclass(frozen=True, slots=True)
class AgentRuntimeEventQuery:
    """Bounded cohort filters shared by every storage adapter."""

    occurred_from: datetime
    occurred_to: datetime
    requesting_user_id: str | None = None
    include_private: bool = False
    event_id: str | None = None
    operation_id: str | None = None
    execution_id: str | None = None
    source_id: str | None = None
    source_type: str | None = None
    event_name: str | None = None
    outcome: AgentRuntimeOutcome | None = None
    reason_code: str | None = None
    trace_id: str | None = None
    model: str | None = None
    provider: str | None = None
    deployment_revision: str | None = None
    extraction_contract_version: str | None = None
    contract_version: str | None = None
    newest_first: bool = False
    limit: int = 100
    offset: int = 0

    def __post_init__(self) -> None:
        if self.occurred_to <= self.occurred_from:
            raise ValueError("evaluation cohort requires a non-empty half-open time range")
        if not 1 <= self.limit <= 1000:
            raise ValueError("evaluation cohort limit must be between 1 and 1000")
        if self.offset < 0:
            raise ValueError("evaluation cohort offset must be non-negative")


@dataclass(frozen=True, slots=True)
class AgentAssessment:
    """One versioned judgment over one online event, offline result, or calibration candidate."""

    assessment_id: str
    target_event_id: str | None
    criterion: str
    status: AgentAssessmentStatus
    label: AgentAssessmentLabel | None
    reason_code: str
    annotator_kind: AgentAssessmentAnnotatorKind
    evaluator_name: str
    evaluator_version: str
    created_at: datetime
    target_result_id: str | None = None
    target_candidate_id: str | None = None
    annotator_id: str | None = None
    content_policy_id: str | None = None
    input_fingerprint: str | None = None
    confidence: AgentAssessmentConfidence | None = None
    reused_from_assessment_id: str | None = None
    occurrence_count: int = 1
    schema_version: str = AGENT_ASSESSMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "assessment_id",
            "criterion",
            "reason_code",
            "evaluator_name",
            "evaluator_version",
        ):
            _require_safe_identifier(name, getattr(self, name))
        targets = (
            self.target_event_id,
            self.target_result_id,
            self.target_candidate_id,
        )
        if sum(value is not None for value in targets) != 1:
            raise ValueError(
                "assessment requires exactly one event, result, or candidate target"
            )
        if self.target_event_id is not None:
            _require_safe_identifier("target_event_id", self.target_event_id)
        if self.target_result_id is not None:
            _require_safe_identifier("target_result_id", self.target_result_id)
        if self.target_candidate_id is not None:
            _require_safe_identifier("target_candidate_id", self.target_candidate_id)
        if self.annotator_id is not None:
            _require_bounded_principal_id("annotator_id", self.annotator_id)
        if self.content_policy_id is not None:
            _require_safe_identifier("content_policy_id", self.content_policy_id)
        _require_hash("input_fingerprint", self.input_fingerprint)
        _require_safe_identifier(
            "reused_from_assessment_id", self.reused_from_assessment_id
        )
        if (
            self.annotator_kind == "human"
            and self.schema_version == AGENT_ASSESSMENT_SCHEMA_VERSION
            and (self.annotator_id is None or self.content_policy_id is None)
        ):
            raise ValueError(
                "human assessment requires annotator and content-policy provenance"
            )
        if self.annotator_kind != "human" and self.annotator_id is not None:
            raise ValueError("only human assessments carry reviewer provenance")
        if self.annotator_kind == "code" and self.content_policy_id is not None:
            raise ValueError("code assessments cannot carry human-review provenance")
        if (
            self.annotator_kind == "llm"
            and self.target_result_id is not None
            and self.schema_version == AGENT_ASSESSMENT_SCHEMA_VERSION
            and (self.content_policy_id is None or self.input_fingerprint is None)
        ):
            raise ValueError(
                "offline LLM assessment requires content-policy and input provenance"
            )
        if self.confidence is not None and self.annotator_kind != "llm":
            raise ValueError("only LLM assessments carry model confidence")
        if self.input_fingerprint is not None and self.annotator_kind != "llm":
            raise ValueError("only LLM assessments carry semantic input fingerprints")
        if self.confidence not in {None, "low", "medium", "high"}:
            raise ValueError("assessment confidence must be low, medium, or high")
        if self.status == "failed" and self.confidence is not None:
            raise ValueError("failed assessment cannot carry confidence")
        if (
            self.reused_from_assessment_id is not None
            and self.annotator_kind != "llm"
        ):
            raise ValueError("only LLM assessments can reuse an assessment decision")
        if self.reused_from_assessment_id == self.assessment_id:
            raise ValueError("assessment cannot reuse itself")
        if self.status == "completed" and self.label is None:
            raise ValueError("completed assessment requires a label")
        if self.status == "failed" and self.label is not None:
            raise ValueError("failed assessment cannot carry a label")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("assessment timestamp requires a timezone")
        if self.occurrence_count < 1:
            raise ValueError("assessment occurrence_count must be positive")


@dataclass(frozen=True, slots=True)
class AgentRuntimeBundle:
    """One terminal fact and its deterministic assessment as an atomic unit."""

    events: tuple[AgentRuntimeEvent, ...]
    assessments: tuple[AgentAssessment, ...]

    def __post_init__(self) -> None:
        event_ids = {event.event_id for event in self.events}
        if any(
            item.target_result_id is not None or item.target_event_id not in event_ids
            for item in self.assessments
        ):
            raise ValueError("agent assessment target must belong to the runtime bundle")

    @property
    def event(self) -> AgentRuntimeEvent:
        if len(self.events) != 1:
            raise ValueError("runtime bundle does not contain exactly one event")
        return self.events[0]

    @property
    def assessment(self) -> AgentAssessment:
        if len(self.assessments) != 1:
            raise ValueError("runtime bundle does not contain exactly one assessment")
        return self.assessments[0]


@dataclass(frozen=True, slots=True)
class AgentAssessmentQuery:
    """Bounded assessment filters with runtime-event visibility semantics."""

    occurred_from: datetime
    occurred_to: datetime
    requesting_user_id: str | None = None
    include_private: bool = False
    assessment_id: str | None = None
    target_event_id: str | None = None
    target_result_id: str | None = None
    source_id: str | None = None
    criterion: str | None = None
    status: AgentAssessmentStatus | None = None
    label: AgentAssessmentLabel | None = None
    evaluator_name: str | None = None
    newest_first: bool = False
    limit: int = 100
    offset: int = 0

    def __post_init__(self) -> None:
        if self.occurred_to <= self.occurred_from:
            raise ValueError("assessment cohort requires a non-empty half-open time range")
        if not 1 <= self.limit <= 1000:
            raise ValueError("assessment cohort limit must be between 1 and 1000")
        if self.offset < 0:
            raise ValueError("assessment cohort offset must be non-negative")


@dataclass(frozen=True, slots=True)
class AgentRuntimeCohortReport:
    """Content-free aggregate with explicit per-event-name denominators."""

    total_events: int
    event_name_counts: Mapping[str, int]
    outcome_counts: Mapping[str, int]
    reason_counts: Mapping[str, int]
    rates_by_event_name: Mapping[str, Mapping[str, float]]

    def to_payload(self) -> dict[str, object]:
        return {
            "total_events": self.total_events,
            "event_name_counts": dict(self.event_name_counts),
            "outcome_counts": dict(self.outcome_counts),
            "reason_counts": dict(self.reason_counts),
            "rates_by_event_name": {
                event_name: dict(rates)
                for event_name, rates in self.rates_by_event_name.items()
            },
        }


class AgentRuntimeEventStore(Protocol):
    async def record_agent_runtime_events(
        self,
        events: tuple[AgentRuntimeEvent, ...],
    ) -> None: ...

    async def list_agent_runtime_events(
        self,
        query: AgentRuntimeEventQuery,
    ) -> list[AgentRuntimeEvent]: ...

    async def purge_agent_runtime_events(
        self,
        *,
        occurred_before: datetime,
        limit: int,
    ) -> int: ...


class AgentAssessmentStore(Protocol):
    async def record_agent_assessments(
        self,
        assessments: tuple[AgentAssessment, ...],
    ) -> None: ...

    async def list_agent_assessments(
        self,
        query: AgentAssessmentQuery,
    ) -> list[AgentAssessment]: ...


class RuntimeEventTraceSink(Protocol):
    """Best-effort post-commit projection seam for one durable event batch."""

    def publish(self, events: tuple[AgentRuntimeEvent, ...]) -> None: ...


class AgentAssessmentSink(Protocol):
    """Best-effort post-commit projection seam for durable assessments."""

    def publish(
        self,
        assessments: tuple[AgentAssessment, ...],
        events: tuple[AgentRuntimeEvent, ...],
    ) -> None: ...


class LangfuseObservation(Protocol):
    @property
    def id(self) -> str: ...

    def end(self) -> None: ...


class LangfuseClient(Protocol):
    def start_observation(self, **kwargs: object) -> LangfuseObservation: ...

    def create_event(self, **kwargs: object) -> object: ...

    def create_score(self, **kwargs: object) -> None: ...

    def shutdown(self) -> None: ...


class NoOpRuntimeEventTraceSink:
    def publish(self, events: tuple[AgentRuntimeEvent, ...]) -> None:
        del events


class NoOpAgentAssessmentSink:
    def publish(
        self,
        assessments: tuple[AgentAssessment, ...],
        events: tuple[AgentRuntimeEvent, ...],
    ) -> None:
        del assessments, events


class LangfuseAgentAssessmentSink:
    """Project DB-authoritative assessments as trace-level Langfuse Scores."""

    def __init__(self, client: LangfuseClient) -> None:
        self._client = client

    def publish(
        self,
        assessments: tuple[AgentAssessment, ...],
        events: tuple[AgentRuntimeEvent, ...],
    ) -> None:
        event_by_id = {event.event_id: event for event in events}
        for assessment in assessments:
            event = event_by_id.get(assessment.target_event_id)
            if event is None or assessment.status != "completed" or assessment.label is None:
                continue
            trace_id = _event_trace_id(event)
            self._client.create_score(
                score_id=assessment.assessment_id,
                name=f"memforge.{assessment.criterion}",
                value=assessment.label,
                data_type="CATEGORICAL",
                trace_id=trace_id,
                metadata=_langfuse_assessment_metadata(assessment),
                timestamp=assessment.created_at,
            )


class LangfuseRuntimeEventTraceSink:
    """Metadata-only Langfuse projection isolated from product correctness."""

    def __init__(
        self,
        client: LangfuseClient,
        attribute_scope: Callable[..., ContextManager[object]],
    ) -> None:
        self._client = client
        self._attribute_scope = attribute_scope

    def assessment_sink(self) -> AgentAssessmentSink:
        return LangfuseAgentAssessmentSink(self._client)

    def publish(self, events: tuple[AgentRuntimeEvent, ...]) -> None:
        if not events:
            return
        first = events[0]
        trace_id = first.trace_id or _event_trace_id(first)
        session_id, trace_name, version = _runtime_projection_profile(first)
        if any(_event_trace_id(event) != trace_id for event in events):
            raise ValueError("runtime event projection requires one execution per publish")
        with self._attribute_scope(
            session_id=session_id,
            trace_name=trace_name,
            version=version,
            tags=_langfuse_tags(first),
        ):
            root = self._client.start_observation(
                name=trace_name,
                as_type="span",
                trace_context={"trace_id": trace_id},
                metadata=_langfuse_batch_metadata(first, len(events)),
            )
            try:
                for event in events:
                    self._client.create_event(
                        name=event.event_name,
                        trace_context={
                            "trace_id": trace_id,
                            "parent_span_id": root.id,
                        },
                        metadata=_langfuse_event_metadata(event),
                        level=_langfuse_level(event.outcome),
                        status_message=event.reason_code,
                    )
            finally:
                root.end()


@lru_cache(maxsize=1)
def runtime_event_trace_sink_from_env() -> RuntimeEventTraceSink:
    """Build one process-level sink; environment changes require a restart."""

    if os.environ.get("MEMFORGE_LANGFUSE_ENABLED", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return NoOpRuntimeEventTraceSink()
    required_config = (
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_BASE_URL",
    )
    if any(not os.environ.get(name, "").strip() for name in required_config):
        logger.warning("Langfuse tracing is enabled but configuration is incomplete")
        return NoOpRuntimeEventTraceSink()
    try:
        langfuse_module = importlib.import_module("langfuse")
        span_filter_module = importlib.import_module("langfuse.span_filter")
        otel_trace_module = importlib.import_module("opentelemetry.sdk.trace")
        client = langfuse_module.Langfuse(
            tracer_provider=otel_trace_module.TracerProvider(),
            should_export_span=span_filter_module.is_langfuse_span,
            release=current_deployment_revision(),
        )
        attribute_scope = langfuse_module.propagate_attributes
    except Exception as exc:
        logger.warning(
            "Langfuse tracing is enabled but client initialization failed error_type=%s",
            type(exc).__name__,
        )
        return NoOpRuntimeEventTraceSink()
    atexit.register(client.shutdown)
    return LangfuseRuntimeEventTraceSink(client, attribute_scope)


def publish_runtime_events(
    sink: RuntimeEventTraceSink,
    events: tuple[AgentRuntimeEvent, ...],
) -> None:
    """Publish after commit; sink failure never changes the product result."""

    try:
        sink.publish(events)
    except Exception as exc:
        first = events[0] if events else None
        session_id = (
            _runtime_projection_profile(first)[0] if first is not None else "none"
        )
        trace_id = (
            _event_trace_id(first)
            if first is not None
            else "none"
        )
        logger.warning(
            "Agent runtime trace projection failed session_id=%s trace_id=%s event_count=%d error_type=%s",
            session_id,
            trace_id,
            len(events),
            type(exc).__name__,
        )


def assessment_sink_for_runtime_sink(
    sink: RuntimeEventTraceSink,
) -> AgentAssessmentSink:
    """Reuse the configured Langfuse client without coupling the core to it."""

    factory = getattr(sink, "assessment_sink", None)
    if callable(factory):
        return factory()
    return NoOpAgentAssessmentSink()


def publish_agent_assessments(
    sink: AgentAssessmentSink,
    assessments: tuple[AgentAssessment, ...],
    events: tuple[AgentRuntimeEvent, ...],
) -> None:
    """Publish after commit; projection failure never changes extraction."""

    try:
        sink.publish(assessments, events)
    except Exception as exc:
        first = events[0] if events else None
        logger.warning(
            "Agent assessment projection failed session_id=%s assessment_count=%d error_type=%s",
            _runtime_projection_profile(first)[0] if first is not None else "none",
            len(assessments),
            type(exc).__name__,
        )


class QualitySignalCollector:
    """Thread-safe request-local collection with a fixed cardinality bound."""

    def __init__(self, *, max_signals: int = 256) -> None:
        self._max_signals = max(1, int(max_signals))
        self._signals: list[QualitySignal] = []
        self._overflow_counts: Counter[tuple[str, AgentRuntimeOutcome, str]] = Counter()
        self._lock = Lock()

    def record(self, signal: QualitySignal) -> None:
        with self._lock:
            if len(self._signals) >= self._max_signals:
                self._overflow_counts[(signal.event_name, signal.outcome, signal.reason_code)] += 1
                return
            self._signals.append(signal)

    def snapshot(self) -> tuple[QualitySignal, ...]:
        with self._lock:
            values = tuple(self._signals)
            overflow_counts = dict(self._overflow_counts)
        if overflow_counts:
            values += tuple(
                QualitySignal(
                    event_name=event_name,
                    outcome=outcome,
                    reason_code=reason_code,
                    occurrence_count=count,
                )
                for (event_name, outcome, reason_code), count in sorted(overflow_counts.items())
            )
        return values


_CURRENT_QUALITY_COLLECTOR: ContextVar[QualitySignalCollector | None] = ContextVar(
    "memforge_agent_evaluation_quality_collector",
    default=None,
)


@contextmanager
def quality_signal_scope(
    collector: QualitySignalCollector,
) -> Iterator[QualitySignalCollector]:
    token = _CURRENT_QUALITY_COLLECTOR.set(collector)
    try:
        yield collector
    finally:
        _CURRENT_QUALITY_COLLECTOR.reset(token)


def record_quality_signal(signal: QualitySignal) -> None:
    collector = _CURRENT_QUALITY_COLLECTOR.get()
    if collector is not None:
        collector.record(signal)


def bind_quality_signals(
    signals: tuple[QualitySignal, ...],
    *,
    source_id: str,
    source_type: str,
    doc_id: str,
    source_unit_id: str,
    target_unit_revision_id: str,
    projection_run_id: str,
    derivation_id: str,
    batch_id: str,
    batch_attempt: int,
    extraction_contract_version: str,
    occurred_at: datetime | None = None,
    deployment_revision: str | None = None,
    trace_context: RuntimeTraceContext | None = None,
    observation_revision_ids: Mapping[str, str] | None = None,
) -> tuple[AgentRuntimeEvent, ...]:
    """Attach stable lineage and deterministic replay-safe event identity."""

    if batch_attempt < 1:
        raise ValueError("batch_attempt must be positive")
    timestamp = occurred_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("agent runtime event timestamp requires a timezone")
    timestamp = timestamp.astimezone(timezone.utc)
    resolved_trace_context = trace_context or RuntimeTraceContext(
        trace_id=runtime_trace_id(
            derivation_id=derivation_id,
            batch_id=batch_id,
            batch_attempt=batch_attempt,
        ),
        span_id="0" * 16,
        trace_flags=None,
    )
    operation_input_hash = hashlib.sha256(
        json.dumps(
            {
                "source_id": source_id,
                "source_unit_id": source_unit_id,
                "target_unit_revision_id": target_unit_revision_id,
                "projection_run_id": projection_run_id,
                "derivation_id": derivation_id,
                "batch_id": batch_id,
                "contract_version": extraction_contract_version,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    operation_id = _runtime_identity(
        "source-derivation-batch-operation-v1",
        derivation_id,
        batch_id,
        operation_input_hash,
        prefix="aop",
    )
    execution_id = _runtime_identity(
        "source-derivation-batch-execution-v1",
        operation_id,
        str(batch_attempt),
        prefix="aex",
    )
    events = []
    for index, signal in enumerate(signals):
        if (
            signal.observation_id is not None
            and signal.observation_revision_id is None
            and observation_revision_ids is not None
        ):
            revision_id = observation_revision_ids.get(signal.observation_id)
            if revision_id is not None:
                signal = replace(signal, observation_revision_id=revision_id)
        identity = {
            "schema_version": AGENT_RUNTIME_EVENT_SCHEMA_VERSION,
            "derivation_id": derivation_id,
            "batch_id": batch_id,
            "batch_attempt": batch_attempt,
            "index": index,
            "signal": asdict(signal),
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:24]
        payload_hash = hashlib.sha256(
            json.dumps(
                {
                    "identity": identity,
                    "source_id": source_id,
                    "source_type": source_type,
                    "doc_id": doc_id,
                    "source_unit_id": source_unit_id,
                    "target_unit_revision_id": target_unit_revision_id,
                    "projection_run_id": projection_run_id,
                    "contract_version": extraction_contract_version,
                    "deployment_revision": deployment_revision,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        events.append(
            AgentRuntimeEvent(
                event_id=f"are-{digest}",
                operation_id=operation_id,
                execution_id=execution_id,
                contract_version=extraction_contract_version,
                payload_hash=payload_hash,
                operation_input_hash=operation_input_hash,
                execution_owner_id=f"{derivation_id}:{batch_id}:{batch_attempt}",
                occurred_at=timestamp,
                source_id=source_id,
                source_type=source_type,
                doc_id=doc_id,
                source_unit_id=source_unit_id,
                target_unit_revision_id=target_unit_revision_id,
                projection_run_id=projection_run_id,
                derivation_id=derivation_id,
                batch_id=batch_id,
                batch_attempt=batch_attempt,
                extraction_contract_version=extraction_contract_version,
                deployment_revision=deployment_revision,
                trace_id=resolved_trace_context.trace_id,
                span_id=(
                    resolved_trace_context.span_id
                    if resolved_trace_context.span_id != "0" * 16
                    else None
                ),
                trace_flags=resolved_trace_context.trace_flags,
                **asdict(signal),
            )
        )
    return tuple(events)


def bind_source_lifecycle_outcome(
    *,
    source_id: str,
    source_type: str,
    doc_id: str,
    source_unit_id: str,
    base_unit_revision_id: str | None,
    target_unit_revision_id: str,
    projection_run_id: str,
    operation_input_hash: str,
    execution_owner_id: str,
    outcome: AgentRuntimeOutcome,
    reason_code: str,
    attempt_count: int,
    duration_ms: int,
    incumbent_count: int,
    relation_pair_count: int,
    mutation_count: int,
    review_count: int,
    model_call_count: int,
    occurred_at: datetime | None = None,
    deployment_revision: str | None = None,
) -> AgentRuntimeBundle:
    """Bind one durable Source Unit lifecycle terminal result and assessment."""

    _require_hash("operation_input_hash", operation_input_hash)
    if attempt_count < 1:
        raise ValueError("attempt_count must be positive")
    for name, value in (
        ("duration_ms", duration_ms),
        ("incumbent_count", incumbent_count),
        ("relation_pair_count", relation_pair_count),
        ("mutation_count", mutation_count),
        ("review_count", review_count),
        ("model_call_count", model_call_count),
    ):
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    timestamp = occurred_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("agent runtime event timestamp requires a timezone")
    timestamp = timestamp.astimezone(timezone.utc)
    operation_id = _runtime_identity(
        "source-unit-lifecycle-operation-v1",
        source_id,
        source_unit_id,
        base_unit_revision_id or "none",
        target_unit_revision_id,
        operation_input_hash,
        SOURCE_UNIT_LIFECYCLE_CONTRACT_VERSION,
        prefix="aop",
    )
    execution_id = _runtime_identity(
        "source-unit-lifecycle-execution-v1",
        operation_id,
        execution_owner_id,
        prefix="aex",
    )
    event_id = _runtime_identity(
        AGENT_RUNTIME_EVENT_SCHEMA_VERSION,
        execution_id,
        "source_unit_lifecycle_outcome",
        prefix="are",
    )
    payload = {
        "schema_version": AGENT_RUNTIME_EVENT_SCHEMA_VERSION,
        "event_id": event_id,
        "event_name": "source_unit_lifecycle_outcome",
        "operation_id": operation_id,
        "execution_id": execution_id,
        "contract_version": SOURCE_UNIT_LIFECYCLE_CONTRACT_VERSION,
        "operation_input_hash": operation_input_hash,
        "execution_owner_id": execution_owner_id,
        "outcome": outcome,
        "reason_code": reason_code,
        "source_id": source_id,
        "source_type": source_type,
        "doc_id": doc_id,
        "source_unit_id": source_unit_id,
        "base_unit_revision_id": base_unit_revision_id,
        "target_unit_revision_id": target_unit_revision_id,
        "projection_run_id": projection_run_id,
        "deployment_revision": deployment_revision,
        "attempt_count": attempt_count,
        "duration_ms": duration_ms,
        "recovered": attempt_count > 1 and outcome != "failed",
        "incumbent_count": incumbent_count,
        "relation_pair_count": relation_pair_count,
        "mutation_count": mutation_count,
        "review_count": review_count,
        "model_call_count": model_call_count,
    }
    payload_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    trace_id = runtime_execution_trace_id(execution_id)
    event = AgentRuntimeEvent(
        **payload,
        occurred_at=timestamp,
        payload_hash=payload_hash,
        trace_id=trace_id,
        operation="source_unit_lifecycle_reconciliation",
    )
    label: AgentAssessmentLabel = "fail" if outcome == "failed" else "pass"
    assessment_id = _runtime_identity(
        AGENT_ASSESSMENT_SCHEMA_VERSION,
        event_id,
        "source_unit_lifecycle_completion",
        f"{DETERMINISTIC_RUNTIME_EVALUATOR_NAME}:{DETERMINISTIC_RUNTIME_EVALUATOR_VERSION}",
        prefix="aas",
    )
    assessment = AgentAssessment(
        assessment_id=assessment_id,
        target_event_id=event_id,
        criterion="source_unit_lifecycle_completion",
        status="completed",
        label=label,
        reason_code=reason_code,
        annotator_kind="code",
        evaluator_name=DETERMINISTIC_RUNTIME_EVALUATOR_NAME,
        evaluator_version=DETERMINISTIC_RUNTIME_EVALUATOR_VERSION,
        created_at=timestamp,
    )
    return AgentRuntimeBundle(events=(event,), assessments=(assessment,))


def _runtime_identity(seed_version: str, *parts: str, prefix: str) -> str:
    seed = ":".join((seed_version, *parts))
    return f"{prefix}-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]


def evaluate_runtime_events(
    events: tuple[AgentRuntimeEvent, ...],
) -> tuple[AgentAssessment, ...]:
    """Apply the small, explicit deterministic online-evaluation contract."""

    assessments: list[AgentAssessment] = []
    for event in events:
        decision = _deterministic_assessment_decision(event)
        if decision is None:
            continue
        criterion, label = decision
        identity = (
            f"{AGENT_ASSESSMENT_SCHEMA_VERSION}:{event.event_id}:{criterion}:"
            f"{DETERMINISTIC_RUNTIME_EVALUATOR_NAME}:"
            f"{DETERMINISTIC_RUNTIME_EVALUATOR_VERSION}"
        )
        assessment_id = "aas-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
        assessments.append(
            AgentAssessment(
                assessment_id=assessment_id,
                target_event_id=event.event_id,
                criterion=criterion,
                status="completed",
                label=label,
                reason_code=event.reason_code,
                annotator_kind="code",
                evaluator_name=DETERMINISTIC_RUNTIME_EVALUATOR_NAME,
                evaluator_version=DETERMINISTIC_RUNTIME_EVALUATOR_VERSION,
                created_at=event.occurred_at,
                occurrence_count=event.occurrence_count,
            )
        )
    return tuple(assessments)


def _deterministic_assessment_decision(
    event: AgentRuntimeEvent,
) -> tuple[str, AgentAssessmentLabel] | None:
    if event.event_name == "source_unit_lifecycle_outcome":
        return (
            "source_unit_lifecycle_completion",
            "fail" if event.outcome == "failed" else "pass",
        )
    if event.event_name == "structured_output_outcome":
        return (
            "structured_output_contract",
            "pass" if event.outcome in {"expected", "degraded"} else "fail",
        )
    if event.event_name == "evidence_admission_outcome":
        return "evidence_reference_validity", "fail"
    if event.event_name == "evidence_localization_outcome":
        return (
            "evidence_localization",
            "needs_review" if event.reason_code == "whole_block_fallback" else "pass",
        )
    if event.event_name == "extraction_batch_outcome":
        if event.outcome == "failed":
            return "extraction_completion", "fail"
        if event.reason_code == "candidates_extracted":
            return "extraction_completion", "pass"
    return None


def runtime_trace_id(
    *,
    derivation_id: str,
    batch_id: str,
    batch_attempt: int,
) -> str:
    """Return the stable W3C-compatible trace correlation for one attempt."""

    seed = f"memforge-agent-runtime-trace-v1:{derivation_id}:{batch_id}:{batch_attempt}"
    trace_id = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:32]
    return trace_id if trace_id != "0" * 32 else "0" * 31 + "1"


def runtime_execution_trace_id(execution_id: str) -> str:
    """Return the W3C-compatible trace projection for one durable execution."""

    trace_id = hashlib.sha256(
        f"memforge-agent-execution-trace-v1:{execution_id}".encode("utf-8")
    ).hexdigest()[:32]
    return trace_id if trace_id != "0" * 32 else "0" * 31 + "1"


def runtime_operation_session_id(operation_id: str) -> str:
    """Return the bounded Langfuse Session projection for one operation."""

    seed = f"memforge-agent-operation-session-v1:{operation_id}"
    return "mfo1-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:32]


def _event_trace_id(event: AgentRuntimeEvent) -> str:
    if event.trace_id is not None:
        return event.trace_id
    if event.execution_id is not None:
        return runtime_execution_trace_id(event.execution_id)
    if event.derivation_id is None or event.batch_id is None or event.batch_attempt is None:
        raise ValueError("runtime event has no traceable execution identity")
    return runtime_trace_id(
        derivation_id=event.derivation_id,
        batch_id=event.batch_id,
        batch_attempt=event.batch_attempt,
    )


def _runtime_projection_profile(event: AgentRuntimeEvent) -> tuple[str, str, str | None]:
    if event.event_name == "source_unit_lifecycle_outcome":
        if event.operation_id is None:
            raise ValueError("lifecycle runtime event requires operation_id")
        return (
            runtime_operation_session_id(event.operation_id),
            "memforge.agent.reconcile_source_unit",
            event.contract_version,
        )
    return (
        runtime_session_id(event.projection_run_id),
        "memforge.agent.extraction_batch",
        event.extraction_contract_version,
    )


def runtime_session_id(projection_run_id: str) -> str:
    """Return the bounded Langfuse Session correlation for one Projection."""

    seed = f"memforge-agent-runtime-session-v1:{projection_run_id}"
    return "mfs1-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:32]


def current_deployment_revision() -> str | None:
    """Return a bounded deploy identity without depending on a Cloud adapter."""

    configured = os.environ.get("MEMFORGE_DEPLOYMENT_REVISION", "").strip()
    if configured:
        return _safe_revision(configured)
    raw_vcap = os.environ.get("VCAP_APPLICATION", "").strip()
    if not raw_vcap:
        return None
    try:
        payload = json.loads(raw_vcap)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, Mapping):
        return None
    for key in ("application_version", "version"):
        value = payload.get(key)
        if isinstance(value, str) and (revision := _safe_revision(value)) is not None:
            return revision
    return None


def runtime_event_otel_attributes(event: AgentRuntimeEvent) -> Mapping[str, object]:
    """Return a bounded, content-free projection for an OTLP exporter."""

    values: dict[str, object | None] = {
        "memforge.agent.event_id": event.event_id,
        "memforge.agent.schema_version": event.schema_version,
        "memforge.agent.operation_id": event.operation_id,
        "memforge.agent.execution_id": event.execution_id,
        "memforge.agent.contract.version": event.contract_version,
        "memforge.agent.outcome": event.outcome,
        "memforge.agent.reason_code": event.reason_code,
        "memforge.source.type": event.source_type,
        "memforge.extraction.contract.version": event.extraction_contract_version,
        "memforge.deployment.revision": event.deployment_revision,
        "gen_ai.operation.name": event.operation,
        "gen_ai.provider.name": event.provider,
        "gen_ai.request.model": event.model,
        "memforge.batch.attempt": event.batch_attempt,
        "memforge.agent.attempt_count": event.attempt_count,
        "memforge.agent.duration_ms": event.duration_ms,
        "memforge.agent.recovered": event.recovered,
        "memforge.agent.incumbent_count": event.incumbent_count,
        "memforge.agent.relation_pair_count": event.relation_pair_count,
        "memforge.agent.mutation_count": event.mutation_count,
        "memforge.agent.review_count": event.review_count,
        "memforge.agent.model_call_count": event.model_call_count,
        "gen_ai.request.max_tokens": event.requested_max_tokens,
        "gen_ai.response.finish_reasons": (
            [event.finish_reason] if event.finish_reason is not None else None
        ),
        "gen_ai.usage.input_tokens": event.prompt_tokens,
        "gen_ai.usage.output_tokens": event.completion_tokens,
        "memforge.structured.attempt_index": event.attempt_index,
        "memforge.structured.mode": event.structured_mode,
        "memforge.structured.schema_transport": event.schema_transport,
        "memforge.structured.terminal_category": event.terminal_category,
        "memforge.structured.response_chars": event.response_chars,
        "memforge.evidence.localization_mode": event.localization_mode,
        "memforge.agent.candidate_count": event.candidate_count,
        "memforge.agent.rejected_count": event.rejected_count,
        "memforge.agent.occurrence_count": event.occurrence_count,
    }
    return {name: value for name, value in values.items() if value is not None}


def event_public_payload(event: AgentRuntimeEvent) -> Mapping[str, object]:
    """Return the fixed, content-free payload suitable for logs and exports."""

    payload = asdict(event)
    payload["occurred_at"] = event.occurred_at.isoformat()
    return payload


def assessment_public_payload(
    assessment: AgentAssessment,
) -> Mapping[str, object]:
    """Return the fixed, content-free assessment payload."""

    payload = asdict(assessment)
    payload["created_at"] = assessment.created_at.isoformat()
    return payload


def summarize_agent_assessments(
    assessments: tuple[AgentAssessment, ...] | list[AgentAssessment],
) -> dict[str, object]:
    """Return bounded counts for an authorized assessment cohort."""

    label_counts: Counter[str] = Counter()
    criterion_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    for assessment in assessments:
        if assessment.label is not None:
            label_counts[assessment.label] += assessment.occurrence_count
        criterion_counts[assessment.criterion] += assessment.occurrence_count
        status_counts[assessment.status] += assessment.occurrence_count
    return {
        "total_assessments": sum(item.occurrence_count for item in assessments),
        "label_counts": dict(sorted(label_counts.items())),
        "criterion_counts": dict(sorted(criterion_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
    }


def build_source_online_evaluation_view(
    events: tuple[AgentRuntimeEvent, ...] | list[AgentRuntimeEvent],
    assessments: tuple[AgentAssessment, ...] | list[AgentAssessment],
    *,
    representative_limit: int = 3,
) -> dict[str, object]:
    """Return one actionable, content-free Source evaluation presentation.

    Coverage is semantic: a persisted assessment covers the expected check when
    its event, criterion, evaluator, and evaluator version match. Assessment
    schema versions and storage IDs remain idempotency details and do not create
    false user-facing coverage gaps.
    """

    if not 1 <= representative_limit <= 10:
        raise ValueError("representative_limit must be between 1 and 10")
    event_rows = list(events)
    event_by_id = {event.event_id: event for event in event_rows}
    expected = list(evaluate_runtime_events(tuple(event_rows)))
    effective = _effective_event_assessments(
        assessments,
        visible_event_ids=frozenset(event_by_id),
    )
    effective_by_key = {
        _event_assessment_semantic_key(assessment): assessment
        for assessment in effective
    }

    eligible_occurrences = sum(item.occurrence_count for item in expected)
    assessed_occurrences = 0
    pending_occurrences = 0
    pending_times: list[datetime] = []
    for item in expected:
        persisted = effective_by_key.get(_event_assessment_semantic_key(item))
        if persisted is not None and persisted.status == "completed":
            assessed_occurrences += item.occurrence_count
        else:
            pending_occurrences += item.occurrence_count
            target = event_by_id.get(item.target_event_id or "")
            if target is not None:
                pending_times.append(target.occurred_at)

    evaluator_failure_occurrences = sum(
        item.occurrence_count for item in effective if item.status == "failed"
    )
    issue_groups = _source_online_evaluation_issue_groups(
        effective,
        event_by_id=event_by_id,
        representative_limit=representative_limit,
    )
    summary = summarize_agent_assessments(effective)
    summary.update(
        {
            "runtime_event_count": sum(event.occurrence_count for event in event_rows),
            "eligible_assessment_count": eligible_occurrences,
            "missing_assessment_count": pending_occurrences,
            "action_issue_group_count": sum(
                1 for group in issue_groups if group["label"] == "fail"
            ),
            "review_issue_group_count": sum(
                1 for group in issue_groups if group["label"] == "needs_review"
            ),
        }
    )
    return {
        "summary": summary,
        "coverage": {
            "policy": ONLINE_EVALUATION_COVERAGE_POLICY,
            "eligible_occurrences": eligible_occurrences,
            "assessed_occurrences": assessed_occurrences,
            "pending_occurrences": pending_occurrences,
            "coverage_rate": (
                assessed_occurrences / eligible_occurrences
                if eligible_occurrences
                else 1.0
            ),
            "oldest_pending_at": (
                min(pending_times).isoformat() if pending_times else None
            ),
            "evaluator_failure_occurrences": evaluator_failure_occurrences,
        },
        "issue_groups": issue_groups,
        "runtime_events": [event_public_payload(event) for event in event_rows[:50]],
        "assessments": [
            assessment_public_payload(assessment) for assessment in effective[:50]
        ],
    }


def build_workspace_online_evaluation_view(
    sources: Sequence[Mapping[str, object]],
    events: tuple[AgentRuntimeEvent, ...] | list[AgentRuntimeEvent],
    assessments: tuple[AgentAssessment, ...] | list[AgentAssessment],
    *,
    representative_limit: int = 3,
) -> dict[str, object]:
    """Return one workspace-scoped presentation over an authorized Source cohort.

    Callers must supply only Sources the principal may discover. Events and
    assessments are filtered against that explicit cohort again so a drifting
    adapter cannot leak a hidden Source through aggregate counts.
    """

    normalized_sources = [_online_evaluation_source(source) for source in sources]
    source_ids = frozenset(source["source_id"] for source in normalized_sources)
    event_rows = [event for event in events if event.source_id in source_ids]
    visible_event_ids = frozenset(event.event_id for event in event_rows)
    assessment_rows = [
        assessment
        for assessment in assessments
        if assessment.target_event_id in visible_event_ids
    ]
    workspace_view = build_source_online_evaluation_view(
        event_rows,
        assessment_rows,
        representative_limit=representative_limit,
    )

    event_by_source: dict[str, list[AgentRuntimeEvent]] = {
        source_id: [] for source_id in source_ids
    }
    for event in event_rows:
        event_by_source[event.source_id].append(event)
    assessment_by_event: dict[str, list[AgentAssessment]] = {}
    for assessment in assessment_rows:
        if assessment.target_event_id is None:
            continue
        assessment_by_event.setdefault(assessment.target_event_id, []).append(assessment)

    source_health: list[dict[str, object]] = []
    for source in normalized_sources:
        source_id = source["source_id"]
        source_events = event_by_source[source_id]
        source_event_ids = frozenset(event.event_id for event in source_events)
        source_assessments = [
            assessment
            for event_id in source_event_ids
            for assessment in assessment_by_event.get(event_id, ())
        ]
        view = build_source_online_evaluation_view(
            source_events,
            source_assessments,
            representative_limit=representative_limit,
        )
        summary = view["summary"]
        coverage = view["coverage"]
        assert isinstance(summary, Mapping)
        assert isinstance(coverage, Mapping)
        action_groups = int(summary["action_issue_group_count"])
        review_groups = int(summary["review_issue_group_count"])
        pending = int(coverage["pending_occurrences"])
        evaluator_failures = int(coverage["evaluator_failure_occurrences"])
        eligible = int(coverage["eligible_occurrences"])
        if action_groups:
            evaluation_status = "attention"
        elif pending or evaluator_failures:
            evaluation_status = "coverage_gap"
        elif review_groups:
            evaluation_status = "review"
        elif eligible == 0:
            evaluation_status = "no_data"
        else:
            evaluation_status = "healthy"
        label_counts = summary.get("label_counts")
        assert isinstance(label_counts, Mapping)
        source_health.append(
            {
                **source,
                "evaluation_status": evaluation_status,
                "action_issue_group_count": action_groups,
                "review_issue_group_count": review_groups,
                "fail_occurrences": int(label_counts.get("fail") or 0),
                "review_occurrences": int(label_counts.get("needs_review") or 0),
                "coverage": dict(coverage),
                "last_event_at": (
                    max(event.occurred_at for event in source_events).isoformat()
                    if source_events
                    else None
                ),
            }
        )

    status_rank = {
        "attention": 0,
        "coverage_gap": 1,
        "review": 2,
        "healthy": 3,
        "no_data": 4,
    }
    source_health.sort(
        key=lambda source: (
            status_rank[str(source["evaluation_status"])],
            -int(source["fail_occurrences"]),
            -int(source["review_occurrences"]),
            str(source["name"]).casefold(),
            str(source["source_id"]),
        )
    )
    affected_source_ids = {
        str(source["source_id"])
        for source in source_health
        if source["evaluation_status"] in {"attention", "coverage_gap", "review"}
    }
    summary = dict(workspace_view["summary"])
    summary.update(
        {
            "source_count": len(source_health),
            "affected_source_count": len(affected_source_ids),
        }
    )
    workspace_view["summary"] = summary
    workspace_view["sources"] = source_health
    return workspace_view


def _online_evaluation_source(source: Mapping[str, object]) -> dict[str, str]:
    source_id = str(source.get("id") or source.get("source_id") or "").strip()
    if not source_id:
        raise ValueError("online evaluation source_id is required")
    return {
        "source_id": source_id,
        "name": str(source.get("name") or source_id),
        "type": str(source.get("type") or "unknown"),
        "source_status": str(source.get("status") or "unknown"),
    }


def _event_assessment_semantic_key(
    assessment: AgentAssessment,
) -> tuple[str | None, str, str, str]:
    return (
        assessment.target_event_id,
        assessment.criterion,
        assessment.evaluator_name,
        assessment.evaluator_version,
    )


def _effective_event_assessments(
    assessments: tuple[AgentAssessment, ...] | list[AgentAssessment],
    *,
    visible_event_ids: frozenset[str],
) -> list[AgentAssessment]:
    by_key: dict[tuple[str | None, str, str, str], AgentAssessment] = {}
    for assessment in assessments:
        if assessment.target_event_id not in visible_event_ids:
            continue
        key = _event_assessment_semantic_key(assessment)
        incumbent = by_key.get(key)
        if incumbent is None or _assessment_preference(assessment) > _assessment_preference(
            incumbent
        ):
            by_key[key] = assessment
    return sorted(
        by_key.values(),
        key=lambda item: (item.created_at, item.assessment_id),
        reverse=True,
    )


def _assessment_preference(assessment: AgentAssessment) -> tuple[bool, datetime, str]:
    return (
        assessment.schema_version == AGENT_ASSESSMENT_SCHEMA_VERSION,
        assessment.created_at,
        assessment.assessment_id,
    )


def _source_online_evaluation_issue_groups(
    assessments: list[AgentAssessment],
    *,
    event_by_id: Mapping[str, AgentRuntimeEvent],
    representative_limit: int,
) -> list[dict[str, object]]:
    criterion_occurrences: Counter[str] = Counter()
    for assessment in assessments:
        if assessment.status == "completed":
            criterion_occurrences[assessment.criterion] += assessment.occurrence_count
    grouped: dict[tuple[str, str, str, str, str], dict[str, object]] = {}
    for assessment in assessments:
        if assessment.status != "completed" or assessment.label not in {
            "fail",
            "needs_review",
        }:
            continue
        event = event_by_id.get(assessment.target_event_id or "")
        if event is None:
            continue
        key = (
            assessment.label,
            assessment.criterion,
            assessment.reason_code,
            assessment.evaluator_name,
            assessment.evaluator_version,
        )
        group = grouped.setdefault(
            key,
            {
                "group_id": "aeg-"
                + hashlib.sha256(":".join(key).encode("utf-8")).hexdigest()[:16],
                "label": assessment.label,
                "criterion": assessment.criterion,
                "reason_code": assessment.reason_code,
                "evaluator_name": assessment.evaluator_name,
                "evaluator_version": assessment.evaluator_version,
                "occurrence_count": 0,
                "distinct_event_count": 0,
                "first_seen_at": event.occurred_at,
                "last_seen_at": event.occurred_at,
                "affected_source_ids": [],
                "source_types": [],
                "representative_cases": [],
            },
        )
        group["occurrence_count"] = int(group["occurrence_count"]) + assessment.occurrence_count
        group["distinct_event_count"] = int(group["distinct_event_count"]) + 1
        group["first_seen_at"] = min(group["first_seen_at"], event.occurred_at)
        group["last_seen_at"] = max(group["last_seen_at"], event.occurred_at)
        affected_source_ids = group["affected_source_ids"]
        source_types = group["source_types"]
        assert isinstance(affected_source_ids, list)
        assert isinstance(source_types, list)
        if event.source_id not in affected_source_ids:
            affected_source_ids.append(event.source_id)
        if event.source_type not in source_types:
            source_types.append(event.source_type)
        representatives = group["representative_cases"]
        assert isinstance(representatives, list)
        if len(representatives) < representative_limit:
            representatives.append(_source_online_evaluation_case(assessment, event))

    values = list(grouped.values())
    for group in values:
        group["first_seen_at"] = group["first_seen_at"].isoformat()
        group["last_seen_at"] = group["last_seen_at"].isoformat()
        denominator = criterion_occurrences[str(group["criterion"])]
        group["criterion_occurrence_count"] = denominator
        group["criterion_rate"] = (
            int(group["occurrence_count"]) / denominator if denominator else 0.0
        )
        group["affected_source_ids"] = sorted(group["affected_source_ids"])
        group["source_types"] = sorted(group["source_types"])
        group["affected_source_count"] = len(group["affected_source_ids"])
    values.sort(
        key=lambda group: (
            0 if group["label"] == "fail" else 1,
            -int(group["occurrence_count"]),
            str(group["criterion"]),
            str(group["reason_code"]),
        )
    )
    return values


def _source_online_evaluation_case(
    assessment: AgentAssessment,
    event: AgentRuntimeEvent,
) -> dict[str, object]:
    return {
        "assessment_id": assessment.assessment_id,
        "event_id": event.event_id,
        "label": assessment.label,
        "criterion": assessment.criterion,
        "reason_code": assessment.reason_code,
        "occurred_at": event.occurred_at.isoformat(),
        "occurrence_count": assessment.occurrence_count,
        "source_id": event.source_id,
        "source_type": event.source_type,
        "doc_id": event.doc_id,
        "source_unit_id": event.source_unit_id,
        "target_unit_revision_id": event.target_unit_revision_id,
        "observation_id": event.observation_id,
        "observation_revision_id": event.observation_revision_id,
        "projection_run_id": event.projection_run_id,
        "operation_id": event.operation_id,
        "execution_id": event.execution_id,
        "derivation_id": event.derivation_id,
        "batch_id": event.batch_id,
        "trace_id": event.trace_id,
        "provider": event.provider,
        "model": event.model,
        "contract_version": event.contract_version,
        "extraction_contract_version": event.extraction_contract_version,
        "deployment_revision": event.deployment_revision,
    }


def _langfuse_batch_metadata(
    event: AgentRuntimeEvent,
    event_count: int,
) -> dict[str, object]:
    values: dict[str, object | None] = {
        "schema_version": event.schema_version,
        "operation_id": event.operation_id,
        "execution_id": event.execution_id,
        "contract_version": event.contract_version,
        "source_type": event.source_type,
        "extraction_contract_version": event.extraction_contract_version,
        "deployment_revision": event.deployment_revision,
        "provider": event.provider,
        "model": event.model,
        "batch_attempt": event.batch_attempt,
        "attempt_count": event.attempt_count,
        "duration_ms": event.duration_ms,
        "recovered": event.recovered,
        "event_count": event_count,
    }
    return {name: value for name, value in values.items() if value is not None}


def _langfuse_tags(event: AgentRuntimeEvent) -> list[str]:
    tags = [
        "memforge-agent-eval",
        (
            "source-unit-lifecycle"
            if event.event_name == "source_unit_lifecycle_outcome"
            else "memory-extraction"
        ),
    ]
    if event.source_type and len(event.source_type) <= 128 and all(
        ch in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-"
        for ch in event.source_type
    ):
        tags.append(f"source-type:{event.source_type}")
    return tags


def _langfuse_event_metadata(event: AgentRuntimeEvent) -> dict[str, object]:
    """Return the allowlisted, content-free Langfuse event projection."""

    values: dict[str, object | None] = {
        "event_id": event.event_id,
        "schema_version": event.schema_version,
        "operation_id": event.operation_id,
        "execution_id": event.execution_id,
        "contract_version": event.contract_version,
        "outcome": event.outcome,
        "reason_code": event.reason_code,
        "source_type": event.source_type,
        "extraction_contract_version": event.extraction_contract_version,
        "deployment_revision": event.deployment_revision,
        "operation": event.operation,
        "provider": event.provider,
        "model": event.model,
        "batch_attempt": event.batch_attempt,
        "localization_mode": event.localization_mode,
        "attempt_count": event.attempt_count,
        "duration_ms": event.duration_ms,
        "recovered": event.recovered,
        "incumbent_count": event.incumbent_count,
        "relation_pair_count": event.relation_pair_count,
        "mutation_count": event.mutation_count,
        "review_count": event.review_count,
        "model_call_count": event.model_call_count,
        "retry_count": event.retry_count,
        "fallback_count": event.fallback_count,
        "attempt_index": event.attempt_index,
        "structured_mode": event.structured_mode,
        "schema_transport": event.schema_transport,
        "requested_max_tokens": event.requested_max_tokens,
        "terminal_category": event.terminal_category,
        "error_code": event.error_code,
        "finish_reason": event.finish_reason,
        "stop_reason": event.stop_reason,
        "prompt_tokens": event.prompt_tokens,
        "completion_tokens": event.completion_tokens,
        "total_tokens": event.total_tokens,
        "response_chars": event.response_chars,
        "validation_location": event.validation_location,
        "validation_rule": event.validation_rule,
        "json_error_line": event.json_error_line,
        "json_error_column": event.json_error_column,
        "candidate_count": event.candidate_count,
        "rejected_count": event.rejected_count,
        "occurrence_count": event.occurrence_count,
    }
    return {name: value for name, value in values.items() if value is not None}


def _langfuse_assessment_metadata(
    assessment: AgentAssessment,
) -> dict[str, object]:
    """Return the bounded, content-free Langfuse Score metadata."""

    return {
        "assessment_id": assessment.assessment_id,
        "target_event_id": assessment.target_event_id,
        "target_result_id": assessment.target_result_id,
        "target_candidate_id": assessment.target_candidate_id,
        "schema_version": assessment.schema_version,
        "reason_code": assessment.reason_code,
        "annotator_kind": assessment.annotator_kind,
        "evaluator_name": assessment.evaluator_name,
        "evaluator_version": assessment.evaluator_version,
        "occurrence_count": assessment.occurrence_count,
    }


def _langfuse_level(outcome: AgentRuntimeOutcome) -> str:
    if outcome == "failed":
        return "ERROR"
    if outcome in {"degraded", "rejected"}:
        return "WARNING"
    return "DEFAULT"


def summarize_agent_runtime_events(
    events: tuple[AgentRuntimeEvent, ...] | list[AgentRuntimeEvent],
) -> AgentRuntimeCohortReport:
    """Aggregate a bounded cohort without treating degraded events as labels."""

    event_name_counts = Counter()
    outcome_counts = Counter()
    reason_counts = Counter()
    outcome_by_event_name: dict[str, Counter[str]] = {}
    for event in events:
        event_name_counts[event.event_name] += event.occurrence_count
        outcome_counts[event.outcome] += event.occurrence_count
        reason_counts[event.reason_code] += event.occurrence_count
        outcome_by_event_name.setdefault(event.event_name, Counter())[event.outcome] += (
            event.occurrence_count
        )
    rates_by_event_name = {
        event_name: {
            outcome: count / event_name_counts[event_name]
            for outcome, count in sorted(outcomes.items())
        }
        for event_name, outcomes in sorted(outcome_by_event_name.items())
    }
    return AgentRuntimeCohortReport(
        total_events=sum(event.occurrence_count for event in events),
        event_name_counts=dict(sorted(event_name_counts.items())),
        outcome_counts=dict(sorted(outcome_counts.items())),
        reason_counts=dict(sorted(reason_counts.items())),
        rates_by_event_name=rates_by_event_name,
    )


def _require_safe_label(name: str, value: str | None) -> None:
    if value is None:
        return
    if not value or len(value) > 128 or any(
        ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-"
        for ch in value
    ):
        raise ValueError(f"{name} must be a bounded machine-readable label")


def _require_safe_identifier(name: str, value: str | None) -> None:
    if value is None:
        return
    if not value or len(value) > 255 or any(
        ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:/-"
        for ch in value
    ):
        raise ValueError(f"{name} must be a bounded machine-readable identifier")


def _require_bounded_principal_id(name: str, value: str | None) -> None:
    if value is None:
        return
    if not value or len(value) > 255 or any(ord(ch) < 33 or ord(ch) == 127 for ch in value):
        raise ValueError(f"{name} must be a bounded principal identifier")


def _require_safe_diagnostic_path(name: str, value: str | None) -> None:
    if value is None:
        return
    if not value or len(value) > 255 or any(
        ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.$[]:-"
        for ch in value
    ):
        raise ValueError(f"{name} must be a bounded diagnostic path")


def _require_hash(name: str, value: str | None) -> None:
    if value is None:
        return
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _require_model(value: str | None) -> None:
    if value is None:
        return
    if not value or len(value) > 256 or any(
        ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./:-"
        for ch in value
    ):
        raise ValueError("model must be a bounded machine-readable identifier")


def _safe_revision(value: str) -> str | None:
    if not value or len(value) > 255 or any(
        ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-"
        for ch in value
    ):
        return None
    return value
