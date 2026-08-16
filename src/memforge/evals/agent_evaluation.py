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
from typing import Callable, ContextManager, Iterator, Literal, Mapping, Protocol


AGENT_RUNTIME_EVENT_SCHEMA_VERSION = "agent-runtime-event-v2"
AgentRuntimeOutcome = Literal["expected", "degraded", "rejected", "failed"]
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
    derivation_id: str
    batch_id: str
    batch_attempt: int
    extraction_contract_version: str
    schema_version: str = AGENT_RUNTIME_EVENT_SCHEMA_VERSION
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


class RuntimeEventTraceSink(Protocol):
    """Best-effort post-commit projection seam for one durable event batch."""

    def publish(self, events: tuple[AgentRuntimeEvent, ...]) -> None: ...


class LangfuseObservation(Protocol):
    @property
    def id(self) -> str: ...

    def end(self) -> None: ...


class LangfuseClient(Protocol):
    def start_observation(self, **kwargs: object) -> LangfuseObservation: ...

    def create_event(self, **kwargs: object) -> object: ...

    def shutdown(self) -> None: ...


class NoOpRuntimeEventTraceSink:
    def publish(self, events: tuple[AgentRuntimeEvent, ...]) -> None:
        del events


class LangfuseRuntimeEventTraceSink:
    """Metadata-only Langfuse projection isolated from product correctness."""

    def __init__(
        self,
        client: LangfuseClient,
        attribute_scope: Callable[..., ContextManager[object]],
    ) -> None:
        self._client = client
        self._attribute_scope = attribute_scope

    def publish(self, events: tuple[AgentRuntimeEvent, ...]) -> None:
        if not events:
            return
        first = events[0]
        trace_id = first.trace_id or runtime_trace_id(
            derivation_id=first.derivation_id,
            batch_id=first.batch_id,
            batch_attempt=first.batch_attempt,
        )
        with self._attribute_scope(
            session_id=runtime_session_id(first.projection_run_id),
            trace_name="memforge.agent.extraction_batch",
            version=first.extraction_contract_version,
            tags=_langfuse_tags(first),
        ):
            root = self._client.start_observation(
                name="memforge.agent.extraction_batch",
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
        session_id = runtime_session_id(first.projection_run_id) if first is not None else "none"
        trace_id = (
            first.trace_id
            or runtime_trace_id(
                derivation_id=first.derivation_id,
                batch_id=first.batch_id,
                batch_attempt=first.batch_attempt,
            )
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
        events.append(
            AgentRuntimeEvent(
                event_id=f"are-{digest}",
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
        "memforge.agent.outcome": event.outcome,
        "memforge.agent.reason_code": event.reason_code,
        "memforge.source.type": event.source_type,
        "memforge.extraction.contract.version": event.extraction_contract_version,
        "memforge.deployment.revision": event.deployment_revision,
        "gen_ai.operation.name": event.operation,
        "gen_ai.provider.name": event.provider,
        "gen_ai.request.model": event.model,
        "memforge.batch.attempt": event.batch_attempt,
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


def _langfuse_batch_metadata(
    event: AgentRuntimeEvent,
    event_count: int,
) -> dict[str, object]:
    values: dict[str, object | None] = {
        "schema_version": event.schema_version,
        "source_type": event.source_type,
        "extraction_contract_version": event.extraction_contract_version,
        "deployment_revision": event.deployment_revision,
        "provider": event.provider,
        "model": event.model,
        "batch_attempt": event.batch_attempt,
        "event_count": event_count,
    }
    return {name: value for name, value in values.items() if value is not None}


def _langfuse_tags(event: AgentRuntimeEvent) -> list[str]:
    tags = ["memforge-agent-eval", "memory-extraction"]
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
