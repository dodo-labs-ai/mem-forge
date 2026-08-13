"""Privacy-safe online quality events for agent runtime.

Producers emit small, content-free ``QualitySignal`` values.  The Source
Derivation seam binds them to stable source lineage before persistence, so a
signal remains actionable even when no Memory was created.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from threading import Lock
from typing import Iterator, Literal, Mapping, Protocol


AGENT_RUNTIME_EVENT_SCHEMA_VERSION = "agent-runtime-event-v1"
AgentRuntimeOutcome = Literal["expected", "degraded", "rejected", "failed"]


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
    structured_mode: str | None = None
    terminal_category: str | None = None
    error_code: str | None = None
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
            "terminal_category",
            "error_code",
        ):
            _require_safe_label(name, getattr(self, name))
        _require_model(self.model)
        for name in (
            "prompt_hash",
            "candidate_hash",
            "block_hash",
            "quote_hash",
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
    structured_mode: str | None = None
    terminal_category: str | None = None
    error_code: str | None = None
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
                trace_id=trace_context.trace_id if trace_context else None,
                span_id=trace_context.span_id if trace_context else None,
                trace_flags=trace_context.trace_flags if trace_context else None,
                **asdict(signal),
            )
        )
    return tuple(events)


def current_trace_context() -> RuntimeTraceContext | None:
    """Read the active OTel span when the optional API is installed."""

    try:
        from opentelemetry import trace  # type: ignore[import-not-found]
    except ImportError:
        return None
    span_context = trace.get_current_span().get_span_context()
    if not span_context.is_valid:
        return None
    return RuntimeTraceContext(
        trace_id=f"{span_context.trace_id:032x}",
        span_id=f"{span_context.span_id:016x}",
        trace_flags=int(span_context.trace_flags),
    )


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
