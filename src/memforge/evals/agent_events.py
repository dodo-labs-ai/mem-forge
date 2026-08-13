"""Privacy-safe online quality events for agent evaluation.

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


AGENT_EVALUATION_EVENT_SCHEMA_VERSION = "agent-evaluation-event-v1"
AgentEvaluationOutcome = Literal["expected", "degraded", "rejected", "failed"]


@dataclass(frozen=True, slots=True)
class QualitySignal:
    """One content-free occurrence before durable lineage is attached."""

    event_type: str
    outcome: AgentEvaluationOutcome
    reason_code: str
    operation: str | None = None
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

    def __post_init__(self) -> None:
        for name in (
            "event_type",
            "reason_code",
            "operation",
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


@dataclass(frozen=True, slots=True)
class AgentEvaluationEvent:
    """One append-only, lineage-bound online evaluation event."""

    event_id: str
    event_type: str
    outcome: AgentEvaluationOutcome
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
    extraction_contract_version: str
    schema_version: str = AGENT_EVALUATION_EVENT_SCHEMA_VERSION
    deployment_revision: str | None = None
    operation: str | None = None
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


@dataclass(frozen=True, slots=True)
class AgentEvaluationEventQuery:
    """Bounded cohort filters shared by every storage adapter."""

    occurred_from: datetime
    occurred_to: datetime
    source_id: str | None = None
    source_type: str | None = None
    event_type: str | None = None
    outcome: AgentEvaluationOutcome | None = None
    reason_code: str | None = None
    model: str | None = None
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
class AgentEvaluationCohortReport:
    """Content-free aggregate with explicit per-event-type denominators."""

    total_events: int
    event_type_counts: Mapping[str, int]
    outcome_counts: Mapping[str, int]
    reason_counts: Mapping[str, int]
    rates_by_event_type: Mapping[str, Mapping[str, float]]

    def to_payload(self) -> dict[str, object]:
        return {
            "total_events": self.total_events,
            "event_type_counts": dict(self.event_type_counts),
            "outcome_counts": dict(self.outcome_counts),
            "reason_counts": dict(self.reason_counts),
            "rates_by_event_type": {
                event_type: dict(rates)
                for event_type, rates in self.rates_by_event_type.items()
            },
        }


class AgentEvaluationEventStore(Protocol):
    async def record_agent_evaluation_events(
        self,
        events: tuple[AgentEvaluationEvent, ...],
    ) -> None: ...

    async def list_agent_evaluation_events(
        self,
        query: AgentEvaluationEventQuery,
    ) -> list[AgentEvaluationEvent]: ...


class QualitySignalCollector:
    """Thread-safe request-local collection with a fixed cardinality bound."""

    def __init__(self, *, max_signals: int = 256) -> None:
        self._max_signals = max(1, int(max_signals))
        self._signals: list[QualitySignal] = []
        self._dropped_count = 0
        self._lock = Lock()

    def record(self, signal: QualitySignal) -> None:
        with self._lock:
            if len(self._signals) >= self._max_signals:
                self._dropped_count += 1
                return
            self._signals.append(signal)

    def snapshot(self) -> tuple[QualitySignal, ...]:
        with self._lock:
            values = tuple(self._signals)
            dropped_count = self._dropped_count
        if dropped_count:
            values += (
                QualitySignal(
                    event_type="agent_evaluation_signal_overflow",
                    outcome="degraded",
                    reason_code="signal_cardinality_limit",
                    rejected_count=dropped_count,
                ),
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
    extraction_contract_version: str,
    occurred_at: datetime | None = None,
    deployment_revision: str | None = None,
    observation_revision_ids: Mapping[str, str] | None = None,
) -> tuple[AgentEvaluationEvent, ...]:
    """Attach stable lineage and deterministic replay-safe event identity."""

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
            "schema_version": AGENT_EVALUATION_EVENT_SCHEMA_VERSION,
            "derivation_id": derivation_id,
            "batch_id": batch_id,
            "index": index,
            "signal": asdict(signal),
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:24]
        events.append(
            AgentEvaluationEvent(
                event_id=f"aee-{digest}",
                occurred_at=timestamp,
                source_id=source_id,
                source_type=source_type,
                doc_id=doc_id,
                source_unit_id=source_unit_id,
                target_unit_revision_id=target_unit_revision_id,
                projection_run_id=projection_run_id,
                derivation_id=derivation_id,
                batch_id=batch_id,
                extraction_contract_version=extraction_contract_version,
                deployment_revision=deployment_revision,
                **asdict(signal),
            )
        )
    return tuple(events)


def event_public_payload(event: AgentEvaluationEvent) -> Mapping[str, object]:
    """Return the fixed, content-free payload suitable for logs and exports."""

    payload = asdict(event)
    payload["occurred_at"] = event.occurred_at.isoformat()
    return payload


def summarize_agent_evaluation_events(
    events: tuple[AgentEvaluationEvent, ...] | list[AgentEvaluationEvent],
) -> AgentEvaluationCohortReport:
    """Aggregate a bounded cohort without treating degraded events as labels."""

    event_type_counts = Counter(event.event_type for event in events)
    outcome_counts = Counter(event.outcome for event in events)
    reason_counts = Counter(event.reason_code for event in events)
    outcome_by_event_type: dict[str, Counter[str]] = {}
    for event in events:
        outcome_by_event_type.setdefault(event.event_type, Counter())[event.outcome] += 1
    rates_by_event_type = {
        event_type: {
            outcome: count / event_type_counts[event_type]
            for outcome, count in sorted(outcomes.items())
        }
        for event_type, outcomes in sorted(outcome_by_event_type.items())
    }
    return AgentEvaluationCohortReport(
        total_events=len(events),
        event_type_counts=dict(sorted(event_type_counts.items())),
        outcome_counts=dict(sorted(outcome_counts.items())),
        reason_counts=dict(sorted(reason_counts.items())),
        rates_by_event_type=rates_by_event_type,
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
