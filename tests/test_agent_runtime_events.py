from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta, timezone

import pytest

from memforge.evals.agent_evaluation import (
    AgentRuntimeEventQuery,
    QualitySignal,
    QualitySignalCollector,
    RuntimeTraceContext,
    bind_quality_signals,
    event_public_payload,
    runtime_event_otel_attributes,
    summarize_agent_runtime_events,
)
from memforge.storage.database import Database


NOW = datetime(2026, 8, 13, 6, 0, tzinfo=timezone.utc)


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "agent-evaluation.db"))
    await database.connect()
    await database.db.execute(
        """INSERT INTO sources (
               id, type, name, config, owner_user_id, access_policy
           ) VALUES (?, ?, ?, ?, ?, ?)""",
        ("src-teams", "teams", "Teams", "{}", "user-1", "private"),
    )
    await database.db.commit()
    yield database
    await database.close()


def _events(*signals: QualitySignal):
    return bind_quality_signals(
        tuple(signals),
        source_id="src-teams",
        source_type="teams",
        doc_id="doc-window",
        source_unit_id="unit-window",
        target_unit_revision_id="sur-current",
        projection_run_id="spr-current",
        derivation_id="sda-current",
        batch_id="batch-1",
        batch_attempt=1,
        extraction_contract_version="projection-extraction-v8",
        occurred_at=NOW,
    )


def test_bind_quality_signal_is_replay_stable_without_memory_id() -> None:
    signal = QualitySignal(
        event_name="structured_output_outcome",
        outcome="rejected",
        reason_code="schema_validation_failed",
        operation="memory_extraction",
        terminal_category="invalid_response",
        error_code="ValidationError",
        attempt_count=3,
        retry_count=1,
        fallback_count=1,
        structured_mode="json_text",
    )

    [first] = _events(signal)
    [replayed] = _events(signal)

    assert first.event_id == replayed.event_id
    assert first.source_id == "src-teams"
    assert first.derivation_id == "sda-current"
    assert first.batch_id == "batch-1"
    assert not hasattr(first, "memory_content")


def test_retry_attempt_and_trace_correlation_do_not_alias_runtime_identity() -> None:
    signal = QualitySignal(
        event_name="structured_output_outcome",
        outcome="failed",
        reason_code="provider_error",
    )
    [first] = _events(signal)
    [retry] = bind_quality_signals(
        (signal,),
        source_id=first.source_id,
        source_type=first.source_type,
        doc_id=first.doc_id,
        source_unit_id=first.source_unit_id,
        target_unit_revision_id=first.target_unit_revision_id,
        projection_run_id=first.projection_run_id,
        derivation_id=first.derivation_id,
        batch_id=first.batch_id,
        batch_attempt=2,
        extraction_contract_version=first.extraction_contract_version,
        occurred_at=NOW,
        trace_context=RuntimeTraceContext("1" * 32, "2" * 16, 1),
    )

    assert retry.event_id != first.event_id
    assert retry.trace_id == "1" * 32
    assert retry.span_id == "2" * 16
    assert "trace_id" not in runtime_event_otel_attributes(retry)


def test_bounded_collector_coalesces_overflow_without_losing_denominator() -> None:
    collector = QualitySignalCollector(max_signals=1)
    signal = QualitySignal(
        event_name="evidence_admission_outcome",
        outcome="rejected",
        reason_code="unknown_evidence_block_id",
    )
    for _ in range(4):
        collector.record(signal)

    events = _events(*collector.snapshot())
    report = summarize_agent_runtime_events(events)

    assert len(events) == 2
    assert report.total_events == 4
    assert report.reason_counts == {"unknown_evidence_block_id": 4}


def test_signal_rejects_content_bearing_or_unbounded_diagnostics() -> None:
    with pytest.raises(ValueError, match="machine-readable"):
        QualitySignal(
            event_name="structured_output_outcome",
            outcome="failed",
            reason_code="model said the private document body was invalid",
        )
    with pytest.raises(ValueError, match="SHA-256"):
        QualitySignal(
            event_name="evidence_localization_outcome",
            outcome="degraded",
            reason_code="whole_block_fallback",
            quote_hash="raw quote text",
        )


def test_public_event_payload_contains_only_fixed_contract_fields() -> None:
    [event] = _events(
        QualitySignal(
            event_name="evidence_localization_outcome",
            outcome="degraded",
            reason_code="whole_block_fallback",
            quote_hash="a" * 64,
            block_hash="b" * 64,
            quote_chars=42,
            range_start=10,
            range_end=80,
            localization_mode="block_fallback",
        )
    )

    payload = event_public_payload(event)
    expected = asdict(event)
    expected["occurred_at"] = NOW.isoformat()
    assert payload == expected
    assert not {"prompt", "quote", "source_content", "memory_content", "provider_error_body"}.intersection(payload)


@pytest.mark.asyncio
async def test_sqlite_event_store_is_idempotent_and_supports_bounded_cohorts(db) -> None:
    events = _events(
        QualitySignal(
            event_name="extraction_batch_outcome",
            outcome="expected",
            reason_code="candidates_extracted",
            candidate_count=2,
        ),
        QualitySignal(
            event_name="evidence_localization_outcome",
            outcome="degraded",
            reason_code="whole_block_fallback",
            localization_mode="block_fallback",
            block_hash="b" * 64,
            quote_hash="a" * 64,
            quote_chars=42,
            range_start=10,
            range_end=80,
        ),
    )
    await db.record_agent_runtime_events(events)
    await db.record_agent_runtime_events(events)

    rows = await db.list_agent_runtime_events(
        AgentRuntimeEventQuery(
            occurred_from=NOW - timedelta(seconds=1),
            occurred_to=NOW + timedelta(seconds=1),
            source_id="src-teams",
            event_name="evidence_localization_outcome",
            reason_code="whole_block_fallback",
            limit=10,
        )
    )

    assert rows == [events[1]]


def test_cohort_report_uses_event_name_denominators() -> None:
    events = _events(
        QualitySignal(
            event_name="evidence_localization_outcome",
            outcome="expected",
            reason_code="exact_quote",
        ),
        QualitySignal(
            event_name="evidence_localization_outcome",
            outcome="degraded",
            reason_code="whole_block_fallback",
        ),
        QualitySignal(
            event_name="structured_output_outcome",
            outcome="failed",
            reason_code="provider_error",
        ),
    )

    report = summarize_agent_runtime_events(events)

    assert report.event_name_counts == {
        "evidence_localization_outcome": 2,
        "structured_output_outcome": 1,
    }
    assert report.rates_by_event_name["evidence_localization_outcome"] == {
        "degraded": 0.5,
        "expected": 0.5,
    }
