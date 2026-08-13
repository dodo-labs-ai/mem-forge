from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta, timezone

import pytest

from memforge.evals.agent_events import (
    AgentEvaluationEventQuery,
    QualitySignal,
    bind_quality_signals,
    event_public_payload,
    summarize_agent_evaluation_events,
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
        extraction_contract_version="projection-extraction-v8",
        occurred_at=NOW,
    )


def test_bind_quality_signal_is_replay_stable_without_memory_id() -> None:
    signal = QualitySignal(
        event_type="structured_output_outcome",
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


def test_signal_rejects_content_bearing_or_unbounded_diagnostics() -> None:
    with pytest.raises(ValueError, match="machine-readable"):
        QualitySignal(
            event_type="structured_output_outcome",
            outcome="failed",
            reason_code="model said the private document body was invalid",
        )
    with pytest.raises(ValueError, match="SHA-256"):
        QualitySignal(
            event_type="evidence_localization_outcome",
            outcome="degraded",
            reason_code="whole_block_fallback",
            quote_hash="raw quote text",
        )


def test_public_event_payload_contains_only_fixed_contract_fields() -> None:
    [event] = _events(
        QualitySignal(
            event_type="evidence_localization_outcome",
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
            event_type="extraction_batch_outcome",
            outcome="expected",
            reason_code="candidates_extracted",
            candidate_count=2,
        ),
        QualitySignal(
            event_type="evidence_localization_outcome",
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
    await db.record_agent_evaluation_events(events)
    await db.record_agent_evaluation_events(events)

    rows = await db.list_agent_evaluation_events(
        AgentEvaluationEventQuery(
            occurred_from=NOW - timedelta(seconds=1),
            occurred_to=NOW + timedelta(seconds=1),
            source_id="src-teams",
            event_type="evidence_localization_outcome",
            reason_code="whole_block_fallback",
            limit=10,
        )
    )

    assert rows == [events[1]]


def test_cohort_report_uses_event_type_denominators() -> None:
    events = _events(
        QualitySignal(
            event_type="evidence_localization_outcome",
            outcome="expected",
            reason_code="exact_quote",
        ),
        QualitySignal(
            event_type="evidence_localization_outcome",
            outcome="degraded",
            reason_code="whole_block_fallback",
        ),
        QualitySignal(
            event_type="structured_output_outcome",
            outcome="failed",
            reason_code="provider_error",
        ),
    )

    report = summarize_agent_evaluation_events(events)

    assert report.event_type_counts == {
        "evidence_localization_outcome": 2,
        "structured_output_outcome": 1,
    }
    assert report.rates_by_event_type["evidence_localization_outcome"] == {
        "degraded": 0.5,
        "expected": 0.5,
    }
