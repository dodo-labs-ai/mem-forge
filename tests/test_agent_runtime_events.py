from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from memforge.evals.agent_evaluation import (
    AgentRuntimeEventQuery,
    QualitySignal,
    QualitySignalCollector,
    RuntimeTraceContext,
    LangfuseRuntimeEventTraceSink,
    NoOpRuntimeEventTraceSink,
    bind_quality_signals,
    current_deployment_revision,
    event_public_payload,
    runtime_event_otel_attributes,
    publish_runtime_events,
    runtime_event_trace_sink_from_env,
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
    assert len(first.trace_id or "") == 32


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


def test_default_trace_id_is_deterministic_but_not_event_identity() -> None:
    first, second = _events(
        QualitySignal("structured_output_outcome", "expected", "schema_conformant"),
        QualitySignal("extraction_batch_outcome", "expected", "candidates_extracted"),
    )

    assert first.event_id != second.event_id
    assert first.trace_id == second.trace_id
    assert first.trace_id == _events(
        QualitySignal("structured_output_outcome", "expected", "schema_conformant")
    )[0].trace_id


def test_deployment_revision_prefers_explicit_and_safely_reads_vcap(monkeypatch) -> None:
    monkeypatch.setenv("VCAP_APPLICATION", '{"application_version":"cf-version"}')
    assert current_deployment_revision() == "cf-version"

    monkeypatch.setenv("MEMFORGE_DEPLOYMENT_REVISION", "cloud-pr-338")
    assert current_deployment_revision() == "cloud-pr-338"

    monkeypatch.setenv("MEMFORGE_DEPLOYMENT_REVISION", "private revision text")
    assert current_deployment_revision() is None


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
            requesting_user_id="user-1",
            include_private=True,
            event_id=events[1].event_id,
            source_id="src-teams",
            event_name="evidence_localization_outcome",
            reason_code="whole_block_fallback",
            limit=10,
        )
    )

    assert rows == [events[1]]


@pytest.mark.asyncio
async def test_private_runtime_events_require_the_source_owner_scope(db) -> None:
    events = _events(
        QualitySignal(
            event_name="extraction_batch_outcome",
            outcome="expected",
            reason_code="zero_candidates",
        )
    )
    await db.record_agent_runtime_events(events)
    window = {
        "occurred_from": NOW - timedelta(seconds=1),
        "occurred_to": NOW + timedelta(seconds=1),
    }

    assert await db.list_agent_runtime_events(AgentRuntimeEventQuery(**window)) == []
    assert await db.list_agent_runtime_events(
        AgentRuntimeEventQuery(
            **window,
            requesting_user_id="someone-else",
            include_private=True,
        )
    ) == []
    assert await db.list_agent_runtime_events(
        AgentRuntimeEventQuery(
            **window,
            requesting_user_id="user-1",
            include_private=True,
        )
    ) == list(events)


@pytest.mark.asyncio
async def test_sqlite_runtime_event_retention_is_bounded_and_exclusive(db) -> None:
    old_event = _events(
        QualitySignal(
            event_name="extraction_batch_outcome",
            outcome="expected",
            reason_code="zero_candidates",
        )
    )[0]
    cutoff_event = bind_quality_signals(
        (QualitySignal("extraction_batch_outcome", "expected", "zero_candidates"),),
        source_id=old_event.source_id,
        source_type=old_event.source_type,
        doc_id=old_event.doc_id,
        source_unit_id=old_event.source_unit_id,
        target_unit_revision_id=old_event.target_unit_revision_id,
        projection_run_id=old_event.projection_run_id,
        derivation_id="sda-cutoff",
        batch_id="batch-cutoff",
        batch_attempt=1,
        extraction_contract_version=old_event.extraction_contract_version,
        occurred_at=NOW + timedelta(days=1),
    )[0]
    await db.record_agent_runtime_events((old_event, cutoff_event))

    assert await db.purge_agent_runtime_events(
        occurred_before=NOW + timedelta(days=1),
        limit=1,
    ) == 1
    remaining = await db.list_agent_runtime_events(
        AgentRuntimeEventQuery(
            occurred_from=NOW - timedelta(seconds=1),
            occurred_to=NOW + timedelta(days=2),
            requesting_user_id="user-1",
            include_private=True,
        )
    )
    assert remaining == [cutoff_event]


class _FakeObservation:
    def __init__(self, calls: list[tuple[str, dict]], kind: str, kwargs: dict) -> None:
        self._calls = calls
        self._kind = kind
        self._kwargs = kwargs

    def start_observation(self, **kwargs):
        self._calls.append(("child", kwargs))
        return _FakeObservation(self._calls, "child", kwargs)

    def end(self) -> None:
        self._calls.append((f"{self._kind}_end", self._kwargs))


class _FakeLangfuseClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def start_observation(self, **kwargs):
        self.calls.append(("root", kwargs))
        return _FakeObservation(self.calls, "root", kwargs)


def test_langfuse_sink_projects_allowlisted_metadata_and_event_id() -> None:
    [event] = _events(
        QualitySignal(
            event_name="evidence_localization_outcome",
            outcome="degraded",
            reason_code="whole_block_fallback",
            block_hash="b" * 64,
            quote_hash="a" * 64,
            localization_mode="block_fallback",
        )
    )
    client = _FakeLangfuseClient()

    LangfuseRuntimeEventTraceSink(client).publish((event,))

    child = next(payload for kind, payload in client.calls if kind == "child")
    assert child["metadata"]["event_id"] == event.event_id
    assert child["metadata"]["reason_code"] == "whole_block_fallback"
    assert not {
        "source_id",
        "doc_id",
        "source_unit_id",
        "prompt_hash",
        "block_hash",
        "quote_hash",
        "observation_id",
    }.intersection(child["metadata"])


def test_runtime_trace_sink_failure_is_best_effort(caplog) -> None:
    class FailingSink:
        def publish(self, events):
            raise RuntimeError("backend unavailable")

    publish_runtime_events(FailingSink(), _events(QualitySignal("batch", "failed", "provider_error")))
    NoOpRuntimeEventTraceSink().publish(())
    assert "trace projection failed" in caplog.text


def test_runtime_trace_sink_is_disabled_without_importing_langfuse(monkeypatch) -> None:
    def unexpected_import(name: str):
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.delenv("MEMFORGE_LANGFUSE_ENABLED", raising=False)
    monkeypatch.setattr("memforge.evals.agent_evaluation.importlib.import_module", unexpected_import)

    assert isinstance(runtime_event_trace_sink_from_env(), NoOpRuntimeEventTraceSink)


def test_runtime_trace_sink_uses_langfuse_only_when_enabled(monkeypatch) -> None:
    client = _FakeLangfuseClient()
    monkeypatch.setenv("MEMFORGE_LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setattr(
        "memforge.evals.agent_evaluation.importlib.import_module",
        lambda name: SimpleNamespace(get_client=lambda: client) if name == "langfuse" else None,
    )

    sink = runtime_event_trace_sink_from_env()

    assert isinstance(sink, LangfuseRuntimeEventTraceSink)
    sink.publish(_events(QualitySignal("batch", "expected", "candidates_extracted")))
    assert any(kind == "root" for kind, _ in client.calls)


def test_runtime_trace_sink_initialization_failure_is_nonfatal(monkeypatch, caplog) -> None:
    monkeypatch.setenv("MEMFORGE_LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setattr(
        "memforge.evals.agent_evaluation.importlib.import_module",
        lambda _name: SimpleNamespace(get_client=lambda: (_ for _ in ()).throw(RuntimeError("bad config"))),
    )

    assert isinstance(runtime_event_trace_sink_from_env(), NoOpRuntimeEventTraceSink)
    assert "client initialization failed" in caplog.text


def test_runtime_trace_sink_requires_credentials_when_enabled(monkeypatch, caplog) -> None:
    monkeypatch.setenv("MEMFORGE_LANGFUSE_ENABLED", "true")
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    assert isinstance(runtime_event_trace_sink_from_env(), NoOpRuntimeEventTraceSink)
    assert "credentials are incomplete" in caplog.text


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
