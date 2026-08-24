from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from memforge.evals.agent_evaluation import (
    AgentAssessmentQuery,
    AgentRuntimeEventQuery,
    QualitySignal,
    QualitySignalCollector,
    RuntimeTraceContext,
    LangfuseRuntimeEventTraceSink,
    NoOpRuntimeEventTraceSink,
    assessment_sink_for_runtime_sink,
    bind_source_lifecycle_outcome,
    bind_quality_signals,
    build_workspace_online_evaluation_view,
    current_deployment_revision,
    evaluate_runtime_events,
    event_public_payload,
    runtime_event_otel_attributes,
    publish_runtime_events,
    runtime_event_trace_sink_from_env,
    runtime_session_id,
    runtime_trace_id,
    summarize_agent_assessments,
    summarize_agent_runtime_events,
)
from memforge.storage.database import Database, MIGRATIONS


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


def _events(
    *signals: QualitySignal,
    projection_run_id: str = "spr-current",
    derivation_id: str = "sda-current",
    batch_id: str = "batch-1",
    batch_attempt: int = 1,
):
    return bind_quality_signals(
        tuple(signals),
        source_id="src-teams",
        source_type="teams",
        doc_id="doc-window",
        source_unit_id="unit-window",
        target_unit_revision_id="sur-current",
        projection_run_id=projection_run_id,
        derivation_id=derivation_id,
        batch_id=batch_id,
        batch_attempt=batch_attempt,
        extraction_contract_version="projection-extraction-v8",
        occurred_at=NOW,
    )


def _lifecycle_bundle(
    *,
    execution_owner_id: str = "sync-run-7:lease-1",
    outcome: str = "expected",
    reason_code: str = "lifecycle_plan_applied",
):
    return bind_source_lifecycle_outcome(
        source_id="src-teams",
        source_type="teams",
        doc_id="doc-window",
        source_unit_id="unit-window",
        base_unit_revision_id="sur-before",
        target_unit_revision_id="sur-current",
        projection_run_id="spr-current",
        operation_input_hash="a" * 64,
        execution_owner_id=execution_owner_id,
        outcome=outcome,
        reason_code=reason_code,
        attempt_count=3,
        duration_ms=250,
        incumbent_count=2,
        relation_pair_count=4,
        mutation_count=2,
        review_count=0,
        model_call_count=2,
        occurred_at=NOW,
        deployment_revision="cloud-pr-258",
    )


def test_source_lifecycle_terminal_identity_separates_operation_execution_and_event() -> None:
    first = _lifecycle_bundle()
    replayed = _lifecycle_bundle()
    recovered = _lifecycle_bundle(execution_owner_id="sync-run-7:lease-2")

    assert replayed == first
    assert recovered.event.operation_id == first.event.operation_id
    assert recovered.event.execution_id != first.event.execution_id
    assert recovered.event.event_id != first.event.event_id
    assert first.event.event_name == "source_unit_lifecycle_outcome"
    assert first.event.payload_hash
    assert first.assessment.criterion == "source_unit_lifecycle_completion"
    assert first.assessment.label == "pass"


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
    assert first.operation_id
    assert first.execution_id
    assert first.contract_version == "projection-extraction-v8"
    assert first.payload_hash
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
async def test_sqlite_event_store_rejects_conflicting_terminal_payload(db) -> None:
    bundle = _lifecycle_bundle()
    await db.record_agent_runtime_events(bundle.events)
    conflicting = replace(
        bundle.event,
        outcome="failed",
        reason_code="lifecycle_commit_failed",
        payload_hash="b" * 64,
    )

    with pytest.raises(ValueError, match="conflicting agent runtime event payload"):
        await db.record_agent_runtime_events((conflicting,))


@pytest.mark.asyncio
async def test_sqlite_structured_attempt_diagnostics_round_trip(db) -> None:
    [event] = _events(
        QualitySignal(
            event_name="structured_llm_attempt_outcome",
            outcome="rejected",
            reason_code="schema_validation_failed",
            operation="memory_extraction",
            provider="sap",
            model="sap/anthropic--claude-4.6-sonnet",
            attempt_index=1,
            structured_mode="native_schema",
            schema_transport="json_schema_response_format",
            requested_max_tokens=32_768,
            terminal_category="invalid_response",
            error_code="ValidationError",
            finish_reason="max_tokens",
            stop_reason="max_tokens",
            provider_request_id="msg-provider-123",
            prompt_tokens=1_864,
            completion_tokens=32_768,
            total_tokens=34_632,
            response_chars=98_304,
            response_hash="c" * 64,
            validation_location="$",
            validation_rule="json_invalid",
            json_error_line=1,
            json_error_column=98_305,
        )
    )

    await db.record_agent_runtime_events((event,))
    rows = await db.list_agent_runtime_events(
        AgentRuntimeEventQuery(
            occurred_from=NOW - timedelta(seconds=1),
            occurred_to=NOW + timedelta(seconds=1),
            requesting_user_id="user-1",
            include_private=True,
            event_id=event.event_id,
        )
    )

    assert rows == [event]


@pytest.mark.asyncio
async def test_sqlite_v3_migration_preserves_v2_event_and_assessment(db) -> None:
    await db.db.execute("DROP TABLE agent_assessments")
    await db.db.execute("DROP TABLE agent_runtime_events")
    for version, _description, statements in MIGRATIONS:
        if version not in {76, 78}:
            continue
        for statement in statements:
            await db.db.execute(statement)
    await db.db.execute(
        """INSERT INTO agent_runtime_events (
               event_id, schema_version, event_name, outcome, reason_code,
               occurred_at, source_id, source_type, doc_id, source_unit_id,
               target_unit_revision_id, projection_run_id, derivation_id,
               batch_id, batch_attempt, extraction_contract_version
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "are-legacy",
            "agent-runtime-event-v2",
            "extraction_batch_outcome",
            "expected",
            "zero_candidates",
            NOW.isoformat(),
            "src-teams",
            "teams",
            "doc-window",
            "unit-window",
            "sur-current",
            "spr-current",
            "sda-legacy",
            "batch-legacy",
            1,
            "projection-extraction-v8",
        ),
    )
    await db.db.execute(
        """INSERT INTO agent_assessments (
               assessment_id, schema_version, target_event_id, criterion,
               status, label, reason_code, annotator_kind, evaluator_name,
               evaluator_version, created_at
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "aas-legacy",
            "agent-assessment-v1",
            "are-legacy",
            "extraction_completion",
            "completed",
            "pass",
            "zero_candidates",
            "code",
            "memforge.deterministic.runtime_contract",
            "1",
            NOW.isoformat(),
        ),
    )
    await db.db.execute("DELETE FROM schema_migrations WHERE version IN (79, 82, 85)")
    await db.db.commit()

    await db._run_migrations()

    [event] = await db.list_agent_runtime_events(
        AgentRuntimeEventQuery(
            occurred_from=NOW - timedelta(seconds=1),
            occurred_to=NOW + timedelta(seconds=1),
            requesting_user_id="user-1",
            include_private=True,
            event_id="are-legacy",
        )
    )
    [assessment] = await db.list_agent_assessments(
        AgentAssessmentQuery(
            occurred_from=NOW - timedelta(seconds=1),
            occurred_to=NOW + timedelta(seconds=1),
            requesting_user_id="user-1",
            include_private=True,
            assessment_id="aas-legacy",
        )
    )
    assert event.schema_version == "agent-runtime-event-v2"
    assert event.operation_id is None
    assert event.derivation_id == "sda-legacy"
    assert assessment.target_event_id == event.event_id


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
        self.id = "root-span-id"

    def end(self) -> None:
        self._calls.append((f"{self._kind}_end", self._kwargs))


class _FakeLangfuseClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def start_observation(self, **kwargs):
        self.calls.append(("root", kwargs))
        return _FakeObservation(self.calls, "root", kwargs)

    def create_event(self, **kwargs):
        self.calls.append(("event", kwargs))

    def create_score(self, **kwargs):
        self.calls.append(("score", kwargs))

    def shutdown(self) -> None:
        self.calls.append(("shutdown", {}))


def _langfuse_sink(
    client: _FakeLangfuseClient,
    attribute_scopes: list[dict[str, object]] | None = None,
) -> LangfuseRuntimeEventTraceSink:
    scopes = attribute_scopes if attribute_scopes is not None else []

    @contextmanager
    def attribute_scope(**kwargs):
        scopes.append(kwargs)
        yield

    return LangfuseRuntimeEventTraceSink(client, attribute_scope)


@pytest.fixture(autouse=True)
def _reset_runtime_trace_sink_cache():
    runtime_event_trace_sink_from_env.cache_clear()
    yield
    runtime_event_trace_sink_from_env.cache_clear()


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

    _langfuse_sink(client).publish((event,))

    projected_event = next(payload for kind, payload in client.calls if kind == "event")
    assert projected_event["metadata"]["event_id"] == event.event_id
    assert projected_event["metadata"]["reason_code"] == "whole_block_fallback"
    assert projected_event["trace_context"] == {
        "trace_id": event.trace_id,
        "parent_span_id": "root-span-id",
    }
    assert not {
        "source_id",
        "doc_id",
        "source_unit_id",
        "prompt_hash",
        "block_hash",
        "quote_hash",
        "observation_id",
    }.intersection(projected_event["metadata"])


def test_deterministic_evaluator_only_judges_supported_runtime_contracts() -> None:
    events = _events(
        QualitySignal("structured_output_outcome", "expected", "schema_conformant"),
        QualitySignal("structured_llm_attempt_outcome", "rejected", "schema_validation_failed"),
        QualitySignal("evidence_admission_outcome", "rejected", "unknown_evidence_block_id"),
        QualitySignal("evidence_localization_outcome", "degraded", "whole_block_fallback"),
        QualitySignal("extraction_batch_outcome", "expected", "zero_candidates"),
    )

    assessments = evaluate_runtime_events(events)

    assert [(item.criterion, item.label, item.reason_code) for item in assessments] == [
        ("structured_output_contract", "pass", "schema_conformant"),
        ("evidence_reference_validity", "fail", "unknown_evidence_block_id"),
        ("evidence_localization", "needs_review", "whole_block_fallback"),
    ]
    assert evaluate_runtime_events(events) == assessments


def test_deterministic_evaluator_reproduces_persisted_lifecycle_assessment() -> None:
    bundle = _lifecycle_bundle()

    assert evaluate_runtime_events((bundle.event,)) == bundle.assessments


def test_workspace_evaluation_distinguishes_coverage_gap_from_no_recent_data() -> None:
    [pending_event] = _events(
        QualitySignal("structured_output_outcome", "expected", "schema_conformant")
    )

    view = build_workspace_online_evaluation_view(
        [
            {
                "id": "src-teams",
                "name": "Teams",
                "type": "teams",
                "status": "active",
            },
            {
                "id": "src-empty",
                "name": "Empty Source",
                "type": "jira",
                "status": "active",
            },
        ],
        [pending_event],
        [],
    )

    assert [(source["source_id"], source["evaluation_status"]) for source in view["sources"]] == [
        ("src-teams", "coverage_gap"),
        ("src-empty", "no_data"),
    ]
    assert view["summary"]["affected_source_count"] == 1
    assert view["coverage"]["pending_occurrences"] == 1


def test_assessment_summary_preserves_coalesced_occurrence_denominator() -> None:
    [event] = _events(
        QualitySignal(
            "evidence_admission_outcome",
            "rejected",
            "unknown_evidence_block_id",
            occurrence_count=4,
        )
    )
    assessments = evaluate_runtime_events((event,))

    assert assessments[0].occurrence_count == 4
    assert summarize_agent_assessments(assessments) == {
        "total_assessments": 4,
        "label_counts": {"fail": 4},
        "criterion_counts": {"evidence_reference_validity": 4},
        "status_counts": {"completed": 4},
    }


def test_langfuse_assessment_sink_projects_categorical_score_to_event_trace() -> None:
    [event] = _events(
        QualitySignal("evidence_admission_outcome", "rejected", "unknown_evidence_block_id")
    )
    [assessment] = evaluate_runtime_events((event,))
    client = _FakeLangfuseClient()
    sink = _langfuse_sink(client)

    assessment_sink_for_runtime_sink(sink).publish((assessment,), (event,))

    score = next(payload for kind, payload in client.calls if kind == "score")
    assert score["score_id"] == assessment.assessment_id
    assert score["trace_id"] == event.trace_id
    assert score["name"] == "memforge.evidence_reference_validity"
    assert score["value"] == "fail"
    assert score["data_type"] == "CATEGORICAL"
    assert score["metadata"]["target_event_id"] == event.event_id


@pytest.mark.asyncio
async def test_sqlite_assessments_are_idempotent_visible_and_cascade_with_events(db) -> None:
    [event] = _events(
        QualitySignal("evidence_admission_outcome", "rejected", "unknown_evidence_block_id")
    )
    assessments = evaluate_runtime_events((event,))
    await db.record_agent_runtime_events((event,))
    await db.record_agent_assessments(assessments)
    await db.record_agent_assessments(assessments)

    query = AgentAssessmentQuery(
        occurred_from=NOW - timedelta(seconds=1),
        occurred_to=NOW + timedelta(seconds=1),
        requesting_user_id="user-1",
        include_private=True,
        source_id="src-teams",
        label="fail",
    )
    assert await db.list_agent_assessments(query) == list(assessments)
    assert await db.list_agent_assessments(
        AgentAssessmentQuery(
            occurred_from=query.occurred_from,
            occurred_to=query.occurred_to,
        )
    ) == []

    assert await db.purge_agent_runtime_events(
        occurred_before=NOW + timedelta(seconds=1),
        limit=10,
    ) == 1
    assert await db.list_agent_assessments(query) == []


@pytest.mark.asyncio
async def test_sqlite_assessment_batch_rolls_back_on_immutable_collision(db) -> None:
    [event] = _events(
        QualitySignal("evidence_admission_outcome", "rejected", "unknown_evidence_block_id")
    )
    [existing] = evaluate_runtime_events((event,))
    await db.record_agent_runtime_events((event,))
    await db.record_agent_assessments((existing,))

    new_assessment = replace(
        existing,
        assessment_id="aas-new-before-conflict",
    )
    conflicting = replace(
        existing,
        label="pass",
        reason_code="conflicting_immutable_payload",
    )

    with pytest.raises(ValueError, match="conflicting immutable agent assessment"):
        await db.record_agent_assessments((new_assessment, conflicting))

    assert await db.get_agent_assessment(new_assessment.assessment_id) is None


def test_langfuse_sink_omits_durable_only_structured_attempt_identifiers() -> None:
    [event] = _events(
        QualitySignal(
            event_name="structured_llm_attempt_outcome",
            outcome="rejected",
            reason_code="schema_validation_failed",
            operation="memory_extraction",
            attempt_index=1,
            structured_mode="native_schema",
            schema_transport="json_schema_response_format",
            requested_max_tokens=32_768,
            terminal_category="invalid_response",
            error_code="ValidationError",
            provider_request_id="sap-request-456",
            response_hash="c" * 64,
            response_chars=98_304,
        )
    )
    client = _FakeLangfuseClient()

    _langfuse_sink(client).publish((event,))

    projected_event = next(payload for kind, payload in client.calls if kind == "event")
    assert projected_event["metadata"]["attempt_index"] == 1
    assert projected_event["metadata"]["requested_max_tokens"] == 32_768
    assert projected_event["metadata"]["response_chars"] == 98_304
    assert "provider_request_id" not in projected_event["metadata"]
    assert "response_hash" not in projected_event["metadata"]


def test_runtime_trace_sink_failure_is_best_effort(caplog) -> None:
    class FailingSink:
        def publish(self, events):
            raise RuntimeError("backend unavailable")

    publish_runtime_events(FailingSink(), _events(QualitySignal("batch", "failed", "provider_error")))
    NoOpRuntimeEventTraceSink().publish(())
    assert "trace projection failed" in caplog.text
    assert "session_id=mfs1-83cf15f58a5beafef62844440de30fbc" in caplog.text
    assert f"trace_id={_events(QualitySignal('batch', 'failed', 'provider_error'))[0].trace_id}" in caplog.text
    assert "event_count=1" in caplog.text


def test_langfuse_sink_groups_fallback_derivations_in_one_projection_session() -> None:
    [diff_event] = _events(
        QualitySignal("extraction_batch_outcome", "failed", "diff_guided_extraction_error"),
        derivation_id="sdrv-diff",
        batch_id="dbatch-diff",
    )
    [structural_event] = _events(
        QualitySignal("extraction_batch_outcome", "expected", "candidates_extracted"),
        derivation_id="sdrv-structural",
        batch_id="sbatch-structural",
    )
    client = _FakeLangfuseClient()
    attribute_scopes: list[dict[str, object]] = []
    sink = _langfuse_sink(client, attribute_scopes)

    sink.publish((diff_event,))
    sink.publish((structural_event,))

    assert [scope["session_id"] for scope in attribute_scopes] == [
        "mfs1-83cf15f58a5beafef62844440de30fbc",
        "mfs1-83cf15f58a5beafef62844440de30fbc",
    ]
    assert all(scope["trace_name"] == "memforge.agent.extraction_batch" for scope in attribute_scopes)
    assert all(scope["version"] == "projection-extraction-v8" for scope in attribute_scopes)
    assert all(
        scope["tags"] == ["memforge-agent-eval", "memory-extraction", "source-type:teams"] for scope in attribute_scopes
    )
    root_trace_ids = [payload["trace_context"]["trace_id"] for kind, payload in client.calls if kind == "root"]
    assert root_trace_ids == [diff_event.trace_id, structural_event.trace_id]
    assert diff_event.trace_id != structural_event.trace_id


def test_langfuse_sink_groups_lifecycle_execution_under_operation() -> None:
    bundle = _lifecycle_bundle()
    client = _FakeLangfuseClient()
    attribute_scopes: list[dict[str, object]] = []

    _langfuse_sink(client, attribute_scopes).publish(bundle.events)

    [scope] = attribute_scopes
    assert scope["session_id"].startswith("mfo1-")
    assert scope["trace_name"] == "memforge.agent.reconcile_source_unit"
    assert scope["version"] == "source-unit-lifecycle-v1"
    assert scope["tags"] == [
        "memforge-agent-eval",
        "source-unit-lifecycle",
        "source-type:teams",
    ]
    root = next(payload for kind, payload in client.calls if kind == "root")
    assert root["name"] == "memforge.agent.reconcile_source_unit"
    assert root["trace_context"]["trace_id"] == bundle.event.trace_id


def test_runtime_session_id_is_bounded_versioned_and_projection_specific() -> None:
    assert runtime_session_id("spr-current") == "mfs1-83cf15f58a5beafef62844440de30fbc"
    assert runtime_session_id("spr-next") == "mfs1-d3f07dda65d93f39803da948c533759b"


def test_runtime_trace_id_guards_w3c_all_zero_value(monkeypatch) -> None:
    class ZeroDigest:
        def hexdigest(self) -> str:
            return "0" * 64

    monkeypatch.setattr(
        "memforge.evals.agent_evaluation.hashlib.sha256",
        lambda _value: ZeroDigest(),
    )

    assert (
        runtime_trace_id(
            derivation_id="sdrv-zero",
            batch_id="batch-zero",
            batch_attempt=1,
        )
        == "0" * 31 + "1"
    )


def test_runtime_trace_sink_is_disabled_without_importing_langfuse(monkeypatch) -> None:
    def unexpected_import(name: str):
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.delenv("MEMFORGE_LANGFUSE_ENABLED", raising=False)
    monkeypatch.setattr("memforge.evals.agent_evaluation.importlib.import_module", unexpected_import)

    assert isinstance(runtime_event_trace_sink_from_env(), NoOpRuntimeEventTraceSink)


def test_runtime_trace_sink_uses_langfuse_only_when_enabled(monkeypatch) -> None:
    client = _FakeLangfuseClient()
    tracer_provider = object()
    span_filter = object()
    registered: list[object] = []
    monkeypatch.setenv("MEMFORGE_LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://langfuse.example.invalid")

    def fake_import(name: str):
        if name == "langfuse":
            return SimpleNamespace(
                propagate_attributes=lambda **_kwargs: nullcontext(),
                Langfuse=lambda **kwargs: (
                    client
                    if kwargs
                    == {
                        "tracer_provider": tracer_provider,
                        "should_export_span": span_filter,
                        "release": None,
                    }
                    else (_ for _ in ()).throw(AssertionError(kwargs))
                )
            )
        if name == "langfuse.span_filter":
            return SimpleNamespace(is_langfuse_span=span_filter)
        if name == "opentelemetry.sdk.trace":
            return SimpleNamespace(TracerProvider=lambda: tracer_provider)
        raise AssertionError(name)

    monkeypatch.setattr(
        "memforge.evals.agent_evaluation.importlib.import_module",
        fake_import,
    )
    monkeypatch.setattr(
        "memforge.evals.agent_evaluation.atexit.register",
        registered.append,
    )

    sink = runtime_event_trace_sink_from_env()

    assert isinstance(sink, LangfuseRuntimeEventTraceSink)
    assert runtime_event_trace_sink_from_env() is sink
    assert registered == [client.shutdown]
    sink.publish(_events(QualitySignal("batch", "expected", "candidates_extracted")))
    assert any(kind == "root" for kind, _ in client.calls)


def test_runtime_trace_sink_initialization_failure_is_nonfatal(monkeypatch, caplog) -> None:
    monkeypatch.setenv("MEMFORGE_LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://langfuse.example.invalid")
    monkeypatch.setattr(
        "memforge.evals.agent_evaluation.importlib.import_module",
        lambda _name: (_ for _ in ()).throw(RuntimeError("bad config")),
    )

    assert isinstance(runtime_event_trace_sink_from_env(), NoOpRuntimeEventTraceSink)
    assert "client initialization failed" in caplog.text


def test_runtime_trace_sink_requires_credentials_when_enabled(monkeypatch, caplog) -> None:
    monkeypatch.setenv("MEMFORGE_LANGFUSE_ENABLED", "true")
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)

    assert isinstance(runtime_event_trace_sink_from_env(), NoOpRuntimeEventTraceSink)
    assert "configuration is incomplete" in caplog.text


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
    assessment_sink_for_runtime_sink,
    evaluate_runtime_events,
