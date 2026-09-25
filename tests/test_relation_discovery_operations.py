"""Operator paths for cross-document discovery: exhausted work, re-runs and Review conversion."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from memforge.config import AppConfig
from memforge.evals.cross_document_relation_cases import RELATION_CASE_POLICY_VERSION
from memforge.memory.cross_document_relation import CrossDocumentRelationLabel, pair_key
from memforge.memory.cross_source_conflict_reviews import CROSS_SOURCE_CONFLICT_REVIEW_KIND
from memforge.memory.cross_source_review_conversion import (
    CONVERSION_APPLIED_EVENT,
    CrossSourceReviewConversionConflict,
    CrossSourceReviewConversionReceipt,
    apply_conversion,
    build_conversion_plan,
    delete_converted_reviews,
)
from memforge.memory.evidence import EvidenceContentProvenance, EvidenceUnit
from memforge.memory.relation_discovery import DEFAULT_RELATION_DISCOVERY_BUDGET
from memforge.memory.relation_discovery_contract import (
    RELATION_DISCOVERY_RERUN_EVENT,
    RelationDiscoveryRequest,
    RelationDiscoveryWorkSelection,
    RelationDiscoveryWorkState,
)
from memforge.models import Memory, MemoryReview, ReviewStatus, content_hash
from memforge.storage.adapters.context import LOCAL_DEV_USER_ID
from memforge.storage.database import Database

MAX_ATTEMPTS = DEFAULT_RELATION_DISCOVERY_BUDGET.max_attempts
REVIEWED_AT = datetime(2026, 7, 20, 8, 0, tzinfo=timezone.utc)
OPERATOR = "operator@example.test"
REVIEWER = "reviewer@example.test"
UPDATED = "2026-07-23T00:00:00+00:00"


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "relation-operations.db"))
    await database.connect()
    await database.upsert_source(
        id="src-confluence",
        type="confluence",
        name="Confluence",
        config_json="{}",
        access_policy="workspace",
        owner_user_id=LOCAL_DEV_USER_ID,
    )
    await database.db.execute(
        """INSERT INTO lifecycle_plans (
               id, reconciliation_scope_id, source_id, source_unit_id,
               target_unit_revision_id, status, payload_json, payload_hash, created_at
           ) VALUES ('plan-1', 'scope-1', 'src-confluence', 'unit-1', 'unitrev-1',
                     'applied', '{}', 'plan-hash', '2026-07-22T00:00:00+00:00')"""
    )
    await database.db.commit()
    yield database
    await database.close()


def _client(db: Database, tmp_path, *, operator: bool = True) -> TestClient:
    from memforge.server.admin_api import create_admin_app

    config = AppConfig(base_dir=tmp_path / "memforge")
    config.sync.worker_enabled = False
    return TestClient(
        create_admin_app(
            db=db,
            config=config,
            principal_resolver=lambda _request: OPERATOR,
            workspace_role_resolver=lambda _request: "workspace_admin",
            maintenance_operator_resolver=lambda _request: operator,
        )
    )


async def _memory(db: Database, memory_id: str, content: str | None = None) -> Memory:
    text = content or f"Statement of {memory_id}."
    memory = Memory(
        id=memory_id,
        memory_type="decision",
        content=text,
        content_hash=content_hash(text),
        created_at=REVIEWED_AT,
        updated_at=REVIEWED_AT,
    )
    await db.insert_memory(memory)
    return memory


async def _change(db: Database, memory: Memory) -> None:
    changed = f"{memory.content} Revised."
    await db.db.execute(
        "UPDATE memories SET content = ?, content_hash = ?, updated_at = ? WHERE id = ?",
        (changed, content_hash(changed), "2026-07-21T00:00:00+00:00", memory.id),
    )
    await db.db.commit()


async def _work(
    db: Database,
    memory: Memory,
    *,
    status: str,
    attempts: int,
    error_code: str | None = None,
    classifier_version: str | None = None,
    updated_at: str = UPDATED,
) -> str:
    work_id = f"work-{memory.id}"
    await db._enqueue_relation_discovery_work_unlocked(  # noqa: SLF001
        "plan-1",
        RelationDiscoveryRequest(
            id=work_id,
            memory_id=memory.id,
            expected_content_hash=memory.content_hash,
            source_id="src-confluence",
            source_unit_id="unit-1",
            source_unit_revision_id="unitrev-1",
            doc_id="doc-1",
            actor_user_id=None,
        ),
        now="2026-07-22T00:00:00+00:00",
    )
    await db.db.execute(
        """UPDATE relation_discovery_work
              SET status = ?, attempts = ?, error = ?, error_code = ?,
                  classifier_version = ?, updated_at = ?
            WHERE id = ?""",
        (
            status,
            attempts,
            f"{error_code}: failed" if error_code else None,
            error_code,
            classifier_version,
            updated_at,
            work_id,
        ),
    )
    await db.db.commit()
    return work_id


async def _review(
    db: Database,
    review_id: str,
    status: ReviewStatus,
    *,
    challenger: Memory,
    incumbent: Memory,
) -> None:
    await db.insert_memory_review(
        MemoryReview(
            id=review_id,
            kind=CROSS_SOURCE_CONFLICT_REVIEW_KIND,
            status=status.value,
            incumbent_memory_id=incumbent.id,
            challenger_memory_id=challenger.id,
            reason="contradiction: the statements disagree",
            review_note="different payroll runs" if status is ReviewStatus.REJECTED else None,
            reviewer=REVIEWER if status in {ReviewStatus.APPROVED, ReviewStatus.REJECTED} else None,
            expected_incumbent_updated_at=REVIEWED_AT.isoformat(),
            expected_challenger_updated_at=REVIEWED_AT.isoformat(),
            created_at=REVIEWED_AT,
            resolved_at=REVIEWED_AT if status in {ReviewStatus.APPROVED, ReviewStatus.REJECTED} else None,
        )
    )


async def _legacy_discovery_relation(db: Database, memory: Memory) -> None:
    await _memory(db, "mem-lifecycle-support")
    await db.upsert_evidence_unit(
        EvidenceUnit(
            id="eu-legacy",
            source_id="src-confluence",
            doc_id="doc-1",
            doc_revision_id="rev-1",
            source_type="confluence",
            source_anchor=None,
            source_lineage_id="unit-1",
            project_key=None,
            visibility="workspace",
            owner_user_id=None,
            repo_identifier=None,
            content="Evidence",
            excerpt="Evidence",
            evidence_provenance=EvidenceContentProvenance.SOURCE_EXCERPT,
        )
    )
    for run_id, source in (("run-discovery", "relation_discovery"), ("run-lifecycle", "lifecycle")):
        await db.db.execute(
            """INSERT INTO relation_runs (id, evidence_unit_id, classifier_version, status, audit_json)
               VALUES (?, 'eu-legacy', 'memory-relation-v4-sparse', 'checked', ?)""",
            (run_id, json.dumps({"source": source})),
        )
    for run_id, memory_id, relation_type, authoritative in (
        ("run-discovery", memory.id, "equivalent", 0),
        ("run-lifecycle", "mem-lifecycle-support", "supports", 1),
    ):
        await db.db.execute(
            """INSERT INTO evidence_relations (
                   evidence_unit_id, memory_id, relation_type, authority_case,
                   is_authoritative_support, classifier_version, relation_run_id
               ) VALUES ('eu-legacy', ?, ?, 'same_source_unit', ?, 'memory-relation-v4-sparse', ?)""",
            (memory_id, relation_type, authoritative, run_id),
        )
    await db.db.commit()


# ---------------------------------------------------------------------------
# Exhausted work and re-runs
# ---------------------------------------------------------------------------


async def _seed_work_states(db: Database) -> dict[str, str]:
    return {
        "exhausted_timeout": await _work(
            db, await _memory(db, "mem-w1"), status="failed", attempts=MAX_ATTEMPTS, error_code="TimeoutError"
        ),
        "exhausted_invalid": await _work(
            db,
            await _memory(db, "mem-w2"),
            status="failed",
            attempts=MAX_ATTEMPTS + 1,
            error_code="output_invalid",
            updated_at="2026-07-25T00:00:00+00:00",
        ),
        "retrying": await _work(
            db, await _memory(db, "mem-w3"), status="failed", attempts=1, error_code="TimeoutError"
        ),
        "completed_old": await _work(
            db,
            await _memory(db, "mem-w4"),
            status="completed",
            attempts=1,
            classifier_version="memory-relation-v4-sparse",
        ),
        "completed_new": await _work(
            db,
            await _memory(db, "mem-w5"),
            status="completed",
            attempts=1,
            classifier_version="cross-document-relation-v1",
        ),
    }


@pytest.mark.asyncio
async def test_work_listing_filters_by_state_error_time_and_classifier(db: Database, tmp_path) -> None:
    work = await _seed_work_states(db)

    with _client(db, tmp_path) as client:
        exhausted = client.get("/api/v1/relation-discovery/work", params={"state": "exhausted"}).json()
        by_code = client.get(
            "/api/v1/relation-discovery/work",
            params={"state": "exhausted", "error_code": "TimeoutError"},
        ).json()
        by_time = client.get(
            "/api/v1/relation-discovery/work",
            params={"state": "exhausted", "updated_from": "2026-07-24T00:00:00+00:00"},
        ).json()
        retrying = client.get("/api/v1/relation-discovery/work", params={"state": "failed"}).json()
        old = client.get(
            "/api/v1/relation-discovery/work",
            params={"state": "completed", "classifier_version": "memory-relation-v4-sparse"},
        ).json()
        paged = client.get(
            "/api/v1/relation-discovery/work",
            params={"state": "exhausted", "limit": 1, "offset": 1},
        ).json()

    assert [item["id"] for item in exhausted["data"]] == [work["exhausted_invalid"], work["exhausted_timeout"]]
    assert exhausted["total"] == 2
    assert exhausted["data"][0]["error"] == "output_invalid: failed"
    assert exhausted["data"][0]["error_code"] == "output_invalid"
    assert [item["id"] for item in by_code["data"]] == [work["exhausted_timeout"]]
    assert [item["id"] for item in by_time["data"]] == [work["exhausted_invalid"]]
    assert [item["id"] for item in retrying["data"]] == [work["retrying"]]
    assert [item["id"] for item in old["data"]] == [work["completed_old"]]
    assert [item["id"] for item in paged["data"]] == [work["exhausted_timeout"]]
    assert paged["total"] == 2


@pytest.mark.asyncio
async def test_work_routes_require_a_maintenance_operator(db: Database, tmp_path) -> None:
    await _seed_work_states(db)

    with _client(db, tmp_path, operator=False) as client:
        listing = client.get("/api/v1/relation-discovery/work", params={"state": "exhausted"})
        rerun = client.post("/api/v1/relation-discovery/work/rerun", json={"state": "exhausted"})

    assert listing.status_code == 403
    assert rerun.status_code == 403
    assert await db.count_relation_discovery_work(
        RelationDiscoveryWorkSelection(state=RelationDiscoveryWorkState.EXHAUSTED, max_attempts=MAX_ATTEMPTS)
    ) == 2


@pytest.mark.asyncio
async def test_rerun_audits_the_last_state_and_queues_a_new_generation(db: Database, tmp_path) -> None:
    work = await _seed_work_states(db)

    with _client(db, tmp_path) as client:
        retrying = client.post("/api/v1/relation-discovery/work/rerun", json={"state": "failed"})
        rerun = client.post(
            "/api/v1/relation-discovery/work/rerun",
            json={"state": "exhausted", "error_code": "TimeoutError"},
        )

    assert retrying.status_code == 422
    assert rerun.json() == {"rerun_count": 1}
    rows = {
        row["id"]: dict(row)
        for row in await db.db.execute_fetchall(
            "SELECT id, status, attempts, run_generation, error, error_code FROM relation_discovery_work"
        )
    }
    assert rows[work["exhausted_timeout"]] == {
        "id": work["exhausted_timeout"],
        "status": "pending",
        "attempts": 0,
        "run_generation": 1,
        "error": None,
        "error_code": None,
    }
    assert rows[work["exhausted_invalid"]]["status"] == "failed"
    assert rows[work["retrying"]]["attempts"] == 1
    [event] = await db.db.execute_fetchall(
        "SELECT actor_type, actor_id, memory_id, before_snapshot, payload FROM memory_audit_events WHERE event_type = ?",
        (RELATION_DISCOVERY_RERUN_EVENT,),
    )
    assert (event["actor_type"], event["actor_id"], event["memory_id"]) == ("maintenance_operator", OPERATOR, "mem-w1")
    assert json.loads(event["before_snapshot"]) == {
        "status": "failed",
        "error": "TimeoutError: failed",
        "error_code": "TimeoutError",
        "attempts": MAX_ATTEMPTS,
        "run_generation": 0,
        "classifier_version": None,
    }
    assert json.loads(event["payload"]) == {"work_id": work["exhausted_timeout"]}

    [leased] = await db.lease_relation_discovery_work(
        worker_id="relation-worker",
        limit=1,
        lease_seconds=60,
        max_attempts=MAX_ATTEMPTS,
    )
    assert (leased.request.id, leased.attempts, leased.run_generation) == (work["exhausted_timeout"], 1, 1)


@pytest.mark.asyncio
async def test_rerun_of_completed_work_keeps_its_classifier_version_until_it_completes_again(db: Database) -> None:
    work = await _seed_work_states(db)

    rerun = await db.rerun_relation_discovery_work(
        RelationDiscoveryWorkSelection(
            state=RelationDiscoveryWorkState.COMPLETED,
            max_attempts=MAX_ATTEMPTS,
            classifier_version="memory-relation-v4-sparse",
        ),
        actor=OPERATOR,
    )

    assert rerun == 1
    [row] = await db.db.execute_fetchall(
        "SELECT status, completed_at, classifier_version FROM relation_discovery_work WHERE id = ?",
        (work["completed_old"],),
    )
    assert dict(row) == {"status": "pending", "completed_at": None, "classifier_version": "memory-relation-v4-sparse"}


# ---------------------------------------------------------------------------
# Cross-Source Conflict Review conversion
# ---------------------------------------------------------------------------


async def _seed_reviews(db: Database) -> dict[str, Memory]:
    memories = {memory_id: await _memory(db, memory_id) for memory_id in "abcdefghij"}
    await _review(db, "rev-confirmed", ReviewStatus.APPROVED, challenger=memories["a"], incumbent=memories["b"])
    await _review(db, "rev-dismissed", ReviewStatus.REJECTED, challenger=memories["c"], incumbent=memories["d"])
    await _review(db, "rev-dismissed-old", ReviewStatus.REJECTED, challenger=memories["e"], incumbent=memories["f"])
    await _review(db, "rev-pending", ReviewStatus.PENDING, challenger=memories["g"], incumbent=memories["h"])
    await _review(db, "rev-confirmed-old", ReviewStatus.APPROVED, challenger=memories["i"], incumbent=memories["j"])
    await db.add_memory_review_related_challenger("rev-pending", "c", reason="same conflict")
    await _change(db, memories["f"])
    await _change(db, memories["j"])
    await _work(db, memories["g"], status="completed", attempts=1, classifier_version="memory-relation-v4-sparse")
    await _work(
        db, await _memory(db, "mem-exhausted"), status="failed", attempts=MAX_ATTEMPTS, error_code="TimeoutError"
    )
    await _legacy_discovery_relation(db, memories["a"])
    return memories


@pytest.mark.asyncio
async def test_conversion_report_classifies_every_review_and_writes_nothing(db: Database, tmp_path) -> None:
    await _seed_reviews(db)

    with _client(db, tmp_path) as client:
        first = client.post("/api/v1/memories/cross-source-review-conversion/report").json()
        second = client.post("/api/v1/memories/cross-source-review-conversion/report").json()

    assert first == second
    assert first["review_count"] == 5
    assert first["relation_labels"] == {"rev-confirmed": "contradicts"}
    assert first["dismissal_review_ids"] == ["rev-dismissed"]
    assert first["discarded_review_ids"] == ["rev-dismissed-old"]
    assert first["rerun_review_ids"] == ["rev-pending"]
    assert first["nothing_to_rerun_review_ids"] == ["rev-confirmed-old"]
    assert first["exhausted_work_ids"] == ["work-mem-exhausted"]
    assert first["planned"] == {
        "relations": 1,
        "dismissals": 2,
        "reruns": 2,
        "legacy_discovery_relations": 1,
    }
    assert first["applied"] is False
    assert await db.db.execute_fetchall("SELECT 1 FROM cross_document_relations") == []


@pytest.mark.asyncio
async def test_conversion_apply_writes_relations_dismissals_reruns_once(db: Database, tmp_path) -> None:
    memories = await _seed_reviews(db)

    with _client(db, tmp_path) as client:
        report = client.post("/api/v1/memories/cross-source-review-conversion/report").json()
        wrong = client.post(
            "/api/v1/memories/cross-source-review-conversion/apply",
            json={"report_id": "csrc-other"},
        )
        applied = client.post(
            "/api/v1/memories/cross-source-review-conversion/apply",
            json={"report_id": report["report_id"]},
        )
        again = client.post(
            "/api/v1/memories/cross-source-review-conversion/apply",
            json={"report_id": report["report_id"]},
        )
        detail = client.get("/api/v1/memories/c").json()
        search_view = client.get("/api/v1/memories/relations").json()

    assert wrong.status_code == 409
    assert applied.status_code == 200
    assert applied.json()["complete"] is True
    assert applied.json()["written"] == report["planned"]
    assert again.json() == applied.json()

    low_id, high_id = pair_key("a", "b")
    [relation] = await db.db.execute_fetchall("SELECT * FROM cross_document_relations")
    assert (relation["memory_low_id"], relation["memory_high_id"]) == (low_id, high_id)
    assert (relation["label"], relation["decided_by"]) == ("contradicts", "review")
    assert relation["decided_at"] == REVIEWED_AT.isoformat()
    assert relation["low_content_hash"] == memories[low_id].content_hash
    assert [item["label"] for item in search_view["data"]] == ["contradicts"]

    dismissals = await db.db.execute_fetchall(
        "SELECT * FROM cross_document_relation_dismissals ORDER BY label"
    )
    assert [row["label"] for row in dismissals] == [
        CrossDocumentRelationLabel.CONTRADICTS.value,
        CrossDocumentRelationLabel.UPDATES.value,
    ]
    for dismissal in dismissals:
        assert (dismissal["memory_low_id"], dismissal["memory_high_id"]) == ("c", "d")
        assert (dismissal["dismissed_by"], dismissal["dismissed_at"]) == (REVIEWER, REVIEWED_AT.isoformat())
        assert dismissal["note"] == "different payroll runs"
    [dismissed] = detail["dismissed_relations"]
    assert dismissed["counterpart"]["memory_id"] == "d"
    # Without source revision times an updates relation reads as a conflict.
    assert dismissed["labels"] == ["contradicts"]

    work = {
        row["id"]: (row["status"], row["run_generation"])
        for row in await db.db.execute_fetchall("SELECT id, status, run_generation FROM relation_discovery_work")
    }
    assert work == {"work-g": ("pending", 1), "work-mem-exhausted": ("pending", 1)}
    remaining = await db.db.execute_fetchall("SELECT memory_id FROM evidence_relations")
    assert [row["memory_id"] for row in remaining] == ["mem-lifecycle-support"]
    [event] = await db.db.execute_fetchall(
        "SELECT operation_id, actor_id, payload FROM memory_audit_events WHERE event_type = ?",
        (CONVERSION_APPLIED_EVENT,),
    )
    payload = json.loads(event["payload"])
    assert (event["operation_id"], event["actor_id"]) == (report["report_id"], OPERATOR)
    assert payload["relations"][0]["review_id"] == "rev-confirmed"
    assert payload["relations"][0]["reviewer"] == REVIEWER
    assert payload["discarded_review_ids"] == ["rev-dismissed-old"]


@pytest.mark.asyncio
async def test_conversion_report_converts_relabeled_reviews_by_their_label(db: Database, tmp_path) -> None:
    await _seed_reviews(db)
    relabels = {
        "rev-confirmed": "updates",
        "rev-dismissed": "equivalent",
        "rev-confirmed-old": "none",
    }

    with _client(db, tmp_path) as client:
        default = client.post("/api/v1/memories/cross-source-review-conversion/report").json()
        relabeled = client.post(
            "/api/v1/memories/cross-source-review-conversion/report",
            json={"label_overrides": relabels},
        ).json()

    assert relabeled["report_id"] != default["report_id"]
    assert relabeled["relation_labels"] == {"rev-confirmed": "updates", "rev-dismissed": "equivalent"}
    assert relabeled["dismissal_review_ids"] == []
    assert relabeled["discarded_review_ids"] == ["rev-confirmed-old", "rev-dismissed-old"]
    assert relabeled["rerun_review_ids"] == ["rev-pending"]
    assert relabeled["nothing_to_rerun_review_ids"] == []
    assert relabeled["planned"]["relations"] == 2
    assert relabeled["planned"]["dismissals"] == 0


@pytest.mark.asyncio
async def test_conversion_apply_writes_a_relabeled_confirmed_review_as_a_confirmed_relation(
    db: Database,
    tmp_path,
) -> None:
    await _seed_reviews(db)
    relabels = {"rev-confirmed": "updates"}

    with _client(db, tmp_path) as client:
        report = client.post(
            "/api/v1/memories/cross-source-review-conversion/report",
            json={"label_overrides": relabels},
        ).json()
        without_relabels = client.post(
            "/api/v1/memories/cross-source-review-conversion/apply",
            json={"report_id": report["report_id"]},
        )
        applied = client.post(
            "/api/v1/memories/cross-source-review-conversion/apply",
            json={"report_id": report["report_id"], "label_overrides": relabels},
        )

    assert report["relation_labels"] == {"rev-confirmed": "updates"}
    assert without_relabels.status_code == 409
    assert applied.status_code == 200
    assert applied.json()["complete"] is True
    [relation] = await db.db.execute_fetchall("SELECT label, decided_by FROM cross_document_relations")
    assert (relation["label"], relation["decided_by"]) == ("updates", "review")
    [event] = await db.db.execute_fetchall(
        "SELECT payload FROM memory_audit_events WHERE event_type = ?",
        (CONVERSION_APPLIED_EVENT,),
    )
    assert [(item["review_id"], item["label"]) for item in json.loads(event["payload"])["relations"]] == [
        ("rev-confirmed", "updates")
    ]


@pytest.mark.asyncio
async def test_conversion_relabels_must_name_decided_reviews(db: Database, tmp_path) -> None:
    await _seed_reviews(db)

    with _client(db, tmp_path) as client:
        responses = [
            client.post(
                "/api/v1/memories/cross-source-review-conversion/report",
                json={"label_overrides": {"rev-missing": "updates"}},
            ),
            client.post(
                "/api/v1/memories/cross-source-review-conversion/apply",
                json={"report_id": "csrc-x", "label_overrides": {"rev-pending": "contradicts"}},
            ),
        ]
        unknown_label = client.post(
            "/api/v1/memories/cross-source-review-conversion/report",
            json={"label_overrides": {"rev-confirmed": "supersedes"}},
        )

    assert [response.status_code for response in responses] == [400, 400]
    assert "rev-missing" in responses[0].json()["detail"]
    assert "rev-pending" in responses[1].json()["detail"]
    assert unknown_label.status_code == 422


@pytest.mark.asyncio
async def test_a_person_can_undo_a_converted_dismissal_later(db: Database, tmp_path) -> None:
    await _seed_reviews(db)

    with _client(db, tmp_path) as client:
        report = client.post("/api/v1/memories/cross-source-review-conversion/report").json()
        client.post("/api/v1/memories/cross-source-review-conversion/apply", json={"report_id": report["report_id"]})
        restored = client.delete("/api/v1/memories/c/relations/d/dismissal")
        detail = client.get("/api/v1/memories/c").json()

    assert restored.status_code == 200
    assert len(restored.json()["restored_dismissal_ids"]) == 2
    assert detail["dismissed_relations"] == []


@pytest.mark.asyncio
async def test_a_later_conversion_keeps_a_dismissal_a_person_undid(db: Database, tmp_path) -> None:
    await _seed_reviews(db)

    with _client(db, tmp_path) as client:
        first = client.post("/api/v1/memories/cross-source-review-conversion/report").json()
        client.post("/api/v1/memories/cross-source-review-conversion/apply", json={"report_id": first["report_id"]})
        client.delete("/api/v1/memories/c/relations/d/dismissal")
        second = client.post("/api/v1/memories/cross-source-review-conversion/report").json()
        applied = client.post(
            "/api/v1/memories/cross-source-review-conversion/apply",
            json={"report_id": second["report_id"]},
        )
        detail = client.get("/api/v1/memories/c").json()

    assert second["report_id"] != first["report_id"]
    assert applied.json()["complete"] is True
    assert detail["dismissed_relations"] == []
    rows = await db.db.execute_fetchall("SELECT restored_at FROM cross_document_relation_dismissals")
    assert len(rows) == 2 and all(row["restored_at"] for row in rows)


@pytest.mark.asyncio
async def test_conversion_counts_rows_whose_content_changed_after_the_report(db: Database) -> None:
    memories = await _seed_reviews(db)
    plan = await build_conversion_plan(db, label_overrides={}, max_attempts=MAX_ATTEMPTS)
    await _change(db, memories["b"])

    receipt = await db.apply_cross_source_review_conversion(plan, actor=OPERATOR)

    assert receipt.planned["relations"] == 1
    assert receipt.written["relations"] == 0
    assert receipt.complete is False
    assert await db.db.execute_fetchall("SELECT 1 FROM cross_document_relations") == []
    with pytest.raises(CrossSourceReviewConversionConflict, match="applied counts differ"):
        await delete_converted_reviews(db, report_id=plan.report_id, actor=OPERATOR)
    with pytest.raises(CrossSourceReviewConversionConflict, match="already applied"):
        await db.apply_cross_source_review_conversion(plan, actor=OPERATOR)


@pytest.mark.asyncio
async def test_converted_reviews_are_deleted_only_after_apply_and_a_frozen_relation_cohort(
    db: Database,
    tmp_path,
    monkeypatch,
) -> None:
    await _seed_reviews(db)
    cohorts = {"aeo-relations": SimpleNamespace(selection_policy_version=RELATION_CASE_POLICY_VERSION)}

    async def get_cohort(cohort_id: str):
        return cohorts.get(cohort_id)

    monkeypatch.setattr(db, "get_agent_evaluation_cohort", get_cohort)
    with _client(db, tmp_path) as client:
        report = client.post("/api/v1/memories/cross-source-review-conversion/report").json()
        request = {"report_id": report["report_id"], "cohort_id": "aeo-relations"}
        before_apply = client.post("/api/v1/memories/cross-source-review-conversion/delete", json=request)
        client.post("/api/v1/memories/cross-source-review-conversion/apply", json={"report_id": report["report_id"]})
        without_cohort = client.post(
            "/api/v1/memories/cross-source-review-conversion/delete",
            json={**request, "cohort_id": "aeo-missing"},
        )
        deleted = client.post("/api/v1/memories/cross-source-review-conversion/delete", json=request)

    assert before_apply.status_code == 409
    assert without_cohort.status_code == 409
    assert deleted.json() == {"report_id": report["report_id"], "deleted_review_count": 5}
    assert await db.list_memory_reviews(kind=CROSS_SOURCE_CONFLICT_REVIEW_KIND) == []
    assert await db.db.execute_fetchall("SELECT 1 FROM memory_review_related_challengers") == []


@pytest.mark.asyncio
async def test_conversion_routes_require_a_maintenance_operator(db: Database, tmp_path) -> None:
    with _client(db, tmp_path, operator=False) as client:
        responses = [
            client.post("/api/v1/memories/cross-source-review-conversion/report"),
            client.post("/api/v1/memories/cross-source-review-conversion/apply", json={"report_id": "csrc-x"}),
            client.post(
                "/api/v1/memories/cross-source-review-conversion/delete",
                json={"report_id": "csrc-x", "cohort_id": "aeo-x"},
            ),
        ]

    assert [response.status_code for response in responses] == [403, 403, 403]


class _ReceiptStore:
    def __init__(self, receipt: CrossSourceReviewConversionReceipt | None) -> None:
        self.receipt = receipt
        self.applied = False

    async def get_cross_source_review_conversion(self, report_id: str):
        return self.receipt if self.receipt and self.receipt.report_id == report_id else None

    async def apply_cross_source_review_conversion(self, plan, *, actor):
        self.applied = True
        raise AssertionError("an applied report is not applied again")


@pytest.mark.asyncio
async def test_applying_an_applied_report_returns_its_receipt_without_writing() -> None:
    receipt = CrossSourceReviewConversionReceipt(
        report_id="csrc-1",
        applied_by=OPERATOR,
        applied_at=REVIEWED_AT.isoformat(),
        planned={"relations": 1},
        written={"relations": 1},
        review_ids=("rev-1",),
    )
    store = _ReceiptStore(receipt)

    applied = await apply_conversion(
        store,
        report_id="csrc-1",
        label_overrides={},
        actor=OPERATOR,
        max_attempts=MAX_ATTEMPTS,
    )

    assert applied == receipt
    assert store.applied is False
    with pytest.raises(CrossSourceReviewConversionConflict, match="not been applied"):
        await delete_converted_reviews(_ReceiptStore(None), report_id="csrc-1", actor=OPERATOR)


@pytest.mark.asyncio
async def test_migration_96_drops_contradiction_counts(db: Database) -> None:
    await db.db.execute("ALTER TABLE memories ADD COLUMN contradiction_count INTEGER NOT NULL DEFAULT 0")
    await db.db.execute(
        "CREATE TABLE memory_contradictions (memory_id_a TEXT, memory_id_b TEXT, classification TEXT)"
    )
    await db.db.execute("DELETE FROM schema_migrations WHERE version = 96")
    await db.db.commit()

    await db._run_migrations()  # noqa: SLF001
    await db._run_migrations()  # noqa: SLF001

    columns = {row[1] for row in await db.db.execute_fetchall("PRAGMA table_info(memories)")}
    tables = {
        row[0] for row in await db.db.execute_fetchall("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    assert "contradiction_count" not in columns
    assert "memory_contradictions" not in tables
