"""Review reads preserve exact eligibility while batching page hydration.

Complete queue enumeration and version checks still scale with the queue.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from memforge.config import AppConfig
from memforge.memory.lifecycle_plan import LifecycleReviewStatus
from memforge.models import (
    Memory,
    MemoryReview,
    ReviewKind,
    ReviewStatus,
    content_hash,
)
from memforge.server.admin_api import create_admin_app
from memforge.storage.database import Database

SOURCE_ID = "src-queue"
OWNER = "dev"
BASE_TIME = datetime(2026, 9, 1, tzinfo=timezone.utc)
SMALL_QUEUE = 3
LARGE_QUEUE = 30
PAGE_SIZE = 5


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "queue.db"))
    await database.connect()
    yield database
    await database.close()


def _config(tmp_path: Path) -> AppConfig:
    config = AppConfig(base_dir=tmp_path / "memforge")
    config.sync.worker_enabled = False
    return config


def _at(minutes: int) -> datetime:
    return BASE_TIME + timedelta(minutes=minutes)


async def _memory(db: Database, memory_id: str, *, status: str = "active", minute: int = 0) -> Memory:
    memory = Memory(
        id=memory_id,
        memory_type="fact",
        content=f"{memory_id} content",
        content_hash=content_hash(f"{memory_id} content"),
        status=status,
        created_at=_at(minute),
        updated_at=_at(minute),
    )
    await db.insert_memory(memory)
    stored = await db.get_memory(memory_id)
    assert stored is not None
    return stored


async def _source(db: Database) -> None:
    await db.upsert_source(
        id=SOURCE_ID,
        type="jira",
        name="Queue source",
        config_json="{}",
        access_policy="workspace",
        owner_user_id=OWNER,
    )


async def _memory_review(db: Database, index: int, *, status: str = ReviewStatus.PENDING.value) -> MemoryReview:
    incumbent = await _memory(db, f"mem-inc-{index:03d}", minute=index)
    challenger = await _memory(db, f"mem-chal-{index:03d}", status="pending_review", minute=index)
    review = MemoryReview(
        id=f"rev-mem-{index:03d}",
        kind=ReviewKind.SUPERSEDE.value,
        status=status,
        incumbent_memory_id=incumbent.id,
        challenger_memory_id=challenger.id,
        reason="Newer document",
        expected_incumbent_updated_at=incumbent.updated_at.isoformat(),
        expected_challenger_updated_at=challenger.updated_at.isoformat(),
        created_at=_at(2 * index),
    )
    await db.insert_memory_review(review)
    return review


async def _lifecycle_review(
    db: Database,
    index: int,
    *,
    candidate_id: str | None = None,
    status: LifecycleReviewStatus = LifecycleReviewStatus.PENDING,
) -> str:
    incumbent = await _memory(db, f"mem-life-{index:03d}", minute=index)
    plan_id = f"plan-{index:03d}"
    review_id = f"rev-life-{index:03d}"
    await db.db.execute(
        """INSERT INTO lifecycle_plans (
               id, reconciliation_scope_id, source_id, source_unit_id,
               target_unit_revision_id, status, payload_json, payload_hash, created_at
           ) VALUES (?, ?, ?, ?, ?, 'applied', ?, 'hash', ?)""",
        (plan_id, f"scope-{index}", SOURCE_ID, f"unit-{index}", f"unitrev-{index}", json.dumps({"scope": {"source_id": SOURCE_ID}}), _at(0).isoformat()),
    )
    staged = {
        "proposed_disposition": "supersede",
        "replacement_memory_id": candidate_id,
        "candidate": {"content": f"Candidate {index}", "memory_type": "fact"},
        "proposed_mutations": [],
    }
    await db.db.execute(
        """INSERT INTO lifecycle_reviews (
               id, lifecycle_plan_id, incumbent_memory_id, status, staged_evidence_json, reason, created_at
           ) VALUES (?, ?, ?, ?, ?, 'candidate_supersede_vs_audit_keep', ?)""",
        (review_id, plan_id, incumbent.id, status.value, json.dumps(staged), _at(2 * index + 1).isoformat()),
    )
    await db.db.commit()
    return review_id


async def _queue(db: Database, size: int) -> None:
    await _source(db)
    for index in range(size):
        await _memory_review(db, index)
        await _lifecycle_review(db, index)


class _CountingDatabase:
    """Records each storage method the admin routes await, and the ids the batched reads receive."""

    def __init__(self, inner: Database) -> None:
        self._inner = inner
        self.calls: Counter[str] = Counter()
        self.ids: dict[str, list[int]] = {}

    def __getattr__(self, name: str):
        value = getattr(self._inner, name)
        if not asyncio.iscoroutinefunction(value):
            return value

        async def counted(*args, **kwargs):
            self.calls[name] += 1
            if args and isinstance(args[0], Sequence) and not isinstance(args[0], str):
                self.ids.setdefault(name, []).append(len(args[0]))
            return await value(*args, **kwargs)

        return counted


@contextmanager
def _counted_client(db: Database, tmp_path: Path) -> Iterator[tuple[TestClient, _CountingDatabase]]:
    """A client whose requests read storage through a counter; the app installs its database on startup."""
    app = create_admin_app(db=db, config=_config(tmp_path))
    counting = _CountingDatabase(db)
    with TestClient(app) as client:
        app.state.db = counting
        yield client, counting


QUEUE_PAGE_CALLS = {
    "list_memory_reviews": 1,
    "list_lifecycle_review_queue_entries": 1,
    "list_memory_review_related_challengers_many": 1,
    "list_memories_by_ids": 2,
    "get_memory_source_ids_many": 1,
    "filter_visible_ids": 1,
    "list_sources": 1,
    "list_lifecycle_reviews_by_ids": 1,
    "get_origin_source_pairs": 1,
}
"""One page of the open queue: the narrow queue reads, then the full reads of the page."""


class TestReviewQueueReads:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("size", [SMALL_QUEUE, LARGE_QUEUE])
    async def test_a_queue_page_batches_hydration_for_small_queues(self, db, tmp_path, size):
        await _queue(db, size)
        with _counted_client(db, tmp_path) as (client, counting):
            response = client.get("/api/v1/memory-reviews", params={"status": "open", "limit": PAGE_SIZE})

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["total"] == 2 * size
        assert len(body["data"]) == PAGE_SIZE
        assert dict(counting.calls) == QUEUE_PAGE_CALLS
        assert counting.ids["list_lifecycle_reviews_by_ids"] == [
            sum(item["review_origin"] == "lifecycle" for item in body["data"])
        ]
        assert max(counting.ids["get_origin_source_pairs"]) <= 2 * PAGE_SIZE

    @pytest.mark.asyncio
    async def test_pages_follow_the_queue_order_across_both_kinds(self, db, tmp_path):
        await _queue(db, SMALL_QUEUE)
        app = create_admin_app(db=db, config=_config(tmp_path))
        with TestClient(app) as client:
            whole = client.get("/api/v1/memory-reviews", params={"status": "open", "limit": 100}).json()
            pages = [
                client.get("/api/v1/memory-reviews", params={"status": "open", "limit": 2, "offset": offset}).json()
                for offset in range(0, 2 * SMALL_QUEUE, 2)
            ]

        ids = [item["id"] for item in whole["data"]]
        assert ids == [
            "rev-life-002", "rev-mem-002", "rev-life-001", "rev-mem-001", "rev-life-000", "rev-mem-000",
        ]
        assert [item["id"] for page in pages for item in page["data"]] == ids
        assert {page["total"] for page in pages} == {2 * SMALL_QUEUE}

    @pytest.mark.asyncio
    async def test_a_lifecycle_candidate_the_caller_cannot_see_hides_its_review(self, db, tmp_path):
        await _source(db)
        hidden = Memory(
            id="mem-hidden-candidate",
            memory_type="fact",
            content="Someone else's candidate",
            content_hash=content_hash("Someone else's candidate"),
            visibility="private",
            owner_user_id="someone-else",
        )
        await db.insert_memory(hidden)
        hidden_review = await _lifecycle_review(db, 0, candidate_id=hidden.id)
        proposed_review = await _lifecycle_review(db, 1, candidate_id="mem-not-created-yet")

        app = create_admin_app(db=db, config=_config(tmp_path))
        with TestClient(app) as client:
            body = client.get("/api/v1/memory-reviews", params={"status": "open"}).json()

        assert [item["id"] for item in body["data"]] == [proposed_review]
        assert body["total"] == 1
        assert hidden_review not in {item["id"] for item in body["data"]}


class TestMemoryPageReads:
    @pytest.mark.asyncio
    async def test_a_waiting_memory_carries_its_review_in_one_batched_read(self, db, tmp_path):
        await _queue(db, LARGE_QUEUE)
        with _counted_client(db, tmp_path) as (client, counting):
            listed = client.get(
                "/api/v1/memories",
                params={"status": "pending_review", "include_private": "true", "limit": PAGE_SIZE},
            )
            waiting_calls = dict(counting.calls)
            counting.calls.clear()
            active = client.get("/api/v1/memories", params={"include_private": "true", "limit": PAGE_SIZE})
            active_calls = dict(counting.calls)

        assert listed.status_code == 200, listed.text
        rows = listed.json()["data"]
        assert len(rows) == PAGE_SIZE
        assert all(row["open_review_id"] == row["id"].replace("mem-chal-", "rev-mem-") for row in rows)
        assert waiting_calls["list_pending_memory_reviews_for_memories"] == 1
        assert counting.ids["list_pending_memory_reviews_for_memories"] == [PAGE_SIZE]
        assert "list_pending_memory_reviews_for_memories" not in active_calls
        assert all(row["open_review_id"] is None for row in active.json()["data"])

    @pytest.mark.asyncio
    async def test_memory_detail_names_the_review_it_waits_for(self, db, tmp_path):
        await _source(db)
        review = await _memory_review(db, 0)
        app = create_admin_app(db=db, config=_config(tmp_path))
        with TestClient(app) as client:
            waiting = client.get(f"/api/v1/memories/{review.challenger_memory_id}").json()
            incumbent = client.get(f"/api/v1/memories/{review.incumbent_memory_id}").json()

        assert waiting["open_review_id"] == review.id
        assert incumbent["open_review_id"] is None


class TestPendingReviewLookup:
    @pytest.mark.asyncio
    async def test_candidate_lookup_includes_all_sides_and_excludes_terminal_reviews(self, db):
        first = await _memory_review(db, 0)
        second = await _memory_review(db, 1)
        await _memory(db, "mem-related", status="pending_review")
        await db.add_memory_review_related_challenger(second.id, "mem-related")
        terminal = await _memory_review(db, 2, status=ReviewStatus.APPROVED.value)
        found = await db.list_pending_memory_reviews_for_memories([
            first.incumbent_memory_id, first.challenger_memory_id,
            "mem-related", terminal.challenger_memory_id, "mem-unknown",
        ])
        assert {review.id for review in found} == {first.id, second.id}
        assert len(found) == 2
        assert await db.list_pending_memory_reviews_for_memories([]) == []


class TestLifecycleQueueEntries:
    @pytest.mark.asyncio
    async def test_entries_name_the_candidate_without_the_staged_evidence(self, db):
        await _source(db)
        first = await _lifecycle_review(db, 0, candidate_id="mem-candidate")
        second = await _lifecycle_review(db, 1)
        await _lifecycle_review(db, 2, status=LifecycleReviewStatus.APPROVED)

        pending = await db.list_lifecycle_review_queue_entries(status=LifecycleReviewStatus.PENDING)
        elsewhere = await db.list_lifecycle_review_queue_entries("src-other")

        assert [(entry.id, entry.candidate_memory_id, entry.source_id) for entry in pending] == [
            (second, None, SOURCE_ID),
            (first, "mem-candidate", SOURCE_ID),
        ]
        assert elsewhere == []

    @pytest.mark.asyncio
    async def test_reviews_by_ids_keep_the_order_given(self, db):
        await _source(db)
        first = await _lifecycle_review(db, 0)
        second = await _lifecycle_review(db, 1)

        reviews = await db.list_lifecycle_reviews_by_ids([second, "rev-unknown", first])

        assert [review.id for review in reviews] == [second, first]
        assert reviews[0].source_id == SOURCE_ID
        assert reviews[0].staged_evidence["candidate"] == {"content": "Candidate 1", "memory_type": "fact"}


async def _attach_source(db: Database, memory_id: str, source_id: str) -> None:
    doc_id = f"doc-{source_id}"
    await db.db.execute(
        """INSERT OR IGNORE INTO documents
           (doc_id, source, source_url, title, space_or_project, last_modified,
            version, content_hash, last_synced)
           VALUES (?, 'jira', '', 'Review source', '', ?, '1', 'hash', ?)""",
        (doc_id, BASE_TIME.isoformat(), BASE_TIME.isoformat()),
    )
    await db.db.execute(
        """INSERT INTO memory_sources (memory_id, doc_id, source_id, source_type)
           VALUES (?, ?, ?, 'jira')""", (memory_id, doc_id, source_id),
    )
    await db.db.commit()


class TestReviewNavigationEligibility:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("case", ["hidden_incumbent", "hidden_related", "hidden_source", "stale_incumbent", "stale_challenger"])
    async def test_list_and_detail_never_link_a_review_excluded_from_open_queue(self, db, tmp_path, case):
        await _source(db)
        review = await _memory_review(db, 0)
        if case == "hidden_related":
            related = await _memory(db, "mem-related", status="pending_review")
            await db.add_memory_review_related_challenger(review.id, related.id)
            hidden_id = related.id
        else:
            hidden_id = review.incumbent_memory_id
        if case.startswith("hidden_") and case != "hidden_source":
            await db.db.execute("UPDATE memories SET visibility = 'private', owner_user_id = 'bob' WHERE id = ?", (hidden_id,))
        elif case == "hidden_source":
            await db.upsert_source(id="src-private", type="jira", name="Bob private", config_json="{}",
                                   access_policy="private", owner_user_id="bob")
            # The incumbent remains readable through one public Source. A hidden
            # additional Source must still hide the complete Review.
            await _attach_source(db, review.incumbent_memory_id, SOURCE_ID)
            await _attach_source(db, review.incumbent_memory_id, "src-private")
        else:
            changed_id = review.incumbent_memory_id if case == "stale_incumbent" else review.challenger_memory_id
            await db.db.execute("UPDATE memories SET updated_at = ? WHERE id = ?", (_at(99).isoformat(), changed_id))
        await db.db.commit()
        app = create_admin_app(db=db, config=_config(tmp_path))
        with TestClient(app) as client:
            listed = client.get("/api/v1/memories", params={"status": "pending_review", "include_private": "true"})
            detail = client.get(f"/api/v1/memories/{review.challenger_memory_id}")
            queue = client.get("/api/v1/memory-reviews", params={"status": "open"})
            assert listed.status_code == detail.status_code == queue.status_code == 200
            row = next(item for item in listed.json()["data"] if item["id"] == review.challenger_memory_id)
            assert row["open_review_id"] is None
            assert detail.json()["open_review_id"] is None
            assert queue.json()["total"] == 0
            # Read paths do not terminalize a dynamically stale or hidden row.
            assert (await db.get_memory_review(review.id)).status == "pending"
            if case.startswith("stale_"):
                history = client.get(f"/api/v1/memory-reviews/{review.id}")
                assert history.status_code == 200
                assert history.json()["is_stale"] is True
                attempted = client.post(f"/api/v1/memory-reviews/{review.id}/approve",
                                        json={"expected_fingerprint": history.json()["decision_fingerprint"]})
                assert attempted.status_code == 409
            elif case in {"hidden_incumbent", "hidden_related", "hidden_source"}:
                assert client.get(f"/api/v1/memory-reviews/{review.id}").status_code == 404
                assert client.post(f"/api/v1/memory-reviews/{review.id}/approve",
                                   json={"expected_fingerprint": "synthetic"}).status_code == 404

    @pytest.mark.asyncio
    @pytest.mark.parametrize("newer_case", ["stale", "hidden"])
    async def test_newer_ineligible_candidate_does_not_shadow_older_open_review(self, db, tmp_path, newer_case):
        older = await _memory_review(db, 0)
        newer = await _memory_review(db, 1)
        challenger = await db.get_memory(older.challenger_memory_id)
        await db.db.execute(
            "UPDATE memory_reviews SET challenger_memory_id = ?, expected_challenger_updated_at = ? WHERE id = ?",
            (challenger.id, challenger.updated_at.isoformat(), newer.id),
        )
        if newer_case == "stale":
            await db.db.execute("UPDATE memories SET updated_at = ? WHERE id = ?", (_at(99).isoformat(), newer.incumbent_memory_id))
        else:
            await db.db.execute("UPDATE memories SET visibility = 'private', owner_user_id = 'bob' WHERE id = ?", (newer.incumbent_memory_id,))
        await db.db.commit()
        app = create_admin_app(db=db, config=_config(tmp_path))
        with TestClient(app) as client:
            listed = client.get("/api/v1/memories", params={"status": "pending_review"}).json()
            row = next(item for item in listed["data"] if item["id"] == challenger.id)
            assert row["open_review_id"] == older.id
            assert client.get(f"/api/v1/memories/{challenger.id}").json()["open_review_id"] == older.id
            assert [item["id"] for item in client.get("/api/v1/memory-reviews").json()["data"]] == [older.id]

    @pytest.mark.asyncio
    async def test_lifecycle_review_link_checks_existing_candidate_visibility(self, db, tmp_path):
        await _source(db)
        candidate = await _memory(db, "mem-life-hidden", status="pending_review")
        review_id = await _lifecycle_review(db, 0, candidate_id=candidate.id)
        await db.db.execute("UPDATE memories SET status = 'pending_review' WHERE id = 'mem-life-000'")
        await db.db.execute("UPDATE memories SET visibility = 'private', owner_user_id = 'bob' WHERE id = ?", (candidate.id,))
        await db.db.commit()
        app = create_admin_app(db=db, config=_config(tmp_path))
        with TestClient(app) as client:
            assert client.get("/api/v1/memories/mem-life-000").json()["open_review_id"] is None
            assert client.get("/api/v1/memory-reviews").json()["total"] == 0
            assert client.get(f"/api/v1/memory-reviews/{review_id}").status_code == 404

    @pytest.mark.asyncio
    async def test_exact_queue_enumeration_crosses_the_storage_page_boundary(self, db, tmp_path):
        for index in range(501):
            await _memory_review(db, index)
        app = create_admin_app(db=db, config=_config(tmp_path))
        with TestClient(app) as client:
            page = client.get("/api/v1/memory-reviews", params={"limit": 1}).json()
            assert page["total"] == 501
            assert [item["id"] for item in page["data"]] == ["rev-mem-500"]
            oldest = client.get("/api/v1/memory-reviews", params={"limit": 1, "offset": 500}).json()
            assert oldest["total"] == 501
            assert [item["id"] for item in oldest["data"]] == ["rev-mem-000"]


@pytest.mark.asyncio
async def test_review_scope_keeps_owned_private_participants_when_memory_page_excludes_private(db, tmp_path):
    review = await _memory_review(db, 0)
    await db.db.execute("UPDATE memories SET visibility = 'private', owner_user_id = ? WHERE id = ?", (OWNER, review.incumbent_memory_id))
    await db.db.commit()
    app = create_admin_app(db=db, config=_config(tmp_path))
    with TestClient(app) as client:
        listed = client.get("/api/v1/memories", params={"status": "pending_review", "include_private": "false"}).json()
        assert next(item for item in listed["data"] if item["id"] == review.challenger_memory_id)["open_review_id"] == review.id
        assert client.get(f"/api/v1/memories/{review.challenger_memory_id}").json()["open_review_id"] == review.id
        assert [item["id"] for item in client.get("/api/v1/memory-reviews").json()["data"]] == [review.id]


@pytest.mark.asyncio
@pytest.mark.parametrize("hidden_participant", ["incumbent", "candidate"])
async def test_lifecycle_review_requires_every_participant_source_visible(db, tmp_path, hidden_participant):
    await _source(db)
    candidate = await _memory(db, "mem-life-candidate", status="pending_review")
    review_id = await _lifecycle_review(db, 0, candidate_id=candidate.id)
    await db.db.execute("UPDATE memories SET status = 'pending_review' WHERE id = 'mem-life-000'")
    await db.upsert_source(id="src-life-private", type="jira", name="Bob private", config_json="{}",
                           access_policy="private", owner_user_id="bob")
    participant_id = "mem-life-000" if hidden_participant == "incumbent" else candidate.id
    await _attach_source(db, participant_id, SOURCE_ID)
    await _attach_source(db, participant_id, "src-life-private")
    app = create_admin_app(db=db, config=_config(tmp_path))
    with TestClient(app) as client:
        assert client.get("/api/v1/memory-reviews").json()["total"] == 0
        listed = client.get("/api/v1/memories", params={"status": "pending_review"}).json()
        assert next(item for item in listed["data"] if item["id"] == "mem-life-000")["open_review_id"] is None
        assert client.get("/api/v1/memories/mem-life-000").json()["open_review_id"] is None
        assert client.get(f"/api/v1/memory-reviews/{review_id}").status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize("hidden", [False, True])
async def test_lifecycle_visibility_precedes_management_for_members(db, tmp_path, hidden):
    await _source(db)
    review_id = await _lifecycle_review(db, 0)
    await db.db.execute("UPDATE sources SET owner_user_id = 'alice' WHERE id = ?", (SOURCE_ID,))
    if hidden:
        await db.upsert_source(id="src-member-private", type="jira", name="Bob private", config_json="{}",
                               access_policy="private", owner_user_id="bob")
        await _attach_source(db, "mem-life-000", SOURCE_ID)
        await _attach_source(db, "mem-life-000", "src-member-private")
    await db.db.commit()
    app = create_admin_app(db=db, config=_config(tmp_path), workspace_role_resolver=lambda request: "member")
    with TestClient(app) as client:
        expected = 404 if hidden else 403
        for path, payload in (
            (f"/api/v1/memory-reviews/{review_id}/approve", {"expected_fingerprint": "synthetic"}),
            (f"/api/v1/memory-reviews/{review_id}/reject", {"expected_fingerprint": "synthetic", "note": "keep current"}),
            (f"/api/v1/memory-reviews/{review_id}/refresh", {"expected_fingerprint": "synthetic"}),
            (f"/api/v1/sources/{SOURCE_ID}/memory-lifecycle/reviews/{review_id}/approve", {"expected_fingerprint": "synthetic"}),
        ):
            response = client.post(path, json=payload)
            assert response.status_code == expected, (path, response.text)
        manifest = client.post("/api/v1/memory-reviews/decisions/validate", json={"decisions": [
            {"review_id": review_id, "decision": "approve", "expected_fingerprint": "synthetic"},
        ]})
        assert manifest.status_code == 200
        assert manifest.json()["results"][0]["outcome"] == ("not_found" if hidden else "forbidden")
    assert (await db.get_lifecycle_review(review_id)).status == LifecycleReviewStatus.PENDING
