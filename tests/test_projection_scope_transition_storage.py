"""Projection Scope transition storage: member rows, supersession and completion without coverage."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone

import aiosqlite
import pytest
import pytest_asyncio

from memforge.memory.evidence import EVIDENCE_UNIT_SUPPORT_SCOPE
from memforge.memory.lifecycle_plan import (
    CoverageProof,
    LifecyclePlan,
    ReconciliationScope,
    StaleGuard,
)
from memforge.models import ContentItem, NormalizedContent, RawContent, SourceUnitInput
from memforge.pipeline.source_projection_adapters import project_source_item, project_source_unit_tombstone
from memforge.source_projection import (
    ProjectionScopeTransition,
    ProjectionScopeTransitionStale,
    ProjectionScopeTransitionStatus,
    SourceProjection,
)
from memforge.storage.database import MIGRATIONS, SCHEMA, Database

SOURCE = "src-docs"
OTHER_SOURCE = "src-other"


@pytest_asyncio.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "transitions.db"))
    await database.connect()
    for source_id in (SOURCE, OTHER_SOURCE):
        await database.upsert_source(
            id=source_id,
            type="local_markdown",
            name=source_id,
            config_json=json.dumps({"path": "/docs"}),
            access_policy="workspace",
            owner_user_id="dev",
        )
    try:
        yield database
    finally:
        await database.close()


def _document(path: str, body: str, *, source_id: str = SOURCE) -> tuple[SourceProjection, SourceUnitInput]:
    item = ContentItem(
        item_id=path,
        title=path,
        source_url="",
        last_modified=datetime(2026, 10, 20, tzinfo=timezone.utc),
    )
    projection = project_source_item(
        source_id=source_id,
        source_type="local_markdown",
        run_id=f"run-{source_id}-{path}-{body}",
        item=item,
        raw=RawContent(item=item, body=body.encode(), content_type="text/markdown"),
        normalized=NormalizedContent(item=item, markdown_body=body),
    )
    unit_input = SourceUnitInput(
        source_unit_id=projection.source_units[0].id,
        unit_revision_id=projection.source_unit_revisions[0].id,
        source_id=source_id,
        document_id=path,
        item=item,
        raw_content_uri=None,
        raw_content_type="text/markdown",
        raw_content_sha256=None,
        normalized_content_uri=None,
        normalized_content_hash=None,
        pdf_content_uri=None,
    )
    return projection, unit_input


def _transition(transition_id: str, target: str, *, predecessor: str | None = None) -> ProjectionScopeTransition:
    return ProjectionScopeTransition(
        id=transition_id,
        source_id=SOURCE,
        previous_scope={"path": "/docs"},
        target_scope={"path": target},
        predecessor_transition_id=predecessor,
        created_at="2026-10-20T10:00:00+00:00",
    )


async def _change_scope(db: Database, transition: ProjectionScopeTransition, **kwargs) -> None:
    await db.upsert_source(
        id=SOURCE,
        type="local_markdown",
        name=SOURCE,
        config_json=json.dumps({"path": transition.target_scope["path"]}),
        access_policy="workspace",
        owner_user_id="dev",
        projection_scope_transition=transition,
        **kwargs,
    )


async def _member_revisions(db: Database, transition_id: str) -> dict[str, str]:
    rows = await db.db.execute_fetchall(
        "SELECT doc_id, unit_revision_id FROM projection_scope_transition_members WHERE transition_id = ?",
        (transition_id,),
    )
    return {str(row["doc_id"]): str(row["unit_revision_id"]) for row in rows}


def _plan(projection: SourceProjection, *, observation_revision_ids: tuple[str, ...] = ()) -> LifecyclePlan:
    delta = projection.deltas[0]
    return LifecyclePlan(
        id=f"plan-{projection.run_id}",
        scope=ReconciliationScope(
            id=f"scope-{projection.run_id}",
            source_id=projection.source_id,
            source_unit_id=delta.source_unit_id,
            base_unit_revision_id=None,
            target_unit_revision_id=delta.current_unit_revision_id,
        ),
        coverage_proof=CoverageProof((), (), (), ()),
        stale_guard=StaleGuard(observation_revision_ids, {}),
        mutations=(),
    )


async def _database_at_schema_124(path: str) -> None:
    """A database at schema 124, whose transitions still record a coverage and know no supersession."""

    connection = await aiosqlite.connect(path)
    await connection.executescript(SCHEMA)
    await connection.executescript(
        """
        DROP TABLE projection_scope_transition_members;
        DROP TABLE projection_scope_transitions;
        CREATE TABLE projection_scope_transitions (
            id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
            previous_scope_json TEXT NOT NULL,
            target_scope_json TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'applied', 'failed')),
            run_id TEXT,
            coverage TEXT,
            error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            completed_at TEXT
        );
        CREATE INDEX idx_projection_scope_transitions_source
            ON projection_scope_transitions(source_id, created_at);
        """
    )
    for version, description, _statements in MIGRATIONS:
        if version > 124:
            break
        await connection.execute(
            "INSERT INTO schema_migrations (version, description, applied_at) VALUES (?, ?, ?)",
            (version, description, "2026-10-01T00:00:00+00:00"),
        )
    await connection.execute(
        "INSERT INTO system_contract_markers (marker_key, marker_value, updated_at) VALUES (?, ?, ?)",
        ("support_scope_version", EVIDENCE_UNIT_SUPPORT_SCOPE, "2026-10-01T00:00:00+00:00"),
    )
    await connection.execute(
        "INSERT INTO sources (id, type, name, config, access_policy, owner_user_id) VALUES (?, ?, ?, ?, ?, ?)",
        (SOURCE, "local_markdown", SOURCE, "{}", "workspace", "dev"),
    )
    await connection.executemany(
        """INSERT INTO projection_scope_transitions (
               id, source_id, previous_scope_json, target_scope_json, status, run_id,
               coverage, error, created_at, updated_at, completed_at
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                "transition-applied",
                SOURCE,
                '{"path":"/a"}',
                '{"path":"/b"}',
                "applied",
                "run-1",
                "complete_snapshot",
                None,
                "2026-10-01T00:00:00+00:00",
                "2026-10-01T01:00:00+00:00",
                "2026-10-01T01:00:00+00:00",
            ),
            (
                "transition-open",
                SOURCE,
                '{"path":"/b"}',
                '{"path":"/c"}',
                "failed",
                "run-2",
                "partial_projection",
                "one fetch failed",
                "2026-10-02T00:00:00+00:00",
                "2026-10-02T01:00:00+00:00",
                None,
            ),
        ],
    )
    await connection.commit()
    await connection.close()


async def test_migration_keeps_transitions_and_drops_their_coverage(tmp_path):
    path = str(tmp_path / "schema-124.db")
    await _database_at_schema_124(path)

    database = Database(path)
    await database.connect()
    try:
        columns = {
            str(row[1]) for row in await database.db.execute_fetchall("PRAGMA table_info(projection_scope_transitions)")
        }
        indexes = {
            str(row[1]) for row in await database.db.execute_fetchall("PRAGMA index_list(projection_scope_transitions)")
        }
        open_transition = await database.get_open_projection_scope_transition(SOURCE)
        transitions = await database.list_projection_scope_transitions(SOURCE)
        member_columns = [
            str(row[1])
            for row in await database.db.execute_fetchall("PRAGMA table_info(projection_scope_transition_members)")
        ]
        await database.db.execute(
            "UPDATE projection_scope_transitions SET status = 'superseded' WHERE id = 'transition-open'"
        )
        foreign_key_violations = await database.db.execute_fetchall("PRAGMA foreign_key_check")
    finally:
        await database.close()

    assert "coverage" not in columns
    assert {"predecessor_transition_id", "excluded_scopes_due"} <= columns
    assert "idx_projection_scope_transitions_source" in indexes
    assert member_columns == ["transition_id", "doc_id", "unit_revision_id"]
    assert open_transition == ProjectionScopeTransition(
        id="transition-open",
        source_id=SOURCE,
        previous_scope={"path": "/b"},
        target_scope={"path": "/c"},
        status=ProjectionScopeTransitionStatus.FAILED,
        run_id="run-2",
        error="one fetch failed",
        created_at="2026-10-02T00:00:00+00:00",
        updated_at="2026-10-02T01:00:00+00:00",
    )
    assert [(t.id, t.status, t.predecessor_transition_id, t.completed_at) for t in transitions] == [
        ("transition-open", ProjectionScopeTransitionStatus.FAILED, None, None),
        ("transition-applied", ProjectionScopeTransitionStatus.APPLIED, None, "2026-10-01T01:00:00+00:00"),
    ]
    assert foreign_key_violations == []


async def test_a_member_keeps_its_latest_revision_and_survives_failure(db: Database) -> None:
    transition = await db.create_projection_scope_transition(_transition("t1", "/new"))
    await db.start_projection_scope_transition(transition.id, run_id="run-1")
    first, first_input = _document("a.md", "first")
    second, second_input = _document("a.md", "second")
    other, other_input = _document("b.md", "other")

    await db.record_source_projection(first, unit_input=first_input, scope_transition_id="t1")
    await db.record_source_projection(other, unit_input=other_input, scope_transition_id="t1")
    failed = await db.fail_projection_scope_transition("t1", run_id="run-1", error="one fetch failed")
    await db.record_source_projection(second, unit_input=second_input, scope_transition_id="t1")

    assert failed.status is ProjectionScopeTransitionStatus.FAILED
    assert await db.list_projection_scope_transition_member_doc_ids("t1") == ("a.md", "b.md")
    assert await _member_revisions(db, "t1") == {
        "a.md": second_input.unit_revision_id,
        "b.md": other_input.unit_revision_id,
    }


async def test_an_unchanged_document_becomes_a_member_when_its_run_is_recorded_again(db: Database) -> None:
    await db.create_projection_scope_transition(_transition("t1", "/new"))
    projection, unit_input = _document("a.md", "same")
    await db.record_source_projection(projection, unit_input=unit_input)

    await db.record_source_projection(projection, unit_input=unit_input, scope_transition_id="t1")

    assert await db.list_projection_scope_transition_member_doc_ids("t1") == ("a.md",)


async def test_a_tombstone_records_no_member(db: Database) -> None:
    await db.create_projection_scope_transition(_transition("t1", "/new"))
    projection, unit_input = _document("a.md", "body")
    await db.record_source_projection(projection, unit_input=unit_input)
    tombstone = project_source_unit_tombstone(
        source_type="local_markdown",
        run_id="run-tombstone",
        source_unit=projection.source_units[0],
        prior_unit_revision=projection.source_unit_revisions[0],
        prior_observation_revisions={revision.observation_id: revision for revision in projection.observation_revisions},
        reason="removed_from_scope",
    )

    await db.record_source_projection(tombstone, scope_transition_id="t1")

    assert await db.list_projection_scope_transition_member_doc_ids("t1") == ()


async def test_a_member_needs_the_committed_unit_input(db: Database) -> None:
    await db.create_projection_scope_transition(_transition("t1", "/new"))
    projection, _unit_input = _document("a.md", "body")

    with pytest.raises(ValueError, match="stored input"):
        await db.record_source_projection(projection, scope_transition_id="t1")

    assert await db.get_source_projection(projection.run_id) is None


async def test_a_member_rolls_back_with_its_lifecycle_commit(db: Database) -> None:
    await db.create_projection_scope_transition(_transition("t1", "/new"))
    refused, refused_input = _document("a.md", "refused")
    committed, committed_input = _document("b.md", "committed")

    with pytest.raises(ValueError, match="observation revision stale guard"):
        await db.apply_source_projection_lifecycle(
            refused,
            _plan(refused, observation_revision_ids=("obsrev-superseded",)),
            unit_input=refused_input,
            scope_transition_id="t1",
        )
    await db.apply_source_projection_lifecycle(
        committed,
        _plan(committed),
        unit_input=committed_input,
        scope_transition_id="t1",
    )

    assert await db.get_source_projection(refused.run_id) is None
    assert await db.list_projection_scope_transition_member_doc_ids("t1") == ("b.md",)


async def test_completion_takes_no_coverage_and_deletes_the_members(db: Database) -> None:
    await db.create_projection_scope_transition(_transition("t1", "/new"))
    await db.start_projection_scope_transition("t1", run_id="run-1")
    projection, unit_input = _document("a.md", "body")
    await db.record_source_projection(projection, unit_input=unit_input, scope_transition_id="t1")

    with pytest.raises(ValueError, match="stale guard"):
        await db.complete_projection_scope_transition("t1", run_id="run-other")
    applied = await db.complete_projection_scope_transition("t1", run_id="run-1")
    later, later_input = _document("b.md", "later")

    assert applied.status is ProjectionScopeTransitionStatus.APPLIED
    assert applied.completed_at is not None
    assert await db.list_projection_scope_transition_member_doc_ids("t1") == ()
    with pytest.raises(ValueError, match="not open"):
        await db.record_source_projection(later, unit_input=later_input, scope_transition_id="t1")
    assert await db.get_source_projection(later.run_id) is None


async def test_supersession_closes_the_expected_transition_and_fences_its_members(db: Database) -> None:
    await _change_scope(db, _transition("t1", "/b"))
    projection, unit_input = _document("a.md", "body")
    await db.record_source_projection(projection, unit_input=unit_input, scope_transition_id="t1")
    successor = _transition("t2", "/c", predecessor="t1")

    await _change_scope(db, successor, expected_open_transition_id="t1")
    await _change_scope(db, successor, expected_open_transition_id="t1")
    late, late_input = _document("b.md", "late")

    transitions = {transition.id: transition for transition in await db.list_projection_scope_transitions(SOURCE)}
    assert transitions["t1"].status is ProjectionScopeTransitionStatus.SUPERSEDED
    assert transitions["t1"].completed_at is not None
    assert transitions["t2"].status is ProjectionScopeTransitionStatus.PENDING
    assert transitions["t2"].predecessor_transition_id == "t1"
    assert (await db.get_open_projection_scope_transition(SOURCE)).id == "t2"
    assert await db.list_projection_scope_transition_member_doc_ids("t1") == ()
    with pytest.raises(ValueError, match="not open"):
        await db.record_source_projection(late, unit_input=late_input, scope_transition_id="t1")
    assert await db.get_source_projection(late.run_id) is None


async def test_a_run_that_reached_the_excluded_scopes_marks_the_transition_until_it_closes(db: Database) -> None:
    async def due(transition_id: str) -> bool:
        transition = await db.get_projection_scope_transition(transition_id)
        assert transition is not None
        return transition.excluded_scopes_due

    await _change_scope(db, _transition("t1", "/b"))
    await db.start_projection_scope_transition("t1", run_id="run-1")
    await db.fail_projection_scope_transition("t1", run_id="run-1", error="excluded listings missing", excluded_scopes_due=True)
    marked = await due("t1")
    await db.start_projection_scope_transition("t1", run_id="run-2")
    kept_while_running = await due("t1")
    await db.fail_projection_scope_transition("t1", run_id="run-2", error="a listed Document failed")
    cleared_by_a_run_that_did_not_reach = await due("t1")
    await db.start_projection_scope_transition("t1", run_id="run-3")
    await db.fail_projection_scope_transition("t1", run_id="run-3", error="excluded listings missing", excluded_scopes_due=True)
    await db.start_projection_scope_transition("t1", run_id="run-4")
    await db.complete_projection_scope_transition("t1", run_id="run-4")
    await _change_scope(db, _transition("t2", "/c"))
    await db.start_projection_scope_transition("t2", run_id="run-5")
    await db.fail_projection_scope_transition("t2", run_id="run-5", error="excluded listings missing", excluded_scopes_due=True)
    await _change_scope(db, _transition("t3", "/d", predecessor="t2"), expected_open_transition_id="t2")

    assert (marked, kept_while_running, cleared_by_a_run_that_did_not_reach) == (True, True, False)
    assert await due("t1") is False
    assert await due("t2") is False


async def test_a_stale_expected_transition_changes_nothing(db: Database) -> None:
    await _change_scope(db, _transition("t1", "/b"))

    with pytest.raises(ProjectionScopeTransitionStale):
        await _change_scope(db, _transition("t2", "/c", predecessor="t0"), expected_open_transition_id="t0")
    # A change that read no open transition read stale facts too.
    with pytest.raises(ProjectionScopeTransitionStale):
        await _change_scope(db, _transition("t3", "/d"))
    with pytest.raises(ValueError, match="must name the superseded one") as misuse:
        await _change_scope(db, _transition("t4", "/e"), expected_open_transition_id="t1")
    assert not isinstance(misuse.value, ProjectionScopeTransitionStale)

    source = await db.get_source(SOURCE)
    assert source is not None
    assert source["config"] == {"path": "/b"}
    assert [transition.id for transition in await db.list_projection_scope_transitions(SOURCE)] == ["t1"]
    assert (await db.get_open_projection_scope_transition(SOURCE)).status is ProjectionScopeTransitionStatus.PENDING


async def test_a_return_to_the_last_applied_scope_supersedes_the_open_transition_and_opens_none(db: Database) -> None:
    await _change_scope(db, _transition("t1", "/b"))
    projection, unit_input = _document("a.md", "body")
    await db.record_source_projection(projection, unit_input=unit_input, scope_transition_id="t1")

    with pytest.raises(ProjectionScopeTransitionStale):
        await db.upsert_source(
            id=SOURCE, type="local_markdown", name=SOURCE, config_json=json.dumps({"path": "/docs"}),
            access_policy="workspace", owner_user_id="dev", expected_open_transition_id="t0",
        )
    await db.upsert_source(
        id=SOURCE, type="local_markdown", name=SOURCE, config_json=json.dumps({"path": "/docs"}),
        access_policy="workspace", owner_user_id="dev", expected_open_transition_id="t1",
    )

    [superseded] = await db.list_projection_scope_transitions(SOURCE)
    assert (superseded.id, superseded.status) == ("t1", ProjectionScopeTransitionStatus.SUPERSEDED)
    assert await db.get_open_projection_scope_transition(SOURCE) is None
    assert await db.list_projection_scope_transition_member_doc_ids("t1") == ()
    assert (await db.get_source(SOURCE))["config"] == {"path": "/docs"}


async def test_an_expected_transition_that_closed_meanwhile_is_stale(db: Database) -> None:
    await _change_scope(db, _transition("t1", "/b"))
    await db.start_projection_scope_transition("t1", run_id="run-1")
    await db.complete_projection_scope_transition("t1", run_id="run-1")

    with pytest.raises(ProjectionScopeTransitionStale):
        await _change_scope(db, _transition("t2", "/c", predecessor="t1"), expected_open_transition_id="t1")

    assert await db.get_open_projection_scope_transition(SOURCE) is None


async def test_a_tombstone_under_a_closed_transition_is_refused(db: Database) -> None:
    await db.create_projection_scope_transition(_transition("t1", "/new"))
    await db.start_projection_scope_transition("t1", run_id="run-1")
    await db.complete_projection_scope_transition("t1", run_id="run-1")
    projection, unit_input = _document("a.md", "body")
    await db.record_source_projection(projection, unit_input=unit_input)
    tombstone = project_source_unit_tombstone(
        source_type="local_markdown",
        run_id="run-tombstone",
        source_unit=projection.source_units[0],
        prior_unit_revision=projection.source_unit_revisions[0],
        prior_observation_revisions={revision.observation_id: revision for revision in projection.observation_revisions},
        reason="removed_from_scope",
    )

    with pytest.raises(ValueError, match="not open"):
        await db.record_source_projection(tombstone, scope_transition_id="t1")

    assert await db.get_source_projection(tombstone.run_id) is None


async def test_a_replayed_older_revision_leaves_the_member_at_the_current_one(db: Database) -> None:
    await db.create_projection_scope_transition(_transition("t1", "/new"))
    first, first_input = _document("a.md", "first")
    second, second_input = _document("a.md", "second")
    await db.record_source_projection(first, unit_input=first_input)
    await db.record_source_projection(second, unit_input=second_input, scope_transition_id="t1")

    await db.record_source_projection(first, unit_input=first_input, scope_transition_id="t1")

    assert await _member_revisions(db, "t1") == {"a.md": second_input.unit_revision_id}


async def test_a_transition_of_another_source_is_refused(db: Database) -> None:
    await db.create_projection_scope_transition(replace(_transition("t-other", "/x"), source_id=OTHER_SOURCE))
    projection, unit_input = _document("a.md", "body")

    with pytest.raises(ValueError, match="not open"):
        await db.record_source_projection(projection, unit_input=unit_input, scope_transition_id="t-other")
    with pytest.raises(ProjectionScopeTransitionStale):
        await _change_scope(
            db, _transition("t2", "/c", predecessor="t-other"), expected_open_transition_id="t-other"
        )

    assert await db.get_source_projection(projection.run_id) is None
    assert await db.get_open_projection_scope_transition(SOURCE) is None
    assert (await db.get_open_projection_scope_transition(OTHER_SOURCE)).id == "t-other"
