"""Cross-project affinity penalty and scope-mode behavior in ranking."""

from __future__ import annotations

import pytest

from memforge.models import SHARED_PROJECT_KEY, UNSORTED_PROJECT_KEY
from memforge.retrieval.search import (
    CROSS_PROJECT_PENALTY,
    W_RECENCY_DEFAULT,
    W_RRF_DEFAULT,
    _affinity_penalty,
)
from memforge.storage.adapters.context import AccessScope


def _scope(*, mode: str = "project-first", active: str | None = "PAY") -> AccessScope:
    return AccessScope(
        user_id="dev",
        include_private=False,
        allowed_statuses=("active",),
        active_project=active,
        scope_mode=mode,
    )


def test_shared_never_penalized():
    assert _affinity_penalty(SHARED_PROJECT_KEY, _scope()) == 0.0


def test_active_project_never_penalized():
    assert _affinity_penalty("PAY", _scope(active="PAY")) == 0.0


def test_unsorted_takes_full_penalty_in_project_first():
    """UNSORTED is NOT exempt: unmapped knowledge must not rank as if it
    were team-wide."""
    assert _affinity_penalty(UNSORTED_PROJECT_KEY, _scope()) == CROSS_PROJECT_PENALTY


def test_other_project_takes_full_penalty_in_project_first():
    assert _affinity_penalty("RISK", _scope(active="PAY")) == CROSS_PROJECT_PENALTY


def test_workspace_mode_no_penalty():
    assert _affinity_penalty("RISK", _scope(mode="workspace", active="PAY")) == 0.0
    assert _affinity_penalty(UNSORTED_PROJECT_KEY, _scope(mode="workspace", active="PAY")) == 0.0


def test_no_active_project_no_penalty():
    """A search request without active_project (legacy default and per-id
    readers) gets no penalty: without a frame of reference every project is
    treated equally so the existing flat ranking stays unchanged."""
    no_active = _scope(active=None)
    assert _affinity_penalty("PAY", no_active) == 0.0
    assert _affinity_penalty("RISK", no_active) == 0.0
    assert _affinity_penalty(UNSORTED_PROJECT_KEY, no_active) == 0.0
    assert _affinity_penalty(SHARED_PROJECT_KEY, no_active) == 0.0


def test_null_project_key_takes_penalty():
    """A NULL project_key (legacy row) is treated as cross-project and
    penalized when an active_project is set."""
    assert _affinity_penalty(None, _scope(active="PAY")) == CROSS_PROJECT_PENALTY


def test_ranking_weights_form_a_valid_distribution():
    """The ranking weights sum to 1.0 and the cross-project penalty stays in
    (0, 1) so a penalized candidate is never lifted above the unpenalized
    active-project candidate."""
    from math import isclose

    assert isclose(W_RRF_DEFAULT + W_RECENCY_DEFAULT, 1.0)
    assert 0.0 < CROSS_PROJECT_PENALTY < 1.0


@pytest.mark.asyncio
async def test_project_mode_upstream_prunes_other_projects(tmp_path):
    """`scope_mode == "project"` must prune at the predicate layer:
    only memories whose project_key is the active_project or SHARED
    survive the SQL/predicate filter. UNSORTED and other projects are
    NOT visible (a hard exclusion, not a penalty).
    """
    from memforge.models import (
        Memory,
        Visibility,
        content_hash,
    )
    from memforge.storage.adapters.sqlite import build_sqlite_adapters
    from memforge.storage.database import Database

    db = Database(str(tmp_path / "p4mode.db"))
    await db.connect()
    try:
        for mid, key in (
            ("m-pay", "PAY"),
            ("m-risk", "RISK"),
            ("m-shared", SHARED_PROJECT_KEY),
            ("m-unsorted", UNSORTED_PROJECT_KEY),
        ):
            await db.insert_memory(
                Memory(
                    id=mid,
                    memory_type="fact",
                    content=mid,
                    content_hash=content_hash(mid),
                    visibility=Visibility.WORKSPACE.value,
                    owner_user_id=None,
                    project_key=key,
                    tags=[],
                )
            )
        adapters = build_sqlite_adapters(db, memory_collection=None)
        scope = _scope(mode="project", active="PAY")
        ids = ["m-pay", "m-risk", "m-shared", "m-unsorted"]
        visible = await adapters.relational.filter_visible_ids(ids, scope)
        # active_project + SHARED in; RISK and UNSORTED out.
        assert visible == {"m-pay", "m-shared"}
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_project_first_mode_keeps_all_projects_visible(tmp_path):
    """`scope_mode == "project-first"` (the default) leaves the predicate
    open: cross-project candidates remain visible and are weighted, not
    pruned."""
    from memforge.models import (
        Memory,
        Visibility,
        content_hash,
    )
    from memforge.storage.adapters.sqlite import build_sqlite_adapters
    from memforge.storage.database import Database

    db = Database(str(tmp_path / "p4mode_first.db"))
    await db.connect()
    try:
        for mid, key in (
            ("m-pay", "PAY"),
            ("m-risk", "RISK"),
            ("m-shared", SHARED_PROJECT_KEY),
            ("m-unsorted", UNSORTED_PROJECT_KEY),
        ):
            await db.insert_memory(
                Memory(
                    id=mid,
                    memory_type="fact",
                    content=mid,
                    content_hash=content_hash(mid),
                    visibility=Visibility.WORKSPACE.value,
                    owner_user_id=None,
                    project_key=key,
                    tags=[],
                )
            )
        adapters = build_sqlite_adapters(db, memory_collection=None)
        scope = _scope(mode="project-first", active="PAY")
        ids = ["m-pay", "m-risk", "m-shared", "m-unsorted"]
        visible = await adapters.relational.filter_visible_ids(ids, scope)
        assert visible == {"m-pay", "m-risk", "m-shared", "m-unsorted"}
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_fetch_ranking_metadata_returns_updated_at_and_project_key(tmp_path):
    """The ranking-metadata fetch returns both columns in a single mapping
    so the ranker can compute recency and the affinity penalty without a
    second roundtrip."""
    from memforge.models import Memory, Visibility, content_hash
    from memforge.storage.adapters.sqlite import build_sqlite_adapters
    from memforge.storage.database import Database

    db = Database(str(tmp_path / "p4meta.db"))
    await db.connect()
    try:
        await db.insert_memory(
            Memory(
                id="m1",
                memory_type="fact",
                content="x",
                content_hash=content_hash("x"),
                visibility=Visibility.WORKSPACE.value,
                owner_user_id=None,
                project_key="PAY",
                tags=[],
            )
        )
        adapters = build_sqlite_adapters(db, memory_collection=None)
        meta = await adapters.relational.fetch_ranking_metadata(["m1", "missing"])
        assert "m1" in meta
        assert meta["m1"]["project_key"] == "PAY"
        assert meta["m1"]["updated_at"] is not None
        assert "missing" not in meta
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_project_first_visibility_survives_real_cross_project_rows(tmp_path):
    """`project-first` mode must keep cross-project candidates visible
    even when those projects have real rows in the `projects` table.

    With PAY and RISK both registered as projects, a `project-first`
    search active on PAY still sees RISK candidates so the ranker can
    apply the affinity penalty.
    """
    from memforge.models import (
        Memory,
        Visibility,
        content_hash,
    )
    from memforge.storage.adapters.sqlite import build_sqlite_adapters
    from memforge.storage.database import Database

    db = Database(str(tmp_path / "p4mode_visibility.db"))
    await db.connect()
    try:
        await db.create_project(key="PAY", name="Pay")
        await db.create_project(key="RISK", name="Risk")
        for mid, key in (("m-pay", "PAY"), ("m-risk", "RISK")):
            await db.insert_memory(
                Memory(
                    id=mid,
                    memory_type="fact",
                    content=mid,
                    content_hash=content_hash(mid),
                    visibility=Visibility.WORKSPACE.value,
                    owner_user_id=None,
                    project_key=key,
                    tags=[],
                )
            )
        adapters = build_sqlite_adapters(db, memory_collection=None)
        # In project-first mode the workspace branch admits every workspace
        # row regardless of project_key; the ranker applies the affinity
        # penalty for cross-project hits later.
        scope = AccessScope(
            user_id="dev",
            include_private=False,
            allowed_statuses=("active",),
            active_project="PAY",
            scope_mode="project-first",
        )
        visible = await adapters.relational.filter_visible_ids(
            ["m-pay", "m-risk"],
            scope,
        )
        assert visible == {"m-pay", "m-risk"}
    finally:
        await db.close()
