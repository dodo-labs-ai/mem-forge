# Cross-datastore isolation is a deployment-time property of the bound
# adapter handle, not of the core predicate; this test pack only exercises
# the predicate.
"""Lifecycle and access ride together but stay independent.

`include_superseded=True` widens the lifecycle window, surfacing superseded
workspace rows. It does NOT widen access: another user's superseded private
row stays hidden. Access and lifecycle are orthogonal in the predicate.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from memforge.memory.lifecycle import allowed_search_statuses
from memforge.models import (
    Memory,
    SHARED_PROJECT_KEY,
    Visibility,
    content_hash,
)
from memforge.retrieval.access_predicate import is_visible
from memforge.storage.adapters.context import AccessScope
from memforge.storage.adapters.sqlite import build_sqlite_adapters
from memforge.storage.database import Database


WORKSPACE = Visibility.WORKSPACE.value
PRIVATE = Visibility.PRIVATE.value


def _mem(mid: str, *, visibility=WORKSPACE, owner=None, status="active") -> Memory:
    return Memory(
        id=mid,
        memory_type="fact",
        content="argocd deploys things",
        content_hash=content_hash("argocd" + mid),
        visibility=visibility,
        owner_user_id=owner,
        project_key=SHARED_PROJECT_KEY,
        status=status,
    )


def _scope(*, include_private: bool, include_superseded: bool) -> AccessScope:
    return AccessScope(
        user_id="u-1",
        include_private=include_private,
        allowed_statuses=allowed_search_statuses(include_superseded),
        active_project=None,
        scope_mode="project-first",
    )


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "lifecycle.db"))
    await database.connect()
    yield database
    await database.close()


async def _seed(db: Database) -> None:
    """Three rows: an active workspace baseline, a superseded workspace row,
    and a superseded U2-private row at the same content."""
    await db.insert_memory(_mem("ws-active"))
    # Insert as active, then mark superseded (the normal write path validates
    # the visibility/owner invariant; we flip status afterward).
    await db.insert_memory(_mem("ws-old"))
    await db.insert_memory(_mem("priv-u2-old", visibility=PRIVATE, owner="u-2"))
    now = datetime.now(timezone.utc).isoformat()
    await db.db.execute(
        "UPDATE memories SET status = 'superseded', superseded_at = ? WHERE id IN ('ws-old', 'priv-u2-old')",
        (now,),
    )
    await db.db.commit()


def test_is_visible_lifecycle_orthogonal_to_access():
    # Workspace superseded row: hidden when status filter excludes superseded,
    # visible when it includes superseded.
    ws_super = {
        "status": "superseded",
        "visibility": WORKSPACE,
        "owner_user_id": None,
        "project_key": SHARED_PROJECT_KEY,
    }
    assert is_visible(ws_super, _scope(include_private=False, include_superseded=False)) is False
    assert is_visible(ws_super, _scope(include_private=False, include_superseded=True)) is True

    # Other-user private superseded row: include_superseded MUST NOT widen
    # access. It stays hidden under TEAM and under PERSONALIZED for u-1.
    priv_super = {
        "status": "superseded",
        "visibility": PRIVATE,
        "owner_user_id": "u-2",
        "project_key": SHARED_PROJECT_KEY,
    }
    assert is_visible(priv_super, _scope(include_private=False, include_superseded=True)) is False
    assert is_visible(priv_super, _scope(include_private=True, include_superseded=True)) is False

    # The caller's own private superseded row IS surfaced when both are on.
    own_super = {
        "status": "superseded",
        "visibility": PRIVATE,
        "owner_user_id": "u-1",
        "project_key": SHARED_PROJECT_KEY,
    }
    assert is_visible(own_super, _scope(include_private=True, include_superseded=True)) is True
    assert is_visible(own_super, _scope(include_private=True, include_superseded=False)) is False


@pytest.mark.asyncio
async def test_keyword_channel_lifecycle_does_not_widen_access(db):
    await _seed(db)
    adapters = build_sqlite_adapters(db, memory_collection=None)

    # TEAM with include_superseded=True: ws-old surfaces, priv-u2-old does not.
    hits = await adapters.keyword.search(
        "argocd",
        _scope(include_private=False, include_superseded=True),
        memory_types=None,
        limit=10,
    )
    ids = {mid for mid, _ in hits}
    assert "ws-active" in ids
    assert "ws-old" in ids
    assert "priv-u2-old" not in ids

    # PERSONALIZED with include_superseded=True from u-1: u-2's private
    # superseded row still must not surface.
    hits = await adapters.keyword.search(
        "argocd",
        _scope(include_private=True, include_superseded=True),
        memory_types=None,
        limit=10,
    )
    ids = {mid for mid, _ in hits}
    assert "ws-active" in ids
    assert "ws-old" in ids
    assert "priv-u2-old" not in ids


@pytest.mark.asyncio
async def test_filter_visible_ids_lifecycle_does_not_widen_access(db):
    await _seed(db)
    adapters = build_sqlite_adapters(db, memory_collection=None)
    survivors = await adapters.relational.filter_visible_ids(
        ["ws-active", "ws-old", "priv-u2-old"],
        _scope(include_private=True, include_superseded=True),
    )
    assert survivors == {"ws-active", "ws-old"}
