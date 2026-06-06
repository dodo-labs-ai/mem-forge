"""Shared fixtures, factory protocol, and builders for the contract suite.

Every concrete adapter family points the suite at one factory that hands
back a freshly bound ``ContractAdapters`` bundle plus a teardown coroutine.
The suite itself never imports a concrete adapter; the only adapter-specific
code lives in the factory the consumer registers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Awaitable, Callable

from memforge.models import Memory, Visibility, content_hash
from memforge.storage.adapters.context import AccessScope, LOCAL_DEV_USER_ID
from memforge.storage.adapters.protocols import (
    KeywordSearch,
    RelationalStore,
    VectorStore,
)


@dataclass(frozen=True)
class ContractAdapters:
    """The three adapter handles a contract suite reads against.

    A keyword channel that is a thin facade over the relational store
    (the SQLite/FTS5 case) and a separate keyword channel both satisfy
    this bundle: the contract suite never assumes shared state.
    """

    relational: RelationalStore
    keyword: KeywordSearch
    vector: VectorStore


@dataclass(frozen=True)
class FactoryResult:
    """A bundle a factory hands the suite, plus the teardown that releases
    every resource it took (db handles, temp dirs, in-memory collections).
    Awaiting the teardown must leave behind no state the next test could
    observe.
    """

    adapters: ContractAdapters
    teardown: Callable[[], Awaitable[None]]


AdaptersFactory = Callable[[], Awaitable[FactoryResult]]


# ---------------------------------------------------------------------------
# Builders: every test in the suite uses these so adapters never see unrelated
# field drift across helpers.
# ---------------------------------------------------------------------------


def make_scope(
    *,
    statuses: tuple[str, ...] = ("active",),
    user_id: str = LOCAL_DEV_USER_ID,
    include_private: bool = False,
    active_project: str | None = None,
    scope_mode: str = "project-first",
) -> AccessScope:
    """Build the per-request caller context with safe defaults.

    ``project-first`` is the default mode in the live API, so it is the
    default here too; tests that need ``project`` or ``workspace`` opt in
    explicitly.
    """
    return AccessScope(
        user_id=user_id,
        include_private=include_private,
        allowed_statuses=statuses,
        active_project=active_project,
        scope_mode=scope_mode,  # type: ignore[arg-type]
    )


def make_memory(
    memory_id: str,
    *,
    content: str | None = None,
    memory_type: str = "fact",
    status: str = "active",
    visibility: str = Visibility.WORKSPACE.value,
    owner_user_id: str | None = None,
    project_key: str | None = None,
) -> Memory:
    """Build a Memory with stamped timestamps and a derived content hash.

    Tests pass an id, type, and visibility; everything else is derived so
    the row is well-formed without per-test boilerplate. The
    owner/visibility invariant is enforced by the relational store on
    insert, so callers that pass a private visibility must also pass an
    ``owner_user_id`` (the suite covers both branches deliberately).
    """
    text = content if content is not None else f"content for {memory_id}"
    now = datetime.now(timezone.utc)
    return Memory(
        id=memory_id,
        memory_type=memory_type,
        content=text,
        content_hash=content_hash(text),
        visibility=visibility,
        owner_user_id=owner_user_id,
        project_key=project_key,
        confidence=0.9,
        created_at=now,
        updated_at=now,
        status=status,
    )


# A small dummy embedding the contract suite uses for vector upserts. The
# vector contract never asserts on ranking quality, only on round-trip and
# scope/visibility filtering, so a fixed-length zero-ish vector is enough.
DEFAULT_EMBEDDING: tuple[float, ...] = (0.1, 0.2, 0.3)


def make_vector_metadata(
    memory: Memory,
    *,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    """Project a Memory into the metadata fields the vector channel filters
    on (visibility, status, project_key, owner_user_id, memory_type).

    Extra keys override the derived ones, so a test that wants to place a
    row in a non-matching status or project does so without constructing
    the whole dict by hand.
    """
    metadata: dict[str, object] = {
        "visibility": memory.visibility,
        "status": memory.status,
        "memory_type": memory.memory_type,
        "project_key": memory.project_key or "UNSORTED",
        "owner_user_id": memory.owner_user_id,
    }
    if extra:
        metadata.update(extra)
    return metadata


__all__ = [
    "AdaptersFactory",
    "ContractAdapters",
    "DEFAULT_EMBEDDING",
    "FactoryResult",
    "make_memory",
    "make_scope",
    "make_vector_metadata",
]
