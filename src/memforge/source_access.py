"""Source-level access policy and discoverability rules.

Source access is the configuration boundary. Memory visibility remains the
materialized query boundary, but it is always derived from an explicit Source
policy rather than inferred from a source type.
"""

from __future__ import annotations

from enum import StrEnum
from collections.abc import Iterable
from typing import Any, Mapping

from memforge.models import Visibility


class SourceAccessPolicy(StrEnum):
    PRIVATE = Visibility.PRIVATE.value
    WORKSPACE = Visibility.WORKSPACE.value


class SourceAccessState(StrEnum):
    ACTIVE = "active"
    CHANGING = "changing"
    ORPHANED_PRIVATE = "orphaned_private"


def infer_legacy_source_access(
    *,
    source_id: str,
    source_type: str,
    provenance_owner_user_id: str | None,
    memory_access: Iterable[tuple[str, str | None]],
) -> tuple[SourceAccessPolicy, str]:
    """Infer one legacy Source's explicit policy or reject ambiguity.

    This function is deliberately shared by SQLite and external adapters so a
    migration cannot preserve different effective access in different stores.
    """

    owner = str(provenance_owner_user_id or "").strip()
    if not owner:
        raise ValueError(f"cannot migrate Source access without an owner: {source_id}")
    if source_type == "agent_session":
        return SourceAccessPolicy.PRIVATE, owner

    rows = tuple(
        (str(visibility or "").strip(), str(owner_user_id or "").strip())
        for visibility, owner_user_id in memory_access
    )
    if not rows:
        return SourceAccessPolicy.WORKSPACE, owner

    policies = {visibility for visibility, _ in rows}
    private_owners = {
        owner_user_id
        for visibility, owner_user_id in rows
        if visibility == SourceAccessPolicy.PRIVATE.value
    }
    if policies == {SourceAccessPolicy.WORKSPACE.value}:
        return SourceAccessPolicy.WORKSPACE, owner
    if (
        policies == {SourceAccessPolicy.PRIVATE.value}
        and len(private_owners) == 1
        and "" not in private_owners
    ):
        return SourceAccessPolicy.PRIVATE, next(iter(private_owners))
    raise ValueError(
        "cannot migrate Source access with mixed or ambiguous "
        f"Memory visibility: {source_id}"
    )


def source_access_policy(source: Mapping[str, Any]) -> SourceAccessPolicy:
    try:
        policy = SourceAccessPolicy(str(source["access_policy"]))
    except (KeyError, ValueError) as exc:
        raise ValueError("source access_policy must be private or workspace") from exc
    source_owner_user_id(source)
    return policy


def source_access_state(source: Mapping[str, Any]) -> SourceAccessState:
    try:
        return SourceAccessState(str(source["access_state"]))
    except (KeyError, ValueError) as exc:
        raise ValueError(
            "source access_state must be active, changing, or orphaned_private"
        ) from exc


def source_owner_user_id(source: Mapping[str, Any]) -> str:
    owner = str(source.get("owner_user_id") or "").strip()
    if not owner:
        raise ValueError("source owner_user_id is required")
    return owner


def source_is_discoverable(source: Mapping[str, Any], *, viewer_id: str) -> bool:
    owner = source_owner_user_id(source)
    policy = source_access_policy(source)
    state = source_access_state(source)

    if state is SourceAccessState.ORPHANED_PRIVATE:
        return False
    if state is SourceAccessState.CHANGING:
        return viewer_id == owner
    if policy is SourceAccessPolicy.PRIVATE:
        return viewer_id == owner
    return True


def memory_visibility_for_source(
    source: Mapping[str, Any],
) -> tuple[str, str | None]:
    owner = source_owner_user_id(source)
    policy = source_access_policy(source)
    state = source_access_state(source)
    if state is SourceAccessState.ORPHANED_PRIVATE:
        raise ValueError("orphaned private sources cannot write memories")
    if state is SourceAccessState.CHANGING:
        return (Visibility.PRIVATE.value, owner)
    if policy is SourceAccessPolicy.PRIVATE:
        return (Visibility.PRIVATE.value, owner)
    return (Visibility.WORKSPACE.value, None)


async def memory_visibility_for_document(
    database: Any,
    *,
    doc_id: str,
) -> tuple[str, str | None]:
    document = await database.get_document(doc_id)
    if document is None:
        raise ValueError(f"document {doc_id!r} has no Source access context")
    if isinstance(document, Mapping):
        source_id = str(document.get("source") or "").strip()
    else:
        source_id = str(getattr(document, "source", "") or "").strip()
    if not source_id:
        raise ValueError(f"document {doc_id!r} has no source_id")
    source = await database.get_source(source_id)
    if source is None:
        raise ValueError(f"Source {source_id!r} has no access policy")
    return memory_visibility_for_source(source)
