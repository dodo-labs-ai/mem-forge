"""Single decision point for the `project_key` written on a memory.

Sources declare a `project_binding` (top-level JSON column on `sources`,
two modes: `fixed` or `by_field`). At extraction, sync.py and the agent-
session intake call `resolve_project_key` rather than reading raw fields,
so the rule lives in one place: a hit maps to its key, a miss resolves
to the binding `default` (which is `UNSORTED` unless the admin set it
to a specific key). Unmapped values never mint a new project row.

Deleting a project releases every binding that names it
(`released_project_bindings`), so no Source keeps writing to a key that no
longer exists.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterable, Mapping

from memforge.models import UNSORTED_PROJECT_KEY

__all__ = [
    "binding_names_project",
    "binding_without_project",
    "released_project_bindings",
    "resolve_project_key",
]

logger = logging.getLogger(__name__)


def resolve_project_key(
    binding: Mapping[str, Any] | None,
    *,
    item_field_value: str | None,
    repo: str | None,
    workspace: str | None,
) -> str:
    """Return the project_key for the memory being written.

    `binding` is the source's `project_binding` JSON or None for a
    legacy/unbound source (resolves to UNSORTED). The caller passes the
    raw values the binding might read: `item_field_value` (the
    `documents.space_or_project` for doc sources), `repo` (for agent
    sources), and `workspace` (intentionally unused by the resolver,
    kept in the signature for symmetry with how callers gather their
    inputs; no junk-key minting from `Path(workspace).name`).
    """
    if not binding:
        return UNSORTED_PROJECT_KEY

    mode = binding.get("mode")
    if mode == "fixed":
        return str(binding.get("project_key") or UNSORTED_PROJECT_KEY)

    if mode == "by_field":
        field = binding.get("field")
        # Doc sources read the documents.space_or_project value the gene
        # populated; agent sources read `repo`. The caller decides which
        # by passing the right argument under `item_field_value` for doc
        # sources or by leaving item_field_value=None and supplying repo.
        observed: str | None
        if field == "repo":
            observed = repo
        else:
            observed = item_field_value
        if observed is None:
            # The `repo` (or other field) is absent. Resolve to the
            # binding default. Never derive from `workspace` basename.
            return str(binding.get("default") or UNSORTED_PROJECT_KEY)

        mapped = (binding.get("map") or {}).get(observed)
        if mapped:
            return str(mapped)
        return str(binding.get("default") or UNSORTED_PROJECT_KEY)

    # Unknown modes resolve to the explicit unassigned bucket.
    return UNSORTED_PROJECT_KEY


def binding_names_project(binding: Mapping[str, Any] | None, project_key: str) -> bool:
    """Whether any memory the binding routes can resolve to `project_key`."""
    if not binding:
        return False
    mode = binding.get("mode")
    if mode == "fixed":
        return binding.get("project_key") == project_key
    if mode == "by_field":
        return binding.get("default") == project_key or project_key in (binding.get("map") or {}).values()
    return False


def binding_without_project(binding: Mapping[str, Any], project_key: str) -> dict[str, Any] | None:
    """The binding, which names `project_key`, once that project no longer exists.

    A fixed binding to the project is removed, so the Source becomes
    unbound. A field binding stays in place: it drops the mappings that
    point to the project and falls back to UNSORTED where its default
    pointed to the project. Either way the Source stops writing to the
    deleted key.
    """
    if binding.get("mode") == "fixed":
        return None
    released = dict(binding)
    if binding.get("map"):
        released["map"] = {value: key for value, key in binding["map"].items() if key != project_key}
    if released.get("default") == project_key:
        released["default"] = UNSORTED_PROJECT_KEY
    return released


def released_project_bindings(
    stored_bindings: Iterable[tuple[str, Any]],
    project_key: str,
) -> list[tuple[str, dict[str, Any] | None]]:
    """The Sources that deleting `project_key` releases, each with the binding it keeps.

    `stored_bindings` pairs the id of every Source that still writes
    memories (retired Sources write none and are left as they are) with
    its stored `project_binding`, as JSON text or already decoded. Every
    store answers both "what would this deletion release" and "release
    it" through this one rule, so the count an admin confirms is the set
    the deletion changes.

    A stored binding that is not a JSON object cannot route any memory to
    `project_key`, so it is skipped and left exactly as stored: one
    unreadable row neither blocks every project deletion in the
    workspace nor gets overwritten by a deletion that cannot read it.
    """
    released: list[tuple[str, dict[str, Any] | None]] = []
    for source_id, stored in stored_bindings:
        binding = _decoded_binding(source_id, stored)
        if binding is None or not binding_names_project(binding, project_key):
            continue
        released.append((source_id, binding_without_project(binding, project_key)))
    return released


def _decoded_binding(source_id: str, stored: Any) -> Mapping[str, Any] | None:
    decoded = stored
    if isinstance(stored, str):
        try:
            decoded = json.loads(stored)
        except json.JSONDecodeError:
            decoded = None
    if isinstance(decoded, Mapping):
        return decoded
    if stored is not None:
        logger.warning(
            "Source %s has a project_binding that is not a JSON object; project deletion leaves it", source_id
        )
    return None
