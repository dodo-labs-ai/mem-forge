"""Single decision point for the `project_key` written on a memory.

Sources declare a `project_binding` (top-level JSON column on `sources`,
two modes: `fixed` or `by_field`). At extraction, sync.py and the agent-
session intake call `resolve_project_key` rather than reading raw fields,
so the rule lives in one place: a hit maps to its key, a miss resolves
to the binding `default` (which is `UNSORTED` unless the admin set it
to a specific key). Unmapped values never mint a new project row.
"""

from __future__ import annotations

from typing import Any, Mapping

from memforge.models import UNSORTED_PROJECT_KEY

__all__ = ["resolve_project_key"]


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
