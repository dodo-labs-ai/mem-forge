"""Notion: every page under configured root pages and databases, read with an internal connection token."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from memforge.source_adapters import notion as adapter
from memforge.source_types.config import SourceConfigPolicy, secret_replaced
from memforge.source_types.descriptor import SourceTypeDescriptor

_ROOT_FIELDS = (("root_pages", "Root pages", True), ("databases", "Databases", False))


def _canonical_scope(scope: dict[str, object]) -> dict[str, object]:
    """Each root as its 32-hex id, sorted, so a pasted link and its id are one scope."""
    from memforge.genes.notion_api import notion_link_id

    for field, _label, open_page in _ROOT_FIELDS:
        if field in scope:
            scope[field] = sorted({notion_link_id(value, open_page=open_page) or value for value in scope[field]})
    return scope


def _validate_config(config: dict[str, Any], existing_config: Mapping[str, Any]) -> None:
    """At least one root, every root a Notion id, and no id in both lists.

    These are the rules the configuration schema cannot state, and each
    refusal names its field: a value that is not an id names the list that
    holds it, and a missing root or an id in both lists names
    ``root_pages``. The token is a required field of the schema, which the
    shared check enforces.
    """
    from memforge.genes.notion_api import notion_id
    from memforge.genes.toolkit.config import FieldIssue, SourceConfigInvalid, list_config

    roots: dict[str, set[str]] = {}
    issues: list[FieldIssue] = []
    for field, label, _open_page in _ROOT_FIELDS:
        values = list_config(config.get(field))
        unread = [value for value in values if notion_id(value) is None]
        if unread:
            message = f"Notion {label} holds {unread[0]!r}, which is not a Notion page or database link or ID"
            issues.append(FieldIssue(field, message))
        roots[field] = {str(notion_id(value)) for value in values if value not in unread}
    if issues:
        raise SourceConfigInvalid(issues)
    if not roots["root_pages"] and not roots["databases"]:
        raise SourceConfigInvalid(
            [FieldIssue("root_pages", "Notion Root pages or Databases must name at least one page or database")]
        )
    shared = sorted(roots["root_pages"] & roots["databases"])
    if shared:
        raise SourceConfigInvalid(
            [FieldIssue("root_pages", f"Notion Root pages and Databases both hold {shared[0]}; name each root in one list")]
        )


def _credentials_changed(config: dict[str, Any], existing_config: Mapping[str, Any]) -> bool:
    return secret_replaced("token", config, existing_config)


DESCRIPTOR = SourceTypeDescriptor(
    source_type="notion",
    gene="memforge.genes.notion_gene:NotionGene",
    unit_noun="page",
    scope_fields=("root_pages", "databases"),
    canonical_scope=_canonical_scope,
    project_native=adapter.project_native,
    evidence_profiles=adapter.CURRENT_OBSERVATION_PROFILES,
    record_schemas=adapter.CANONICAL_RECORD_SCHEMAS,
    config=SourceConfigPolicy(validate=_validate_config, credentials_changed=_credentials_changed),
)
