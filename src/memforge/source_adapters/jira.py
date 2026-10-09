"""Jira-owned immutable record fields and comparison semantics."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from memforge.source_adapters.contracts import CanonicalRecordField, CanonicalRecordSchema
from memforge.source_adapters.jira_html import LEGACY_RENDERED_HTML_FORMAT, RENDERED_HTML_FORMAT
from memforge.source_projection import EvidenceCoordinateSpace, EvidenceRepresentationProfile
from memforge.source_time import latest_source_time, reported_source_time


CANONICAL_RECORD_SCHEMAS: Mapping[tuple[str, int], CanonicalRecordSchema] = {
    ("jira-issue-core", 1): CanonicalRecordSchema(
        name="jira-issue-core",
        version=1,
        fields=(
            CanonicalRecordField("/summary"),
            CanonicalRecordField("/description", nested_profile="markdown-structural"),
            CanonicalRecordField(
                "/status",
                comparison_keys=("id", "key", "name", "value"),
            ),
            CanonicalRecordField(
                "/priority",
                comparison_keys=("id", "key", "name", "value"),
            ),
            CanonicalRecordField(
                "/assignee",
                comparison_keys=(
                    "accountId",
                    "id",
                    "key",
                    "name",
                    "displayName",
                ),
            ),
            CanonicalRecordField("/labels"),
            CanonicalRecordField(
                "/resolution",
                comparison_keys=("id", "key", "name", "value"),
            ),
        ),
    ),
    ("jira-comment", 1): CanonicalRecordSchema(
        name="jira-comment",
        version=1,
        fields=(CanonicalRecordField("/body", nested_profile="markdown-structural"),),
    ),
    ("jira-changelog", 1): CanonicalRecordSchema(
        name="jira-changelog",
        version=1,
        model_interpretation=(
            "fromString is the previous field value; toString is the new value. "
            "field identifies the changed field and created records the change event. "
            "The event time alone does not establish when a business rule becomes effective."
        ),
        fields=(
            CanonicalRecordField("/created", contextual=True),
            CanonicalRecordField("/items/*/field", contextual=True),
            CanonicalRecordField("/items/*/fromString", nested_profile="markdown-structural"),
            CanonicalRecordField("/items/*/toString", nested_profile="markdown-structural"),
        ),
    ),
}


CORE_FIELDS = ("summary", "description", "status", "priority", "assignee", "labels", "resolution")
OPERATIONAL_HISTORY_FIELDS = frozenset({
    "assignee", "due date", "duedate", "fix version", "fix version/s", "fixversion",
    "labels", "priority", "rank", "resolution", "sprint", "status",
})

ISSUE_SCHEMA = replace(
    CANONICAL_RECORD_SCHEMAS[("jira-issue-core", 1)],
    version=2,
    fields=tuple(
        CanonicalRecordField("/description", text_format=LEGACY_RENDERED_HTML_FORMAT)
        if field.json_pointer == "/description" else field
        for field in CANONICAL_RECORD_SCHEMAS[("jira-issue-core", 1)].fields
    ),
)
COMMENT_SCHEMA = CanonicalRecordSchema(
    name="jira-comment", version=2,
    fields=(CanonicalRecordField("/body", text_format=LEGACY_RENDERED_HTML_FORMAT),),
)
CHANGELOG_SCHEMA = replace(
    CANONICAL_RECORD_SCHEMAS[("jira-changelog", 1)],
    version=2,
    fields=tuple(
        replace(field, nested_profile="plain-text") if field.nested_profile else field
        for field in CANONICAL_RECORD_SCHEMAS[("jira-changelog", 1)].fields
    ),
    model_interpretation=(
        "fromString is the previous literal native field value; toString is the new value. "
        "These historical strings have no provider-rendered HTML view and are not declared Markdown. "
        "field identifies the changed field and created records the change event. "
        "The event time alone does not establish when a business rule becomes effective."
    ),
)
CANONICAL_RECORD_SCHEMAS = {
    **CANONICAL_RECORD_SCHEMAS,
    **{(schema.name, schema.version): schema for schema in (ISSUE_SCHEMA, COMMENT_SCHEMA, CHANGELOG_SCHEMA)},
}


ISSUE_SCHEMA = replace(ISSUE_SCHEMA, version=3,
    fields=(*ISSUE_SCHEMA.fields, CanonicalRecordField("/issue_key", contextual=True), CanonicalRecordField("/issue_id", contextual=True)),
    framing_fields=(("/issue_key", "Issue"), ("/issue_id", "Issue ID")))
COMMENT_SCHEMA = replace(COMMENT_SCHEMA, version=3,
    fields=(*COMMENT_SCHEMA.fields, *(CanonicalRecordField(pointer, contextual=True) for pointer in
            ("/issue_key", "/author", "/created", "/updated"))),
    framing_fields=(("/issue_key", "Issue"), ("/author", "Comment author"),
                    ("/created", "Comment created"), ("/updated", "Comment updated")))
CHANGELOG_SCHEMA = replace(CHANGELOG_SCHEMA, version=3,
    fields=(*CHANGELOG_SCHEMA.fields, CanonicalRecordField("/issue_key", contextual=True), CanonicalRecordField("/author", contextual=True)),
    framing_fields=(("/issue_key", "Issue"), ("/author", "Change author")))
CANONICAL_RECORD_SCHEMAS = {**CANONICAL_RECORD_SCHEMAS,
    **{(schema.name, schema.version): schema for schema in (ISSUE_SCHEMA, COMMENT_SCHEMA, CHANGELOG_SCHEMA)}}

# New schemas make event roles explicit; older schemas retain their pinned views.
ISSUE_SCHEMA = replace(ISSUE_SCHEMA, version=4,
    fields=tuple(replace(field, label=field.json_pointer[1:].replace("_", " ").capitalize(),
                         presentation_keys=field.comparison_keys)
                 for field in ISSUE_SCHEMA.fields))
COMMENT_SCHEMA = replace(COMMENT_SCHEMA, version=4,
    fields=tuple(field for field in COMMENT_SCHEMA.fields if field.json_pointer != "/author")
        + (CanonicalRecordField("/author/displayName", contextual=True),
           CanonicalRecordField("/author/key", contextual=True)),
    framing_fields=(("/issue_key", "Issue"), ("/author/displayName", "Comment author"),
                    ("/author/key", "Author ID"), ("/created", "Comment created"),
                    ("/updated", "Comment updated")))
CHANGELOG_SCHEMA = replace(CHANGELOG_SCHEMA, version=4,
    fields=tuple(
        replace(field, label="Previous value" if field.json_pointer.endswith("/fromString") else "New value",
                framing_fields=(("/items/*/field", "Changed field"),),
                comparison_pointer=field.json_pointer)
        if field.json_pointer in {"/items/*/fromString", "/items/*/toString"}
        else replace(field, label="Changed field", comparison_pointer=field.json_pointer,
                     framing_fields=(("/items/*/field", "Changed field"),))
        if field.json_pointer == "/items/*/field" else field
        for field in CHANGELOG_SCHEMA.fields if field.json_pointer != "/author"
    ) + (CanonicalRecordField("/author/displayName", contextual=True),
         CanonicalRecordField("/author/key", contextual=True)),
    framing_fields=(("/issue_key", "Issue"), ("/created", "Change recorded"),
                    ("/author/displayName", "Change author"), ("/author/key", "Author ID")))
CANONICAL_RECORD_SCHEMAS = {**CANONICAL_RECORD_SCHEMAS,
    **{(schema.name, schema.version): schema for schema in (ISSUE_SCHEMA, COMMENT_SCHEMA, CHANGELOG_SCHEMA)}}


# Current Jira formatting semantics; schemas 2–4 retain their pinned interpretation.
ISSUE_SCHEMA, COMMENT_SCHEMA = (
    replace(schema, version=5, fields=tuple(
        replace(field, text_format=RENDERED_HTML_FORMAT) if field.text_format else field
        for field in schema.fields))
    for schema in (ISSUE_SCHEMA, COMMENT_SCHEMA)
)
CANONICAL_RECORD_SCHEMAS = {**CANONICAL_RECORD_SCHEMAS,
    **{(schema.name, schema.version): schema for schema in (ISSUE_SCHEMA, COMMENT_SCHEMA)}}


def _profile(schema_name: str, version: int) -> EvidenceRepresentationProfile:
    return EvidenceRepresentationProfile(
        "canonical-record", 1, EvidenceCoordinateSpace.UNICODE_SCALAR,
        schema_name=schema_name, schema_version=version,
    )


CURRENT_OBSERVATION_PROFILES = {
    ("jira", observation_type): _profile(schema.name, schema.version)
    for observation_type, schema in (
        ("issue_core", ISSUE_SCHEMA), ("comment", COMMENT_SCHEMA),
        ("changelog", CHANGELOG_SCHEMA),
    )
}
LEGACY_OBSERVATION_PROFILES = {
    key: replace(profile, schema_version=1) for key, profile in CURRENT_OBSERVATION_PROFILES.items()
}


def _current_rendering(native: object, rendered: object, field_name: str) -> str | None:
    if native is None or native == "":
        if rendered is not None and rendered != "":
            raise ValueError(f"Jira {field_name} rendering has no matching native value")
        return native
    if not isinstance(native, str):
        raise ValueError(f"Jira {field_name} requires a string native value")
    if not isinstance(rendered, str) or not rendered.strip():
        raise ValueError(f"Jira {field_name} requires its snapshot's explicit provider-rendered HTML")
    # Validate before persistence: unsupported syntax cannot become an apparently
    # complete revision whose meaningful field is silently omitted downstream.
    parsed = RENDERED_HTML_FORMAT.parse(rendered)
    if native.strip() and not parsed.fragments:
        raise ValueError(f"Jira {field_name} rendering has no selectable view of its native value")
    return rendered


def issue_record(data: Mapping[str, object]) -> dict[str, object]:
    """Keep native and rendered values from the same returned issue snapshot."""
    fields = data.get("fields")
    if not isinstance(fields, Mapping):
        raise ValueError("Jira issue record requires its fetched fields")
    rendered = data.get("renderedFields")
    native = fields.get("description")
    description = rendered.get("description") if isinstance(rendered, Mapping) else None
    return {
        **{name: fields.get(name) for name in CORE_FIELDS},
        "description": _current_rendering(native, description, "description"),
        "native_description": native,
        "issue_key": data.get("key"), "issue_id": data.get("id"),
        "representation": f"{ISSUE_SCHEMA.name}:{ISSUE_SCHEMA.version}",
    }


def comment_record(comment: Mapping[str, object], *, issue_key: str | None = None) -> dict[str, object]:
    """The comment body's explicit renderedBody expansion and original value."""
    native = comment.get("body")
    return {
        "body": _current_rendering(native, comment.get("renderedBody"), "comment body"),
        "native_body": native,
        "issue_key": issue_key, "author": comment.get("author"),
        "created": comment.get("created"), "updated": comment.get("updated"),
        "attachments": comment.get("attachments"),
        "representation": f"{COMMENT_SCHEMA.name}:{COMMENT_SCHEMA.version}",
    }


def changelog_record(history: Mapping[str, object], *, issue_key: str | None = None) -> dict[str, object]:
    """Retain prior/new strings as recorded, without borrowing current rendering."""
    items = history.get("items")
    if not isinstance(items, list):
        raise ValueError("Jira changelog record requires its fetched change items")
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError("Jira changelog has a malformed change item")
        if any(item.get(name) is not None and not isinstance(item[name], str) for name in ("fromString", "toString")):
            raise ValueError("Jira changelog values require literal native strings")
    return {**history, "issue_key": issue_key, "representation": f"{CHANGELOG_SCHEMA.name}:{CHANGELOG_SCHEMA.version}"}


def core_revised_at(
    fields: Mapping[str, object], histories: list[object], *, changelog_complete: bool,
) -> str | None:
    """Latest core-field change, or creation when complete history has none.

    A truncated changelog cannot attest this time. The issue updated timestamp
    also moves for comments and other fields and is not a core-field timestamp.
    """
    if not changelog_complete:
        return None
    core_change_times = [
        history.get("created") for history in histories
        if isinstance(history, Mapping) and any(
            isinstance(entry, Mapping)
            and str(entry.get("fieldId") or entry.get("field") or "").strip().lower() in CORE_FIELDS
            for entry in (history.get("items") if isinstance(history.get("items"), list) else [])
        )
    ]
    return latest_source_time(core_change_times) if core_change_times else reported_source_time(fields.get("created"))


def changelog_semantic_class(history: Mapping[str, object]) -> str:
    """Jira field ownership distinguishes attachments, operations and domain changes."""
    items = history.get("items")
    fields = {
        " ".join(str(item.get("field") or "").strip().lower().split())
        for item in (items if isinstance(items, list) else []) if isinstance(item, Mapping)
    } - {""}
    if fields and fields <= {"attachment"}:
        return "attachment_event"
    if fields and fields <= OPERATIONAL_HISTORY_FIELDS:
        return "operational_transition"
    return "domain_transition"


def comment_requires_rendering(comment: Mapping[str, object]) -> bool:
    """Whether an inline native comment needs the explicit renderedBody expansion."""
    return bool(comment.get("body")) and not isinstance(comment.get("renderedBody"), str)

def _provider_name(value: object) -> object:
    """Read the supplied display name of a Jira field object."""
    return value.get("name") if isinstance(value, Mapping) else value


def project_native(*, source_id, item, native, normalized):
    """Build provider identity, immutable Observations, relations and coverage."""
    from memforge.source_projection import ProjectionCoverage, SourceRelationType
    from memforge.source_adapters.contracts import _ObservationInput, _NativeProjection, _unit_title, _canonical_json
    from memforge.genes.jira_gene import LOCAL_AGENT_JIRA_PACKAGE_KIND

    data = native if isinstance(native, dict) else {}
    if data.get("package_kind") == LOCAL_AGENT_JIRA_PACKAGE_KIND and isinstance(data.get("raw_payload"), dict):
        data = data["raw_payload"]
    from memforge.local_agent.jira_contract import validate_jira_observation_identities

    validate_jira_observation_identities(data)
    fields = data.get("fields") if isinstance(data.get("fields"), dict) else {}
    raw_issue_id = data.get("id") or item.extra.get("issue_id")
    issue_id = str(raw_issue_id or "").strip()
    if not issue_id.isdigit():
        raise ValueError("jira projection requires immutable numeric issue id")
    issue_key = str(data.get("key") or item.extra.get("issue_key") or item.item_id)

    core_value = issue_record(data)
    changelog = data.get("changelog") if isinstance(data.get("changelog"), dict) else {}
    raw_histories = changelog.get("histories", [])
    histories = raw_histories if isinstance(raw_histories, list) else []
    changelog_total = changelog.get("total")
    changelog_complete = not data.get("_changelog_truncated") and not (
        isinstance(changelog_total, int) and changelog_total > len(histories)
    )
    inputs = [
        _ObservationInput(
            "issue_core",
            f"{issue_id}:core",
            _canonical_json(core_value),
            core_value,
            {"issue_key": issue_key},
            core_revised_at(
                fields,
                histories,
                # A payload without a changelog says nothing about core changes.
                changelog_complete=changelog_complete and isinstance(data.get("changelog"), dict),
            ),
        )
    ]
    relations: list[tuple[SourceRelationType, str, str, str | None, Mapping[str, object]]] = []
    previous_key = f"{issue_id}:core"
    comments = data.get("_comments") if isinstance(data.get("_comments"), list) else []
    for comment in comments:
        if not isinstance(comment, dict):
            continue
        comment_id = str(comment["id"])
        semantic_comment = comment_record(comment, issue_key=issue_key)
        inputs.append(
            _ObservationInput(
                "comment",
                comment_id,
                _canonical_json(semantic_comment),
                semantic_comment,
                {"issue_key": issue_key},
                str(comment.get("updated") or comment.get("created") or "") or None,
                {"claim_evidence_scope": "atomic"},
            )
        )
        relations.append((SourceRelationType.PRECEDES, previous_key, comment_id, None, {}))
        previous_key = comment_id
    for history in histories:
        if not isinstance(history, dict):
            continue
        history_id = str(history["id"])
        semantic_history = changelog_record(history, issue_key=issue_key)
        inputs.append(
            _ObservationInput(
                "changelog",
                history_id,
                _canonical_json(semantic_history),
                semantic_history,
                {"issue_key": issue_key},
                str(history.get("created") or "") or None,
                {
                    "semantic_class": changelog_semantic_class(
                        history
                    )
                },
            )
        )
    coverage = (
        ProjectionCoverage.PARTIAL_PROJECTION
        if data.get("_comments_truncated") or not changelog_complete
        else ProjectionCoverage.COMPLETE_SNAPSHOT
    )
    return _NativeProjection(
        unit_type="jira_issue",
        provider_key=issue_id,
        observations=tuple(inputs),
        relations=tuple(relations),
        coverage=coverage,
        locator={"issue_id": issue_id, "issue_key": issue_key, "url": item.source_url},
        title=_unit_title(
            "Jira issue",
            ("Key", issue_key),
            ("Type", _provider_name(fields.get("issuetype"))),
            ("Summary", fields.get("summary")),
        ),
    )
