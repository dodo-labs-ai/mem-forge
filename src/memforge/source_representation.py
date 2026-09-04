"""Adapter-owned Evidence representation declarations for Source Projection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from memforge.source_projection import (
    EvidenceCoordinateSpace,
    EvidenceRepresentationProfile,
)


MARKDOWN_STRUCTURAL_PROFILE = EvidenceRepresentationProfile(
    name="markdown-structural",
    version=1,
    coordinate_space=EvidenceCoordinateSpace.UNICODE_SCALAR,
)
BINARY_ARTIFACT_PROFILE = EvidenceRepresentationProfile(
    name="binary-artifact",
    version=1,
    coordinate_space=EvidenceCoordinateSpace.WHOLE_ARTIFACT,
)
PLAIN_TEXT_PROFILE = EvidenceRepresentationProfile(
    name="plain-text",
    version=1,
    coordinate_space=EvidenceCoordinateSpace.UNICODE_SCALAR,
)


@dataclass(frozen=True, slots=True)
class EvidenceProfileBackfillReport:
    """Exact result of classifying legacy Observation Revisions."""

    scanned_revision_count: int
    backfilled_revision_count: int
    unresolved_revision_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CanonicalRecordField:
    json_pointer: str
    nested_profile: str | None = None
    comparison_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.nested_profile not in {None, "markdown-structural", "plain-text"}:
            raise ValueError("unsupported nested canonical-record text profile")


@dataclass(frozen=True, slots=True)
class CanonicalRecordSchema:
    name: str
    version: int
    fields: tuple[CanonicalRecordField, ...]
    tombstone_pointer: str | None = None


@dataclass(frozen=True, slots=True)
class EvidenceRepresentationContract:
    profile: EvidenceRepresentationProfile
    canonical_schema: CanonicalRecordSchema | None = None

    def __post_init__(self) -> None:
        if (self.profile.name == "canonical-record") != (self.canonical_schema is not None):
            raise ValueError("canonical schema ownership must match the representation profile")


def _canonical_record_profile(schema_name: str) -> EvidenceRepresentationProfile:
    return EvidenceRepresentationProfile(
        name="canonical-record",
        version=1,
        coordinate_space=EvidenceCoordinateSpace.UNICODE_SCALAR,
        schema_name=schema_name,
        schema_version=1,
    )


_REPRESENTATION_CONTRACTS: Mapping[tuple[str, str], EvidenceRepresentationProfile] = {
    ("confluence", "page_body"): MARKDOWN_STRUCTURAL_PROFILE,
    ("jira", "issue_core"): _canonical_record_profile("jira-issue-core"),
    ("jira", "comment"): _canonical_record_profile("jira-comment"),
    ("jira", "changelog"): _canonical_record_profile("jira-changelog"),
    ("github_repo", "file_content"): MARKDOWN_STRUCTURAL_PROFILE,
    ("github_pages", "page_content"): MARKDOWN_STRUCTURAL_PROFILE,
    ("local_markdown", "file_content"): MARKDOWN_STRUCTURAL_PROFILE,
    ("teams", "message"): _canonical_record_profile("teams-message"),
    ("agent_session", "session_summary"): MARKDOWN_STRUCTURAL_PROFILE,
    ("agent_session", "agent_concept"): MARKDOWN_STRUCTURAL_PROFILE,
}

_CANONICAL_RECORD_SCHEMAS: Mapping[tuple[str, int], CanonicalRecordSchema] = {
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
        fields=(CanonicalRecordField(""),),
    ),
    ("teams-message", 1): CanonicalRecordSchema(
        name="teams-message",
        version=1,
        fields=(CanonicalRecordField("/content", nested_profile="markdown-structural"),),
        tombstone_pointer="/deleted",
    ),
}


def canonical_field_comparison_value(
    field: CanonicalRecordField,
    value: object,
) -> object:
    """Return the schema-owned stable business value used for authority diff."""

    if not field.comparison_keys or not isinstance(value, Mapping):
        return value
    return tuple(
        (key, value[key])
        for key in field.comparison_keys
        if key in value
    )


def _representation_contract(profile: EvidenceRepresentationProfile) -> EvidenceRepresentationContract:
    schema = (
        _CANONICAL_RECORD_SCHEMAS.get((profile.schema_name or "", profile.schema_version or 0))
        if profile.name == "canonical-record"
        else None
    )
    return EvidenceRepresentationContract(profile=profile, canonical_schema=schema)


_SUPPORTED_REPRESENTATION_CONTRACTS: Mapping[
    EvidenceRepresentationProfile,
    EvidenceRepresentationContract,
] = {
    profile: _representation_contract(profile)
    for profile in {
        MARKDOWN_STRUCTURAL_PROFILE,
        BINARY_ARTIFACT_PROFILE,
        PLAIN_TEXT_PROFILE,
        *_REPRESENTATION_CONTRACTS.values(),
    }
}


def representation_contract_for_profile(
    profile: EvidenceRepresentationProfile | None,
) -> EvidenceRepresentationContract | None:
    if profile is None:
        return None
    return _SUPPORTED_REPRESENTATION_CONTRACTS.get(profile)


def representation_profile_for_observation_contract(
    *,
    source_type: str,
    observation_type: str,
) -> EvidenceRepresentationProfile | None:
    """Declare a stored adapter contract without inspecting content or MIME."""

    if observation_type == "binary_artifact":
        return BINARY_ARTIFACT_PROFILE
    if observation_type == "document_content":
        # The extension-safe projection fallback is explicitly normalized Markdown.
        return MARKDOWN_STRUCTURAL_PROFILE
    return _REPRESENTATION_CONTRACTS.get((source_type, observation_type))
