"""Adapter-owned Evidence representation declarations for Source Projection."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

from memforge.source_artifacts import SOURCE_ARTIFACT_OBSERVATION_TYPE
from memforge.source_adapters.contracts import (
    CanonicalRecordField,
    CanonicalRecordSchema,
    EvidenceRepresentationContract,
)
from memforge.source_adapters.jira import (
    CANONICAL_RECORD_SCHEMAS as JIRA_RECORD_SCHEMAS,
    CURRENT_OBSERVATION_PROFILES as JIRA_OBSERVATION_PROFILES,
    LEGACY_OBSERVATION_PROFILES as JIRA_LEGACY_PROFILES,
)
from memforge.source_adapters.teams import CANONICAL_RECORD_SCHEMAS as TEAMS_RECORD_SCHEMAS
from memforge.source_adapters.confluence import (
    CANONICAL_RECORD_SCHEMAS as CONFLUENCE_RECORD_SCHEMAS,
    LEGACY_OBSERVATION_PROFILES as CONFLUENCE_LEGACY_PROFILES,
)
from memforge.source_projection import (
    EvidenceCoordinateSpace,
    EvidenceRepresentationProfile,
    SourceObservationRevision,
    SourceProjection,
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


def _canonical_record_profile(schema_name: str, schema_version: int = 1) -> EvidenceRepresentationProfile:
    return EvidenceRepresentationProfile(
        name="canonical-record",
        version=1,
        coordinate_space=EvidenceCoordinateSpace.UNICODE_SCALAR,
        schema_name=schema_name,
        schema_version=schema_version,
    )


_REPRESENTATION_CONTRACTS: Mapping[tuple[str, str], EvidenceRepresentationProfile] = {
    ("confluence", "page_body"): _canonical_record_profile("confluence-page-storage"),
    **JIRA_OBSERVATION_PROFILES,
    ("github_repo", "file_content"): MARKDOWN_STRUCTURAL_PROFILE,
    ("github_pages", "page_content"): MARKDOWN_STRUCTURAL_PROFILE,
    ("local_markdown", "file_content"): MARKDOWN_STRUCTURAL_PROFILE,
    ("teams", "message"): _canonical_record_profile("teams-message", 2),
    ("agent_session", "session_summary"): MARKDOWN_STRUCTURAL_PROFILE,
    ("agent_session", "agent_concept"): MARKDOWN_STRUCTURAL_PROFILE,
}

_CANONICAL_RECORD_SCHEMAS: Mapping[tuple[str, int], CanonicalRecordSchema] = {
    **JIRA_RECORD_SCHEMAS,
    **TEAMS_RECORD_SCHEMAS,
    **CONFLUENCE_RECORD_SCHEMAS,
}

_LEGACY_OBSERVATION_PROFILES = {
    **CONFLUENCE_LEGACY_PROFILES,
    **JIRA_LEGACY_PROFILES,
    ("teams", "message"): _canonical_record_profile("teams-message", 1),
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
        *(_canonical_record_profile(name, version) for name, version in _CANONICAL_RECORD_SCHEMAS),
    }
}


def representation_contract_for_profile(
    profile: EvidenceRepresentationProfile | None,
) -> EvidenceRepresentationContract | None:
    if profile is None:
        return None
    return _SUPPORTED_REPRESENTATION_CONTRACTS.get(profile)


def in_current_representation(revision: SourceObservationRevision) -> bool:
    """Whether a stored Observation revision is content the adapters still project.

    Revisions are compared with a new projection only when both sides are in the
    current representation. A stored revision in a profile no adapter projects is
    outside it: no later Unit revision carries it, it is no change of the Unit,
    and Evidence on it remains unavailable for correspondence. A revision stored
    before profiles were recorded counts as current.
    """

    profile = revision.evidence_profile
    return profile is None or profile in _SUPPORTED_REPRESENTATION_CONTRACTS


def current_representation_of(projection: SourceProjection) -> SourceProjection:
    """A stored projection as it is read today, without its revisions outside the current representation.

    A new projection is always in the current representation. A stored one read as
    the revision itself, such as the committed revision a reprocess preview reads
    or a replayed case, may still hold a retired revision; it is read without it.
    Its Unit revision keeps its id: this is how that revision is read, not a new one.
    Its Revision Deltas keep their axes and name no retired revision or Observation,
    so a retired revision is neither changed nor added content of the read projection.
    """

    retired = {r.id for r in projection.observation_revisions if not in_current_representation(r)}
    if not retired:
        return projection
    revisions = tuple(r for r in projection.observation_revisions if r.id not in retired)
    kept = {r.observation_id for r in revisions}
    left = {r.observation_id for r in projection.observation_revisions if r.id in retired} - kept
    return replace(
        projection,
        observations=tuple(o for o in projection.observations if o.id not in left),
        observation_revisions=revisions,
        source_unit_revisions=tuple(
            replace(unit, observation_revision_ids=tuple(i for i in unit.observation_revision_ids if i not in retired))
            for unit in projection.source_unit_revisions
        ),
        deltas=tuple(
            replace(
                delta,
                changed_anchors=tuple(a for a in delta.changed_anchors if a.observation_revision_id not in retired),
                added_observation_ids=tuple(i for i in delta.added_observation_ids if i not in left),
                fragment_mappings=tuple(m for m in delta.fragment_mappings if m.current_revision_id not in retired),
            )
            for delta in projection.deltas
        ),
        carried_observation_revision_ids=tuple(
            i for i in projection.carried_observation_revision_ids if i not in retired
        ),
    )


def representation_profile_for_observation_contract(
    *,
    source_type: str,
    observation_type: str,
) -> EvidenceRepresentationProfile | None:
    """Declare a stored adapter contract without inspecting content or MIME."""

    if observation_type == SOURCE_ARTIFACT_OBSERVATION_TYPE:
        return BINARY_ARTIFACT_PROFILE
    if observation_type == "document_content":
        # The extension-safe projection fallback is explicitly normalized Markdown.
        return MARKDOWN_STRUCTURAL_PROFILE
    return _REPRESENTATION_CONTRACTS.get((source_type, observation_type))


def legacy_representation_profile_for_observation_contract(
    *, source_type: str, observation_type: str,
) -> EvidenceRepresentationProfile | None:
    """A known pre-profile writing contract, independent of today's adapter."""
    return _LEGACY_OBSERVATION_PROFILES.get(
        (source_type, observation_type),
        representation_profile_for_observation_contract(
            source_type=source_type, observation_type=observation_type,
        ),
    )
