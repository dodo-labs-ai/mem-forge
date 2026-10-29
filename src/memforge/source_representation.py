"""Adapter-owned Evidence representation declarations for Source Projection."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

from memforge.source_artifacts import SOURCE_ARTIFACT_OBSERVATION_TYPE
from memforge.source_adapters.contracts import (
    BINARY_ARTIFACT_PROFILE,
    MARKDOWN_STRUCTURAL_PROFILE,
    PLAIN_TEXT_PROFILE,
    UNIT_TITLE_OBSERVATION_PROFILE,
    CanonicalRecordField,
    CanonicalRecordSchema,
    EvidenceRepresentationContract,
    canonical_record_profile,
)
from memforge.source_projection import (
    EvidenceRepresentationProfile,
    SourceObservationRevision,
    SourceProjection,
)
from memforge.source_types import registered_source_types


# Profiles outside the current representation whose content every reading still
# shows. ``shown_as_reading_context`` says why a profile has to be named here.
_PROFILES_SHOWN_AS_READING_CONTEXT = frozenset({UNIT_TITLE_OBSERVATION_PROFILE})


@dataclass(frozen=True, slots=True)
class EvidenceProfileBackfillReport:
    """Exact result of classifying legacy Observation Revisions."""

    scanned_revision_count: int
    backfilled_revision_count: int
    unresolved_revision_ids: tuple[str, ...]


_REPRESENTATION_CONTRACTS: Mapping[tuple[str, str], EvidenceRepresentationProfile] = {
    (descriptor.source_type, observation_type): profile
    for descriptor in registered_source_types()
    for observation_type, profile in descriptor.evidence_profiles.items()
}

_CANONICAL_RECORD_SCHEMAS: Mapping[tuple[str, int], CanonicalRecordSchema] = {
    key: schema
    for descriptor in registered_source_types()
    for key, schema in descriptor.record_schemas.items()
}

_LEGACY_OBSERVATION_PROFILES: Mapping[tuple[str, str], EvidenceRepresentationProfile] = {
    (descriptor.source_type, observation_type): profile
    for descriptor in registered_source_types()
    for observation_type, profile in descriptor.legacy_evidence_profiles.items()
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
        *(canonical_record_profile(name, version) for name, version in _CANONICAL_RECORD_SCHEMAS),
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


def shown_as_reading_context(revision: SourceObservationRevision) -> bool:
    """Whether a stored revision outside the current representation holds content every reading still shows.

    A profile is outside the current representation for one of three reasons,
    and the profile alone does not say which:

    - its content is shown to every reader another way, as the Unit Title is by
      the projection that carries it;
    - its content is no longer collected, so nothing shows it;
    - this build does not know a profile that another build wrote, as after a
      rollback, or lost a profile it should support.

    Evidence on the first kind can leave its Support: the claim keeps what it
    rests on and loses a citation of text its readers still see. Evidence on the
    other two cannot. The claim would lose its ground, or sound Evidence would be
    discarded, and nothing would report either. Such a Support is kept unresolved
    instead, which a corrected build undoes; a citation left out stays out.

    The program cannot tell the first kind from the others, so a profile counts
    only when it is named in ``_PROFILES_SHOWN_AS_READING_CONTEXT``. Name one
    there after establishing that every model reading and every Memory read
    still shows its content.
    """

    return revision.evidence_profile in _PROFILES_SHOWN_AS_READING_CONTEXT


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
    record_schema: str | None = None,
) -> EvidenceRepresentationProfile | None:
    """The profile an Observation of one adapter contract is read under, without inspecting content or MIME.

    It is the profile the contract declares today, unless the payload names the
    canonical record schema it was collected under (``record_schema``, as
    ``name:version``). The payload is then read under that schema, which must be
    one of the contract's ``declared_representation_profiles``; any other
    schema cannot be read, and raises ``ValueError``.
    """

    if observation_type == SOURCE_ARTIFACT_OBSERVATION_TYPE:
        return BINARY_ARTIFACT_PROFILE
    if observation_type == "document_content":
        # The extension-safe projection fallback is explicitly normalized Markdown.
        return MARKDOWN_STRUCTURAL_PROFILE
    declared = _REPRESENTATION_CONTRACTS.get((source_type, observation_type))
    if record_schema is None or declared is None:
        return declared
    name, _, version = record_schema.rpartition(":")
    profile = canonical_record_profile(name, int(version)) if name and version.isdecimal() else None
    if profile not in declared_representation_profiles(source_type=source_type, observation_type=observation_type):
        raise ValueError(
            f"{source_type}/{observation_type} cannot be read under record schema {record_schema!r}: "
            "it is not a registered version of the schema the contract declares"
        )
    return profile


def declared_representation_profiles(
    *, source_type: str, observation_type: str,
) -> frozenset[EvidenceRepresentationProfile]:
    """Every profile a new Observation Revision of one adapter contract may be stored under.

    That is the profile the contract declares and, for a canonical record, every
    registered version of its schema, since a stored payload may name any of
    them as the schema it was collected under. The shared projector alone
    decides which one a revision is read under
    (``representation_profile_for_observation_contract``); storage checks only
    that it is registered for the contract. A contract without a declaration
    has none.
    """

    declared = representation_profile_for_observation_contract(
        source_type=source_type, observation_type=observation_type,
    )
    if declared is None:
        return frozenset()
    if declared.schema_name is None:
        return frozenset({declared})
    return frozenset(
        canonical_record_profile(name, version)
        for name, version in _CANONICAL_RECORD_SCHEMAS
        if name == declared.schema_name
    )


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
