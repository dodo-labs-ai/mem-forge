"""Provider adapters that project fetched Gene items into stable source lineage.

Genes remain responsible for authentication and provider I/O.  This module is
the provider-specific end of the lifecycle seam: it turns native payloads into
provider-neutral Source Units, Observations, immutable revisions, relations,
and deltas.  Downstream extraction and lifecycle code never branches on these
source types.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Mapping

from memforge.github_repo_utils import build_github_repo_doc_id
from memforge.local_agent.source_contract import TEAMS_ROLLING_RETENTION_PRESETS
from memforge.models import ContentItem, NormalizedContent, RawContent
from memforge.source_projection import (
    AnchorKind,
    DeltaAxis,
    ProjectionCoverage,
    ProjectionEnvelope,
    ProjectionScopeAttestation,
    RevisionDelta,
    SourceAnchor,
    SourceObservation,
    SourceObservationRevision,
    SourceProjection,
    ProjectionScopeTransition,
    SourceRelation,
    SourceRelationType,
    SourceUnit,
    SourceUnitRevision,
)
from memforge.source_time import (
    SOURCE_UPDATED_AT_KEY,
    latest_source_time,
    source_time_iso,
)
from memforge.source_representation import (
    UNIT_TITLE_OBSERVATION_TYPE,
    representation_profile_for_observation_contract,
)
from memforge.source_projection_config import (
    projection_access_fingerprint,
    projection_scope_fingerprint,
)
from memforge.source_artifacts import (
    SOURCE_ARTIFACT_OBSERVATION_TYPE,
    StoredSourceArtifact,
    source_artifact_observation_metadata,
    source_artifact_observation_provider_key,
)


BUILTIN_SPECIALIZED_SOURCE_TYPES = frozenset(
    {
        "confluence",
        "jira",
        "github_repo",
        "github_pages",
        "local_markdown",
        "teams",
        "agent_session",
    }
)

# The Jira fields that make up an issue's core Observation.
_JIRA_CORE_FIELDS = (
    "summary",
    "description",
    "status",
    "priority",
    "assignee",
    "labels",
    "resolution",
)

_JIRA_OPERATIONAL_HISTORY_FIELDS = frozenset(
    {
        "assignee",
        "due date",
        "duedate",
        "fix version",
        "fix version/s",
        "fixversion",
        "labels",
        "priority",
        "rank",
        "resolution",
        "sprint",
        "status",
    }
)

def source_run_projection_coverage(
    *,
    source_type: str | None = None,
    incremental: bool,
    authoritative_snapshot: bool,
    discovery_complete: bool = False,
) -> ProjectionCoverage:
    """Declare absence authority for a complete source discovery run."""

    del source_type
    if authoritative_snapshot:
        return ProjectionCoverage.COMPLETE_SNAPSHOT
    if incremental:
        return ProjectionCoverage.PARTIAL_PROJECTION
    if discovery_complete:
        return ProjectionCoverage.COMPLETE_SNAPSHOT
    # Absence authority comes from run-scoped provider evidence, never from a
    # source-type allowlist. Extension genes and conversational sources remain
    # partial until they explicitly prove enumeration completion.
    return ProjectionCoverage.PARTIAL_PROJECTION


@dataclass(frozen=True, slots=True)
class _ObservationInput:
    """One Observation as the provider payload gives it.

    ``observed_at`` is the source's own time for this content (design 0.9):
    when the provider last changed it, or ``None`` when the provider records
    no such time. It is never a discovery, fetch, submission or sync time.
    """

    observation_type: str
    provider_key: str
    content: str
    semantic_value: object
    locator: Mapping[str, object]
    observed_at: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    semantic_hash: str | None = None


_REVISION_SEMANTIC_METADATA_KEYS = ("claim_evidence_scope",)
# The Unit Title's provider key; native provider keys never start with "$".
_UNIT_TITLE_PROVIDER_KEY = "$unit_identity"


@dataclass(frozen=True, slots=True)
class _UnitTitle:
    """The provider's human-facing name of one Source Unit: its kind and named values.

    Adapters supply only values present in the provider payload; absent values
    are omitted, never guessed.
    """

    kind: str
    fields: tuple[tuple[str, str], ...]

    @classmethod
    def of(cls, kind: str, *fields: tuple[str, object]) -> _UnitTitle:
        present = tuple(
            (name, " ".join(str(value).split()))
            for name, value in fields
            if value is not None and str(value).strip()
        )
        return cls(kind=kind, fields=present)

    def observation(self) -> _ObservationInput:
        return _ObservationInput(
            UNIT_TITLE_OBSERVATION_TYPE,
            _UNIT_TITLE_PROVIDER_KEY,
            "\n".join((self.kind, *(f"{name}: {value}" for name, value in self.fields))),
            {"kind": self.kind, "fields": [list(field) for field in self.fields]},
            {},
        )


@dataclass(frozen=True, slots=True)
class _NativeProjection:
    """One provider payload as a Source Unit, its Observations and its Unit Title."""

    unit_type: str
    provider_key: str
    observations: tuple[_ObservationInput, ...]
    relations: tuple[tuple[SourceRelationType, str, str, str | None, Mapping[str, object]], ...]
    coverage: ProjectionCoverage
    locator: Mapping[str, object]
    # None only when the payload tombstones the whole Unit.
    title: _UnitTitle | None


def _observation_semantic_hash(value: _ObservationInput) -> str:
    if value.semantic_hash is not None:
        return value.semantic_hash
    inference_contract = {
        key: value.metadata[key]
        for key in _REVISION_SEMANTIC_METADATA_KEYS
        if key in value.metadata
    }
    if not inference_contract:
        return _canonical_hash(value.semantic_value)
    return _canonical_hash(
        {
            "semantic_value": value.semantic_value,
            "inference_contract": inference_contract,
        }
    )


class GeneSourceProjectionAdapter:
    """Unified adapter used by every Gene-backed document/conversation source."""

    async def project(self, envelope: ProjectionEnvelope) -> SourceProjection:
        request = envelope.request
        return project_source_item(
            source_id=request.source_id,
            source_type=request.source_type,
            run_id=request.run_id,
            item=envelope.item,
            raw=envelope.raw,
            normalized=envelope.normalized,
            artifacts=envelope.artifacts,
            scope=request.scope,
            access_context=request.access_context,
            prior_unit_revision=envelope.prior_unit_revision,
            prior_observation_revisions=envelope.prior_observation_revisions,
            scope_attestations=request.scope_attestations,
        )

    def reconciliation_coverage(
        self,
        *,
        source_type: str,
        transition: ProjectionScopeTransition,
        current_units: tuple[SourceUnit, ...],
        run_attestations: tuple[ProjectionScopeAttestation, ...] = (),
    ) -> ProjectionCoverage | None:
        """Return scoped absence proof after provider tombstones were applied."""

        if source_type != "teams":
            return None
        selector_fields = {
            "conversation_ids",
            "channels",
            "group_chats",
            "individual_chats",
        }
        supported_fields = selector_fields | {"rolling_retention_days"}
        changed_fields = {
            key
            for key in set(transition.previous_scope) | set(transition.target_scope)
            if transition.previous_scope.get(key) != transition.target_scope.get(key)
        }
        if not changed_fields or not changed_fields.issubset(supported_fields):
            return None
        from memforge.local_agent.source_contract import (
            canonical_teams_conversation_ids,
        )

        try:
            target_conversations = set(
                canonical_teams_conversation_ids(
                    transition.target_scope,
                    require_nonempty=True,
                )
            )
        except ValueError:
            return None
        if not _teams_run_attests_target_scope(
            transition=transition,
            target_conversations=target_conversations,
            run_attestations=run_attestations,
        ):
            return None
        current_window_units = tuple(unit for unit in current_units if unit.unit_type == "teams_window")
        if any(unit.unit_type != "teams_window" for unit in current_units):
            return None
        if (changed_fields & selector_fields or target_conversations) and not all(
            str(unit.locator.get("conversation_id") or "").strip() in target_conversations
            for unit in current_window_units
        ):
            return None

        return ProjectionCoverage.TOMBSTONED_DELTA


DEFAULT_SOURCE_PROJECTION_ADAPTER = GeneSourceProjectionAdapter()


def project_source_item(
    *,
    source_id: str,
    source_type: str,
    run_id: str,
    item: ContentItem,
    raw: RawContent,
    normalized: NormalizedContent,
    artifacts: tuple[StoredSourceArtifact, ...] = (),
    scope: Mapping[str, object] | None = None,
    access_context: Mapping[str, object] | None = None,
    prior_unit_revision: SourceUnitRevision | None = None,
    prior_observation_revisions: Mapping[str, SourceObservationRevision] | None = None,
    scope_attestations: tuple[ProjectionScopeAttestation, ...] = (),
) -> SourceProjection:
    """Project one completely fetched Source Unit.

    The run scope is exactly this unit. Source-wide absence is handled by the
    enclosing manifest projection; a unit projection never claims another unit
    was deleted merely because it was not part of this call.
    """

    prior_observation_revisions = prior_observation_revisions or {}
    native = _native_payload(raw)
    projected_scope = dict(scope or {})
    native_projection = _project_native(
        source_id=source_id,
        source_type=source_type,
        item=item,
        native=native,
        normalized=normalized,
    )
    unit_type = native_projection.unit_type
    provider_key = native_projection.provider_key
    relations_input = native_projection.relations
    coverage = native_projection.coverage
    locator = native_projection.locator
    coverage = _provider_authoritative_unit_coverage(
        source_type=source_type,
        native=native,
        coverage=coverage,
        projected_scope=projected_scope,
        scope_attestations=scope_attestations,
    )
    persisted_unit_id = projected_scope.get("source_unit_id")
    persisted_provider_key = projected_scope.get("source_unit_provider_key")
    incarnation = projected_scope.get("source_unit_incarnation")
    if persisted_unit_id is not None and not persisted_provider_key:
        raise ValueError("persisted Source Unit identity requires its provider key")
    if persisted_provider_key is not None:
        provider_key = str(persisted_provider_key)
    elif incarnation is not None:
        provider_key = f"{provider_key}#incarnation:{incarnation}"
    unit_id = (
        str(persisted_unit_id)
        if persisted_unit_id is not None
        else _stable_id("unit", source_id, unit_type, provider_key)
    )
    unit = SourceUnit(
        id=unit_id,
        source_id=source_id,
        unit_type=unit_type,
        provider_key=provider_key,
        locator={**locator, "document_id": item.item_id},
    )
    # An Artifact is part of its parent Observation's content and carries its time.
    parent_times = {
        (value.observation_type, value.provider_key): value.observed_at
        for value in native_projection.observations
    }
    artifact_inputs = tuple(
        _ObservationInput(
            observation_type=SOURCE_ARTIFACT_OBSERVATION_TYPE,
            provider_key=source_artifact_observation_provider_key(artifact),
            content="",
            semantic_value=artifact.sha256,
            locator=dict(artifact.locator),
            observed_at=parent_times.get(
                (artifact.parent_observation_type, artifact.parent_provider_key)
            ),
            metadata=source_artifact_observation_metadata(
                artifact,
                parent_observation_id=_stable_id(
                    "obs", unit_id, artifact.parent_observation_type, artifact.parent_provider_key,
                ),
            ),
            semantic_hash=artifact.sha256,
        )
        for artifact in artifacts
    )
    # The Unit Title comes first and is returned by every projection of a live Unit.
    title = native_projection.title
    observations_input = (
        *((title.observation(),) if title is not None else ()),
        *native_projection.observations,
        *artifact_inputs,
    )
    observations: list[SourceObservation] = []
    revisions: list[SourceObservationRevision] = []
    carried_revision_ids: list[str] = []
    for value in observations_input:
        evidence_profile = representation_profile_for_observation_contract(
            source_type=source_type,
            observation_type=value.observation_type,
        )
        if evidence_profile is None:
            raise ValueError(
                "Source Observation contract lacks an Evidence Representation Profile: "
                f"{source_type}/{value.observation_type}"
            )
        observation_id = _stable_id("obs", unit_id, value.observation_type, value.provider_key)
        semantic_hash = _observation_semantic_hash(value)
        revision_id = _stable_id("obsrev", observation_id, semantic_hash)
        observations.append(
            SourceObservation(
                id=observation_id,
                source_id=source_id,
                source_unit_id=unit_id,
                observation_type=value.observation_type,
                provider_key=value.provider_key,
                locator=dict(value.locator),
            )
        )
        projected_revision = SourceObservationRevision(
            id=revision_id,
            observation_id=observation_id,
            semantic_hash=semantic_hash,
            content=value.content,
            observed_at=value.observed_at,
            metadata={**dict(value.metadata), "provider_key": value.provider_key},
            evidence_profile=evidence_profile,
        )
        prior_revision = prior_observation_revisions.get(observation_id)
        # Revision identity is semantic. Operational metadata enrichment under
        # an unchanged semantic hash must preserve the exact immutable row; a
        # revision recorded without a source time takes the one the source now
        # gives, and a recorded source time is never replaced.
        if (
            prior_revision is not None
            and prior_revision.observation_id == observation_id
            and prior_revision.semantic_hash == semantic_hash
        ):
            if prior_revision.evidence_profile not in {None, evidence_profile}:
                raise ValueError("immutable Observation Revision changed representation profile")
            reused = prior_revision
            if reused.evidence_profile is None:
                reused = replace(reused, evidence_profile=evidence_profile)
            if reused.observed_at is None and value.observed_at is not None:
                reused = replace(reused, observed_at=value.observed_at)
            revisions.append(reused)
        else:
            revisions.append(projected_revision)
    if coverage is ProjectionCoverage.PARTIAL_PROJECTION:
        projected_observation_ids = {item.observation_id for item in revisions}
        for observation_id, revision in prior_observation_revisions.items():
            if observation_id in projected_observation_ids:
                continue
            revisions.append(revision)
            carried_revision_ids.append(revision.id)
    observation_hashes = sorted((item.observation_id, item.semantic_hash) for item in revisions)
    semantic_hash = _canonical_hash(observation_hashes)
    location_hash = _canonical_hash(unit.locator)
    membership_hash = _canonical_hash(sorted(item.observation_id for item in revisions))
    access_hash = projection_access_fingerprint(access_context) if access_context else None
    unit_revision_id = _stable_id(
        "unitrev",
        unit_id,
        semantic_hash,
        location_hash,
        membership_hash,
        access_hash,
    )
    projected_unit_revision = SourceUnitRevision(
        id=unit_revision_id,
        source_unit_id=unit_id,
        semantic_hash=semantic_hash,
        location_hash=location_hash,
        membership_hash=membership_hash,
        access_hash=access_hash,
        observation_revision_ids=tuple(sorted(item.id for item in revisions)),
        observed_at=latest_source_time(revision.observed_at for revision in revisions),
    )
    unit_revision = (
        prior_unit_revision
        if (
            prior_unit_revision is not None
            and prior_unit_revision.id == projected_unit_revision.id
            and prior_unit_revision.source_unit_id
            == projected_unit_revision.source_unit_id
            and prior_unit_revision.semantic_hash
            == projected_unit_revision.semantic_hash
            and prior_unit_revision.location_hash
            == projected_unit_revision.location_hash
            and prior_unit_revision.membership_hash
            == projected_unit_revision.membership_hash
            and prior_unit_revision.access_hash
            == projected_unit_revision.access_hash
            and tuple(sorted(prior_unit_revision.observation_revision_ids))
            == projected_unit_revision.observation_revision_ids
        )
        else projected_unit_revision
    )
    axes: set[DeltaAxis] = set()
    previous_id = prior_unit_revision.id if prior_unit_revision else None
    if prior_unit_revision is None or prior_unit_revision.semantic_hash != semantic_hash:
        axes.add(DeltaAxis.SEMANTIC)
    if prior_unit_revision is not None and prior_unit_revision.location_hash != location_hash:
        axes.add(DeltaAxis.LOCATION)
    if prior_unit_revision is None or prior_unit_revision.membership_hash != membership_hash:
        axes.add(DeltaAxis.MEMBERSHIP)
    if prior_unit_revision is not None and prior_unit_revision.access_hash != access_hash:
        axes.add(DeltaAxis.ACCESS)

    current_by_observation = {item.observation_id: item for item in revisions}
    previous_ids = set(prior_observation_revisions)
    current_ids = set(current_by_observation)
    added_ids = (
        tuple(sorted(current_ids - previous_ids)) if prior_unit_revision is not None else tuple(sorted(current_ids))
    )
    removed_ids = (
        tuple(sorted(previous_ids - current_ids)) if prior_unit_revision is not None and coverage.proves_absence else ()
    )
    changed_ids = {
        observation_id
        for observation_id, revision in current_by_observation.items()
        if observation_id not in prior_observation_revisions
        or prior_observation_revisions[observation_id].semantic_hash != revision.semantic_hash
    }
    if DeltaAxis.SEMANTIC in axes and not prior_observation_revisions:
        changed_ids = current_ids
    changed_anchors = tuple(
        SourceAnchor(
            kind=AnchorKind.WHOLE_OBSERVATION,
            observation_id=observation_id,
            observation_revision_id=current_by_observation[observation_id].id,
        )
        for observation_id in sorted(changed_ids)
    )
    if removed_ids:
        axes.add(DeltaAxis.MEMBERSHIP)
    delta = RevisionDelta(
        source_unit_id=unit_id,
        previous_unit_revision_id=previous_id,
        current_unit_revision_id=unit_revision.id,
        axes=frozenset(axes),
        coverage=coverage,
        changed_anchors=changed_anchors,
        added_observation_ids=added_ids,
        removed_observation_ids=removed_ids,
    )
    observation_ids_by_provider_key = {
        value.provider_key: observation.id for value, observation in zip(observations_input, observations, strict=True)
    }

    def endpoint(value: str) -> str:
        if value == "$unit":
            return unit_id
        if value in observation_ids_by_provider_key:
            return observation_ids_by_provider_key[value]
        return _relation_endpoint(source_id, unit_type, value)

    relations = tuple(
        SourceRelation(
            relation_type=relation_type,
            from_id=endpoint(from_key),
            to_id=endpoint(to_key),
            provider_relation_id=provider_relation_id,
            metadata=metadata,
        )
        for relation_type, from_key, to_key, provider_relation_id, metadata in relations_input
    )
    return SourceProjection(
        run_id=run_id,
        source_id=source_id,
        source_type=source_type,
        scope={**projected_scope, "source_unit_id": unit_id},
        coverage=coverage,
        observations=tuple(observations),
        observation_revisions=tuple(revisions),
        source_units=(unit,),
        source_unit_revisions=(unit_revision,),
        relations=relations,
        deltas=(delta,),
        checkpoint={"item_id": item.item_id, "version": item.version},
        carried_observation_revision_ids=tuple(carried_revision_ids),
    )


def project_source_unit_tombstone(
    *,
    source_type: str,
    run_id: str,
    source_unit: SourceUnit,
    prior_unit_revision: SourceUnitRevision,
    prior_observation_revisions: Mapping[str, SourceObservationRevision],
    reason: str,
) -> SourceProjection:
    """Project an explicit authoritative tombstone for one known Source Unit."""

    semantic_hash = _canonical_hash({"tombstone": source_unit.id, "reason": reason})
    membership_hash = _canonical_hash([])
    unit_revision = SourceUnitRevision(
        id=_stable_id("unitrev", source_unit.id, semantic_hash, membership_hash),
        source_unit_id=source_unit.id,
        semantic_hash=semantic_hash,
        location_hash=prior_unit_revision.location_hash,
        membership_hash=membership_hash,
        access_hash=prior_unit_revision.access_hash,
        observation_revision_ids=(),
    )
    delta = RevisionDelta(
        source_unit_id=source_unit.id,
        previous_unit_revision_id=prior_unit_revision.id,
        current_unit_revision_id=unit_revision.id,
        axes=frozenset({DeltaAxis.SEMANTIC, DeltaAxis.MEMBERSHIP}),
        coverage=ProjectionCoverage.TOMBSTONED_DELTA,
        removed_observation_ids=tuple(sorted(prior_observation_revisions)),
    )
    return SourceProjection(
        run_id=run_id,
        source_id=source_unit.source_id,
        source_type=source_type,
        scope={"source_unit_id": source_unit.id},
        coverage=ProjectionCoverage.TOMBSTONED_DELTA,
        observations=(),
        observation_revisions=(),
        source_units=(
            SourceUnit(
                id=source_unit.id,
                source_id=source_unit.source_id,
                unit_type=source_unit.unit_type,
                provider_key=source_unit.provider_key,
                locator={**source_unit.locator, "tombstone_reason": reason},
            ),
        ),
        source_unit_revisions=(unit_revision,),
        relations=(),
        deltas=(delta,),
        checkpoint={"tombstoned": True, "reason": reason},
    )


def _provider_authoritative_unit_coverage(
    *,
    source_type: str,
    native: object,
    coverage: ProjectionCoverage,
    projected_scope: Mapping[str, object],
    scope_attestations: tuple[ProjectionScopeAttestation, ...],
) -> ProjectionCoverage:
    """Apply run authority only where the provider unit contract supports it."""

    configured_scope = projected_scope.get("configured_scope")
    teams_native = (
        native.get("raw_payload")
        if isinstance(native, Mapping) and isinstance(native.get("raw_payload"), Mapping)
        else native
    )
    if (
        source_type == "teams"
        and isinstance(teams_native, Mapping)
        and teams_native.get("tombstone_reason") == "outside_rolling_retention"
        and not _teams_retention_attestation_is_valid(
            native=teams_native,
            configured_scope=(configured_scope if isinstance(configured_scope, Mapping) else {}),
            run_attestations=scope_attestations,
        )
    ):
        raise ValueError("Teams rolling-retention tombstone lacks complete run-scoped coverage evidence")
    if coverage.proves_absence or projected_scope.get("authoritative_snapshot") is not True:
        return coverage
    if (
        source_type == "teams"
        and isinstance(native, Mapping)
        and native.get("package_kind") == "teams_window_document"
        and isinstance(native.get("raw_payload"), Mapping)
    ):
        # A force-full local collection attempt is validated against its
        # immutable package manifest before replay. That source-wide proof also
        # makes each canonical window package a complete snapshot of its unit.
        return ProjectionCoverage.COMPLETE_SNAPSHOT
    return coverage


def _teams_run_attests_target_scope(
    *,
    transition: ProjectionScopeTransition,
    target_conversations: set[str],
    run_attestations: tuple[ProjectionScopeAttestation, ...],
) -> bool:
    """Require one exact, successful, same-attempt poll per target conversation."""

    return _teams_attestations_cover_target_scope(
        target_scope=transition.target_scope,
        target_conversations=target_conversations,
        expected_transition_id=transition.id,
        run_attestations=run_attestations,
    )


def _teams_attestations_cover_target_scope(
    *,
    target_scope: Mapping[str, object],
    target_conversations: set[str],
    expected_transition_id: str | None,
    run_attestations: tuple[ProjectionScopeAttestation, ...],
) -> bool:
    target_fingerprint = projection_scope_fingerprint(target_scope)
    expected_conversations = sorted(target_conversations)
    by_conversation: dict[str, ProjectionScopeAttestation] = {}
    attempt_ids: set[str] = set()
    for attestation in run_attestations:
        conversation_id = attestation.subject_key.strip()
        target_values = attestation.evidence.get("target_subject_keys")
        poll = attestation.evidence.get("poll")
        attempt_id = attestation.collection_attempt_id.strip()
        if (
            attestation.subject_type != "conversation"
            or conversation_id not in target_conversations
            or attestation.transition_id != expected_transition_id
            or attestation.target_scope_fingerprint != target_fingerprint
            or target_values != expected_conversations
            or not attempt_id
            or not isinstance(poll, Mapping)
            or not _teams_poll_attestation_is_valid(poll, conversation_id)
        ):
            return False
        if conversation_id in by_conversation:
            return False
        by_conversation[conversation_id] = attestation
        attempt_ids.add(attempt_id)
    return set(by_conversation) == target_conversations and len(attempt_ids) == 1


def _teams_poll_attestation_is_valid(
    poll: Mapping[str, object],
    conversation_id: str,
) -> bool:
    if (
        str(poll.get("raw_conversation_id") or "").strip() != conversation_id
        or str(poll.get("access_probe_status") or "").strip().lower() != "ok"
    ):
        return False
    stop_reason = str(poll.get("stop_reason") or "").strip()
    if stop_reason == "no_backward_link":
        return poll.get("pagination_complete") is True
    if stop_reason != "cutoff_reached":
        return False
    covered_from = _normalized_utc_timestamp(poll.get("absence_covered_from"))
    covered_to = _normalized_utc_timestamp(poll.get("absence_covered_to"))
    return bool(covered_from and covered_to and covered_from <= covered_to)


def _teams_retention_attestation_is_valid(
    *,
    native: Mapping[str, object],
    configured_scope: Mapping[str, object],
    run_attestations: tuple[ProjectionScopeAttestation, ...],
) -> bool:
    conversation_id = str(native.get("conversation_id") or "").strip()
    cutoff = _normalized_utc_timestamp(native.get("rolling_retention_cutoff"))
    observed_to = _normalized_utc_timestamp(native.get("prior_observed_to"))
    try:
        retention_days = int(configured_scope.get("rolling_retention_days") or 0)
    except (TypeError, ValueError):
        return False
    if (
        retention_days not in TEAMS_ROLLING_RETENTION_PRESETS
        or not conversation_id
        or not cutoff
        or not observed_to
        or observed_to >= cutoff
    ):
        return False
    from memforge.local_agent.source_contract import canonical_teams_conversation_ids

    try:
        target_conversations = set(canonical_teams_conversation_ids(configured_scope, require_nonempty=True))
    except ValueError:
        return False
    transition_ids = {item.transition_id for item in run_attestations}
    if len(transition_ids) != 1 or not _teams_attestations_cover_target_scope(
        target_scope=configured_scope,
        target_conversations=target_conversations,
        expected_transition_id=next(iter(transition_ids)),
        run_attestations=run_attestations,
    ):
        return False
    target_fingerprint = projection_scope_fingerprint(configured_scope)
    matches = [
        item
        for item in run_attestations
        if item.subject_type == "conversation"
        and item.subject_key == conversation_id
        and item.target_scope_fingerprint == target_fingerprint
        and _normalized_utc_timestamp(item.evidence.get("rolling_retention_cutoff")) == cutoff
    ]
    if len(matches) != 1:
        return False
    return all(
        _normalized_utc_timestamp(item.evidence.get("rolling_retention_cutoff")) == cutoff
        for item in run_attestations
    )


def _normalized_utc_timestamp(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc).isoformat()


def _project_native(
    *,
    source_id: str,
    source_type: str,
    item: ContentItem,
    native: object,
    normalized: NormalizedContent,
) -> _NativeProjection:
    # The source time of a Unit whose body is one Observation, as the Gene reports it.
    body_time = source_time_iso(normalized.source_semantics.get(SOURCE_UPDATED_AT_KEY))
    if source_type == "confluence":
        page_id = str(item.extra.get("page_id") or item.item_id.removeprefix("confluence-"))
        parent_id = str(item.extra.get("parent_page_id") or "")
        relations = ()
        if parent_id:
            relations = (
                (
                    SourceRelationType.CONTAINED_BY,
                    "$unit",
                    f"confluence_page:{parent_id}",
                    f"{page_id}:parent",
                    {},
                ),
            )
        semantic_body = native if isinstance(native, str) else normalized.markdown_body
        display_body = str(normalized.source_semantics.get("semantic_markdown") or normalized.markdown_body)
        semantic_value = {
            "title": item.title,
            "body": semantic_body,
        }
        semantic_content = f"# {item.title}\n\n{display_body}".strip()
        return _NativeProjection(
            unit_type="confluence_page",
            provider_key=page_id,
            observations=(
                _ObservationInput(
                    "page_body",
                    f"{page_id}:body",
                    semantic_content,
                    semantic_value,
                    {},
                    body_time,
                ),
            ),
            relations=relations,
            coverage=ProjectionCoverage.COMPLETE_SNAPSHOT,
            locator={
                "page_id": page_id,
                "space_key": item.extra.get("space_key") or item.space_or_project,
                "parent_page_id": parent_id or None,
                "url": item.source_url,
            },
            title=_UnitTitle.of(
                "Confluence page",
                ("Space", item.extra.get("space_key") or item.space_or_project),
                ("Title", item.title),
            ),
        )
    if source_type == "jira":
        data = native if isinstance(native, dict) else {}
        if data.get("package_kind") and isinstance(data.get("raw_payload"), dict):
            data = data["raw_payload"]
        from memforge.local_agent.jira_contract import validate_jira_observation_identities

        validate_jira_observation_identities(data)
        fields = data.get("fields") if isinstance(data.get("fields"), dict) else {}
        raw_issue_id = data.get("id") or item.extra.get("issue_id")
        issue_id = str(raw_issue_id or "").strip()
        if not issue_id.isdigit():
            raise ValueError("jira projection requires immutable numeric issue id")
        issue_key = str(data.get("key") or item.extra.get("issue_key") or item.item_id)
        core_value = {name: fields.get(name) for name in _JIRA_CORE_FIELDS}
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
                _jira_core_revised_at(fields, histories, changelog_complete=changelog_complete),
            )
        ]
        relations: list[tuple[SourceRelationType, str, str, str | None, Mapping[str, object]]] = []
        previous_key = f"{issue_id}:core"
        comments = data.get("_comments") if isinstance(data.get("_comments"), list) else []
        for comment in comments:
            if not isinstance(comment, dict):
                continue
            comment_id = str(comment["id"])
            body = comment.get("body")
            semantic_comment = {"body": body, "attachments": comment.get("attachments")}
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
            inputs.append(
                _ObservationInput(
                    "changelog",
                    history_id,
                    _canonical_json(history),
                    history,
                    {"issue_key": issue_key},
                    str(history.get("created") or "") or None,
                    {
                        "semantic_class": _jira_changelog_semantic_class(
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
            title=_UnitTitle.of(
                "Jira issue",
                ("Key", issue_key),
                ("Type", _provider_name(fields.get("issuetype"))),
                ("Summary", fields.get("summary")),
            ),
        )
    if source_type == "github_repo":
        semantics = normalized.source_semantics
        repo = "/".join(
            value
            for value in (
                str(item.extra.get("repo_owner") or semantics.get("repo_owner") or ""),
                str(item.extra.get("repo_name") or semantics.get("repo_name") or ""),
            )
            if value
        ) or str(item.extra.get("repo_url") or semantics.get("repo_url") or item.space_or_project)
        path = str(item.extra.get("relative_path") or semantics.get("relative_path") or item.item_id)
        rename_attested = (
            item.extra.get("rename_evidence_authoritative") is True
            or semantics.get("rename_evidence_authoritative") is True
        )
        previous = (
            item.extra.get("previous_filename") or semantics.get("previous_filename")
            if rename_attested
            else None
        )
        explicit_lineage = item.extra.get("file_lineage_id") or semantics.get("file_lineage_id")
        # Git/GitHub does not expose an immutable file id. Built-in cloud-pull
        # and local-push connectors therefore define path as file identity.
        # Rename continuity is optional and accepted only when a provider
        # adapter explicitly attests authoritative rename evidence (for
        # example, a validated Compare API `renamed` record). Never infer a
        # move from a matching blob SHA because copy+delete is ambiguous.
        lineage = str(explicit_lineage or previous or path)
        relations = ()
        if previous:
            predecessor_document_id = item.extra.get("previous_document_id") or semantics.get("previous_document_id")
            repo_url = str(item.extra.get("repo_url") or semantics.get("repo_url") or "")
            repo_ref = str(item.extra.get("repo_ref") or semantics.get("repo_ref") or "")
            if not predecessor_document_id and repo_url and repo_ref:
                predecessor_document_id = build_github_repo_doc_id(
                    source_id=source_id,
                    repo_url=repo_url,
                    repo_ref=repo_ref,
                    relative_path=str(previous),
                )
            relations = (
                (
                    SourceRelationType.RENAMED_FROM,
                    "$unit",
                    f"github_file:{repo}:{previous}",
                    None,
                    ({"predecessor_document_id": str(predecessor_document_id)} if predecessor_document_id else {}),
                ),
            )
        return _NativeProjection(
            unit_type="github_file",
            provider_key=f"{repo}:{lineage}",
            observations=(
                _ObservationInput(
                    "file_content",
                    "content",
                    normalized.markdown_body,
                    normalized.markdown_body,
                    {"path": path},
                    body_time,
                ),
            ),
            relations=relations,
            coverage=ProjectionCoverage.COMPLETE_SNAPSHOT,
            locator={"repository": repo, "path": path, "ref": item.extra.get("repo_ref"), "url": item.source_url},
            title=_UnitTitle.of(
                "GitHub file",
                ("Repository", repo),
                ("Path", path),
                ("Ref", item.extra.get("repo_ref") or semantics.get("repo_ref")),
            ),
        )
    if source_type == "github_pages":
        canonical_url = str(
            item.extra.get("canonical_url") or normalized.source_semantics.get("canonical_url") or item.source_url
        )
        semantic_value = native if isinstance(native, str) else normalized.markdown_body
        semantic_content = normalized.markdown_body
        return _NativeProjection(
            unit_type="rendered_page",
            provider_key=canonical_url,
            observations=(
                _ObservationInput("page_content", "content", semantic_content, semantic_value, {}, body_time),
            ),
            relations=(),
            coverage=ProjectionCoverage.COMPLETE_SNAPSHOT,
            locator={"canonical_url": canonical_url, "title": item.title},
            title=_UnitTitle.of("GitHub Pages page", ("Title", item.title), ("URL", canonical_url)),
        )
    if source_type == "local_markdown":
        data = native if isinstance(native, dict) else {}
        vault = str(data.get("vault_id") or item.space_or_project or "default")
        path = str(data.get("relative_path") or item.extra.get("relative_path") or item.item_id)
        lineage = str(data.get("file_lineage_id") or item.extra.get("file_lineage_id") or path)
        body = str(data.get("markdown") or normalized.markdown_body)
        return _NativeProjection(
            unit_type="local_file",
            provider_key=f"{vault}:{lineage}",
            observations=(_ObservationInput("file_content", "content", body, body, {"path": path}, body_time),),
            relations=(),
            coverage=ProjectionCoverage.COMPLETE_SNAPSHOT,
            locator={"vault_id": vault, "path": path, "url": item.source_url},
            title=_UnitTitle.of(
                "Markdown file",
                ("Vault", data.get("vault_id") or item.space_or_project),
                ("Path", path),
            ),
        )
    if source_type == "teams":
        data = native if isinstance(native, dict) else {}
        if data.get("package_kind") and isinstance(data.get("raw_payload"), dict):
            data = data["raw_payload"]
        window_id = str(item.extra.get("window_id") or data.get("window_id") or item.item_id)
        conversation_id = str(item.extra.get("conversation_id") or data.get("conversation_id") or "")
        messages = data.get("messages") if isinstance(data.get("messages"), list) else []
        if messages:
            from memforge.local_agent.teams_contract import validate_teams_canonical_messages

            messages = list(validate_teams_canonical_messages(messages))
        inputs = []
        relations = []
        previous_key = None
        for message in messages:
            message_id = str(message["id"])
            semantic_message = {
                "content": message.get("content"),
                "attachments": message.get("attachments"),
                "deleted": message.get("deletedDateTime") or message.get("deleted_at"),
            }
            inputs.append(
                _ObservationInput(
                    "message",
                    message_id,
                    _canonical_json(semantic_message),
                    semantic_message,
                    {"conversation_id": conversation_id},
                    str(message.get("lastModifiedDateTime") or message.get("time") or "") or None,
                    {"claim_evidence_scope": "atomic"},
                )
            )
            reply_to = message.get("reply_to_id") or message.get("replyToId")
            if reply_to:
                relations.append((SourceRelationType.REPLIES_TO, message_id, str(reply_to), None, {}))
            elif previous_key:
                relations.append((SourceRelationType.PRECEDES, previous_key, message_id, None, {}))
            previous_key = message_id
        coverage = (
            ProjectionCoverage.COMPLETE_SNAPSHOT
            if data.get("authoritative_snapshot") or data.get("_authoritative_snapshot")
            else ProjectionCoverage.PARTIAL_PROJECTION
        )
        observed_times = sorted(
            str(message.get("time") or "").strip()
            for message in messages
            if isinstance(message, dict) and str(message.get("time") or "").strip()
        )
        observed_from = str(
            item.extra.get("block_start")
            or data.get("first_message_time")
            or data.get("prior_observed_from")
            or (observed_times[0] if observed_times else "")
        ).strip()
        observed_to = str(
            item.extra.get("block_end")
            or data.get("last_message_time")
            or data.get("prior_observed_to")
            or (observed_times[-1] if observed_times else "")
        ).strip()
        observed_from = _normalized_utc_timestamp(observed_from) or observed_from
        observed_to = _normalized_utc_timestamp(observed_to) or observed_to
        locator = {
            "conversation_id": conversation_id,
            "window_id": window_id,
            "observed_from": observed_from or None,
            "observed_to": observed_to or None,
            "url": item.source_url,
        }
        tombstoned = data.get("_tombstone") is True
        if tombstoned:
            locator["tombstone_reason"] = data.get("tombstone_reason")
        return _NativeProjection(
            unit_type="teams_window",
            provider_key=window_id,
            observations=tuple(inputs),
            relations=tuple(relations),
            coverage=coverage,
            locator=locator,
            # A tombstoned window has no live Unit left to name.
            title=None if tombstoned else _UnitTitle.of(
                "Teams conversation",
                ("Conversation type", data.get("conversation_type")),
                ("Team", data.get("team_name")),
                ("Channel", data.get("channel_name")),
                ("Title", item.title),
                ("From", observed_from),
                ("To", observed_to),
            ),
        )
    if source_type == "agent_session":
        data = native if isinstance(native, dict) else {}
        receipt = data.get("receipt") if isinstance(data.get("receipt"), dict) else {}
        window_id = str(data.get("doc_id") or item.item_id)
        body = str(data.get("markdown") or normalized.markdown_body)
        return _NativeProjection(
            unit_type="agent_session_window",
            provider_key=window_id,
            observations=(_ObservationInput("session_summary", window_id, body, body, {}, body_time),),
            relations=(),
            coverage=ProjectionCoverage.PARTIAL_PROJECTION,
            locator={
                "client": receipt.get("client"),
                "session_id": receipt.get("session_id"),
                "history_window_kind": receipt.get("history_window_kind"),
                "url": item.source_url,
            },
            title=_UnitTitle.of(
                "Agent session",
                ("Client", receipt.get("client")),
                ("Window", receipt.get("history_window_kind")),
                ("Title", item.title),
            ),
        )
    # Extension-safe fallback for document-like genes that have not yet opted
    # into a richer native projection.  It deliberately claims only partial
    # coverage, so it can drive semantic change detection but can never prove
    # that an omitted observation or source unit was deleted.
    body = normalized.markdown_body
    return _NativeProjection(
        unit_type="generic_document",
        provider_key=item.item_id,
        observations=(_ObservationInput("document_content", item.item_id, body, body, {}, body_time),),
        relations=(),
        coverage=ProjectionCoverage.PARTIAL_PROJECTION,
        locator={
            "item_id": item.item_id,
            "url": item.source_url,
            "title": item.title,
            "source_type": source_type,
        },
        title=_UnitTitle.of("Document", ("Title", item.title), ("Source type", source_type)),
    )


def _provider_name(value: object) -> object:
    """A provider object's display name, such as a Jira issue type's name."""

    return value.get("name") if isinstance(value, Mapping) else value


def _jira_core_revised_at(
    fields: Mapping[str, object],
    histories: list[object],
    *,
    changelog_complete: bool,
) -> str | None:
    """When the issue's core fields last changed, from the issue's own records.

    The latest changelog entry that touches a core field gives the time; an
    issue whose complete changelog never touches one has kept its core since
    ``fields.created``. A truncated changelog may omit the latest core change,
    so the time is unknown. ``fields.updated`` is not used: comments and other
    fields move it too.
    """

    if not changelog_complete:
        return None
    core_fields = frozenset(_JIRA_CORE_FIELDS)
    core_change_times = [
        history.get("created")
        for history in histories
        if isinstance(history, Mapping)
        and any(
            isinstance(entry, Mapping)
            and str(entry.get("fieldId") or entry.get("field") or "").strip().lower() in core_fields
            for entry in (history.get("items") if isinstance(history.get("items"), list) else [])
        )
    ]
    if core_change_times:
        return latest_source_time(core_change_times)
    return source_time_iso(fields.get("created"))


def _jira_changelog_semantic_class(history: Mapping[str, object]) -> str:
    items = history.get("items")
    history_items = items if isinstance(items, list) else []
    fields = {
        " ".join(str(item.get("field") or "").strip().lower().split())
        for item in history_items
        if isinstance(item, Mapping)
    }
    fields.discard("")
    if fields and fields.issubset({"attachment"}):
        return "attachment_event"
    if fields and fields.issubset(_JIRA_OPERATIONAL_HISTORY_FIELDS):
        return "operational_transition"
    return "domain_transition"


def _native_payload(raw: RawContent) -> object:
    text = raw.body.decode("utf-8", errors="replace")
    if raw.content_type == "application/json":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    return text


def _relation_endpoint(source_id: str, unit_type: str, provider_key: str) -> str:
    endpoint_type, separator, endpoint_key = provider_key.partition(":")
    if separator and endpoint_type in {"confluence_page", "github_file"}:
        return _stable_id("unit", source_id, endpoint_type, endpoint_key)
    return _stable_id("obs", source_id, unit_type, provider_key)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *values: object) -> str:
    digest = hashlib.sha256("\x1f".join(str(value) for value in values).encode("utf-8")).hexdigest()[:24]
    return f"{prefix}-{digest}"
