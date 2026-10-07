"""Provider adapters that project fetched Gene items into stable source lineage.

Genes remain responsible for authentication and provider I/O. Source adapters
own native payload interpretation. This module turns their declarations into
Source Units, Observations, immutable revisions, relations, and deltas.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Mapping

from memforge.models import ContentItem, NormalizedContent, RawContent
from memforge.source_adapters.contracts import (
    _ObservationInput, _NativeProjection, _UnitEndpoint, _unit_title, _canonical_json,
)
from memforge.source_projection import (
    AnchorKind,
    DeltaAxis,
    ProjectionCoverage,
    ProjectionEnvelope,
    ProjectionIdentityConflict,
    ProjectionScopeAttestation,
    RevisionDelta,
    SourceAnchor,
    SourceObservation,
    SourceObservationRevision,
    SourceProjection,
    ProjectionScopeTransition,
    SourceRelation,
    SourceUnit,
    SourceUnitRevision,
)
from memforge.source_time import (
    SOURCE_UPDATED_AT_KEY,
    latest_source_time,
)
from memforge.source_representation import (
    in_current_representation,
    representation_profile_for_observation_contract,
)
from memforge.source_projection_config import (
    projection_access_fingerprint,
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

def source_run_projection_coverage(
    *,
    authoritative_snapshot: bool,
    scope_transition: bool,
    discovery_complete: bool,
) -> ProjectionCoverage:
    """What a run's discovery alone proves about the Documents it did not return.

    A submitted authoritative snapshot is the whole Source. A complete discovery
    of a newly configured scope is the whole new scope: the user changed the
    scope, so a Unit outside it is removed (ADR 0005). Any other discovery is
    partial, whether or not ``since`` narrowed it, because a query result is
    not proof that the provider no longer has an item; such a run removes only
    what its scope listing proves absent (ADR 0045).
    """

    if authoritative_snapshot:
        return ProjectionCoverage.COMPLETE_SNAPSHOT
    if scope_transition and discovery_complete:
        return ProjectionCoverage.COMPLETE_SNAPSHOT
    return ProjectionCoverage.PARTIAL_PROJECTION




_REVISION_SEMANTIC_METADATA_KEYS = ("claim_evidence_scope",)






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
            stored_observation_revisions=envelope.stored_observation_revisions,
            scope_attestations=request.scope_attestations,
        )

    def reconciliation_coverage(self, *, source_type: str, transition: ProjectionScopeTransition,
        current_units: tuple[SourceUnit, ...], run_attestations: tuple[ProjectionScopeAttestation, ...] = (),
    ) -> ProjectionCoverage | None:
        """Dispatch scoped absence proof to its source owner."""
        if source_type != "teams":
            return None
        from memforge.source_adapters.teams import reconciliation_coverage

        return reconciliation_coverage(transition=transition, current_units=current_units,
            run_attestations=run_attestations)


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
    stored_observation_revisions: Mapping[str, SourceObservationRevision] | None = None,
    scope_attestations: tuple[ProjectionScopeAttestation, ...] = (),
) -> SourceProjection:
    """Project one completely fetched Source Unit.

    The run scope is exactly this unit. Source-wide absence is handled by the
    enclosing manifest projection; a unit projection never claims another unit
    was deleted merely because it was not part of this call.

    Observation Revisions are content-addressed by Observation and semantic
    hash. A projected revision whose id is already stored, as a current
    revision in ``prior_observation_revisions`` or any other revision in
    ``stored_observation_revisions`` (keyed by revision id), is that stored row.

    Only prior revisions in the current representation are compared with this
    projection or carried by it. A prior Observation outside it is no member of
    the projected revision under any coverage; it is retired, not removed, since
    removal means only that the coverage proves the Observation absent.
    """

    prior_observation_revisions = {
        observation_id: revision
        for observation_id, revision in (prior_observation_revisions or {}).items()
        if in_current_representation(revision)
    }
    stored_by_id = {
        **dict(stored_observation_revisions or {}),
        **{revision.id: revision for revision in prior_observation_revisions.values()},
    }
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
    observations_input = (*native_projection.observations, *artifact_inputs)
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
        stored_revision = stored_by_id.get(revision_id)
        # Revision identity is semantic. A stored revision with this id is
        # reused exactly, whatever metadata this projection derived; a revision
        # recorded without a source time takes the one the source now gives,
        # and a recorded source time is never replaced.
        if stored_revision is not None:
            if (
                stored_revision.observation_id != observation_id
                or stored_revision.semantic_hash != semantic_hash
                or stored_revision.evidence_profile not in {None, evidence_profile}
            ):
                raise ProjectionIdentityConflict("source_observation_revisions", revision_id)
            reused = stored_revision
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

    def endpoint(value: str | _UnitEndpoint) -> str:
        if isinstance(value, _UnitEndpoint):
            return _stable_id("unit", source_id, value.unit_type, value.provider_key)
        if value == "$unit":
            return unit_id
        if value in observation_ids_by_provider_key:
            return observation_ids_by_provider_key[value]
        return _stable_id("obs", source_id, unit_type, value)

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
        unit_title=native_projection.title,
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
    *, source_type: str, native: object, coverage: ProjectionCoverage,
    projected_scope: Mapping[str, object], scope_attestations: tuple[ProjectionScopeAttestation, ...],
) -> ProjectionCoverage:
    """Dispatch source-owned collection completeness policy."""
    if source_type == "teams":
        from memforge.source_adapters.teams import authoritative_unit_coverage

        return authoritative_unit_coverage(native=native, coverage=coverage,
            projected_scope=projected_scope, scope_attestations=scope_attestations)
    return coverage


def _project_native(
    *,
    source_id: str,
    source_type: str,
    item: ContentItem,
    native: object,
    normalized: NormalizedContent,
) -> _NativeProjection:
    # The source time of a Unit whose body is one Observation, as the Gene reports it.
    body_time = normalized.source_semantics.get(SOURCE_UPDATED_AT_KEY)
    if source_type == "confluence":
        from memforge.source_adapters.confluence import project_native

        return project_native(source_id=source_id, item=item, native=native, normalized=normalized)
    if source_type == "jira":
        from memforge.source_adapters.jira import project_native

        return project_native(source_id=source_id, item=item, native=native, normalized=normalized)
    if source_type == "github_repo":
        from memforge.source_adapters.github_repo import project_native

        return project_native(source_id=source_id, item=item, native=native, normalized=normalized)
    if source_type == "github_pages":
        from memforge.source_adapters.github_pages import project_native

        return project_native(source_id=source_id, item=item, native=native, normalized=normalized)
    if source_type == "local_markdown":
        from memforge.source_adapters.local_markdown import project_native

        return project_native(source_id=source_id, item=item, native=native, normalized=normalized)
    if source_type == "teams":
        from memforge.source_adapters.teams import project_native

        return project_native(source_id=source_id, item=item, native=native, normalized=normalized)
    if source_type == "agent_session":
        from memforge.source_adapters.agent_session import project_native

        return project_native(source_id=source_id, item=item, native=native, normalized=normalized)
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
        title=_unit_title("Document", ("Title", item.title), ("Source type", source_type)),
    )




def _native_payload(raw: RawContent) -> object:
    text = raw.body.decode("utf-8", errors="replace")
    if raw.content_type == "application/json":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    return text


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *values: object) -> str:
    digest = hashlib.sha256("\x1f".join(str(value) for value in values).encode("utf-8")).hexdigest()[:24]
    return f"{prefix}-{digest}"
