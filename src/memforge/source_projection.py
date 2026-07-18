"""Provider-neutral source identity, revision, coverage, and impact contracts.

Connectors may use provider-specific APIs internally, but source lifecycle starts
at :class:`SourceProjectionAdapter`.  Everything downstream can therefore reason
about the same stable units, immutable observation revisions, controlled anchors,
and run-scoped coverage for documents, conversations, and hybrid sources.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Mapping, Protocol, runtime_checkable

if TYPE_CHECKING:
    from memforge.models import ContentItem, NormalizedContent, RawContent


class ProjectionRunMode(str, Enum):
    FULL_SNAPSHOT = "full_snapshot"
    DELTA = "delta"
    APPEND = "append"


class ProjectionCoverage(str, Enum):
    COMPLETE_SNAPSHOT = "complete_snapshot"
    TOMBSTONED_DELTA = "tombstoned_delta"
    PARTIAL_PROJECTION = "partial_projection"

    @property
    def proves_absence(self) -> bool:
        return self in {self.COMPLETE_SNAPSHOT, self.TOMBSTONED_DELTA}


class ProjectionScopeTransitionStatus(str, Enum):
    """Durable state of one configured-source membership transition."""

    PENDING = "pending"
    RUNNING = "running"
    APPLIED = "applied"
    FAILED = "failed"


class DeltaAxis(str, Enum):
    SEMANTIC = "semantic"
    LOCATION = "location"
    MEMBERSHIP = "membership"
    ACCESS = "access"


class AnchorKind(str, Enum):
    WHOLE_OBSERVATION = "whole_observation"
    STABLE_FRAGMENT = "stable_fragment"
    REVISION_RANGE = "revision_range"


class ImpactResult(str, Enum):
    AFFECTED = "affected"
    DISJOINT = "disjoint"
    UNKNOWN = "unknown"


class SourceRelationType(str, Enum):
    CONTAINED_BY = "contained_by"
    REPLIES_TO = "replies_to"
    PRECEDES = "precedes"
    RENAMED_FROM = "renamed_from"
    REDIRECTS_TO = "redirects_to"
    REFERENCES = "references"


@dataclass(frozen=True, slots=True)
class SourceAnchor:
    """A controlled, revision-pinned location inside one observation."""

    kind: AnchorKind
    observation_id: str
    observation_revision_id: str
    fragment_id: str | None = None
    range_start: int | None = None
    range_end: int | None = None

    def __post_init__(self) -> None:
        if not self.observation_id or not self.observation_revision_id:
            raise ValueError("anchor requires observation and revision ids")
        if self.kind is AnchorKind.WHOLE_OBSERVATION:
            if self.fragment_id is not None or self.range_start is not None or self.range_end is not None:
                raise ValueError("whole observation anchor cannot contain fragment_id or range")
        elif self.kind is AnchorKind.STABLE_FRAGMENT:
            if not self.fragment_id:
                raise ValueError("stable fragment anchor requires fragment_id")
            if self.range_start is not None or self.range_end is not None:
                raise ValueError("stable fragment anchor cannot contain a range")
        elif self.kind is AnchorKind.REVISION_RANGE:
            if (
                self.range_start is None
                or self.range_end is None
                or self.range_start < 0
                or self.range_end <= self.range_start
            ):
                raise ValueError("revision range anchor requires a valid half-open range")
            if self.fragment_id is not None:
                raise ValueError("revision range anchor cannot contain fragment_id")


@dataclass(frozen=True, slots=True)
class FragmentMapping:
    """Provider-backed fragment identity across two observation revisions."""

    observation_id: str
    previous_revision_id: str
    current_revision_id: str
    previous_fragment_id: str
    current_fragment_id: str


@dataclass(frozen=True, slots=True)
class SourceObservation:
    id: str
    source_id: str
    source_unit_id: str
    observation_type: str
    provider_key: str
    locator: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SourceObservationRevision:
    id: str
    observation_id: str
    semantic_hash: str
    content: str
    observed_at: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SourceUnit:
    id: str
    source_id: str
    unit_type: str
    provider_key: str
    locator: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SourceUnitInventoryFilter:
    """Provider-neutral predicates over active Source Unit locator metadata."""

    unit_type: str | None = None
    locator_equals: Mapping[str, str] = field(default_factory=dict)
    observed_from_lte: str | None = None
    observed_to_gte: str | None = None
    observed_to_lt: str | None = None


@dataclass(frozen=True, slots=True)
class SourceUnitInventoryPage:
    units: tuple[SourceUnit, ...]
    next_cursor: str | None = None


@dataclass(frozen=True, slots=True)
class SourceUnitRevision:
    id: str
    source_unit_id: str
    semantic_hash: str
    observation_revision_ids: tuple[str, ...]
    location_hash: str | None = None
    membership_hash: str | None = None
    access_hash: str | None = None
    observed_at: str | None = None


@dataclass(frozen=True, slots=True)
class SourceRelation:
    relation_type: SourceRelationType
    from_id: str
    to_id: str
    provider_relation_id: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RevisionDelta:
    source_unit_id: str
    previous_unit_revision_id: str | None
    current_unit_revision_id: str | None
    axes: frozenset[DeltaAxis]
    coverage: ProjectionCoverage
    changed_anchors: tuple[SourceAnchor, ...] = ()
    added_observation_ids: tuple[str, ...] = ()
    removed_observation_ids: tuple[str, ...] = ()
    fragment_mappings: tuple[FragmentMapping, ...] = ()

    def __post_init__(self) -> None:
        if self.removed_observation_ids and not self.coverage.proves_absence:
            raise ValueError("removed_observation_ids require absence-proving coverage")
        if self.removed_observation_ids and DeltaAxis.MEMBERSHIP not in self.axes:
            raise ValueError("removed observations require the membership delta axis")

    @property
    def requires_extraction(self) -> bool:
        return bool(
            self.axes.intersection({DeltaAxis.SEMANTIC, DeltaAxis.MEMBERSHIP})
            or self.added_observation_ids
            or self.removed_observation_ids
        )


@dataclass(frozen=True, slots=True)
class ProjectionRequest:
    run_id: str
    source_id: str
    source_type: str
    scope: Mapping[str, object]
    run_mode: ProjectionRunMode
    previous_checkpoint: Mapping[str, object] | None = None
    scope_transition: Mapping[str, object] | None = None
    access_context: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ProjectionScopeTransition:
    """Old and target Projection Scope plus its complete-snapshot proof state."""

    id: str
    source_id: str
    previous_scope: Mapping[str, object]
    target_scope: Mapping[str, object]
    status: ProjectionScopeTransitionStatus = ProjectionScopeTransitionStatus.PENDING
    run_id: str | None = None
    coverage: ProjectionCoverage | None = None
    error: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    completed_at: str | None = None

    def __post_init__(self) -> None:
        if not self.id or not self.source_id:
            raise ValueError("scope transition requires id and source_id")
        if dict(self.previous_scope) == dict(self.target_scope):
            raise ValueError("scope transition requires distinct old and target scopes")


@dataclass(frozen=True, slots=True)
class ProjectionEnvelope:
    """One fully fetched connector item plus prior provider-neutral lineage."""

    request: ProjectionRequest
    item: ContentItem
    raw: RawContent
    normalized: NormalizedContent
    prior_unit_revision: SourceUnitRevision | None = None
    prior_observation_revisions: Mapping[str, SourceObservationRevision] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SourceProjection:
    run_id: str
    source_id: str
    source_type: str
    scope: Mapping[str, object]
    coverage: ProjectionCoverage
    observations: tuple[SourceObservation, ...]
    observation_revisions: tuple[SourceObservationRevision, ...]
    source_units: tuple[SourceUnit, ...]
    source_unit_revisions: tuple[SourceUnitRevision, ...]
    relations: tuple[SourceRelation, ...]
    deltas: tuple[RevisionDelta, ...]
    checkpoint: Mapping[str, object]
    # Partial projections retain exact prior revisions for observations that
    # the provider did not return; this run annotation never mutates revision identity.
    carried_observation_revision_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_unique("observation", tuple(item.id for item in self.observations))
        _require_unique("observation revision", tuple(item.id for item in self.observation_revisions))
        _require_unique("source unit", tuple(item.id for item in self.source_units))
        _require_unique("source unit revision", tuple(item.id for item in self.source_unit_revisions))
        unit_ids = {item.id for item in self.source_units}
        observations_by_id = {item.id: item for item in self.observations}
        observation_revisions_by_id = {item.id: item for item in self.observation_revisions}
        unit_revisions_by_id = {item.id: item for item in self.source_unit_revisions}
        carried_revision_ids = set(self.carried_observation_revision_ids)
        if len(carried_revision_ids) != len(self.carried_observation_revision_ids):
            raise ValueError("carried Observation Revision ids must be unique")
        if carried_revision_ids and self.coverage is not ProjectionCoverage.PARTIAL_PROJECTION:
            raise ValueError("carried Observation Revisions require partial projection coverage")
        absent_observation_revision_ids = {
            revision.id
            for revision in self.observation_revisions
            if revision.observation_id not in observations_by_id
        }
        if carried_revision_ids != absent_observation_revision_ids:
            raise ValueError(
                "carried Observation Revision ids must exactly identify revisions without projected Observations"
            )
        carried_revision_unit_ids: dict[str, str] = {}
        for unit_revision in self.source_unit_revisions:
            if len(set(unit_revision.observation_revision_ids)) != len(unit_revision.observation_revision_ids):
                raise ValueError("Source Unit Revision has duplicate Observation Revision membership")
            for revision_id in carried_revision_ids.intersection(unit_revision.observation_revision_ids):
                if revision_id in carried_revision_unit_ids:
                    raise ValueError("carried Observation Revision must belong to exactly one Source Unit Revision")
                carried_revision_unit_ids[revision_id] = unit_revision.source_unit_id
        if carried_revision_ids != set(carried_revision_unit_ids):
            raise ValueError("carried Observation Revision must belong to a Source Unit Revision")
        if any(item.source_id != self.source_id for item in self.source_units):
            raise ValueError("all source units must belong to the projection source")
        if any(item.source_unit_id not in unit_ids for item in self.observations):
            raise ValueError("every observation must reference a projected source unit")
        if any(item.source_id != self.source_id for item in self.observations):
            raise ValueError("all observations must belong to the projection source")
        for item in self.observation_revisions:
            if item.observation_id in observations_by_id:
                continue
            if item.id not in carried_revision_unit_ids:
                raise ValueError("every Observation Revision must reference a projected Observation")
        for unit_revision in self.source_unit_revisions:
            if unit_revision.source_unit_id not in unit_ids:
                raise ValueError("every Source Unit Revision must reference a projected Source Unit")
            for observation_revision_id in unit_revision.observation_revision_ids:
                observation_revision = observation_revisions_by_id.get(observation_revision_id)
                if observation_revision is None:
                    raise ValueError("Source Unit Revision references an unknown Observation Revision")
                observation = observations_by_id.get(observation_revision.observation_id)
                carried_unit_id = carried_revision_unit_ids.get(observation_revision.id)
                if (observation is not None and observation.source_unit_id != unit_revision.source_unit_id) or (
                    observation is None and carried_unit_id != unit_revision.source_unit_id
                ):
                    raise ValueError("Source Unit Revision references another unit's Observation")
        for delta in self.deltas:
            if delta.source_unit_id not in unit_ids:
                raise ValueError("Revision Delta references an unknown Source Unit")
            if delta.current_unit_revision_id is not None:
                current = unit_revisions_by_id.get(delta.current_unit_revision_id)
                if current is None or current.source_unit_id != delta.source_unit_id:
                    raise ValueError("Revision Delta current revision is not projected for its Source Unit")
            for observation_id in delta.added_observation_ids:
                observation = observations_by_id.get(observation_id)
                if observation is None or observation.source_unit_id != delta.source_unit_id:
                    raise ValueError("Revision Delta added Observation is not projected for its Source Unit")
            for anchor in delta.changed_anchors:
                revision = observation_revisions_by_id.get(anchor.observation_revision_id)
                if (
                    revision is None
                    or revision.observation_id != anchor.observation_id
                    or observations_by_id[anchor.observation_id].source_unit_id != delta.source_unit_id
                ):
                    raise ValueError("Revision Delta anchor is not pinned to its projected Source Unit")
            for mapping in delta.fragment_mappings:
                current_revision = observation_revisions_by_id.get(mapping.current_revision_id)
                if (
                    current_revision is None
                    or current_revision.observation_id != mapping.observation_id
                    or observations_by_id[mapping.observation_id].source_unit_id != delta.source_unit_id
                ):
                    raise ValueError("Fragment Mapping current revision is not projected for its Source Unit")


@runtime_checkable
class SourceProjectionAdapter(Protocol):
    async def project(self, envelope: ProjectionEnvelope) -> SourceProjection: ...
    def reconciliation_coverage(
        self,
        *,
        source_type: str,
        transition: ProjectionScopeTransition,
        current_units: tuple[SourceUnit, ...],
        run_attestations: tuple[Mapping[str, object], ...] = (),
    ) -> ProjectionCoverage | None: ...


@runtime_checkable
class ProjectionStore(Protocol):
    async def record_source_projection(self, projection: SourceProjection) -> None: ...
    async def get_source_projection(self, run_id: str) -> SourceProjection | None: ...
    async def get_current_source_unit_revision(
        self,
        source_unit_id: str,
    ) -> SourceUnitRevision | None: ...
    async def get_current_source_observation_revisions(
        self,
        source_unit_id: str,
    ) -> Mapping[str, SourceObservationRevision]: ...
    async def find_source_unit_by_document_id(
        self,
        source_id: str,
        document_id: str,
        *,
        current_only: bool = False,
    ) -> SourceUnit | None: ...
    async def list_source_unit_document_ids(
        self,
        source_unit_id: str,
    ) -> tuple[str, ...]: ...
    async def list_current_source_units_page(
        self,
        source_id: str,
        *,
        filters: SourceUnitInventoryFilter,
        cursor: str | None = None,
        limit: int = 200,
    ) -> SourceUnitInventoryPage: ...


def source_projection_to_payload(projection: SourceProjection) -> dict[str, object]:
    """Canonical JSON-compatible projection payload shared by all adapters."""

    def anchor(item: SourceAnchor) -> dict[str, object]:
        return {
            "kind": item.kind.value,
            "observation_id": item.observation_id,
            "observation_revision_id": item.observation_revision_id,
            "fragment_id": item.fragment_id,
            "range_start": item.range_start,
            "range_end": item.range_end,
        }

    return {
        "run_id": projection.run_id,
        "source_id": projection.source_id,
        "source_type": projection.source_type,
        "scope": dict(projection.scope),
        "coverage": projection.coverage.value,
        "observations": [
            {
                "id": item.id,
                "source_id": item.source_id,
                "source_unit_id": item.source_unit_id,
                "observation_type": item.observation_type,
                "provider_key": item.provider_key,
                "locator": dict(item.locator),
            }
            for item in projection.observations
        ],
        "observation_revisions": [
            {
                "id": item.id,
                "observation_id": item.observation_id,
                "semantic_hash": item.semantic_hash,
                "content": item.content,
                "observed_at": item.observed_at,
                "metadata": dict(item.metadata),
            }
            for item in projection.observation_revisions
        ],
        "source_units": [
            {
                "id": item.id,
                "source_id": item.source_id,
                "unit_type": item.unit_type,
                "provider_key": item.provider_key,
                "locator": dict(item.locator),
            }
            for item in projection.source_units
        ],
        "source_unit_revisions": [
            {
                "id": item.id,
                "source_unit_id": item.source_unit_id,
                "semantic_hash": item.semantic_hash,
                "observation_revision_ids": list(item.observation_revision_ids),
                "location_hash": item.location_hash,
                "membership_hash": item.membership_hash,
                "access_hash": item.access_hash,
                "observed_at": item.observed_at,
            }
            for item in projection.source_unit_revisions
        ],
        "relations": [
            {
                "relation_type": item.relation_type.value,
                "from_id": item.from_id,
                "to_id": item.to_id,
                "provider_relation_id": item.provider_relation_id,
                "metadata": dict(item.metadata),
            }
            for item in projection.relations
        ],
        "deltas": [
            {
                "source_unit_id": item.source_unit_id,
                "previous_unit_revision_id": item.previous_unit_revision_id,
                "current_unit_revision_id": item.current_unit_revision_id,
                "axes": sorted(axis.value for axis in item.axes),
                "coverage": item.coverage.value,
                "changed_anchors": [anchor(value) for value in item.changed_anchors],
                "added_observation_ids": list(item.added_observation_ids),
                "removed_observation_ids": list(item.removed_observation_ids),
                "fragment_mappings": [
                    {
                        "observation_id": value.observation_id,
                        "previous_revision_id": value.previous_revision_id,
                        "current_revision_id": value.current_revision_id,
                        "previous_fragment_id": value.previous_fragment_id,
                        "current_fragment_id": value.current_fragment_id,
                    }
                    for value in item.fragment_mappings
                ],
            }
            for item in projection.deltas
        ],
        "checkpoint": dict(projection.checkpoint),
        "carried_observation_revision_ids": list(projection.carried_observation_revision_ids),
    }


def source_projection_from_payload(payload: Mapping[str, object]) -> SourceProjection:
    """Rehydrate the canonical shared payload returned by a ProjectionStore."""

    def mappings(name: str) -> list[Mapping[str, object]]:
        value = payload.get(name, [])
        if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
            raise ValueError(f"invalid source projection {name}")
        return value

    def anchor(value: Mapping[str, object]) -> SourceAnchor:
        return SourceAnchor(
            kind=AnchorKind(str(value["kind"])),
            observation_id=str(value["observation_id"]),
            observation_revision_id=str(value["observation_revision_id"]),
            fragment_id=_optional_str(value.get("fragment_id")),
            range_start=_optional_int(value.get("range_start")),
            range_end=_optional_int(value.get("range_end")),
        )

    return SourceProjection(
        run_id=str(payload["run_id"]),
        source_id=str(payload["source_id"]),
        source_type=str(payload["source_type"]),
        scope=_mapping(payload.get("scope")),
        coverage=ProjectionCoverage(str(payload["coverage"])),
        observations=tuple(
            SourceObservation(
                id=str(item["id"]),
                source_id=str(item["source_id"]),
                source_unit_id=str(item["source_unit_id"]),
                observation_type=str(item["observation_type"]),
                provider_key=str(item["provider_key"]),
                locator=_mapping(item.get("locator")),
            )
            for item in mappings("observations")
        ),
        observation_revisions=tuple(
            SourceObservationRevision(
                id=str(item["id"]),
                observation_id=str(item["observation_id"]),
                semantic_hash=str(item["semantic_hash"]),
                content=str(item["content"]),
                observed_at=_optional_str(item.get("observed_at")),
                metadata=_mapping(item.get("metadata")),
            )
            for item in mappings("observation_revisions")
        ),
        source_units=tuple(
            SourceUnit(
                id=str(item["id"]),
                source_id=str(item["source_id"]),
                unit_type=str(item["unit_type"]),
                provider_key=str(item["provider_key"]),
                locator=_mapping(item.get("locator")),
            )
            for item in mappings("source_units")
        ),
        source_unit_revisions=tuple(
            SourceUnitRevision(
                id=str(item["id"]),
                source_unit_id=str(item["source_unit_id"]),
                semantic_hash=str(item["semantic_hash"]),
                observation_revision_ids=tuple(str(value) for value in item.get("observation_revision_ids", [])),
                location_hash=_optional_str(item.get("location_hash")),
                membership_hash=_optional_str(item.get("membership_hash")),
                access_hash=_optional_str(item.get("access_hash")),
                observed_at=_optional_str(item.get("observed_at")),
            )
            for item in mappings("source_unit_revisions")
        ),
        relations=tuple(
            SourceRelation(
                relation_type=SourceRelationType(str(item["relation_type"])),
                from_id=str(item["from_id"]),
                to_id=str(item["to_id"]),
                provider_relation_id=_optional_str(item.get("provider_relation_id")),
                metadata=_mapping(item.get("metadata")),
            )
            for item in mappings("relations")
        ),
        deltas=tuple(
            RevisionDelta(
                source_unit_id=str(item["source_unit_id"]),
                previous_unit_revision_id=_optional_str(item.get("previous_unit_revision_id")),
                current_unit_revision_id=_optional_str(item.get("current_unit_revision_id")),
                axes=frozenset(DeltaAxis(str(value)) for value in item.get("axes", [])),
                coverage=ProjectionCoverage(str(item["coverage"])),
                changed_anchors=tuple(anchor(value) for value in item.get("changed_anchors", [])),
                added_observation_ids=tuple(str(value) for value in item.get("added_observation_ids", [])),
                removed_observation_ids=tuple(str(value) for value in item.get("removed_observation_ids", [])),
                fragment_mappings=tuple(
                    FragmentMapping(
                        observation_id=str(value["observation_id"]),
                        previous_revision_id=str(value["previous_revision_id"]),
                        current_revision_id=str(value["current_revision_id"]),
                        previous_fragment_id=str(value["previous_fragment_id"]),
                        current_fragment_id=str(value["current_fragment_id"]),
                    )
                    for value in item.get("fragment_mappings", [])
                ),
            )
            for item in mappings("deltas")
        ),
        checkpoint=_mapping(payload.get("checkpoint")),
        carried_observation_revision_ids=tuple(
            str(value) for value in payload.get("carried_observation_revision_ids", [])
        ),
    )


def resolve_anchor_impact(anchor: SourceAnchor, delta: RevisionDelta) -> ImpactResult:
    """Resolve source impact using only the controlled anchor contract.

    Precision is an optimization.  Missing or unreliable mapping returns
    ``UNKNOWN`` so callers expand toward the whole Source Unit rather than
    silently keeping stale support.
    """

    if anchor.observation_id in delta.removed_observation_ids:
        return ImpactResult.AFFECTED
    if anchor.observation_id in delta.added_observation_ids:
        return ImpactResult.AFFECTED
    if DeltaAxis.SEMANTIC not in delta.axes and DeltaAxis.MEMBERSHIP not in delta.axes:
        return ImpactResult.DISJOINT

    changed = tuple(item for item in delta.changed_anchors if item.observation_id == anchor.observation_id)
    if not changed:
        return ImpactResult.DISJOINT
    if anchor.kind is AnchorKind.WHOLE_OBSERVATION:
        # A whole-observation support anchor is affected by any semantic
        # change to that same Observation.  Returning UNKNOWN here prevented
        # partial Jira/Teams projections from reconciling the exact incumbent
        # they had deterministically changed, leaving stale Memory active.
        return ImpactResult.AFFECTED

    if anchor.kind is AnchorKind.REVISION_RANGE:
        if not all(item.kind is AnchorKind.REVISION_RANGE for item in changed):
            return ImpactResult.UNKNOWN
        for item in changed:
            if anchor.observation_revision_id != item.observation_revision_id:
                return ImpactResult.UNKNOWN
            assert anchor.range_start is not None and anchor.range_end is not None
            assert item.range_start is not None and item.range_end is not None
            if anchor.range_start < item.range_end and item.range_start < anchor.range_end:
                return ImpactResult.AFFECTED
        return ImpactResult.DISJOINT

    assert anchor.fragment_id is not None
    current_fragment_id = anchor.fragment_id
    current_revision_id = anchor.observation_revision_id
    for mapping in delta.fragment_mappings:
        if (
            mapping.observation_id == anchor.observation_id
            and mapping.previous_revision_id == current_revision_id
            and mapping.previous_fragment_id == current_fragment_id
        ):
            current_fragment_id = mapping.current_fragment_id
            current_revision_id = mapping.current_revision_id
            break
    if not delta.fragment_mappings and any(item.observation_revision_id != current_revision_id for item in changed):
        return ImpactResult.UNKNOWN
    for item in changed:
        if item.kind is not AnchorKind.STABLE_FRAGMENT:
            return ImpactResult.UNKNOWN
        if item.fragment_id == current_fragment_id:
            return ImpactResult.AFFECTED
    return ImpactResult.DISJOINT


def _require_unique(label: str, values: tuple[str, ...]) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"duplicate {label} id")


def _mapping(value: object) -> Mapping[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("invalid source projection mapping")
    return dict(value)


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("invalid source projection integer")
    return value
