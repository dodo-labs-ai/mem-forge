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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Mapping

from memforge.github_repo_utils import build_github_repo_doc_id
from memforge.models import ContentItem, NormalizedContent, RawContent
from memforge.source_projection import (
    AnchorKind,
    DeltaAxis,
    ProjectionCoverage,
    ProjectionEnvelope,
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
from memforge.source_projection_config import projection_scope_fingerprint


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
    observation_type: str
    provider_key: str
    content: str
    semantic_value: object
    locator: Mapping[str, object]
    observed_at: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


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
            scope=request.scope,
            access_context=request.access_context,
            prior_unit_revision=envelope.prior_unit_revision,
            prior_observation_revisions=envelope.prior_observation_revisions,
        )

    def reconciliation_coverage(
        self,
        *,
        source_type: str,
        transition: ProjectionScopeTransition,
        current_units: tuple[SourceUnit, ...],
        run_attestations: tuple[Mapping[str, object], ...] = (),
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
        time_scope_fields = {"max_age_days"}
        membership_partition_fields = {
            "conversation_gap_minutes",
            "max_block_messages",
        }
        supported_fields = selector_fields | time_scope_fields | membership_partition_fields
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
        if any(unit.unit_type not in {"teams_window", "teams_scope_attestation"} for unit in current_units):
            return None
        if (changed_fields & selector_fields or target_conversations) and not all(
            str(unit.locator.get("conversation_id") or "").strip() in target_conversations
            for unit in current_window_units
        ):
            return None

        target_fingerprint = projection_scope_fingerprint(transition.target_scope)
        if changed_fields & membership_partition_fields:
            target_partition_fingerprint = _teams_partition_scope_fingerprint(transition.target_scope)
            if not all(
                unit.locator.get("projection_scope_fingerprint") == target_fingerprint
                or unit.locator.get("partition_scope_fingerprint") == target_partition_fingerprint
                for unit in current_window_units
            ):
                return None
        if changed_fields & time_scope_fields:
            if not all(_teams_unit_attests_time_scope(unit, target_fingerprint) for unit in current_window_units):
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
    scope: Mapping[str, object] | None = None,
    access_context: Mapping[str, object] | None = None,
    prior_unit_revision: SourceUnitRevision | None = None,
    prior_observation_revisions: Mapping[str, SourceObservationRevision] | None = None,
) -> SourceProjection:
    """Project one completely fetched Source Unit.

    The run scope is exactly this unit. Source-wide absence is handled by the
    enclosing manifest projection; a unit projection never claims another unit
    was deleted merely because it was not part of this call.
    """

    prior_observation_revisions = prior_observation_revisions or {}
    native = _native_payload(raw)
    projected_scope = dict(scope or {})
    unit_type, provider_key, observations_input, relations_input, coverage, locator = _project_native(
        source_id=source_id,
        source_type=source_type,
        item=item,
        native=native,
        normalized=normalized,
    )
    coverage = _provider_authoritative_unit_coverage(
        source_type=source_type,
        native=native,
        coverage=coverage,
        projected_scope=projected_scope,
    )
    if source_type == "teams":
        locator = _teams_scope_attested_locator(
            locator=locator,
            native=native,
            coverage=coverage,
            projected_scope=projected_scope,
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
    observations: list[SourceObservation] = []
    revisions: list[SourceObservationRevision] = []
    carried_revision_ids: list[str] = []
    for value in observations_input:
        observation_id = _stable_id("obs", unit_id, value.observation_type, value.provider_key)
        semantic_hash = _canonical_hash(value.semantic_value)
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
        )
        prior_revision = prior_observation_revisions.get(observation_id)
        # Revision identity is semantic. Operational metadata enrichment under
        # an unchanged semantic hash must preserve the exact immutable row.
        revisions.append(
            prior_revision
            if (
                prior_revision is not None
                and prior_revision.observation_id == observation_id
                and prior_revision.semantic_hash == semantic_hash
            )
            else projected_revision
        )
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
    access_hash = _canonical_hash(dict(access_context)) if access_context else None
    unit_revision_id = _stable_id(
        "unitrev",
        unit_id,
        semantic_hash,
        location_hash,
        membership_hash,
        access_hash,
    )
    unit_revision = SourceUnitRevision(
        id=unit_revision_id,
        source_unit_id=unit_id,
        semantic_hash=semantic_hash,
        location_hash=location_hash,
        membership_hash=membership_hash,
        access_hash=access_hash,
        observation_revision_ids=tuple(item.id for item in revisions),
        observed_at=item.last_modified.isoformat(),
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


def _teams_scope_attested_locator(
    *,
    locator: Mapping[str, object],
    native: object,
    coverage: ProjectionCoverage,
    projected_scope: Mapping[str, object],
) -> Mapping[str, object]:
    configured_scope = projected_scope.get("configured_scope")
    if not isinstance(configured_scope, Mapping):
        return locator
    fingerprint = projection_scope_fingerprint(configured_scope)
    result = dict(locator)
    result["partition_scope_fingerprint"] = _teams_partition_scope_fingerprint(configured_scope)
    if coverage is ProjectionCoverage.COMPLETE_SNAPSHOT:
        result["projection_scope_fingerprint"] = fingerprint
        return result
    if not isinstance(native, Mapping):
        return result
    covered_from = _normalized_utc_timestamp(native.get("_scope_coverage_from"))
    covered_to = _normalized_utc_timestamp(native.get("_scope_coverage_to"))
    if covered_from and covered_to and covered_from <= covered_to:
        result.update(
            {
                "time_scope_fingerprint": fingerprint,
                "time_scope_coverage_from": covered_from,
                "time_scope_coverage_to": covered_to,
            }
        )
    return result


def _provider_authoritative_unit_coverage(
    *,
    source_type: str,
    native: object,
    coverage: ProjectionCoverage,
    projected_scope: Mapping[str, object],
) -> ProjectionCoverage:
    """Apply run authority only where the provider unit contract supports it."""

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


def _teams_partition_scope_fingerprint(scope: Mapping[str, object]) -> str:
    values: dict[str, object] = {}
    for field_name, default in (
        ("conversation_gap_minutes", 60),
        ("max_block_messages", 100),
    ):
        try:
            values[field_name] = int(scope.get(field_name, default))
        except (TypeError, ValueError):
            values[field_name] = default
    return projection_scope_fingerprint(values)


def _teams_run_attests_target_scope(
    *,
    transition: ProjectionScopeTransition,
    target_conversations: set[str],
    run_attestations: tuple[Mapping[str, object], ...],
) -> bool:
    """Require one exact, successful, same-attempt poll per target conversation."""

    target_fingerprint = projection_scope_fingerprint(transition.target_scope)
    expected_conversations = sorted(target_conversations)
    by_conversation: dict[str, Mapping[str, object]] = {}
    attempt_ids: set[str] = set()
    for attestation in run_attestations:
        conversation_id = str(attestation.get("conversation_id") or "").strip()
        target_values = attestation.get("target_conversation_ids")
        poll = attestation.get("poll")
        attempt_id = str(attestation.get("collection_attempt_id") or "").strip()
        if (
            conversation_id not in target_conversations
            or str(attestation.get("transition_id") or "").strip() != transition.id
            or attestation.get("target_scope_fingerprint") != target_fingerprint
            or target_values != expected_conversations
            or not attempt_id
            or not isinstance(poll, Mapping)
            or str(poll.get("raw_conversation_id") or "").strip() != conversation_id
            or str(poll.get("access_probe_status") or "").strip().lower() != "ok"
        ):
            return False
        stop_reason = str(poll.get("stop_reason") or "").strip()
        if stop_reason == "no_backward_link":
            if poll.get("pagination_complete") is not True:
                return False
        elif stop_reason == "cutoff_reached":
            covered_from = _normalized_utc_timestamp(poll.get("absence_covered_from"))
            covered_to = _normalized_utc_timestamp(poll.get("absence_covered_to"))
            if not covered_from or not covered_to or covered_from > covered_to:
                return False
        else:
            return False
        if conversation_id in by_conversation:
            return False
        by_conversation[conversation_id] = attestation
        attempt_ids.add(attempt_id)
    return set(by_conversation) == target_conversations and len(attempt_ids) == 1


def _teams_unit_attests_time_scope(
    unit: SourceUnit,
    target_fingerprint: str,
) -> bool:
    locator = unit.locator
    if locator.get("projection_scope_fingerprint") == target_fingerprint:
        return True
    if locator.get("time_scope_fingerprint") != target_fingerprint:
        return False
    covered_from = _normalized_utc_timestamp(locator.get("time_scope_coverage_from"))
    covered_to = _normalized_utc_timestamp(locator.get("time_scope_coverage_to"))
    observed_to = _normalized_utc_timestamp(locator.get("observed_to"))
    return bool(covered_from and covered_to and observed_to and covered_from <= observed_to <= covered_to)


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
) -> tuple[
    str,
    str,
    tuple[_ObservationInput, ...],
    tuple[tuple[SourceRelationType, str, str, str | None, Mapping[str, object]], ...],
    ProjectionCoverage,
    Mapping[str, object],
]:
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
        return (
            "confluence_page",
            page_id,
            (
                _ObservationInput(
                    "page_body",
                    f"{page_id}:body",
                    semantic_content,
                    semantic_value,
                    {},
                ),
            ),
            relations,
            ProjectionCoverage.COMPLETE_SNAPSHOT,
            {
                "page_id": page_id,
                "space_key": item.extra.get("space_key") or item.space_or_project,
                "parent_page_id": parent_id or None,
                "url": item.source_url,
            },
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
        core_value = {
            "summary": fields.get("summary"),
            "description": fields.get("description"),
            "status": fields.get("status"),
            "priority": fields.get("priority"),
            "assignee": fields.get("assignee"),
            "labels": fields.get("labels"),
            "resolution": fields.get("resolution"),
        }
        inputs = [
            _ObservationInput(
                "issue_core",
                f"{issue_id}:core",
                _canonical_json(core_value),
                core_value,
                {"issue_key": issue_key},
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
                )
            )
            relations.append((SourceRelationType.PRECEDES, previous_key, comment_id, None, {}))
            previous_key = comment_id
        changelog = data.get("changelog") if isinstance(data.get("changelog"), dict) else {}
        histories = changelog.get("histories", [])
        for history in histories if isinstance(histories, list) else []:
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
        changelog_total = changelog.get("total")
        changelog_incomplete = isinstance(changelog_total, int) and changelog_total > len(
            histories if isinstance(histories, list) else []
        )
        coverage = (
            ProjectionCoverage.PARTIAL_PROJECTION
            if data.get("_comments_truncated") or data.get("_changelog_truncated") or changelog_incomplete
            else ProjectionCoverage.COMPLETE_SNAPSHOT
        )
        return (
            "jira_issue",
            issue_id,
            tuple(inputs),
            tuple(relations),
            coverage,
            {"issue_id": issue_id, "issue_key": issue_key, "url": item.source_url},
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
        return (
            "github_file",
            f"{repo}:{lineage}",
            (
                _ObservationInput(
                    "file_content",
                    "content",
                    normalized.markdown_body,
                    normalized.markdown_body,
                    {"path": path},
                ),
            ),
            relations,
            ProjectionCoverage.COMPLETE_SNAPSHOT,
            {"repository": repo, "path": path, "ref": item.extra.get("repo_ref"), "url": item.source_url},
        )
    if source_type == "github_pages":
        canonical_url = str(
            item.extra.get("canonical_url") or normalized.source_semantics.get("canonical_url") or item.source_url
        )
        semantic_value = native if isinstance(native, str) else normalized.markdown_body
        semantic_content = normalized.markdown_body
        return (
            "rendered_page",
            canonical_url,
            (_ObservationInput("page_content", "content", semantic_content, semantic_value, {}),),
            (),
            ProjectionCoverage.COMPLETE_SNAPSHOT,
            {"canonical_url": canonical_url, "title": item.title},
        )
    if source_type == "local_markdown":
        data = native if isinstance(native, dict) else {}
        vault = str(data.get("vault_id") or item.space_or_project or "default")
        path = str(data.get("relative_path") or item.extra.get("relative_path") or item.item_id)
        lineage = str(data.get("file_lineage_id") or item.extra.get("file_lineage_id") or path)
        body = str(data.get("markdown") or normalized.markdown_body)
        return (
            "local_file",
            f"{vault}:{lineage}",
            (_ObservationInput("file_content", "content", body, body, {"path": path}),),
            (),
            ProjectionCoverage.COMPLETE_SNAPSHOT,
            {"vault_id": vault, "path": path, "url": item.source_url},
        )
    if source_type == "teams":
        data = native if isinstance(native, dict) else {}
        if data.get("package_kind") and isinstance(data.get("raw_payload"), dict):
            data = data["raw_payload"]
        if data.get("_scope_attestation") is True:
            conversation_id = str(data.get("conversation_id") or "").strip()
            poll = data.get("poll")
            if not conversation_id or not isinstance(poll, Mapping):
                raise ValueError("Teams scope attestation is invalid")
            return (
                "teams_scope_attestation",
                conversation_id,
                (),
                (),
                ProjectionCoverage.COMPLETE_SNAPSHOT,
                {
                    "conversation_id": conversation_id,
                    "transition_id": data.get("transition_id"),
                    "target_scope_fingerprint": data.get("target_scope_fingerprint"),
                    "target_conversation_ids": data.get("target_conversation_ids"),
                    "collection_attempt_id": data.get("collection_attempt_id"),
                    "poll": dict(poll),
                },
            )
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
            or (observed_times[0] if observed_times else "")
        ).strip()
        observed_to = str(
            item.extra.get("block_end")
            or data.get("last_message_time")
            or (observed_times[-1] if observed_times else "")
        ).strip()
        observed_from = _normalized_utc_timestamp(observed_from) or observed_from
        observed_to = _normalized_utc_timestamp(observed_to) or observed_to
        return (
            "teams_window",
            window_id,
            tuple(inputs),
            tuple(relations),
            coverage,
            {
                "conversation_id": conversation_id,
                "window_id": window_id,
                "observed_from": observed_from or None,
                "observed_to": observed_to or None,
                "url": item.source_url,
            },
        )
    if source_type == "agent_session":
        data = native if isinstance(native, dict) else {}
        receipt = data.get("receipt") if isinstance(data.get("receipt"), dict) else {}
        window_id = str(data.get("doc_id") or item.item_id)
        body = str(data.get("markdown") or normalized.markdown_body)
        return (
            "agent_session_window",
            window_id,
            (_ObservationInput("session_summary", window_id, body, body, {}),),
            (),
            ProjectionCoverage.PARTIAL_PROJECTION,
            {
                "client": receipt.get("client"),
                "session_id": receipt.get("session_id"),
                "history_window_kind": receipt.get("history_window_kind"),
                "url": item.source_url,
            },
        )
    # Extension-safe fallback for document-like genes that have not yet opted
    # into a richer native projection.  It deliberately claims only partial
    # coverage, so it can drive semantic change detection but can never prove
    # that an omitted observation or source unit was deleted.
    body = normalized.markdown_body
    return (
        "generic_document",
        item.item_id,
        (_ObservationInput("document_content", item.item_id, body, body, {}),),
        (),
        ProjectionCoverage.PARTIAL_PROJECTION,
        {
            "item_id": item.item_id,
            "url": item.source_url,
            "title": item.title,
            "source_type": source_type,
        },
    )


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
