"""Teams-owned immutable record fields and comparison semantics."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from memforge.source_adapters.contracts import CanonicalRecordField, CanonicalRecordSchema, _normalized_utc_timestamp
from memforge.source_projection import ProjectionCoverage, ProjectionScopeAttestation, ProjectionScopeTransition, SourceUnit
from memforge.source_projection_config import projection_scope_fingerprint
from memforge.local_agent.source_contract import TEAMS_ROLLING_RETENTION_PRESETS


CANONICAL_RECORD_SCHEMAS: Mapping[tuple[str, int], CanonicalRecordSchema] = {
    ("teams-message", 1): CanonicalRecordSchema(
        name="teams-message",
        version=1,
        fields=(CanonicalRecordField("/content", nested_profile="markdown-structural"),),
        tombstone_pointer="/deleted",
    ),
}


MESSAGE_SCHEMA = replace(
    CANONICAL_RECORD_SCHEMAS[("teams-message", 1)], version=2,
    fields=(CanonicalRecordField("/content", nested_profile="markdown-structural"),
            *(CanonicalRecordField(pointer, contextual=True) for pointer in
              ("/from", "/time", "/edited_time", "/reply_to_id", "/conversation_id", "/conversation_name"))),
    framing_fields=(("/from", "Reported sender"), ("/time", "Message created"),
                    ("/edited_time", "Message edited"), ("/reply_to_id", "Reply referent"),
                    ("/conversation_id", "Conversation ID"), ("/conversation_name", "Conversation")),
    model_interpretation="Sender and message times are supplied source framing. Missing sender or a reported Unknown sender does not establish identity. Message time is not a rule effective date. An authenticated reply referent establishes a relationship; chronological adjacency alone does not.",
)
CANONICAL_RECORD_SCHEMAS = {**CANONICAL_RECORD_SCHEMAS, (MESSAGE_SCHEMA.name, MESSAGE_SCHEMA.version): MESSAGE_SCHEMA}


def reply_referent(message: Mapping[str, object]) -> object:
    """Retain a supplied immediate referent; never infer a reply from ordering."""
    referent = (message.get("reply_to_id") or message.get("replyToId")
                or message.get("parentMessageId") or message.get("rootMessageId"))
    return referent if referent != message.get("id") else None


def message_record(message: Mapping[str, object], conversation: Mapping[str, object], *, conversation_id: str) -> dict[str, object]:
    """Retain only supplied message framing and scoped reply identity."""
    return {
        "content": message.get("content"), "attachments": message.get("attachments"),
        "deleted": message.get("deletedDateTime") or message.get("deleted_at"),
        "from": message.get("from"), "time": message.get("time"),
        "edited_time": message.get("edited_time"),
        "reply_to_id": reply_referent(message),
        "conversation_id": conversation_id,
        "conversation_name": conversation.get("conversation_name") or conversation.get("channel_name"),
        "representation": f"{MESSAGE_SCHEMA.name}:{MESSAGE_SCHEMA.version}",
    }

def project_native(*, source_id, item, native, normalized):
    """Build provider identity, immutable Observations, relations and coverage."""
    from memforge.source_projection import ProjectionCoverage, SourceRelationType
    from memforge.source_adapters.contracts import _ObservationInput, _NativeProjection, _unit_title, _canonical_json, _normalized_utc_timestamp
    from memforge.genes.teams_gene import LOCAL_AGENT_TEAMS_PACKAGE_KIND

    data = native if isinstance(native, dict) else {}
    if data.get("package_kind") == LOCAL_AGENT_TEAMS_PACKAGE_KIND and isinstance(data.get("raw_payload"), dict):
        data = data["raw_payload"]
    window_id = str(item.extra.get("window_id") or data.get("window_id") or item.item_id)
    conversation_id = str(item.extra.get("conversation_id") or data.get("conversation_id") or "")
    from memforge.local_agent.teams_contract import (
        teams_message_source_time,
        validate_teams_canonical_messages,
    )

    messages = data.get("messages") if isinstance(data.get("messages"), list) else []
    if messages:
        messages = list(validate_teams_canonical_messages(messages))
    inputs = []
    relations = []
    previous_key = None
    for message in messages:
        message_id = str(message["id"])

        semantic_message = message_record(message, data, conversation_id=conversation_id)
        inputs.append(
            _ObservationInput(
                "message",
                message_id,
                _canonical_json(semantic_message),
                semantic_message,
                {"conversation_id": conversation_id},
                teams_message_source_time(message),
                {"claim_evidence_scope": "atomic"},
            )
        )
        reply_to = reply_referent(message)
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
        # A tombstoned window has no live Unit left to name. A live window is named by its
        # conversation and its start; its end moves with every new message, so it is no part of the name.
        title=None if tombstoned else _unit_title(
            "Teams conversation",
            ("Conversation type", data.get("conversation_type")),
            ("Team", data.get("team_name")),
            ("Conversation", data.get("conversation_name") or data.get("channel_name")),
            ("From", observed_from),
        ),
    )


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


def authoritative_unit_coverage(
    *,
    native: object,
    coverage: ProjectionCoverage,
    projected_scope: Mapping[str, object],
    scope_attestations: tuple[ProjectionScopeAttestation, ...],
) -> ProjectionCoverage:
    """Apply run authority only where the provider unit contract supports it."""

    from memforge.genes.teams_gene import LOCAL_AGENT_TEAMS_PACKAGE_KIND

    configured_scope = projected_scope.get("configured_scope")
    teams_native = (
        native.get("raw_payload")
        if isinstance(native, Mapping) and isinstance(native.get("raw_payload"), Mapping)
        else native
    )
    if (
        isinstance(teams_native, Mapping)
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
        isinstance(native, Mapping)
        and native.get("package_kind") == LOCAL_AGENT_TEAMS_PACKAGE_KIND
        and isinstance(native.get("raw_payload"), Mapping)
    ):
        # A force-full local collection attempt is validated against its
        # immutable package manifest before replay. That source-wide proof also
        # makes each canonical window package a complete snapshot of its unit.
        return ProjectionCoverage.COMPLETE_SNAPSHOT
    return coverage


def reconciliation_coverage(*, transition: ProjectionScopeTransition, current_units: tuple[SourceUnit, ...], run_attestations: tuple[ProjectionScopeAttestation, ...] = ()) -> ProjectionCoverage | None:
    """Prove window absence from authenticated target-scope collection."""
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
