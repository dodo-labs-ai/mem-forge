"""Shared execution contract for local-agent-backed sources."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from memforge.conversations.time_scope import MAX_INITIAL_HISTORY_DAYS
from memforge.source_types import (
    SOURCE_EXECUTOR_DEVICE,
    SOURCE_EXECUTOR_SERVER,
    registered_source_types,
    source_type_descriptor,
)


DOCUMENT_ACQUISITION_OPERATION = "document_acquisition"
"""The local-agent operation that acquires a document Source on its owner's device with the type's own Gene."""
ACQUISITION_CONTRACT_CHANGED = "acquisition_contract_changed"
"""Why a device run's submission was refused: the server's acquisition contract for the type changed."""
LOCAL_AGENT_SYNC_OPERATIONS = frozenset(
    {
        DOCUMENT_ACQUISITION_OPERATION,
        *(
            descriptor.local_agent.operation
            for descriptor in registered_source_types()
            if descriptor.local_agent is not None
        ),
    }
)
# ``retry_failed_items`` marks a job a person requested: the sync its snapshot
# starts retries every failed Document of the Source. ``dry_run`` asks a
# conversation Source's job to report what its attempt would change instead of
# processing it.
LOCAL_AGENT_SYNC_PAYLOAD_CONTROL_FIELDS = frozenset(
    {
        "dry_run",
        "force_full_sync",
        "retry_failed_items",
    }
)
LOCAL_AGENT_JOB_MAX_ATTEMPTS = 5

TEAMS_ROLLING_RETENTION_PRESETS = (365, 730, 1095)


@dataclass(frozen=True, slots=True)
class LocalAgentLease:
    """A current lease of one local-agent job attempt, as the server validates it for each submission.

    ``payload`` is the job's payload, and ``leased_at`` when the attempt was
    leased: a device run reads its provider from then on.
    ``acquisition_contract`` is the acquisition contract a
    ``document_acquisition`` attempt was leased under, and ``None`` for any
    other job.
    """

    payload: Mapping[str, Any]
    leased_at: datetime
    acquisition_contract: int | None = None


def teams_config_within_history_bound(config: Mapping[str, Any]) -> dict[str, Any] | None:
    """A stored Teams configuration with its Initial History cut to the bound, or ``None`` when it is within it.

    The bound is the one every conversation Source shares
    (:data:`~memforge.conversations.time_scope.MAX_INITIAL_HISTORY_DAYS`).

    Initial History is read as the Teams Gene reads it: ``initial_history_days``,
    otherwise a stored ``max_age_days``.
    """

    try:
        days = int(config.get("initial_history_days", config.get("max_age_days")))
    except (TypeError, ValueError):
        return None
    if days <= MAX_INITIAL_HISTORY_DAYS:
        return None
    return {**config, "initial_history_days": MAX_INITIAL_HISTORY_DAYS}


TEAMS_CONVERSATION_SELECTOR_FIELDS = (
    "conversation_ids",
    "channels",
    "group_chats",
    "individual_chats",
)
_TEAMS_CONVERSATION_ID_RE = re.compile(r"^19:[^\s@]+@[^\s@]+$")


def jira_config_with_auth_mode(config: Mapping[str, Any]) -> dict[str, Any] | None:
    """A stored Jira configuration that names its authentication method, or ``None`` when it already does.

    The named method is the one a configuration without it authenticated
    with when it was stored: Jira's local agent sync (``sync_mode``
    ``local_agent``) used the shared browser session; otherwise a
    configuration with no ``auth_mode`` key uses its token when it holds one
    (``pat`` or ``pat_encrypted``), and a blank ``auth_mode`` or no token
    means the browser session.
    """

    configured = config.get("auth_mode")
    if str(configured or "").strip():
        return None
    local_sync = str(config.get("sync_mode") or "").strip().lower() == "local_agent"
    holds_token = bool(config.get("pat") or config.get("pat_encrypted"))
    auth_mode = "pat" if configured is None and holds_token and not local_sync else "browser_cookie"
    return {**config, "auth_mode": auth_mode}


class SourceSyncRunReceiptError(ValueError):
    """Raised when a successful local sync cannot identify its durable server run."""


def source_processing_receipt(sync_result: object) -> dict[str, str]:
    """Project the durable server-run identity returned after local collection.

    An explicit server error has no receipt. Every non-error response must name
    one durable SourceSyncRun so the broker cannot report success before the
    server-side lifecycle transaction can be followed to completion.
    """

    if not isinstance(sync_result, Mapping):
        raise SourceSyncRunReceiptError("successful source processing response must be an object")
    if sync_result.get("error"):
        return {}
    run_id = _stable_execution_code(sync_result.get("run_id"))
    if run_id is None:
        raise SourceSyncRunReceiptError("successful source processing response omitted run_id")
    return {"source_sync_run_id": run_id}


def source_sync_run_id_from_completion(result: object) -> str:
    """Return the immutable SourceSyncRun receipt from a successful broker result."""

    if not isinstance(result, Mapping):
        raise SourceSyncRunReceiptError("successful local-agent sync result must be an object")
    run_id = _stable_execution_code(result.get("source_sync_run_id"))
    if run_id is None:
        raise SourceSyncRunReceiptError("successful local-agent sync result omitted source_sync_run_id")
    return run_id


def _stable_execution_code(value: object) -> str | None:
    normalized = str(value or "").strip()
    if not normalized or any(character.isspace() for character in normalized):
        return None
    return normalized


def is_direct_teams_conversation_id(value: object) -> bool:
    """Return whether a selector is already a stable Teams conversation id."""

    normalized = str(value or "").strip()
    return bool(_TEAMS_CONVERSATION_ID_RE.fullmatch(normalized))


def canonical_teams_conversation_ids(
    config: Mapping[str, Any] | str | None,
    *,
    require_nonempty: bool = False,
) -> tuple[str, ...]:
    """Collapse every accepted Teams selector field into verified direct ids.

    Legacy selector fields historically also accepted display names.  Those names
    are not stable provider identity and therefore must never reach lifecycle or
    inventory reconciliation as if they were conversation ids.
    """

    normalized_config = _source_config(config)
    result: list[str] = []
    seen: set[str] = set()
    for field in TEAMS_CONVERSATION_SELECTOR_FIELDS:
        raw_values = normalized_config.get(field, ())
        if isinstance(raw_values, str):
            values = [value.strip() for value in raw_values.split(",") if value.strip()]
        elif isinstance(raw_values, (list, tuple)):
            values = [str(value).strip() for value in raw_values if str(value).strip()]
        elif raw_values in (None, ()):
            values = []
        else:
            raise ValueError("teams conversation selectors must be strings or lists")
        for value in values:
            if not is_direct_teams_conversation_id(value):
                raise ValueError("teams_sync_requires_direct_conversation_ids")
            if value not in seen:
                seen.add(value)
                result.append(value)
    if require_nonempty and not result:
        raise ValueError("teams_sync_requires_direct_conversation_ids")
    return tuple(result)


def local_agent_completion_status(
    status: str,
    *,
    retryable: bool,
    attempt_count: int,
) -> str:
    """Return the durable broker status for a daemon completion."""
    if status == "failed" and retryable and attempt_count < LOCAL_AGENT_JOB_MAX_ATTEMPTS:
        return "queued"
    return "succeeded" if status == "succeeded" else "failed"


def local_agent_retry_delay(
    status: str,
    *,
    retryable: bool,
    attempt_count: int,
) -> timedelta | None:
    """Return the server-controlled delay before another daemon lease."""
    if local_agent_completion_status(
        status,
        retryable=retryable,
        attempt_count=attempt_count,
    ) != "queued":
        return None
    return timedelta(hours=1 if attempt_count == 1 else 12)


def local_agent_sync_snapshot_id(job_id: object, attempt_count: object) -> str:
    """Identify one complete collection attempt without sharing partial membership."""
    normalized_job_id = str(job_id or "").strip()
    try:
        normalized_attempt = int(attempt_count)
    except (TypeError, ValueError) as exc:
        raise ValueError("cloud sync job is missing attempt_count") from exc
    if not normalized_job_id:
        raise ValueError("cloud sync job is missing job_id")
    if normalized_attempt < 1:
        raise ValueError("cloud sync job is missing attempt_count")
    return f"{normalized_job_id}:attempt:{normalized_attempt}"


def local_agent_collection_attempt_id(
    source_type: object,
    job_id: object,
    attempt_count: object,
    requested_snapshot_id: object = None,
) -> str:
    """Return the server-owned immutable identity for one collection attempt."""

    normalized_source_type = str(source_type or "").strip().lower()
    requested = str(requested_snapshot_id or "").strip()
    descriptor = source_type_descriptor(normalized_source_type)
    if descriptor is None or descriptor.local_agent is None:
        raise ValueError(f"{normalized_source_type or 'unknown'} sources are not collected by a local agent")
    canonical = local_agent_sync_snapshot_id(job_id, attempt_count)
    if requested and requested != canonical:
        raise ValueError("local agent snapshot does not match the leased job attempt")
    return canonical


def _source_config(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return {}
        if isinstance(parsed, Mapping):
            return parsed
    return {}


def local_agent_sync_operation(
    source_type: str,
    config: Mapping[str, Any] | str | None,
) -> str | None:
    """Return the daemon sync operation for a canonical source configuration.

    A document Source that replays an evaluation snapshot set runs on the
    server with its type's recorded Gene over the set, whoever collects its
    type otherwise; a conversation Source replays its set through the device
    executor.
    """
    from memforge.evals.source_snapshots import SNAPSHOT_CONFIG_FIELD
    from memforge.genes.family import CollectionFamilyKind

    descriptor = source_type_descriptor(str(source_type or "").strip().lower())
    collection = descriptor.local_agent if descriptor is not None else None
    source_config = _source_config(config)
    if collection is None or not collection.collects(source_config):
        return None
    if SNAPSHOT_CONFIG_FIELD in source_config and descriptor.family is CollectionFamilyKind.DOCUMENT:
        return None
    return collection.operation


def is_local_agent_backed_source(source: Mapping[str, Any]) -> bool:
    return (
        local_agent_sync_operation(
            str(source.get("type") or ""),
            source.get("config"),
        )
        is not None
    )


def source_execution_descriptor(source: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical execution contract exposed to source clients.

    ``kind`` is ``local_agent`` for a Source collected on its owner's device,
    by the device executor (ADR 0058) or a type's local-agent operation, and
    ``server`` otherwise.
    """
    operation = source_collection_operation(source)
    return {
        "kind": "local_agent" if operation is not None else "server",
        "operation": operation,
        # Deprecated (admin-ui-v1): V1 setup reads it; always empty, removed with V1.
        "immutable_config_fields": [],
    }


def source_collection_operation(source: Mapping[str, Any]) -> str | None:
    """The local-agent operation that collects the Source on its owner's device, or ``None`` when the server does.

    A device Source is acquired by ``document_acquisition``; a Source on its
    type's local-agent path by that type's operation.
    """

    if source_runs_on_device(source):
        return DOCUMENT_ACQUISITION_OPERATION
    return local_agent_sync_operation(str(source.get("type") or ""), source.get("config"))


def source_executor(source: Mapping[str, Any]) -> str:
    """Where the Source's acquisition runs: ``server`` or ``device`` (ADR 0058)."""

    return SOURCE_EXECUTOR_DEVICE if source_runs_on_device(source) else SOURCE_EXECUTOR_SERVER


def source_runs_on_device(source: Mapping[str, Any]) -> bool:
    """Whether the Source's acquisition runs on its execution owner's device (ADR 0058).

    The server holds no credential for such a Source and never reaches its
    provider; only the device does.
    """

    return source.get("executor") == SOURCE_EXECUTOR_DEVICE


def execution_owner_user_id(source: Mapping[str, Any]) -> str | None:
    value = str(source.get("execution_owner_user_id") or "").strip()
    return value or None


def local_agent_job_config(
    source_type: str,
    config: Mapping[str, Any] | str | None,
) -> dict[str, Any]:
    """Return connector config safe and useful for the owner's daemon.

    A Source that replays an evaluation snapshot set also hands the device
    the set it replays.
    """
    descriptor = source_type_descriptor(str(source_type or "").strip().lower())
    collection = descriptor.local_agent if descriptor is not None else None
    if collection is None:
        return {}
    from memforge.evals.source_snapshots import SNAPSHOT_CONFIG_FIELD

    source_config = dict(_source_config(config))
    if collection.job_config_defaults is not None:
        source_config = collection.job_config_defaults(source_config)
    return {
        key: value
        for key, value in source_config.items()
        if key in collection.job_config_fields or key == SNAPSHOT_CONFIG_FIELD
    }


LOCAL_AGENT_TRANSITION_PAYLOAD_FIELD = "projection_scope_transition"
"""The job payload field naming the Source's open scope transition and its canonical scopes."""
LOCAL_AGENT_EXCLUDED_SCOPES_PAYLOAD_FIELD = "excluded_scopes"
"""The job payload field carrying the configuration of each scope an open transition excludes.

Each entry is ``{"scope_key": ..., "config": ...}``: the scope's fingerprint,
which the device's manifest names its listing by, and the job configuration
that lists that scope.
"""


async def local_agent_sync_job_payload(
    db: Any,
    source: Mapping[str, Any],
    request_payload: Mapping[str, Any] | str | None = None,
) -> dict[str, Any]:
    """The daemon sync-job payload for a saved Source, as of the moment the job is enqueued.

    Every place that enqueues a local-agent sync job builds its payload here:
    the saved collection configuration, the request's control fields, and,
    while a scope transition is open, that transition.
    """

    from memforge.source_projection_config import canonical_projection_scope, canonical_scope_transition

    source_type = str(source.get("type") or "").strip()
    config = _source_config(source.get("config"))
    payload = local_agent_job_config(source_type, config)
    payload.update(
        {
            key: value
            for key, value in _source_config(request_payload).items()
            if key in LOCAL_AGENT_SYNC_PAYLOAD_CONTROL_FIELDS
        }
    )
    payload["source_id"] = str(source.get("id") or "").strip()
    payload["source_type"] = source_type
    payload["source_config_revision"] = local_agent_source_config_revision(source)
    transition = await db.get_open_projection_scope_transition(payload["source_id"])
    if transition is None:
        return payload
    transition = canonical_scope_transition(source_type, transition, canonical_projection_scope(source_type, config))
    payload[LOCAL_AGENT_TRANSITION_PAYLOAD_FIELD] = {
        "id": transition.id,
        "previous_scope": dict(transition.previous_scope),
        "target_scope": dict(transition.target_scope),
    }
    return payload


def local_agent_source_config_revision(source: Mapping[str, Any]) -> str:
    """Fingerprint the saved configuration one daemon job collects with.

    It is the configuration the job hands the device: the fields of the
    type's local-agent collection for a conversation Source, and, for a
    document Source the device acquires with its type's own Gene (ADR 0058),
    every field but the secrets.
    """
    from memforge.genes.family import CollectionFamilyKind
    from memforge.local_agent.document_acquisition import acquisition_job_config

    source_type = str(source.get("type") or "").strip()
    descriptor = source_type_descriptor(source_type)
    if (
        descriptor is not None
        and descriptor.family is CollectionFamilyKind.DOCUMENT
        and local_agent_sync_operation(source_type, source.get("config")) is None
    ):
        job_config = acquisition_job_config(source_type, _source_config(source.get("config")))
    else:
        job_config = local_agent_job_config(source_type, source.get("config"))
    canonical = {
        "source_type": source_type,
        "config": job_config,
    }
    encoded = json.dumps(
        canonical,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
