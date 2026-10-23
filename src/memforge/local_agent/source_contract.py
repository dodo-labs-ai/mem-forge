"""Shared execution contract for local-agent-backed sources."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from datetime import timedelta
from typing import Any

from memforge.conversations.time_scope import MAX_INITIAL_HISTORY_DAYS
from memforge.local_agent.snapshot_manifest import UPSERT
from memforge.source_types import registered_source_types, source_type_descriptor


LOCAL_AGENT_SYNC_OPERATIONS = frozenset(
    descriptor.local_agent.operation
    for descriptor in registered_source_types()
    if descriptor.local_agent is not None
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
LOCAL_AGENT_SEMANTIC_INPUT_VERSION = "canonical-v1"
LOCAL_PACKAGE_CONTRACT_VERSION = 3
"""The package contract this local agent collects under, declared in its manifest.

An agent that declares no version predates the numbered contract.
"""
LOCAL_PACKAGE_SOURCE_TIME_CONTRACT_VERSION = 2
"""The first contract under which a file package carries the file's source time.

A collection under this contract or a later one does not reuse a retained file
package without a source time, so the agent uploads that file once more with
its time (design ``source-sync-to-memory.md`` 0.9).
"""
LOCAL_PACKAGE_CONFIRMATION_CONTRACT_VERSION = 3
"""The first contract under which an agent confirms by identifier what its query manifest leaves out.

The manifest response then names the items to confirm, and the agent posts
its answers before it asks for processing.
"""
TEAMS_ROLLING_RETENTION_PRESETS = (365, 730, 1095)


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


class SourceSyncRunReceiptError(ValueError):
    """Raised when a successful local sync cannot identify its durable server run."""


class GitHubProviderConnectionError(ConnectionError):
    """Raised when GitHub CLI cannot reach the configured GitHub host."""


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


def local_agent_input_sha256(doc_id: object, document_hash: object) -> str:
    """Hash stable document identity and semantic content version."""
    normalized_doc_id = str(doc_id or "").strip()
    normalized_hash = str(document_hash or "").strip()
    if not normalized_doc_id or not normalized_hash:
        return ""
    identity = json.dumps(
        {"doc_id": normalized_doc_id, "document_hash": normalized_hash},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def local_agent_semantic_input_sha256(
    doc_id: object,
    semantic_hash: object,
) -> str:
    """Version the canonical semantic identity independently of legacy raw hashes."""
    return local_agent_input_sha256(
        doc_id,
        f"{LOCAL_AGENT_SEMANTIC_INPUT_VERSION}:{semantic_hash}",
    )


def local_file_package_semantic_hash(content_hash: str, source_updated_at: str | None) -> str:
    """The semantic identity of a file package: its content and that content's source time.

    The same content uploaded with and without its source time makes two
    inputs, so a package that adds the time is retained and projected instead
    of collapsing into the earlier one. A package without a time is identified
    by its content hash alone.
    """

    if source_updated_at is None:
        return content_hash
    identity = json.dumps(
        {"content_hash": content_hash, "source_updated_at": source_updated_at},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def local_package_requires_source_time(
    source_type: object,
    package_contract_version: int | None,
) -> bool:
    """Whether a collection may reuse only retained packages that carry a source time.

    True for a file source collected under the source-time package contract.
    """

    from memforge.local_agent.replay_adapter import get_local_source_replay_adapter

    if package_contract_version is None or package_contract_version < LOCAL_PACKAGE_SOURCE_TIME_CONTRACT_VERSION:
        return False
    try:
        adapter = get_local_source_replay_adapter(str(source_type or "").strip().lower())
    except ValueError:
        return False
    return adapter.package_source_time


def local_package_confirms_absence(package_contract_version: int | None) -> bool:
    """Whether an agent collecting under this package contract confirms unlisted items by identifier."""

    return package_contract_version is not None and (
        package_contract_version >= LOCAL_PACKAGE_CONFIRMATION_CONTRACT_VERSION
    )


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
    """Return the daemon sync operation for a canonical source configuration."""
    descriptor = source_type_descriptor(str(source_type or "").strip().lower())
    collection = descriptor.local_agent if descriptor is not None else None
    if collection is None or not collection.collects(_source_config(config)):
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


def source_execution_descriptor(
    source_type: str,
    config: Mapping[str, Any] | str | None,
) -> dict[str, Any]:
    """Return the canonical execution contract exposed to source clients."""
    operation = local_agent_sync_operation(source_type, config)
    descriptor = source_type_descriptor(str(source_type or "").strip().lower())
    collection = descriptor.local_agent if descriptor is not None else None
    return {
        "kind": "local_agent" if operation is not None else "server",
        "operation": operation,
        "immutable_config_fields": list(collection.immutable_execution_fields if collection else ()),
    }


def execution_owner_user_id(source: Mapping[str, Any]) -> str | None:
    value = str(source.get("execution_owner_user_id") or "").strip()
    return value or None


def local_agent_job_config(
    source_type: str,
    config: Mapping[str, Any] | str | None,
) -> dict[str, Any]:
    """Return connector config safe and useful for the owner's daemon."""
    descriptor = source_type_descriptor(str(source_type or "").strip().lower())
    collection = descriptor.local_agent if descriptor is not None else None
    if collection is None:
        return {}
    source_config = dict(_source_config(config))
    if collection.job_config_defaults is not None:
        source_config = collection.job_config_defaults(source_config)
    return {key: value for key, value in source_config.items() if key in collection.job_config_fields}


LOCAL_AGENT_TRANSITION_PAYLOAD_FIELD = "projection_scope_transition"
"""The job payload field naming the Source's open scope transition and its canonical scopes."""
LOCAL_AGENT_EXCLUDED_SCOPES_PAYLOAD_FIELD = "excluded_scopes"
"""The job payload field carrying the configuration of each scope an open transition excludes.

Each entry is ``{"scope_key": ..., "config": ...}``: the scope's fingerprint,
which the device's manifest names its listing by, and the job configuration
that lists that scope. A device collector removes both transition fields
before it configures its Gene.
"""
LOCAL_AGENT_TRANSITION_PAYLOAD_FIELDS = frozenset(
    {LOCAL_AGENT_TRANSITION_PAYLOAD_FIELD, LOCAL_AGENT_EXCLUDED_SCOPES_PAYLOAD_FIELD}
)
"""The job payload fields about the open scope transition, which never reach a device Gene's configuration."""


async def local_agent_sync_job_payload(
    db: Any,
    source: Mapping[str, Any],
    request_payload: Mapping[str, Any] | str | None = None,
) -> dict[str, Any]:
    """The daemon sync-job payload for a saved Source, as of the moment the job is enqueued.

    Every place that enqueues a local-agent sync job builds its payload here:
    the saved collection configuration, the request's control fields, and,
    while a document scope transition is open, that transition. Once the
    transition's previous run left no listed Document that is not yet a member,
    the next run can complete it, so the payload also carries the configuration
    of every distinct scope the transition excludes, for the device to list
    (:data:`LOCAL_AGENT_EXCLUDED_SCOPES_PAYLOAD_FIELD`). Only a query listing
    needs them; an existence listing completes from the new listing alone.
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
    excluded_scopes = await _excluded_scope_job_configs(db, source_type, config, transition)
    if excluded_scopes:
        payload[LOCAL_AGENT_EXCLUDED_SCOPES_PAYLOAD_FIELD] = excluded_scopes
    return payload


async def _excluded_scope_job_configs(
    db: Any,
    source_type: str,
    config: Mapping[str, Any],
    transition: Any,
) -> list[dict[str, Any]]:
    """The job configuration of each scope the transition excludes, once its next run can complete it.

    The next run can complete it when the transition's latest run reached the
    step that lists the excluded scopes (``excluded_scopes_due``), or when
    that run, bound to a device snapshot, listed only Documents that are
    already members. Before any such run, and while a listed Document waits
    for anything but the removals completion makes, none is carried.
    """

    from memforge.genes.family import CollectionFamilyKind
    from memforge.models import ScopeListingKind
    from memforge.source_projection_config import (
        ExcludedScopesUnavailable,
        excluded_projection_scopes,
        excluded_scope_config,
        projection_scope_fingerprint,
    )

    descriptor = source_type_descriptor(source_type)
    if descriptor is None or descriptor.local_agent is None or descriptor.family is not CollectionFamilyKind.DOCUMENT:
        return []
    if descriptor.gene_class().snapshot_listing_kind() is not ScopeListingKind.QUERY:
        return []
    if not transition.excluded_scopes_due and not await _previous_run_left_none_waiting(db, transition):
        return []
    try:
        scopes = await excluded_projection_scopes(db, transition, source_type=source_type, config=config)
    except ExcludedScopesUnavailable:
        return []
    return [
        {
            "scope_key": projection_scope_fingerprint(scope),
            "config": local_agent_job_config(source_type, excluded_scope_config(source_type, config, scope)),
        }
        for scope in scopes
    ]


async def _previous_run_left_none_waiting(db: Any, transition: Any) -> bool:
    """Whether the transition's latest run, bound to a device snapshot, listed only Documents that are members."""

    if transition.run_id is None:
        return False
    run = await db.get_source_sync_run(transition.run_id)
    if run is None or run.input_snapshot_id is None:
        return False
    listed = snapshot_document_ids(
        await db.list_source_sync_inputs(
            source_id=transition.source_id,
            workspace_id=run.workspace_id,
            input_snapshot_id=run.input_snapshot_id,
        )
    )
    return not listed - set(await db.list_projection_scope_transition_member_doc_ids(transition.id))


def local_agent_source_config_revision(source: Mapping[str, Any]) -> str:
    """Fingerprint the saved collection scope used by one daemon job."""
    source_type = str(source.get("type") or "").strip()
    canonical = {
        "source_type": source_type,
        "config": local_agent_job_config(source_type, source.get("config")),
    }
    encoded = json.dumps(
        canonical,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def source_sync_input_metadata_with_artifact_attestation(
    metadata: Mapping[str, Any],
    *,
    package_sha256: str,
    input_id: str,
) -> dict[str, Any]:
    """Fill or validate the immutable package attestation for a retained input."""
    artifact_hash = str(package_sha256 or "").strip()
    if not artifact_hash:
        raise ValueError("source sync input artifact attestation is required")
    result = dict(metadata)
    manifest_entry = result.get("manifest_entry")
    if not isinstance(manifest_entry, Mapping):
        raise ValueError(f"source sync input manifest is missing: {input_id}")
    manifest = dict(manifest_entry)
    declared_hashes = {
        value
        for value in (
            str(result.get("package_sha256") or "").strip(),
            str(manifest.get("package_sha256") or "").strip(),
        )
        if value
    }
    if declared_hashes and declared_hashes != {artifact_hash}:
        raise ValueError(f"source sync input artifact attestation conflict: {input_id}")
    result["package_sha256"] = artifact_hash
    manifest["package_sha256"] = artifact_hash
    result["manifest_entry"] = manifest
    return result


def _manifest_entry(source_input: Any) -> tuple[str, Mapping[str, Any]] | None:
    """The Document id and manifest entry one raw input carries, if it names both."""

    metadata = getattr(source_input, "metadata", {})
    entry = metadata.get("manifest_entry") if isinstance(metadata, Mapping) else None
    if not isinstance(entry, Mapping):
        return None
    doc_id = str(entry.get("doc_id") or "").strip()
    raw_uri = str(getattr(source_input, "raw_uri", "") or "").strip()
    if not doc_id or not raw_uri:
        return None
    return doc_id, entry


def snapshot_document_ids(inputs: list[Any]) -> frozenset[str]:
    """The Documents a complete input snapshot holds: every item it names.

    An item stored as a tombstone, which only a manifest written before every
    item was an upsert can hold, names a Document the snapshot does not hold.
    """

    return frozenset(
        doc_id
        for source_input in inputs
        if (named := _manifest_entry(source_input)) is not None
        for doc_id, entry in (named,)
        if str(entry.get("change_kind") or UPSERT) == UPSERT
    )


def source_with_sync_inputs(
    source: Mapping[str, Any],
    inputs: list[Any],
    *,
    input_snapshot_supplied: bool = False,
) -> dict[str, Any]:
    """Project immutable raw inputs into the connector's runtime manifest.

    A run bound to an input snapshot gets a manifest even when the snapshot
    is empty, because an empty complete snapshot says the scope holds nothing.
    """
    latest_entries: dict[str, dict[str, Any]] = {}
    for source_input in sorted(
        inputs,
        key=lambda item: int(getattr(item, "input_generation", 0)),
    ):
        named = _manifest_entry(source_input)
        if named is None:
            continue
        doc_id, entry = named
        metadata = getattr(source_input, "metadata", {})
        raw_uri = str(getattr(source_input, "raw_uri", "") or "").strip()
        package_sha256 = str(metadata.get("package_sha256") or "").strip()
        projected_entry = {
            **entry,
            "package_uri": raw_uri,
            "input_sha256": str(getattr(source_input, "raw_sha256", "") or "").strip(),
            **({"package_sha256": package_sha256} if package_sha256 else {}),
        }
        latest_entries[doc_id] = projected_entry
    projected = dict(source)
    if latest_entries or input_snapshot_supplied:
        config = dict(_source_config(source.get("config")))
        config["local_agent_package_manifest"] = list(latest_entries.values())
        projected["config"] = config
    return projected


def validate_local_agent_replay_package(
    source_type: str,
    body: bytes,
    *,
    expected_doc_id: str,
    expected_version: str,
    expected_input_sha256: str,
    expected_package_sha256: str,
) -> Mapping[str, Any]:
    """Validate one local-agent package with its source type's replay adapter."""
    from memforge.local_agent.replay_adapter import get_local_source_replay_adapter

    return get_local_source_replay_adapter(source_type).validate(
        body,
        expected_doc_id=expected_doc_id,
        expected_version=expected_version,
        expected_input_sha256=expected_input_sha256,
        expected_package_sha256=expected_package_sha256,
    )
