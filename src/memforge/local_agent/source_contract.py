"""Shared execution contract for local-agent-backed sources."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any


LOCAL_AGENT_SYNC_OPERATIONS = frozenset(
    {
        "github_repo_sync",
        "jira_sync",
        "local_markdown_sync",
        "teams_sync",
    }
)
_IMMUTABLE_EXECUTION_MODE_FIELDS = {
    "github_repo": ("connection_mode",),
    "jira": ("sync_mode",),
}

_LOCAL_AGENT_JOB_CONFIG_FIELDS_BY_SOURCE_TYPE = {
    "github_repo": frozenset(
        {
            "repo_url",
            "ref",
            "include_paths",
            "exclude_paths",
            "include_extensions",
        }
    ),
    "jira": frozenset(
        {
            "base_url",
            "auth_mode",
            "sync_mode",
            "projects",
            "issue_types",
            "include_comments",
            "query_mode",
            "jql",
            "jql_filter",
        }
    ),
    "local_markdown": frozenset(
        {
            "root",
            "vault_id",
            "include",
            "exclude",
        }
    ),
    "teams": frozenset(
        {
            "region",
            "conversation_ids",
            "channels",
            "group_chats",
            "individual_chats",
            "conversation_gap_minutes",
            "max_age_days",
            "max_block_messages",
            "incremental_overlap_hours",
            "page_size",
        }
    ),
}
LOCAL_AGENT_SYNC_PAYLOAD_CONTROL_FIELDS = frozenset(
    {
        "force_full_sync",
    }
)
LOCAL_AGENT_JOB_MAX_ATTEMPTS = 5
LOCAL_AGENT_SEMANTIC_INPUT_VERSION = "canonical-v1"
TEAMS_TOMBSTONE_REASONS = frozenset(
    {
        "not_returned_by_complete_conversation_poll",
        "not_returned_by_bounded_conversation_poll",
        "conversation_removed_from_projection_scope",
        "outside_configured_time_scope",
    }
)
TEAMS_CONVERSATION_SELECTOR_FIELDS = (
    "conversation_ids",
    "channels",
    "group_chats",
    "individual_chats",
)
_TEAMS_CONVERSATION_ID_RE = re.compile(r"^19:[^\s@]+@[^\s@]+$")


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
    from memforge.local_agent.replay_adapter import get_local_source_replay_adapter

    get_local_source_replay_adapter(normalized_source_type)
    canonical = local_agent_sync_snapshot_id(job_id, attempt_count)
    if requested and requested != canonical:
        raise ValueError("local agent snapshot does not match the leased job attempt")
    return canonical


def local_agent_collection_is_authoritative(source_type: object) -> bool:
    """Return whether one attempt proves the complete source collection."""

    from memforge.local_agent.replay_adapter import get_local_source_replay_adapter

    try:
        return get_local_source_replay_adapter(str(source_type or "").strip().lower()).authoritative_collection
    except ValueError:
        return False


def local_agent_rebaseline_snapshot_is_authoritative(
    source_type: object,
    *,
    force_full_sync: bool,
    input_snapshot_id: str | None,
) -> bool:
    """Return whether an immutable attempt defines the rebaseline corpus."""

    from memforge.local_agent.replay_adapter import get_local_source_replay_adapter

    try:
        adapter = get_local_source_replay_adapter(str(source_type or "").strip().lower())
    except ValueError:
        return False
    return adapter.rebaseline_snapshot_is_authoritative(
        force_full_sync=force_full_sync,
        input_snapshot_id=input_snapshot_id,
    )


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
    normalized_type = str(source_type or "").strip().lower()
    normalized_config = _source_config(config)
    if normalized_type == "teams":
        return "teams_sync"
    if normalized_type == "jira" and str(normalized_config.get("sync_mode") or "").strip().lower() == "local_agent":
        return "jira_sync"
    if normalized_type == "local_markdown":
        return "local_markdown_sync"
    if (
        normalized_type == "github_repo"
        and str(normalized_config.get("connection_mode") or "").strip().lower() == "local_push"
    ):
        return "github_repo_sync"
    return None


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
    normalized_type = str(source_type or "").strip().lower()
    operation = local_agent_sync_operation(normalized_type, config)
    return {
        "kind": "local_agent" if operation is not None else "server",
        "operation": operation,
        "immutable_config_fields": list(_IMMUTABLE_EXECUTION_MODE_FIELDS.get(normalized_type, ())),
    }


def execution_owner_user_id(source: Mapping[str, Any]) -> str | None:
    value = str(source.get("execution_owner_user_id") or "").strip()
    return value or None


def local_agent_job_config(
    source_type: str,
    config: Mapping[str, Any] | str | None,
) -> dict[str, Any]:
    """Return connector config safe and useful for the owner's daemon."""
    allowed = _LOCAL_AGENT_JOB_CONFIG_FIELDS_BY_SOURCE_TYPE.get(
        str(source_type or "").strip().lower(),
        frozenset(),
    )
    return {key: value for key, value in _source_config(config).items() if key in allowed}


def local_agent_sync_job_payload(
    source: Mapping[str, Any],
    request_payload: Mapping[str, Any] | str | None = None,
) -> dict[str, Any]:
    """Return the canonical daemon sync-job payload for a saved source."""
    source_type = str(source.get("type") or "").strip()
    payload = local_agent_job_config(source_type, source.get("config"))
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
    payload["source_activity_epoch"] = int(source.get("activity_epoch") or 0)
    return payload


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


def source_with_sync_inputs(
    source: Mapping[str, Any],
    inputs: list[Any],
    *,
    authoritative_snapshot: bool = False,
    preserve_version_history: bool = False,
) -> dict[str, Any]:
    """Project immutable raw inputs into the connector's runtime manifest."""
    latest_entries: dict[str, dict[str, Any]] = {}
    historical_entries: list[dict[str, Any]] = []
    for source_input in sorted(
        inputs,
        key=lambda item: int(getattr(item, "input_generation", 0)),
    ):
        metadata = getattr(source_input, "metadata", {})
        entry = metadata.get("manifest_entry") if isinstance(metadata, Mapping) else None
        if not isinstance(entry, Mapping):
            continue
        doc_id = str(entry.get("doc_id") or "").strip()
        raw_uri = str(getattr(source_input, "raw_uri", "") or "").strip()
        if not doc_id or not raw_uri:
            continue
        package_sha256 = str(metadata.get("package_sha256") or "").strip()
        projected_entry = {
            **entry,
            "package_uri": raw_uri,
            "input_sha256": str(getattr(source_input, "raw_sha256", "") or "").strip(),
            **({"package_sha256": package_sha256} if package_sha256 else {}),
        }
        latest_entries[doc_id] = projected_entry
        historical_entries.append(projected_entry)
    projected = dict(source)
    if latest_entries or authoritative_snapshot:
        config = dict(_source_config(source.get("config")))
        config["local_agent_package_manifest"] = (
            historical_entries if preserve_version_history else list(latest_entries.values())
        )
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
    """Compatibility wrapper around the registered provider adapter."""
    from memforge.local_agent.replay_adapter import get_local_source_replay_adapter

    return get_local_source_replay_adapter(source_type).validate(
        body,
        expected_doc_id=expected_doc_id,
        expected_version=expected_version,
        expected_input_sha256=expected_input_sha256,
        expected_package_sha256=expected_package_sha256,
    )
