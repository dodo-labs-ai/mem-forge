"""Canonical admin source response assembly."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, get_args

from memforge.genes import source_type_supports_sync
from memforge.genes.family import CollectionFamilyKind
from memforge.models import AbsenceCheckSkippedReason
from memforge.local_agent.source_contract import (
    execution_owner_user_id,
    is_local_agent_backed_source,
    source_execution_descriptor,
)
from memforge.source_access import (
    SourceAccessPolicy,
    source_access_policy,
    source_is_discoverable,
    source_owner_user_id,
)
from memforge.release import current_release
from memforge.source_artifacts import source_artifact_inference_eligibility
from memforge.source_item_failures import SourceItemFailure
from memforge.source_projection_config import canonical_projection_scope, canonical_scope_transition
from memforge.source_types import source_type_descriptor
from memforge.sync_progress import source_progress_unit, source_sync_progress_from_pipeline
from memforge.storage.admin_source import SourceAdminReader

WORKSPACE_ADMIN_ROLE = "workspace_admin"
MEMBER_ROLE = "member"
VIEWER_ROLE = "viewer"
SELF_HOSTED_OWNER_ROLE = "owner"
LOCAL_WORKSPACE_ROLE = SELF_HOSTED_OWNER_ROLE
MANAGED_SOURCE_TYPES = frozenset({"agent_session"})
# The failed Documents a Source row lists; its count covers every one.
MAX_LISTED_FAILED_ITEMS = 100


async def image_documents_by_source(db: Any, *, viewer_id: str, viewer_role: str) -> dict[str, tuple[str, ...]]:
    """The current Documents with an image a model can read, of each Source the viewer can reprocess.

    A Source counts when the viewer manages it and it is active and syncs, as
    a reprocess requires. A Document counts when one of its current Artifacts
    is eligible for inference, which only an image of a supported format and
    size is.
    """
    documents: dict[str, tuple[str, ...]] = {}
    for source in await db.list_sources():
        if source.get("status") != "active" or not source_type_supports_sync(str(source.get("type") or "")):
            continue
        if not can_manage_source(source, viewer_id=viewer_id, viewer_role=viewer_role):
            continue
        rows = await db.list_current_document_artifacts(str(source["id"]))
        document_ids = tuple(
            dict.fromkeys(
                document_id
                for document_id, metadata in rows
                if source_artifact_inference_eligibility(metadata) is True
            )
        )
        if document_ids:
            documents[str(source["id"])] = document_ids
    return documents


def normalize_workspace_role(role: str | None) -> str:
    value = str(role or "").strip()
    return (
        value
        if value in {SELF_HOSTED_OWNER_ROLE, WORKSPACE_ADMIN_ROLE, MEMBER_ROLE, VIEWER_ROLE}
        else MEMBER_ROLE
    )


def can_manage_workspace(viewer_role: str) -> bool:
    """Whether the role may change workspace-wide structure and memory."""
    return viewer_role in {SELF_HOSTED_OWNER_ROLE, WORKSPACE_ADMIN_ROLE}


def _is_managed_source_type(source_type: str) -> bool:
    return source_type in MANAGED_SOURCE_TYPES


def source_viewer_relationship(
    source: dict[str, Any], *, viewer_id: str, viewer_role: str
) -> str:
    if source_owner_user_id(source) == viewer_id:
        return "owner"
    if viewer_role == SELF_HOSTED_OWNER_ROLE:
        return SELF_HOSTED_OWNER_ROLE
    if viewer_role == WORKSPACE_ADMIN_ROLE:
        return WORKSPACE_ADMIN_ROLE
    return normalize_workspace_role(viewer_role)


def can_manage_source(
    source: dict[str, Any], *, viewer_id: str, viewer_role: str
) -> bool:
    if not source_is_discoverable(source, viewer_id=viewer_id):
        return False
    if source_owner_user_id(source) == viewer_id:
        return True
    if viewer_role == SELF_HOSTED_OWNER_ROLE:
        return True
    return (
        source_access_policy(source) is SourceAccessPolicy.WORKSPACE
        and viewer_role == WORKSPACE_ADMIN_ROLE
    )


def source_ownership_and_capabilities(
    source: dict[str, Any], *, viewer_id: str, viewer_role: str
) -> tuple[dict[str, Any], dict[str, bool]]:
    relationship = source_viewer_relationship(
        source, viewer_id=viewer_id, viewer_role=viewer_role
    )
    can_manage = can_manage_source(source, viewer_id=viewer_id, viewer_role=viewer_role)
    managed_source = _is_managed_source_type(str(source.get("type") or ""))
    sync_supported = source_type_supports_sync(str(source.get("type") or ""))
    local_agent_backed = is_local_agent_backed_source(source)
    execution_owner = execution_owner_user_id(source)
    can_execute_locally = execution_owner is not None and execution_owner == viewer_id
    ownership = {
        "created_by_user_id": source.get("created_by_user_id"),
        "owner_user_id": source_owner_user_id(source),
        "execution_owner_user_id": execution_owner,
        "viewer_role": viewer_role,
        "viewer_relationship": relationship,
    }
    capabilities = {
        "can_subscribe": True,
        "can_configure": can_manage and not managed_source,
        "can_configure_connection": (
            can_execute_locally if local_agent_backed else can_manage and not managed_source
        ),
        "can_sync": (
            sync_supported
            and (
                can_execute_locally
                if local_agent_backed
                else can_manage and not managed_source
            )
        ),
        "can_force_resync": (
            sync_supported
            and (
                can_execute_locally
                if local_agent_backed
                else can_manage and not managed_source
            )
        ),
        "can_delete": can_manage and not managed_source,
        "can_change_access": can_manage,
    }
    return ownership, capabilities


def _sync_is_running(sync_service: Any, source_id: str) -> bool:
    return bool(sync_service is not None and sync_service.is_running(source_id))


def _durable_sync_payload(run: Any) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    status = str(run.status)
    if status == "running" and (
        run.lease_expires_at is None or run.lease_expires_at <= now
    ):
        status = "recovering"
    return {
        "run_id": run.run_id,
        "status": status,
        "trigger": run.trigger,
        "force_full_sync": run.force_full_sync,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.completed_at.isoformat() if run.completed_at else None,
        "next_attempt_at": (
            run.next_attempt_at.isoformat() if run.next_attempt_at else None
        ),
        "recovery_count": run.recovery_count,
        "error_message": run.error_message,
        "progress": run.progress,
        "progress_revision": run.progress_revision,
        "progress_updated_at": (
            run.progress_updated_at.isoformat() if run.progress_updated_at else None
        ),
        "stop_requested_at": (
            run.stop_requested_at.isoformat() if run.stop_requested_at else None
        ),
    }


async def list_source_admin_rows(
    reader: SourceAdminReader,
    *,
    sync_service: Any,
    viewer_id: str,
    viewer_role: str,
    workspace_id: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    viewer_role = normalize_workspace_role(viewer_role)
    for source in await reader.list_sources():
        if not source_is_discoverable(source, viewer_id=viewer_id):
            continue
        row = dict(source)
        source_id = str(row.get("id") or "")
        ownership, capabilities = source_ownership_and_capabilities(
            row, viewer_id=viewer_id, viewer_role=viewer_role
        )
        row["ownership"] = ownership
        row["capabilities"] = capabilities
        row["execution"] = source_execution_descriptor(
            str(row.get("type") or ""),
            row.get("config"),
        )
        enabled_for_me = await reader.is_source_enabled_for_user(source_id, viewer_id)
        row["subscription"] = {"enabled": enabled_for_me}
        row["enabled_for_me"] = enabled_for_me
        row["pinned_for_me"] = await reader.is_source_pinned_for_user(
            source_id, viewer_id
        )
        row["memory_count"] = await reader.count_source_memories(
            source_id,
            include_private=True,
            owner_user_id=viewer_id,
        )
        row["doc_count"] = await reader.count_documents(source=source_id)
        row["access_transition"] = await reader.get_active_source_access_transition(
            source_id
        )
        row.setdefault("client", None)
        if not source_type_supports_sync(str(row.get("type") or "")):
            row["sync"] = None
            row["failed_items"] = None
            row["scope_transition"] = None
            rows.append(row)
            continue
        failures = await reader.list_source_item_failures(source_id=source_id)
        row["failed_items"] = source_failed_items_payload(failures, release=current_release())
        row["scope_transition"] = await source_scope_transition_payload(reader, row, failures)
        durable_run = await reader.get_latest_source_sync_run(
            source_id=source_id,
            workspace_id=workspace_id,
        )
        if durable_run is not None and durable_run.status in {"pending", "running"}:
            row["sync"] = _durable_sync_payload(durable_run)
        elif _sync_is_running(sync_service, source_id):
            progress = sync_service.progress.get(source_id, {})
            progress_snapshot = source_sync_progress_from_pipeline(
                {
                    "phase": progress.get("phase"),
                    "current": progress.get("docs_processed", 0),
                    "total": progress.get("docs_total", 0),
                    "docs_updated": progress.get("docs_updated", 0),
                    "docs_failed": progress.get("docs_failed", 0),
                    "memories_extracted": progress.get("memories_extracted", 0),
                },
                source_type=str(row.get("type") or ""),
            )
            row["sync"] = {
                "status": "running",
                "phase": progress.get("phase"),
                "started_at": progress.get("started_at"),
                "finished_at": None,
                "docs_processed": progress.get("docs_processed", 0),
                "docs_total": progress.get("docs_total", 0),
                "docs_updated": progress.get("docs_updated", 0),
                "docs_failed": progress.get("docs_failed", 0),
                "memories_extracted": progress.get("memories_extracted", 0),
                "docs_stored": row["doc_count"],
                "memories_stored": row["memory_count"],
                "current_title": progress.get("title"),
                "error_message": None,
                "progress": progress_snapshot,
            }
        else:
            history = await reader.get_sync_history(source=source_id, limit=1)
            if history:
                latest = history[0]
                row["sync"] = {
                    "run_id": latest.get("run_id"),
                    "status": latest.get("status", "success"),
                    "started_at": latest.get("started_at"),
                    "finished_at": latest.get("finished_at"),
                    "docs_processed": latest.get("docs_processed", 0),
                    "docs_updated": latest.get("docs_updated", 0),
                    "docs_failed": latest.get("docs_failed", 0),
                    "memories_extracted": latest.get("memories_extracted", 0),
                    "error_message": latest.get("error_message"),
                    "failed_docs": latest.get("failed_docs", []),
                    "absence_check_skipped_reason": _absence_check_skipped_reason(
                        latest.get("absence_check_skipped_reason")
                    ),
                    "unlisted_present_count": latest.get("unlisted_present_count"),
                    "skipped_reading_group_count": latest.get("skipped_reading_group_count", 0),
                    "progress": {
                        "schema_version": 1,
                        "phase": "processing",
                        "progress": {
                            "completed": latest.get("docs_processed", 0),
                            "unit": source_progress_unit(str(row.get("type") or "")),
                        },
                        "counts": {
                            "changed": latest.get("docs_updated", 0),
                            "failed": latest.get("docs_failed", 0),
                            "memories_created": latest.get("memories_extracted", 0),
                        },
                    },
                }
            elif durable_run is not None:
                row["sync"] = _durable_sync_payload(durable_run)
            else:
                row["sync"] = None
        rows.append(row)
    return rows


def source_failed_items_payload(failures: tuple[SourceItemFailure, ...], *, release: str) -> dict[str, Any]:
    """The Source's failed Documents, the most recently failed first, and how many this release is retrying."""

    listed = sorted(failures, key=lambda failure: (failure.last_failed_at, failure.document_id), reverse=True)
    return {
        "count": len(failures),
        "retrying_after_upgrade": sum(failure.recorded_by_another_release(release) for failure in failures),
        "items": [
            {
                "document_id": failure.document_id,
                "title": failure.title,
                "stage": failure.stage,
                "error_code": failure.error_code,
                "message": failure.message,
                "retryable": failure.retryable,
                "release": failure.release,
                "failures_in_release": failure.failures_in_release,
                "first_failed_at": failure.first_failed_at.isoformat(),
                "last_failed_at": failure.last_failed_at.isoformat(),
                "retrying_after_upgrade": failure.recorded_by_another_release(release),
            }
            for failure in listed[:MAX_LISTED_FAILED_ITEMS]
        ],
    }


ScopeChangeRemoval = Literal["at_completion", "next_sync"]


def scope_change_removes(source_type: str) -> ScopeChangeRemoval:
    """When a scope change of a Source of this type removes what its new scope leaves out.

    A document transition removes it once it completes (``at_completion``):
    until then nothing is removed. A conversation transition removes the
    conversations its new selection leaves out in its next sync
    (``next_sync``), and completes once every selected conversation it can
    read is synced.
    """

    descriptor = source_type_descriptor(source_type)
    if descriptor is not None and descriptor.family is CollectionFamilyKind.CONVERSATION:
        return "next_sync"
    return "at_completion"


async def source_scope_transition_payload(
    reader: SourceAdminReader,
    source: dict[str, Any],
    failures: tuple[SourceItemFailure, ...],
) -> dict[str, Any] | None:
    """The scope change the Source is applying, as Source health shows it; ``None`` when none is open.

    Both scopes are canonical scope fields, read as the sync reads them
    (:func:`canonical_scope_transition`), and hold no credential.
    ``members_committed`` counts the items committed under the change, and
    ``listed_count`` the items the latest complete listing of the target scope
    holds. ``failing_items`` are the failure ledger's rows of Documents not
    committed under it: one the target scope lists keeps it open until a sync
    commits it, and one only the previous scope held is settled when it
    completes. The listing itself is not stored, so both are shown. For
    example, a Confluence Source changed from space ``ENG`` to ``ENG`` and
    ``OPS`` whose latest sync listed 120 pages, committed 117 and failed 3
    reads ``members_committed`` 117, ``listed_count`` 120 and three failing
    items. ``restore_config`` is filled by the route, which alone holds the
    configuration the viewer may read. ``removes`` says when the change
    removes what the target scope leaves out (:func:`scope_change_removes`).
    """

    source_id = str(source.get("id") or "")
    source_type = str(source.get("type") or "")
    transition = await reader.get_open_projection_scope_transition(source_id)
    if transition is None:
        return None
    transition = canonical_scope_transition(
        source_type,
        transition,
        canonical_projection_scope(source_type, source.get("config") or {}),
    )
    committed = frozenset(await reader.list_projection_scope_transition_failed_member_doc_ids(transition.id))
    failing = tuple(failure for failure in failures if failure.document_id not in committed)
    return {
        "id": transition.id,
        "status": transition.status.value,
        "previous_scope": dict(transition.previous_scope),
        "target_scope": dict(transition.target_scope),
        "created_at": transition.created_at,
        "updated_at": transition.updated_at,
        "members_committed": await reader.count_projection_scope_transition_members(transition.id),
        "listed_count": transition.listed_count,
        "excluded_scopes_due": transition.excluded_scopes_due,
        "error": transition.error,
        "failing_items": source_failed_items_payload(failing, release=current_release()),
        "restore_config": None,
        "removes": scope_change_removes(source_type),
    }


def _absence_check_skipped_reason(stored: Any) -> AbsenceCheckSkippedReason | None:
    """The step a sync run's absence check stopped at, as recorded with the run.

    Any other stored text is a provider's error from a scope listing that did
    not complete.
    """

    if not stored:
        return None
    if stored in get_args(AbsenceCheckSkippedReason):
        return stored
    return "scope_listing_failed"
