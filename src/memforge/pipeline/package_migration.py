"""Move Sources collected through a device package path onto device acquisition (ADR 0058, design §10.9).

Before device acquisition, a Jira Source with ``sync_mode: local_agent`` and
a GitHub repository Source with ``connection_mode: local_push`` were collected
through a package path of their own: the owner's daemon submitted one package
per item, the server stored the package as the Unit's stored input, and Jira
named the Document ``jira-<source>-<key>-<digest>``. Device acquisition runs
the type's own Gene on the device instead, so these Units hold inputs that
only the package path reads, under ids the Gene does not give.

This recoverable migration (ADR 0032) moves each such Source once:

1. **Inventory.** :func:`package_path_sources` lists the Sources whose
   configuration names a package path.
2. **Dry run.** :func:`plan_package_migration` builds, for each current Unit
   whose stored input is a package, the input the type's Gene would have
   produced: the item, with the Gene's id, ``version`` and ``last_modified``,
   and the raw body unwrapped from its package. It projects that input
   against the Unit's current revision and counts the Units whose
   Observations it would change. A package the Gene's form cannot be rebuilt
   from (a Jira issue without ``fields.updated``), two Units that map to one
   id, and a GitHub file whose id from the configured ``repo_url`` and
   ``ref`` differs from the stored one each refuse the Source. It writes
   nothing.
3. **Apply.** :func:`apply_package_migration` holds the Source's sync
   activity, plans again, and replays the rebuilt inputs through a REPLAY
   run of the type's recorded Gene: each is processed under the Gene's id
   without forced extraction, so the projection delta decides extraction as
   in an ordinary run, and the run records it as the Unit's stored input.
   Where the id changes, the rebuilt item names the stored Document as the
   one it was renamed from, so it continues that Document's Unit, which
   moves to its new Document with its Support
   (``_finalize_projected_document_moves``). Only once every rebuilt input
   committed does the Source move to the device executor without its package
   fields; the cursor, the failure ledger and everything else it holds stay.

Every step is idempotent. A Unit already in the Gene's form is not planned
again, and a Source whose replay did not commit every input keeps its
package path, so running the migration again continues where it stopped.
``memforge maintenance migrate-device-packages --apply`` takes a recovery
point of the local data before it applies anything, so the whole run can be
undone with ``memforge backup restore``.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from memforge.genes import gene_class
from memforge.genes.recorded import AcquiredInput, AcquisitionRecordMismatch
from memforge.github_repo_utils import GITHUB_SYMLINK_MODE, build_github_repo_doc_id, parse_github_repo_url
from memforge.local_agent.document_acquisition import acquisition_job_config
from memforge.models import AbsenceAnswer, ContentItem, ScopeListing, SourceUnitInput
from memforge.pipeline.source_projection_adapters import DEFAULT_SOURCE_PROJECTION_ADAPTER
from memforge.pipeline.stored_document import StoredDocumentUnavailable, committed_source_artifacts
from memforge.pipeline.sync import SourceSyncMode
from memforge.source_artifacts import RawSourceArtifact, SourceArtifactDownload, StoredSourceArtifact
from memforge.source_projection import ProjectionEnvelope, ProjectionRequest, ProjectionRunMode
from memforge.source_projection_config import canonical_projection_scope, source_access_context
from memforge.source_types.descriptor import SOURCE_EXECUTOR_DEVICE

if TYPE_CHECKING:
    from memforge.config import AppConfig
    from memforge.runtime import SyncRuntime
    from memforge.storage.database import Database
    from memforge.storage.document_store import DocumentStore

__all__ = [
    "MigratedUnit",
    "PackageMigrationResult",
    "SourceMigrationPlan",
    "apply_package_migration",
    "package_path_sources",
    "plan_package_migration",
]


class Unrebuildable(ValueError):
    """The Gene's form of one stored package cannot be rebuilt from what the package holds."""


@dataclass(frozen=True, slots=True)
class _Rebuilt:
    item: ContentItem
    body: bytes
    content_type: str


@dataclass(frozen=True, slots=True)
class _PackagePath:
    """One type's package path: how to recognize it, and how its stored packages become the Gene's input."""

    package_kind: str
    on_path: Callable[[Mapping[str, Any]], bool]
    # The configuration fields only the package path reads.
    package_fields: tuple[str, ...]
    rebuild: Callable[[str, Mapping[str, Any], dict[str, Any], SourceUnitInput], _Rebuilt]
    # Whether the package path already named each Document as the Gene does,
    # so a rebuilt id that differs refuses the Source instead of moving a Unit.
    keeps_document_ids: bool


def _rebuild_jira(source_id: str, config: Mapping[str, Any], package: dict[str, Any], stored: SourceUnitInput) -> _Rebuilt:
    """The issue as ``JiraGene`` discovers and fetches it: ``jira-<KEY>``, its ``fields.updated`` as version and time."""

    from memforge.genes.jira_gene import issue_content_item

    payload = package.get("raw_payload")
    if not isinstance(payload, dict):
        raise Unrebuildable("the package holds no issue payload")
    fields = payload.get("fields")
    if not isinstance(fields, dict) or not str(fields.get("updated") or "").strip():
        raise Unrebuildable("the package's issue has no fields.updated")
    if not str(payload.get("id") or "").strip().isdigit() or not str(payload.get("key") or "").strip():
        raise Unrebuildable("the package's issue has no stable id or key")
    base_url = str(config.get("base_url") or "").rstrip("/")
    try:
        item = issue_content_item(payload, base_url)
    except RuntimeError as exc:
        raise Unrebuildable(str(exc)) from exc
    # The Gene names the issue's Document jira-<KEY>: the Unit moves to it.
    item.renamed_from = stored.document_id
    return _Rebuilt(item=item, body=json.dumps(payload).encode("utf-8"), content_type="application/json")


def _rebuild_github_repo(
    source_id: str, config: Mapping[str, Any], package: dict[str, Any], stored: SourceUnitInput
) -> _Rebuilt:
    """The file as ``GitHubRepoGene`` discovers and fetches it, at the configured repository and ref."""

    from memforge.genes.github_repo_gene import LAST_COMMIT_AT_KEY, title_from_path

    relative_path = str(package.get("relative_path") or "").strip()
    markdown = package.get("markdown")
    blob_sha = str(package.get("blob_sha") or "").strip()
    if not relative_path or not isinstance(markdown, str) or not blob_sha:
        raise Unrebuildable("the package names no file path, text or blob")
    try:
        repo = parse_github_repo_url(str(config.get("repo_url") or ""))
    except ValueError as exc:
        raise Unrebuildable(f"the configured repo_url is invalid: {exc}") from exc
    ref = str(config.get("ref") or "main").strip() or "main"
    doc_id = build_github_repo_doc_id(
        source_id=source_id, repo_url=repo["repo_url"], repo_ref=ref, relative_path=relative_path
    )
    content_type = str(package.get("content_type") or "text/markdown")
    symlink_chain = package.get("symlink_chain")
    extra: dict[str, Any] = {
        "file_identity_contract": "repository_path",
        "repo_url": repo["repo_url"],
        "repo_host": repo["host"],
        "repo_owner": repo["owner"],
        "repo_name": repo["repo"],
        "repo_ref": ref,
        "relative_path": relative_path,
        "blob_sha": blob_sha,
        LAST_COMMIT_AT_KEY: package.get("source_updated_at"),
    }
    if symlink_chain:
        extra.update(
            {
                "logical_file_mode": GITHUB_SYMLINK_MODE,
                "symlink_chain": [dict(link) for link in symlink_chain],
                "resolved_relative_path": package.get("resolved_relative_path"),
            }
        )
    item = ContentItem(
        item_id=doc_id,
        title=title_from_path(relative_path),
        source_url=str(package.get("source_url") or stored.item.source_url),
        last_modified=stored.item.last_modified,
        content_type=content_type,
        version=blob_sha,
        space_or_project=f"{repo['owner']}/{repo['repo']}",
        labels=["github_repo"],
        extra=extra,
    )
    return _Rebuilt(item=item, body=markdown.encode("utf-8"), content_type=content_type)


_PACKAGE_PATHS: Mapping[str, _PackagePath] = {
    "jira": _PackagePath(
        package_kind="jira_document",
        on_path=lambda config: str(config.get("sync_mode") or "").strip().lower() == "local_agent",
        package_fields=("sync_mode", "local_agent_documents_dir"),
        rebuild=_rebuild_jira,
        keeps_document_ids=False,
    ),
    "github_repo": _PackagePath(
        package_kind="github_repo_document",
        on_path=lambda config: str(config.get("connection_mode") or "").strip().lower() == "local_push",
        package_fields=("connection_mode", "documents_dir"),
        rebuild=_rebuild_github_repo,
        keeps_document_ids=True,
    ),
}


def _package_path(source: Mapping[str, Any]) -> _PackagePath | None:
    path = _PACKAGE_PATHS.get(str(source.get("type") or ""))
    if path is None or not path.on_path(source.get("config") or {}):
        return None
    return path


def _decoded_package(body: bytes, package_kind: str) -> dict[str, Any] | None:
    try:
        package = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return package if isinstance(package, dict) and package.get("package_kind") == package_kind else None


@dataclass(frozen=True, slots=True)
class MigratedUnit:
    """One Unit whose stored package becomes the Gene's input.

    ``stored_document_id`` is the Document the Unit is on now and
    ``document_id`` the one the Gene names. ``changes_content`` says whether
    the rewrite changes the Unit's Observations, so its replay extracts.
    """

    source_unit_id: str
    stored_document_id: str
    document_id: str
    item: ContentItem
    body: bytes
    content_type: str
    artifacts: tuple[StoredSourceArtifact, ...]
    changes_content: bool


@dataclass(frozen=True, slots=True)
class SourceMigrationPlan:
    """What applying the migration would do to one Source; a refused Source is left as it is."""

    source_id: str
    source_type: str
    source_name: str
    units: tuple[MigratedUnit, ...] = ()
    # Units whose stored input is already the Gene's.
    current_units: int = 0
    # (stored Document id, reason) of each package the Gene's form cannot be rebuilt from.
    unrebuildable: tuple[tuple[str, str], ...] = ()
    # (Gene id, the stored Document ids that map to it) of each collision.
    collisions: tuple[tuple[str, tuple[str, ...]], ...] = ()
    # (stored Document id, Gene id) of each file whose id the Gene gives differently.
    id_changes_refused: tuple[tuple[str, str], ...] = ()

    @property
    def refused(self) -> bool:
        return bool(self.unrebuildable or self.collisions or self.id_changes_refused)

    @property
    def changed_document_ids(self) -> tuple[str, ...]:
        return tuple(unit.stored_document_id for unit in self.units if unit.changes_content)

    @property
    def moved_document_ids(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (unit.stored_document_id, unit.document_id)
            for unit in self.units
            if unit.stored_document_id != unit.document_id
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_type": self.source_type,
            "source_name": self.source_name,
            "units": len(self.units),
            "current_units": self.current_units,
            "changed_document_ids": list(self.changed_document_ids),
            "moved_document_ids": [list(pair) for pair in self.moved_document_ids],
            "unrebuildable": [list(pair) for pair in self.unrebuildable],
            "collisions": [[doc_id, list(stored)] for doc_id, stored in self.collisions],
            "id_changes_refused": [list(pair) for pair in self.id_changes_refused],
            "refused": self.refused,
        }


@dataclass(frozen=True, slots=True)
class PackageMigrationResult:
    """What applying the migration did to one Source.

    ``status`` is ``migrated`` once the Source runs on the device, ``refused``
    when its plan is refused, and ``replay_failed`` when not every rebuilt
    input committed; ``sync_status`` and ``error`` then come from the replay.
    """

    plan: SourceMigrationPlan
    status: str
    sync_status: str | None = None
    error: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {"status": self.status, "sync_status": self.sync_status, "error": self.error, **self.plan.to_payload()}


async def package_path_sources(db: Database) -> list[dict[str, Any]]:
    """The inventory: every Source, in every workspace, whose configuration names a device package path."""

    return [source for source in await db.list_sources() if _package_path(source) is not None]


async def plan_package_migration(db: Database, doc_store: DocumentStore, source: Mapping[str, Any]) -> SourceMigrationPlan:
    """The dry run for one Source on a package path; it writes nothing (see the module docstring)."""

    path = _package_path(source)
    source_id = str(source["id"])
    source_type = str(source["type"])
    if path is None:
        raise ValueError(f"Source {source_id} is not on a device package path")
    config = dict(source.get("config") or {})
    gene = gene_class(source_type)(config=config, source_id=source_id)
    projection_scope = canonical_projection_scope(source_type, config)
    access_context = source_access_context(source)
    units: list[MigratedUnit] = []
    current = 0
    unrebuildable: list[tuple[str, str]] = []
    id_changes: list[tuple[str, str]] = []
    for stored_doc_id in sorted(await db.list_indexed_doc_ids(source_id)):
        unit = await db.find_source_unit_by_document_id(source_id, stored_doc_id, current_only=True)
        stored = await db.get_source_unit_input(unit.id) if unit is not None else None
        if unit is None or stored is None:
            unrebuildable.append((stored_doc_id, "the Unit has no stored input"))
            continue
        try:
            body = await asyncio.to_thread(doc_store.read_artifact, str(stored.raw_content_uri))
        except Exception as exc:
            unrebuildable.append((stored_doc_id, f"the stored input cannot be read: {exc}"))
            continue
        package = _decoded_package(body, path.package_kind)
        if package is None:
            current += 1
            continue
        try:
            rebuilt = path.rebuild(source_id, config, package, stored)
        except Unrebuildable as exc:
            unrebuildable.append((stored_doc_id, str(exc)))
            continue
        if path.keeps_document_ids and rebuilt.item.item_id != stored_doc_id:
            id_changes.append((stored_doc_id, rebuilt.item.item_id))
            continue
        committed = await db.get_current_source_unit_projection(unit.id)
        if committed is None:
            unrebuildable.append((stored_doc_id, "the Unit has no committed revision"))
            continue
        try:
            artifacts = committed_source_artifacts(committed, document_id=stored_doc_id)
            changes = await _changes_content(
                db, gene, unit, rebuilt, artifacts, projection_scope=projection_scope, access_context=access_context
            )
        except (StoredDocumentUnavailable, ValueError) as exc:
            unrebuildable.append((stored_doc_id, f"the rebuilt input does not project: {exc}"))
            continue
        units.append(
            MigratedUnit(
                source_unit_id=unit.id,
                stored_document_id=stored_doc_id,
                document_id=rebuilt.item.item_id,
                item=rebuilt.item,
                body=rebuilt.body,
                content_type=rebuilt.content_type,
                artifacts=artifacts,
                changes_content=changes,
            )
        )
    claims: dict[str, set[str]] = {}
    for migrated in units:
        claims.setdefault(migrated.document_id, set()).add(migrated.stored_document_id)
    migrating = {migrated.stored_document_id for migrated in units}
    for doc_id in set(claims) & set(await db.list_indexed_doc_ids(source_id)):
        # The Gene's id is already held by a Unit that is not moving off it.
        if doc_id not in migrating:
            claims[doc_id].add(doc_id)
    collisions = tuple(
        (doc_id, tuple(sorted(stored))) for doc_id, stored in sorted(claims.items()) if len(stored) > 1
    )
    return SourceMigrationPlan(
        source_id=source_id,
        source_type=source_type,
        source_name=str(source.get("name") or source_id),
        units=tuple(units),
        current_units=current,
        unrebuildable=tuple(unrebuildable),
        collisions=collisions,
        id_changes_refused=tuple(id_changes),
    )


async def _changes_content(
    db: Database,
    gene: Any,
    unit: Any,
    rebuilt: _Rebuilt,
    artifacts: tuple[StoredSourceArtifact, ...],
    *,
    projection_scope: dict[str, object],
    access_context: dict[str, object],
) -> bool:
    """Whether the rebuilt input changes the Unit's Observations from its current revision, as its replay decides."""

    raw = gene.raw_from_stored_input(rebuilt.item, rebuilt.body, rebuilt.content_type)
    normalized = await gene.normalize(raw)
    prior_unit_revision = await db.get_current_source_unit_revision(unit.id)
    projection = await DEFAULT_SOURCE_PROJECTION_ADAPTER.project(
        ProjectionEnvelope(
            request=ProjectionRequest(
                run_id="package-migration-probe",
                source_id=str(unit.source_id),
                source_type=gene.metadata().name,
                scope={
                    "configured_scope": dict(projection_scope),
                    "source_unit_id": unit.id,
                    "source_unit_provider_key": unit.provider_key,
                    "authoritative_snapshot": False,
                },
                run_mode=ProjectionRunMode.DELTA if prior_unit_revision is not None else ProjectionRunMode.FULL_SNAPSHOT,
                access_context=dict(access_context),
            ),
            item=rebuilt.item,
            raw=raw,
            normalized=normalized,
            artifacts=artifacts,
            prior_unit_revision=prior_unit_revision,
            prior_observation_revisions=await db.get_current_source_observation_revisions(unit.id),
            stored_observation_revisions={},
        )
    )
    return projection.deltas[0].requires_extraction


def _artifact_key(artifact: RawSourceArtifact | StoredSourceArtifact) -> tuple[str, str, str, str]:
    return (
        artifact.parent_observation_type,
        artifact.parent_provider_key,
        artifact.provider_key,
        artifact.provider_revision,
    )


def _raw_artifact(stored: StoredSourceArtifact) -> RawSourceArtifact:
    return RawSourceArtifact(
        provider_key=stored.provider_key,
        parent_observation_type=stored.parent_observation_type,
        parent_provider_key=stored.parent_provider_key,
        provider_revision=stored.provider_revision,
        filename=stored.filename,
        media_type=stored.media_type,
        declared_size_bytes=stored.size_bytes,
        locator=dict(stored.locator),
    )


class _ReplayRecord:
    """The rebuilt inputs of one Source, open as the record of its REPLAY run."""

    def __init__(self, units: Sequence[MigratedUnit], doc_store: DocumentStore) -> None:
        self._units = {unit.document_id: unit for unit in units}
        self._artifacts = {_artifact_key(artifact): artifact for unit in units for artifact in unit.artifacts}
        self._doc_store = doc_store

    def discovered(self) -> Sequence[ContentItem]:
        return [unit.item for unit in self._units.values()]

    def listing(self) -> ScopeListing:
        raise AcquisitionRecordMismatch("a replay lists no scope")

    def absence_answer(self, item_id: str) -> AbsenceAnswer:
        raise AcquisitionRecordMismatch("a replay confirms no absence")

    def rediscovered(self, item_id: str) -> ContentItem | None:
        raise AcquisitionRecordMismatch("a replay rediscovers nothing")

    async def acquired(self, item_id: str) -> AcquiredInput:
        unit = self._units.get(item_id)
        if unit is None:
            raise AcquisitionRecordMismatch(f"item {item_id} is not replayed")
        return AcquiredInput(
            item=copy.deepcopy(unit.item),
            body=unit.body,
            content_type=unit.content_type,
            artifacts=tuple(_raw_artifact(artifact) for artifact in unit.artifacts),
        )

    @contextlib.asynccontextmanager
    async def open_artifact(self, artifact: RawSourceArtifact) -> AsyncIterator[SourceArtifactDownload]:
        stored = self._artifacts.get(_artifact_key(artifact))
        if stored is None:
            raise AcquisitionRecordMismatch(f"artifact {artifact.provider_key} is not replayed")
        body = await asyncio.to_thread(self._doc_store.read_artifact, stored.uri)

        async def chunks() -> AsyncIterator[bytes]:
            yield body

        yield SourceArtifactDownload(chunks=chunks(), media_type=stored.media_type, content_length=len(body))

    async def close(self) -> None:
        return None


@dataclass(frozen=True, slots=True)
class _ReplayRecordStore:
    units: tuple[MigratedUnit, ...]
    doc_store: Any = field(repr=False)

    async def open(self) -> _ReplayRecord:
        return _ReplayRecord(self.units, self.doc_store)


async def apply_package_migration(
    db: Database,
    config: AppConfig,
    source_id: str,
    *,
    runtime: SyncRuntime,
) -> PackageMigrationResult:
    """Apply the migration to one Source on a package path (see the module docstring).

    The Source's sync activity is held from the plan to the executor change,
    so no other run of the Source interleaves. Raises
    ``SourceSyncActivityConflict`` while another activity holds the Source,
    ``SourceSyncRunActive`` while one of its runs is pending, and
    ``ValueError`` for a Source that is not on a package path.
    """

    from memforge.runtime import held_source_activity, run_source_sync

    async with held_source_activity(db, source_id) as (lease, _heartbeat):
        source = await db.get_source(source_id)
        if source is None:
            raise LookupError("source_not_found")
        path = _package_path(source)
        if path is None:
            raise ValueError(f"Source {source_id} is not on a device package path")
        plan = await plan_package_migration(db, runtime.doc_store, source)
        if plan.refused:
            return PackageMigrationResult(plan=plan, status="refused")
        if plan.units:
            state = await run_source_sync(
                db,
                config,
                source,
                runtime=runtime,
                execution_mode=SourceSyncMode.REPLAY,
                replay=_ReplayRecordStore(plan.units, runtime.doc_store),
                source_activity=lease,
            )
            if state.last_sync_status != "success":
                return PackageMigrationResult(
                    plan=plan, status="replay_failed", sync_status=state.last_sync_status, error=state.error_message
                )
        device_config = {
            key: value
            for key, value in acquisition_job_config(plan.source_type, source.get("config") or {}).items()
            if key not in path.package_fields
        }
        await db.change_source_executor(
            source_id,
            executor=SOURCE_EXECUTOR_DEVICE,
            config_json=json.dumps(device_config),
            now=datetime.now(timezone.utc),
        )
    return PackageMigrationResult(plan=plan, status="migrated")

