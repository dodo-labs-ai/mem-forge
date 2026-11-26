"""Service-side intake for local daemon/adapter source packages.

Local clients send raw file text or structured source payloads. The service owns
the inbox directory and package layout; local clients never touch MemForge
storage directly and never own canonical document normalization.

A configured ``local_markdown`` source has a stable per-source inbox under
``{docs_path}/../local-adapter-submissions/{source_id}/``. Each push writes one
JSON package, then the source's sync pipeline picks it up via
the corresponding source gene.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from memforge.config import AppConfig
from memforge.genes.local_markdown_gene import (
    LOCAL_MARKDOWN_CONTENT_ROLE,
    LOCAL_MARKDOWN_PACKAGE_KIND,
)
from memforge.local_agent.document_identity import build_local_markdown_doc_id
from memforge.local_agent.snapshot_manifest import UPSERT
from memforge.local_agent.source_contract import local_file_package_semantic_hash
from memforge.models import content_hash, slugify
from memforge.storage.database import Database
from memforge.storage.document_store import DocumentStore
from memforge.source_time import SOURCE_UPDATED_AT_KEY, source_time_iso

LOCAL_MARKDOWN_SOURCE_TYPE = "local_markdown"

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_local_adapter_inbox(config: AppConfig, source_id: str) -> Path:
    """Return the per-source inbox directory used by the local adapter."""
    base = Path(config.storage.docs_path).parent / "local-adapter-submissions"
    return base / slugify(source_id)


def _normalize_relative_path(value: str) -> str:
    """Reject paths that try to escape the vault or use absolute paths."""
    candidate = (value or "").strip().lstrip("/").lstrip("\\")
    if not candidate:
        raise ValueError("relative_path is required")
    parts = candidate.replace("\\", "/").split("/")
    cleaned = [part for part in parts if part not in ("", ".")]
    if any(part == ".." for part in cleaned):
        raise ValueError("relative_path must not contain '..' segments")
    return "/".join(cleaned)


def _markdown_title(markdown_body: str, fallback: str) -> str:
    for line in markdown_body.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            extracted = stripped[2:].strip()
            if extracted:
                return extracted
    return fallback


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _store_package_artifact(
    *,
    document_store: DocumentStore | None,
    source: dict[str, Any],
    source_id: str,
    doc_id: str,
    payload_text: str,
    extension: str,
) -> tuple[str | None, str]:
    payload_bytes = payload_text.encode("utf-8")
    package_sha256 = _hash_bytes(payload_bytes)
    if document_store is None:
        return None, package_sha256
    # A push does not hold the Source activity lease, so each attempt writes a
    # key of its own: a cleanup task queued by an earlier failed push can never
    # name the object written here.
    package_uri = document_store.store_raw(
        source_id,
        doc_id,
        # The Document namespace is already in the key; the stem stays within
        # the slug length limit so the attempt part is never cut off.
        f"package-{uuid.uuid4().hex}-{package_sha256}",
        payload_bytes,
        "application/json",
        extension=extension,
    )
    return package_uri, package_sha256


def _write_inbox_package(
    *,
    config: AppConfig,
    source_id: str,
    doc_id: str,
    payload_text: str,
    document_store: DocumentStore | None,
) -> str | None:
    """Write the legacy/direct local inbox only when no artifact store exists.

    Server intake owns one canonical artifact URI.  Creating a second mutable
    inbox copy made dedupe, rollback, and Source deletion impossible to close
    transactionally across both representations.
    """
    if document_store is not None:
        return None
    inbox = default_local_adapter_inbox(config, source_id)
    inbox.mkdir(parents=True, exist_ok=True)
    package_path = inbox / f"{doc_id}.json"
    tmp_fd, tmp_name = tempfile.mkstemp(
        dir=str(package_path.parent),
        suffix=".json.tmp",
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
            handle.write(payload_text)
        os.replace(tmp_name, package_path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return str(package_path)


async def _persist_package(
    *,
    config: AppConfig,
    document_store: DocumentStore | None,
    source: dict[str, Any],
    source_id: str,
    doc_id: str,
    payload_text: str,
    extension: str,
) -> tuple[str | None, str, str | None]:
    """Persist one raw package without blocking the async request loop."""

    def persist() -> tuple[str | None, str, str | None]:
        package_uri, package_sha256 = _store_package_artifact(
            document_store=document_store,
            source=source,
            source_id=source_id,
            doc_id=doc_id,
            payload_text=payload_text,
            extension=extension,
        )
        package_path = _write_inbox_package(
            config=config,
            source_id=source_id,
            doc_id=doc_id,
            payload_text=payload_text,
            document_store=document_store,
        )
        return package_uri, package_sha256, package_path

    return await asyncio.to_thread(persist)


def _package_manifest_entry(
    *,
    doc_id: str,
    title: str,
    source_url: str,
    last_modified: str,
    space_or_project: str,
    version: str,
    package_uri: str,
    package_path: str | None,
    submitted_at: str,
    submitted_by: str | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    entry = {
        "doc_id": doc_id,
        "title": title,
        "source_url": source_url,
        "last_modified": last_modified,
        "space_or_project": space_or_project,
        "version": version,
        "package_uri": package_uri,
        "submitted_at": submitted_at,
        "submitted_by": submitted_by,
        "change_kind": UPSERT,
    }
    if package_path:
        entry["package_path"] = package_path
    if extra:
        entry.update(extra)
    return entry


async def submit_local_markdown_document(
    *,
    db: Database,
    config: AppConfig,
    source: dict[str, Any],
    vault_id: str,
    relative_path: str,
    markdown_body: str,
    content_type: str = "text/markdown",
    title: str | None = None,
    raw_hash: str | None = None,
    source_updated_at: str | None = None,
    submitted_by: str | None = None,
    submitted_at: str | None = None,
    document_store: DocumentStore | None = None,
) -> dict[str, Any]:
    """Validate, package, and persist one local repository file push.

    ``markdown_body`` is the raw file text and ``content_type`` declares its
    format. Conversion to markdown happens later, in the gene's ``normalize``.
    ``source_updated_at`` is the file's own time as the local agent reads it
    (its commit time, or its modification time outside a clean Git checkout);
    without it the file has no source time. ``submitted_at``, the current time
    when omitted, is the package's time, never its source time: discovery
    orders packages by it, an incremental run reads only packages submitted
    after its cursor, and a package submitted again reads as changed.
    """
    if source.get("type") != LOCAL_MARKDOWN_SOURCE_TYPE:
        raise ValueError(f"source {source.get('id')} is type {source.get('type')!r}, not 'local_markdown'")

    configured_vault = str((source.get("config") or {}).get("vault_id") or "").strip()
    if not configured_vault:
        raise ValueError("source has no configured vault_id")
    if vault_id.strip() != configured_vault:
        raise ValueError(f"vault_id {vault_id!r} does not match the source's configured vault_id {configured_vault!r}")
    if not isinstance(markdown_body, str):
        raise ValueError("markdown_body is required")

    relative = _normalize_relative_path(relative_path)
    content_time = source_time_iso(source_updated_at)
    submitted_at = submitted_at or _now_iso()
    source_id = str(source["id"])
    document_hash = content_hash(markdown_body)
    doc_id = build_local_markdown_doc_id(
        source_id=source_id,
        vault_id=configured_vault,
        relative_path=relative,
    )
    doc_title = (title or "").strip() or _markdown_title(markdown_body, fallback=relative)
    source_url = f"local-adapter://{slugify(source_id)}/{slugify(configured_vault)}/{relative}"
    package = {
        "package_kind": LOCAL_MARKDOWN_PACKAGE_KIND,
        "content_role": LOCAL_MARKDOWN_CONTENT_ROLE,
        "doc_id": doc_id,
        "title": doc_title,
        "source_url": source_url,
        "last_modified": submitted_at,
        "space_or_project": configured_vault,
        "version": document_hash,
        "vault_id": configured_vault,
        "relative_path": relative,
        "content_type": content_type,
        "raw_hash": raw_hash,
        "submitted_at": submitted_at,
        "submitted_by": submitted_by,
        "markdown": markdown_body,
    }
    if content_time is not None:
        package[SOURCE_UPDATED_AT_KEY] = content_time

    payload_text = json.dumps(package, indent=2, sort_keys=True)
    package_uri, package_sha256, package_path = await _persist_package(
        config=config,
        document_store=document_store,
        source=source,
        source_id=source_id,
        doc_id=doc_id,
        payload_text=payload_text,
        extension=".local-package.json",
    )

    manifest_entry = None
    if package_uri:
        manifest_entry = _package_manifest_entry(
            doc_id=doc_id,
            title=doc_title,
            source_url=source_url,
            last_modified=submitted_at,
            space_or_project=configured_vault,
            version=document_hash,
            package_uri=package_uri,
            package_path=package_path,
            submitted_at=submitted_at,
            submitted_by=submitted_by,
            extra={
                "vault_id": configured_vault,
                "relative_path": relative,
                "content_type": content_type,
                "raw_hash": raw_hash,
                **({SOURCE_UPDATED_AT_KEY: content_time} if content_time is not None else {}),
            },
        )
    return {
        "source_id": source_id,
        "doc_id": doc_id,
        "vault_id": configured_vault,
        "relative_path": relative,
        "document_hash": document_hash,
        "semantic_hash": local_file_package_semantic_hash(document_hash, content_time),
        "package_path": package_path,
        "package_uri": package_uri,
        "package_sha256": package_sha256,
        "package_manifest_entry": manifest_entry,
        "submitted_at": submitted_at,
    }


async def submit_local_markdown_request(
    req: Any,
    *,
    db: Any,
    workspace_id: str,
    config: AppConfig,
    source: dict[str, Any],
    document_store: Any,
    sync_snapshot_id: str,
) -> dict[str, Any]:
    """Submit one local markdown file the local agent snapshotted."""

    return await submit_local_markdown_document(
        db=db,
        config=config,
        source=source,
        vault_id=req.vault_id or "",
        relative_path=req.relative_path,
        markdown_body=req.markdown_body,
        content_type=req.content_type,
        title=req.title,
        raw_hash=req.raw_hash,
        source_updated_at=req.source_updated_at,
        submitted_by=req.submitted_by,
        submitted_at=req.submitted_at,
        document_store=document_store,
    )
