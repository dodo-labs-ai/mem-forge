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

import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from memforge.config import AppConfig
from memforge.genes.local_markdown_gene import (
    LOCAL_MARKDOWN_CONTENT_ROLE,
    LOCAL_MARKDOWN_PACKAGE_KIND,
)
from memforge.github_repo_utils import (
    build_github_repo_doc_id,
    github_extension_allowed,
    github_include_extensions,
    github_include_paths,
    github_path_in_scope,
    normalize_github_relative_path,
    parse_github_repo_url,
)
from memforge.models import content_hash, slugify
from memforge.storage.database import Database
from memforge.storage.document_store import DocumentStore

LOCAL_MARKDOWN_SOURCE_TYPE = "local_markdown"
GITHUB_REPO_SOURCE_TYPE = "github_repo"
JIRA_SOURCE_TYPE = "jira"
TEAMS_SOURCE_TYPE = "teams"
GITHUB_REPO_PACKAGE_KIND = "github_repo_document"
GITHUB_REPO_CONTENT_ROLE = "repository_file"
JIRA_PACKAGE_KIND = "jira_document"
JIRA_CONTENT_ROLE = "jira_issue"
TEAMS_PACKAGE_KIND = "teams_window_document"
TEAMS_CONTENT_ROLE = "teams_conversation_window"

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_local_adapter_inbox(config: AppConfig, source_id: str) -> Path:
    """Return the per-source inbox directory used by the local adapter."""
    base = Path(config.storage.docs_path).parent / "local-adapter-submissions"
    return base / slugify(source_id)


def build_local_markdown_doc_id(*, source_id: str, vault_id: str, relative_path: str) -> str:
    """Stable doc id for one markdown file in a configured local source."""
    identity = "|".join([source_id.strip(), vault_id.strip(), relative_path.strip()])
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return "-".join([
        "local-md",
        slugify(source_id)[:30],
        slugify(relative_path)[:50] or "doc",
        digest,
    ])


def build_jira_doc_id(*, source_id: str, issue_key: str) -> str:
    """Stable doc id for one Jira issue pushed by the local daemon."""
    identity = "|".join([source_id.strip(), issue_key.strip().upper()])
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return "-".join([
        "jira",
        slugify(source_id)[:30],
        slugify(issue_key)[:30] or "issue",
        digest,
    ])


def build_teams_doc_id(*, source_id: str, window_id: str) -> str:
    """Stable doc id for one Teams conversation window pushed by the local daemon."""
    identity = "|".join([source_id.strip(), window_id.strip()])
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return "-".join([
        "teams",
        slugify(source_id)[:30],
        slugify(window_id)[:50] or "window",
        digest,
    ])


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


def _normalize_issue_key(value: str) -> str:
    key = (value or "").strip().upper()
    if not key:
        raise ValueError("issue_key is required")
    if not all(ch.isalnum() or ch in {"-", "_"} for ch in key):
        raise ValueError("issue_key contains unsupported characters")
    return key


def _markdown_title(markdown_body: str, fallback: str) -> str:
    for line in markdown_body.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            extracted = stripped[2:].strip()
            if extracted:
                return extracted
    return fallback


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _hash_json(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


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
    package_uri = document_store.store_raw(
        source.get("name") or source_id,
        f"{doc_id}-{package_sha256}-package",
        payload_bytes,
        "application/json",
        extension=extension,
    )
    return package_uri, package_sha256


def _package_manifest_entry(
    *,
    doc_id: str,
    title: str,
    source_url: str,
    last_modified: str,
    space_or_project: str,
    version: str,
    package_uri: str,
    package_path: str,
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
        "package_path": package_path,
        "submitted_at": submitted_at,
        "submitted_by": submitted_by,
    }
    if extra:
        entry.update(extra)
    return entry


def _jira_title_from_payload(payload: dict[str, Any], fallback: str) -> str:
    fields = payload.get("fields") if isinstance(payload.get("fields"), dict) else {}
    summary = str(fields.get("summary") or "").strip()
    key = str(payload.get("key") or fallback).strip()
    return summary or key or fallback


def _teams_title_from_payload(payload: dict[str, Any], fallback: str) -> str:
    title = str(payload.get("title") or "").strip()
    if title:
        return title
    channel = str(payload.get("channel_name") or "").strip()
    team = str(payload.get("team_name") or "").strip()
    if channel:
        return f"{team} / {channel}" if team else channel
    conversation_type = str(payload.get("conversation_type") or "").strip()
    if conversation_type == "group_chat":
        return str(payload.get("team_name") or "Group chat").strip()
    if conversation_type == "individual_chat":
        participants = payload.get("participants")
        if isinstance(participants, list):
            names = ", ".join(str(name) for name in participants if str(name).strip())
            if names:
                return names
        return "Direct message"
    return fallback


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
    submitted_by: str | None = None,
    submitted_at: str | None = None,
    document_store: DocumentStore | None = None,
) -> dict[str, Any]:
    """Validate, package, and persist one local repository file push.

    ``markdown_body`` is the raw file text and ``content_type`` declares its
    format. Conversion to markdown happens later, in the gene's ``normalize``.
    """
    if source.get("type") != LOCAL_MARKDOWN_SOURCE_TYPE:
        raise ValueError(
            f"source {source.get('id')} is type {source.get('type')!r}, not 'local_markdown'"
        )

    configured_vault = str((source.get("config") or {}).get("vault_id") or "").strip()
    if not configured_vault:
        raise ValueError("source has no configured vault_id")
    if vault_id.strip() != configured_vault:
        raise ValueError(
            f"vault_id {vault_id!r} does not match the source's configured vault_id "
            f"{configured_vault!r}"
        )
    if not markdown_body or not markdown_body.strip():
        raise ValueError("markdown_body is required")

    relative = _normalize_relative_path(relative_path)
    submitted_at = submitted_at or _now_iso()
    source_id = str(source["id"])
    inbox = default_local_adapter_inbox(config, source_id)
    inbox.mkdir(parents=True, exist_ok=True)

    document_hash = content_hash(markdown_body)
    doc_id = build_local_markdown_doc_id(
        source_id=source_id,
        vault_id=configured_vault,
        relative_path=relative,
    )
    doc_title = (title or "").strip() or _markdown_title(markdown_body, fallback=relative)
    source_url = f"local-adapter://{slugify(source_id)}/{slugify(configured_vault)}/{relative}"
    package_path = inbox / f"{doc_id}.json"
    package_path.parent.mkdir(parents=True, exist_ok=True)

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

    payload_text = json.dumps(package, indent=2, sort_keys=True)
    package_uri, package_sha256 = _store_package_artifact(
        document_store=document_store,
        source=source,
        source_id=source_id,
        doc_id=doc_id,
        payload_text=payload_text,
        extension=".local-package.json",
    )
    package_existed = package_path.exists()
    package_written = False
    tmp_fd, tmp_name = tempfile.mkstemp(dir=str(package_path.parent), suffix=".json.tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
            handle.write(payload_text)
        os.replace(tmp_name, package_path)
        package_written = True
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        if package_written and not package_existed:
            try:
                os.unlink(package_path)
            except OSError:
                pass
        raise

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
            package_path=str(package_path),
            submitted_at=submitted_at,
            submitted_by=submitted_by,
            extra={
                "vault_id": configured_vault,
                "relative_path": relative,
                "content_type": content_type,
                "raw_hash": raw_hash,
            },
        )
    return {
        "source_id": source_id,
        "doc_id": doc_id,
        "vault_id": configured_vault,
        "relative_path": relative,
        "document_hash": document_hash,
        "package_path": str(package_path),
        "package_uri": package_uri,
        "package_sha256": package_sha256,
        "package_manifest_entry": manifest_entry,
        "submitted_at": submitted_at,
    }


async def submit_github_repo_document(
    *,
    db: Database,
    config: AppConfig,
    source: dict[str, Any],
    repo_url: str,
    repo_ref: str,
    relative_path: str,
    markdown_body: str,
    content_type: str = "text/markdown",
    title: str | None = None,
    raw_hash: str | None = None,
    blob_sha: str | None = None,
    submitted_by: str | None = None,
    submitted_at: str | None = None,
    document_store: DocumentStore | None = None,
) -> dict[str, Any]:
    """Validate, package, and persist one GitHub repository file push."""
    if source.get("type") != GITHUB_REPO_SOURCE_TYPE:
        raise ValueError(
            f"source {source.get('id')} is type {source.get('type')!r}, not 'github_repo'"
        )

    configured_repo = str((source.get("config") or {}).get("repo_url") or "").strip()
    if not configured_repo:
        raise ValueError("source has no configured repo_url")
    connection_mode = str((source.get("config") or {}).get("connection_mode") or "cloud_pull").strip().lower()
    if connection_mode != "local_push":
        raise ValueError("GitHub Repository adapter push requires Internal network / VPN access")
    if _canonical_repo_url(repo_url) != _canonical_repo_url(configured_repo):
        raise ValueError(
            f"repo_url {repo_url!r} does not match the source's configured repo_url "
            f"{configured_repo!r}"
        )
    if not markdown_body or not markdown_body.strip():
        raise ValueError("markdown_body is required")

    relative = normalize_github_relative_path(relative_path)
    submitted_at = submitted_at or _now_iso()
    source_id = str(source["id"])
    source_config = dict(source.get("config") or {})
    configured_ref = str(source_config.get("ref") or "main").strip() or "main"
    ref = (repo_ref or configured_ref).strip() or configured_ref
    if ref != configured_ref:
        raise ValueError(f"repo_ref {ref!r} does not match the source's configured ref {configured_ref!r}")
    include_paths = github_include_paths(source_config)
    if not github_path_in_scope(relative, include_paths):
        raise ValueError("relative_path is outside the source's configured include_paths")
    include_extensions = github_include_extensions(source_config)
    if not github_extension_allowed(relative, include_extensions):
        raise ValueError("relative_path extension is outside the source's configured include_extensions")
    repo = _repo_parts(configured_repo)
    inbox = default_local_adapter_inbox(config, source_id)
    inbox.mkdir(parents=True, exist_ok=True)

    document_hash = content_hash(markdown_body)
    doc_id = build_github_repo_doc_id(
        source_id=source_id,
        repo_url=repo["repo_url"],
        repo_ref=ref,
        relative_path=relative,
    )
    doc_title = (title or "").strip() or _markdown_title(markdown_body, fallback=relative)
    source_url = _github_file_url(repo["repo_url"], ref, relative)
    package_path = inbox / f"{doc_id}.json"
    package_path.parent.mkdir(parents=True, exist_ok=True)
    max_files = _positive_int(source_config.get("max_files"), default=500)
    if not package_path.exists() and _github_package_count(inbox, source_config) >= max_files:
        raise ValueError(f"GitHub Repository Internal network / VPN source already has max_files={max_files} packages")

    package = {
        "package_kind": GITHUB_REPO_PACKAGE_KIND,
        "content_role": GITHUB_REPO_CONTENT_ROLE,
        "doc_id": doc_id,
        "title": doc_title,
        "source_url": source_url,
        "last_modified": submitted_at,
        "space_or_project": f"{repo['owner']}/{repo['name']}",
        "version": blob_sha or raw_hash or document_hash,
        "repo_url": repo["repo_url"],
        "repo_host": repo["host"],
        "repo_owner": repo["owner"],
        "repo_name": repo["name"],
        "repo_ref": ref,
        "relative_path": relative,
        "blob_sha": blob_sha,
        "content_type": content_type,
        "raw_hash": raw_hash,
        "submitted_at": submitted_at,
        "submitted_by": submitted_by,
        "markdown": markdown_body,
    }

    payload_text = json.dumps(package, indent=2, sort_keys=True)
    package_uri, package_sha256 = _store_package_artifact(
        document_store=document_store,
        source=source,
        source_id=source_id,
        doc_id=doc_id,
        payload_text=payload_text,
        extension=".github-repo-package.json",
    )
    package_existed = package_path.exists()
    package_written = False
    tmp_fd, tmp_name = tempfile.mkstemp(dir=str(package_path.parent), suffix=".json.tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
            handle.write(payload_text)
        os.replace(tmp_name, package_path)
        package_written = True
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        if package_written and not package_existed:
            try:
                os.unlink(package_path)
            except OSError:
                pass
        raise

    manifest_entry = None
    if package_uri:
        manifest_entry = _package_manifest_entry(
            doc_id=doc_id,
            title=doc_title,
            source_url=source_url,
            last_modified=submitted_at,
            space_or_project=f"{repo['owner']}/{repo['name']}",
            version=blob_sha or raw_hash or document_hash,
            package_uri=package_uri,
            package_path=str(package_path),
            submitted_at=submitted_at,
            submitted_by=submitted_by,
            extra={
                "repo_url": repo["repo_url"],
                "repo_host": repo["host"],
                "repo_owner": repo["owner"],
                "repo_name": repo["name"],
                "repo_ref": ref,
                "relative_path": relative,
                "blob_sha": blob_sha,
                "content_type": content_type,
                "raw_hash": raw_hash,
            },
        )
    return {
        "source_id": source_id,
        "doc_id": doc_id,
        "repo_url": repo["repo_url"],
        "repo_ref": ref,
        "relative_path": relative,
        "document_hash": document_hash,
        "package_path": str(package_path),
        "package_uri": package_uri,
        "package_sha256": package_sha256,
        "package_manifest_entry": manifest_entry,
        "submitted_at": submitted_at,
    }


async def submit_jira_package(
    *,
    db: Database,
    config: AppConfig,
    source: dict[str, Any],
    base_url: str,
    issue_key: str,
    raw_payload: dict[str, Any],
    source_url: str | None = None,
    title: str | None = None,
    raw_hash: str | None = None,
    submitted_by: str | None = None,
    submitted_at: str | None = None,
    document_store: DocumentStore | None = None,
) -> dict[str, Any]:
    """Validate, package, and persist one raw Jira issue captured by the local daemon."""
    if source.get("type") != JIRA_SOURCE_TYPE:
        raise ValueError(f"source {source.get('id')} is type {source.get('type')!r}, not 'jira'")

    source_config = dict(source.get("config") or {})
    if str(source_config.get("sync_mode") or "cloud").strip().lower() != "local_agent":
        raise ValueError("Jira local adapter pushes require sync_mode=local_agent")
    if not isinstance(raw_payload, dict) or not raw_payload:
        raise ValueError("raw_payload is required")
    configured_base_url = str(source_config.get("base_url") or "").strip().rstrip("/")
    if not configured_base_url:
        raise ValueError("source has no configured base_url")
    actual_base_url = (base_url or configured_base_url).strip().rstrip("/")
    if actual_base_url != configured_base_url:
        raise ValueError(
            f"base_url {actual_base_url!r} does not match the source's configured base_url {configured_base_url!r}"
        )

    normalized_issue_key = _normalize_issue_key(issue_key or str(raw_payload.get("key") or ""))
    submitted_at = submitted_at or _now_iso()
    source_id = str(source["id"])
    inbox = default_local_adapter_inbox(config, source_id)
    inbox.mkdir(parents=True, exist_ok=True)

    payload_hash = raw_hash or _hash_json(raw_payload)
    doc_id = build_jira_doc_id(source_id=source_id, issue_key=normalized_issue_key)
    doc_title = (title or "").strip() or _jira_title_from_payload(raw_payload, normalized_issue_key)
    issue_url = (source_url or f"{configured_base_url}/browse/{normalized_issue_key}").strip()
    package_path = inbox / f"{doc_id}.json"
    package_path.parent.mkdir(parents=True, exist_ok=True)

    package = {
        "package_kind": JIRA_PACKAGE_KIND,
        "content_role": JIRA_CONTENT_ROLE,
        "doc_id": doc_id,
        "title": doc_title,
        "source_url": issue_url,
        "last_modified": submitted_at,
        "space_or_project": normalized_issue_key.split("-", 1)[0],
        "version": payload_hash,
        "base_url": configured_base_url,
        "issue_key": normalized_issue_key,
        "content_type": "application/json",
        "raw_hash": payload_hash,
        "submitted_at": submitted_at,
        "submitted_by": submitted_by,
        "raw_payload": raw_payload,
    }

    payload_text = json.dumps(package, indent=2, sort_keys=True)
    package_uri, package_sha256 = _store_package_artifact(
        document_store=document_store,
        source=source,
        source_id=source_id,
        doc_id=doc_id,
        payload_text=payload_text,
        extension=".jira-package.json",
    )
    tmp_fd, tmp_name = tempfile.mkstemp(dir=str(package_path.parent), suffix=".json.tmp")
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

    manifest_entry = None
    if package_uri:
        manifest_entry = _package_manifest_entry(
            doc_id=doc_id,
            title=doc_title,
            source_url=issue_url,
            last_modified=submitted_at,
            space_or_project=normalized_issue_key.split("-", 1)[0],
            version=payload_hash,
            package_uri=package_uri,
            package_path=str(package_path),
            submitted_at=submitted_at,
            submitted_by=submitted_by,
            extra={
                "base_url": configured_base_url,
                "issue_key": normalized_issue_key,
                "content_type": "application/json",
                "raw_hash": payload_hash,
            },
        )
    return {
        "source_id": source_id,
        "doc_id": doc_id,
        "base_url": configured_base_url,
        "issue_key": normalized_issue_key,
        "document_hash": payload_hash,
        "package_path": str(package_path),
        "package_uri": package_uri,
        "package_sha256": package_sha256,
        "package_manifest_entry": manifest_entry,
        "submitted_at": submitted_at,
    }


async def submit_teams_window_package(
    *,
    db: Database,
    config: AppConfig,
    source: dict[str, Any],
    conversation_id: str,
    window_id: str,
    revision_hash: str,
    raw_payload: dict[str, Any],
    title: str | None = None,
    root_message_id: str | None = None,
    window_type: str | None = None,
    source_url: str | None = None,
    raw_hash: str | None = None,
    submitted_by: str | None = None,
    submitted_at: str | None = None,
    document_store: DocumentStore | None = None,
) -> dict[str, Any]:
    """Validate, package, and persist one raw Teams window captured by the local daemon."""
    if source.get("type") != TEAMS_SOURCE_TYPE:
        raise ValueError(f"source {source.get('id')} is type {source.get('type')!r}, not 'teams'")
    if not isinstance(raw_payload, dict) or not raw_payload:
        raise ValueError("raw_payload is required")

    normalized_conversation_id = (conversation_id or "").strip()
    normalized_window_id = (window_id or "").strip()
    normalized_revision_hash = (revision_hash or "").strip()
    if not normalized_conversation_id:
        raise ValueError("conversation_id is required")
    if not normalized_window_id:
        raise ValueError("window_id is required")
    if not normalized_revision_hash:
        raise ValueError("revision_hash is required")

    submitted_at = submitted_at or _now_iso()
    source_id = str(source["id"])
    inbox = default_local_adapter_inbox(config, source_id)
    inbox.mkdir(parents=True, exist_ok=True)

    payload_hash = raw_hash or _hash_json(raw_payload)
    doc_id = build_teams_doc_id(source_id=source_id, window_id=normalized_window_id)
    doc_title = (title or "").strip() or _teams_title_from_payload(raw_payload, normalized_window_id)
    package_path = inbox / f"{doc_id}.json"
    package_path.parent.mkdir(parents=True, exist_ok=True)

    package = {
        "package_kind": TEAMS_PACKAGE_KIND,
        "content_role": TEAMS_CONTENT_ROLE,
        "doc_id": doc_id,
        "title": doc_title,
        "source_url": (source_url or f"teams-window://{source_id}/{doc_id}/{normalized_revision_hash}").strip(),
        "last_modified": submitted_at,
        "space_or_project": str(
            raw_payload.get("team_name") or raw_payload.get("channel_name") or raw_payload.get("title") or source.get("name") or ""
        ),
        "version": normalized_revision_hash,
        "conversation_id": normalized_conversation_id,
        "root_message_id": (root_message_id or "").strip(),
        "window_id": normalized_window_id,
        "window_type": (window_type or "").strip(),
        "revision_hash": normalized_revision_hash,
        "content_type": "application/json",
        "raw_hash": payload_hash,
        "submitted_at": submitted_at,
        "submitted_by": submitted_by,
        "raw_payload": raw_payload,
    }

    payload_text = json.dumps(package, indent=2, sort_keys=True)
    package_uri, package_sha256 = _store_package_artifact(
        document_store=document_store,
        source=source,
        source_id=source_id,
        doc_id=doc_id,
        payload_text=payload_text,
        extension=".teams-package.json",
    )
    tmp_fd, tmp_name = tempfile.mkstemp(dir=str(package_path.parent), suffix=".json.tmp")
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

    manifest_entry = None
    if package_uri:
        manifest_entry = _package_manifest_entry(
            doc_id=doc_id,
            title=doc_title,
            source_url=package["source_url"],
            last_modified=submitted_at,
            space_or_project=package["space_or_project"],
            version=normalized_revision_hash,
            package_uri=package_uri,
            package_path=str(package_path),
            submitted_at=submitted_at,
            submitted_by=submitted_by,
            extra={
                "conversation_id": normalized_conversation_id,
                "root_message_id": package["root_message_id"],
                "window_id": normalized_window_id,
                "window_type": package["window_type"],
                "revision_hash": normalized_revision_hash,
            },
        )
    return {
        "source_id": source_id,
        "doc_id": doc_id,
        "conversation_id": normalized_conversation_id,
        "window_id": normalized_window_id,
        "revision_hash": normalized_revision_hash,
        "document_hash": payload_hash,
        "package_path": str(package_path),
        "package_uri": package_uri,
        "package_sha256": package_sha256,
        "package_manifest_entry": manifest_entry,
        "submitted_at": submitted_at,
    }


def _canonical_repo_url(repo_url: str) -> str:
    return _repo_parts(repo_url)["repo_url"]


def _repo_parts(repo_url: str) -> dict[str, str]:
    parsed = parse_github_repo_url(repo_url)
    return {"repo_url": parsed["repo_url"], "host": parsed["host"], "owner": parsed["owner"], "name": parsed["repo"]}


def _github_file_url(repo_url: str, repo_ref: str, relative_path: str) -> str:
    from urllib.parse import quote

    return f"{repo_url}/blob/{quote(repo_ref, safe='')}/{quote(relative_path, safe='/')}"


def _positive_int(value: object, *, default: int) -> int:
    try:
        return max(int(value), 1)
    except (TypeError, ValueError):
        return default


def _github_package_count(inbox: Path, config: dict[str, Any]) -> int:
    count = 0
    for package_path in inbox.glob("*.json"):
        try:
            package = json.loads(package_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if package.get("package_kind") == GITHUB_REPO_PACKAGE_KIND:
            try:
                repo_matches = _canonical_repo_url(str(package.get("repo_url") or "")) == _canonical_repo_url(
                    str(config.get("repo_url") or "")
                )
                ref_matches = str(package.get("repo_ref") or "").strip() == (
                    str(config.get("ref") or "main").strip() or "main"
                )
                path = normalize_github_relative_path(str(package.get("relative_path") or ""))
            except ValueError:
                continue
            if (
                repo_matches
                and ref_matches
                and github_path_in_scope(path, github_include_paths(config))
                and github_extension_allowed(path, github_include_extensions(config))
            ):
                count += 1
    return count
