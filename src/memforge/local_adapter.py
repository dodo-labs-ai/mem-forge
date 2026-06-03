"""Service-side intake for the local CLI adapter push flow.

The local CLI adapter (``memforge adapter kb push``) sends one normalized
markdown document at a time. The service owns the inbox directory and the
package layout; the CLI never touches MemForge storage directly.

A configured ``local_markdown`` source has a stable per-source inbox under
``{docs_path}/../local-adapter-submissions/{source_id}/``. Each push writes one
JSON package, then the source's sync pipeline picks it up via
``LocalMarkdownGene.discover``.
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
from memforge.models import content_hash, slugify
from memforge.storage.database import Database

LOCAL_MARKDOWN_SOURCE_TYPE = "local_markdown"

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

    # The DB upsert refreshes documents_dir so a freshly created source picks
    # up its inbox even if the gene has not authenticated yet.
    refreshed_config = dict(source.get("config") or {})
    refreshed_config["documents_dir"] = str(inbox)
    await db.upsert_source(
        id=source_id,
        type=LOCAL_MARKDOWN_SOURCE_TYPE,
        name=source.get("name") or source_id,
        config_json=json.dumps(refreshed_config),
    )

    return {
        "source_id": source_id,
        "doc_id": doc_id,
        "vault_id": configured_vault,
        "relative_path": relative,
        "document_hash": document_hash,
        "package_path": str(package_path),
        "submitted_at": submitted_at,
    }
