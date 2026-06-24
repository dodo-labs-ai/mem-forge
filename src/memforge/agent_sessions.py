"""Client-generated agent session document intake."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from memforge.agent_session_contract import (
    AGENT_SESSION_CONTENT_ROLE,
    AGENT_SESSION_PACKAGE_KIND,
)
from memforge.config import AppConfig
from memforge.agent_knowledge import (
    AgentKnowledgeBundleService,
    AgentKnowledgePatchProposal,
    render_agent_knowledge_patch_prompt,
)
from memforge.memory.project_resolver import resolve_project_key
from memforge.models import AgentHookReceipt, AgentSessionReceipt, content_hash, slugify
from memforge.storage.database import Database

# Historical singleton source id. New writes use the per-client source derived
# by agent_session_source_id().
AGENT_SESSION_SOURCE_ID = "src-agent-sessions"
AGENT_SESSION_SOURCE_TYPE = "agent_session"
AGENT_SESSION_SOURCE_KIND = "generated_agent_summary"
AGENT_SESSION_WINDOW_SOURCE_KIND = "generated_agent_window_summary"
AGENT_SESSION_KNOWLEDGE_PATCH_MAX_TOKENS = 8192

# Whitelisted clients with well-known source ids and display names.
_KNOWN_CLIENT_SOURCE_IDS: dict[str, str] = {
    "codex": "src-agent-sessions-codex",
    "claude-code": "src-agent-sessions-claude-code",
}
_KNOWN_CLIENT_SOURCE_NAMES: dict[str, str] = {
    "codex": "Codex Session",
    "claude-code": "Claude Code Session",
}


def agent_session_source_id(client: str) -> str:
    """Return the canonical source id for the given client.

    Whitelisted clients ('codex', 'claude-code') map to their well-known ids.
    Any other client falls back to 'src-agent-sessions-<slug>'.
    """
    return _KNOWN_CLIENT_SOURCE_IDS.get(client, f"src-agent-sessions-{slugify(client)}")


def agent_session_source_name(client: str) -> str:
    """Return the display name for the given client's agent-session source."""
    return _KNOWN_CLIENT_SOURCE_NAMES.get(client, f"{client.title()} Session")


def normalize_repo_identifier(repo: str | None) -> str | None:
    """Return the stable repository identity used for agent-session grouping.

    Remote URLs are normalized to ``host/org/repo`` without protocol, user, or
    ``.git`` suffix. Plain repo slugs are lower-cased and returned unchanged.
    """
    if repo is None:
        return None
    value = repo.strip()
    if not value:
        return None

    ssh_match = re.match(r"^[^/@]+@([^:/]+):(.+)$", value)
    if ssh_match:
        host, path = ssh_match.groups()
        value = f"{host}/{path}"
    else:
        value = re.sub(r"^[a-z][a-z0-9+.-]*://", "", value, flags=re.IGNORECASE)
        value = re.sub(r"^[^@/]+@", "", value)

    value = value.split("?", 1)[0].split("#", 1)[0].rstrip("/")
    if value.endswith(".git"):
        value = value[:-4]
    value = re.sub(r"/+", "/", value)
    return value.lower() or None


# Reverse-lookup: given a per-client source id, return the originating client.
# Used by /api/sources to attach `client` to each row so the UI can pick a brand.
_AGENT_SESSION_ID_TO_CLIENT: dict[str, str] = {
    source_id: client for client, source_id in _KNOWN_CLIENT_SOURCE_IDS.items()
}
_AGENT_SESSION_ID_PREFIX = "src-agent-sessions-"


def agent_session_client_for_source_id(source_id: str) -> str | None:
    """Return the client slug for an agent-session source id, or None.

    Returns the whitelisted client for known ids; for unknown agent-session
    sources prefixed with 'src-agent-sessions-' returns the trailing slug.
    Returns None for any other source id (jira, local_markdown, etc.).
    """
    if source_id in _AGENT_SESSION_ID_TO_CLIENT:
        return _AGENT_SESSION_ID_TO_CLIENT[source_id]
    if source_id.startswith(_AGENT_SESSION_ID_PREFIX):
        return source_id[len(_AGENT_SESSION_ID_PREFIX) :] or None
    return None


_SECRET_PATTERNS = [
    ("assignment", re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*([^\s`'\"]+)")),
    ("json_assignment", re.compile(r"(?i)([\"']?(?:api[_-]?key|token|secret|password)[\"']?\s*:\s*[\"'])[^\"']+")),
    ("bearer", re.compile(r"(?i)(bearer\s+)[a-z0-9._~+/=-]{16,}")),
    ("openai_key", re.compile(r"(?i)sk-[a-z0-9_-]{12,}")),
]
_CANONICAL_EVENT_KINDS = {
    "user_message",
    "assistant_message",
    "tool_call",
    "tool_result",
    "file_change",
    "command_result",
    "decision",
    "error",
}
_NOISE_EVENT_TYPES = {
    "session_meta",
    "turn_context",
    "compacted",
    "context_compaction",
    "compaction",
    "compaction_trigger",
    "task_started",
    "reasoning",
    "thinking",
    "system",
    "queue-operation",
    "last-prompt",
    "attachment",
}
_TOOL_CALL_TYPES = {
    "function_call",
    "tool_call",
    "tool_use",
    "custom_tool_call",
    "local_shell_call",
    "web_search_call",
}
_TOOL_RESULT_TYPES = {
    "function_call_output",
    "tool_result",
    "tool_output",
    "tool_use_result",
    "custom_tool_call_output",
}
_MAX_CANONICAL_EVENT_TEXT_CHARS = 4_000


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_submitted_at(value: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_source_observed_at(value: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("source_observed_at must include an explicit timezone offset")
    return parsed.astimezone(timezone.utc)


def _normalize_source_observed_at(value: str | None) -> str | None:
    if value is None:
        return None
    return _parse_source_observed_at(value).isoformat()


def _receipt_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    return {key: value for key, value in dict(metadata or {}).items() if key != "source_observed_at"}


def default_agent_session_documents_dir(config: AppConfig) -> Path:
    """Return the local inbox directory for generated session packages."""
    return Path(config.storage.docs_path).parent / "agent-session-submissions"


def redact_agent_session_markdown(markdown: str) -> str:
    """Redact obvious secrets before storing generated agent session content."""
    redacted = markdown
    for kind, pattern in _SECRET_PATTERNS:
        if kind == "bearer":
            redacted = pattern.sub(r"\1[REDACTED]", redacted)
        elif kind == "json_assignment":
            redacted = pattern.sub(r"\1[REDACTED]", redacted)
        elif kind == "openai_key":
            redacted = pattern.sub("[REDACTED]", redacted)
        else:
            redacted = pattern.sub(lambda m: f"{m.group(1)}: [REDACTED]", redacted)
    return redacted


def agent_session_window_hash(payload: dict[str, Any]) -> str:
    """Return a stable hash for the redacted content of an uploaded window."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def redact_agent_session_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Redact obvious secrets from uploaded structured transcript events."""
    return [redact_agent_session_payload(event) for event in events]


def redact_agent_session_payload(value: Any) -> Any:
    """Redact obvious secrets from nested uploaded agent-session window data."""
    if isinstance(value, str):
        return redact_agent_session_markdown(value)
    if isinstance(value, list):
        return [redact_agent_session_payload(item) for item in value]
    if isinstance(value, dict):
        return {key: redact_agent_session_payload(item) for key, item in value.items()}
    return value


def canonicalize_agent_session_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the service-owned evidence stream used for package generation."""
    canonical_events: list[dict[str, Any]] = []
    for event in events:
        canonical = _canonicalize_agent_session_event(event)
        if canonical is not None:
            canonical_events.append(canonical)
    return canonical_events


def _canonicalize_agent_session_event(event: dict[str, Any]) -> dict[str, Any] | None:
    kind = _string_value(event.get("kind"))
    native_type = (
        _string_value(event.get("native_type"))
        or _string_value(event.get("type"))
        or _string_value(event.get("source_type"))
    )
    role = _string_value(event.get("actor")) or _string_value(event.get("role"))

    if (kind and kind in _NOISE_EVENT_TYPES) or (native_type and native_type in _NOISE_EVENT_TYPES):
        return None
    if kind not in _CANONICAL_EVENT_KINDS:
        kind, role = _infer_agent_session_event_kind(kind, native_type, role)
    if kind is None:
        return None

    text = _first_text_value(event, "text", "summary", "content", "preview", "message", "output", "input")
    name = _string_value(event.get("name") or event.get("tool_name"))
    if not text and not name:
        return None

    canonical: dict[str, Any] = {
        "kind": kind,
        "actor": role or _default_actor_for_kind(kind),
    }
    if name:
        canonical["name"] = name
    if text:
        canonical["text"] = _truncate_middle(text, _MAX_CANONICAL_EVENT_TEXT_CHARS)
    timestamp = _string_value(event.get("timestamp"))
    if timestamp:
        canonical["timestamp"] = timestamp
    if native_type:
        canonical["native_type"] = native_type
    source_type = _string_value(event.get("source_type"))
    if source_type:
        canonical["source_type"] = source_type
    truncation = event.get("truncation")
    if isinstance(truncation, dict):
        canonical["truncation"] = truncation
    return canonical


def _infer_agent_session_event_kind(
    kind: str | None,
    native_type: str | None,
    role: str | None,
) -> tuple[str | None, str | None]:
    candidate = native_type or kind or role
    if candidate in _TOOL_CALL_TYPES:
        return "tool_call", role or "assistant"
    if candidate in _TOOL_RESULT_TYPES or role == "tool":
        return "tool_result", role or "tool"
    if candidate == "agent_message":
        return "assistant_message", role or "assistant"
    if role == "user" or candidate == "user":
        return "user_message", "user"
    if role == "assistant" or candidate == "assistant":
        return "assistant_message", "assistant"
    return None, role


def _default_actor_for_kind(kind: str) -> str:
    if kind == "user_message":
        return "user"
    if kind in {"tool_result", "command_result"}:
        return "tool"
    return "assistant"


def _string_value(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _first_text_value(event: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = event.get(key)
        if value in (None, "", [], {}):
            continue
        if isinstance(value, str):
            return value
        if isinstance(value, (dict, list)):
            return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return str(value)
    return None


def _truncate_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    marker = "...truncated..."
    left = (max_chars - len(marker)) // 2
    right = max_chars - len(marker) - left
    return text[:left] + marker + text[-right:]


def build_agent_session_doc_id(
    *,
    client: str,
    session_id: str,
    trigger: str,
    workspace: str,
    history_window_kind: str,
    history_window_start: str | None,
    history_window_end: str | None,
    window_hash: str | None = None,
) -> str:
    """Build a stable document id for one client history window.

    Identity combines the event range with the content hash: an event range
    fixes where the window sits, and ``window_hash`` makes documents
    content-distinct so a window that reuses a range with different content gets
    a new id instead of overwriting the earlier one.
    """
    identity = "|".join(
        [
            client.strip(),
            session_id.strip(),
            trigger.strip(),
            workspace.strip(),
            history_window_kind.strip(),
            history_window_start or "",
            history_window_end or "",
            window_hash or "",
        ]
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return "-".join(
        [
            "agent-session",
            slugify(client)[:30],
            slugify(session_id)[:50],
            slugify(trigger)[:30],
            digest,
        ]
    )


def build_agent_hook_receipt_id(
    *,
    client: str,
    session_id: str,
    hook: str,
    workspace: str,
    repo: str | None,
    branch: str | None,
    commit_sha: str | None,
) -> str:
    """Build a stable receipt id for one client lifecycle hook event."""
    identity = "|".join(
        [
            client.strip(),
            session_id.strip(),
            hook.strip(),
            workspace.strip(),
            repo or "",
            branch or "",
            commit_sha or "",
        ]
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return "-".join(
        [
            "agent-hook",
            slugify(client)[:30],
            slugify(session_id)[:50],
            slugify(hook)[:30],
            digest,
        ]
    )


async def submit_agent_hook_receipt(
    *,
    db: Database,
    client: str,
    session_id: str,
    hook: str,
    workspace: str,
    repo: str | None = None,
    branch: str | None = None,
    commit_sha: str | None = None,
    metadata: dict[str, Any] | None = None,
    submitted_at: str | None = None,
) -> dict:
    """Store lineage for a lifecycle hook without creating source material."""
    if not client.strip():
        raise ValueError("client is required")
    if not session_id.strip():
        raise ValueError("session_id is required")
    if not hook.strip():
        raise ValueError("hook is required")
    if not workspace.strip():
        raise ValueError("workspace is required")

    submitted_at = submitted_at or _now_iso()
    receipt = AgentHookReceipt(
        receipt_id=build_agent_hook_receipt_id(
            client=client,
            session_id=session_id,
            hook=hook,
            workspace=workspace,
            repo=repo,
            branch=branch,
            commit_sha=commit_sha,
        ),
        client=client,
        session_id=session_id,
        hook=hook,
        workspace=workspace,
        repo=repo,
        branch=branch,
        commit_sha=commit_sha,
        submitted_at=submitted_at,
        metadata=metadata or {},
        updated_at=submitted_at,
    )
    await db.upsert_agent_hook_receipt(receipt)
    return {
        "receipt_id": receipt.receipt_id,
        "receipt": receipt.__dict__,
    }


async def ensure_agent_session_source(
    db: Database,
    config: AppConfig,
    *,
    client: str,
    documents_dir: str | None = None,
) -> dict:
    """Ensure the per-client agent-session source exists and return it."""
    source_id = agent_session_source_id(client)
    source_name = agent_session_source_name(client)
    inbox = Path(documents_dir) if documents_dir else default_agent_session_documents_dir(config)
    inbox.mkdir(parents=True, exist_ok=True)
    # The "client" key tells AgentSessionGene which packages in the shared
    # documents_dir belong to this source. Without it the gene would rglob the
    # entire inbox and stamp foreign clients' documents with this source id.
    source_config = {"documents_dir": str(inbox), "client": client}
    # Preserve any admin-attached project_binding across the idempotent
    # upsert so a binding configured through the admin API is not silently
    # cleared the next time a session document arrives.
    existing = await db.get_source(source_id)
    existing_binding = existing.get("project_binding") if existing else None
    await db.upsert_source(
        id=source_id,
        type=AGENT_SESSION_SOURCE_TYPE,
        name=source_name,
        config_json=json.dumps(source_config),
        project_binding=existing_binding,
    )
    source = await db.get_source(source_id)
    assert source is not None
    return source


async def submit_agent_session_document(
    *,
    db: Database,
    config: AppConfig,
    client: str,
    session_id: str,
    trigger: str,
    document_markdown: str,
    workspace: str,
    repo: str | None = None,
    branch: str | None = None,
    commit_sha: str | None = None,
    history_window_kind: str = "session",
    history_window_start: str | None = None,
    history_window_end: str | None = None,
    title: str | None = None,
    metadata: dict[str, Any] | None = None,
    source_kind: str = AGENT_SESSION_SOURCE_KIND,
    window_hash: str | None = None,
    submitted_at: str | None = None,
    source_observed_at: str | None = None,
    user_id: str | None = None,
) -> dict:
    """Store a generated session document package and receipt lineage."""
    if not client.strip():
        raise ValueError("client is required")
    if not session_id.strip():
        raise ValueError("session_id is required")
    if not trigger.strip():
        raise ValueError("trigger is required")
    if not workspace.strip():
        raise ValueError("workspace is required")
    if not document_markdown.strip():
        raise ValueError("document_markdown is required")

    source = await ensure_agent_session_source(db, config, client=client)
    documents_dir = Path(source["config"]["documents_dir"])

    submitted_at = submitted_at or _now_iso()
    normalized_source_observed_at = _normalize_source_observed_at(source_observed_at)
    redacted_markdown = redact_agent_session_markdown(document_markdown)
    document_hash = content_hash(redacted_markdown)
    doc_id = build_agent_session_doc_id(
        client=client,
        session_id=session_id,
        trigger=trigger,
        workspace=workspace,
        history_window_kind=history_window_kind,
        history_window_start=history_window_start,
        history_window_end=history_window_end,
        window_hash=window_hash,
    )
    per_client_source_id = agent_session_source_id(client)
    source_url = f"agent-session://{slugify(client)}/{slugify(session_id)}/{slugify(trigger)}/{doc_id}"
    doc_title = title or f"Agent Session: {client} {session_id} {trigger}"
    project = resolve_project_key(
        source.get("project_binding"),
        item_field_value=None,
        repo=repo,
        workspace=workspace,
    )
    package_path = documents_dir / slugify(project) / f"{doc_id}.json"
    package_path.parent.mkdir(parents=True, exist_ok=True)

    receipt_metadata = _receipt_metadata(metadata)
    if user_id is not None:
        receipt_metadata["user_id"] = user_id
    receipt_metadata.setdefault("repo_identifier", normalize_repo_identifier(repo))

    receipt = AgentSessionReceipt(
        doc_id=doc_id,
        source_id=per_client_source_id,
        client=client,
        session_id=session_id,
        trigger=trigger,
        workspace=workspace,
        repo=repo,
        branch=branch,
        commit_sha=commit_sha,
        history_window_kind=history_window_kind,
        history_window_start=history_window_start,
        history_window_end=history_window_end,
        submitted_at=submitted_at,
        document_hash=document_hash,
        source_kind=source_kind,
        document_uri=str(package_path),
        metadata=receipt_metadata,
        updated_at=submitted_at,
    )
    package = {
        "package_kind": AGENT_SESSION_PACKAGE_KIND,
        "content_role": AGENT_SESSION_CONTENT_ROLE,
        "doc_id": doc_id,
        "title": doc_title,
        "source_url": source_url,
        "last_modified": submitted_at,
        "space_or_project": project,
        "version": document_hash,
        "markdown": redacted_markdown,
        "receipt": receipt.__dict__,
    }
    if normalized_source_observed_at is not None:
        package["source_observed_at"] = normalized_source_observed_at
    # Write the package atomically: serialize to a sibling temp file on the same
    # filesystem, then rename it into place. A reader (or a concurrent same-id
    # write) sees either the previous package or the complete new one, never a
    # half-written file.
    payload_text = json.dumps(package, indent=2, sort_keys=True)
    package_existed = package_path.exists()
    package_written = False
    tmp_fd, tmp_name = tempfile.mkstemp(dir=str(package_path.parent), suffix=".json.tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
            handle.write(payload_text)
        os.replace(tmp_name, package_path)
        package_written = True
        await db.upsert_agent_session_receipt(receipt)
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

    return {
        "doc_id": doc_id,
        "source_id": per_client_source_id,
        "source_type": AGENT_SESSION_SOURCE_TYPE,
        "document_uri": str(package_path),
        "document_hash": document_hash,
        "receipt": receipt.__dict__,
    }


async def _record_window_outcome(
    *,
    db: Database,
    client: str,
    session_id: str,
    trigger: str,
    workspace: str,
    repo: str | None,
    branch: str | None,
    commit_sha: str | None,
    history_window_kind: str,
    history_window_start: str | None,
    history_window_end: str | None,
    submitted_at: str,
    source_observed_at: str | None,
    window_hash: str,
    receipt: dict[str, Any] | None,
    outcome: str,
    reason: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Persist lineage for a window that produced no stored document.

    Every uploaded window records its fate (no_output or failed), so a window
    that was processed but kept nothing is never indistinguishable from one that
    was lost or never sent. The receipt carries no package file, only the outcome
    and reason, keyed by the same window identity a stored package would use.
    """
    doc_id = build_agent_session_doc_id(
        client=client,
        session_id=session_id,
        trigger=trigger,
        workspace=workspace,
        history_window_kind=history_window_kind,
        history_window_start=history_window_start,
        history_window_end=history_window_end,
        window_hash=window_hash,
    )
    stored_metadata = {
        "outcome": outcome,
        "reason": reason,
        "window_hash": f"sha256:{window_hash}",
        "window_retention": "none",
        "receipt": receipt or {},
    }
    stored_metadata.update(_receipt_metadata(metadata))
    receipt_record = AgentSessionReceipt(
        doc_id=doc_id,
        source_id=agent_session_source_id(client),
        client=client,
        session_id=session_id,
        trigger=trigger,
        workspace=workspace,
        repo=repo,
        branch=branch,
        commit_sha=commit_sha,
        history_window_kind=history_window_kind,
        history_window_start=history_window_start,
        history_window_end=history_window_end,
        submitted_at=submitted_at,
        document_hash=f"sha256:{window_hash}",
        source_kind=AGENT_SESSION_WINDOW_SOURCE_KIND,
        document_uri="",
        metadata=stored_metadata,
        updated_at=submitted_at,
    )
    await db.upsert_agent_session_receipt(receipt_record)
    return doc_id


async def _existing_window_result(
    *,
    db: Database,
    client: str,
    session_id: str,
    trigger: str,
    workspace: str,
    history_window_kind: str,
    history_window_start: str | None,
    history_window_end: str | None,
    window_hash: str,
) -> dict[str, Any] | None:
    doc_id = build_agent_session_doc_id(
        client=client,
        session_id=session_id,
        trigger=trigger,
        workspace=workspace,
        history_window_kind=history_window_kind,
        history_window_start=history_window_start,
        history_window_end=history_window_end,
        window_hash=window_hash,
    )
    receipt = await db.get_agent_session_receipt(doc_id)
    if not receipt:
        return None
    metadata = receipt.get("metadata") or {}
    outcome = metadata.get("outcome")
    if outcome == "failed":
        return None
    result = {
        "accepted": True,
        "window_hash": f"sha256:{window_hash}",
        "status": "processed",
        "result": outcome,
        "source_id": agent_session_source_id(client),
        "source_type": AGENT_SESSION_SOURCE_TYPE,
        "process_now": False,
        "idempotent": True,
    }
    for key in ("patch_outcome", "concept_id", "claim_id", "memory_id", "reason"):
        if key in metadata:
            result[key] = metadata[key]
    return result


async def submit_agent_session_window(
    *,
    db: Database,
    config: AppConfig,
    memory_store,
    structured_llm_client: Any,
    client: str,
    session_id: str,
    trigger: str,
    workspace: str,
    events: list[dict[str, Any]],
    history_window: dict[str, Any] | None = None,
    transcript_markdown: str | None = None,
    repo: str | None = None,
    branch: str | None = None,
    commit_sha: str | None = None,
    receipt: dict[str, Any] | None = None,
    retention: str = "none",
    submitted_at: str | None = None,
    source_observed_at: str | None = None,
    process_now: bool = True,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Patch private agent knowledge from an uploaded transcript window."""
    if not client.strip():
        raise ValueError("client is required")
    if not session_id.strip():
        raise ValueError("session_id is required")
    if not trigger.strip():
        raise ValueError("trigger is required")
    if not workspace.strip():
        raise ValueError("workspace is required")
    if retention != "none":
        raise ValueError("retention must be none")

    submitted_at = submitted_at or _now_iso()
    normalized_source_observed_at = _normalize_source_observed_at(source_observed_at)
    history_window = history_window or {"kind": "boundary"}
    redacted_history_window = redact_agent_session_payload(history_window)
    redacted_events = redact_agent_session_events(events)
    redacted_transcript = redact_agent_session_markdown(transcript_markdown or "")
    canonical_events = canonicalize_agent_session_events(redacted_events)
    transcript_fallback = "" if canonical_events else redacted_transcript
    window_content = {
        "events": canonical_events,
        "transcript_markdown": transcript_fallback,
    }
    window_hash = agent_session_window_hash(window_content)
    # Window identity combines the event range with the content hash. The range
    # (possibly absent for transcript windows) fixes where the window sits; the
    # hash, passed to build_agent_session_doc_id, makes documents content-distinct
    # so an identical window is idempotent and a same-range window with different
    # content gets a new id instead of overwriting the earlier one.
    window_kind = str(history_window.get("kind") or "boundary")
    window_start = history_window.get("start_event_id") or history_window.get("start")
    window_end = history_window.get("end_event_id") or history_window.get("end")
    outcome_identity = {
        "client": client,
        "session_id": session_id,
        "trigger": trigger,
        "workspace": workspace,
        "repo": repo,
        "branch": branch,
        "commit_sha": commit_sha,
        "history_window_kind": window_kind,
        "history_window_start": window_start,
        "history_window_end": window_end,
        "submitted_at": submitted_at,
        "source_observed_at": normalized_source_observed_at,
        "window_hash": window_hash,
        "receipt": receipt,
    }
    existing_result = await _existing_window_result(
        db=db,
        client=client,
        session_id=session_id,
        trigger=trigger,
        workspace=workspace,
        history_window_kind=window_kind,
        history_window_start=window_start,
        history_window_end=window_end,
        window_hash=window_hash,
    )
    if existing_result is not None:
        return existing_result

    # An empty window keeps nothing durable, but its outcome is still recorded so
    # completeness auditing can tell "processed, nothing to keep" from "lost".
    if not canonical_events and not transcript_fallback.strip():
        await _record_window_outcome(db=db, **outcome_identity, outcome="no_output", reason="empty window")
        return {
            "accepted": True,
            "window_hash": f"sha256:{window_hash}",
            "status": "processed",
            "result": "no_output",
            "reason": "empty window",
        }

    if structured_llm_client is None:
        await _record_window_outcome(
            db=db,
            **outcome_identity,
            outcome="failed",
            reason="agent session window summarization LLM unavailable",
        )
        raise ValueError("agent session window summarization LLM unavailable")

    repo_identifier = normalize_repo_identifier(repo)
    prompt = await render_agent_knowledge_patch_prompt(
        db=db,
        owner_user_id=user_id or "local",
        client=client,
        session_id=session_id,
        trigger=trigger,
        workspace=workspace,
        repo_identifier=repo_identifier,
        branch=branch,
        history_window=redacted_history_window,
        events=canonical_events,
        transcript_markdown=transcript_fallback,
    )
    try:
        generated = await structured_llm_client.generate_agent_knowledge_patch(
            prompt,
            max_tokens=min(
                config.llm.enrichment_max_tokens,
                AGENT_SESSION_KNOWLEDGE_PATCH_MAX_TOKENS,
            ),
        )
    except Exception as exc:
        await _record_window_outcome(
            db=db,
            **outcome_identity,
            outcome="failed",
            reason=f"{type(exc).__name__}: {exc}"[:500],
        )
        raise

    citation = f"agent-window://{slugify(client)}/{slugify(session_id)}/sha256-{window_hash}"
    try:
        proposal = (
            generated
            if isinstance(generated, AgentKnowledgePatchProposal)
            else AgentKnowledgePatchProposal.model_validate(generated)
        )
    except Exception as exc:
        await _record_window_outcome(
            db=db,
            **outcome_identity,
            outcome="failed",
            reason=f"{type(exc).__name__}: {exc}"[:500],
        )
        raise ValueError(f"agent knowledge patch validation failed: {exc}") from exc
    if citation not in proposal.citations:
        proposal.citations.append(citation)

    if proposal.action == "no_output":
        patch_service = AgentKnowledgeBundleService(db=db, memory_store=memory_store)
        patch = await patch_service.apply_patch_proposal(
            proposal=proposal,
            owner_user_id=user_id or "local",
            source_id=agent_session_source_id(client),
            client=client,
            session_id=session_id,
            workspace=workspace,
            repo_identifier=repo_identifier,
            project_key=None,
            submitted_at=_parse_submitted_at(submitted_at),
            source_observed_at=(
                _parse_source_observed_at(normalized_source_observed_at)
                if normalized_source_observed_at is not None
                else None
            ),
        )
        reason = patch.reason or proposal.reason or "window had no durable memory value"
        await _record_window_outcome(
            db=db,
            **outcome_identity,
            outcome="no_output",
            reason=reason,
            metadata={"patch_outcome": patch.outcome},
        )
        return {
            "accepted": True,
            "window_hash": f"sha256:{window_hash}",
            "status": "processed",
            "result": "no_output",
            "patch_outcome": patch.outcome,
            "reason": reason,
        }

    source = await ensure_agent_session_source(db, config, client=client)
    project = resolve_project_key(
        source.get("project_binding"),
        item_field_value=None,
        repo=repo,
        workspace=workspace,
    )
    patch_service = AgentKnowledgeBundleService(db=db, memory_store=memory_store)

    try:
        patch = await patch_service.apply_patch_proposal(
            proposal=proposal,
            owner_user_id=user_id or "local",
            source_id=agent_session_source_id(client),
            client=client,
            session_id=session_id,
            workspace=workspace,
            repo_identifier=repo_identifier,
            project_key=project,
            submitted_at=_parse_submitted_at(submitted_at),
            source_observed_at=(
                _parse_source_observed_at(normalized_source_observed_at)
                if normalized_source_observed_at is not None
                else None
            ),
        )
    except Exception as exc:
        await _record_window_outcome(
            db=db,
            **outcome_identity,
            outcome="failed",
            reason=f"{type(exc).__name__}: {exc}"[:500],
        )
        raise

    if patch.outcome != "applied":
        reason = patch.reason or proposal.reason or "window had no durable memory value"
        result = patch.result_bucket
        await _record_window_outcome(
            db=db,
            **outcome_identity,
            outcome=result,
            reason=reason,
            metadata={"patch_outcome": patch.outcome},
        )
        return {
            "accepted": True,
            "window_hash": f"sha256:{window_hash}",
            "status": "processed",
            "result": result,
            "patch_outcome": patch.outcome,
            "reason": reason,
        }

    await _record_window_outcome(
        db=db,
        **outcome_identity,
        outcome="knowledge_patched",
        reason=patch.reason or "agent knowledge patch applied",
        metadata={
            "patch_outcome": patch.outcome,
            "concept_id": patch.concept_id,
            "claim_id": patch.claim_id,
            "memory_id": patch.memory_id,
        },
    )
    return {
        "accepted": True,
        "window_hash": f"sha256:{window_hash}",
        "status": "processed",
        "result": "knowledge_patched",
        "patch_outcome": patch.outcome,
        "concept_id": patch.concept_id,
        "claim_id": patch.claim_id,
        "memory_id": patch.memory_id,
        "source_id": agent_session_source_id(client),
        "source_type": AGENT_SESSION_SOURCE_TYPE,
        "process_now": process_now,
    }
