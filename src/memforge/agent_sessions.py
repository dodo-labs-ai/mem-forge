"""Client-generated agent session document intake."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import re
import tempfile
import uuid
from collections.abc import Awaitable, Callable
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
    render_agent_session_authority_prompt,
    render_agent_knowledge_patch_prompt,
)
from memforge.memory.project_resolver import resolve_project_key
from memforge.llm.structured import AgentSessionAuthorityResponse
from memforge.models import AgentHookReceipt, AgentSessionReceipt, content_hash, slugify
from memforge.storage.database import Database
from memforge.source_activity import SourceActivityConflict, SourceActivityKind

AGENT_SESSION_SOURCE_TYPE = "agent_session"
AGENT_SESSION_SOURCE_KIND = "generated_agent_summary"
AGENT_SESSION_WINDOW_SOURCE_KIND = "generated_agent_window_summary"
AGENT_SESSION_KNOWLEDGE_PATCH_MAX_TOKENS = 8192


async def _run_agent_patch_with_activity(
    *,
    db: Database,
    source_id: str,
    expected_epoch: int | None,
    operation: Callable[[], Awaitable[Any]],
    lease_seconds: int = 300,
    heartbeat_seconds: float = 60.0,
) -> Any:
    """Run one Agent patch only while its durable Source lease is current."""

    activity_id = f"agent-patch-{uuid.uuid4().hex}"
    await db.acquire_source_activity(
        activity_id=activity_id,
        source_id=source_id,
        kind=SourceActivityKind.AGENT_PATCH,
        expected_epoch=expected_epoch,
        lease_seconds=lease_seconds,
    )

    async def heartbeat() -> None:
        while True:
            await asyncio.sleep(heartbeat_seconds)
            await db.renew_source_activity(
                activity_id=activity_id,
                lease_seconds=lease_seconds,
            )

    patch_task = asyncio.create_task(operation())
    heartbeat_task = asyncio.create_task(heartbeat())
    try:
        done, _ = await asyncio.wait(
            {patch_task, heartbeat_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if heartbeat_task in done:
            try:
                await heartbeat_task
            except Exception as exc:
                patch_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await patch_task
                raise SourceActivityConflict(
                    f"agent patch activity heartbeat stopped: {activity_id}"
                ) from exc
            raise SourceActivityConflict(
                f"agent patch activity heartbeat stopped: {activity_id}"
            )
        return await patch_task
    finally:
        heartbeat_task.cancel()
        if not heartbeat_task.done():
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat_task
        if not patch_task.done():
            patch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await patch_task
        await db.release_source_activity(activity_id=activity_id)

_KNOWN_CLIENT_SOURCE_NAMES: dict[str, str] = {
    "codex": "Codex Session",
    "claude-code": "Claude Code Session",
}


def agent_session_source_id(client: str, owner_user_id: str) -> str:
    """Return the opaque, stable source id for one client and owner."""
    normalized_client = client.strip()
    normalized_owner = owner_user_id.strip()
    if not normalized_client:
        raise ValueError("client is required")
    if not normalized_owner:
        raise ValueError("owner_user_id is required")
    owner_fingerprint = hashlib.sha256(normalized_owner.encode("utf-8")).hexdigest()[:16]
    return f"src-agent-sessions-{slugify(normalized_client)[:32]}-{owner_fingerprint}"


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
AGENT_SESSION_AUTHORITY_CLASSIFIER_MAX_TOKENS = 1024


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_submitted_at(value: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_source_updated_at(value: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("source_updated_at must include an explicit timezone offset")
    return parsed.astimezone(timezone.utc)


def _normalize_source_updated_at(value: str | None) -> str | None:
    if value is None:
        return None
    return _parse_source_updated_at(value).isoformat()


def _receipt_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    return {key: value for key, value in dict(metadata or {}).items() if key != "source_updated_at"}


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
    """Return the service-owned evidence stream used for package generation.

    Evidence IDs are local to one canonicalized submission. They are prompt
    handles, not persistent identifiers for historical rows.
    """
    canonical_events: list[dict[str, Any]] = []
    for event in events:
        canonical = _canonicalize_agent_session_event(event)
        if canonical is not None:
            canonical["evidence_id"] = f"E{len(canonical_events) + 1}"
            canonical["evidence_role"] = _agent_session_evidence_role(canonical)
            canonical["authority_candidate"] = _is_agent_session_authority_candidate(canonical)
            canonical_events.append(canonical)
    return canonical_events


def _agent_session_evidence_role(event: dict[str, Any]) -> str:
    """Return the initial evidence role before semantic authority classification."""
    return "supporting"


def _is_agent_session_authority_candidate(event: dict[str, Any]) -> bool:
    return (
        event.get("kind") == "user_message"
        and event.get("actor") == "user"
        and event.get("actor_is_explicit")
        and bool(event.get("text"))
    )


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
        "actor_is_explicit": bool(role),
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


def _primary_agent_session_evidence_ids(events: list[dict[str, Any]]) -> set[str]:
    return {
        str(event["evidence_id"])
        for event in events
        if event.get("evidence_role") == "primary" and event.get("evidence_id")
    }


def _candidate_agent_session_authority_ids(events: list[dict[str, Any]]) -> set[str]:
    return {
        str(event["evidence_id"])
        for event in events
        if event.get("authority_candidate") and event.get("evidence_id")
    }


def _apply_agent_session_authority_response(
    events: list[dict[str, Any]],
    response: AgentSessionAuthorityResponse,
) -> list[dict[str, Any]]:
    """Return events with semantic authority decisions applied."""
    candidate_ids = _candidate_agent_session_authority_ids(events)
    seen_ids: set[str] = set()
    authoritative_ids: set[str] = set()
    unknown_ids: set[str] = set()
    duplicate_ids: set[str] = set()
    for decision in response.decisions:
        evidence_id = decision.evidence_id.strip()
        if evidence_id in seen_ids:
            duplicate_ids.add(evidence_id)
            continue
        seen_ids.add(evidence_id)
        if evidence_id not in candidate_ids:
            unknown_ids.add(evidence_id)
            continue
        if decision.is_authoritative:
            authoritative_ids.add(evidence_id)
    missing_ids = candidate_ids - seen_ids
    if missing_ids:
        raise ValueError(
            "authority classifier omitted candidate evidence ids: "
            + ", ".join(sorted(missing_ids))
        )
    if unknown_ids:
        raise ValueError(
            "authority classifier returned non-candidate evidence ids: "
            + ", ".join(sorted(unknown_ids))
        )
    if duplicate_ids:
        raise ValueError(
            "authority classifier returned duplicate evidence ids: "
            + ", ".join(sorted(duplicate_ids))
        )
    classified_events = []
    for event in events:
        classified = dict(event)
        classified["evidence_role"] = (
            "primary" if classified.get("evidence_id") in authoritative_ids else "supporting"
        )
        classified_events.append(classified)
    return classified_events


async def _classify_agent_session_authority(
    *,
    structured_llm_client: Any,
    owner_user_id: str,
    client: str,
    session_id: str,
    trigger: str,
    workspace: str,
    repo_identifier: str | None,
    branch: str | None,
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidate_ids = _candidate_agent_session_authority_ids(events)
    if not candidate_ids:
        return events
    prompt = render_agent_session_authority_prompt(
        owner_user_id=owner_user_id,
        client=client,
        session_id=session_id,
        trigger=trigger,
        workspace=workspace,
        repo_identifier=repo_identifier,
        branch=branch,
        events=events,
    )
    generated = await structured_llm_client.classify_agent_session_evidence_authority(
        prompt,
        max_tokens=AGENT_SESSION_AUTHORITY_CLASSIFIER_MAX_TOKENS,
    )
    response = (
        generated
        if isinstance(generated, AgentSessionAuthorityResponse)
        else AgentSessionAuthorityResponse.model_validate(generated)
    )
    return _apply_agent_session_authority_response(events, response)


def _agent_patch_primary_evidence_error(
    proposal: AgentKnowledgePatchProposal,
    events: list[dict[str, Any]],
) -> str | None:
    if proposal.action == "no_output":
        return None
    primary_ids = _primary_agent_session_evidence_ids(events)
    cited_primary_ids = {evidence_id.strip() for evidence_id in proposal.primary_evidence_ids if evidence_id.strip()}
    if not cited_primary_ids:
        return "Agent-session patch has no primary evidence authorization."
    if not cited_primary_ids <= primary_ids:
        return "Agent-session patch cites non-primary evidence as authorization."
    return None


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
    owner_user_id: str,
    client: str,
    session_id: str,
    trigger: str,
    workspace: str,
    history_window_kind: str,
    history_window_start: object | None,
    history_window_end: object | None,
    window_hash: str | None = None,
) -> str:
    """Build a stable document id for one client history window.

    Identity combines the event range with the content hash: an event range
    fixes where the window sits, and ``window_hash`` makes documents
    content-distinct so a window that reuses a range with different content gets
    a new id instead of overwriting the earlier one.
    """
    normalized_owner = owner_user_id.strip()
    if not normalized_owner:
        raise ValueError("owner_user_id is required")
    identity = "|".join(
        [
            normalized_owner,
            client.strip(),
            session_id.strip(),
            trigger.strip(),
            workspace.strip(),
            history_window_kind.strip(),
            "" if history_window_start is None else str(history_window_start),
            "" if history_window_end is None else str(history_window_end),
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
    owner_user_id: str,
    documents_dir: str | None = None,
) -> dict:
    """Ensure the private source for one coding client and user exists."""
    source_id = agent_session_source_id(client, owner_user_id)
    source_name = agent_session_source_name(client)
    inbox_root = Path(documents_dir) if documents_dir else default_agent_session_documents_dir(config)
    inbox = inbox_root / source_id
    inbox.mkdir(parents=True, exist_ok=True)
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
        access_policy="private",
        owner_user_id=owner_user_id,
        project_binding=existing_binding,
        created_by_user_id=owner_user_id,
    )
    if existing is None:
        await db.enable_lifecycle_gate(source_id)
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
    source_updated_at: str | None = None,
    user_id: str,
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

    source = await ensure_agent_session_source(
        db,
        config,
        client=client,
        owner_user_id=user_id,
    )
    documents_dir = Path(source["config"]["documents_dir"])

    submitted_at = submitted_at or _now_iso()
    normalized_source_updated_at = _normalize_source_updated_at(source_updated_at)
    redacted_markdown = redact_agent_session_markdown(document_markdown)
    document_hash = content_hash(redacted_markdown)
    doc_id = build_agent_session_doc_id(
        owner_user_id=user_id,
        client=client,
        session_id=session_id,
        trigger=trigger,
        workspace=workspace,
        history_window_kind=history_window_kind,
        history_window_start=history_window_start,
        history_window_end=history_window_end,
        window_hash=window_hash,
    )
    per_client_source_id = agent_session_source_id(client, user_id)
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
    if normalized_source_updated_at is not None:
        package["source_updated_at"] = normalized_source_updated_at
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
    owner_user_id: str,
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
    source_updated_at: str | None,
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
        owner_user_id=owner_user_id,
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
        "user_id": owner_user_id,
    }
    stored_metadata.update(_receipt_metadata(metadata))
    receipt_record = AgentSessionReceipt(
        doc_id=doc_id,
        source_id=agent_session_source_id(client, owner_user_id),
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
    owner_user_id: str,
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
        owner_user_id=owner_user_id,
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
        "source_id": agent_session_source_id(client, owner_user_id),
        "source_type": AGENT_SESSION_SOURCE_TYPE,
        "process_now": False,
        "idempotent": True,
    }
    for key in (
        "patch_outcome",
        "concept_id",
        "claim_id",
        "memory_id",
        "covered_concept_id",
        "covered_claim_id",
        "reason",
    ):
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
    source_updated_at: str | None = None,
    process_now: bool = True,
    user_id: str,
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
    owner_user_id = user_id.strip()
    if not owner_user_id:
        raise ValueError("user_id is required")
    if retention != "none":
        raise ValueError("retention must be none")

    submitted_at = submitted_at or _now_iso()
    normalized_source_updated_at = _normalize_source_updated_at(source_updated_at)
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
        "owner_user_id": owner_user_id,
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
        "source_updated_at": normalized_source_updated_at,
        "window_hash": window_hash,
        "receipt": receipt,
    }
    existing_result = await _existing_window_result(
        db=db,
        owner_user_id=owner_user_id,
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

    source_id = agent_session_source_id(client, owner_user_id)
    source = await db.get_source(source_id)
    source_activity_epoch = (
        int(source.get("activity_epoch") or 0)
        if source is not None
        else None
    )

    if structured_llm_client is None:
        await _record_window_outcome(
            db=db,
            **outcome_identity,
            outcome="failed",
            reason="agent session window summarization LLM unavailable",
        )
        raise ValueError("agent session window summarization LLM unavailable")

    repo_identifier = normalize_repo_identifier(repo)
    try:
        canonical_events = await _classify_agent_session_authority(
            structured_llm_client=structured_llm_client,
            owner_user_id=owner_user_id,
            client=client,
            session_id=session_id,
            trigger=trigger,
            workspace=workspace,
            repo_identifier=repo_identifier,
            branch=branch,
            events=canonical_events,
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

    async def generate_proposal() -> AgentKnowledgePatchProposal:
        prompt = await render_agent_knowledge_patch_prompt(
            db=db,
            owner_user_id=owner_user_id,
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
        primary_evidence_error = _agent_patch_primary_evidence_error(
            proposal,
            canonical_events,
        )
        if primary_evidence_error:
            return AgentKnowledgePatchProposal(
                action="no_output",
                reason=primary_evidence_error,
                covered_concept_id=proposal.covered_concept_id,
                covered_claim_id=proposal.covered_claim_id,
            )
        return proposal

    async def apply_proposal(
        proposal: AgentKnowledgePatchProposal,
        admitted_source: dict[str, Any] | None,
    ) -> dict[str, Any]:
        patch_service = AgentKnowledgeBundleService(db=db, memory_store=memory_store)
        project_key = (
            resolve_project_key(
                admitted_source.get("project_binding"),
                item_field_value=None,
                repo=repo,
                workspace=workspace,
            )
            if admitted_source is not None
            else None
        )
        patch = await patch_service.apply_patch_proposal(
            proposal=proposal,
            owner_user_id=owner_user_id,
            source_id=source_id,
            client=client,
            session_id=session_id,
            workspace=workspace,
            repo_identifier=repo_identifier,
            project_key=project_key,
            submitted_at=_parse_submitted_at(submitted_at),
            source_updated_at=(
                _parse_source_updated_at(normalized_source_updated_at)
                if normalized_source_updated_at is not None
                else None
            ),
        )

        if proposal.action != "no_output" and patch.outcome == "applied":
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
                "source_id": source_id,
                "source_type": AGENT_SESSION_SOURCE_TYPE,
                "process_now": process_now,
            }

        if proposal.action != "no_output":
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

        reason = patch.reason or proposal.reason or "window had no durable memory value"
        await _record_window_outcome(
            db=db,
            **outcome_identity,
            outcome="no_output",
            reason=reason,
            metadata={
                "patch_outcome": patch.outcome,
                "covered_concept_id": patch.covered_concept_id,
                "covered_claim_id": patch.covered_claim_id,
            },
        )
        return {
            "accepted": True,
            "window_hash": f"sha256:{window_hash}",
            "status": "processed",
            "result": "no_output",
            "patch_outcome": patch.outcome,
            "reason": reason,
            "covered_concept_id": patch.covered_concept_id,
            "covered_claim_id": patch.covered_claim_id,
        }

    async def generate_and_apply(admitted_source: dict[str, Any]) -> dict[str, Any]:
        return await apply_proposal(await generate_proposal(), admitted_source)

    if source is None:
        preliminary = await generate_proposal()
        if preliminary.action == "no_output":
            return await apply_proposal(preliminary, None)
        source = await ensure_agent_session_source(
            db,
            config,
            client=client,
            owner_user_id=owner_user_id,
        )
        source_activity_epoch = int(source.get("activity_epoch") or 0)

    assert source_activity_epoch is not None
    try:
        return await _run_agent_patch_with_activity(
            db=db,
            source_id=source_id,
            expected_epoch=source_activity_epoch,
            operation=lambda: generate_and_apply(source),
        )
    except Exception as exc:
        await _record_window_outcome(
            db=db,
            **outcome_identity,
            outcome="failed",
            reason=f"{type(exc).__name__}: {exc}"[:500],
        )
        raise
