"""Shared contract for generated agent-session source packages."""

from datetime import datetime, timezone

from memforge.models import AgentSessionReceipt

AGENT_SESSION_PACKAGE_KIND = "agent_session_document"
AGENT_SESSION_CONTENT_ROLE = "generated_summary"
AGENT_SESSION_WINDOW_SOURCE_KIND = "generated_agent_window_summary"

LLM_VISIBLE_METADATA_KEYS = {
    "has_transcript_path",
    "hook_event_name",
    "permission_mode",
    "turn_id",
}

_SUCCESSFUL_WINDOW_OUTCOMES = frozenset(
    {
        "knowledge_patched",
        "package_created",  # Legacy name for a successfully kept patch.
        "no_output",
    }
)


def successful_agent_session_activity_at(
    receipt: AgentSessionReceipt,
) -> str | None:
    """Return the canonical activity time for one successful window receipt.

    Failed or non-window receipts do not advance the Configured Source's
    successful-activity watermark.  Normalizing to UTC keeps the monotonic SQL
    comparison stable across ``Z`` and explicit-offset inputs.
    """

    if receipt.source_kind != AGENT_SESSION_WINDOW_SOURCE_KIND:
        return None
    if receipt.metadata.get("outcome") not in _SUCCESSFUL_WINDOW_OUTCOMES:
        return None
    value = receipt.updated_at or receipt.submitted_at
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()
