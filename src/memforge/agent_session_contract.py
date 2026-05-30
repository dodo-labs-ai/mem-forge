"""Shared contract for generated agent-session source packages."""

AGENT_SESSION_PACKAGE_KIND = "agent_session_document"
AGENT_SESSION_CONTENT_ROLE = "generated_summary"

LLM_VISIBLE_METADATA_KEYS = {
    "has_transcript_path",
    "hook_event_name",
    "permission_mode",
    "turn_id",
}
