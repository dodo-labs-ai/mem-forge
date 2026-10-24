"""Identity of the private agent-session Source of one coding client and owner.

An agent-session Source holds the private Agent Knowledge claims kept for one
coding client (for example ``codex`` or ``claude-code``) and one owner.
MemForge no longer creates these Sources: coding agents write memory only
through the confirmed MCP ``create_memory`` tool. Existing ones stay readable,
and their owner retires them through ordinary Source deletion.

Example::

    agent_session_source_id("codex", "alice")
    # -> "src-agent-sessions-codex-2bd806c97f0e00af"
"""

from __future__ import annotations

import hashlib

from memforge.models import slugify

AGENT_SESSION_SOURCE_TYPE = "agent_session"


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
