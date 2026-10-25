"""Seed an existing private agent-session Source, as earlier releases created it."""

from __future__ import annotations

import json

from memforge.agent_sessions import AGENT_SESSION_SOURCE_TYPE, agent_session_source_id
from memforge.storage.database import Database

AGENT_SESSION_SOURCE_NAMES = {"codex": "Codex Session", "claude-code": "Claude Code Session"}


async def seed_agent_session_source(db: Database, *, client: str, owner_user_id: str) -> dict:
    """Store the private agent-session Source of one client and owner and return it."""
    source_id = agent_session_source_id(client, owner_user_id)
    await db.upsert_source(
        id=source_id,
        type=AGENT_SESSION_SOURCE_TYPE,
        name=AGENT_SESSION_SOURCE_NAMES[client],
        config_json=json.dumps({"client": client}),
        access_policy="private",
        owner_user_id=owner_user_id,
        created_by_user_id=owner_user_id,
    )
    source = await db.get_source(source_id)
    assert source is not None
    return source
