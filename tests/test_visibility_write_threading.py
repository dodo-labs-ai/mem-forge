"""Visibility and owner reach the Memory API response."""

from __future__ import annotations


def test_memory_response_exposes_visibility_not_scope():
    from memforge.models import Memory, Visibility, content_hash
    from memforge.server.admin_api import _memory_to_response

    memory = Memory(
        id="mem-api",
        memory_type="fact",
        content="x",
        content_hash=content_hash("x"),
    )
    body = _memory_to_response(memory).model_dump()
    assert body["visibility"] == Visibility.WORKSPACE.value
    assert body["owner_user_id"] is None
    assert "scope" not in body
