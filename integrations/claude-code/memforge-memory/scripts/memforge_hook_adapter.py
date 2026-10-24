"""SessionStart hook for the MemForge Codex and Claude Code plugins.

The plugins register one command hook, ``SessionStart`` with the ``context``
mode. It adds static MemForge usage guidance to a new, resumed or compacted
session so the agent knows how to use the MemForge MCP tools. The hook reads
the provider's hook JSON only to name the event; it makes no network call,
reads no MemForge configuration and writes no local state. Agents write memory
only through the confirmed MCP ``create_memory`` tool.

Hook contract: every invocation prints one JSON object with
``"continue": true`` and exits 0, so the hook can never block the coding
client. That holds for an event other than ``SessionStart``, for unreadable
hook input, and for any mode other than ``context``.

Example::

    $ echo '{"hook_event_name": "SessionStart"}' | memforge_hook.py context
    {"continue": true, "hookSpecificOutput": {"hookEventName": "SessionStart",
     "additionalContext": "## MemForge Usage Guidance ..."}}
"""

from __future__ import annotations

import json
import sys
from typing import Any

CONTEXT_MODE = "context"
SESSION_START_EVENT = "SessionStart"
PLUGIN_VERSION = "0.2.0-beta.2"
SESSION_START_USAGE_GUIDANCE = (
    "## MemForge Usage Guidance\n\n"
    "MemForge is long-term memory for prior decisions, conventions, debugging "
    "history, source context, and user preferences. Use it agentically:\n\n"
    "- Before non-trivial repo or workspace work, use the MemForge MCP search "
    "tool with a natural language query. Omit filters when unsure so you do not "
    "accidentally hide relevant memories.\n"
    "- Use source filters only when the user or task gives a clear facet. Do "
    "not send `current_repo_only`; exact repo filtering is not exposed by the "
    "MCP tool. Pass `repository_context` only when the exact current working "
    "directory is known; never guess or use a plugin directory. When the user "
    "names a configured knowledge source, call `list_sources` first and pass "
    "exact `source_ids` from that result; do not guess source ids from source "
    "names or source types.\n"
    "- If a memory affects an answer, review, or code change, call `get_memory` "
    "for full content and provenance before relying on it.\n"
    "- Search results do not include source links or artifact URLs. Use "
    "`search -> get_memory -> get_resource` when the user needs source "
    "evidence, quotes, or exact document context.\n"
    "- Treat memory as context, not current truth. Verify current files, tests, "
    "runtime state, or external systems when facts may have changed.\n"
    "- Proactively detect memory creation, corrections, or retirements (`remember this`, "
    "`not X, actually Y`, `don't use this anymore`). For create, search first "
    "to avoid duplicates. For correction/retirement, locate the memory with "
    "`search`/`get_memory`. Show a readable preview: new claim or old/new claim, "
    "provenance/evidence, scope, and type.\n"
    "- For `create_memory`, confirmed content must be the durable memory only. Keep "
    "confirmation details, test/deploy notes, and why-the-tool-was-called out "
    "of content; put source evidence/details in `provenance`.\n"
    "- Never mutate memory silently. Before `create_memory`, `propose_memory_correction`, "
    "`retire_memory`, `dismiss_memory_relation`, `restore_memory_relation`, or "
    "`resolve_memory_review`, confirm via "
    "`request_user_input` if available; "
    "otherwise ask a concise text question. Do not show raw tool arguments unless needed.\n"
    "- If no relevant memory is found, continue normally and say so only when it "
    "matters to the user.\n"
)


def main(argv: list[str] | None = None) -> int:
    """Run one hook invocation and always let the client continue.

    ``argv`` is the hook command's arguments: the mode first, then options such
    as ``--client codex``. Options are accepted and need no value check,
    because the guidance is the same for every client.
    """
    arguments = sys.argv[1:] if argv is None else argv
    mode = arguments[0] if arguments else None
    try:
        event_name = _event_name(_read_hook_payload()) if mode == CONTEXT_MODE else None
    except Exception:  # Unreadable hook input must not block the session.
        event_name = None
    if event_name == SESSION_START_EVENT:
        _emit_additional_context(event_name, SESSION_START_USAGE_GUIDANCE)
    else:
        print(json.dumps({"continue": True}))
    return 0


def _emit_additional_context(event_name: str, context: str) -> None:
    output: dict[str, Any] = {
        "continue": True,
        "hookSpecificOutput": {
            "hookEventName": event_name,
            "additionalContext": context,
        },
    }
    print(json.dumps(output))


def _read_hook_payload() -> dict[str, Any]:
    raw = sys.stdin.read().strip()
    if not raw:
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("hook input must be a JSON object")
    return data


def _event_name(payload: dict[str, Any]) -> str | None:
    value = payload.get("hook_event_name") or payload.get("hookEventName") or payload.get("event")
    return str(value) if value else None


if __name__ == "__main__":
    raise SystemExit(main())
