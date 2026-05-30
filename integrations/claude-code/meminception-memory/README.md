# MemInception Memory for Claude Code

This plugin connects Claude Code lifecycle hooks to a local MemInception Admin
API.
It also registers the MemInception MCP server for explicit memory tools.

Set `MEMINCEPTION_API_URL` if the API is not running at
`http://127.0.0.1:8765`. The URL can point at a local instance or a hosted
service. Set `MEMINCEPTION_API_TOKEN` when the service requires bearer auth.

The bundled MCP server starts `meminception serve` from `PATH`. For a local
development checkout, set `MEMINCEPTION_MCP_COMMAND` to the desired
`meminception` executable path.

The plugin adds context during `SessionStart` and `UserPromptSubmit`, records
hook lifecycle receipts during `PreCompact`, `Stop`, and `SubagentStop`, and
queues bounded, redacted transcript-window uploads to
`/api/agent-sessions/windows`.

Default capture flow:

```text
hook -> local queue -> window upload with process_now=false
     -> MemInception package generation -> service-owned source sync
```

The hook worker does not call `/api/sources/{source_id}/sync`.
It stores retry state in `~/.meminception-agent/queue.sqlite` unless
`MEMINCEPTION_AGENT_QUEUE_DB` points somewhere else.

The bundled MCP server exposes tools such as `search`, `get_memory`,
`list_recent_changes`, and `submit_agent_session_document` for explicit
already-generated summaries.

Hooks do not write canonical memories directly.
