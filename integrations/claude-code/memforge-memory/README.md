# MemForge Memory for Claude Code

This plugin connects Claude Code lifecycle hooks to a local MemForge Admin
API.
It also registers the MemForge MCP server for explicit memory tools.

Set `MEMFORGE_API_URL` if the API is not running at
`http://127.0.0.1:8765`. The URL can point at a local instance or a hosted
service. Set `MEMFORGE_API_TOKEN` when the service requires bearer auth.

The bundled MCP server starts `memforge serve` from `PATH`. For a local
development checkout, set `MEMFORGE_MCP_COMMAND` to the desired
`memforge` executable path. `search` and `get_memory` run in that MCP process;
hook calls and `get_resource` artifact reads use `MEMFORGE_API_URL`.

Install from the repository root:

```bash
uv sync
export MEMFORGE_MCP_COMMAND="$PWD/.venv/bin/memforge"
export MEMFORGE_API_URL="http://127.0.0.1:8765"

claude plugin marketplace add ./
claude plugin install memforge-memory@memforge
```

The plugin adds context during `SessionStart` and `UserPromptSubmit`, records
hook lifecycle receipts during `PreCompact`, `Stop`, and `SubagentStop`, and
queues bounded, redacted transcript-window uploads to
`/api/agent-sessions/windows`.

Default capture flow:

```text
hook -> local queue -> window upload with process_now=false
     -> MemForge package generation -> service-owned source sync
```

The hook worker does not call `/api/sources/{source_id}/sync`.
It stores retry state in `~/.memforge-agent/queue.sqlite` unless
`MEMFORGE_AGENT_QUEUE_DB` points somewhere else.

The bundled MCP server exposes tools such as `search`, `get_memory`,
`get_resource`, `list_recent_changes`, and `submit_agent_session_document`.
`get_resource` fetches `content_url` / `pdf_url` artifacts through
`MEMFORGE_API_URL`, so it works when MemForge storage lives in Docker or a
hosted service.

Hooks do not write canonical memories directly.
