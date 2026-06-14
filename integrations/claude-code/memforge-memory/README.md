# MemForge Memory for Claude Code

This plugin connects Claude Code lifecycle hooks to a MemForge API.
It also registers a thin local MCP proxy for explicit memory tools.

Set `MEMFORGE_API_URL` if the API is not running at
`http://127.0.0.1:8765`. The URL can point at a local instance or a hosted
service. Set `MEMFORGE_API_TOKEN` when the service requires bearer auth.
For hosted multi-workspace deployments, also set `MEMFORGE_WORKSPACE_ID` so
the proxy targets `/api/workspaces/<workspace>/api/...` while the token remains
a user identity credential.

The bundled MCP proxy does not need a local MemForge CLI or local-DB MCP
process. It forwards search, memory detail, recent-change, and session
document calls to `MEMFORGE_API_URL`. `get_resource(mode="file")` is handled
locally so returned `local_path` values point to the agent machine.

```text
Claude Code MCP stdio -> plugin-local proxy -> HTTP(S) MemForge API
get_resource(mode=file) -> ~/.memforge-agent/artifacts -> local_path
```

Install from GitHub (run inside an active Claude Code session):

```text
/plugin marketplace add shno-labs/mem-forge
/plugin install memory@memforge
```

Start a new Claude Code session after install.

To push a local folder as a source, open the MemForge Admin UI, choose
**Add Source -> Local Repository**, and run the printed CLI command.

Try a search:

```text
Use MemForge to search for "<topic>". Show the top memories with source_url,
content_url, and pdf_url when present.
```

Fetch backing evidence:

```text
Search MemForge for "<topic>". If a result has content_url or pdf_url, call
get_resource with mode="file" and show the local_path.
```

The plugin adds context during `SessionStart`, records hook lifecycle receipts
during `PreCompact`, `Stop`, and `SubagentStop`, and queues bounded, redacted
transcript-window uploads to `/api/agent-sessions/windows`. Per-prompt memory
retrieval is left to the MCP `search` tool, which fetches query-aware context
on demand.

Default capture flow:

```text
hook -> local queue -> window upload with process_now=false
     -> MemForge package generation -> service-owned source sync
```

The hook worker does not call `/api/sources/{source_id}/sync`.
It stores retry state in `~/.memforge-agent/queue.sqlite` unless
`MEMFORGE_AGENT_QUEUE_DB` points somewhere else.

The bundled MCP proxy exposes tools such as `search`, `get_memory`,
`get_resource`, `list_recent_changes`, and `submit_agent_session_document`.
`get_resource` fetches `content_url` / `pdf_url` artifacts through
`MEMFORGE_API_URL`; in `file` mode it writes the artifact to
`~/.memforge-agent/artifacts`.

Hooks do not write canonical memories directly.
