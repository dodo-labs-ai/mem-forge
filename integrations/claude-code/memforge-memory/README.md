# MemForge Memory for Claude Code

This plugin connects Claude Code lifecycle hooks to a MemForge API.
It also registers a thin local MCP proxy for explicit memory tools.
The packaged runtime and plugin version is `0.1.45`.

With no routing variables, the plugin targets local OSS at
`http://127.0.0.1:8765/api/v1`. Otherwise put the target in the top-level `env`
object in `~/.claude/settings.json`. Lifecycle hooks do not
inherit MCP-server-only environment, so do not put these routing values only in
an MCP server entry. `MEMFORGE_API_URL` must be an HTTP(S) origin without
an API path. OSS and Cloud share the same `/api/v1` contract.

```json
{
  "env": {
    "MEMFORGE_API_URL": "https://memforge-dev.cfapps.eu12.hana.ondemand.com"
  }
}
```

Set `MEMFORGE_API_TOKEN` separately in the process or
top-level agent environment when bearer authentication is required; the token
is an identity credential, not a workspace selector. Call `list_workspaces`
when discovery is useful and pass optional `workspace_id` to any data tool for
a one-request override. It may be omitted when the account has a default or
exactly one accessible workspace. After the user explicitly confirms a choice,
`set_default_workspace` persists the server-side default used by automatic
hooks and later calls that omit `workspace_id`. Repository context remains
provenance attribution; it never selects a workspace.

Do not add a manual MCP server block for MemForge. The plugin's `.mcp.json`
registers the MCP server; duplicating it in config can pin the agent to a stale
plugin cache path after upgrades.

The bundled MCP proxy does not need a local MemForge CLI or local-DB MCP
process. It forwards search, current recent-Memory listing, memory detail, and session document
calls through the configured immutable target. MCP and lifecycle hooks read the
same top-level agent routing values. `get_resource(mode="file")` is handled
locally so returned `local_path` values point to the agent machine.

```text
Claude Code MCP stdio -> plugin-local proxy -> HTTP(S) MemForge API
get_resource(mode=file) -> ~/.memforge-agent/artifacts -> local_path
```

Claude Code may pass its exact current working directory through the optional
MCP `repository_context` argument on every MemForge tool. The local proxy uses
it to derive a normalized Git remote, then
discards the path before calling OSS or Cloud. Negotiated request-scoped host
context, when supported by a client, sits behind an explicit argument and ahead
of compatible MCP Roots. Malformed
negotiated or explicit context fails before HTTP instead of silently reading
another workspace.

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
Use MemForge to search for "<topic>". If source evidence matters, call
get_memory on the relevant result before citing source details.
```

List current Memories observed from active sources during a resolved time
window without inventing a topic query:

```text
Use MemForge list_recent_memories for 2026-07-21T00:00:00+08:00 through
2026-07-28T00:00:00+08:00. This is a current-Memory view, not a source
changelog. Follow next_cursor with the same filters until has_more is false.
```

Fetch backing evidence:

```text
Search MemForge for "<topic>". Call get_memory for the relevant memory, then
call get_resource with mode="file" on the best content_url or pdf_url and show
the local_path.
```

The plugin adds context during `SessionStart`, records hook lifecycle receipts
during `PreCompact`, `Stop`, and `SubagentStop`, and queues bounded, redacted
transcript-window uploads to `/api/agent-sessions/windows`. Per-prompt memory
retrieval is left to the MCP `search` tool, which fetches query-aware context
on demand.

Default capture flow:

```text
hook -> local queue -> window upload with process_now=false
     -> MemForge service-owned extraction
```

The hook worker does not call `/api/sources/{source_id}/sync`.
It stores retry state in `~/.memforge-agent/queue.sqlite` unless
`MEMFORGE_AGENT_QUEUE_DB` points somewhere else.

The bundled MCP proxy exposes tools such as `search`, `list_recent_memories`,
`get_memory`, and `get_resource`.
`get_resource` fetches `content_url` / `pdf_url` artifacts through
`MEMFORGE_API_URL`; in `file` mode it writes the artifact to
`~/.memforge-agent/artifacts`.

Hooks do not write canonical memories directly.
