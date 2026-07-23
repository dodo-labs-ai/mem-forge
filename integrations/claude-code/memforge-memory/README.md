# MemForge Memory for Claude Code

This plugin connects Claude Code lifecycle hooks to a MemForge API.
It also registers a thin local MCP proxy for explicit memory tools.
The packaged runtime and plugin version is `0.1.29`.

With no routing variables, the plugin targets local OSS at
`http://127.0.0.1:8765/api`. Otherwise put the target in the top-level `env`
object in `~/.claude/settings.json`. Lifecycle hooks do not
inherit MCP-server-only environment, so do not put these routing values only in
an MCP server entry. `MEMFORGE_API_URL` must be an HTTP(S) origin without
`/api`. Origins whose hostname is `hana.ondemand.com` or one of its subdomains
are Cloud targets and require `MEMFORGE_WORKSPACE_ID`; every other origin is OSS
and forbids a workspace.

```json
{
  "env": {
    "MEMFORGE_API_URL": "https://memforge-dev.cfapps.eu12.hana.ondemand.com",
    "MEMFORGE_WORKSPACE_ID": "mount_tai"
  }
}
```

For remote OSS, use its origin and omit `MEMFORGE_WORKSPACE_ID`. Set
`MEMFORGE_API_TOKEN` separately in the process or
top-level agent environment when bearer authentication is required; the token
is an identity credential, not a workspace selector. Invalid or partial targets
fail locally before any MCP or hook network request.

Do not add a manual MCP server block for MemForge. The plugin's `.mcp.json`
registers the MCP server; duplicating it in config can pin the agent to a stale
plugin cache path after upgrades.

The bundled MCP proxy does not need a local MemForge CLI or local-DB MCP
process. It forwards search, memory detail, recent-change, and session document
calls through the configured immutable target. MCP and lifecycle hooks read the
same top-level agent routing values. `get_resource(mode="file")` is handled
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
Use MemForge to search for "<topic>". If source evidence matters, call
get_memory on the relevant result before citing source details.
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

The bundled MCP proxy exposes tools such as `search`, `get_memory`, and
`get_resource`.
`get_resource` fetches `content_url` / `pdf_url` artifacts through
`MEMFORGE_API_URL`; in `file` mode it writes the artifact to
`~/.memforge-agent/artifacts`.

Hooks do not write canonical memories directly.
