# MemForge Memory for Codex

This plugin connects Codex lifecycle hooks to a MemForge API.
It also registers a thin local MCP proxy for explicit memory tools.
The packaged runtime and plugin version is `0.1.37`.

With no routing variables, the plugin targets local OSS at
`http://127.0.0.1:8765/api`. Otherwise set the target in `~/.codex/config.toml`.
`MEMFORGE_API_URL` must be an HTTP(S) origin without `/api`. Origins whose
hostname is `hana.ondemand.com` or one of its subdomains are Cloud targets and
require `MEMFORGE_WORKSPACE_ID`; every other origin is OSS and forbids a
workspace. Set `MEMFORGE_API_TOKEN` separately when bearer authentication is
required.

```toml
[memforge]
MEMFORGE_API_URL = "https://memforge-dev.cfapps.eu12.hana.ondemand.com"
MEMFORGE_API_TOKEN = "..."
MEMFORGE_WORKSPACE_ID = "mount_tai"
```

For remote OSS, use its origin and omit `MEMFORGE_WORKSPACE_ID`. Invalid or
partial targets fail locally before any MCP or hook network request.

The user-level setting is the global default. To select a different Cloud
workspace for one Git repository, create the uncommitted
`<repository>/.memforge/config.toml`:

```toml
[memforge]
workspace_id = "repository_workspace"
```

The repository file is ignored by Git and only selects the workspace; API
origins and credentials retain their process-then-user precedence. For workspace
selection, a valid repository override has highest priority, followed by a
process `MEMFORGE_WORKSPACE_ID` and then the user-level default. Missing or
invalid repository configuration leaves the process or user default unchanged.
The bundled MCP uses the host repository root and the hooks use the hook
workspace, so both resolve the same override.

Do not add a manual `[mcp_servers.memforge]` block. The plugin's `.mcp.json`
registers the MCP server; duplicating it in `config.toml` can pin Codex to a
stale plugin cache path after upgrades.

The bundled MCP proxy does not need a local MemForge CLI or local-DB MCP
process. It forwards search, current recent-Memory listing, memory detail, and session document
calls through the configured immutable target. MCP and lifecycle hooks read the
same agent-level routing values. `get_resource(mode="file")` is handled locally
so returned `local_path` values point to the agent machine.

```text
Codex MCP stdio -> plugin-local proxy -> HTTP(S) MemForge API
get_resource(mode=file) -> ~/.memforge-agent/artifacts -> local_path
```

For `search` and `create_memory`, Codex may pass its exact current working
directory through the optional MCP `repository_context` argument. The local
proxy converts it to a normalized Git remote and discards the path before
calling OSS or Cloud. No workspace environment variable is required; when
context is unavailable, the operation remains valid and unscoped.

Install from GitHub (no checkout required):

```bash
codex plugin marketplace add shno-labs/mem-forge
codex plugin add memory@memforge
```

Start a new Codex session after install.

To push a local folder as a source, open the MemForge Admin UI, choose
**Add Source -> Local Repository**, and run the printed CLI command.

```bash
# optional
codex mcp get memforge --json
```

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
during `PreCompact` and `Stop`, and queues bounded, redacted transcript-window
uploads to `/api/agent-sessions/windows`. Per-prompt memory retrieval is left to
the MCP `search` tool, which fetches query-aware context on demand.

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

For guided workspace setup, ask Codex to use **MemForge Setup** to configure a
global default, manage the current repository override, or inspect the
effective selection. The skill previews and validates changes before asking for
confirmation.

Hooks do not write canonical memories directly.
