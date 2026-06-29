# MemForge Memory for Codex

This plugin connects Codex lifecycle hooks to a MemForge API.
It also registers a thin local MCP proxy for explicit memory tools.

Set MemForge connection values in `~/.codex/config.toml` if the API is not
running at `http://127.0.0.1:8765`. The URL can point at a local instance or a
hosted service. Set `MEMFORGE_API_TOKEN` when the service requires bearer auth.
For hosted multi-workspace deployments, also set `MEMFORGE_WORKSPACE_ID` so the
proxy targets `/api/workspaces/<workspace>/api/...` while the token remains a
user identity credential.

```toml
[memforge]
MEMFORGE_API_URL = "https://memforge.example"
MEMFORGE_API_TOKEN = "..."
MEMFORGE_WORKSPACE_ID = "mount_tai"
```

Do not add a manual `[mcp_servers.memforge]` block. The plugin's `.mcp.json`
registers the MCP server; duplicating it in `config.toml` can pin Codex to a
stale plugin cache path after upgrades.

The bundled MCP proxy does not need a local MemForge CLI or local-DB MCP
process. It forwards search, memory detail, recent-change, and session
document calls to `MEMFORGE_API_URL`. `get_resource(mode="file")` is handled
locally so returned `local_path` values point to the agent machine.

```text
Codex MCP stdio -> plugin-local proxy -> HTTP(S) MemForge API
get_resource(mode=file) -> ~/.memforge-agent/artifacts -> local_path
```

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

The bundled MCP proxy exposes tools such as `search`, `get_memory`, and
`get_resource`.
`get_resource` fetches `content_url` / `pdf_url` artifacts through
`MEMFORGE_API_URL`; in `file` mode it writes the artifact to
`~/.memforge-agent/artifacts`.

Hooks do not write canonical memories directly.
