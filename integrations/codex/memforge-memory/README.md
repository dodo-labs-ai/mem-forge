# MemForge Memory for Codex

This plugin connects Codex to a MemForge API through a thin local MCP proxy
for memory tools, and adds MemForge usage guidance when a session starts.
The packaged runtime and plugin version is `0.2.0-beta.2`.

With no routing variables, the plugin targets local OSS at
`http://127.0.0.1:8765/api/v1`. Otherwise set the origin in
`~/.codex/config.toml`. `MEMFORGE_API_URL` must be an HTTP(S) origin without an
API path. Set `MEMFORGE_API_TOKEN` separately when bearer authentication is
required.

```toml
[memforge]
MEMFORGE_API_URL = "https://memforge-dev.cfapps.eu12.hana.ondemand.com"
MEMFORGE_API_TOKEN = "..."
```

The same `/api/v1` surface is used for OSS and Cloud. Call `list_workspaces` to
discover accessible workspaces when useful, then pass optional `workspace_id`
to any data tool for a one-request override. It may be omitted when the account
has exactly one accessible workspace or the project has a user-confirmed local
binding. Use the `memforge-setup` skill to bind a Git repository or ordinary
directory.
Use `memforge-daemon-setup` when the agent should securely install, configure,
verify, troubleshoot, or remove the managed local collection daemon.
Repository context remains provenance attribution; it never selects or
authorizes a workspace.

Do not add a manual `[mcp_servers.memforge]` block. The plugin's `.mcp.json`
registers the MCP server; duplicating it in `config.toml` can pin Codex to a
stale plugin cache path after upgrades.

The bundled MCP proxy does not need a local MemForge CLI or local-DB MCP
process. It forwards search, current recent-Memory listing, memory detail, memory write, and Review
calls through the configured immutable target. `get_resource(mode="file")` is
handled locally so returned `local_path` values point to the agent machine.

```text
Codex MCP stdio -> plugin-local proxy -> HTTP(S) MemForge API
get_resource(mode=file) -> ~/.memforge-agent/artifacts -> local_path
```

The proxy negotiates Codex's request-scoped `codex/sandbox-state-meta`
capability. Codex then attaches the current task environment's `sandboxCwd` to
each MCP tool call, allowing the proxy to derive a normalized Git remote
without relying on a process-global directory.
An explicit `repository_context.working_directory` remains available for other
hosts and takes precedence when supplied. Compatible MCP Roots are attribution
fallbacks. Local paths are discarded before OSS or Cloud calls;
malformed negotiated or explicit context fails before HTTP instead of silently
reading another workspace.

Install from GitHub (no checkout required):

```bash
codex plugin marketplace add dodo-labs-ai/mem-forge
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

The plugin registers one hook, `SessionStart`. It adds static MemForge usage
guidance to a new, resumed or compacted session, and it makes no network call
and writes no local state. Per-prompt memory retrieval is left to the MCP
`search` tool, which fetches query-aware context on demand.

The bundled MCP proxy exposes tools such as `search`, `list_recent_memories`,
`get_memory`, and `get_resource`.
`get_resource` fetches `content_url` / `pdf_url` artifacts through
`MEMFORGE_API_URL`; in `file` mode it writes the artifact to
`~/.memforge-agent/artifacts`.

The plugin captures nothing on its own. To keep something, ask Codex to
remember it: the agent shows a preview, and after you confirm it writes one
private Memory with the MCP `create_memory` tool.

## Upgrading from 0.2.0-beta.1 or earlier

Earlier plugin releases uploaded session transcripts automatically. That
capture is removed, and MemForge refuses those uploads. Update the plugin and
restart Codex:

```bash
codex plugin marketplace upgrade memforge
codex plugin add memory@memforge
```

After the update:

- You can delete `~/.memforge-agent/queue.sqlite` and its lock files; nothing
  reads them. Keep `~/.memforge-agent/artifacts` if you use
  `get_resource(mode="file")`.
- A `hook_workspace_id` entry in `~/.memforge/workspace-bindings.json` is
  ignored. The `memforge-setup` skill reports it and can remove it.
- Memories captured from earlier sessions belong to the Codex Session
  Source. Delete that Source in the admin UI to retire them.

## Troubleshooting

**Codex shows `MemForge hook skipped: /hooks/receipts returned HTTP 410` at
the end of a turn or at compaction.** An earlier plugin release is still
installed, and MemForge no longer accepts its session uploads. Update the
plugin with the commands above and restart Codex.
