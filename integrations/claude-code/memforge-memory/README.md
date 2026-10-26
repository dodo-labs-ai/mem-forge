# MemForge Memory for Claude Code

This plugin connects Claude Code to a MemForge API through a thin local MCP proxy
for memory tools, and adds MemForge usage guidance when a session starts.
The packaged runtime and plugin version is `0.2.0-beta.2`.

With no routing variables, the plugin targets local OSS at
`http://127.0.0.1:8765/api/v1`. Otherwise put the target in the top-level `env`
object in `~/.claude/settings.json`. `MEMFORGE_API_URL` must be an HTTP(S)
origin without an API path. OSS and Cloud share the same `/api/v1` contract.

```json
{
  "env": {
    "MEMFORGE_API_URL": "https://memforge.example.com"
  }
}
```

Set `MEMFORGE_API_TOKEN` separately in the process or
top-level agent environment when bearer authentication is required; the token
is an identity credential, not a workspace selector. Call `list_workspaces`
when discovery is useful and pass optional `workspace_id` to any data tool for
a one-request override. It may be omitted when the account has exactly one
accessible workspace or the project has a user-confirmed local binding. Use the
`memforge-setup` skill to bind a Git repository or ordinary directory. Use
`memforge-daemon-setup` when the agent should securely install, configure,
verify, troubleshoot, or remove the managed local collection daemon.
Repository context remains
provenance attribution; it never selects or authorizes a workspace.

Do not add a manual MCP server block for MemForge. The plugin's `.mcp.json`
registers the MCP server; duplicating it in config can pin the agent to a stale
plugin cache path after upgrades.

The bundled MCP proxy does not need a local MemForge CLI or local-DB MCP
process. It forwards search, current recent-Memory listing, memory detail, memory write, and Review
calls through the configured immutable target. `get_resource(mode="file")` is handled
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
/plugin marketplace add dodo-labs-ai/mem-forge
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

The plugin registers one hook, `SessionStart`. It adds static MemForge usage
guidance to a new, resumed or compacted session, and it makes no network call
and writes no local state. Per-prompt memory retrieval is left to the MCP
`search` tool, which fetches query-aware context on demand.

The bundled MCP proxy exposes tools such as `search`, `list_recent_memories`,
`get_memory`, and `get_resource`.
`get_resource` fetches `content_url` / `pdf_url` artifacts through
`MEMFORGE_API_URL`; in `file` mode it writes the artifact to
`~/.memforge-agent/artifacts`.

The plugin captures nothing on its own. To keep something, ask Claude Code to
remember it: the agent shows a preview, and after you confirm it writes one
private Memory with the MCP `create_memory` tool.

## Upgrading from 0.2.0-beta.1 or earlier

Earlier plugin releases uploaded redacted session windows automatically. That
capture is removed, and MemForge refuses those uploads. Update the plugin and
restart Claude Code:

```bash
claude plugin marketplace update memforge
claude plugin update memory@memforge
```

After the update:

- You can delete `queue.sqlite`, `queue.sqlite-wal`, `queue.sqlite-shm` and
  `worker.lock` in `~/.memforge-agent`; nothing reads them. Keep `~/.memforge-agent/artifacts` if you use
  `get_resource(mode="file")`.
- A `hook_workspace_id` entry in `~/.memforge/workspace-bindings.json` is
  ignored. The `memforge-setup` skill reports it and can remove it.
- Memories captured from earlier sessions belong to the Claude Code Session
  Source. A MemForge server on a release after 0.2.0b2 retires that Source
  when it upgrades, with the Memories no other Source supports. On a server
  still on 0.2.0b2, delete the Source in the admin UI to retire it the same way.

## Troubleshooting

**Claude Code shows `MemForge hook skipped: /hooks/receipts returned HTTP 410` or
`MemForge hook skipped: /hooks/receipts returned HTTP 404` at the end of a turn
or at compaction.** An earlier plugin release is still installed, and MemForge
no longer accepts its session uploads: a 0.2.0b2 server answers `410`, and a
later server, which no longer has the upload routes, answers `404`. Update the
plugin with the commands above and restart Claude Code.
