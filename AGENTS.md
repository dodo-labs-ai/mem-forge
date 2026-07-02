# AGENTS.md — Project conventions for Codex

## Code Quality

- Code should look like "first-place design", not "bug-fix archaeology." Comments, docstrings, and prompts should read as if the current approach was always the intended design. A new developer reading the code shouldn't see refactoring history.
- Prefer clean, robust ownership boundaries over fallback workarounds. Do not add DB-only fallback paths for lifecycle operations that also require search/vector cleanup; route those operations through the owning service instead.

## Project

- See `README.md` for setup and project orientation.
- See `docs/architecture.md` for the full system design.
- See `docs/design/agent-session-saas-plugin-flow.md` for the Codex and Claude Code adapter flow.

## Plugin Release Validation

- Test MemForge plugin changes through the same remote plugin install/update path that users use. Do not hand-edit files under `~/.codex/plugins/cache`, and do not add a manual MCP server entry as a workaround.
- For pre-release validation, publish or install a dev/RC plugin artifact from the remote GitHub marketplace source (`shno-labs/mem-forge`), such as `0.1.17-rc.1`, then promote the same commit/tag to the final release after validation.
- When cutting an MCP plugin version, commit the version bump and integration copies first, push the branch, then create a remote tag that names the plugin version, for example `memforge-memory-v0.1.21-rc.4`. Point Codex at that tag with `codex plugin marketplace add https://github.com/shno-labs/mem-forge.git --ref <tag>` and install with `codex plugin add memory@memforge`.
- After installing, verify `codex plugin list --json --marketplace memforge --available`, `codex mcp list`, the marketplace checkout commit/tag, and the cache directory under `~/.codex/plugins/cache/memforge/memory/<version>`. Restart Codex and repeat the version and MCP cwd checks; a restart must not fall back to an older cache.
- Keep Codex and Claude Code on the same remote plugin source. Neither client should reference `/Users/i551096/Dev/mem-inception` as a marketplace or plugin source, because local path installs hide packaging and cache drift.
- If Codex Desktop still falls back after a restart, check for workspace plugin sources such as `.agents/plugins/marketplace.json` in saved workspace roots. A local checkout can silently reinstall `memory@memforge` from its own `integrations/codex/memforge-memory` tree even when the configured marketplace points at the remote tag.
- Release gates for plugin changes should include unit tests, lint, package/install parity, MCP `initialize` and `tools/list` version/tool checks after restart, SessionStart and Stop/PreCompact hook smoke tests, and at least one harmless read/write MCP smoke in a test workspace when write tools change.
