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
- Keep Codex and Claude Code on the same remote plugin source. Neither client should reference `/Users/i551096/Dev/mem-inception` as a marketplace or plugin source, because local path installs hide packaging and cache drift.
- Release gates for plugin changes should include unit tests, lint, package/install parity, MCP `initialize` and `tools/list` version/tool checks after restart, SessionStart and Stop/PreCompact hook smoke tests, and at least one harmless read/write MCP smoke in a test workspace when write tools change.
