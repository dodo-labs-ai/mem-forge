# AGENTS.md — Project conventions for Codex

## Code Quality

- Code should look like "first-place design", not "bug-fix archaeology." Comments, docstrings, and prompts should read as if the current approach was always the intended design. A new developer reading the code shouldn't see refactoring history.
- Prefer clean, robust ownership boundaries over fallback workarounds. Do not add DB-only fallback paths for lifecycle operations that also require search/vector cleanup; route those operations through the owning service instead.

## Project

- See `README.md` for setup and project orientation.
- See `docs/architecture.md` for the full system design.
- See `docs/design/agent-session-saas-plugin-flow.md` for the Codex and Claude Code adapter flow.
