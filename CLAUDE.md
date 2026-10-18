# CLAUDE.md — Project conventions for Claude Code

Agent workflow rules shared with other coding agents live in AGENTS.md:

@AGENTS.md

## Code Quality

- Code should look like "first-place design", not "bug-fix archaeology." Comments, docstrings, and prompts should read as if the current approach was always the intended design. A new developer reading the code shouldn't see refactoring history.

## Project

- See `README.md` for setup and project orientation.
- See `docs/architecture.md` for the system overview.
- Design records (the full architecture, ADRs, subsystem designs, research and the domain glossary) live in the MemForge Cloud repository under `docs/oss/`. Do not add them to this repository.
