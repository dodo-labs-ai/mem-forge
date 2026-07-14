# Contributing

Thanks for helping improve MemForge. The project is young, so the best
contributions are small, well-tested, and easy to review.

## Development Setup

```bash
git clone https://github.com/shno-labs/mem-forge.git
cd mem-forge
uv sync --extra dev
cd admin-ui && npm ci && cd ..
```

Copy `.env.example` to `.env` for local overrides. Never commit `.env`, local
databases, transcript exports, or source-system credentials.

## Before You Open A PR

Run the checks that match the files you changed:

```bash
make lint
make test
make ui-lint
make ui-test
make ui-build
```

These Makefile targets are maintainer shortcuts for the underlying `uv` and
`npm` commands. The public quickstart uses Docker Compose instead.

For agent-session changes, also run the focused Python tests:

```bash
uv run pytest tests/test_hook_adapter.py tests/test_agent_session_api.py tests/test_agent_session_gene.py -q
```

### MCP proxy integration copies

Edit only `src/memforge/plugin_mcp_proxy.py`. The Codex and Claude Code plugin
copies are generated delivery artifacts so each marketplace package can start
standalone with system Python. Refresh both copies after a canonical change:

```bash
make sync-plugin-mcp
```

`make lint` and CI run the same `--check` validation and fail if either
generated copy differs from the canonical source.

## Design Principles

- Keep client adapters thin. They translate hook payloads, redact obvious
  secrets, and upload bounded windows.
- Keep memory extraction, package generation, lifecycle, and source sync inside
  the service.
- Prefer clear ownership boundaries over fallback paths that bypass the owning
  service.
- Add tests where behavior changes. Keep docs in sync with public workflows.

## Pull Request Checklist

- The change is scoped and described in the PR body.
- Tests or verification commands are listed.
- User-facing behavior is documented when relevant.
- No generated caches, local databases, private URLs, or credentials are
  included.
