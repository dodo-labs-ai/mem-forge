# Contributing

Thanks for helping improve MemForge. The project is young, so the best
contributions are small, well-tested, and easy to review.

## Development Setup

```bash
git clone https://github.com/dodo-labs-ai/mem-forge.git
cd mem-forge
uv sync --extra dev
cd admin-ui && npm ci && cd ..
```

Copy `.env.example` to `.env` for local overrides. Never commit `.env`,
`.env.test`, local databases, transcript exports, or source-system credentials.

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

CI runs the lint, unit tests and the provider request contract tests on every
pull request. Some changes also need a check that reaches real providers or a
real Docker install, which CI cannot run. Run the ones your change calls for
and paste their output into the PR:

| Your change touches | Also run |
| --- | --- |
| The LiteLLM version, `src/memforge/llm/transport.py`, `src/memforge/llm/providers.py`, or adds a provider or route | `uv run python scripts/provider_smoke.py` |
| The Dockerfile, Compose files, database migrations, startup, or configuration defaults | `scripts/compose_smoke.sh` |
| Prompts, extraction, judgement, decoding, decision contracts, or default models | `scripts/compose_smoke.sh --quality` |

A change that fits several rows needs every check they name. When you are not
sure whether a change belongs to the prompts and extraction row, treat it as
if it does. Releases run all of them.

`--quality` adds the quick offline quality evaluation to the Compose smoke: it
syncs the synthetic documents in `evals/quality/quick`, pins each as a case
with the rubric in `cohort.json` (how many memories its extraction should give
and terms they must hold), replays extraction with the configured model and
prints a summary with a verdict below the smoke table. A `fail` names the case
and the check; run it again before you conclude, since a model's answers vary.
Maintainers also run the same evaluation over cases pinned from a real
workspace, which stay in that workspace.

### Provider keys for the smoke scripts

`provider_smoke.py` checks every provider you have a key for and reports the
rest as skipped; `compose_smoke.sh` uses OpenRouter for its model steps, or a
local Ollama with `--ollama`. Keep your keys in one file instead of exporting
them in every terminal:

```bash
cp .env.test.example .env.test
```

Then fill in the keys you have. Git ignores `.env.test`, and both scripts read
it on start; a variable exported in your shell takes precedence. Keep these
keys out of `.env`, which configures the MemForge you run.

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
