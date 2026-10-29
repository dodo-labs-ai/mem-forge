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

`--quality` adds the quick offline quality evaluation to the Compose smoke. It
syncs the synthetic documents in `evals/quality/quick/docs`, then replaces the
ones in `revisions/` with their second version and syncs again, and evaluates
three kinds of cases with the rubrics in `cohort.json`:

- extraction: how many memories each document should give and terms they must
  hold;
- Support after an update: whether each old memory still has support in the
  second version, naming memories by a term their claims hold;
- reconciliation: the hand-written situations in `reconciliation/`, each an old
  memory, a new candidate with its evidence and the expected outcome for the
  old memory (`kept`, `review`, `revised` or `retired`).

Each kind replays with the configured model and prints a summary with a verdict
below the smoke table. A `fail` names the case and the check; run it again
before you conclude, since a model's answers vary. A model may write two
labelled facts into one memory; the memory is then expected unsupported when any
of its terms is, since a claim holds only when all of it does. To test a new
situation, add a document pair or a reconciliation file and list it in
`cohort.json`.

Maintainers also run the same evaluation over cases pinned from a real
workspace, which stay in that workspace.

The rubrics are written against one reference model, Claude Haiku 5.5, which
is the default generation model of `compose_smoke.sh` with OpenRouter. The
verdict your pull request needs is the one on that model, so today it needs an
OpenRouter key. A run with `--ollama` or another `--generation-model` is still
worth reporting, but it is advisory: a `fail` there may come from the model
rather than your change. Name the provider and model of every run you paste.

### Provider keys for the smoke scripts

`provider_smoke.py` checks OpenRouter, OpenAI and Anthropic when you have their
keys, and a local Ollama when it answers, and reports the rest as skipped. Add
any other LiteLLM route with `--litellm-route`, with `--endpoint` when it needs
one and `--kind embedding` for an embedding model.
`compose_smoke.sh` configures its models through OpenRouter, or through a
local Ollama with `--ollama`; with neither, it skips the model, sync and search
steps, and your pull request should say so. Keep your keys in one file instead
of exporting them in every terminal:

```bash
cp .env.test.example .env.test
```

Then fill in the keys you have. Git ignores `.env.test`, and both scripts read
it on start; a variable exported in your shell takes precedence. Keep these
keys out of `.env`, which configures the MemForge you run.

For memory plugin changes, also run the focused Python tests:

```bash
uv run pytest tests/test_hook_adapter.py tests/test_workspace_bindings.py tests/test_mcp_workspace_contract.py -q
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

- Keep client adapters thin. The hook adds static usage guidance, and the MCP
  proxy forwards tool calls; neither captures sessions.
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
