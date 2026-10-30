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

### Quality snapshots

Every source type has a synthetic snapshot in the quick cohort, so a change to
how a type is read is checked by `--quality` as Markdown is. A snapshot is
what the type's collection hands to the rest of MemForge for one item at one
revision; `--quality` creates one Source per type that replays it in place of
the provider, syncs revision 1 and then revision 2, and pins the cases
`cohort.json` lists for it. Everything after collection is the type's own
code: normalization, projection, readings, extraction and Support.

**A new source type adds a snapshot.** `tests/test_quality_snapshots.py`
fails until it has one, and its message names the directory to add and a
snapshot of the same family to copy. The rule for a snapshot:

- It lives in `evals/quality/quick/snapshots/<source_type>/`, next to
  `source.json`, the configuration `--quality` creates its Source with. Use
  example hosts such as `jira.example.test` and a placeholder for any secret;
  the Source never reaches them.
- Each item is a directory holding one numbered directory per revision,
  starting at `1`. At least one item has a revision `2` that changes text
  earlier memories were drawn from, so its update has Support to judge. A
  revision directory holds a whole item; an item that does not change in a
  revision simply has no directory for it.
- All content is synthetic: no real names, hosts, tickets or messages.
- `cohort.json` lists a derivation case for each item and a Support case for
  each revised item, with `"snapshot": "<source_type>"` and the Document's
  title. A conversation Unit's title ends with the time of its first message,
  and replay moves it by whole days, so write the title the snapshot's own
  times give it; `--quality` moves the date.

A **document type** (Jira, Confluence, a repository file, a page) holds the
stored input of one fetch: `item.json` names the discovered item, the raw
file and its content type, an optional `export.pdf` when the type exports a
PDF, and each artifact in `artifacts/`:

```json
{
  "item": {
    "item_id": "jira-KES-42",
    "title": "KES-42: Webhook delivery timeout is too short for slow receivers",
    "source_url": "https://jira.example.test/browse/KES-42",
    "last_modified": "2026-10-20T11:31:00+00:00",
    "content_type": "application/json",
    "space_or_project": "KES",
    "version": "2026-10-20T11:31:00.000+0000",
    "author": "Lena Ortiz",
    "labels": ["webhooks"],
    "extra": {"issue_id": "10042", "issue_key": "KES-42"}
  },
  "raw": {"file": "raw.json", "content_type": "application/json"},
  "artifacts": [
    {
      "file": "artifacts/webhook-latency.png",
      "provider_key": "20077",
      "parent_observation_type": "comment",
      "parent_provider_key": "30102",
      "provider_revision": "2026-10-20T11:30:00.000+0000",
      "filename": "webhook-latency.png",
      "media_type": "image/png"
    }
  ]
}
```

`item` is the item as the type's discovery and fetch leave it, `raw.json` is
the body exactly as its fetch returns it, and each artifact is named as its
fetch describes it. Start from the snapshot of a type whose provider answers
in the same form.

A **conversation type** (Teams) holds the provider's messages for one
conversation: `item.json` names the conversation and how the provider threads
it, and `messages.json` lists each message with its creation time, the thread
root it replies to, and its record in the type's message schema:

```json
[
  {
    "message_id": "1761041100000",
    "created_at": "2026-10-21T10:05:00+00:00",
    "thread_root_id": "1761040800000",
    "record": {
      "content": "Agreed. The canary covers 5% of tenants for the first 24 hours.",
      "from": "Marco Bianchi",
      "time": "2026-10-21T10:05:00+00:00",
      "reply_to_id": "1761040800000",
      "representation": "teams-message:3"
    }
  }
]
```

The record above leaves out fields for brevity; copy a full one from
`snapshots/teams/`. Each revision lists the whole conversation as the provider
would return it then: an edited message carries its new text and
`edited_time`, and a new reply is one more entry.

To check a snapshot locally, run the L1 test, which reads it through the
type's own normalization and projection without a model:

```bash
uv run pytest tests/test_quality_snapshots.py -q
```

Then run `scripts/compose_smoke.sh --quality --quality-outputs DIR` with the
reference model, which saves every case's pinned input, rubric and model output
in `DIR`, and read your type's cases to set their rubrics. A Support rubric
labels every fact the revision changes, so write each one as something a
reader would keep: a stated decision, a limit or a constraint, not an aside.
Reword or drop a term only when the rubric itself is wrong, for example when
the term is not in the source or a paraphrase is just as right. Never drop
one because the model left out a fact that matters: that is the regression
the case exists to catch. Replay is off
in every other deployment: a Source that names a snapshot is refused unless
`MEMFORGE_EVALUATION_SNAPSHOT_ROOT` names the folder its set lies below, which
only `--quality` sets.

The rubrics are written against one reference model, Claude Haiku 5.5. The
verdict your pull request needs is the one on that model, through whichever
provider serves it: `anthropic/claude-haiku-5.5` on OpenRouter, the default
generation model of `compose_smoke.sh`, or `claude-haiku-5-5` with an
Anthropic key (`--generation-provider anthropic`). The embedding model is not
part of the reference. A run on another generation model, for example with
`--ollama` or another `--generation-model`, is still worth reporting, but it
is advisory: a `fail` there may come from the model rather than your change.
The report and each summary name the provider and model of both kinds; paste
them with the run.

### Provider keys for the smoke scripts

`provider_smoke.py` checks OpenRouter, OpenAI and Anthropic when you have their
keys, and a local Ollama when it answers, and reports the rest as skipped. Add
any other LiteLLM route with `--litellm-route`, with `--endpoint` when it needs
one and `--kind embedding` for an embedding model.
`compose_smoke.sh` configures the generation and the embedding model each
through its own provider, as Settings does. By default both use OpenRouter
when `OPENROUTER_API_KEY` is set, or a local Ollama with `--ollama`. Name a
kind's provider with `--generation-provider` or `--embedding-provider`, and
its model, base URL and key variable with `--KIND-model`, `--KIND-endpoint`
and `--KIND-key-env`; a key is read from `<PROVIDER>_API_KEY` unless you name
another variable. A kind you leave unnamed keeps the default, so a generation
key alone needs an embedding provider beside it. For example, a generation
model behind an OpenAI-compatible endpoint, with OpenRouter for embedding:

```bash
scripts/compose_smoke.sh --generation-provider openai_compatible \
  --generation-endpoint https://llm.example.com/v1 --generation-model my-model \
  --generation-key-env MY_LLM_API_KEY --embedding-provider openrouter
```

Or Claude Haiku 5.5 with an Anthropic key, and OpenAI for embedding:

```bash
scripts/compose_smoke.sh --quality --generation-provider anthropic --embedding-provider openai
```

Without a provider for either kind, the smoke skips the model, sync and search
steps, and your pull request should say so. `scripts/compose_smoke.sh --help`
lists each provider's default model. Keep your keys in one file instead of
exporting them in every terminal:

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
