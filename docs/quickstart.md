# Quickstart

This guide starts MemForge on your machine, connects Codex or Claude Code, adds
a first source, and shows how agents read memory and its evidence.

## 1. Start The Stack

Requirements:

- Docker with a current Compose v2
- An API key for the generation model that writes memories (Anthropic by
  default) and, for vector search, an embedding endpoint (OpenAI by default)

```bash
git clone https://github.com/shno-labs/mem-forge.git
cd mem-forge
cp .env.example .env
```

Leave the model settings in `.env` empty: you choose the models and enter
their keys in the admin UI (step 2). To name them in `.env` instead, see
[Configuring models in `.env`](#configuring-models-in-env).

Then start the stack:

```bash
docker compose up --build
```

Open `http://localhost:5174`. The UI is served by the `admin-ui` container and
proxies `/api/*` requests to the `api` container. The API is also available
directly at `http://localhost:8765`.

Runtime data is stored in the `memforge-data` Docker volume. Remove that volume
only when you intentionally want a clean local instance.

### Network scope

This OSS profile is single-user and authentication-free. Compose publishes both
ports on host loopback only (`127.0.0.1`), so the browser, the CLI, the local
collection daemon, and host-side Codex or Claude Code plugins on the same
machine can connect, while other LAN devices cannot. Do not use this profile as
a shared or remote service.

A client running in another container has its own `localhost`; give it an
explicit host/container route instead of widening the published host binding.

If either port is already in use, set `MEMFORGE_API_HOST_PORT` or
`MEMFORGE_ADMIN_UI_HOST_PORT` in `.env` and restart with the same
`docker compose up --build` command.

### Restricted or slow registry networks

If base images fail to pull from Docker Hub, set a mirror prefix in `.env`
before rebuilding:

```bash
sed -i.bak 's#^MEMFORGE_DOCKERHUB_PREFIX=.*#MEMFORGE_DOCKERHUB_PREFIX=docker.m.daocloud.io/library/#' .env
docker compose up --build
```

The repository also includes a build mirror profile covering Docker Hub, Debian
apt, PyPI/uv, and npm:

```bash
docker compose --env-file .env.mirrors.example up --build
```

Copy `.env.mirrors.example` to `.env` only when you also want to edit local
model keys or ports.

### Reaching model providers through a proxy

The API container calls the model providers itself. If they are reachable only
through a proxy on your machine, add it to `.env` and restart the stack.
`host.docker.internal` is the host as the container sees it, and the proxy must
accept connections from the container:

```bash
HTTPS_PROXY=http://host.docker.internal:7890
HTTP_PROXY=http://host.docker.internal:7890
NO_PROXY=localhost,127.0.0.1,api,admin-ui,host.docker.internal
```

A model server on your machine, such as Ollama at `host.docker.internal`, is
reached directly through `NO_PROXY`.

The API image uses WeasyPrint for Confluence PDF export. This keeps the image
much lighter than bundling a browser runtime while preserving print-oriented
HTML layout for source evidence PDFs.

## 2. Configure Models

MemForge uses two models:

- A **generation model** reads your sources and writes memories. Without one,
  a sync stores source content but produces no memories.
- An **embedding model** turns memories into vectors for semantic search. It
  is optional: without one, search falls back to full-text ranking.

Open `http://localhost:5174/v2/settings`. For each model, choose a provider
and fill in what it asks for:

| Provider | Use it for | You enter |
| --- | --- | --- |
| OpenAI | OpenAI models | API key, model |
| Anthropic | Claude models (generation only) | API key, model |
| OpenRouter | Any model OpenRouter offers | API key, model |
| Ollama | Models running on your machine | Model; the address if it is not `http://localhost:11434` |
| OpenAI-compatible endpoint | A proxy or local server that speaks the OpenAI API, such as LM Studio or vLLM | Base URL, model, and an API key if the server wants one |
| LiteLLM route (advanced) | Any other provider LiteLLM supports | The model as `provider/model`, and an API key or the provider's own environment variables |

Fields you can leave blank are marked *(optional)*; the others are required,
except that an API key already saved can be left blank to keep it. A provider
with an address of its own uses it when the base URL is left as it is.

Where the provider can list its models, **Load models** offers the models of
the kind you are setting up: OpenRouter's embedding list for the embedding
model, OpenAI's embedding models for the embedding model and its other text
models for the generation model, and Ollama's models by what each one can do.
An OpenAI-compatible endpoint does not say which of its models embed, so all of
them are offered. When the endpoint lists no model of the kind, Settings says
so and you type the model id. Load models fills the model field only when the
endpoint lists exactly one model of the kind; otherwise pick one, or type in
the field to narrow the list.

### Save and validate

**Save and validate** stores your edit without changing the model MemForge is
using, then sends one small real request to the model:

- The generation model must return a short structured answer.
- The embedding model must return two vectors of the same size, twice.

When the request succeeds, your edit becomes the model in use, and the next
sync and search use it. An embedding model that changes what the vectors mean
is the one exception: it is validated and then waits for a reindex (see
[Changing the embedding model](#changing-the-embedding-model)). When the
request fails, MemForge keeps using the previous model and shows why (for
example a rejected API key or an unknown model), so a typing mistake cannot
break a working setup. The failed edit stays in the form until you correct it
or choose **Discard changes**.

Each card shows the model in use above the form. **Validate again** repeats
the request for that model at any time; the model stays in use whatever the
result. Until a generation model is ready, every other page of the new admin
(the pages under `http://localhost:5174/v2/`) shows a notice that links to
Settings.

Each model shows one of three states:

| State | Meaning |
| --- | --- |
| Ready | The model passed its latest check. |
| Not validated | The model has not passed a check yet. Settings that came from `.env` or from an earlier MemForge version start here and are checked in the background when the server starts. |
| Needs revalidation | The model passed a check before, but something changed since (its settings, its API key or the installed LiteLLM version), or its latest check failed. It stays in use; open Settings and choose **Validate again**. |

### Changing the embedding model

Vectors from two embedding models cannot be compared. Neither can vectors of
one model at two output sizes. So the vectors of all memories form one index
that belongs to one embedding model, and changing that model means building
the index again.

Editing only the API key of the embedding model changes nothing about the
vectors: the edit is validated and put to use at once. An edit that changes
the provider, the base URL, the model or its dimensions is different:

1. **Save and validate** checks the new model. When it passes, Settings shows
   it as validated and waiting for a reindex. The previous model stays in use
   and search keeps working as before.
2. **Rebuild search index** runs the reindex: it embeds every searchable
   memory again with the new model into a new index, verifies that the new
   index holds one valid vector for each of them, and then switches: the new
   model and the new index come into use together, and the previous index is
   deleted. Settings shows the progress.

While a reindex runs, search keeps answering from the previous index. Sources
do not sync, and memories cannot be created, corrected, retired or decided on
in a review; such a request is refused with the code `reindex_in_progress`
and can be repeated afterwards. A reindex does not start while a source is
syncing. A reindex that fails, or is interrupted by a restart, changes
nothing: the previous model and index stay in use, Settings shows the reason,
and you can run it again. A reindex whose process was killed is ended when
the server next starts on the same machine, and otherwise within a minute;
until then it still counts as running.

The same rebuild runs from the command line:

```bash
memforge maintenance reindex --check   # shows the index and what a rebuild would do
memforge maintenance reindex --apply   # rebuilds the index and switches to it
```

**When a reindex is required.** If the embedding model in use is not the one
the index was built with, MemForge does not mix the two: search answers
without its semantic channel, creating a memory is refused with the code
`reindex_required` (`vector_index_missing` when the stored vectors fit no
index) until the index is rebuilt, vector updates that were already queued
wait, and the health endpoint and Settings say that a reindex is required.
Every other page of the new admin shows a notice that vector search is off,
with a link to Settings, also while the reindex that ends this state runs.
This happens when you change `MEMFORGE_EMBEDDING_MODEL` or
`MEMFORGE_EMBEDDING_BASE_URL` in `.env` for an installation that already has
memories. Run the reindex with **Rebuild search index** on the card of the
embedding model, or put the previous values back.

**Upgrading.** An installation that already has vectors keeps them. The first
time its embedding model answers, MemForge checks that the stored vectors have
the length the model returns and adopts them as the index in use, marked as
adopted because the model they came from is assumed rather than known. If the
lengths differ, a reindex is required.

**Similarity thresholds.** The thresholds MemForge uses to recognize
duplicates and near matches were tuned on OpenAI `text-embedding-3-small`.
They apply unchanged to every other model, and Settings marks such a model as
*thresholds not calibrated*: expect duplicate detection to be somewhat more or
less strict than with the tuned model.

### API keys

A key you enter in Settings is stored encrypted in the local database and is
never shown again; Settings displays only its last four characters. A saved
key belongs to the address it was saved for. If you point a model at a
different host, port or scheme, enter the key again: MemForge does not send a
saved key to a new address.

Choose **No authentication** for a local server that needs no key.

A model that takes its credentials from the provider's own environment
variables is reached at the provider's own address. To send its requests to
another base URL, enter an API key for that address. A model from `.env` can
name a `provider/model` route with a base URL and no key; Settings shows such
a model and, before it saves a change to it, asks you to either remove the
base URL or choose **API key** (or, for a server that needs no key, the Ollama
or OpenAI-compatible provider). A base URL is stored as the address the API's
paths start from: a request path pasted at its end, such as
`/chat/completions`, `/embeddings` or `/models`, is left out. It must not
contain a user name or a password.

### Models on your own machine

If MemForge runs in Docker and your model server runs on your Mac, use
`http://host.docker.internal:<port>` instead of `http://localhost:<port>`.
Inside the container, `localhost` points back to the container itself.

MemForge sizes its requests to the token limits of the generation model: its
context window and the most it can write in one answer. LiteLLM knows these
limits for most hosted models. For a model it has no limits for, which is
common for local models, validation stops with the code
`token_limits_required` and Settings shows one more step: enter the
**context window** and the **maximum output**, in tokens, with the generation
model, and validate again. Take the numbers from the model card or from your
model server; a context window of 32768 and a maximum output of 4096 are
typical for a local model. When you pick a model from **Load models** and the
provider's list gives its limits, as OpenRouter's does, Settings fills the
empty limit fields with them and says they came from that list; you can still
change them.

You can also set the limits in `.env`. A limit entered in Settings is the one
used; the `.env` values apply to whatever Settings leaves empty:

```bash
MEMFORGE_LLM_MAX_INPUT_TOKENS=32768
MEMFORGE_LLM_CONTEXT_WINDOW_TOKENS=32768
MEMFORGE_LLM_MAX_OUTPUT_TOKENS=4096
```

### Configuring models in `.env`

You can name the models in `.env` instead of Settings:

```bash
MEMFORGE_ENRICHMENT_MODEL=...
MEMFORGE_ENRICHMENT_BASE_URL=...
MEMFORGE_ENRICHMENT_API_KEY=...
MEMFORGE_EMBEDDING_MODEL=...
MEMFORGE_EMBEDDING_BASE_URL=...
MEMFORGE_EMBEDDING_API_KEY=...
```

These values apply to a model until you save and validate one in Settings;
from then on the saved model is used. A plain generation model name is an
Anthropic model, and `provider/model` is a LiteLLM route. A plain embedding
model name is sent to the OpenAI-compatible endpoint at the base URL, and a
name that starts with a provider LiteLLM knows (`ollama/nomic-embed-text`) is a
LiteLLM route. A base URL left unset is `https://api.anthropic.com` for
generation and `https://api.openai.com/v1` for embedding.

Restart the stack after editing `.env`:

```bash
docker compose up --build
```

`memforge config show` prints the model each kind is using, where its settings
come from and its state. It never prints a key.

## 3. Install Agent Plugins

Add this repository as a plugin marketplace and install the plugin. No checkout
is required; the marketplace is fetched from GitHub.

```bash
# Codex
codex plugin marketplace add shno-labs/mem-forge
codex plugin add memory@memforge
```

```text
# Claude Code (run inside an active Claude Code session)
/plugin marketplace add shno-labs/mem-forge
/plugin install memory@memforge
```

Start a new Codex or Claude Code session after installing the plugin, and keep
Docker running while the agent client is open.

Optional Codex check:

```bash
codex mcp get memforge --json
```

The MCP server should be named `memforge`.

Each plugin does two things:

- **Session capture.** Lifecycle hooks read the local session, redact obvious
  secrets, and upload bounded evidence windows to
  `POST /api/v1/agent-sessions/windows`. The service redacts again, decides
  which user messages carry durable authority, and patches the user's private
  Agent Knowledge. Raw windows are not stored. The local collection daemon is
  not required for session capture.
- **Memory tools.** A plugin-local MCP proxy speaks stdio to the agent and
  calls the MemForge API over HTTP(S). Search and provenance logic stay in the
  service, while `get_resource(mode="file")` writes a real file on the agent
  machine under `~/.memforge-agent/artifacts`. The proxy does not need the
  `memforge` CLI.

### Plugin routing

By default the plugins use `http://127.0.0.1:8765`. Set `MEMFORGE_API_URL` and
optional `MEMFORGE_API_TOKEN` only when pointing the plugin at another MemForge
service:

```bash
export MEMFORGE_API_URL=https://api.example.memforge
export MEMFORGE_API_TOKEN=...
```

For Claude Code, put these values in the top-level `env` object in
`~/.claude/settings.json`. Lifecycle hooks do not inherit MCP-server-only
environment.

MCP offers `list_workspaces`; every other tool accepts an optional
`workspace_id`. Installed clients resolve a user-confirmed local project
binding and send it as an explicit selector. Omission is safe only when exactly
one accessible workspace remains. Self-hosted MemForge exposes the single
workspace id `local`.

## 4. Add A Source

Use the **Sources** screen in the admin UI to add a source and run a sync.

Each source can also run on its own server-side schedule. Open the source's
**Configure** dialog, enable **Sync on a schedule**, and choose an interval.
Scheduled runs use the same backend queue and source permissions as a manual
**Sync** click; an already-running source is skipped until the next interval.

To sync a local folder or repository, choose **Add Source -> Local Repository**
and run the printed CLI command. See [local-repo-sync.md](local-repo-sync.md).

| Source | What is synced |
| --- | --- |
| Claude Code, Codex | Decisions, conventions and corrections from your sessions |
| Confluence | Pages, including exported PDFs; accepts a root, space or page URL |
| Jira | Tickets, decisions and work items |
| GitHub Repository | Repository files from GitHub or GitHub Enterprise, scoped by folders and file types |
| GitHub Pages | Rendered documentation pages |
| Microsoft Teams | Channel messages, group chats and direct messages |
| Local Repository | Markdown, text, JSON and HTML files from a local folder or repo, uploaded by the local daemon |

### Confluence

The Confluence source accepts a root, space, or page URL in the **Wiki URL**
field. This keeps standard and corporate Confluence deployments on the same
source type.

Examples:

```text
https://team.atlassian.net/wiki
https://team.atlassian.net/wiki/spaces/ENG
https://wiki.company.example/wiki/spaces/PAY/pages/5695886009/Flexible+Payroll
https://confluence.example.com
```

When a page URL is pasted, MemForge infers the space key, page ID, REST API
path, and page-tree sync scope. `spaces` is required only for whole-space sync.
Plain Confluence roots first try `/wiki/rest/api` and then `/rest/api`; use the
advanced REST API path field only for deployments that serve Confluence below a
custom path.

Confluence PDF artifacts are rendered with WeasyPrint. When running the Python
service directly on macOS, install WeasyPrint's native text-rendering libraries
with `brew install pango`. The Docker image already includes them.

## 5. Query Memory From An Agent

After a source has synced, ask:

```text
I'm about to change the agent-session capture flow.
Check MemForge for the decisions, conventions, and source evidence that matter.
If a memory points to a backing page or PDF, inspect it when the original context
could change your recommendation.
```

Search returns compact memory cards. When a memory conflicts with or has been
updated by another memory, the card carries a `relation_notice` and a
`follow_up` hint. Agents call `get_memory` for provenance. Revision-pinned Evidence carries readable citation `text`,
which may be model-produced focused wording, and a `resource_url` for the pinned
citation material. Manually created memories may instead return an `excerpt`
without a resource link. Revision and range locators identify source-backed
Evidence where available. Full source inspection
retains the underlying excerpt; readable text need not be verbatim. Artifact
evidence uses `evidence[].items[].artifact.url`. The Document `content_url` and
`pdf_url` describe current stored content, which may differ from the cited
revision. Managed-session citations point to generated concept documents,
not retained raw dialogue.

To fetch backing evidence:

```text
Search MemForge for "<topic>". Call get_memory for the relevant memory, then
call get_resource with mode="file" on its resource_url or pinned artifact.url
and show the local_path. Use content_url or pdf_url when you want current content.
```

These URLs are served by the API, so they work even when service storage lives
inside a Docker volume. `local_path` is written by the local plugin under
`~/.memforge-agent/artifacts`, not by the Docker container.

You can also ask the agent to create, correct or retire a memory; the plugin
tells the agent to confirm with you before it writes.

## 6. Install The Host CLI

Install the host-side CLI in an isolated environment when you want to query the
service from a terminal or run local-source adapters from this machine:

```bash
pipx install memforge-ai
memforge --help
```

`memforge-ai` is the Python distribution name; the installed command and import
package remain `memforge`.

Configure the current target and install the local collection daemon as a login
user service with one guided command:

```bash
# Guided setup; press Enter to use the local self-hosted target
memforge setup

# Or configure a hosted target; the token is prompted and saved in the OS keyring
memforge setup --api-url https://memory.example.com
```

With no active target, the guided command prompts for the API URL and offers
the local self-hosted endpoint as its default. MemForge discovers the exact
origin's edition, authentication requirement, API base, and health path from
`/.well-known/memforge`; it does not infer service type from the hostname.

The setup command manages launchd on macOS and the systemd user manager on
Linux. Use `memforge daemon status`, `check`, `restart`, `logs`, `stop`,
`start`, and `uninstall` for subsequent operations; you do not need to write
native service files or keep a terminal open. Status and check include the
server-observed heartbeat, so you can prove connectivity before scheduling or
triggering a source sync.

## 7. Query Memory From The CLI

The CLI mirrors the MCP read flow for local debugging and scripted checks. It
calls the MemForge API instead of reading SQLite directly.

```bash
memforge
memforge search "docker artifact provenance"
memforge get-memory mem-123
memforge get-resource /api/v1/documents/doc-456/pdf --mode file
```

Running `memforge` with no subcommand opens the interactive Clack menu. Keep
Node.js available on `PATH`; MemForge installs the packaged menu dependencies
into `~/.cache/memforge/interactive-cli` on first use, so no manual
`npm install` step is required.

`get-resource --mode file` writes to the same client-local artifact cache used
by the MCP proxy: `~/.memforge-agent/artifacts`.

The CLI uses `MEMFORGE_API_URL` and optional `MEMFORGE_API_TOKEN` when set;
otherwise it targets the local Admin API port from config.

You can configure the same per-source automatic sync from the CLI:

```bash
memforge sources schedule src-123 --every-minutes 60
memforge sources schedule-show src-123
memforge sources schedule src-123 --disable
```

To run a one-off sync of every configured source, or of one source by name:

```bash
memforge sync
memforge sync --source "Source name"
```

## 8. Development From Source

Use the source path when you are changing MemForge itself rather than just
running it. Requirements: Python 3.12 or newer, Node.js 22.12+ (or 20.19+), and `uv`
for Python dependency management.

```bash
uv sync --extra dev
cp .env.example .env
uv run memforge api
```

In another terminal, start V2 at `http://localhost:5175/v2/`:

```bash
cd admin
npm ci
npm run dev
```

V1 remains in `admin-ui/` until V2 replaces it. Source setup still opens V1 pages. For that flow, also start V1 in a third
terminal at `http://localhost:5174`:

```bash
cd admin-ui
npm ci
npm run dev
```

Run checks for the packages you changed before opening a pull request:

```bash
uv run ruff check src tests
uv run pytest -q

cd admin
npm ci
npm run lint
npm run typecheck
npm test
npm run build
```

The same checks run in GitHub Actions. See
[CONTRIBUTING.md](../CONTRIBUTING.md) before opening a pull request.

By default, source-mode runtime data lives under `.memforge/` when you use the
example environment file. That folder is ignored by git. In source mode, prefix
CLI commands with `uv run`, for example `uv run memforge sync`.

Project layout:

```text
src/memforge/        Python service, CLI, pipeline, genes, plugin MCP proxy
admin/               V2 React admin console, served at /v2/
admin-ui/            V1 console, until V2 replaces it
integrations/        Codex and Claude Code plugin packages
docs/                Quickstart, architecture overview and API notes
tests/               Python tests
```
