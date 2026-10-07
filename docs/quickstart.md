# Quickstart

This guide starts MemForge on your machine, connects Codex or Claude Code, adds
a first source, and shows how agents read memory and its evidence.

## 1. Start The Stack

Requirements:

- Docker with a current Compose v2
- An API key for the extraction model (Anthropic by default) and, for vector
  search, an embedding endpoint (OpenAI by default)

```bash
git clone https://github.com/shno-labs/mem-forge.git
cd mem-forge
cp .env.example .env
```

Set the model keys in `.env`, or leave them empty and set them later in the
admin UI (step 2):

```bash
MEMFORGE_ENRICHMENT_API_KEY=...
MEMFORGE_EMBEDDING_API_KEY=...
```

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

The API image uses WeasyPrint for Confluence PDF export. This keeps the image
much lighter than bundling a browser runtime while preserving print-oriented
HTML layout for source evidence PDFs.

## 2. Configure Models

MemForge needs an extraction model to create or re-check memories. Without it,
syncs store source content but produce no memories. Embeddings are optional:
when the embedding endpoint is unavailable, search falls back to full-text
ranking.

Open `http://localhost:5174/settings` and configure the enrichment and
embedding endpoints. Use **Test connection** to verify that the MemForge API
container can reach the URL and to load model ids when the endpoint exposes a
model list. Values saved in the UI are stored in the local database and are
used by the next sync.

Model calls go through LiteLLM. A bare model name is treated as an Anthropic
model; a `provider/model` name selects another LiteLLM provider.

If MemForge runs in Docker and your model proxy runs on your Mac, use
`http://host.docker.internal:<port>` instead of `http://localhost:<port>`.
Inside the container, `localhost` points back to the container itself.

For file-based configuration, edit only the values you want to manage outside
the UI:

```bash
MEMFORGE_ENRICHMENT_MODEL=...
MEMFORGE_ENRICHMENT_BASE_URL=...
MEMFORGE_ENRICHMENT_API_KEY=...
MEMFORGE_EMBEDDING_MODEL=...
MEMFORGE_EMBEDDING_BASE_URL=...
MEMFORGE_EMBEDDING_API_KEY=...
```

Then restart the stack:

```bash
docker compose up --build
```

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

See [integrations/agent-clients.md](integrations/agent-clients.md) for the
client-side versus service-side design.

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

V1 remains in `admin-ui/` during the parallel run described by
[ADR 0044](adr/0044-build-the-admin-ui-v2-beside-v1-and-remove-v1-after-a-parallel-run.md).
Source setup still opens V1 pages. For that flow, also start V1 in a third
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
admin-ui/            V1 console during the parallel run (ADR 0044)
integrations/        Codex and Claude Code plugin packages
docs/design/         Design notes for memory extraction and agent sessions
tests/               Python tests
```
