# Quickstart

This guide starts MemForge locally, opens the admin UI, and shows where
agent-client integrations plug in.

## 1. Start The Self-Hosted Stack

Requirements:

- Docker with a current Compose v2

```bash
git clone https://github.com/shno-labs/mem-forge.git
cd mem-forge
docker compose up --build
```

If base images fail to pull from Docker Hub, copy `.env.example` to `.env` and
set a mirror prefix before rebuilding:

```bash
cp .env.example .env
sed -i.bak 's#^MEMFORGE_DOCKERHUB_PREFIX=.*#MEMFORGE_DOCKERHUB_PREFIX=docker.m.daocloud.io/library/#' .env
docker compose up --build
```

For restricted or slow registry networks, the repository includes a build mirror
profile covering Docker Hub, Debian apt, PyPI/uv, and npm:

```bash
docker compose --env-file .env.mirrors.example up --build
```

Copy `.env.mirrors.example` to `.env` only when you also want to edit local model
keys or ports.

Open `http://localhost:5174`. The UI is served by the `admin-ui` container and
proxies `/api/*` requests to the `api` container.

The API is also available directly at `http://localhost:8765`.

If either port is already in use, set `MEMFORGE_API_HOST_PORT` or
`MEMFORGE_ADMIN_UI_HOST_PORT` in `.env` and restart with the same
`docker compose up --build` command.

The API image uses WeasyPrint for Confluence PDF export. This keeps the
self-hosted image much lighter than bundling a browser runtime while preserving
print-oriented HTML layout for source evidence PDFs.

Runtime data is stored in the `memforge-data` Docker volume. Remove that volume
only when you intentionally want a clean local instance.

When Codex or another host-side agent queries a Docker-hosted MemForge, memory
search works over HTTP as usual. For backing source artifacts, call
`get_memory` for provenance and use its `content_url` or `pdf_url`; those URLs
are served by the API and work even when the service storage lives inside a
Docker volume.

## 2. Configure Models

Open `http://localhost:5174/settings` and configure the enrichment and
embedding endpoints. Use **Test connection** to verify that the MemForge API
container can reach the URL and to load model ids when the endpoint exposes a
model list.

If MemForge runs in Docker and your model proxy runs on your Mac, use
`http://host.docker.internal:<port>` instead of `http://localhost:<port>`.
Inside the container, `localhost` points back to the container itself.

You can also use `.env` for file-based configuration:

```bash
cp .env.example .env
```

Then edit only the values you want to manage outside the UI:

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

## 3. Use The Admin UI

Use the Sources screen to add a source and run a sync. Settings values saved in
the UI are stored in the local database and are used by the next sync.

Each source can also run on its own server-side schedule. Open the source's
**Configure** dialog, enable **Sync on a schedule**, and choose an interval.
Scheduled runs use the same backend queue and source permissions as a manual
**Sync** click; an already-running source is skipped until the next interval.

## 4. Install Agent Plugins

Agent clients call MemForge through a plugin-local MCP proxy. The proxy does
not need the `memforge` CLI; it talks to the MemForge API started by Docker and
only writes client-local artifact cache files when `get_resource(mode="file")`
is called.

Install the local plugin marketplace:

```bash
# Codex
codex plugin marketplace add ./
codex plugin add memory@memforge

# Claude Code
claude plugin marketplace add ./
claude plugin install memory@memforge
```

By default the plugins use `http://127.0.0.1:8765`. Keep Docker running while
the agent client is open. For a hosted service, set `MEMFORGE_API_URL` and
optional `MEMFORGE_API_TOKEN` before starting the agent.

Start a new Codex or Claude Code session after installing the plugin.

Optional Codex check:

```bash
codex mcp get memforge --json
```

The MCP server should be named `memforge`.

## 5. Query Memory From An Agent

After a source has synced, ask:

```text
Use MemForge to search for "<topic>". If source evidence matters, call
get_memory on the relevant result before citing source details.
```

To fetch backing evidence:

```text
Search MemForge for "<topic>". Call get_memory for the relevant memory, then
call get_resource with mode="file" on the best content_url or pdf_url and show
the local_path.
```

`local_path` is written by the local plugin under
`~/.memforge-agent/artifacts`, not by the Docker container.

## 6. Query Memory From The CLI

The CLI mirrors the practical MCP read flow for local debugging and scripted
checks. It calls the MemForge API instead of reading SQLite directly.

```bash
uv run memforge search "docker artifact provenance"
uv run memforge get-memory mem-123
uv run memforge get-resource /api/documents/doc-456/pdf --mode file
```

Running `uv run memforge` with no subcommand opens the interactive Clack menu.
Keep Node.js available on `PATH`; MemForge installs the packaged menu
dependencies into `~/.cache/memforge/interactive-cli` on first use.

`get-resource --mode file` writes to the same client-local artifact cache used
by the MCP proxy: `~/.memforge-agent/artifacts`.

For hosted MemForge, set the same environment variables used by the agent
plugins before running these commands:

```bash
export MEMFORGE_API_URL=https://api.example.memforge
export MEMFORGE_API_TOKEN=...
```

You can configure the same per-source automatic sync from the CLI:

```bash
uv run memforge sources schedule src-123 --every-minutes 60
uv run memforge sources schedule-show src-123
uv run memforge sources schedule src-123 --disable
```

## 7. Development From Source

Use the source path when you are changing MemForge itself rather than just
running it locally.

```bash
uv sync --extra dev
cp .env.example .env
uv run memforge api
```

In another terminal:

```bash
cd admin-ui
npm ci
npm run dev
```

Run checks before opening a pull request:

```bash
uv run ruff check src tests
uv run pytest -q

cd admin-ui
npm run lint
npm test
npm run build
```

By default, source-mode runtime data lives under `.memforge/` when you use the
example environment file. That folder is ignored by git.

## 7. Run A One-Off Sync

```bash
uv run memforge sync
```

Use `uv run memforge sync --source "Source name"` to sync one configured
source.

## 8. Configure Confluence

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

Confluence PDF artifacts are rendered with WeasyPrint.

When running the Python service directly on macOS, install WeasyPrint's native
text-rendering libraries with `brew install pango`. The Docker image already
includes the required runtime libraries.
