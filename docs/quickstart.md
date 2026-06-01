# Quickstart

This guide starts MemForge locally, opens the admin UI, and shows where
agent-client integrations plug in.

## 1. Start The Self-Hosted Stack

Requirements:

- Docker with a current Compose v2

```bash
git clone https://github.com/dodoman-sun/mem-forge.git
cd mem-forge
docker compose up --build
```

Open `http://localhost:5174`. The UI is served by the `admin-ui` container and
proxies `/api/*` requests to the `api` container.

The API is also available directly at `http://localhost:8765`.

Runtime data is stored in the `memforge-data` Docker volume. Remove that volume
only when you intentionally want a clean local instance.

When Codex or another host-side agent queries a Docker-hosted MemForge, memory
search works over HTTP as usual. For backing source artifacts, prefer
`content_url` and `pdf_url` from provenance; those URLs are served by the API
and work even when the service storage lives inside a Docker volume.

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
Use MemForge to search for "<topic>". Include source_url, content_url, and
pdf_url when present.
```

To fetch backing evidence:

```text
Search MemForge for "<topic>". If a result has content_url or pdf_url, call
get_resource with mode="file" and show the local_path.
```

`local_path` is written by the local plugin under
`~/.memforge-agent/artifacts`, not by the Docker container.

## 6. Development From Source

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
