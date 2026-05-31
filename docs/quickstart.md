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

## 2. Configure Models And Secrets

Copy the example environment file when you want to set LLM and embedding API
keys or local overrides:

```bash
cp .env.example .env
```

Then edit `.env`:

```bash
MEMFORGE_ENRICHMENT_API_KEY=...
MEMFORGE_EMBEDDING_API_KEY=...
```

Then restart the stack:

```bash
docker compose up --build
```

## 3. Use The Admin UI

Use the Sources screen to add a source and run a sync. The Settings screen can
also store runtime LLM configuration in the local database.

## 4. Start MCP Tools

Agent clients can call MemForge through MCP. From a source checkout:

```bash
uv sync --extra dev
uv run memforge serve
```

The integration packages under `integrations/` register this command by default.
For a development checkout, set:

```bash
export MEMFORGE_MCP_COMMAND="uv run memforge"
```

## 5. Development From Source

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

## 6. Run A One-Off Sync

```bash
uv run memforge sync
```

Use `uv run memforge sync --source "Source name"` to sync one configured
source.
