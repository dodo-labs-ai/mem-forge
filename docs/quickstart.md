# Quickstart

This guide starts a local MemForge service, opens the admin UI, and shows
where agent-client integrations plug in.

## 1. Install Dependencies

```bash
git clone https://github.com/DoDoMan-TTT/mem-forge.git
cd mem-forge
uv sync --extra dev
cd admin-ui && npm ci && cd ..
```

## 2. Configure Local Runtime

```bash
cp .env.example .env
```

Set the LLM and embedding API keys your environment uses:

```bash
export MEMFORGE_ENRICHMENT_API_KEY=...
export MEMFORGE_EMBEDDING_API_KEY=...
```

By default, local runtime data lives under `.memforge/` when you use the
example environment file. That folder is ignored by git.

## 3. Initialize And Start The API

```bash
uv run memforge init
uv run memforge api
```

The API listens on `http://127.0.0.1:8765` unless
`MEMFORGE_ADMIN_API_PORT` changes it.

## 4. Start The Admin UI

```bash
cd admin-ui
npm run dev
```

Open `http://localhost:5174`. Use the Sources screen to add a source and run a
sync. The Settings screen can also store runtime LLM configuration in the local
database.

## 5. Start MCP Tools

Agent clients can call MemForge through MCP:

```bash
uv run memforge serve
```

The integration packages under `integrations/` register this command by default.
For a development checkout, set:

```bash
export MEMFORGE_MCP_COMMAND="uv run memforge"
```

## 6. Run A One-Off Sync

```bash
uv run memforge sync
```

Use `uv run memforge sync --source "Source name"` to sync one configured
source.
