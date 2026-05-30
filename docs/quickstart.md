# Quickstart

This guide starts a local MemInception service, opens the admin UI, and shows
where agent-client integrations plug in.

## 1. Install Dependencies

```bash
git clone https://github.com/DoDoMan-TTT/mem-inception.git
cd mem-inception
uv sync --extra dev
cd admin-ui && npm ci && cd ..
```

## 2. Configure Local Runtime

```bash
cp .env.example .env
```

Set the LLM and embedding API keys your environment uses:

```bash
export MEMINCEPTION_ENRICHMENT_API_KEY=...
export MEMINCEPTION_EMBEDDING_API_KEY=...
```

By default, local runtime data lives under `.meminception/` when you use the
example environment file. That folder is ignored by git.

## 3. Initialize And Start The API

```bash
uv run meminception init
uv run meminception api
```

The API listens on `http://127.0.0.1:8765` unless
`MEMINCEPTION_ADMIN_API_PORT` changes it.

## 4. Start The Admin UI

```bash
cd admin-ui
npm run dev
```

Open `http://localhost:5174`. Use the Sources screen to add a source and run a
sync. The Settings screen can also store runtime LLM configuration in the local
database.

## 5. Start MCP Tools

Agent clients can call MemInception through MCP:

```bash
uv run meminception serve
```

The integration packages under `integrations/` register this command by default.
For a development checkout, set:

```bash
export MEMINCEPTION_MCP_COMMAND="uv run meminception"
```

## 6. Run A One-Off Sync

```bash
uv run meminception sync
```

Use `uv run meminception sync --source "Source name"` to sync one configured
source.
