# Architecture overview

MemForge is a self-hosted service that turns a team's documents and coding-agent
sessions into short, source-cited memories and serves them to agents over MCP.
This page names the parts and follows the data between them, for people who run
MemForge or contribute to it.

## Components

| Component | What it does | Code |
| --- | --- | --- |
| API service | A FastAPI application that owns sources, sync, memories, reviews and settings | `src/memforge/server` |
| Source connectors | One per source type: Confluence, Jira, GitHub Pages, GitHub repositories, Microsoft Teams, local folders and agent sessions. The code calls a connector a *gene*. Each fetches documents and hands them over in one common form | `src/memforge/genes` |
| Pipeline | Extracts memories from each document with an LLM, keeps the Evidence each memory rests on, and rechecks memories when their source changes | `src/memforge/pipeline` |
| Storage | SQLite for documents, memories and sync state, with FTS5 for keyword search, and Chroma for vectors | `src/memforge/storage` |
| Retrieval | Vector, keyword and entity channels run in parallel and are fused into one ranking | `src/memforge/retrieval` |
| Agent plugins | Claude Code and Codex plugins. Lifecycle hooks upload redacted session windows, and a local MCP proxy forwards tool calls to the API | `integrations` |
| Local daemon | Uploads files from a local folder or repository for a Local Repository source | `src/memforge/local_agent` |
| Admin UI | The web UI for sources, memories, reviews, projects and settings. `admin` is replacing `admin-ui`; both ship until the replacement is complete | `admin`, `admin-ui` |

## How data moves

1. **Capture.** A connector syncs documents on demand or on a schedule. Agent
   plugins upload bounded, redacted windows of each session.
2. **Extract.** The pipeline turns each document into memories: facts,
   decisions, procedures and conventions. Each memory keeps Evidence that
   points back to the text it came from.
3. **Reconcile.** When a source changes, the memories built on it are rechecked
   against the new revision. A memory is kept, replaced or retired, or it waits
   for a review. Memories from different documents are related when they state
   the same thing, update one another or contradict one another.
4. **Recall.** An agent calls `search` for memory cards, `get_memory` for a
   memory's Evidence and relations, and `get_resource` for the backing
   document. An agent can also propose creating, correcting or retiring a
   memory, and confirms with the person first.

## Deployment

`docker compose up` starts two containers, the API and the admin UI, with one
data volume. The open-source build serves a single user in one workspace named
`local`, has no authentication and binds to `127.0.0.1`.

Model calls go to the endpoints you configure: one model for extraction and
semantic judgments, one for embeddings. Without an extraction model a sync
stores content and creates no memories.

## Adding a source

A connector implements the interface in `src/memforge/genes/base.py`: it
describes itself and its configuration fields, authenticates, discovers the
content in scope, fetches each item and normalizes it into the common form. The
existing connectors in the same directory are the reference.

## Further reading

- [Quickstart](quickstart.md): setup, models, plugins, sources and the CLI.
- [API overview](api.md): the core Admin API endpoints.
- [Local repository sync](local-repo-sync.md): the local daemon and its jobs.
