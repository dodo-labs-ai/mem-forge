# API Overview

The Admin API is served by:

```bash
uv run memforge api
```

Default base URL: `http://127.0.0.1:8765`.

## Core Endpoints

| Area | Endpoint | Purpose |
| --- | --- | --- |
| Health | `GET /api/health` | Runtime and storage health |
| Sources | `GET /api/sources` | List configured sources |
| Sources | `POST /api/sources` | Add a source configuration |
| Sources | `POST /api/sources/{source_id}/sync` | Queue or run a source sync |
| Memories | `GET /api/memories` | Search and filter memories |
| Memories | `GET /api/memories/{memory_id}` | Inspect memory detail and provenance |
| Documents | `GET /api/documents/{doc_id}/content` | Fetch normalized source content from service storage |
| Documents | `GET /api/documents/{doc_id}/pdf` | Fetch a stored PDF rendition when available |
| Review | `GET /api/review` | List review queue items |
| Agent sessions | `POST /api/agent-sessions/windows` | Submit a redacted evidence window |

The API schema is also available from FastAPI's generated OpenAPI endpoint while
the service is running.
