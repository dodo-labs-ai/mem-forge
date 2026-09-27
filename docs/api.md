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
| Memories | `POST /api/memories/search` | Service-owned ranked memory search for agent MCP proxies |
| Memories | `GET /api/memories/{memory_id}` | Inspect memory detail and provenance |
| Recent changes | `GET /api/recent-changes` | List changed source documents and optionally changed memories |
| Source Units | `GET /api/source-units/{source_unit_id}/artifacts` | List the artifacts one Source stored for its Unit's current revision |
| Source Units | `GET /api/source-units/{source_unit_id}/artifacts/{kind}` | Fetch an explicit artifact such as `normalized_markdown`, `raw_source`, or `pdf` |
| Source Units | `GET /api/source-units/{source_unit_id}/content` | Fetch the Unit's normalized source content from service storage |
| Source Units | `GET /api/source-units/{source_unit_id}/pdf` | Fetch the Unit's stored PDF rendition when available |
| Documents | `GET /api/documents/{doc_id}/artifacts`, `.../artifacts/{kind}`, `.../content`, `.../pdf` | The same reads for the newest copy of the Document stored by a Source the caller can read |
| Review | `GET /api/review` | List review queue items |
| Agent sessions | `POST /api/agent-sessions/windows` | Submit a redacted evidence window |

The API schema is also available from FastAPI's generated OpenAPI endpoint while
the service is running.
