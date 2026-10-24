# How MemForge Compares

These projects address related needs with different data models. The following
summary uses their public documentation checked on October 7, 2026; it is not a
quality or latency benchmark. Corrections are welcome with a source link.

| | MemForge | Mem0 | Zep / Graphiti | claude-mem | Unblocked |
| --- | --- | --- | --- | --- | --- |
| Main focus | Source-backed Memory and confirmed agent memories | Memory APIs for applications and agents | Temporal context graphs | Persistent coding-session observations | Context from connected engineering tools |
| Source input | Confluence, Jira, GitHub, Teams, local files and memories an agent saves on request | Application-provided content and integrations | Episodes supplied through APIs, loaders or MCP | Coding-agent hooks | Code, tickets, documents, conversations and other connected tools |
| Changing knowledge | Recheck Source revisions; retain lifecycle history and gated Reviews | v3 extraction adds facts with temporal context; manual update/delete APIs are separate | Facts have temporal validity and old facts can be invalidated | Session observations and retrieval | Reconciles sources using context, recency and authority signals |
| Evidence | Readable citation text and backing source links; pinned locators where available | Consult the selected library or Platform API contract | Source episodes and temporal facts | Observation identifiers | Citations to files, PRs, tickets and documents |
| Open-source option | Apache-2.0 local service | Apache-2.0 library; hosted Platform is separate | Apache-2.0 Graphiti; hosted Zep is separate | Apache-2.0 local service | Hosted product |

## Sources for this table

- Mem0: [README](https://github.com/mem0ai/mem0), [OSS v3 migration](https://docs.mem0.ai/migration/oss-v2-to-v3), [Platform v3 migration](https://docs.mem0.ai/migration/platform-v2-to-v3), [Claude Code plugin](https://docs.mem0.ai/integrations/claude-code), [pricing](https://mem0.ai/pricing)
- Zep / Graphiti: [Graphiti README](https://github.com/getzep/graphiti), [edge model](https://github.com/getzep/graphiti/blob/main/graphiti_core/edges.py), [Community Edition notice](https://blog.getzep.com/announcing-a-new-direction-for-zeps-open-source-strategy/), [Zep Ingest](https://help.getzep.com/zep-ingest), [pricing](https://www.getzep.com/pricing)
- claude-mem: [README](https://github.com/thedotmack/claude-mem), [search tools](https://docs.claude-mem.ai/usage/search-tools.md)
- Unblocked: [product overview](https://getunblocked.com/ai-info/), [context engine](https://getunblocked.com/blog/inside-the-unblocked-context-engine/), [MCP](https://getunblocked.com/unblocked-mcp/), [pricing](https://getunblocked.com/pricing/)
