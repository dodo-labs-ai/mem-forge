# How MemForge Compares

MemForge builds team memory from the docs, tickets and chat your team already
has, keeps it current as they change, and cites the source behind each memory
it extracts. The tools below solve related problems in different ways, and for some jobs they
are the better choice.

## Which one fits

- **MemForge:** your agents need what your team already wrote in its wikis,
  tickets, repositories and chat, with a source behind each extracted memory. The
  open-source edition is for one person on one machine; a
  [shared Team Cloud Edition](https://github.com/dodo-labs-ai/mem-forge/discussions/13)
  is on the way.
- **Mem0:** your application needs a memory API for what its users and agents
  say.
- **Zep / Graphiti:** you want a graph of facts that change over time, from data
  you send it.
- **claude-mem:** you want your coding sessions remembered automatically.
- **Unblocked:** you want a team product you can buy today that connects code,
  docs, tickets and chat.

## Side by side

| | MemForge | Mem0 | Zep / Graphiti | claude-mem | Unblocked |
| --- | --- | --- | --- | --- | --- |
| Built from | Your team's wikis, tickets, repositories, chat and local files, and memories an agent saves when you ask | Content your application sends; its Claude Code plugin also captures coding sessions | Data sent through APIs, loaders or MCP | Coding-agent sessions | Code, tickets, docs, conversations and other tools |
| When knowledge changes | Rechecks memories against the new revision, keeps history, links conflicts | Adds facts with temporal context; update and delete are separate APIs | Facts have validity periods; old facts can be invalidated | Records new session observations | Weighs recency, expertise and code across sources; shows both sides of a real conflict, with citations |
| How you check a memory | Cited text and a source link, pinned to the revision where available | Change history for each memory | Source episodes behind each fact | Observation IDs, with the files each one read or changed | Citations to files, PRs, tickets and docs |
| Open source | Apache-2.0 local service | Apache-2.0 library; hosted Platform is separate | Apache-2.0 Graphiti; hosted Zep is separate | Apache-2.0 local service | Not open source; cloud by default, on-premises on Enterprise |

This summary comes from each project's public documentation, checked between
October 7 and November 3, 2026. It is not a quality or latency benchmark.
Corrections are welcome with a source link.

## Sources

- Mem0: [README](https://github.com/mem0ai/mem0), [OSS v3 migration](https://docs.mem0.ai/migration/oss-v2-to-v3), [Platform v3 migration](https://docs.mem0.ai/migration/platform-v2-to-v3), [memory history](https://docs.mem0.ai/api-reference/memory/history-memory), [Claude Code plugin](https://docs.mem0.ai/integrations/claude-code), [pricing](https://mem0.ai/pricing)
- Zep / Graphiti: [Graphiti README](https://github.com/getzep/graphiti), [edge model](https://github.com/getzep/graphiti/blob/main/graphiti_core/edges.py), [Community Edition notice](https://blog.getzep.com/announcing-a-new-direction-for-zeps-open-source-strategy/), [Zep Ingest](https://help.getzep.com/zep-ingest), [pricing](https://www.getzep.com/pricing)
- claude-mem: [README](https://github.com/thedotmack/claude-mem), [search tools](https://docs.claude-mem.ai/usage/search-tools.md)
- Unblocked: [product overview](https://getunblocked.com/ai-info/), [context engine](https://getunblocked.com/blog/inside-the-unblocked-context-engine/), [MCP](https://getunblocked.com/unblocked-mcp/), [pricing](https://getunblocked.com/pricing/)
