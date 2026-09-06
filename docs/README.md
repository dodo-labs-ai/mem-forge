# Documentation

Start here if you are new to the repository:

- [Quickstart](quickstart.md): local setup, API/admin UI startup, agent plugin
  installation, and first memory queries.
- [API overview](api.md): core Admin API endpoints.
- [Architecture](architecture.md): service design, memory model, pipeline,
  storage, retrieval, MCP tools, and admin UI boundaries.
- [Source sync to Memory](design/source-sync-to-memory.md): complete lifecycle,
  model responsibilities, scenarios, and current-versus-target implementation review.
- [Agent client integrations](integrations/agent-clients.md): Codex and Claude
  Code adapter responsibilities, MCP request flow, and service boundaries.
- [Python distribution release](releasing-python-distribution.md): package
  naming, Trusted Publisher configuration, release tags, and install smoke.

## Design ownership

ADRs record durable shared decisions. Subsystem design documents explain their
execution. Accepted design does not imply implemented or deployed behavior;
read each document's status and current-versus-target comparison.

| Question | Maintained document |
|---|---|
| Full Source sync to Memory flow and model stages | [Source sync to Memory](design/source-sync-to-memory.md) |
| Memory, Support, lifecycle actions and Review meanings | [Document Memory Lifecycle](design/document-memory-lifecycle.md) |
| Current extraction/catalog/selector contract | [Source-Agnostic Memory Extraction](design/source-agnostic-memory-extraction.md) |
| Representation-specific incremental Primary algorithm | [Incremental Primary authority](design/representation-scoped-incremental-primary-authority.md) |
| Unified support/claim assessment orchestration | [ADR 0034](adr/0034-unify-incremental-support-and-claim-assessment.md) |
| Historical large-document incident and unapproved Review options | [Large-document recovery analysis](design/large-document-reconciliation-recovery.md) |
| Managed agent-session entrypoint | [Agent-session flow](design/agent-session-saas-plugin-flow.md) |

Cloud documents link shared OSS decisions and describe adapter/hosting
consequences. Historical design analysis is not a second live specification or
an execution backlog; use the issue tracker for execution status.
