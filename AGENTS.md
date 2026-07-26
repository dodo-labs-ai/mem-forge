# AGENTS.md — Project conventions for Codex

## Code Quality

- Code should look like "first-place design", not "bug-fix archaeology." Comments, docstrings, and prompts should read as if the current approach was always the intended design. A new developer reading the code shouldn't see refactoring history.
- Prefer clean, robust ownership boundaries over fallback workarounds. Do not add DB-only fallback paths for lifecycle operations that also require search/vector cleanup; route those operations through the owning service instead.

## Project

- See `README.md` for setup and project orientation.
- See `docs/architecture.md` for the full system design.
- See `docs/design/agent-session-saas-plugin-flow.md` for the Codex and Claude Code adapter flow.

## Plugin Release Validation

- Test MemForge plugin changes through the same remote plugin install/update path that users use. Do not hand-edit files under `~/.codex/plugins/cache`, and do not add a manual MCP server entry as a workaround.
- For pre-release validation, publish or install a dev/RC plugin artifact from the remote GitHub marketplace source (`shno-labs/mem-forge`), such as `0.1.17-rc.1`, then promote the same commit/tag to the final release after validation.
- When cutting an MCP plugin version, commit the version bump and integration copies first, push the branch, then create a remote tag that names the plugin version, for example `memforge-memory-v0.1.21-rc.4`. Point Codex at that tag with `codex plugin marketplace add https://github.com/shno-labs/mem-forge.git --ref <tag>` and install with `codex plugin add memory@memforge`.
- After installing, verify `codex plugin list --json --marketplace memforge --available`, `codex mcp list`, the marketplace checkout commit/tag, and the cache directory under `~/.codex/plugins/cache/memforge/memory/<version>`. Restart Codex and repeat the version and MCP cwd checks; a restart must not fall back to an older cache.
- Keep Codex and Claude Code on the same remote plugin source. Neither client should reference `/Users/i551096/Dev/mem-inception` as a marketplace or plugin source, because local path installs hide packaging and cache drift.
- If Codex Desktop still falls back after a restart, check for workspace plugin sources such as `.agents/plugins/marketplace.json` in saved workspace roots. A local checkout can silently reinstall `memory@memforge` from its own `integrations/codex/memforge-memory` tree even when the configured marketplace points at the remote tag.
- Release gates for plugin changes should include unit tests, lint, package/install parity, MCP `initialize` and `tools/list` version/tool checks after restart, SessionStart and Stop/PreCompact hook smoke tests, and at least one harmless read/write MCP smoke in a test workspace when write tools change.

## Refactor/Deploy Goal Workflow

- Before starting a goal-scoped refactor, pull `main` first in every affected repo, then create a fresh `codex/` branch or isolated worktree from the updated main. Do not keep implementing on an older feature branch just because it is convenient.
- Before starting any PR merge, check the affected repo for existing open PRs and local/remote `codex/` branches that might be stale, superseded, or still awaiting action. Report the findings and explicitly call out whether any branch or PR should be handled before the merge.
- Treat the approved design and implementation plan as the completion contract. Audit the current implementation against every design requirement before claiming the design is fully covered.
- Record every user-approved deferred MemForge feature or optimization as one GitHub Issue in the coordinating repository. The issue must state the problem and evidence, affected repositories, execution trigger, acceptance criteria, explicit non-goals, type/area, `status:candidate` or `status:ready`, and `priority:P0` through `priority:P3`. Do not treat chat, handoff notes, ADR asides, or duplicate cross-repo issues as the execution backlog, and do not prescribe an unreviewed implementation as settled design.
- When a validated change materially corrects or clarifies the design, update the repo-owned `docs/adr` record in the same goal/PR. Persist the durable decision and consequences, not the debugging chronology.
- Use online research when making design or technical judgment calls, especially for retrieval, ranking, database search, GraphRAG, deployment, security, or API-contract decisions. Prefer primary sources and current official docs; when useful, inspect relevant open-source project designs or code for implementation insight before settling the approach.
- When explaining or refining retrieval behavior, lead with the runtime flow: user query, independent retrieval channels, fusion/ranking, and exact source/time/visibility predicates. Distinguish vector search, content BM25, metadata lexical channels, graph expansion, and deterministic source/time listing.
- Run review incrementally while implementing. When the user asks for two reviewers, use exactly two independent reviewer lanes: one Claude Code reviewer and one Codex reviewer. Do not defer review until a large final diff.
- Reviewer feedback is evidence to evaluate. For each finding, decide whether it is valid for the design/runtime contract, whether the proposed fix is proportionate, and whether it should be implemented, rejected, or deferred; record that decision in a handoff log outside the repo.
- Continue the review/fix loop until both reviewers and the implementer approve the result, with no accepted blocker left open.
- For cloud-impacting changes, CF deploy and detailed smoke testing are required before the goal can be considered complete. Record the deployment target, smoke commands or UI/API steps, and redacted evidence in the handoff log.
- Open PRs for every repo with committed changes. The goal is not complete until the relevant PRs exist, verification evidence is recorded, and no required repo is left with uncommitted goal work.

## OSS Self-Hosted Release Validation

- Validate self-hosted releases through the documented newcomer path: a clean Docker Compose project, loopback-only ports unless explicitly testing exposure, the default SQLite-backed data volume, Admin UI/API startup, first-run model configuration, a representative source sync, memory retrieval, and the packaged local MCP path.
- Use an isolated Compose project and dedicated volume for smoke tests. Never reuse or delete a developer's existing MemForge volume, source data, plugin cache, or external service state merely to obtain a clean result.
- Prove SQLite persistence by restarting the API or Compose stack against the same isolated data volume and rechecking sources, documents, memories, keyword/vector health, and MCP retrieval.
- Treat source support as an execution matrix, not a registry count. For each registered source type, state whether server collection, local-agent collection, managed intake, credentials, browser sessions, VPN/internal reachability, or an external provider is required. Classify each type as verified, unsupported by design, blocked by a missing prerequisite, or failed.
- Exercise local-agent sources through the real server-issued job and daemon path. A direct package injection may diagnose the service boundary, but it does not prove that the documented daemon workflow works.
- For public-beta readiness, separately assess newcomer reproducibility, first-run UX, error messages, documentation accuracy, default data/privacy boundaries, and obvious exposure or secret-handling footguns. Give a conservative `GO`, `CONDITIONAL GO`, or `NO-GO`, with the smallest required fixes or disclosure language.
- Do not present fixture-backed LLM/provider behavior as production-provider validation. Record the fixture contract, then distinguish it from real connector credentials and model-provider coverage.

## Lifecycle Canary Convergence

- Before another retry or canary, review the complete design contract and static call path for the affected state transition: transaction ownership, operation ordering, idempotency, stale guards, retry semantics, and SQLite/Cloud adapter parity.
- Define a falsifiable hypothesis and the exact confirming or rejecting evidence before changing code or data. Classify historical data, migration residue, and one-off provider/runtime failures separately from reproducible shared-contract violations.
- Prefer one bounded joint snapshot that correlates durable runs/jobs, Lifecycle Plans and mutations, current and stale Support, Source Projection lineage, Reviews/Findings, vector outbox, and process health. Do not replace it with repeated narrow probes that guess at adjacent causes.
- Use a bounded convergence sequence: one report-only inventory, one exact dry-run, one controlled apply when authorized, and one strict incremental audit. Stop when the incremental audit is clean and no relevant state changed.
- Never fabricate Evidence, infer missing provenance from semantic similarity, or silently rewrite lifecycle history to make an audit pass. Preserve historical Plans, Relations, Reviews, Findings, and failed jobs.
- Report current and historical populations separately. Distinguish active Memories from superseded or retired lifecycle records and classify unexpected growth by lifecycle reason and bounded time cohort.

## Architecture Decision Ownership

- Record shared Memory lifecycle, retrieval, and storage-contract decisions in an OSS `docs/adr` record as the canonical design source. This includes transaction boundaries and work/outbox semantics that every storage adapter must implement.
- Do not duplicate shared semantics in a Cloud ADR. Update a Cloud ADR only for a material HANA, Cloud Foundry, or other cloud-only consequence; link the canonical OSS ADR and describe only how Cloud implements or constrains that shared contract.

## OSS and Cloud Contract Consistency

- Treat OSS storage/service protocols as the source of truth for shared behavior. Cloud adapters must implement the same method names, signatures, visibility semantics, filters, pagination, transaction/work semantics, and error behavior unless a documented cloud-only route applies.
- Keep caller identity and visibility explicit in contracts, including `include_private`, `owner_user_id`, `AccessScope`, source/status/project filters, limit, and offset.
- Test shared behavior at the boundary users hit: route/service input, generated store query or adapter call, and returned count/list/detail semantics. A method-existence assertion alone is not contract coverage.
- For shared routes and services, keep adapter-level contract coverage that proves SQLite/OSS and Cloud interpret the contract consistently. Test fixtures and fake stores must implement current protocol signatures; a permissive or narrower fake is contract drift.
- Do not add source-type special cases, compatibility bridges, or route-level fallbacks to hide adapter drift. Update the shared protocol and each adapter implementation directly.
