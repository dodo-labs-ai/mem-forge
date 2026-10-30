# AGENTS.md — Project conventions for coding agents

## Code Quality

- Code should look like "first-place design", not "bug-fix archaeology." Comments, docstrings, and prompts should read as if the current approach was always the intended design. A new developer reading the code shouldn't see refactoring history.
- Prefer clean, robust ownership boundaries over fallback workarounds. Do not add DB-only fallback paths for lifecycle operations that also require search/vector cleanup; route those operations through the owning service instead.
- Do not introduce batching, reduce a batch size, or split a schema merely to bypass a provider limit or failing path. First identify whether the limit exposes a better schema, transport, ownership, or execution-boundary design; present the alternatives and tradeoffs to the user and obtain explicit confirmation before choosing batching as the remediation.
- When batching is explicitly approved or is already part of the workload contract, keep it a transport/computation detail: it must not create additional business or lifecycle states, weaken complete coverage, or change atomic and fail-closed correctness semantics.

## Project

- See `README.md` for setup and project orientation.
- See `docs/architecture.md` for the system overview.
- Design records (the full architecture, ADRs, subsystem designs, research and the domain glossary) live in the MemForge Cloud repository under `docs/oss/`. Do not add them to this repository.

## Situational Procedures

Long procedures for specific tasks live in skills under `.agents/skills/`, linked from `.claude/skills/`; load the matching skill before starting such a task.

- `plugin-release-validation`: before changing, cutting or testing the memory plugin for Codex or Claude Code.
- `self-hosted-release-validation`: before validating a self-hosted release or public-beta readiness through Docker Compose.
- `lifecycle-canary-convergence`: before another lifecycle retry, canary or data repair.

## Refactor/Deploy Goal Workflow

- Before starting a goal-scoped refactor, pull `main` first in every affected repo, then create a fresh feature branch or isolated worktree from the updated main. Do not keep implementing on an older feature branch just because it is convenient.
- Before starting any PR merge, check the affected repo for existing open PRs and local and remote feature branches that might be stale, superseded, or still awaiting action. Report the findings and explicitly call out whether any branch or PR should be handled before the merge.
- Treat the approved design and implementation plan as the completion contract. Audit the current implementation against every design requirement before claiming the design is fully covered.
- Record every user-approved deferred MemForge feature or optimization as one GitHub Issue in the coordinating repository. The issue must state the problem and evidence, affected repositories, execution trigger, acceptance criteria, explicit non-goals, type/area, `status:candidate` or `status:ready`, and `priority:P0` through `priority:P3`. Do not treat chat, handoff notes, ADR asides, or duplicate cross-repo issues as the execution backlog, and do not prescribe an unreviewed implementation as settled design.
- When a validated change materially corrects or clarifies the design, update the owning record in the Cloud repository in the same goal. A change to a decision updates its ADR under `docs/oss/adr` with the current rule plus a dated history line, not an appended narrative. Execution detail, such as algorithms, flows, schemas and evidence, goes to the owning design document under `docs/oss/design`. Persist the durable decision and consequences, not the debugging chronology.
- Use online research when making design or technical judgment calls, especially for retrieval, ranking, database search, GraphRAG, deployment, security, or API-contract decisions. Prefer primary sources and current official docs; when useful, inspect relevant open-source project designs or code for implementation insight before settling the approach.
- When explaining or refining retrieval behavior, lead with the runtime flow: user query, independent retrieval channels, fusion/ranking, and exact source/time/visibility predicates. Distinguish vector search, content BM25, metadata lexical channels, graph expansion, and deterministic source/time listing.
- Run review incrementally while implementing, with independent subagents as reviewers. Do not defer review until a large final diff.
- Reviewer feedback is evidence to evaluate. For each finding, decide whether it is valid for the design/runtime contract, whether the proposed fix is proportionate, and whether it should be implemented, rejected, or deferred; record that decision in a handoff log outside the repo.
- Continue the review/fix loop until every reviewer and the implementer approve the result, with no accepted blocker left open.
- A change to a prompt, extraction, judgment, decoding, decision contract or default model runs the `quick` offline quality evaluation before merge: `scripts/compose_smoke.sh --quality` with `OPENROUTER_API_KEY` for the public synthetic cohort in `evals/quality/quick`, plus, for maintainers, `memforge eval offline-run` on the pinned workspace cohorts the change can affect. The verdict that gates is the one on the cohort's reference model; a run on another model, such as through `--ollama`, is advisory. Put the verdict with its provider and model in the PR, and rerun a `fail` once before calling it a regression. The test strategy under the Cloud repository's `docs/oss/design` owns the full rule.
- For cloud-impacting changes, CF deploy and detailed smoke testing are required before the goal can be considered complete. Record the deployment target, smoke commands or UI/API steps, and redacted evidence in the handoff log.
- Open PRs for every repo with committed changes. The goal is not complete until the relevant PRs exist, verification evidence is recorded, and no required repo is left with uncommitted goal work.

## Architecture Decision Ownership

- Record shared Memory lifecycle, retrieval, and storage-contract decisions in an OSS ADR under the Cloud repository's `docs/oss/adr` as the canonical design source. This includes transaction boundaries and work/outbox semantics that every storage adapter must implement.
- Do not duplicate shared semantics in a Cloud ADR (`docs/adr` in the Cloud repository). Update a Cloud ADR only for a material HANA, Cloud Foundry, or other cloud-only consequence; link the canonical OSS ADR and describe only how Cloud implements or constrains that shared contract.
- For a new design, write an ADR when it settles a durable decision and a design document when it needs execution detail; work that does both gets both, the ADR linking the design. Refining execution alone updates only the design document.
- Before changing a subsystem's behaviour, find its owning ADR and design document through the design ownership table in the Cloud repository's `docs/oss/README.md`, and read the record's status first. An Accepted ADR is binding; a Proposed one is context, not settled design; a superseded one is history, so follow its replacement. Accepted design does not mean implemented: check the document's current-versus-target statement. When a task conflicts with an Accepted ADR, stop and raise it with the user instead of working around it.
- A diagram belongs to the design document it explains: write it as mermaid in that document, or keep the rendered image next to its editable source under the Cloud repository's `docs/oss/design/images/`, link it from the document, and name it in the index row.

## OSS and Cloud Contract Consistency

- Treat OSS storage/service protocols as the source of truth for shared behavior. Cloud adapters must implement the same method names, signatures, visibility semantics, filters, pagination, transaction/work semantics, and error behavior unless a documented cloud-only route applies.
- Keep caller identity and visibility explicit in contracts, including `include_private`, `owner_user_id`, `AccessScope`, source/status/project filters, limit, and offset.
- Test shared behavior at the boundary users hit: route/service input, generated store query or adapter call, and returned count/list/detail semantics. A method-existence assertion alone is not contract coverage.
- For shared routes and services, keep adapter-level contract coverage that proves SQLite/OSS and Cloud interpret the contract consistently. Test fixtures and fake stores must implement current protocol signatures; a permissive or narrower fake is contract drift.
- Do not add source-type special cases, compatibility bridges, or route-level fallbacks to hide adapter drift. Update the shared protocol and each adapter implementation directly.
