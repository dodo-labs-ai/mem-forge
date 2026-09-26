# Changelog

## Unreleased

- Every Observation Revision records the source's own time for its content: a
  Confluence page version, a GitHub file's latest commit, a GitHub Pages commit,
  sitemap `lastmod` or `Last-Modified`, the latest change to a Jira issue's core
  fields, a comment or changelog entry time, a Teams message's edit or post
  time, a local file's commit or modification time, and the time of the agent
  session event that authorized a concept change. A source without such a time
  records none; discovery, fetch, submission and sync times are never used.
  Times are stored in UTC, and a provider time that cannot be read counts as
  unknown instead of failing the projection. Relation discovery reads this one
  time for every Source, so `updates` between Memories from GitHub, local
  files, Jira issue fields or agent sessions is no longer recorded as
  `contradicts` for want of a time.
- A Memory's `source_updated_at` is the time the Source reports and is empty
  when it reports none; it no longer falls back to the sync time (GitHub
  Repository, GitHub Pages) or the local agent's submission time. Searches that
  filter on `source_updated_at` now match these Memories by their source time.
- GitHub Repository cloud pull reads a file's latest commit when its blob is
  new or has no recorded time: one extra GitHub request for such a file (two
  for a symlink). An unchanged blob keeps the time recorded on an earlier sync.
  A refused commit request leaves the time unknown and is asked again on the
  next sync; it no longer fails the file.
- The local agent sends `source_updated_at` for GitHub Repository and local
  Markdown files, and the edit time of edited Teams messages. Upgrade the local
  agent to record these times; an older one sends none and its files have no
  source time. The upgraded agent declares package contract version 2, and the
  service then asks once more for every retained GitHub Repository or local
  Markdown file that has no source time: the first collection after the upgrade
  uploads those files again, one time.
- After upgrading, run one force full sync of each Jira and GitHub Pages Source
  to record times on existing revisions. GitHub Repository cloud pull records
  them on its next sync, GitHub Repository local push and local Markdown on the
  first collection by the upgraded local agent, and a migration gives current
  Confluence page bodies their page version time. Teams messages edited before
  the upgrade keep their post time. Relations recorded as `contradicts` because
  a time was missing do not change by themselves: re-run relation discovery for
  them afterwards (`POST /api/v1/relation-discovery/work/rerun`). Source
  derivations in progress or deferred during the upgrade may run their
  extraction once more.

- Evidence Unit Support is the only Support model (ADR 0038). A workspace that
  still holds reference-scoped Support refuses to start and is left unchanged;
  move the database aside and rebuild the workspace from its Sources. Other
  workspaces drop the reference-scoped Support and lifecycle cutover tables on
  start.
- `POST /sources/{id}/memory-lifecycle/gate` enables destructive lifecycle for a
  Source recorded before lifecycle gates existed, once every active Memory of
  that Source has complete Support. Until then, its destructive Reviews cannot
  be approved.
- Remove Source lifecycle backfill, rebaseline and cutover finding repair:
  `POST /sources/{id}/memory-lifecycle/backfill`, `.../rebaseline` and
  `.../findings/{finding_id}/repair` are gone, `GET
  /sources/{id}/memory-lifecycle` no longer returns `jobs` and `findings`, and
  the Source list no longer returns `lifecycle_maintenance`.
- Memory Evidence no longer returns `support_scope_version` or `legacy_limited`,
  and the admin UI no longer shows the Legacy limited badge.
- `projection-extraction-v9` is the only extraction contract. Offline
  derivation cases must pin `access_context_hash` and
  `inference_capability_hash`.

## 0.1.63 - 2026-09-26

- Cross-document discovery records one relation label per Memory pair
  (`equivalent`, `updates`, `contradicts`, or nothing) and no longer creates
  Cross-Source Conflict Reviews (ADR 0037). `supersede` is the only Memory Review
  kind; `list_memory_reviews` no longer takes `kind`.
- Search and `get_memory` return `relations` and `relation_notice` in place of
  `conflict_contexts` and `contradiction_warning`. An `equivalent` Memory is
  returned once, and the newer Memory of an `updates` pair ranks directly ahead
  of the older one.
- New MCP tools `dismiss_memory_relation` and `restore_memory_relation`, and
  `get_memory` lists `dismissed_relations` that can be undone. A dismissal
  hides one relation until either Memory changes and changes no Memory. Memory
  detail in the admin UI shows relations with dismiss and undo, and the
  Memories list can show the current relations by the label readers see.
- Maintenance operators can list exhausted relation discovery work and re-run
  exhausted or completed work by error code, time range or classifier version.
- A one-time conversion turns existing Cross-Source Conflict Reviews into
  relations and dismissals (a dismissed Review dismisses both `contradicts` and
  `updates`; decided Reviews can be relabeled by id, as for the evaluation set), re-runs discovery for pending ones, and deletes the converted
  Reviews as a separate step. Decided Reviews can be pinned first as the
  relation evaluation set.
- Relation discovery completes when its Source Unit has a newer revision and the
  Memory's evidence is still current.
- Remove the unused contradiction count and the `/memories/contradictions`
  route.

## 0.1.62 - 2026-09-25

- Retry failed Agent Session capture uploads with exponential backoff from 60
  seconds up to one hour, and reset the backoff after a successful upload.
- Stop sending captures that cannot select a workspace. They wait until the
  session directory resolves to a workspace, either on a later hook or at the
  hourly check; waiting captures whose transcript is older than 7 days are
  dropped and recorded in the queue.
- Show the workspace binding hint at SessionStart while captures wait for a
  workspace and the current project is unbound.
- Add `--workspace-id` to workspace-scoped CLI commands. `search` and `memory`
  commands use the current directory's workspace binding when the option is
  omitted; other commands report the selectable workspace IDs when the account
  has several.
- `memforge adapter auth jira refresh` and `watch` upload one captured session
  to every workspace with a Jira Source for the origin and report each result.
  Only the service's `jira_principal_changed` conflict is reported as a
  principal change.
- `POST /api/v1/agent-sessions/windows` answers a failed model call with a
  retryable 503 (`agent_session_llm_failed`, with `category`, `error_code` and
  `Retry-After`) instead of 400 or 500.

## 0.1.61 - 2026-09-09

- Coalesce Stop, PreCompact, and SessionStart capture requests under one on-demand
  upload owner, with durable fresh-request priority and safe shutdown handoff.
- Bound historical recovery to five sessions per activation and retry failures
  only on later events after a 60-second cooldown.
- Restart both clients after existing workers exit when upgrading from 0.1.60.

## 0.1.60 - 2026-09-08

- Preserve the hook that wakes Agent Session capture while another local worker
  is active. The detached worker waits for the shared queue lock and claims its
  requesting session before unrelated historical backlog.
- Keep the existing bounded claim, lease, bookmark, retry, and server upload
  contracts unchanged.

## 0.1.59 - 2026-09-05

- Publish the Codex and Claude Code plugins with grouped `get_memory.evidence[]`
  provenance. Source links, original-document locators, Evidence excerpts, and
  Artifact locators survive response compaction.
- Upgrade existing plugin installations and reload their MCP connections when
  using a service that returns grouped Evidence. Earlier release tags still
  read the removed flat `sources[]` and `evidence_artifacts[]` fields.

## 0.1.0 - Unreleased

- Breaking: `get_memory` now exposes provenance only through grouped
  `evidence[]`. The deprecated flat `sources[]` and `evidence_artifacts[]`
  fields have been removed; document and PDF locators live under
  `evidence[].document`, and Artifact locators remain under
  `evidence[].items[].artifact.url`.
- Initial public repository preparation.
- Self-hosted MemForge service with FastAPI admin API, SQLite persistence,
  FTS search, Chroma vector search, and MCP tools.
- React admin UI for memories, entities, sources, review, and settings.
- Codex and Claude Code integration packages with thin hook adapters and
  service-owned agent-session package generation.
- Jira browser-session capture now runs in the client CLI and uploads to the
  server over POST /api/auth/jira-session; the server no longer scrapes a
  browser. New `memforge adapter auth jira watch` keeps the session fresh
  proactively. PAT mode is unchanged.
