# Changelog

## Unreleased

- Entity admin routes follow Memory visibility. The entity list, detail, alias
  list, linked Memory count and the `/api/v1/stats` entity total admit an
  Entity only through a linked active Memory the caller can see, so a name
  learned from another user's private Memory is not discoverable. Adding or
  removing a manual alias and merging entities require workspace
  administration (403 `entity_curation_forbidden`) and every Entity involved
  must be discoverable (404). Entity detail returns `can_curate`, and
  `/api/v1/stats` counts only Sources the caller can discover. Storage:
  `list_entities`, `count_entities` and `count_memories_for_entity` take
  `scope`. Cloud: HANA implements the same signatures together with the pin
  to this version.
- Review queues and Memory `open_review_id` links share caller-scoped
  participant and Source visibility and pinned-version checks. A hidden or
  dynamically stale newer Review cannot displace an older eligible Review.
  Candidate lookup covers every pending Review involving the waiting Memories
  on the requested page, including related challengers; read paths never change
  durable Review status. SQLite and HANA implement the same candidate contract.
- Review pages hydrate staged Lifecycle evidence only for the requested page,
  and the V2 waiting count requests one item. Exact totals still evaluate all
  matching candidates and participant versions; backend work grows with queue
  size. Memory navigation no longer downloads the full Review queue.

- The runtime provider owns its vector backend. `RuntimeProvider.build_adapters`
  now takes `(db, config, *, audit_logger)` instead of a caller-opened
  `memory_collection`, and the admin memory-store and project routes no longer
  open Chroma themselves. `DefaultRuntimeProvider` opens the `memories`
  collection under `storage.chroma_path` exactly as before, through the same
  helper its search engine and sync runtime use, so OSS behavior and data are
  unchanged. Cloud: a provider backed by HANA never opens a local Chroma store
  on the container disk, including on every `/api/v1/projects` and memory
  write request. Cloud's `CloudRuntimeProvider.build_adapters` takes the new
  signature together with the pin to this version; no HANA schema, `sap/`
  route or environment configuration change.
- `GET /api/v1/projects` returns each project's `memory_count`: the active
  memories in that project the caller can see, including the caller's private
  ones and leaving out Sources the caller turned off. One grouped read,
  `count_memory_admin_projects(scope=...)`, joins `MemoryAdminPageReader` and
  counts with exactly the predicates of `query_memory_admin_page`, so each
  count equals the total the memory list reports for that project. Creating a
  project no longer requires the deprecated `kind`, and a duplicate code is
  refused with a sentence that names the code. The admin UI V2 Projects pages
  use both. Cloud: the HANA workspace store must implement
  `count_memory_admin_projects` (one `GROUP BY M.PROJECT_KEY` over the admin
  list predicates) before it pins this change, or the project list fails.
- The memory list and detail routes name each Memory's Sources and the
  memory list carries its Cross-Document Relations, so the admin UI shows
  project, source, access and relation hints on every row without a request
  per Memory. `GET /api/v1/memories` rows and `GET /api/v1/memories/{id}` gain
  `sources` (the caller-readable `MemorySourceRef`s, as related Memories
  already report them) and the rows gain `relations`; the detail gains
  `source_backed`, true when active Source Units support the Memory so
  `POST /memories/{id}/retire` would refuse it. `POST /api/v1/memories/search`
  declares its response model and leaves unset fields out, so its JSON is
  unchanged for MCP proxies. Cloud: arrives with the pin; the list reads
  `get_memory_source_refs_many` and `list_cross_document_relations`, and the
  detail reads `get_agent_claim_by_memory_id` and
  `get_active_memory_support_states`, all of which the HANA workspace store
  already implements; no schema, protocol or configuration change.
- `GET /api/v1/memories?status=...` lists the Memories in exactly that
  lifecycle status (`pending_review`, `superseded`, `retired`, or the
  `decayed` alias), and `GET /api/v1/memories/{id}` opens a Memory in any
  lifecycle status, so the admin UI can filter by status, open a Memory that
  waits for a review, and follow "Replaced by" links. Without `status` the
  list still holds active Memories only, so its total matches each project's
  `memory_count`. Visibility is unchanged: another user's private Memory stays
  hidden in every status. Cloud: arrives with the pin; the HANA store reads
  through `_hana_visible_sql`, which delegates to the OSS `visible_sql`, so it
  follows the scope's statuses with no storage change.
- `GET /api/v1/memory-reviews` items, `GET /api/v1/memory-reviews/{id}` and
  the decision routes report `can_decide`: whether the caller manages every
  Source behind the Review, so the admin UI shows a view-only Review up front
  and leaves it out of bulk decisions. Every decision route checks the same
  rule, and a Review with no configured Source stays decidable by anyone who
  can see it. The list computes it from the Sources it already reads in one
  `list_sources` call, with no read per Review. Cloud: arrives with the pin;
  the flag is computed in the route from `get_source`, `list_sources` and
  `get_memory_source_ids_many`, which the HANA store already implements, with
  no storage change.
- `GET /api/v1/projects` returns `{data, can_manage}` instead of a bare list.
  `can_manage` says whether the caller may create, rename and delete projects,
  from the same workspace-admin check the project write routes enforce, and
  the admin UI V2 hides New project, Edit and Delete without it. The V1 admin
  UI reads the list from `data`. Cloud: arrives with the pin and follows the
  role the Cloud proxy stamps (`workspace_admin` may manage projects, `member`
  and `viewer` may not); clients that read the list as an array must read
  `data`.
- Deleting a project also releases the Sources that write to it, so none
  keeps writing new memories to the deleted key. In the same relational
  transaction that moves the project's memories to UNSORTED and drops the row,
  a fixed binding to the project is removed (the Source becomes unbound) and a
  field binding drops its mappings to the project, with its default moving to
  UNSORTED if it pointed there. Retired Sources are left as they are, and so is
  a stored binding that is not a JSON object: it routes no memory to the
  project, so it neither blocks the deletion nor gets rewritten. The memories
  keep today's behaviour: they move to UNSORTED and are not retired.
  `DELETE /api/v1/projects/{id}` reports `released_source_count`, and the new
  admin-only `GET /api/v1/projects/{id}/deletion-impact` counts the memories
  and Sources the delete changes across the workspace, so the admin UI V2 can
  state both before a two-step confirmation that asks for the project code.
  The V1 admin UI no longer deletes projects; its project list and detail link
  to the project in V2 instead. `released_project_bindings` is the one rule
  for which Sources a deletion releases: the new
  `RelationalStore.list_sources_released_by_project_deletion` (behind the
  impact count) and `RelationalStore.commit_project_deletion`, which now
  returns the released Source ids, both apply it. Cloud: before it pins this
  change, the HANA workspace store must implement
  `list_sources_released_by_project_deletion` and release the bindings in its
  `commit_project_deletion` transaction through `released_project_bindings`,
  reading the live bound Sources `FOR UPDATE ... ORDER BY ID`, and return
  their ids; no schema, `sap/` route or configuration change. The V1 link
  opens `/v2/projects/<code>`, where Cloud already serves the V2 bundle.
- `GET /api/v1/llm-config` reports `writable`, false when
  `MEMFORGE_LLM_CONFIG_WRITABLE` hands LLM settings to the deployment
  environment, and `PUT /api/v1/llm-config` returns the stored configuration
  instead of `{"ok": true}`. Both routes and `POST /api/v1/llm-config/probe`
  declare response models. Cloud: arrives with the pin; Cloud sets the flag to
  false and its admin UI redirects Settings to `/cloud/settings`.
- `GET /api/v1/agent-evaluations/online-overview` declares its response model,
  and its `summary` and the per-Source `agent-evaluation` summary report
  `row_limit`, the most runtime events and the most assessments one window
  reads, so the admin UI states the bound `truncated` refers to. Cloud:
  arrives with the pin; the HANA reads already honor the query limit.
- Every sync of a document Source removes the Documents its provider no longer
  has, whether the run is incremental, first or force-full, and never removes
  one only because it left the configured query. After discovery each run
  lists its configured scope by identifier, independent of `since`
  (`Gene.list_scope()`), and the listing declares its kind. An existence
  listing (the GitHub repository tree, the GitHub Pages repository tree or
  declared exhaustive sitemap) makes an unlisted Document absent. A query
  listing (Jira JQL, a Confluence page tree or space) confirms each unlisted
  Document by id in the same run (`Gene.confirm_absent()`): only not found or
  gone is absence, so an issue outside a relative-date window, a closed issue
  or a page moved out of its tree keeps its Unit and Support. GitHub, GitHub
  Pages and Confluence list from the walk their discovery already makes; Jira
  lists with an identifier-only search ordered by key and confirms in
  unvalidated `issuekey in (...)` batches of a hundred, reading by id what the
  search does not return. A listing or confirmation that fails or is
  incomplete removes nothing in that run, which still succeeds and logs the
  skipped check; a rejected credential fails discovery and the run; a 403 for
  one item keeps it. `source_run_projection_coverage` no longer takes `incremental`:
  discovery alone proves absence only for an authoritative snapshot or a
  complete discovery of a newly configured scope, and scope transitions are
  unchanged (ADR 0005). Force-full keeps meaning "re-read every Unit". Cloud:
  arrives with the pin; no HANA schema, protocol, configuration or `sap/` route
  change. The first scheduled run after the pin retires what EU12 dev has
  accumulated: 183 absent Units (169 renamed or moved files in one GitHub
  repository, 9 GitHub Pages pages, 5 Confluence pages) and the 844 active
  Memories whose last Support they hold; 678 unlisted Jira issues and 2
  Confluence pages still exist and are kept. See ADR 0045.

- Genes decode stored input by the bytes themselves. A local-agent package is
  recognized by its own `package_kind`, exactly that Gene's kind, never by
  `item.extra.package_uri` or `package_path`, so a repository file that
  happens to be JSON stays file text. What a package attests about its content
  (an empty file, a tombstoned Teams window) is read back from the stored bytes
  through `Gene.raw_from_stored_input`. Reprocess and its `dry_run` preview run
  the same check (`project_stored_input`: normalize, attested content,
  projection, committed location); a Unit that fails it is unavailable with
  `stored_input_invalid` or `stored_input_incomplete` in both, and the run no
  longer retries it. This repairs reprocess of the local-push GitHub inputs
  that the ADR 0041 upgrade migrated with an empty `extra`. Cloud: arrives with
  the pin; no HANA schema, protocol or configuration change, no data fix, no
  resync. See the ADR 0041 amendment of 2026-09-30.

- Sources sync only on their own schedules. The workspace-wide "sync every
  active Source" schedule is removed: `GET/PUT /api/v1/schedule`, its scheduler
  job and the `schedule_all` sync trigger no longer exist, and migration 108
  drops the `schedule_config` table. Per-source schedules
  (`/api/v1/sources/{source_id}/schedule`) are unchanged. Cloud: the workspace
  store no longer needs `get_schedule_config` or `set_schedule_config`; Cloud
  drops its `SCHEDULE_CONFIG` table together with the pin to this version.

- Candidate Admission judges value by one source-neutral definition
  (`candidate-admission-v5`): a supported Candidate is rejected as `low_value`
  when it is a record of what happened once, such as a status transition, an
  assignment, a field change, a version bump or a bare link between items, and
  kept when it is knowledge that holds apart from the event that produced it,
  such as a rule, a design, a decision and its reason, a cause and its fix, or
  a configuration or limit. When unsure, it is kept. The response schema's
  `low_value` reason names this definition. On a held-out set of 207 Candidates
  the rule caught 69% of one-off records, against 12% before, with the same
  knowledge lost. Cloud: prompt text and contract version only, arriving with
  the pin; no configuration, HANA or protocol change. Admission work completed
  under `candidate-admission-v4` is not reinterpreted, and admitted Memories are
  not judged again. See ADR 0034 (Value) and the ADR 0043 amendment of
  2026-09-30.

- The Unit Title is reading context carried by the Source Projection
  (`SourceProjection.unit_title`), no longer a Unit's first Observation. It
  takes no part in Unit revision identity, so a change of only the Unit Title's
  values, such as a renamed Teams chat, creates no revision and no model work; a
  name that also appears in the content or the locator, such as a Confluence
  page title or a file path, changes as content or location does. Claim
  Extraction, Candidate Admission, Support Assessment and Change Impact show the
  current Unit Title in every request, rendered once from the adapter's values,
  and it is never selectable Evidence. The complete-support definition lets a
  claim state every Unit Title value, including a Jira issue type and summary,
  and name its Unit by an earlier name; Candidate Admission reads the selected
  Evidence with the Unit Title, and another Unit's key that neither the Unit
  Title nor the Evidence contains is still unsupported. Stored revisions are
  compared in the current representation only: a Unit that stored its Unit Title
  as an Observation gets one new revision at its next fetch with no changed
  content, its exact Supports are rebound by the program without a model call
  and drop the title part, and a partial projection no longer carries the old
  title. A stored projection read in the current representation, as a reprocess
  preview or an offline replay reads it, drops the stored title from its
  Revision Delta too. The old title is retired, not removed: removal still means
  only proven absence. A store keeps a Unit's current Observations equal to the
  members of its current revision and clears the current revision of every
  Observation that is not a member. Contracts: `revision-input-v8`,
  `projection-extraction-v11`, model presentation policy 6,
  `candidate-admission-v4`, `support-ordered-reading-v6`, `revision-support-v8`
  and `change-impact-v3`. Cloud: no HANA schema, storage protocol, configuration
  or `proxy/external_runtime.py` change; the HANA store's
  `_record_source_projection_sync` must clear the pointers of Observations that
  are not members of the new current revision, shipped together with the pin, or
  reading the current Unit fails for partial Units that stored a Unit Title
  Observation. Derivation attempts staged before the upgrade are not resumed.
  See ADR 0034 and the ADR 0043 amendment on replaying model input changes.

- Relation group evaluation cases (`cross_document_relation_group_v1`) replay
  discovery's requests: one case pins a challenger with every candidate
  discovery asked about for it, in order, and labels for some of them. A
  maintenance operator seeds them with
  `POST /api/v1/agent-evaluations/relation-group-cases/seed`, naming Memories
  by id with the content hash each had when labelled; a group with a changed
  Memory is skipped as `memory_changed`, and the server pins each Memory's
  classifier input and each label as the label the program records. Runs
  classify every candidate of a group and score only labelled ones; each
  check's reason code names its candidate (`expected:actual/memory_id`). Run
  reports add `relation_summary` with `labelled_pairs`, `false_relations`,
  `none_recall` and `unlabelled_relations` (relations given to unlabelled
  candidates, left for people to label).
- Cross-document relation classifier `cross-document-relation-v4`: every
  request states one challenger once as the subject and asks one question per
  candidate, with sharper relation rules (ADR 0043 amendment of 2026-09-29).
  Evaluation cases pinned under `v2` and `v3` are still replayed. Search returns
  both Memories of an `equivalent` pair, each naming the other, instead of
  leaving one out. A completed discovery run replaces every relation the
  classifier recorded for its work, so a re-run keeps no relation for a pair it
  no longer judges, and a re-run that finds its work obsolete keeps none from
  that work; relations a person confirmed and dismissals stay. To apply
  `v4` to existing Memories, re-run completed work with
  `POST /api/v1/relation-discovery/work/rerun` and `{"state": "completed"}`.
- A Memory has no confidence. The Memory API, Review summaries, search
  results, the MCP `create_memory`, `search` and `get_memory` tools, the CLI
  `memories list` table, hook context text and the Admin UI no longer return or
  show `confidence`, and `PUT /memories/{id}` no longer accepts it. A
  `confidence` sent to `create_memory` by an older plugin is ignored. SQLite
  migration 106 drops `memories.confidence` and `agent_claims.confidence`.
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
- Remove the Source activity epoch (ADR 0042). Source activity leases,
  local-agent job leases and derivation identities fence Source writes.
  Local-agent sync job payloads no longer carry `source_activity_epoch`, and
  heartbeats no longer fail jobs with `local_agent_source_activity_epoch_stale`.
  The upgrade supersedes derivations staged before it: the next incremental
  sync derives those Units again, and a reprocess or full sync that was in
  progress must be requested again. Upgrading when no sync is running, no
  derivation is unapplied, and no local-agent job is queued or leased avoids
  that repeated work.

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
