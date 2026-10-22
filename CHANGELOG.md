# Changelog

## Unreleased

- A query listing's scope transition, such as a JQL or Confluence space
  change, removes only what the user's change excluded. The run that can
  complete it lists each distinct excluded scope once (the last applied scope
  and the targets of every transition it superseded) with the current
  credential and only that scope's fields (`excluded_scope_config`), and
  removes the held Documents an excluded scope lists and the new one does not
  as `removed_from_scope`. Every other unlisted Document had left the query
  before the change and follows the ordinary rule: confirmed absent is
  `absent_from_source`, still present is kept. A transient failure listing an
  excluded scope keeps the transition open; an excluded scope the provider
  rejects (`SourceConfigurationError`) completes it with the ordinary rule.
  Until a device lists its excluded scopes, a local-agent Jira transition
  stays open. `sync_gene` takes `excluded_scope_gene`, which `run_source_sync`
  builds, and both stores gain `get_projection_scope_transition`.
  - A Confluence Page Tree Root that answers 404 and a JQL Jira refuses (HTTP
    400, `JiraQueryRejected`) are configuration errors: the run fails without
    a retry.
  - GitHub Pages single-page and explicit-list Sources list their configured
    pages, so a page dropped from the configuration is removed and a page that
    cannot be read fails the run's listing. A sitemap declared complete that
    is missing, unreadable or not a urlset fails discovery; it no longer falls
    back to the link crawl.

- A document Source's scope transition completes by accumulated commitment.
  Each Document that commits under an open transition, through an unchanged
  item's projection, a lifecycle commit, deferred convergence or the
  recovery of a derivation staged since the transition opened, is recorded as
  one of its members in the same transaction; a deferred commit is recorded
  only when it commits, and a tombstone never. An older recovered derivation
  records no member, so its Document is fetched again.
  Each transition run discovers everything, lists the new scope and fetches
  only the listed Documents that are not members yet (`force_full_sync`
  fetches them all). The transition completes once that listing is complete
  and every listed Document is a member, so a large Source no longer needs a
  whole run without failures. Only the completing run removes: every held
  Document the new listing leaves out, as `removed_from_scope` under the
  transition's lifecycle cycle. A listed Document whose commit waits only on
  Units of this run and Units that completion removes counts as committed:
  the run removes them, converges, and completes only if that Document then
  committed; otherwise
  the transition stays open with those removals applied. A run that cannot
  complete removes nothing and records why in the transition's error.
  - The run that completes a transition sets the sync cursor to the time the
    transition opened, so the next run discovers every change made while it
    was open. Other transition runs keep the cursor; a successful one that
    finds none, as after a credential change, records the opening time.
  - Changing a document Source's scope no longer resets its sync cursor or
    wipes its sync history. A credential change, and a conversation Source's
    scope change, still do.
  - Projection reuse is off while a transition is open, failed included.
  - `Gene.attest_discovery_complete`, `discovery_complete`,
    `discovery_completion_reason` and `begin_discovery` are removed, with
    `source_run_projection_coverage`: the listing is the only evidence.
    `DocumentGene.reset_scope_listing()` forgets a recorded listing.
  - Cloud: the HANA reuse predicate also excludes `failed` transitions.

- A document sync run takes one listing of its scope after discovery, before
  it retries failed Documents, and decides both absence and the failure
  ledger from it. A device's snapshot manifest is that listing, with the
  Gene's single listing kind.
  - Item failures no longer suspend absence. A Document that failed to fetch,
    normalize, project, derive or commit says nothing about one the listing
    left out, so the unlisted one is removed in the same run. The exception
    is a rename: `ContentItem.renamed_from` names the Document id the item
    was renamed from. The item commits to that Document's current Unit, and
    the Unit is kept until the item commits, while the item is in the run or
    in an open ledger row. `renamed_from` is serialized only when set. No
    built-in collector attests renames yet; `github_renamed_from` is the
    mapping a GitHub collector uses for an attested `previous_filename`, and
    the GitHub adapter no longer reads rename evidence itself.
  - An existence listing settles every failure ledger row it does not name,
    including a row whose Document never became a Unit and a row a scheduled
    run would leave alone. A server query listing retries an unlisted row
    through `rediscover` (settled when the item is gone or left the query) or,
    without one, keeps it untried. A listing that fails settles nothing.
  - Local-agent Jira no longer removes an issue that dropped out of its
    manifest. Until the device confirms by identifier, such a run removes
    nothing and records `absence_confirmation_missing`.
  - A server query listing records how many unlisted Documents the provider
    confirmed it still has or hides as `unlisted_present_count` in the sync
    history; Source health returns it with the latest sync and the Admin UI
    names those kept items.

- A document connector says whether and how it lists its scope. A
  `DocumentGene` must implement the class method `lists_scope(config)` and
  `list_scope()`, which returns a complete `ScopeListing` or raises; startup
  refuses a Gene, including one outside this repository, that leaves either
  out. `confirm_absent` returns an `AbsenceAnswer` (`absent`, `exists` or
  `hidden`) per Document id instead of the set of absent ids, and
  `atlassian_record_state` replaces `atlassian_record_is_gone`.
  `tests/contracts/document_fakes` holds the fakes each registered document
  type needs for the connector contract suite.
- Jira looks one issue up again, when a failed issue is retried or a Document
  is reprocessed at its current revision, by searching within the Source's
  JQL: `issuekey = <id> AND (<where clause>)`, without the incremental
  `updated >=` filter or ORDER BY. Before, it read the issue by id whether or
  not the JQL still matched it. An issue the JQL no longer matches, or the
  credential can no longer see, now fails with `provider_document_missing`,
  as a Confluence page outside its page tree or space already does, instead
  of being refreshed from outside the Source's scope. Confluence's lookup also
  skips content that is not a page, such as a blog post.
- A person can stop a Source's sync in progress: `POST
  /api/v1/sources/{id}/sync/stop`, the Admin UI's "Stop sync" and `memforge
  sources stop`. The Source stays active and later syncs run as scheduled. A
  server run that has not started, and the local daemon's collection job, end
  at once; a running server run ends at its worker's next heartbeat (about 30
  seconds). The run fails with `cancelled_by_user`, is not retried, and is
  shown as "Stopped". It ends like a run whose lease was lost: what it
  committed stays, nothing is removed by absence detection, an open scope
  transition stays open for the next run, and the sync cursor does not move.
  Pausing a Source also stops its sync. A configuration change refused while a
  sync runs answers 409 `source_sync_running` naming the run, and the Admin UI
  offers "Stop the running sync and save". Migration 125 adds
  `source_sync_runs.stop_requested_at`; `heartbeat_source_sync_run` returns a
  `SourceSyncHeartbeat` instead of a boolean, and a run that fails for good
  marks a scope transition it left running as failed.
  - Cloud: HANA adds `SOURCE_SYNC_RUNS.STOP_REQUESTED_AT` and implements
    `request_source_sync_stop`; the Cloud local-agent job store stops the
    Source's device jobs and releases their collection activity.
- Each source type declares its facts once, in a descriptor in
  `memforge.source_types`: its Gene, native projection, evidence profiles,
  namespace and scope fields, progress noun, local-agent collection, PDF
  export and configuration rules. Shared sync, projection, local-agent and
  admin code reads the descriptor instead of branching on the type, so a new
  source type is one connector plus one descriptor. `GENE_REGISTRY` is a
  read-only view of the descriptors, and `@register_gene` is removed: register
  a `SourceTypeDescriptor` with `register_source_type`. A built-in source type
  that fails to load now fails startup instead of being skipped.
- The changelog is written through the store's `insert_changelog`, missing
  PDFs are counted without naming a source type, and projected document moves
  always go through the memory store. The memory health issue
  `confluence_pdf_uri_missing` is renamed `pdf_uri_missing`.
  - Cloud: HANA `count_missing_pdf_uris` drops its Confluence predicate, and
    the Cloud runtime no longer overrides the changelog write, so a failed
    changelog write is logged instead of failing the Document, as in OSS.
- A document Source's next run reads from the start of the previous run, not
  its end, so an item changed while a long run was still going is read again.
  "Last synced" shows that start time.
- A Teams poll that reached its requested start names the time Chatsvc
  certified its read complete from, and the server refuses a manifest whose
  poll evidence does not support its claim. Agents installed before this
  release are refused for Teams until they are upgraded.
- A conversation run logs how many Units it removes per conversation and
  reason (`conversation_units_removed`).
- Reading a conversation Unit fails on an Observation revision that is neither
  a message record nor a Source Artifact, instead of leaving it out.
- Teams Initial History is bounded to 365 days; migration 124 cuts any stored
  longer value to the bound.
  - Cloud: the HANA migration `teams-initial-history-bound-v1` does the same.

- Conversation Sources collect through one server-owned conversation layer
  (`memforge.conversations`), and Teams is its first Source. A connector
  implements only the provider port; the layer decides thread and window
  Units per message, proven intervals, absence, Rolling Retention and scope
  transitions from each attempt's Collection Manifest. Teams keeps its stored
  Unit and Observation identities and records: migration 122 adds the
  collection tables and renames Teams Unit Types to `conversation_window` and
  `conversation_thread`, and migration 123 drops the Projection Scope
  Attestation column. The local agent asks
  `POST /api/v1/sources/{id}/conversation-collection/request`, uploads the
  manifest, artifacts and bodies, then starts processing, or asks for the plan
  when the sync was requested with `dry_run` to verify a migration. A
  conversation the Source cannot read keeps its Support, and the run is
  `partial` and names it. Older agents get `local_agent_upgrade_required`; the
  Teams window-package route and the device ledger and audit log are removed.
  - Cloud: a store implements the `CollectionStore` protocol (collection
    request, manifest and body methods and `prune_conversation_collections`)
    and `list_current_observation_holders`; `list_active_local_agent_jobs` returns
    every queued and leased job so a dry run never joins a real collection.
- A derivation evaluation case can pin a Document's later update: a seed request
  names it with `derivation_id`, and the case also pins the projection of the
  committed base revision the update was derived against, so replay plans the
  same Evidence work as production. Without `derivation_id` the first derivation
  is pinned. Seeding skips a named derivation that is not a completed derivation
  of the Document (`derivation_not_found`) or whose base can no longer be read
  (`base_unavailable`). A replayed derivation whose Evidence work cannot be
  planned fails its case with the failure code
  (`derivation_failed:<code>`) instead of erroring.
  - Cloud: `get_source_unit_revision_projection` is part of the derivation case
    store contract.
- Support Assessment evaluation cases (`source_unit_support_v1`) replay whether
  an old Memory's Support still holds in a new revision. A case pins one applied
  update of a Document from the receipts its Support reads left in the
  derivation work journal: the revision as synced, its committed base and
  validation baselines, every work item the model judged with the claim it saw
  and its Evidence, and the recorded result. Seed with
  `POST /api/v1/agent-evaluations/support-cases/seed` or
  `memforge eval seed-support-cases`. Results are per Support: the rubric's
  `expected_support` labels Memories `supported` or `unsupported`, each label
  applies to the Memory's pinned Supports, and an unlabelled Support the replay
  finds unsupported is `unknown`. Lifecycle and replay plan Support work
  through one shared planner.
  - Cloud: a store implements `list_source_derivation_work` and
    `get_evidence_references`, which returns each stored column as stored;
    `get_source_observation_revisions` is part of the Support case store
    contract.

- A Memory's source text is its Evidence excerpt only. `Memory.extraction_context`
  and `RawMemory.extraction_context` are removed, with the `memories` column
  (migration 121 drops it) and the field of `GET /api/v1/memories/{id}`.
  Candidate quality checks read the candidate's `evidence_quote`.
  - Cloud: `apply_authorized_memory_correction` takes the correction's
    `provenance`, which becomes the `user_correction` source excerpt.

- An image Evidence item's `artifact` in `GET /api/v1/memories/{id}` describes
  only the Artifact: `artifact_id`, `parent_observation_id`, `filename`,
  `content_type`, `size_bytes` and `url`. The item gives its Observation,
  revision, reference, role and digest, and its group the Evidence Unit. The
  item's `text` says what the image shows for the claim. `summary` is gone, with the extraction result,
  derivation payload and telemetry fields that carried it; MCP `get_memory`
  keeps `filename`, `content_type`, `size_bytes` and `url`.
  - Cloud: `SourceArtifactSummary`, `MAX_SOURCE_ARTIFACT_SUMMARY_CHARS`,
    `with_source_artifact_summaries`, `MemoryExtractionResult.artifact_summaries`
    and `SourceArtifactRevision.summary` are removed.

- Claim Extraction writes a Memory in the language of its Primary Evidence,
  even when its Required Evidence is in another language: a Chinese reply that
  settles an English message gives a Chinese Memory. Identifiers, names and
  quoted values stay as the source writes them. The extraction contract is
  `projection-extraction-v24`, and a test ties the extraction prompt to that
  version. The quick quality cohort gains a Chinese case.

- A structured request with images is checked as it is sent: rendered once,
  with no room kept for a correction, which only the packing of a first
  request keeps. A request that does not fit, or that exceeds the encoded size
  its images allow, fails as `input_capacity_exceeded`, a capacity failure the
  batch runner splits or skips, instead of `image_delivery_input_capacity_exceeded`.
  A repair request names as many failed fields as the request has room for; a
  re-ask with no room for its note is not repaired, and its invalid answer
  stands.

- Every structured model request asks for the route's output capacity: the
  model's maximum output, up to a quarter of its context window. A route whose
  limits are unknown sends no maximum and the provider's default applies. An
  answer cut off at that capacity is a capacity failure, like an input that does
  not fit: a request of several items is split, and a single item is skipped or
  rejected with `output_capacity_exceeded`.
  - An Ollama binding states its context window, and every request asks Ollama
    for it as `num_ctx`. The window is no longer read from the Ollama server,
    so validating an Ollama model always asks for the token limits. The
    provider list gains `sets_context_window`, and Settings says, for such a
    provider, that the window costs the server memory.
  - `MEMFORGE_ENRICHMENT_MAX_TOKENS` and `MEMFORGE_MEMORY_EXTRACTION_MAX_TOKENS`
    set nothing and log a warning. `MEMFORGE_LLM_MAX_OUTPUT_TOKENS` or the
    binding's maximum output bounds the output.
  - Structured call telemetry drops `reasoning_budget_retries` and
    `reasoning_budget_recovered`; `requested_max_tokens` is empty when no
    maximum was sent.
  - Cloud: `MemoryExtractor` takes no `max_tokens`, `LlmConfig` has no
    `enrichment_max_tokens` or `memory_extraction_max_tokens`, and
    `LlmRequest` takes no output limit (its third field is `images`).

- Offline replay of a derivation case now shows the model the images its
  derivation read, such as screenshots in a Jira issue. Each image is read from
  the document store by the hash its pinned Observation Revision records, as a
  sync reads it; an image that was deleted or no longer matches its hash ends
  the case as `artifact_unavailable`. The `offline-run` summary counts a check
  that waits for review as unknown instead of leaving its row empty.
  - Cloud: pass the workspace's document store to
    `build_offline_evaluation_for_run(..., document_store=...)`.

- Contributors: the smoke scripts default to Claude Haiku 5.5 for generation
  (`anthropic/claude-haiku-5.5` through OpenRouter, `claude-haiku-5-5` with an
  Anthropic key). On OpenRouter it costs a tenth of Haiku 4.5 per token and
  reads up to 1M tokens. `--generation-model` still picks another model.

- Offline quality evaluation can check extraction quality in one command.
  - `scripts/compose_smoke.sh --quality` syncs the synthetic documents of
    `evals/quality/quick`, pins them as derivation cases and runs the quick
    evaluation with the configured model; its summary and verdict follow the
    smoke table.
  - `memforge eval seed-derivation-cases --spec FILE` pins synced Documents, by
    id or by title within a Source, as cases with a rubric and freezes their
    cohort (maintenance operator only; the self-hosted owner is one). It calls
    the new `POST /api/v1/agent-evaluations/derivation-cases/seed`.
  - `memforge eval offline-run --spec FILE` submits a run, waits for it and
    prints a Markdown summary; it exits 0 on `pass`, 1 on `fail` or `unknown`
    and 2 when the run does not finish in time.
  - Derivation cases gain three checks: every ReadingGroup was read, the
    Memory count is within the rubric's `expected_memory_count`, and the
    Memories hold the rubric's `required_terms`. Run reports gain a `verdict`.
  - A run admitted without `code_revision` or `model` is pinned to the release
    the server runs and its generation model.
  - Cloud: the HANA store's `list_source_derivation_attempts` needs the same
    `source_unit_id` filter.

- Contributors: provider keys for `scripts/provider_smoke.py` and
  `scripts/compose_smoke.sh` can live in `.env.test` (copied from
  `.env.test.example`, ignored by git), which both scripts read on start; a
  variable exported in the shell still takes precedence. `CONTRIBUTING.md` now
  says which of these checks a change needs.

- Creating a memory by hand (MCP `create_memory` or the admin API) works before
  an embedding model is configured, instead of failing with a server error.
  The memory is kept and found by keyword search; near-duplicates are not
  checked until there is an embedding model. When the first embedding model is
  then saved, Settings asks for one rebuild of the vector index, which gives
  those memories their vectors.
  - A `create_memory` request repeated with the same `idempotency_key` answers
    with the memory it created, whether or not an embedding model is configured.
  - Cloud: nothing to do; Cloud always has its embedding model.

- A sync that finishes while some parts of its documents went unread by the
  generation model now says so. A part is unread when the model's answer for it
  is rejected even on its own, or when the part does not fit the model; such a
  part yields no memories, and before, the run still showed as fully synced.
  The admin UI Source drawer notes "N parts of the synced documents could not
  be read by the generation model, so no memories were taken from them", the
  old admin UI lists the count in Last sync details, and `memforge sync` adds
  it to its Done line. A run that reads everything and finds nothing worth
  remembering shows no notice.
  - The count is `SyncState.skipped_reading_group_count`, stored in the new
    `sync_history.skipped_reading_group_count` column (SQLite migration 120,
    default 0) and returned as `sync.skipped_reading_group_count` on each
    Source in `GET /api/v1/sources`. It includes derivations a run resumes from
    an earlier run.
  - Cloud: the HANA workspace store needs the same `SYNC_HISTORY` column,
    written with every sync history row and returned from `get_sync_history`,
    and its completed derivation batch reads must keep each batch's stored
    evidence counts, as the SQLite store does. Existing rows read as 0.

- Contributors: `scripts/compose_smoke.sh` checks a fresh Docker Compose install
  end to end in an isolated project and prints a Markdown table for the PR or
  release notes.

- The vector index status in Settings no longer reports "reindexing" next to
  a rebuild that has already ended. Both now come from one read of the latest
  rebuild, so a rebuild that ends while the status is read shows as running or
  as ended, never both.
  - Cloud: nothing to do; the Cloud admin reads the same view.

- Local Ollama models now answer in the exact JSON shape MemForge asks for.
  MemForge sends the schema as Ollama's own `format`, so Ollama holds the
  answer to it, instead of writing the schema into the prompt, which small
  models such as `qwen3:0.6b` tended to echo back instead of answering.
  - OpenRouter models that support structured outputs get the same: MemForge
    reads what OpenRouter's model list says about the model
    (`structured_outputs`) and asks it for a `json_schema` response format.
    Other OpenRouter models keep the schema in the prompt. OpenRouter can add or
    drop support for a model, so MemForge reads the list again whenever it
    checks the model: at Save and validate, at Check again and when the server
    starts. When the list cannot be reached, it keeps what it read last.
  - The same holds for an Ollama model typed as a LiteLLM route
    (`ollama_chat/...` or `ollama/...`, with the Other provider option or
    `MEMFORGE_ENRICHMENT_MODEL`).
  - Whether a provider natively supports a schema is now decided by what the
    provider says, not by LiteLLM's built-in model list, which does not know
    local Ollama models or most OpenRouter models. OpenAI, Anthropic,
    OpenAI-compatible endpoints and other LiteLLM routes work as before.
  - A provider that rejects the schema request still gets the request again
    with the schema in the prompt, so nothing that works today stops working.
  - Existing installs keep their validated models; nothing needs to be checked
    again, and what the model list says never makes a model need a new check.
    An OpenRouter model saved before picks up structured outputs the next time
    the server starts.
  - The admin API's model listing, candidate and binding responses gain a
    `native_schema` field, and the model bindings table a `native_schema`
    column (migration 119).
  - Cloud: nothing to change. Cloud's SAP AI Core models use the generic
    LiteLLM route provider, whose requests are unchanged, and the
    `native_schema_transport` Cloud sets is still honoured.

- Contributors: `scripts/provider_smoke.py` checks that real providers
  (OpenRouter, OpenAI, Anthropic, a local Ollama, or any LiteLLM route) accept
  what MemForge sends, through the same model listing and capability check as
  "Save and validate", and prints a Markdown table for the PR or release notes.
  It replaces `scripts/smoke_litellm_source_support.py`. A failed model check
  now also records the HTTP status the provider answered with (`status_code`
  in the validation facts).

- Ollama embedding models now work with an Ollama server that requires an API
  key, and a stalled embedding request ends at its timeout. MemForge sends
  embedding requests for the Ollama provider to the server's OpenAI-compatible
  API (`<address>/v1/embeddings`) with the saved key as a Bearer token; LiteLLM's
  own Ollama embedding route sends neither the key nor the timeout
  (https://github.com/BerriAI/litellm/issues/47261).
  - A keyless Ollama server keeps working, and a set embedding dimension is
    still sent. The Ollama server must serve the OpenAI-compatible API, which
    current Ollama releases do.
  - The model, its address and its vector space are unchanged, so an existing
    install keeps its vector index, needs no reindex and keeps its validated
    model. Generation with Ollama and Load models are unchanged.
  - A model typed as a LiteLLM route such as `ollama/nomic-embed-text` (the
    Other provider option, or `MEMFORGE_EMBEDDING_MODEL`) is still sent the way
    LiteLLM sends it, without the key; choose the Ollama provider for a server
    behind authentication.
  - Cloud: nothing to change. Only bindings with the Ollama provider send
    embeddings differently; Cloud's SAP AI Core models use the generic LiteLLM
    route provider, whose requests are unchanged.

- The quickstart no longer offers AWS Bedrock or Google Vertex AI as LiteLLM
  route examples. Their routes need Python packages MemForge does not ship
  (`boto3`, `google-auth`), so they fail before any request is sent.
  - Cloud: nothing to change.

- A generation call the provider refuses is no longer retried as if the
  provider were down. When a generation model answers 400, 401, 403, 404 or
  422, for example after its key is revoked or the model is removed, the
  failed Source Unit is recorded as not retryable, the sync is not retried
  automatically, and relation discovery stops retrying the work, as already
  happened for an embedding model. Timeouts, connection failures, 408, 409,
  429 and 5xx are still retried. After fixing the key or the model, reprocess
  the affected Documents or wait for them to change.
  - Cloud: no code or configuration change. Cloud's SAP AI Core generation
    calls go through the same classification, so one SAP AI Core refuses (for
    example an expired credential or an undeployed model) is no longer retried
    at the run level.

- Memories you write by hand, and corrections you make, now appear in the
  recent-memories listing (`list_recent_memories` and
  `POST /api/v1/memories/recent`) and in searches filtered only by date. They
  were left out because they have no configured Source.
  - Search without an explicit access scope now runs as the local owner and
    includes the owner's private Memories, such as agent-session and hand-made
    ones. Other users' private Memories are still never returned.
  - A new test checks every OSS read surface (search and recent APIs, the
    Memory list, ToolClient, both CLI search commands, the MCP search and
    recent tools, hook context, and the search engine called directly) returns
    all of the owner's Memories and no one else's private ones.
  - Cloud: the search default does not affect Cloud, which always passes its
    own access scope. If Cloud's HANA store mirrors the SQLite
    `list_recent_memory_ids` and `list_ids_by_source_and_time` queries, it
    needs the same change: a support row from `user_memory` or
    `user_correction` counts as current even though no Source row exists for
    it.

- The Sources page warns when the Codex or Claude Code plugin cannot read
  lines of its session transcripts, for example after the client changes its
  transcript format, so a plugin update is not missed.
  - The hook adapter counts the transcript lines it cannot recognize in each
    uploaded window and reports them as the `unrecognized_line` omission in the
    window receipt. Malformed JSON and conversation records whose shape it does
    not understand count; records it skips on purpose, such as session
    metadata, reasoning and injected context, do not. Unrecognized lines are
    still skipped and are not uploaded again.
  - Each window receipt records the uploading `plugin_version`. The Source List
    reports `unrecognized_session_lines` (count, plugin version and when last
    seen) for an agent-session Source, summed over its 50 most recent windows
    from the plugin version that uploaded the latest one, so a window from an
    updated plugin clears it.
  - The Source drawer shows the warning, for example "3 session lines from
    Claude Code were not recognized", with a prompt to update the MemForge
    plugin. Nothing is shown when every line was recognized.
  - Cloud: no HANA schema or configuration change; the count and plugin version
    live in the existing agent-session receipt metadata. The Source List reads
    them through `list_agent_session_receipts(source_id=..., limit=...)`, which
    Cloud's HANA store must serve with parsed `metadata`, `source_kind` and
    `submitted_at`, newest first.

- Claim Extraction no longer fails a Document when the model cites Evidence its
  request cannot bind. On an incremental update only the changed content may be
  a Memory's Primary Evidence, and earlier content may only be Required; a model
  that chose an earlier message as Primary, for example for a new Teams reply
  that agrees with it, failed the Document with `EXTRACTION_EVIDENCE_UNRESOLVED`
  at every sync.
  - A response binds whole or not at all. When any of its candidates names an
    unknown, duplicate or Required-only ref as it is used, the request is sent
    once more with a correction that names each rejected selection, its
    location in the response and the refs it may use. A request still unbound
    is halved, and a ReadingGroup that alone stays unbound is skipped with the
    `extraction_reading_group_unread` diagnostic, like one whose output stays
    invalid; the Document commits with the rest and records no failure. A
    skipped ReadingGroup is read again when its Document changes or is
    reprocessed. Duplicate refs are still never removed on the model's behalf.
  - The extraction prompt (`projection-extraction-v23`) says that a new reply
    that agrees with, answers or decides on earlier content states its own
    stance, so the reply is Primary and the earlier content Required, and that
    only a pure repeat that adds no stance yields no Memory; the rule against
    a signpost as Primary now applies among the refs eligible as Primary. A
    derivation staged under `projection-extraction-v22` is superseded and
    derived again.
  - Extraction results and derivation batch telemetry count the rejected
    selections per reason over every attempt
    (`fragment_selection_rejection_counts`, `rejected_fragment_selection_count`),
    the requests bound after their correction
    (`selection_corrected_request_count`) and the ReadingGroups skipped unbound
    (`selection_skipped_reading_group_count`), summed per Source Unit. Every
    rejected candidate still records an `evidence_admission_outcome` quality
    signal with its reason.
  - The unused selector correction module, the structured client's
    `correct_projection_fragment_selectors` with its
    `ProjectionFragmentSelectorCorrectionResponse`, and the
    `selector_correction_*` and `selector_normalization_*` telemetry are
    removed; stored batch outputs that carry those keys still load. The batch
    runner's `ItemTask` takes the `correction` block its task sends back.
  - Cloud: no HANA schema, storage protocol or configuration change; the new
    contract version arrives with the pin. Cloud's HANA workspace store test
    that expects derivation batch telemetry to keep the `selector_correction_*`
    keys must expect the selection keys above instead.

- A Document that fails no longer holds its Source back, and a release that
  fixes the failure retries it without any action from you.
  - Every Document that fails in a sync, at any step from fetching it to
    committing its Memories, is recorded in a failure ledger with the step, the
    reason, whether the failure may pass on its own, and the release that met
    it. The sync advances its cursor and ends `partial`; a failure of the sync
    itself, such as an authentication error, still keeps the cursor. A Document
    leaves the ledger as soon as it commits, or when it leaves the Source.
  - A failure may pass on its own only when its cause proves it transient: a
    model or provider call that timed out, was rate limited (429), met a server
    error (5xx) or a dropped connection, including a Gene's own HTTP errors; a
    storage error the store classifies as transient; or the loss of the Source
    activity fence. A scheduled sync retries such a failure until it has failed
    at the same step with the same error class in three consecutive runs. It
    leaves any other failure of the same release alone while the provider
    returns the Document unchanged, and processes it again once the provider
    changes it.
  - Clicking **Sync** retries every failed Document of the Source, also for a
    Source collected by the local agent and when the click joins a sync that is
    already pending or running. A retried Document is read again from its
    provider when the Source can ask for one Document by id; otherwise the
    revision that failed is fetched again.
  - At startup a new release retries once each failure an earlier release
    recorded: it queues a reprocess of those Documents for each Source, or a
    sync for a Source that never completed one. A release is the installed
    `memforge-ai` version, joined with `MEMFORGE_DEPLOYMENT_REVISION` (or the
    Cloud Foundry application version) when one is set. A derivation another
    release staged is superseded with `DERIVATION_RELEASE_SUPERSEDED` and
    derived again, and batch outputs and model work journals are keyed by the
    release.
  - Relation discovery work records the release of its last failure, and at
    startup each release re-runs once, through the audited operator re-run, the
    exhausted work an earlier release failed. Work whose lease expires on its
    last attempt ends exhausted with `lease_expired` instead of staying
    `running`.
  - `GET /api/v1/sources` lists each Source's failed Documents under
    `failed_items`, and `GET /api/v1/health` counts them under `failed_items`.
    The admin UI V2 Sources page shows "Some items failed (N)" with each item's
    step and reason, and "Retrying N items after upgrade" while a new release
    retries them. The admin UI V1 last sync details count the items that still
    fail, and its "Refresh source" action is named "Rebuild memories".
  - Cloud: the HANA stores implement the failure ledger
    (`upsert_source_item_failure` and `delete_source_item_failures`, both fenced
    by an optional `source_activity`, and `list_source_item_failures`) over a
    `source_item_failures` table keyed by Source and Document and deleted with
    its Source; a nullable `release` column on Source derivations, read into
    `SourceDerivationAttempt.release`; `is_transient_error(error)`, true for the
    `hdbcli` errors of a lock wait timeout, a deadlock rollback and a lost
    connection, with samples in `ContractAdapters.transient_errors`; and
    `retry_failed_items` and `rerun_retry_failed_items` columns (`INTEGER NOT
    NULL DEFAULT 0`) on `source_sync_runs`, set by a `manual` or `force` enqueue
    or `retry_failed_items=True`, set on a pending run a request coalesces into,
    set as `rerun_retry_failed_items` with `rerun_requested` on a running run
    without it, copied into the rerun successor and into a failed run that
    returns to pending, and read into `SourceSyncRun`. The HANA relational store
    adds a nullable `release` column to `relation_discovery_work`, written by
    `fail_relation_discovery_work(..., release=...)`, cleared on completion and
    re-run, and filtered by `RelationDiscoveryWorkSelection.release_other_than`
    as `release IS NULL OR release <> ?`; it implements
    `exhaust_expired_relation_discovery_work`, counts such work as ready in
    `has_ready_relation_discovery_work`, and takes `reason` in
    `rerun_relation_discovery_work`. Core-integration `WorkspaceDatabase` and
    `WORKSPACE_SYNC_RUNTIME_METHODS` add `upsert_source_item_failure`,
    `list_source_item_failures`, `delete_source_item_failures`,
    `is_transient_error`, `exhaust_expired_relation_discovery_work` and
    `rerun_relation_discovery_work`. Cloud's runtime provider's
    `run_source_sync` accepts `failed_item_retry`, its sync service's
    `enqueue_source` accepts `retry_failed_items`, its local-agent job enqueuer
    keeps a job payload's `retry_failed_items` when a scheduled job coalesces
    into it and does not coalesce a job that sets it into a leased job without
    it, its source admin reader implements `list_source_item_failures`, and its
    health report fills `failed_items`. Each workspace has its own store, so the
    Cloud worker's workspace loop calls `SyncService.enqueue_release_recovery()`
    and `rerun_relation_discovery_failed_by_another_release(store)` once per
    workspace for each release it runs (for example by tracking the last release
    per workspace id); the per-workspace admin app also calls both when its
    lifespan starts.
- Upgrading an installation keeps its data recoverable. When a release has
  schema migrations the database has not applied yet, it first saves a
  recovery point, a copy of the database made with the SQLite backup API and
  a copy of the Chroma vector index, under `backups/` in the data directory
  (`/data/backups` in the `memforge-data` volume). It keeps the newest
  recovery point of each of the three most recent schema versions it upgraded
  from, and the newest three taken before a restore. Each migration then
  applies in one transaction together with the record that it was applied,
  so a migration that fails, or a process stopped part-way, leaves the
  database at the previous migration, and the next start applies it again
  and keeps the recovery point taken before the upgrade began; the error
  names the migration and that recovery point. A release refuses to start on
  a database a newer release has upgraded, changes nothing, and names the
  recovery point to restore. The new `memforge backup list` and
  `memforge backup restore ID` commands list and restore recovery points; a
  restore keeps the data it replaces as another recovery point, and MemForge
  refuses to start after a restore that stopped part-way until it is run
  again. Upgrading and restoring are refused while another MemForge process
  has the data open. Recovery points do not include source documents, which
  no migration changes. The quickstart's new Upgrading section gives the
  Docker commands; an installation started with 0.2.0b1 gets its first
  recovery point when the first release that changes the schema starts.
  Cloud: no change needed; Cloud keeps its data in HANA with its own schema
  handling, and its SQLite store opens databases without recovery points,
  with each migration in its own transaction.

- A Source Unit revision whose re-read Support chooses as its Primary an
  unchanged Observation that a partial sync did not return now commits, with
  its Evidence read from that Observation as the Unit last stored it. It
  previously failed with a `KeyError` while building the lifecycle plan, which
  stopped every later sync of the Source. Cloud: no change needed.

- An embedding request that fails with a retryable error after the embedding
  deadline has already passed now ends the call as a deadline overrun
  (`EmbeddingDeadlineExceeded`, reported as `timeout`), the same as when the
  deadline interrupts the request. It previously depended on event-loop timing
  and could end as a provider error instead. Cloud: no change needed.

## 0.2.0b1 - 2026-10-16

The first beta. MemForge runs for one user on one machine from the Docker
Compose stack, with the models chosen in Settings. APIs, storage formats and
plugin packaging may still change before 0.2.0, and upgrading an installation
or backing up and restoring its data is not supported yet: start a beta
installation on a fresh `memforge-data` volume, and if an upgrade fails,
start again on a fresh volume and sync the sources again. The package is `memforge-ai` 0.2.0b1; the agent
plugins are versioned `0.2.0-beta.1`, the same version in semver spelling.

- Settings is easier to follow while models are set up. A model id typed,
  pasted or filled in by the browser that exactly matches a model from Load
  models now counts as picking it, so the token limits the provider's list
  gives fill the empty limit fields, as a click on the suggestion did. When
  validation stops at the token limits step and the provider's list can give
  them (OpenRouter and OpenAI-compatible endpoints), the step offers **Fill
  from the model list**, and says plainly when the list does not include the
  model or gives no limits for it; the providers expose this as the new
  `lists_token_limits` field of `GET /api/v1/model-settings/providers`. A card
  with no model in use shows only the Not set up badge and one line on what
  MemForge does without one, instead of repeating it in an In use panel and in
  the notice of a saved edit. An empty search index still reads "Vector search
  is on." and adds that memories become searchable once a source is synced.
  The provider "LiteLLM route (advanced)" is renamed "Other provider (LiteLLM
  route)" and says under the Provider field which providers it is for; a
  LiteLLM Proxy, which speaks the OpenAI API, belongs under OpenAI-compatible
  endpoint. A generation model LiteLLM has no metadata for, whose token limits
  are all configured, no longer logs `request_budget_metadata_unavailable` as a
  warning on every request: it is logged at debug level, and a warning remains
  for a limit that nothing configures. Cloud: no change needed; no Cloud test
  or UI matches the old provider label. Cloud's AI Core models are bindings of
  that provider, so wherever Cloud's Settings shows them the provider reads
  "Other provider (LiteLLM route)".
- `memforge search` and `memforge memory search` find your own private
  memories, which include every memory from a Claude Code or Codex session and
  every memory created by hand, so they no longer come back empty on an
  installation that holds only those. Memory reads through the API include
  the caller's own private memories unless the request sets `include_private`
  to `false`: `POST /api/v1/memories/search`, `POST /api/v1/memories/recent`,
  `GET /api/v1/memories` and `ToolClient.search` now default it to `true`.
  Other users' private memories are never returned. In a workspace of one, the
  admin UIs no longer show access choices: a new Source is created private
  without asking who can use it, and the "Only me" labels, the Access column
  and detail row, the Change access action and the access-change banners are
  left out. Cloud: `include_private` defaults to `true` on the same routes, so
  a client that leaves it out now also gets the caller's own private memories;
  other members' private memories stay hidden. The admin UIs show the access
  controls only for a workspace target marked `shared`, so the Cloud extension
  should pass `shared: true` to `setTarget` and add `shared` to its mirror of
  `WorkspaceTarget`; a target that leaves `shared` out keeps the access
  controls.
- `.env.example` leaves the models empty, so a new installation starts with no
  model and you choose one in Settings. It no longer names
  `claude-sonnet-4-20250514` and `text-embedding-3-small`; set
  `MEMFORGE_ENRICHMENT_MODEL` and `MEMFORGE_EMBEDDING_MODEL` to configure the
  models in `.env` instead. An existing `.env` is not affected.
- Closed-choice model calls no longer fail with `output_truncated` because the
  model spent their small output budget reasoning before it answered. These
  calls (relation classification, cross-document relations, change impact,
  candidate admission, agent-session authority, entity adjudication,
  reranking, the offline semantic judge and the generation model check) are
  sized for a short JSON answer, and a model that reasons by default, such as
  `anthropic/claude-haiku-5.5` on OpenRouter, could use all of it and return
  no answer or half of one. They now ask the model to answer directly, as each
  provider allows: OpenRouter is sent `reasoning: {"enabled": false}`, OpenAI
  the lowest `reasoning_effort` LiteLLM knows the model takes, and Ollama
  `think: false` (`"low"` for gpt-oss, which cannot stop thinking). Anthropic
  models do not think unless asked, and OpenAI-compatible endpoints and
  LiteLLM routes are sent nothing. A provider whose rejection names the
  parameter is asked again without it, and is not sent it for that model
  again; other rejections are handled as before. Claim extraction, claim
  revision, Support assessment and knowledge patches keep the provider's
  default. When any call is still cut off at its output limit after reasoning
  used most of it, it is sent once more with the limit raised by that
  reasoning, within the model's output limit and context window, and a repair
  or JSON-text attempt of the same call keeps that limit; the call's
  telemetry, the Source Unit LLM summary (`reasoning_budget_retries`) and the
  `reasoning_budget_recovered` quality signal show it. Cloud: no change
  needed. Cloud's AI Core models are bindings of the LiteLLM route provider,
  so their requests gain no parameter; only the once-raised `max_tokens` can
  differ, and only when an AI Core reply reports reasoning that used up its
  budget.
- Settings → Load models ends a listing request at five seconds, counted from
  sending it to the last byte of the answer. Before, the five seconds applied
  to each wait on its own, so an endpoint that sent a byte now and then could
  keep Load models waiting indefinitely. LiteLLM requests made through the
  HTTP clients MemForge sets up, on the shared OpenAI-compatible client and on
  the clients built for `openrouter/`, `anthropic/` and every other provider
  route, now also end at their timeout as a whole and fail as a timeout that
  is retried like any other; before, a provider that kept the connection busy
  could hold a direct LiteLLM call open far past its timeout. Generation,
  embedding and capability checks were already bounded by their own
  deadlines and behave as before. Cloud: no change needed; no request
  parameter is added, so the `sap/` routes of the `litellm` provider receive
  the same requests as before, and their requests through LiteLLM's SAP
  client end at the timeout MemForge already sends them with.
- Settings → Load models lists the models of the kind being set up. OpenRouter
  embedding models come from its embedding list (`/embeddings/models`) instead
  of its chat models; OpenAI's one list is split by model id; Ollama is asked
  for its local models (`/api/tags`) and each is sorted by its capabilities,
  family or name. An OpenAI-compatible endpoint does not say which models
  embed, so all of them are offered and the message says so. A list with no
  model of the kind says that, and that the model id can be typed, instead of
  offering the others. Load models fills the model field only when exactly one
  model of the kind is listed, and typing in the field narrows the list, also
  once it holds a whole model id. A listed model carries the token limits the
  listing gives (OpenRouter's `context_length` and
  `top_provider.max_completion_tokens`); picking it fills empty token limit
  fields of the generation model and says where they came from; typing another
  model id clears limits filled in this way. A validation that fails only for
  want of token limits shows the step that is left in amber, on Settings and
  in the notice on other pages, and a passed validation is reported once, by
  the model in use. A
  base URL pasted with a request path at its end (`/chat/completions`,
  `/embeddings`, `/models`, `/embeddings/models`, Ollama's `/api/...` or
  `/v1`, Anthropic's `/v1/messages`) is stored without it. Optional fields
  are marked "(optional)", and a base URL with a provider default names it.
  Cloud: no change needed; the discovery response only gains fields
  (`kind_filtered`, and token limits on each model), and Cloud's AI Core
  models are LiteLLM routes with no base URL, which the new base URL reading
  leaves alone.
- A fresh install no longer reports its health as degraded after the first
  sync. A Memory created by a sync wrote the entity names as the extraction
  spelled them into its keyword search row, while the entities it was
  linked to were stored under their canonical names, so the index
  consistency check counted one `fts_entities_mismatch` for every Memory with
  an entity. The keyword search row of a new Memory now holds the canonical
  names of the entities it is linked to, written in the same transaction as
  the links. Linking an entity to a Memory and merging two entities update
  the row in the same way. Keyword search used the extraction's spelling
  until now, so a Memory whose entity was resolved to a differently named
  entity or alias was not found by that entity's name. Upgrading rebuilds
  every keyword search row once, which clears the reported issues;
  `memforge maintenance repair-indexes` clears them too. Cloud: no change.
  The HANA store has no entity keyword row and searches Memory content only.
  The local-dev SQLite store uses this package's database, so it gets the
  fix and the one-time rebuild with the pin.
- A Markdown document with an HTML part that has no text to cite, such as a
  row of badges, an image or `<picture>`, a centred `<div>` wrapper, a
  `</details>` closer or a lone `<br>`, no longer makes every Memory taken
  from that document unreadable. Before, opening such a Memory in the admin
  UI or through MCP `get_memory` failed with "Pinned text Evidence integrity
  verification failed", every relation discovery job that read it failed,
  and an edit to the document ended its next sync without extracting
  anything. Most GitHub READMEs have such parts. Evidence still has to match
  its source exactly. Only a document that cannot be read at all, such as one
  in an unsupported format, still fails this check. Memories already stored
  read correctly without a re-sync. Relation discovery work that already used
  up its retries can be queued again from the relation discovery work admin
  action (`POST /api/v1/relation-discovery/work/rerun` with state
  `exhausted`). Cloud: arrives with the pin; the HANA store uses the same
  check, so it needs no change.
- Logs written where stderr is not a terminal, such as a Docker container or a
  service manager, are one plain line per record with the timestamp, level and
  logger, and tracebacks print in full. Before, boxed tracebacks wrapped into
  an 80-column frame that cut every line apart. A terminal keeps the formatted
  output. Cloud: arrives with the pin; Cloud's CF logs become plain lines.

- A sync run that skipped its absence check records which step failed instead
  of the provider's error text: `sync.absence_check_skipped_reason` is
  `scope_listing_failed` or `absence_confirmation_failed`. The provider's error
  stays in the log. A run recorded with error text reads as
  `scope_listing_failed`. The admin UI V2 Source drawer explains the step in
  plain words and says that nothing was removed and the next sync checks
  again. Cloud: arrives with the pin; the HANA store keeps the value as text,
  so no schema change.

- A search question that names a product written in camel case, such as
  "How do I configure workspaces in MemForge?" or "GitHub 的同步是怎么工作的？",
  is ranked as an open question (`general_hybrid`) instead of a lookup of one
  item (`known_item`), so the Memory that answers it is no longer pushed below
  documents that only share the product name. Without a requested intent, a
  search is a `known_item` lookup when it contains an issue key, `#123`, a
  UUID, a URL, a snake_case name, a path or a dotted name. A path is relative
  to the home or current directory, has two segments under the root, has
  three or more segments, or ends in a file name (`src/App.tsx`, but not a
  pair of capitalised names such as `React/Node.js`). A dotted
  name is a module, attribute or file name such as `litellm.aembedding` or
  `docker-compose.yml`. A search is also a `known_item` lookup when it is
  made only of camel-case names, quoted phrases, and slash or dotted terms
  that read as prose or a product as readily as a name (`and/or`, `TCP/IP`,
  `pkg/errors`, `e.g.`, `Node.js`, `Cargo.toml`), or when a source title or
  alias equals the whole query. Inside a question these terms do not make it
  a lookup. A code symbol with other words, such as
  "PersistCutOffBlockerHintsCommand blocker hint", is ranked as an open
  question and its Memory still comes first through the exact metadata match.
  The words searched as exact metadata anchors are unchanged, and a requested
  intent is honoured as before. Cloud: arrives with the pin; Cloud search
  runs the OSS `SearchEngine`, which resolves the intent and hands the HANA
  keyword adapter the same metadata query plan as before.
- The Admin API answers browser requests only from the Admin UI dev server on
  loopback (`http://localhost:5174,http://127.0.0.1:5174`) by default, instead
  of from every origin. The loopback bind keeps other machines out but not a
  web page open in the operator's own browser, which could otherwise read and
  change a local install that has no login. Both admin UIs reach the API
  through their own dev-server proxy and are unaffected, as are the CLI, the
  local-agent daemon and MCP clients. Set `MEMFORGE_CORS_ORIGINS` to answer
  other origins; `*` answers every origin and never carries credentials.
  Cloud: arrives with the pin; the Cloud UI calls the mounted Admin API from
  the same origin, so no environment change.
- The Memory vector index belongs to one embedding model, and changing that
  model rebuilds it. An Embedding Profile is the identity of a vector space:
  the provider, the LiteLLM route, the endpoint, the model id, the requested
  output dimensions and the vector length the model answered with; its
  fingerprint is a hash of those six fields. The endpoint enters it in one
  spelling (scheme and host in lower case, no default port, no trailing
  slash), so writing the base URL of the same server another way needs no
  reindex. SQLite migration 114 adds
  `vector_indexes`, which records each physical index once: its fingerprint,
  collection name, state (`building`, `active`, `failed`, `retired`), whether
  it was adopted, vector length, counts, times, error code and the process
  that runs its rebuild; at most one is `active`. A new index is the Chroma
  collection named `memories-` and the first 32 characters of its
  fingerprint.
  Every reader and writer opens the index through
  `memforge.retrieval.vector_index.open_memory_vector_index(db, chroma_path)`,
  at each operation, so all of them follow a rebuilt index the moment it is
  in use; `memforge.retrieval.embeddings.get_chroma_collection` is removed.
  An Embedding Runtime knows the fingerprint it embeds in once its model has
  answered, and its vectors reach an index only when the fingerprints are
  equal: with another model, search answers from its keyword and entity
  channels without an error, a write that needs a new vector fails as
  retryable and leaves nothing behind (`EmbeddingIndexMismatch`, answered by
  the API as 409 `reindex_required`, or 409 `vector_index_missing` when no
  index is in use), pending vector work in the outbox waits without using up
  an attempt, and index repair reports the Memory as unrepaired. A cache of vectors belongs to one
  runtime and is checked against the index like a fresh vector. Validating an
  embedding candidate whose fingerprint is the index's makes it active at
  once. Any other candidate that passes stays the candidate, `ready`, with
  `waits_for_reindex: true`, and `POST /{kind}/candidate/validate` answers
  `promoted: false, reindex_required: true`. A reindex is started with
  `POST /api/v1/model-settings/vector-index/reindex` (202) and runs beside the
  server: it embeds every searchable Memory from SQLite into a new index,
  verifies that the vector count equals the count of searchable Memories and
  that every vector has the profile's length and finite values, then in one
  transaction makes the new index the one in use and the candidate the active
  binding, and afterwards deletes the previous collection, together with
  stored vectors in the `memories` collection that no index adopted. Deleting
  a collection also removes the directory of its vector segment, which
  ChromaDB leaves on disk. A
  deletion that fails leaves the switch in place and is made up for when the
  server next starts or a reindex is next started. Search reads the
  previous index until that transaction. A reindex that fails changes nothing
  in use, drops what it built, records an error code and can be started again.
  A running reindex holds a 60-second lease that it renews every 15 seconds
  whatever it is doing, and records the host, process ID and a per-process
  token of its runner. One whose process was killed is recorded as
  `interrupted` when a server starts, or a reindex is started, on the same
  host and finds that process gone, and otherwise once the lease has run out;
  a server brings the index into order at start before it checks any model.
  `memforge maintenance reindex --apply` records `interrupted` at once on
  SIGTERM, as on Ctrl-C, and exits with status 1.
  `GET /api/v1/model-settings/vector-index` and the new `vector_index`
  field of `GET /api/v1/model-settings` report `status` (`ready`,
  `reindex_required`, `reindexing`, `awaiting_validation`, `not_configured`,
  `not_managed`), `serves_search` (whether search reads the index in use,
  which stays false through a reindex that started while one was required),
  `reindex_target` (`candidate`, `active` or null; `active` also for a model
  in use that has not passed a check as it stands, when its settings are not
  the index's, the index was adopted, or vectors are stored and no index is
  in use, in which case starting the reindex checks the model first), the index
  in use and the latest reindex with its progress; `GET /api/v1/health` gains
  `models.embedding.vector_index` and `models.embedding.reindex_required`. A
  reindex is refused with 409 `sync_in_progress` while any Source activity
  lease is current, `reindex_in_progress` while another runs,
  `reindex_not_required` when the index fits the model and was not adopted,
  and `embedding_not_validated` when the model has not passed a check. The
  first three are decided in the transaction that admits the reindex, so a
  request made before another reindex switched cannot build over its result.
  While
  one runs, no Source activity lease is granted, so a Source sync does not
  start and a queued run stays pending; index repair and the worker's idle
  vector and relation work do not run; the embedding candidate cannot be
  saved, validated or discarded; and a write that inserts or deletes a Memory
  row, or changes its type, content, visibility, owner, project, repository
  or status, is refused by database triggers with the code
  `reindex_in_progress`, which the API answers as 409 with that code in
  `detail.error`. A write refused this way is rolled back on its connection,
  and the daily expiry job leaves expired Memories to its next run. The
  routes that create, correct, retire, purge or decide on
  Memories and that delete a project check for a running reindex first.
  `memforge maintenance reindex --check` reports the index and what a rebuild
  would do, and `--apply` runs the same rebuild in the command's own process,
  with exit status 1 when it is refused or fails. An installation with no
  recorded index gets one the first time its embedding model answers, at
  validation or at the first embedding request: stored vectors in the
  `memories` collection whose sampled length is the model's are adopted where
  they are (`adopted: true`, the model they came from assumed), an
  installation with no vectors gets a new empty index, and one whose stored
  vectors have another length has no index in use and reports
  `reindex_required` until it is rebuilt. Stored vectors are adopted for a
  candidate only when it names the model the active binding names. Changing
  `MEMFORGE_EMBEDDING_MODEL` or `MEMFORGE_EMBEDDING_BASE_URL` for an
  installation with an index therefore requires a reindex. A profile whose
  model the similarity thresholds were not tuned on reads
  `thresholds_calibrated: false`; only OpenAI `text-embedding-3-small` at
  1536 dimensions reads true, and the thresholds themselves are unchanged. A
  binding's digest now covers the key it refers to and its token limits, so
  replacing or removing the saved key of a ready binding makes it
  `needs_revalidation`. A generation binding may carry `token_limits`
  (`context_window_tokens`, `max_output_tokens`), accepted by
  `PUT /{kind}/candidate` and returned with every binding; the runtime sizes
  requests with them and takes the `MEMFORGE_LLM_*` limits for whatever the
  binding leaves open. A generation model whose limits neither LiteLLM, the
  binding nor the environment knows now fails validation with
  `token_limits_required`, after its request has been answered; an active
  binding of such a model reads `needs_revalidation` or `not_validated` until
  the limits are given. In Python, `VectorStore` gains
  `async def index_fingerprint() -> str | None`; `EmbeddingSettings` takes
  `identity=` and `EmbeddingRuntime` has `fingerprint`;
  `memforge.llm.embedding.embed_documents_for_index` and
  `embed_query_for_index` embed for an index under the fingerprint rule;
  `ResolvedModels` carries `embedding_identity`; `MemoryStore.collection` is
  replaced by `await MemoryStore.index_collection()`; a runtime provider that
  has this service manage its index implements
  `memory_vector_indexes(db, config)`, as `DefaultRuntimeProvider` does.
  Cloud: the vector store contract gains exactly one method,
  `async def index_fingerprint(self) -> str | None`, which returns the
  fingerprint recorded for the whole index, or `None` for an index that
  records none. `MemoryStore`, `SearchEngine` and index repair call it before
  they embed and refuse vectors of another fingerprint; with `None` no check
  is made, so a store that returns `None` behaves as before. A store without
  the method fails every Memory write that needs a vector with
  `AttributeError`, while search catches the error, logs
  `Embedding computation failed for query` for each query and answers without
  vector hits, so the method has to be in place before this version is taken
  up. The HANA vector
  store has to add the method and decide where it records the fingerprint of
  its table; to have it checked, build the Embedding Runtime with
  `EmbeddingSettings(invocation=..., identity=embedding_identity(binding))`
  and record `identity.profile(dimension).fingerprint`. A provider that does
  not implement `memory_vector_indexes` gets no `vector_indexes` reads or
  writes on its workspace store, no rebuild, no write guard, and
  `vector_index.status: "not_managed"`; `POST .../vector-index/reindex`
  answers 405 `reindex_not_available`. For such a provider, a store whose
  fingerprint is not the runtime's shows in shared code only as refused
  writes (409 `reindex_required`) and as one warning per search engine,
  `Vector search is off: the vector index was built with another embedding
  model`; health reads `not_managed` and `reindex_required: false`, so the
  deployment has to report the mismatch by its own means.
  `DefaultRuntimeProvider` implements
  it for a `memforge.storage.database.Database` and returns `None` for any
  other store, so a provider that subclasses it stays unmanaged as long as
  its workspace store is not a `Database`. `SourceSyncWorker`,
  `ModelSettings` and the admin routes call the new storage methods only for
  a managed index. An environment generation model
  LiteLLM has no limits for (a `sap/...` route without an entry in its
  registry) now needs `MEMFORGE_LLM_MAX_INPUT_TOKENS`,
  `MEMFORGE_LLM_CONTEXT_WINDOW_TOKENS` and `MEMFORGE_LLM_MAX_OUTPUT_TOKENS`
  to read `ready` in `GET /api/v1/health`; it is used either way. No HANA
  schema change is required by this release.
- Model settings are two bindings, one for the generation model and one for
  the embedding model, and replace the single LLM configuration. A binding
  names a provider descriptor (`openai`, `anthropic`, `openrouter`, `ollama`,
  `openai_compatible`, or `litellm` for a full `provider/model` route), an
  endpoint, an auth strategy (`api_key`, `none`, or `environment` for
  credentials LiteLLM reads from the process environment), a model id and the
  LiteLLM route it compiles to. Each kind has an active binding, which the
  runtime uses, and a candidate, which Settings edits. Saving writes the
  candidate; validating it makes one real request through the runtime's own
  client (generation must return a small strict nested object under the
  runtime's structured-output policy; embedding must return one valid vector
  per input for two inputs, twice, with one stable dimension) and, when that
  passes, makes the candidate the active binding. A binding is
  `not_validated`, `ready` or `needs_revalidation`; only the latest check is
  kept, on the binding, with its time, observed facts (latency, dimension,
  the model id the provider reported, whether the schema fell back to JSON
  text, whether the model's token limits are known) and an error code, never
  a prompt or a response. A ready binding needs revalidation when its
  endpoint, provider, route, auth strategy, model, requested dimensions or
  the installed LiteLLM version changes, or when a later check of it fails;
  the active binding stays in use whatever its state. A starting server
  checks every active binding that is not ready in the background, without
  delaying its start. One check of a binding runs at a time: a request for a
  check made while one of the same binding is running waits for it and takes
  its outcome, so simultaneous requests make one model call. When a candidate
  becomes active, the search engine and the embedded sync worker
  (`SourceSyncWorker.models_changed()`) build their model clients again, so
  the replaced model, endpoint and key are no longer used; a Source sync that
  is running finishes with the runtime it started with. The
  `/api/v1/llm-config` routes are removed;
  `/api/v1/model-settings` replaces them: `GET /providers`, `GET` (both
  kinds, never a key), `PUT` and `DELETE /{kind}/candidate`,
  `POST /{kind}/candidate/validate`, `POST /{kind}/active/validate` and
  `POST /{kind}/discover-models`, with machine-readable error codes in
  `detail.error`. A request that does not fit a route's schema is answered
  422 with the field and the problem of each error and without the value that
  was sent, so a submitted key is never repeated; `provider` and `model_id`
  may be left out of the body and are then refused as `unknown_provider` and
  `model_required`. An endpoint that carries a user name or a password is
  refused as `endpoint_invalid`, on saving and on model discovery. A
  candidate with the `environment` strategy is accepted only for the default
  endpoint of its provider, and for no endpoint with `litellm`
  (`environment_endpoint_not_allowed` otherwise), so saving a base URL cannot
  send a credential of the process environment to another server; the
  `MEMFORGE_*` variables may still pair a route with a base URL. Model
  discovery does not follow redirects, and a redirect to another origin fails
  the listing. A listing request carries an API key or nothing, never a
  credential of the process environment: an endpoint that refuses it under
  the `environment` strategy is reported as `listing_needs_api_key`, with a
  message that says to enter the model, and as `api_key_required` otherwise. An embedding binding's `dimensions` is sent to every model of
  an `openai/` route, whatever the model is called; on a route LiteLLM does
  not send it for, the check fails with `parameter_not_supported_by_route`
  before any request. `GET /api/v1/health` gains `models.generation` and
  `models.embedding`, each `{configured, state}`, so a client can tell that no
  generation model is ready. An API key entered in Settings is stored
  encrypted with the source-secret key in the new `model_secrets` table and
  the binding holds a reference to it; reads return where the key is kept and
  its last four characters. A saved key is bound to the origin (scheme, host,
  port) it was saved for: saving a candidate for another origin, or listing
  the models of another origin, needs the key again. The `none` strategy
  sends the placeholder key `no-api-key` with every model request and never a
  key from the environment, and a keyless binding counts as configured, so
  `memforge maintenance repair-indexes` embeds through a keyless endpoint.
  That command no longer fails with `AttributeError` after printing its
  table when issues remain. LiteLLM's own lookup of an Ollama model's limits
  still goes to the server `OLLAMA_API_BASE` names (`http://localhost:11434`
  when unset) with `OLLAMA_API_KEY`; it is no longer sent to the endpoint of
  an Ollama request. SQLite migration 113 adds `model_bindings` and
  `model_secrets`. An existing `llm_config` row becomes active bindings in
  state `not_validated` when a process opens the database, with the
  configuration that process runs with: a plain-text key is encrypted, an
  empty column takes the value of that configuration (a file given with
  `--config` and an `AppConfig` a host built included), and a stored model
  with no key anywhere refers to its `MEMFORGE_*_API_KEY` variable and is
  used once that is set. The table is then dropped, overwriting the pages
  that held the keys, and the write-ahead log is emptied as soon as no other
  connection is reading the database. Stored settings that no binding accepts
  are dropped with a warning in the log; a stored key that cannot be
  encrypted stops the process with the table left as it is. `memforge api`
  and every `memforge` command do this; a host that opens a `Database`
  itself calls `await db.translate_llm_config(config.llm)` after `connect()`,
  and reading a model binding before that raises. `MEMFORGE_DECISION_MODEL` is a model id
  of the generation model's provider or the full route of one
  (`anthropic/claude-haiku-4` beside a bare Anthropic generation model). In a
  writable installation a stored binding wins and the
  `MEMFORGE_ENRICHMENT_*` and `MEMFORGE_EMBEDDING_*` variables describe a kind
  with nothing stored, read as before: a bare generation model is an
  Anthropic model, `provider/model` is a route, and a bare embedding model
  is a model id of the OpenAI-compatible endpoint at its base URL (the
  `openai` descriptor when that is `api.openai.com`). A bare Anthropic model
  with no key and an OpenAI embedding model with no key are not configured.
  The runtime names the generation model by its route, so a bare
  `MEMFORGE_ENRICHMENT_MODEL=claude-x` is now `anthropic/claude-x` in work
  identities and recorded model names, as it would be after a change of
  model. `memforge config show` prints the model each kind uses, its source
  and its state. In Python,
  `memforge.runtime.resolve_models(db, config)` returns `ResolvedModels`
  (`generation` and `embedding` as `ResolvedInvocation`, or `None`) and
  replaces `get_effective_llm_config` and `EffectiveLlmConfig`;
  `RuntimeProvider.build_structured_llm_client` takes that `ResolvedModels`;
  `build_embedding_runtime` takes a `ResolvedInvocation | None`;
  `EmbeddingSettings` takes `invocation=` in place of `model`, `base_url` and
  `api_key`; `StructuredLlmConfig.for_invocation(...)` builds a client
  configuration from an invocation; `Database.get_llm_config` and
  `set_llm_config`, `is_litellm_provider_model` and `litellm_embedding_route`
  are removed. Cloud: with `MEMFORGE_LLM_CONFIG_WRITABLE=false` every
  resolution compiles the bindings in memory from the environment with
  `config_source: "deployment"` and neither reads nor writes the workspace
  store, so the HANA store needs no new table or method and its
  `get_llm_config` and `set_llm_config` are no longer called; a `sap/...`
  model resolves to the `litellm` descriptor with the `environment` strategy
  (or `api_key` when `MEMFORGE_*_API_KEY` is set). Change
  the runtime provider's `build_structured_llm_client(llm, ...)` to take
  `ResolvedModels` (`models.generation.route`, `.api_base`, `.api_key`,
  `models.decision_route`; return `None` when `models.generation` is `None`),
  build embedding runtimes with `build_embedding_runtime(models.embedding,
  config)` or `EmbeddingSettings(invocation=...)`, replace
  `get_effective_llm_config` with `resolve_models`, and point any client of
  `/api/v1/llm-config` at `/api/v1/model-settings`. Each API process checks
  its environment bindings once at start with one generation request and four
  embedding inputs, keeps the outcome in memory, and reports it in
  `GET /api/v1/health`; confirm that this start-up request is acceptable on
  AI Core. No HANA schema change.
- The admin UI V2 Settings page shows the search index, which is the Memory
  vector index, on the card of the embedding model: the model that built it,
  the number of memories it holds, whether vector search is on, and the marks
  **Carried over** for an adopted index and **Thresholds not calibrated** for
  a model the similarity thresholds were not tuned on. An embedding edit that
  passes validation and waits for the index says so, and the action of its
  form becomes **Rebuild search index**. An installation whose index does not
  fit its embedding model, or was adopted, starts the rebuild from the index
  itself, also with `MEMFORGE_LLM_CONFIG_WRITABLE=false`. While a rebuild
  runs, the page reads the model settings every two seconds, shows how many
  memories are embedded and that search keeps working on the current index,
  or, for a rebuild that started while vector search was off, that search
  ranks by matching words until the index is ready, and holds the form of the
  embedding model. An index that does not fit a model of the same name says
  what differs: the base URL, the vector size asked for or the size returned.
  A rebuild that fails shows its
  reason and **Run the rebuild again**; one that succeeds shows the new model
  in use. A deployment whose index reads `not_managed` sees none of this.
  While vector search is off because a rebuild is required, and through the
  rebuild that follows, every other product page shows a notice with a link
  to Settings. A change that is
  refused with `reindex_in_progress` is explained in the same words on every
  page, and a Review decision refused for the state of the index keeps the
  Review as it is instead of offering a reload; `ApiError` carries the code of
  a coded refusal as `code`. The card of the generation model has **Context
  window** and **Maximum output** fields for a model that states token limits
  or whose validation failed with `token_limits_required`. The V1 admin UI's
  Settings page says in one line whether the search index is ready, being
  rebuilt or has to be rebuilt.
- The admin UI V2 Settings page has one card for the generation model and one
  for the embedding model. Each shows the model in use with its state (Ready,
  Not validated, Needs revalidation), what its latest validation found and a
  **Validate again** action, and below it a form: a provider choice, then the
  fields that provider needs (base URL, the way of authenticating, API key,
  model with **Load models** where the provider lists them, organization for
  OpenAI, output dimensions for an embedding model). **Save and validate**
  saves the edit and validates it in one step; an edit that fails stays in the
  form with the reason, and the page names the model that is still in use.
  Moving the base URL to another origin asks for the API key again before
  anything is sent. The form sends the base URL it shows: for a model that
  pairs a base URL with credentials from the server environment, as the
  `MEMFORGE_*` variables can, it says so and saves nothing until the user
  removes the base URL or chooses another way of authenticating. For a model
  whose key the server does not have, the key field names the unset
  `MEMFORGE_*_API_KEY` variable and asks for a key. With `MEMFORGE_LLM_CONFIG_WRITABLE=false` the page shows
  the models without a form. Until a generation model is ready, every other
  product page shows a notice with a link to Settings; an extension that
  redirects Settings (`reservedRouteRedirects`) gets the link to its own page,
  or no link for a user its `visibleWhen` hides the page from. The V1 admin
  UI's Settings page shows the two models in use (provider, endpoint, model,
  state) and links to the V2 page for changes.
- Embedding goes through the LiteLLM SDK only, behind an Embedding Runtime
  (`memforge.llm.embedding.EmbeddingRuntime`, with `embed_query` and
  `embed_documents`). An embedding model name whose first segment is a
  provider LiteLLM knows (`ollama/nomic-embed-text`, `sap/...`) is used as
  that route. Any other name, including one with a slash such as
  `BAAI/bge-m3`, is a model id sent on the `openai/<model>` route to the
  configured base URL, and never to an endpoint named by the environment. The
  direct HTTP embedding call, `embed_texts` and `EmbeddingCache` are removed.
  The runtime retries timeouts, connection failures, 408, 409, 429 and 5xx
  (three retries by default, pausing 2, 4 and 8 seconds, or longer when the
  provider sends `Retry-After`) inside one 60 second deadline for the whole
  call; a pause that would outlast the deadline ends the call at once. It
  sends at most 32 inputs per request, so that decoding one response holds
  the event loop only briefly, and returns every vector or none:
  a failed or invalid request fails the call and leaves nothing cached. A
  response must hold one vector per input, complete indices, finite numbers
  and one dimension; an answer the SDK cannot read as an embedding payload at
  all counts as a provider failure and is retried. Each runtime owns its
  vector cache; the Memory store and the search engine no longer hold one.
  `MemoryStore`, `SearchEngine`, `MemoryEngine`, `EntityResolver` and
  `MemoryIndexRepairer` take `embedding=` (an `EmbeddingRuntime`) where they
  took `embed_cfg=`, and `memforge.runtime.build_embedding_runtime` builds
  one. Importing a model client applies one
  transport policy to generation and embedding: no HTTP client LiteLLM uses
  follows a redirect (`litellm.aclient_session` is an `httpx.AsyncClient`
  that refuses them, with one connection pool per event loop, and the clients
  LiteLLM builds for the other routes, the `anthropic/` route of a bare
  generation model among them, refuse them too), so an endpoint that answers
  with a redirect fails and neither the API key nor the request body is sent
  on; LiteLLM's default SDK retry count is zero, so one attempt is one
  request; and LiteLLM's help notice is no longer printed to standard output
  on a failed request. A provider answer that LiteLLM reports as a connection
  error is retried or not by the status the provider gave, so a 400, 401 or
  404 on the `ollama/` or `voyage/` routes is one request. `litellm` is
  declared as `>=1.86,<1.88`. Cloud: pass `embedding=EmbeddingRuntime(...)` in place of
  `embed_cfg=` to those five constructors (`SearchEngine` needs it even with
  `embedding_provider=`, and no longer reads
  `RetrievalConfig.embedding_cache_size`: pass `cache_size` in
  `EmbeddingSettings`) and replace `embed_texts` calls with
  `await runtime.embed_documents(...)` or `embed_query(...)`: embedding is
  asynchronous and needs no thread hand-off. The attributes `embed_cfg` and
  `SearchEngine._embed_cfg` are `embedding` and `SearchEngine._embedding`,
  and a test that replaced `embed_texts`, `litellm.embedding` or `httpx`
  replaces `litellm.aembedding`. SDK retries are disabled for `sap/` routes
  as well (`max_retries=0` on every request), so confirm that AI Core accepts
  requests without them and that embedding on a `sap/` route does not hold
  the event loop. No HANA change.
- A document Source run whose scope listing or absence confirmation does not
  complete still succeeds and removes nothing (ADR 0045), and now records why
  on the run: `SyncState.absence_check_skipped_reason`, stored in the new
  nullable `sync_history.absence_check_skipped_reason` column (SQLite
  migration 112) and returned as `sync.absence_check_skipped_reason` on each
  Source in `GET /api/v1/sources`. The field is null when the check ran or when
  the connector cannot list its scope. The admin UI V2 Source drawer shows it
  as a notice, "Deletion check skipped: <reason>". Cloud: the HANA workspace
  store adds the nullable `SYNC_HISTORY.ABSENCE_CHECK_SKIPPED_REASON` column at
  startup with `_add_column_if_missing`, writes it with every sync history row
  and returns it from `get_sync_history`. Existing rows read as null; no data
  migration, `sap/` route or environment configuration change.
- A Support that cites the stored Unit Title is judged again
  (`revision-support-v12`). Evidence on a stored revision in a representation
  no adapter projects kept its whole Support unresolved, so a Memory that
  cited the Unit Title as Required Evidence was never revalidated after its
  Unit moved to a title-free revision. Such a part now leaves its Support,
  which is rebound by its other parts or assessed like any other; the Unit
  Title is still shown by every model reading and Memory read. Only profiles
  named in `source_representation.py` are left out, today `unit-identity`
  alone. Evidence in any other representation this build does not read still
  keeps its Support unresolved, and the recorded reason now says so instead
  of reporting partial projection coverage. Cloud: arrives with the pin; no
  HANA change.
- Context Evidence is the Observation a claim's Primary answers, shown as a
  reader sees it. A claim carries Context only when its provider declares the
  reply; an Observation that merely precedes the Primary is read during
  selection and is Evidence only when selected as Required. A whole
  Observation reads as its record framing once and then its entries instead
  of its stored JSON record, for new Context and for Context already stored
  without a presentation. SQLite migration 111 deactivates stored Context
  associations whose Observation the Evidence Unit's Primary does not answer;
  Evidence References and supporting Evidence are unchanged. Cloud: the HANA
  evidence read presents a whole Observation the same way and the HANA
  migration `context-for-declared-reply-v1` deactivates the same
  associations, together with the pin to this version. The unused storage
  method `replace_evidence_context_associations` is removed from the protocol
  and both stores.
- Model catalogs read in source-time order and list record framing only as
  context (`revision-input-v11`). Observations are ordered by the time their
  source reports, then each one's Fragments by position, so a Teams window or
  an issue's comments read in the order they were written instead of in
  revision-id order. A contextual record field is never Primary and never an
  entry read alone; fields a schema declares as record framing (Teams sender,
  times, reply referent and conversation; Jira issue key and id, author and
  event times) stay displayed on every entry and are no longer listed as separate
  entries. Evidence that selected such a field is assessed again the next time
  its Unit is derived. Stored revisions and committed Evidence are unchanged.
  Cloud: arrives with the pin; no HANA change.
- MCP `get_memory` returns what an agent reads and can act on. Each Evidence
  group keeps its Source, currency, document locators and pinned
  `resource_url`; each item keeps its role, kind, currency, `resource_url`,
  readable `text` and Artifact summary, filename, media type, size and URL.
  Storage identities, anchors, hashes and `text_view` stay in the REST
  projection: no tool accepts them. Tool results are written with every script
  as text instead of `\uXXXX` escapes. Plugin 0.1.65; the server is unchanged.
- Entity admin routes follow Memory visibility. The entity list, detail, alias
  list and linked Memory count admit an Entity only through a linked active
  Memory the caller can see, so a name learned from another user's private
  Memory is not discoverable. Adding or removing a manual alias and merging
  entities require workspace administration (403 `entity_curation_forbidden`)
  and every Entity involved must be discoverable (404). Entity detail returns
  `can_curate`. Storage: `list_entities` and `count_memories_for_entity` take
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
