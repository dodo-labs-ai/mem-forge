# Changelog

## Unreleased

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
  (`kind_filtered`, and token limits on each model), and Cloud's LiteLLM gateway
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
  LiteLLM has no limits for (a `litellm_proxy/...` route without an entry in its
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
  `get_llm_config` and `set_llm_config` are no longer called; a `litellm_proxy/...`
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
  LiteLLM gateway. No HANA schema change.
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
  provider LiteLLM knows (`ollama/nomic-embed-text`, `litellm_proxy/...`) is used as
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
  replaces `litellm.aembedding`. SDK retries are disabled for `litellm_proxy/` routes
  as well (`max_retries=0` on every request), so confirm that LiteLLM gateway accepts
  requests without them and that embedding on a `litellm_proxy/` route does not hold
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
  migration, `litellm_proxy/` route or environment configuration change.
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
  signature together with the pin to this version; no HANA schema, `litellm_proxy/`
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
  their ids; no schema, `litellm_proxy/` route or configuration change. The V1 link
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
  arrives with the pin; no HANA schema, protocol, configuration or `litellm_proxy/` route
  change. The first scheduled run after the pin retires what the dev landscape has
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
