# Architecture overview

MemForge is a self-hosted service that turns a team's documents and coding-agent
sessions into short, source-cited memories and serves them to agents over MCP.
This page names the parts and follows the data between them, for people who run
MemForge or contribute to it.

## Components

| Component | What it does | Code |
| --- | --- | --- |
| API service | A FastAPI application that owns sources, sync, memories, reviews and settings | `src/memforge/server` |
| Source connectors | One per source type: Confluence, Jira, Notion, GitHub Pages, GitHub repositories, Microsoft Teams, local folders and agent sessions. The code calls a connector a *gene*. Each fetches documents and hands them over in one common form | `src/memforge/genes` |
| Pipeline | Extracts memories from each document with an LLM, keeps the Evidence each memory rests on, and rechecks memories when their source changes | `src/memforge/pipeline` |
| Storage | SQLite for documents, memories and sync state, with FTS5 for keyword search, and Chroma for vectors | `src/memforge/storage` |
| Retrieval | Vector, keyword and entity channels run in parallel and are fused into one ranking | `src/memforge/retrieval` |
| Agent plugins | Claude Code and Codex plugins. Lifecycle hooks upload redacted session windows, and a local MCP proxy forwards tool calls to the API | `integrations` |
| Local daemon | Uploads files from a local folder or repository for a Local Repository source | `src/memforge/local_agent` |
| Admin UI | The web UI for sources, memories, reviews, projects and settings. `admin` is replacing `admin-ui`; both ship until the replacement is complete | `admin`, `admin-ui` |

## How data moves

1. **Capture.** A connector syncs documents on demand or on a schedule. Agent
   plugins upload bounded, redacted windows of each session.
2. **Extract.** The pipeline turns each document into memories: facts,
   decisions, procedures and conventions. Each memory keeps Evidence that
   points back to the text it came from.
3. **Reconcile.** When a source changes, the memories built on it are rechecked
   against the new revision. A memory is kept, replaced or retired, or it waits
   for a review. Memories from different documents are related when they state
   the same thing, update one another or contradict one another.
4. **Recall.** An agent calls `search` for memory cards, `get_memory` for a
   memory's Evidence and relations, and `get_resource` for the backing
   document. An agent can also propose creating, correcting or retiring a
   memory, and confirms with the person first.

## Deployment

`docker compose up` starts two containers, the API and the admin UI, with one
data volume. The open-source build serves a single user in one workspace named
`local`, has no authentication and binds to `127.0.0.1`.

Model calls go to the endpoints you configure: a generation model, which
extracts memories and makes semantic judgments, and an embedding model.
Without a generation model a sync stores content and creates no memories.

The vectors of all memories form one index, built with one embedding model and
recorded with a fingerprint of that model's settings and vector length.
Vectors are read and written only by the model the index was built with.
Changing the embedding model rebuilds the index from the memories in SQLite
and switches to it in one step; the [quickstart](quickstart.md#changing-the-embedding-model)
describes the procedure.

## Adding a source

A source type is two pieces: a connector (a Gene) in `src/memforge/genes`, and
a descriptor in `src/memforge/source_types`. Every connector describes itself
and its configuration fields, authenticates, discovers the content in scope,
fetches each item and normalizes it into the common form
(`src/memforge/genes/base.py`). The module docstrings named below are the
full guide, and the existing connectors are the reference.

**Choose the family.** The Gene's base class decides how a sync run collects
the source and how it learns that something is gone:

- `DocumentGene` (`genes/document_gene.py`) is for a provider whose items have
  their own identifiers and can be listed: pages, issues, files. Each run
  fetches the items changed since the last run, then lists the configured
  scope to find the Documents that are gone. Confluence, Jira, GitHub and local
  folders work this way.
- `ConversationGene` (`genes/conversation_gene.py`) is for messages in channels
  and chats. The shared conversation layer (`src/memforge/conversations`) cuts
  messages into thread and window Units and decides what to remove. The
  connector supplies a provider that implements the port in
  `conversations/port.py`. Teams works this way.

A Gene of neither family has no sync run; agent sessions are the only such type.

**Write the descriptor.** A `SourceTypeDescriptor` (`source_types/descriptor.py`)
holds every fact shared code needs about the type: its Gene, the configuration
fields that name the provider and those that define the scope (with
`scope_defaults` and `canonical_scope`), how items project into Units and
Evidence, secret fields and configuration checks, whether the local daemon
collects it, and, for a conversation type, its provider and the factory that
opens it. Each built-in type has a module in `source_types` that exports
`DESCRIPTOR`, and `source_types/__init__.py` registers it. Shared code reads the
descriptor and never branches on a source type name, so a new type adds no
`if` elsewhere.

**Describe the form.** The Gene's `config_schema()` is the whole setup form;
the admin UI draws it without per-type code, and the server checks every saved
configuration against it (`source_types/form.py`). Group fields into steps,
each with a `purpose`: `connection` to reach the provider, `content` to choose
what to collect, `options` for settings with working defaults. Give a `select`
field labelled `choices`. Show a field or a choice only for some modes with
`visible_when`, a list of `ConfigCondition(field, values)` on an earlier
`select` field; `required` then means required while visible. Name an
interactive control with `widget`, or a panel on a group with `panels`. A
blank field reads the descriptor's `scope_defaults` entry, then its declared
`default`, so declare a mode's default as the value the connector already
assumes. Write any value the type infers, such as a mode a pasted URL implies,
in `normalize_config`, so the stored configuration names it and config-check
can show it. Keep in `validate` only what a form cannot state, and raise
`SourceConfigInvalid` naming the field you refuse (`genes/toolkit/config.py`).
State what a person should see about a configuration, such as the repository a
URL names, with the descriptor's `facts`.

**Choose the listing kind** (document connectors). A scope listing names every
item in the configured scope by Document id, completely and without content.
The run compares it with the Documents the source holds. Ask what defines the
scope:

- **The listing itself: existence.** An item the listing leaves out is outside
  the scope by definition, so its Document is removed. Examples: a repository
  tree, where a moved file gets a new Document id; a sitemap the site declares
  complete; pages the user lists; a folder snapshot from the local daemon.
- **A condition over items that exist on their own: query.** An item can stop
  matching and keep its identifier, so a Document the listing leaves out is
  removed only when `confirm_absent` asks the provider by identifier and hears
  that it is gone. Leaving the query is not deletion. Examples: a Jira JQL, a
  Confluence page tree.

When unsure, choose query: it never removes an item the provider still has.
Declare the kinds in `listing_kinds`. Answer `lists_scope(config)` for each
configuration, and answer `False` for one that cannot list completely, such as
a bounded link crawl; its sources then never remove a Document through a
listing. Override `unlisted_scope_mode(config)` to name that mode and suggest
one that lists; a scope change into such a configuration is refused with that
suggestion. `list_scope()` returns a complete listing or raises; it never returns
less. A type the local daemon collects is listed by its snapshot manifest and
declares exactly one kind. When that kind is query, the Gene also implements
`provider_key` and `confirmation_item`, so the daemon can confirm items by
identifier.

**Startup validation.** Once every type has registered, loading `memforge.genes`
checks each one against its family's contract (`genes/contract.py`). A type
that breaks it stops the service with a `SourceTypeContractError` naming the
type and the missing piece: a collecting Gene outside both families, an
unimplemented abstract method, undeclared `listing_kinds`, `lists_scope` not
declared as a class method, a query listing without `confirm_absent`,
`rediscovers_documents` without `rediscover`, a daemon-collected Gene with more
than one listing kind, a conversation provider missing a member of the port, or
a form description that a client could not draw: a condition on a later field
or on a value the field does not offer, a repeated choice, or an unknown
widget, panel or group purpose.

**Add the contract-suite fakes.** `tests/contracts/test_document_connector_contract.py`
runs every registered document type against fakes of its provider, kept in
`tests/contracts/document_fakes/<source_type>.py`. A registered type without
that module fails the suite. The package docstring says what each fake must
answer. The suite fakes only the provider's transport, so it checks the Gene's
own paging, listing and error handling.

## Further reading

- [Quickstart](quickstart.md): setup, models, plugins, sources and the CLI.
- [API overview](api.md): the core Admin API endpoints.
- [Local repository sync](local-repo-sync.md): the local daemon and its jobs.
