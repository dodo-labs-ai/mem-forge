<h1><img src="https://raw.githubusercontent.com/dodo-labs-ai/mem-forge/main/.github/assets/logo.svg" width="36" align="absmiddle" alt=""> MemForge</h1>

**Give every agent your team's brain on day one.**

The memory layer for AI agents. Everything your team knows. Cited, and always
up to date. Built from your docs, tickets and chat.

<p>
  <a href="https://github.com/dodo-labs-ai/mem-forge/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/dodo-labs-ai/mem-forge/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="License Apache 2.0" src="https://img.shields.io/badge/license-Apache--2.0-blue">
  <img alt="Status beta" src="https://img.shields.io/badge/status-beta-3b82f6">
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/dodo-labs-ai/mem-forge/main/.github/assets/demo-dark.gif">
    <img src="https://raw.githubusercontent.com/dodo-labs-ai/mem-forge/main/.github/assets/demo-light.gif" alt="On day one, a new hire asks a coding agent how to fix a ticket about a customer charged twice. Without team memory the agent starts patching the payment code. With MemForge it checks team memory first: not a bug, a pending card hold, citing the support playbook and the team's decision log in Notion. The admin UI shows the connected sources syncing on their own, and a memory the decision log updated." width="100%">
  </picture>
</p>

<p align="center">
  The open-source edition runs on your machine. The <b>Team Cloud Edition</b> is on the way:
  we host it and provide the models, your team just subscribes.
  <a href="https://github.com/dodo-labs-ai/mem-forge/discussions/13">Join the waitlist</a>.
</p>

## Why MemForge

**Your team's knowledge isn't in the repo.** A new hire's agent reads the code
and guesses, while the answer sits in a Notion page nobody linked. A `CLAUDE.md`
or `AGENTS.md` file only fits so many rules. MemForge connects to Confluence,
Jira, Notion, GitHub and Teams, so the agent can check what your team already
decided before it acts.

**Docs change, and old answers linger.** When a doc changes, the next sync
rechecks every memory built on it and keeps, replaces or retires each one, with
no manual cleanup. The old version stays as history. MemForge also looks for
memories that repeat, update or contradict each other and links them. When it
finds a conflict, it keeps both and tells the agent, so the disagreement is
visible instead of blended into one answer.

**Agents shouldn't read whole docs to find one rule.** A model extracts short
memories from your docs, and MemForge searches those instead. The agent searches
by keyword and, with an embedding model, by meaning, and gets back only the
memories that match. It opens the evidence or source when it needs more.
Extraction spends model tokens when a source syncs, so agents have less to read
when they answer. Nobody has to keep an index or folder structure tidy for the
agent to find things.

**Team memory with evidence.** A memory from a source cites the text it rests on
and links to the page, ticket or file, pinned to the revision where the source
allows it. The cited text can be the model's wording, so the agent can open the
original to confirm it.

**Run it yourself, or share it across the team.** The open-source edition is
Apache-2.0 and runs on your machine for one person, with local storage and your
own model keys. Source text goes to the model providers you choose, and the
plugins don't capture your sessions. The
[Team Cloud Edition](https://github.com/dodo-labs-ai/mem-forge/discussions/13),
on the way, will give the whole team one shared memory, so every agent starts
from the same knowledge.

## Quick start

> **Beta.** APIs, storage formats and plugin packaging may change. An upgrade
> that migrates your data saves a recovery point first
> ([Upgrading](https://github.com/dodo-labs-ai/mem-forge/blob/main/docs/quickstart.md#upgrading)).

You need Docker with Compose v2 and an API key for OpenRouter, OpenAI or
Anthropic.

**Let your agent set it up.** Paste this into Claude Code or Codex:

```text
Set up MemForge for me. Fetch this guide with curl and follow it exactly:
https://raw.githubusercontent.com/dodo-labs-ai/mem-forge/main/docs/agent-setup.md
```

The agent starts MemForge and installs its plugin, then hands you the admin UI
to enter your model key and connect a source, so your keys never pass through
the agent.

**Or set it up yourself:**

1. **Start MemForge.**

   ```bash
   git clone https://github.com/dodo-labs-ai/mem-forge.git && cd mem-forge
   cp .env.example .env
   docker compose up --build
   ```

   The admin UI is at http://localhost:5174.

2. **Choose models.** In **Settings**, choose a generation model, which writes
   the memories, and optionally an embedding model for semantic search. After
   **Save and validate**, each one shows **Ready**.

3. **Connect a source.** In **Sources**, add Notion pages, a Confluence
   space, a Jira project or a GitHub repository, and run a sync. Turn on
   **Sync automatically** to keep it current. The source shows the memories it
   made as the sync runs.

4. **Install the agent plugin.**

   ```text
   # Claude Code (inside a session)
   /plugin marketplace add dodo-labs-ai/mem-forge
   /plugin install memory@memforge
   ```

   ```bash
   # Codex
   codex plugin marketplace add dodo-labs-ai/mem-forge
   codex plugin add memory@memforge
   ```

5. **Ask.** Start a new agent session in your project and ask:

   ```text
   Before you change this, check MemForge for decisions and conventions that
   apply, and show the source for each one you rely on.
   ```

   The agent answers with the memories that apply and names the source of each.

The [quickstart](https://github.com/dodo-labs-ai/mem-forge/blob/main/docs/quickstart.md) covers every model provider,
local folders, the host CLI and network details.

## How it works

```mermaid
flowchart LR
  subgraph You["Your machine"]
    CC["Claude Code / Codex"]
    Plugin["MemForge plugin"]
    API["MemForge"]
    Extract["Write short memories,<br/>each citing its source"]
    Recheck["Recheck when a source changes,<br/>link related memories"]
    Store[("Local storage")]
  end
  Docs["Confluence, Jira, Notion,<br/>GitHub, Teams, local files"]

  CC --> Plugin
  Docs -- "sync" --> API
  API --> Extract --> Recheck --> Store
  Plugin -- "search, read a memory,<br/>open its source" --> API
  API -- "read" --> Store
```

1. **Sync.** Connectors read your docs, tickets and chat, on a schedule or
   when you ask.
2. **Extract.** A model turns each source into short memories, each citing the
   passage it came from.
3. **Recheck.** When a source changes, the memories built on it are kept,
   replaced or retired, and related memories are linked.
4. **Recall.** Agents search memory and open its sources over MCP.

More detail: [architecture overview](https://github.com/dodo-labs-ai/mem-forge/blob/main/docs/architecture.md).

## Integrations

MemForge sits between the systems where your team writes things down and the
agents that do the work.

**Agent integrations**

<p>
  <img src="https://raw.githubusercontent.com/dodo-labs-ai/mem-forge/main/.github/assets/icons/claude.svg" width="18" height="18" align="absmiddle" alt="">&nbsp;Claude&nbsp;Code&nbsp;&nbsp;&nbsp;
  <picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/dodo-labs-ai/mem-forge/main/.github/assets/icons/openai-dark.svg"><img src="https://raw.githubusercontent.com/dodo-labs-ai/mem-forge/main/.github/assets/icons/openai-light.svg" width="18" height="18" align="absmiddle" alt=""></picture>&nbsp;Codex
</p>

**Memory sources**

<table>
  <tr>
    <td>Docs and wikis</td>
    <td nowrap>
      <img src="https://raw.githubusercontent.com/dodo-labs-ai/mem-forge/main/.github/assets/icons/confluence.svg" width="18" height="18" align="absmiddle" alt="">&nbsp;Confluence&nbsp;&nbsp;&nbsp;
      <picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/dodo-labs-ai/mem-forge/main/.github/assets/icons/notion-dark.svg"><img src="https://raw.githubusercontent.com/dodo-labs-ai/mem-forge/main/.github/assets/icons/notion-light.svg" width="18" height="18" align="absmiddle" alt=""></picture>&nbsp;Notion&nbsp;&nbsp;&nbsp;
      <picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/dodo-labs-ai/mem-forge/main/.github/assets/icons/github-dark.svg"><img src="https://raw.githubusercontent.com/dodo-labs-ai/mem-forge/main/.github/assets/icons/github-light.svg" width="18" height="18" align="absmiddle" alt=""></picture>&nbsp;GitHub&nbsp;Pages&nbsp;&nbsp;&nbsp;
      <picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/dodo-labs-ai/mem-forge/main/.github/assets/icons/github-dark.svg"><img src="https://raw.githubusercontent.com/dodo-labs-ai/mem-forge/main/.github/assets/icons/github-light.svg" width="18" height="18" align="absmiddle" alt=""></picture>&nbsp;GitHub&nbsp;repo&nbsp;docs&nbsp;&nbsp;&nbsp;
      <img src="https://raw.githubusercontent.com/dodo-labs-ai/mem-forge/main/.github/assets/icons/folder.svg" width="18" height="18" align="absmiddle" alt="">&nbsp;Local&nbsp;folders
    </td>
  </tr>
  <tr>
    <td>Tickets</td>
    <td nowrap><img src="https://raw.githubusercontent.com/dodo-labs-ai/mem-forge/main/.github/assets/icons/jira.svg" width="18" height="18" align="absmiddle" alt="">&nbsp;Jira</td>
  </tr>
  <tr>
    <td>Chat</td>
    <td nowrap><img src="https://raw.githubusercontent.com/dodo-labs-ai/mem-forge/main/.github/assets/icons/teams.svg" width="18" height="18" align="absmiddle" alt="">&nbsp;Teams</td>
  </tr>
  <tr>
    <td>Planned</td>
    <td nowrap>
      <img src="https://raw.githubusercontent.com/dodo-labs-ai/mem-forge/main/.github/assets/icons/obsidian.svg" width="18" height="18" align="absmiddle" alt="">&nbsp;Obsidian
    </td>
  </tr>
</table>

[Add a source](https://github.com/dodo-labs-ai/mem-forge/blob/main/docs/quickstart.md#4-add-a-source) lists what each one syncs and
how to scope it. Need one that is missing?
[Vote for the next source](https://github.com/dodo-labs-ai/mem-forge/discussions/12).

## Limits

- The open-source edition runs for one person on one machine. It has no
  accounts or authentication and listens on `127.0.0.1` only, so do not expose
  it on a network. For memory your whole team shares, join the
  [Team Cloud Edition waitlist](https://github.com/dodo-labs-ai/mem-forge/discussions/13).
- Model extraction and semantic judgments can miss useful knowledge or make
  mistakes. Check the cited Evidence before relying on a consequential claim.
- A generation model is required; without one, syncs store content but create
  no memories.

## Documentation

- [Quickstart](https://github.com/dodo-labs-ai/mem-forge/blob/main/docs/quickstart.md): setup, models, plugins, sources, CLI,
  development from source
- [Docs index](https://github.com/dodo-labs-ai/mem-forge/blob/main/docs/README.md): quickstart, architecture, API and release guides
- [How MemForge compares](https://github.com/dodo-labs-ai/mem-forge/blob/main/docs/comparison.md): when to choose MemForge and when
  another tool fits better
- [Contributing](https://github.com/dodo-labs-ai/mem-forge/blob/main/CONTRIBUTING.md)

## License

Apache License 2.0. See [LICENSE](https://github.com/dodo-labs-ai/mem-forge/blob/main/LICENSE).
