# Set up MemForge with a coding agent

This guide is for a coding agent, such as Claude Code or Codex, that a person
asked to set up MemForge on their machine. Follow the steps in order and run
each command yourself. Where a step says the person acts, tell them exactly
what to do, then wait for them.

Rules for the whole setup:

- Never ask for, read, print or write an API key, token or other credential.
  The person enters them in the admin UI, so they never pass through this
  conversation. Do not read `.env`, and leave its model settings empty.
- Never remove MemForge's data: do not run `docker compose down -v` or
  `docker volume rm`.
- MemForge listens on `127.0.0.1` only. Do not change the port bindings in
  `compose.yaml` or expose the service on a network.
- If a command fails, show the person the error and stop. The
  [quickstart](https://github.com/dodo-labs-ai/mem-forge/blob/main/docs/quickstart.md)
  covers slow registries, proxies and port conflicts.
- Your sandbox may need the person's approval to reach the network or the
  Docker socket. Ask for it when a command is blocked.

## 1. Check Docker and an existing install

```bash
docker compose version
docker compose ls --all --filter name=memforge
docker volume ls -q --filter name=memforge_memforge-data
```

MemForge needs Docker with Compose v2. If the first command fails, ask the
person to install or start Docker, and continue when it works.

If the second command lists a `memforge` project, or the third lists
`memforge_memforge-data`, MemForge is already set up on this machine and has
data. Stop and ask the person whether to use that install: starting another
checkout would upgrade its data to that checkout's release. To use it, run the
remaining steps from the checkout the second command lists under CONFIG FILES,
or the one the person names.

## 2. Start MemForge

Ask the person where to keep MemForge, suggesting `~/mem-forge`. Do not clone
it inside their own project. Then clone it there:

```bash
git clone https://github.com/dodo-labs-ai/mem-forge.git ~/mem-forge
cd ~/mem-forge
```

In the checkout, create `.env` unless one exists, and start the stack:

```bash
[ -f .env ] || cp .env.example .env
docker compose up -d --build --wait
```

`--wait` returns once the API reports healthy. The first build can take
several minutes, longer than a shell tool's default timeout, so allow up to
15 minutes. If it fails, show the person `docker compose logs api` and stop.

The commands below use the default ports. If the person set other ports in
`.env`, `docker compose port api 8765` shows the API's address.

## 3. Install the agent plugin

Install the plugin for the agent you are. A marketplace that is already added
or a plugin that is already installed counts as done.

```bash
# Claude Code
claude plugin marketplace add dodo-labs-ai/mem-forge
claude plugin install memory@memforge
```

```bash
# Codex
codex plugin marketplace add dodo-labs-ai/mem-forge
codex plugin add memory@memforge
```

Ask the person whether they also use the other agent, and install it there
only if they say so. The plugin's memory tools load when a new session starts,
so they are not available in this one.

## 4. The person chooses models

Tell the person:

> Open http://localhost:5174/settings. Choose a generation model from
> OpenRouter, OpenAI or Anthropic and paste its API key, then choose **Save
> and validate**. An embedding model is optional and makes search semantic.
> Each model shows **Ready** when its test request succeeds.

When they say it is done, confirm it:

```bash
curl -fsS http://127.0.0.1:8765/api/v1/health
```

The generation model is ready when `models.generation.state` is `ready`. If it
is not, ask the person to read the error that Settings shows for the model.

## 5. The person connects a first source

Tell the person:

> Open http://localhost:5174/sources and choose **Add source**. Pick Notion,
> Confluence, Jira or a GitHub repository, enter the credentials the form asks
> for, and run a sync. Turn on **Sync automatically** to keep it current.

Link them to
[Add a source](https://github.com/dodo-labs-ai/mem-forge/blob/main/docs/quickstart.md#4-add-a-source)
for where each source's credentials come from. To follow the sync, list the
sources:

```bash
curl -fsS http://127.0.0.1:8765/api/v1/sources
```

Each source's `memory_count` grows as the sync runs.

## 6. Hand over

Tell the person what you set up: the checkout directory, the admin UI at
http://localhost:5174, and that `docker compose up -d` in the checkout starts
MemForge again later. Then ask them to start a new agent session in their
project and try:

```text
Before you change this, check MemForge for decisions and conventions that
apply, and show the source for each one you rely on.
```
