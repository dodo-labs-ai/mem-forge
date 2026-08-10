---
name: memforge-setup
description: Inspect and configure MemForge workspace routing for Git repositories, ordinary directories, and automatic hooks. Use when the user asks to bind a local project to a workspace, change or remove that binding, configure the hook fallback, or diagnose workspace selection.
compatibility: Requires Python 3.12+ and access to the configured MemForge workspace directory.
---

# MemForge Workspace Setup

Use the bundled `scripts/workspace_config.py` helper. It owns only
`~/.memforge/workspace-bindings.json`; never ask the user to edit the file by
hand.

## Routing model

- A Git checkout with an `origin` is stored in `repository_bindings` using the
  normalized origin, so its worktrees and moved checkouts share a binding.
- Any ordinary non-Git directory is stored in `directory_bindings` by absolute
  path. Descendants inherit it and the most-specific directory wins.
- An explicit tool `workspace_id` overrides either binding for that call.
- `hook_workspace_id` is only a fallback for automatic hooks. Interactive MCP
  tools never use it.
- API URL and credentials remain in the client's existing configuration. This
  skill never writes or displays credentials and never edits MCP registration.

## Commands

Resolve the helper relative to this skill, then use:

```bash
python3 "<skill-dir>/scripts/workspace_config.py" --client CLIENT inspect --path "$PWD"
python3 "<skill-dir>/scripts/workspace_config.py" --client CLIENT bind --path "$PWD" --workspace WORKSPACE
python3 "<skill-dir>/scripts/workspace_config.py" --client CLIENT unbind --path "$PWD"
python3 "<skill-dir>/scripts/workspace_config.py" --client CLIENT set-hook-workspace --workspace WORKSPACE
python3 "<skill-dir>/scripts/workspace_config.py" --client CLIENT clear-hook-workspace
```

`CLIENT` is `codex` or `claude-code`. Binding commands validate the workspace
against the live configured target. Mutations are previews unless `--apply` is
present.

## Required confirmation workflow

1. Use `list_workspaces` when available and let the user choose the workspace;
   never infer it from the server default.
2. Run the selected command without `--apply`.
3. Show the operation, binding kind/key, workspace, target origin, exact file,
   and planned document. Do not show credentials.
4. Ask for explicit confirmation of that preview.
5. After confirmation, repeat with `--apply --expect-digest DIGEST`, using the
   exact `document_digest` from the preview.
6. If the digest changed, preview and confirm again. Do not bypass the guard.

An initial request to set up routing is not confirmation of the generated
preview. Inspection is read-only and needs no confirmation.

## Completion report

Report the local path or repository identity, workspace, binding source, file
changed, and that the next MCP call/hook will use it. State that no credentials
or MCP registration changed. A restart is not required.
