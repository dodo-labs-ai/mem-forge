---
name: memforge-setup
description: Configure and inspect MemForge workspace selection for Codex or Claude Code. Use whenever the user asks to set or update the global MemForge workspace, add or remove a repository workspace override, inspect which workspace is effective, or fix workspace routing. Guide the change in natural language, preview exact files and values, and require confirmation before applying it.
compatibility: Requires Python 3.12+ and a Git repository for repository overrides.
---

# MemForge Workspace Setup

Guide workspace routing without asking the user to edit TOML or JSON manually.
Use the bundled `scripts/workspace_config.py` helper relative to this skill.

## Supported operations

Route the request to exactly one operation:

1. Configure or update the global default.
2. Configure or update the current repository override.
3. Inspect the effective workspace selection.
4. Remove the current repository override.

If the client is not evident from the active agent, ask whether the user is
configuring Codex or Claude Code. Do not guess.

## Configuration ownership

- Codex global default: `~/.codex/config.toml`, key
  `[memforge].MEMFORGE_WORKSPACE_ID`.
- Claude Code global default: `~/.claude/settings.json`, key
  `env.MEMFORGE_WORKSPACE_ID`.
- Repository override for either client:
  `<git-root>/.memforge/config.toml`, key `[memforge].workspace_id`.
- Effective workspace precedence: valid repository override, process
  `MEMFORGE_WORKSPACE_ID`, then the client's global default.
- API origin and token keep process-first precedence. This skill changes only
  workspace selection.

Never put `MEMFORGE_API_TOKEN`, API keys, passwords, secrets, or other
credentials in the repository file. Never create, replace, or duplicate a
MemForge MCP entry. In particular, do not edit `.mcp.json`, `.mcp.local.json`,
`[mcp_servers.memforge]`, or plugin cache files.

## Helper usage

Resolve the absolute helper path from this skill directory, then run one of:

```bash
python3 "<skill-dir>/scripts/workspace_config.py" inspect --client codex --repo "$PWD"
python3 "<skill-dir>/scripts/workspace_config.py" inspect --client claude-code --repo "$PWD"
python3 "<skill-dir>/scripts/workspace_config.py" set-global --client codex --workspace WORKSPACE
python3 "<skill-dir>/scripts/workspace_config.py" set-global --client claude-code --workspace WORKSPACE
python3 "<skill-dir>/scripts/workspace_config.py" set-repo --client CLIENT --repo "$PWD" --workspace WORKSPACE
python3 "<skill-dir>/scripts/workspace_config.py" remove-repo --client CLIENT --repo "$PWD"
```

Mutating commands are dry runs unless `--apply` is present. The helper validates
the resulting Cloud/OSS target through the same target builder used by the MCP
and hooks. It never adds, removes, or reports credentials.

## Workflow

### Inspect

1. Run `inspect` without `--apply`.
2. Report the repository root, local-config status, effective source, workspace,
   and validated target. Do not report token values.
3. If the target is invalid, explain the returned validation code. Do not repair
   unrelated API URL or credential configuration without a separate request.

Inspection is read-only and does not require confirmation.

### Configure the global default

1. Obtain the workspace ID if the user did not provide it.
2. Run `set-global` without `--apply`.
3. Show the exact client-specific file, workspace value, validation result, and
   whether a process environment value currently shadows the saved default.
4. Ask for explicit confirmation to change that file.
5. Only after confirmation, repeat the same command with `--apply`.
6. Report the applied value. A new agent session is required because Codex
   caches user config and Claude Code reads top-level `env` at process startup.

Do not edit the other client's global configuration.

### Configure the current repository override

1. Resolve the Git root from the current working directory; never place the
   override in a plugin or installation directory.
2. Obtain the workspace ID if needed.
3. Run `set-repo` without `--apply`.
4. Show the exact `.memforge/config.toml` path, workspace value, target
   validation, and any planned local Git exclude update.
5. Ask for explicit confirmation to change every listed file.
6. Only after confirmation, repeat the same command with `--apply`.
7. Report that the override is effective for both MCP and hooks. No restart is
   needed; it applies on the next MCP request or hook invocation.

The helper refuses tracked repository config, credential-like repository keys,
and invalid Cloud/OSS combinations. Do not bypass those safeguards.

### Remove the current repository override

1. Run `remove-repo` without `--apply`.
2. Show the exact repository file and the validated fallback workspace/source.
3. Ask for explicit confirmation.
4. Only after confirmation, repeat with `--apply`.
5. Report the fallback selection. No restart is needed.

Remove only `workspace_id`. Preserve unrelated non-secret repository
configuration. Keeping the local Git exclude entry is intentional so future
workspace selections remain uncommitted.

## Confirmation boundary

Never infer confirmation from the initial setup request. A dry-run preview must
come first, followed by a clear question naming the file(s) and workspace
selection. Invoke `--apply` only after the user confirms that preview. If the
preview changes before apply, show the new preview and confirm again.

## Completion report

State:

- client and operation;
- configuration file changed, or that inspection was read-only;
- effective workspace and source;
- target validation result;
- whether a new session is required;
- confirmation that no credentials or MCP registration were changed.
