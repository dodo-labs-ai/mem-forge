# Daemon Operations

Read only for troubleshooting, lifecycle actions, target/token changes, or
uninstall requests.

## Diagnose a failed setup

Use the supported surfaces:

```bash
memforge daemon status
memforge daemon logs --lines 100
```

Keep health/discovery, authentication, native service, runtime lock, and server
heartbeat failures distinct. Do not retry blindly, bypass capability discovery,
write a compatibility target, run launchctl/systemctl directly, or use root. If
the server lacks `/.well-known/memforge`, update the server before retrying.

Inspect only bounded relevant logs. Redact credentials, user-specific paths,
and unrelated Source content before relaying excerpts.

Setup is transactional. When it reports rollback failure, stop and report that
terminal condition instead of applying another mutation.

## Ongoing lifecycle

Use only these commands:

```bash
memforge daemon start
memforge daemon stop
memforge daemon restart
memforge daemon logs
memforge daemon status
memforge daemon check
```

Changing the target or token is a new setup operation with the same preview and
authorization boundary.

## Uninstall boundary

Only uninstall after explicit authorization:

```bash
memforge daemon uninstall --yes
```

Verify through the uninstall result and `memforge daemon status` that
`installed`, `loaded`, `running`, and `configured` are false, runtime is
stopped, connection is unconfigured, and any previously reported service PID is
gone. Do not require PID evidence on a platform that does not expose one.

A successful uninstall means the CLI also removed its daemon keyring
credential; do not inspect or display raw Keychain records. Explain that
uninstall intentionally preserves the CLI, `~/.memforge/cli.toml`, logs,
runtime history, Sources, and other MemForge data. Never claim this is a full
machine reset and never delete `~/.memforge` recursively.
