# Local repository sync

Local folders and checked-out repositories are configured in the MemForge UI.
The local daemon reads the selected files and uploads raw packages; the server
owns conversion, normalization, memory extraction, deduplication, and document
lifecycle reconciliation.

## Setup

1. Install the host-side CLI with `pipx install memforge-ai`.
2. Run `memforge setup` to configure the active API target and install the local
   daemon as a login user service.
3. In the Sources UI, add a Local Markdown or GitHub Repository source.
4. Choose the local folder or repository through the source-specific browser.
5. Configure include and exclude patterns, then save the source.

When no target is already configured, `memforge setup` prompts for the API URL;
press Enter to accept the local self-hosted default or enter the hosted URL. The
same URL can be supplied non-interactively, after which Cloud setup prompts for
the token:

```bash
memforge setup --api-url https://memforge.example.hana.ondemand.com
```

`memforge setup` registers a macOS LaunchAgent or Linux systemd user service,
starts it, and waits for the server-observed heartbeat. The API token is stored
in the operating-system keyring rather than in the launchd property list,
systemd unit, or MemForge config file. No `sudo`, hand-written service file, or
user cron entry is required.

When developing MemForge from a checkout, keep using the foreground command
`uv run memforge adapter daemon run` instead of the installed command.

The daemon authenticates with the user's MemForge API key. Self-hosted OSS and
Cloud both use `/api/v1`; self-hosted resolves the readable singleton workspace
`local`. Hosted jobs persist the workspace resolved when the job is admitted,
and the daemon scopes every later job API call to that stored workspace.

## Supported file types

The daemon assigns a content type from the file extension and uploads the raw
text. Conversion happens on the server.

| Extension | Content type | Server processing |
| --- | --- | --- |
| `.md`, `.markdown` | `text/markdown` | passthrough |
| `.txt` | `text/plain` | passthrough |
| `.json` | `application/json` | converted to fenced JSON markdown |
| `.html`, `.htm` | `text/html` | converted with `html_to_markdown` |

Files over 1 MB and non-UTF-8 files are skipped. PDF is not part of this local
text-source contract.

## Sync ownership

The UI's Sync action and the source schedule both create durable server-owned
jobs. The daemon leases those jobs; it has no local profile scheduler or cron
configuration.

Every raw package upload and final processing request carries the leased job id
and attempt count. The server rejects missing, expired, or superseded lease
contexts before writing raw input, so an old daemon attempt cannot publish a
partial snapshot after another daemon has reclaimed the job.

The daemon also enforces that authority while a handler is running. A definitive
heartbeat or package rejection stops the current attempt at its next progress
checkpoint; if renewal remains unavailable, the daemon stops itself when the
last confirmed lease deadline passes. It does not send a stale terminal
completion. The same durable job can then be reclaimed with a higher attempt
number and a new attempt-scoped snapshot.

For each complete folder or repository scan, the daemon derives a snapshot id
from the job id and lease attempt number. Every uploaded package and the final
processing request use that same attempt-scoped id. The server processes only
that snapshot, including an explicit empty snapshot, so deleted or newly
excluded files are reconciled correctly.

If any package upload fails, the daemon does not publish the partial snapshot.
The same job is retried with a new attempt-scoped snapshot, so membership from a
partial attempt cannot leak into the retry. Stable document/content identities
reuse raw inputs and prevent duplicate input generations or memories.

Collection progress reports discovery, content fetching, and Cloud upload as
separate phases. Large remote repositories therefore continue advancing while
the daemon reads each selected file instead of leaving the Source row on the
last discovery count.

## Scheduling

Configure the interval in the source's UI. The server owns the next-run time and
creates a local-agent job when the source becomes due. The daemon only needs to
remain running and authenticated; no user crontab or daemon-side schedule is
created.

## Operations

Manage the installed user service through MemForge rather than calling
`launchctl` or `systemctl` directly:

```bash
memforge daemon status
memforge daemon check
memforge daemon restart
memforge daemon logs --follow
memforge daemon stop
memforge daemon start
memforge daemon uninstall
```

The foreground `memforge adapter daemon run` and one-shot
`memforge adapter daemon once` commands remain available for development and
diagnosis. Automatic service installation currently supports macOS launchd and
Linux systemd.

`memforge daemon status` reports native service state, the local daemon PID and
lock, and the server-observed heartbeat for the daemon's stored target and
Keychain credential. `memforge daemon check` exits nonzero unless all three are
healthy, so installation can be verified without first triggering a source
sync.

Rename, pause, rescope, force-resync, schedule, and delete sources in the UI.
Connection paths and execution ownership remain attached to the saved source,
while project binding controls where extracted memories land.

The user who creates a local source is its execution owner because the source
depends on that user's filesystem and browser credentials. Workspace admins can
rename, bind, schedule, pause, or delete the source, but only the execution
owner's daemon can configure its connection or run a sync.
