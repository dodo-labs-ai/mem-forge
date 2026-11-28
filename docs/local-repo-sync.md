# Local repository sync

A Local Repository Source reads the files of a folder on one of your
computers, its execution device. The MemForge daemon on that computer walks the
folder with the Source's include and exclude patterns, reads each file and
hands what it read to the server; the server owns conversion, normalization,
memory extraction, deduplication and document lifecycle reconciliation.

## Setup

1. Install the host-side CLI with
   `pipx install --python python3.12 --pip-args=--pre memforge-ai`; the
   [quickstart](quickstart.md#6-install-the-host-cli) explains the options.
2. Run `memforge setup` to configure the active API target and install the local
   daemon as a login user service.
3. In the Sources UI, add a Local Repository source. In **Where it runs**,
   choose the computer that has the folder; with one computer it is chosen
   already.
4. Choose the folder with the folder picker, which opens on that computer.
5. Configure include and exclude patterns, then save the source.

A checked-out GitHub repository can also be read from GitHub itself: add a
GitHub Repository source and choose **My computer** in **Where it runs** when
only your computer can reach the repository, then store its token there with
`memforge source credentials set <source id>`.

When no target is already configured, `memforge setup` prompts for the API URL;
press Enter to accept the local self-hosted default or enter the hosted URL. The
same URL can be supplied non-interactively, after which Cloud setup prompts for
the token:

```bash
memforge setup --api-url https://memory.example.com
```

`memforge setup` registers a macOS LaunchAgent or Linux systemd user service,
starts it, and waits for the server-observed heartbeat. The API token is stored
in the operating-system keyring rather than in the launchd property list,
systemd unit, or MemForge config file. No `sudo`, hand-written service file, or
user cron entry is required.

Setup reads `/.well-known/memforge` from the exact origin to discover whether
the service is self-hosted or Cloud and which health/authentication contract it
uses. Custom domains therefore require no client-side DNS allowlist.

When developing MemForge from a checkout, keep using the foreground command
`uv run memforge adapter daemon run` instead of the installed command.

The daemon authenticates with the user's MemForge API key. Self-hosted OSS and
Cloud both use `/api/v1`; self-hosted resolves the readable singleton workspace
`local`. Hosted jobs persist the workspace resolved when the job is admitted,
and the daemon scopes every later job API call to that stored workspace.

## Supported file types

The daemon assigns a content type from the file extension and hands over the
file's text. Conversion happens on the server.

| Extension | Content type | Server processing |
| --- | --- | --- |
| `.md`, `.markdown` | `text/markdown` | passthrough |
| `.txt` | `text/plain` | passthrough |
| `.json` | `application/json` | converted to fenced JSON markdown |
| `.html`, `.htm` | `text/html` | converted with `html_to_markdown` |

A file over 1 MB, a file that is not UTF-8 and a file the daemon cannot read
fail on their own: the run reports them as failed Documents, keeps what each
one held before, and reads them again on the next run. PDF is not part of this
local text-source contract.

## What a run reads

Every run walks the whole folder once. The walk lists every file the patterns
include, by its path relative to the folder; that list is the Source's scope,
so a file that is deleted, renamed or newly excluded leaves the Source, and a
renamed folder's files are read under their new paths. A directory matching an
exclude pattern that ends in `/**`, such as `.git/**`, is not entered.

A missing folder, or a directory inside it that the daemon cannot list, fails
the run and removes nothing: the Source shows **Finish setup** with the
folder's path until the folder is back.

Each file's time is its last commit time when it is committed without local
changes, and its modification time otherwise.

The daemon uploads a file's text only when the server does not already hold
it for the Source, and the server extracts only files whose content changed,
so an unchanged folder costs a walk and no extraction.

## Sync ownership

The UI's Sync action and the source schedule both create durable server-owned
jobs. Only the Source's execution device leases them; the daemon has no local
profile scheduler or cron configuration.

Every upload and the final processing request carry the leased job id and
attempt count. The server rejects missing, expired or superseded lease
contexts, so an old daemon attempt cannot publish what it read after another
attempt has reclaimed the job. A definitive heartbeat or upload rejection
stops the current attempt at its next progress checkpoint; the same job can
then be reclaimed with a higher attempt number, which reads the folder again.

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

The user whose computer runs a Source is its execution owner, because the
Source depends on that computer's files and credentials. Workspace admins can
rename, bind, schedule, pause, or delete the Source, but only its execution
device runs a sync. Another member takes a Source over by choosing one of their
own computers in **Where it runs**.
