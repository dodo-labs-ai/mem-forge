# Local repository sync

Local folders and checked-out repositories are configured in the MemForge UI.
The local daemon reads the selected files and uploads raw packages; the server
owns conversion, normalization, memory extraction, deduplication, and document
lifecycle reconciliation.

## Setup

1. Start the local daemon with `uv run memforge adapter daemon run`.
2. In the Sources UI, add a Local Markdown or GitHub Repository source.
3. Choose the local folder or repository through the source-specific browser.
4. Configure include and exclude patterns, then save the source.

The daemon authenticates with the user's MemForge API key. It does not require
a workspace environment variable: each server-issued job carries the workspace
and source selected in the UI.

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

For each complete folder or repository scan, the daemon derives a snapshot id
from the job id and lease attempt number. Every uploaded package and the final
processing request use that same attempt-scoped id. The server processes only
that snapshot, including an explicit empty snapshot, so deleted or newly
excluded files are reconciled correctly.

If any package upload fails, the daemon does not publish the partial snapshot.
The same job is retried with a new attempt-scoped snapshot, so membership from a
partial attempt cannot leak into the retry. Stable document/content identities
reuse raw inputs and prevent duplicate input generations or memories.

## Scheduling

Configure the interval in the source's UI. The server owns the next-run time and
creates a local-agent job when the source becomes due. The daemon only needs to
remain running and authenticated; no user crontab or daemon-side schedule is
created.

## Operations

Rename, pause, rescope, force-resync, schedule, and delete sources in the UI.
Connection paths and execution ownership remain attached to the saved source,
while project binding controls where extracted memories land.

The user who creates a local source is its execution owner because the source
depends on that user's filesystem and browser credentials. Workspace admins can
rename, bind, schedule, pause, or delete the source, but only the execution
owner's daemon can configure its connection or run a sync.
