# Local repository sync

Register any local folder or repo (an Obsidian vault, a docs directory, a
project's notes) as a first-class MemForge source. The CLI is a thin bridge: it
reads each file's raw text and pushes it to the service, which converts and
extracts it exactly like any other gene. The source shows up in the admin UI,
can be renamed there, and syncs on demand or on a schedule.

## Supported file types

The CLI tags each file with a `content_type` from its extension. The service
converts to markdown during sync:

| Extension | content_type | Conversion (server-side) |
| --- | --- | --- |
| `.md`, `.markdown` | `text/markdown` | passthrough |
| `.txt` | `text/plain` | passthrough |
| `.json` | `application/json` | wrapped in a fenced ` ```json ` block |
| `.html`, `.htm` | `text/html` | converted with `html_to_markdown` (markdownify) |

PDF is intentionally not supported (no PDF text extraction exists in the service
today). See "Which files are picked up" below for what gets scanned and skipped.

## Set up a repo

Interactive: run `memforge` and choose **Local repository → Set up a repository**.

Scriptable:

```bash
memforge adapter kb add my-notes \
  --root /path/to/folder \
  --vault-id my-notes \
  --create-source
```

`--create-source` reuses or creates the matching `local_markdown` source and
stores its id in the profile (`~/.memforge/adapter.toml`). The server URL and
token come from the active target (`memforge target ...`) or
`MEMFORGE_API_URL` / `MEMFORGE_API_TOKEN`; the default is
`http://127.0.0.1:8765`.

### Which files are picked up

By default the **entire folder tree is scanned recursively**, and every `.md`,
`.markdown`, `.txt`, `.json`, `.html`, and `.htm` file (including those in
subfolders) is included. Narrow or widen that with globs:

```bash
memforge adapter kb add my-notes --root /path \
  --include "**/*.md" \
  --exclude "drafts/**" --exclude "archive/**"
```

- `--include` / `--exclude` are repeatable; globs match the path **relative to
  the root** (e.g. `analysis/**`, `**/*.md`).
- **`--include` replaces** the default type set. If you pass any `--include`,
  only those globs are used, so list every type you want (e.g.
  `--include "**/*.md" --include "**/*.txt"`).
- **`--exclude` is added to** the always-on safety excludes (`.obsidian/**`,
  `.trash/**`, `.git/**`, `**/.git/**`) rather than replacing them.
- **Exclude wins over include**: a file matched by both is skipped.

Files over 1 MB and non-UTF-8 files are always skipped.

## Sync

```bash
memforge adapter kb scan --root /path      # dry scan, no profile
memforge adapter kb preview my-notes       # what would sync, for a saved profile
memforge adapter kb push my-notes --process-now   # push and trigger extraction
```

`push` uploads each included file's raw text to
`POST /api/sources/{id}/adapter/packages` with its `content_type`. With
`--process-now`, a source sync runs after the last file. Re-pushing a file
overwrites its package; the pipeline skips documents whose content is unchanged.

Deleting a file locally is not propagated automatically: the push model only
adds and updates. To drop deleted files, remove the source (or run a forced full
resync) so the pipeline reconciles the inbox.

## Schedule

Scheduling installs a job in your **user crontab** that runs
`adapter kb push <name> --process-now` on a timer, logging to
`~/.memforge/kb-<name>.log`.

```bash
memforge adapter kb schedule my-notes --every daily --at 07:30
memforge adapter kb schedule-list
memforge adapter kb unschedule my-notes
```

Presets for `--every`: `15m`, `30m`, `hourly`, `2h`, `4h`, `6h`, `12h`,
`daily`, `weekly`. `--at HH:MM` sets the time for `daily`/`weekly` (default
09:00). `--cron "<5-field expr>"` overrides the preset for full control.

Notes:
- The crontab block is marked (`# >>> memforge:kb:<name> >>>`) so re-running
  `schedule` replaces it in place and `unschedule` removes only that block.
- On macOS, `cron` may require Full Disk Access for the controlling terminal.
- The scheduled command uses the target/credentials resolvable from cron's
  environment; for non-default targets, export `MEMFORGE_API_URL` /
  `MEMFORGE_API_TOKEN` in your crontab.

## Rename a source

A local repository source is renamable in the admin UI: open its **Configure**
dialog and edit **Source name** (this is a `PUT /api/sources/{id}`). The
`vault_id` the CLI uses to address the source is separate and stays stable.

## TODO: background daemon

A long-running `memforge` watcher (file-system events for near-real-time sync,
no crontab) is planned as an alternative to the OS scheduler. Until then, use
`adapter kb schedule` (cron) or manual `push`.
