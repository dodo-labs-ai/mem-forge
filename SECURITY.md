# Security Policy

MemForge handles source-system credentials, memories agents save on a user's
request, and memory derived from private workspaces. Treat all of those as
sensitive.

## Supported Versions

The public project is in alpha. Security fixes target the current `main`
branch until versioned releases are introduced.

## Reporting A Vulnerability

Please do not open a public issue for suspected credential exposure,
authentication bypass, or data leakage. Report privately through GitHub security
advisories for `dodo-labs-ai/mem-forge`.

Include:

- affected commit or version
- reproduction steps
- expected impact
- any logs or screenshots with secrets removed

## Secret Handling Expectations

- Do not commit `.env`, local databases, transcript exports, Chroma data, or
  browser-cookie artifacts.
- Prefer `MEMFORGE_*` environment variables for local secrets.
- The memory plugins send no session content. A coding agent saves memory only
  through a `create_memory` call the user confirms, so review what it proposes
  to save before confirming.
