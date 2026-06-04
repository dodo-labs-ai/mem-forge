# Security Policy

MemForge handles source-system credentials, generated agent-session
evidence, and memory derived from private workspaces. Treat all of those as
sensitive.

## Supported Versions

The public project is in alpha. Security fixes target the current `main`
branch until versioned releases are introduced.

## Reporting A Vulnerability

Please do not open a public issue for suspected credential exposure,
authentication bypass, or data leakage. Report privately through GitHub security
advisories for `shno-labs/mem-forge`.

Include:

- affected commit or version
- reproduction steps
- expected impact
- any logs or screenshots with secrets removed

## Secret Handling Expectations

- Do not commit `.env`, local databases, transcript exports, Chroma data, or
  browser-cookie artifacts.
- Prefer `MEMFORGE_*` environment variables for local secrets.
- Agent adapters redact obvious bearer tokens, API keys, passwords, and nested
  JSON secret fields before upload, but the service also validates and redacts
  incoming windows because client-side redaction is not a security boundary.
