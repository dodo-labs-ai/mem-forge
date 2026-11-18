---
name: self-hosted-release-validation
description: "Validate an OSS self-hosted release through the newcomer Docker Compose path: isolated Compose project, SQLite persistence, source support matrix, local-agent sources and a public-beta GO/NO-GO. Use for compose smoke tests, release validation and beta-readiness reviews."
---

# OSS Self-Hosted Release Validation

- Validate self-hosted releases through the documented newcomer path: a clean Docker Compose project, loopback-only ports unless explicitly testing exposure, the default SQLite-backed data volume, Admin UI/API startup, first-run model configuration, a representative source sync, memory retrieval, and the packaged local MCP path.
- `scripts/compose_smoke.sh` runs this newcomer path in an isolated Compose project (add `--ollama`, export `OPENROUTER_API_KEY`, or name each model kind's provider with `--generation-provider` and `--embedding-provider` for the model, sync and retrieval steps) and prints a table for the PR or release notes. A release runs it with `--quality`, beside `provider_smoke.py`, as the release guide, `docs/releasing-python-distribution.md`, states.
- `compose_smoke.sh` tests the source tree: it builds the checkout under image names of its own. To validate what a newcomer pulls, follow the published-image check at the end of the release sequence in the release guide, whose `docker compose pull` fails on an image the registry refuses or lacks and whose `--no-build` keeps Compose from building one.
- Between releases an isolated Compose project builds under its own image tag (`dev-${COMPOSE_PROJECT_NAME}`), so it validates its own checkout. On a release commit the tag is the version in every project, so use `up --build` or the smoke to test local changes there.
- Use an isolated Compose project and dedicated volume for smoke tests. Never reuse or delete a developer's existing MemForge volume, source data, plugin cache, or external service state merely to obtain a clean result.
- Prove SQLite persistence by restarting the API or Compose stack against the same isolated data volume and rechecking sources, documents, memories, keyword/vector health, and MCP retrieval.
- Treat source support as an execution matrix, not a registry count. For each registered source type, state whether server collection, local-agent collection, credentials, browser sessions, VPN/internal reachability, or an external provider is required. Classify each type as verified, unsupported by design, blocked by a missing prerequisite, or failed.
- Exercise local-agent sources through the real server-issued job and daemon path. A direct package injection may diagnose the service boundary, but it does not prove that the documented daemon workflow works.
- For public-beta readiness, separately assess newcomer reproducibility, first-run UX, error messages, documentation accuracy, default data/privacy boundaries, and obvious exposure or secret-handling footguns. Give a conservative `GO`, `CONDITIONAL GO`, or `NO-GO`, with the smallest required fixes or disclosure language.
- Do not present fixture-backed LLM/provider behavior as production-provider validation. Record the fixture contract, then distinguish it from real connector credentials and model-provider coverage.
