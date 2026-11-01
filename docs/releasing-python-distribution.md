# Releasing the Python distribution

The public Python distribution is named `memforge-ai`. It intentionally keeps the
Python import package and installed command named `memforge`:

| Surface | Name |
| --- | --- |
| PyPI distribution | `memforge-ai` |
| Python import | `memforge` |
| Console command | `memforge` |

## Trusted Publisher

PyPI publishing uses GitHub Actions OpenID Connect and stores no PyPI API token
in the repository. Configure one PyPI pending publisher with these exact values:

| Field | Value |
| --- | --- |
| PyPI project | `memforge-ai` |
| GitHub owner | `dodo-labs-ai` |
| GitHub repository | `mem-forge` |
| Workflow | `publish-pypi.yml` |
| Environment | `pypi` |

The pending publisher does not reserve the project name. Complete the first
release immediately after configuring it.

## Release sequence

1. Update `project.version` in `pyproject.toml` and refresh `uv.lock`.
2. On the release commit, run the checks CI cannot run, with every provider
   key you hold, and keep their output for the release notes:

   ```bash
   uv run python scripts/provider_smoke.py
   scripts/compose_smoke.sh --quality
   ```

   The first checks that each provider still accepts MemForge's requests; the
   release notes name the providers that ran, and say "only the request format
   was verified for this release" for each one that did not. The second
   builds the checkout into a fresh Compose stack, runs the quick quality
   evaluation and checks that data survives a restart. Every step must pass,
   the quality step included rather than skipped, and the notes carry the
   table, the quality summary and the models it names.
3. Merge the version change after CI passes.
4. Create a GitHub release whose tag is exactly `memforge-ai-v<version>`, for
   example `memforge-ai-v0.1.55`.
5. The release workflow builds one wheel and one source distribution, verifies
   their metadata and `memforge` console entry point, and publishes them through
   the `pypi` environment.
6. Verify the public installation from a clean tool environment:

   ```bash
   pipx install memforge-ai
   memforge --help
   python -c 'from importlib.metadata import version; print(version("memforge-ai"))'
   ```

Plugin tags such as `memforge-memory-v0.1.55` do not match the Python release
tag contract and are rejected by the artifact verifier.
