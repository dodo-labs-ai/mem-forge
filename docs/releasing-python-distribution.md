# Releasing the Python distribution and container images

One GitHub release publishes the Python distribution and the container images.
The public Python distribution is named `memforge-ai`. It intentionally keeps the
Python import package and installed command named `memforge`:

| Surface | Name |
| --- | --- |
| PyPI distribution | `memforge-ai` |
| Python import | `memforge` |
| Console command | `memforge` |
| API image | `ghcr.io/dodo-labs-ai/memforge-api:<version>` |
| Admin UI image | `ghcr.io/dodo-labs-ai/memforge-admin-ui:<version>` |

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

## Container images

`publish-images.yml` builds both images for `linux/amd64` and `linux/arm64` and
pushes them to the GitHub Container Registry with the run's `GITHUB_TOKEN`, so
the repository stores no registry credential. It publishes the exact version
only: there is no `latest` tag, because a moving tag would let a checkout run
another release's code.

The image tag in `compose.yaml` belongs to the checkout:

| Checkout | Tag it names | `docker compose up` |
| --- | --- | --- |
| The release commit | The release's version | Pulls that release's images |
| Every other commit | `dev`, which is never published | Fails to pull and builds the checkout |

`scripts/verify_image_tag.py` holds every commit to this. CI fails a commit
after a release that still names that release's version, and the publish
workflow refuses a release whose commit does not name its own version.
`MEMFORGE_IMAGE_REGISTRY` moves the images to a mirror and leaves the tag alone,
so a `.env` kept across an upgrade cannot pin an old release.

A new GHCR package is private, even from a public repository: it inherits the
repository's access but not its visibility. The first time an image is published,
set that package's visibility to public in its package settings, which needs the
organization to allow public packages. Until then an anonymous pull answers
`denied`, and every newcomer falls back to a local build without knowing why.

## Release sequence

1. Update `project.version` in `pyproject.toml`, refresh `uv.lock`, and set the
   tag of both images in `compose.yaml` to the same version.
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
5. Two workflows run, independently of each other. `publish-pypi.yml` builds one
   wheel and one source distribution, verifies their metadata and `memforge`
   console entry point, and publishes them through the `pypi` environment.
   `publish-images.yml` verifies the tag and `compose.yaml`, then builds and
   pushes both images. Rerun whichever failed; the other stays published.
6. Set the tag of both images in `compose.yaml` back to `dev` and merge that. CI
   fails on every other change until it lands.
7. Verify the public installation from a clean tool environment. The last
   command must print the version just released:

   ```bash
   pipx install --python python3.12 --pip-args=--pre memforge-ai
   memforge --help
   pipx list --short | grep memforge-ai
   ```

8. Verify the public images while logged out of `ghcr.io`, from a fresh clone of
   the release under a project name of its own. `--no-build` turns the fallback
   to a local build into an error, so this proves the published images start. If
   MemForge already runs on this machine, set other ports in `.env` first.

   ```bash
   docker buildx imagetools inspect ghcr.io/dodo-labs-ai/memforge-api:<version>
   docker buildx imagetools inspect ghcr.io/dodo-labs-ai/memforge-admin-ui:<version>
   git clone --branch memforge-ai-v<version> https://github.com/dodo-labs-ai/mem-forge.git memforge-release-check
   cd memforge-release-check
   cp .env.example .env
   docker compose --project-name memforge-release-check up -d --no-build --wait
   curl -fsS http://127.0.0.1:8765/api/v1/health
   docker compose --project-name memforge-release-check down -v
   ```

Plugin tags such as `memforge-memory-v0.1.55` do not match the Python release
tag contract and are rejected by the artifact verifier and the image tag check.
