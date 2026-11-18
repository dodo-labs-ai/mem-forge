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
| Every other commit | `dev-${COMPOSE_PROJECT_NAME}`, which is never published | Fails to pull and builds the checkout, under a name only that Compose project uses |

`scripts/verify_image_tag.py` checks this in two places. The `image-tag` CI job
fails a commit that raises `project.version` without naming it, and a commit after
a release that still names that release's version. Both publish workflows refuse
a release whose commit does not name its own version, before they publish
anything. `MEMFORGE_IMAGE_REGISTRY` moves the images to a mirror and leaves the
tag alone, so a `.env` kept across an upgrade cannot pin an old release.

Before the first release with images, once, in the repository's and the
organization's settings:

- Add `image-tag` to the required status checks of `main`'s ruleset. A check
  that is not required reports a failure and does not stop the merge.
- Allow public packages for the organization. A new GHCR package is private, even
  from a public repository: it inherits the repository's access but not its
  visibility. Step 6 of the release sequence makes each package public; until
  then an anonymous pull answers `denied`, and every newcomer falls back to a
  local build without knowing why.

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
3. Merge the version change after CI passes. Merge nothing else until step 7
   lands: `main`'s ruleset does not require a branch to be up to date, so a pull
   request tested before this merge would pass `image-tag` and leave `main`
   naming the release's version without that pull request's code in its images.
4. Create a GitHub release whose tag is exactly `memforge-ai-v<version>`, for
   example `memforge-ai-v0.1.55`.
5. Two workflows run, independently of each other. Each first verifies the tag
   and `compose.yaml`, and a release they refuse publishes nothing. A rerun checks
   out the same commit and is refused again, so correct the commit and release a
   new version. `publish-pypi.yml` then builds one wheel and one source
   distribution, verifies their metadata and `memforge` console entry point, and
   publishes them through the `pypi` environment. `publish-images.yml` builds and
   pushes both images. If one of them fails after its check, rerun it; the other
   stays published.
6. For the first release with images only, set both GHCR packages to public in
   their package settings. GitHub does not let a public package become private
   again.
7. Set the tag of both images in `compose.yaml` back to
   `dev-${COMPOSE_PROJECT_NAME}` and merge that. CI fails on every other change
   until it lands.
8. Verify the public installation from a clean tool environment. The last
   command must print the version just released:

   ```bash
   pipx install --python python3.12 --pip-args=--pre memforge-ai
   memforge --help
   pipx list --short | grep memforge-ai
   ```

9. Verify the public images as a newcomer gets them, logged out of `ghcr.io`, from
   a fresh clone of the release under a project name of its own. `docker compose
   pull` fails when the registry refuses or lacks an image, and replaces any
   image of that name on this machine; `--no-build` then keeps Compose from
   building one. Do not shorten this to `up --pull always`, which on a failed pull
   goes on with a local image of that name and exits 0. If MemForge already runs
   on this machine, set other ports in `.env` first.

   ```bash
   docker logout ghcr.io
   docker buildx imagetools inspect ghcr.io/dodo-labs-ai/memforge-api:<version>
   docker buildx imagetools inspect ghcr.io/dodo-labs-ai/memforge-admin-ui:<version>
   git clone --branch memforge-ai-v<version> https://github.com/dodo-labs-ai/mem-forge.git memforge-release-check
   cd memforge-release-check
   cp .env.example .env
   export COMPOSE_PROJECT_NAME=memforge-release-check
   docker compose pull && docker compose up -d --no-build --wait
   curl -fsS "http://$(docker compose port api 8765)/api/v1/health"
   curl -fsS -o /dev/null "http://$(docker compose port admin-ui 8080)/api/v1/health"
   docker compose down -v
   ```

   Then run the other platform the same way, under emulation. Docker Desktop
   provides it; on Linux, register QEMU first. Every project on this machine
   names the release's images by the same tag, so the run ends by removing the
   other platform's images; a later install of the release then pulls this
   machine's own.

   ```bash
   export DOCKER_DEFAULT_PLATFORM=linux/amd64 # or linux/arm64: the one this machine is not
   docker compose pull && docker compose up -d --no-build --wait
   docker compose exec -T api uname -m
   curl -fsS "http://$(docker compose port api 8765)/api/v1/health"
   curl -fsS -o /dev/null "http://$(docker compose port admin-ui 8080)/api/v1/health"
   docker compose down -v --rmi all
   unset DOCKER_DEFAULT_PLATFORM COMPOSE_PROJECT_NAME
   ```

Plugin tags such as `memforge-memory-v0.1.55` do not match the Python release
tag contract and are rejected by the artifact verifier and the image tag check.
