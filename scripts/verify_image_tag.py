#!/usr/bin/env python3
"""Verify that compose.yaml names the images this checkout should run.

A release commit names its own version, so a clone of that release pulls its
images. Every other commit names a dev tag of its Compose project, which is never
published, so Compose builds that checkout under a name no other project shares.

A version that outlived its release would make later commits run that release's
images, so only the release's own commit may name it. A raised version that
compose.yaml does not name would pass here and be refused only once it is
released, so the commit that raises it must name it. This needs the checkout's
release tags.

With ``--release-tag`` the script checks a release instead, and prints the
version the publish workflows tag the images with.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import tomllib


RELEASE_TAG_PREFIX = "memforge-ai-v"
UNPUBLISHED_TAG = "dev-${COMPOSE_PROJECT_NAME}"
IMAGES = ("memforge-api", "memforge-admin-ui")
_IMAGE_LINE = re.compile(r"^\s*image:\s*\S*/(?P<image>memforge-[a-z-]+):(?P<tag>\S+)\s*$", re.MULTILINE)


def project_version(root: Path) -> str:
    with (root / "pyproject.toml").open("rb") as handle:
        return str(tomllib.load(handle)["project"]["version"])


def compose_image_tag(root: Path) -> str:
    """The one tag compose.yaml names for every MemForge image."""

    named = {
        match["image"]: match["tag"]
        for match in _IMAGE_LINE.finditer((root / "compose.yaml").read_text(encoding="utf-8"))
    }
    if sorted(named) != sorted(IMAGES):
        raise SystemExit(f"compose.yaml names images {sorted(named)}, expected {sorted(IMAGES)}")
    tags = set(named.values())
    if len(tags) != 1:
        raise SystemExit(f"compose.yaml names different image tags: {named}")
    return tags.pop()


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(root), *arguments], capture_output=True, text=True, check=False)


def _release_commit(root: Path, version: str) -> str | None:
    found = _git(root, "rev-parse", "--quiet", "--verify", f"refs/tags/{RELEASE_TAG_PREFIX}{version}^{{commit}}")
    return found.stdout.strip() if found.returncode == 0 else None


def verify_checkout(root: Path) -> str:
    tag = compose_image_tag(root)
    version = project_version(root)
    released_at = _release_commit(root, version)
    if tag == UNPUBLISHED_TAG:
        if released_at is None:
            raise SystemExit(
                f"version {version} has no release tag {RELEASE_TAG_PREFIX}{version} and compose.yaml names"
                f" {UNPUBLISHED_TAG!r}: the commit that raises the version names it as the image tag,"
                " or its release publishes no images"
            )
        return tag
    if tag != version:
        raise SystemExit(
            f"compose.yaml names image tag {tag!r}: a release commit names its version {version!r},"
            f" and any other commit names {UNPUBLISHED_TAG!r}"
        )
    if released_at is not None and _git(root, "rev-parse", "HEAD").stdout.strip() != released_at:
        raise SystemExit(
            f"compose.yaml still names image tag {tag!r}, released at {released_at}:"
            f" set it back to {UNPUBLISHED_TAG!r} so that this checkout does not run that release's images"
        )
    return tag


def verify_release(root: Path, release_tag: str) -> str:
    version = project_version(root)
    if release_tag != f"{RELEASE_TAG_PREFIX}{version}":
        raise SystemExit(f"release tag {release_tag!r} does not match {RELEASE_TAG_PREFIX}{version}")
    tag = compose_image_tag(root)
    if tag != version:
        raise SystemExit(f"compose.yaml names image tag {tag!r}: release {version} must name its own version")
    return version


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-tag")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    print(verify_release(root, args.release_tag) if args.release_tag else verify_checkout(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
