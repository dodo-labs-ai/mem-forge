#!/usr/bin/env python3
"""Verify the publishable MemForge Python distributions without installing them."""

from __future__ import annotations

import argparse
from email.parser import Parser
from pathlib import Path
import tarfile
import tomllib
import zipfile


EXPECTED_DISTRIBUTION_NAME = "memforge-ai"
EXPECTED_ARTIFACT_STEM = "memforge_ai"
EXPECTED_CONSOLE_SCRIPT = "memforge = memforge.main:cli"
RELEASE_TAG_PREFIX = "memforge-ai-v"


def _project_metadata(project_root: Path) -> tuple[str, str]:
    with (project_root / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]
    return str(project["name"]), str(project["version"])


def _single_artifact(dist_dir: Path, pattern: str) -> Path:
    artifacts = sorted(dist_dir.glob(pattern))
    if len(artifacts) != 1:
        raise SystemExit(f"expected exactly one {pattern} artifact in {dist_dir}, found {len(artifacts)}")
    return artifacts[0]


def _metadata_fields(raw: bytes) -> tuple[str, str]:
    parsed = Parser().parsestr(raw.decode("utf-8"))
    return str(parsed["Name"]), str(parsed["Version"])


def _verify_wheel(wheel: Path, *, version: str) -> None:
    expected_prefix = f"{EXPECTED_ARTIFACT_STEM}-{version}-"
    if not wheel.name.startswith(expected_prefix):
        raise SystemExit(f"unexpected wheel filename: {wheel.name}; expected prefix {expected_prefix}")

    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata_path = next((name for name in names if name.endswith(".dist-info/METADATA")), None)
        entry_points_path = next((name for name in names if name.endswith(".dist-info/entry_points.txt")), None)
        if metadata_path is None or entry_points_path is None:
            raise SystemExit("wheel is missing METADATA or entry_points.txt")
        name, artifact_version = _metadata_fields(archive.read(metadata_path))
        if (name, artifact_version) != (EXPECTED_DISTRIBUTION_NAME, version):
            raise SystemExit(f"unexpected wheel metadata: Name={name!r} Version={artifact_version!r}")
        entry_points = archive.read(entry_points_path).decode("utf-8")
        if EXPECTED_CONSOLE_SCRIPT not in entry_points:
            raise SystemExit("wheel does not expose the memforge console script")
        required_members = {
            "memforge/__init__.py",
            "memforge/main.py",
            "memforge/interactive_cli/index.mjs",
        }
        missing = sorted(required_members.difference(names))
        if missing:
            raise SystemExit(f"wheel is missing required packaged files: {', '.join(missing)}")


def _verify_sdist(sdist: Path, *, version: str) -> None:
    expected_name = f"{EXPECTED_ARTIFACT_STEM}-{version}.tar.gz"
    if sdist.name != expected_name:
        raise SystemExit(f"unexpected sdist filename: {sdist.name}; expected {expected_name}")

    with tarfile.open(sdist, "r:gz") as archive:
        pkg_info = next((member for member in archive.getmembers() if member.name.endswith("/PKG-INFO")), None)
        if pkg_info is None:
            raise SystemExit("sdist is missing PKG-INFO")
        extracted = archive.extractfile(pkg_info)
        if extracted is None:
            raise SystemExit("could not read sdist PKG-INFO")
        name, artifact_version = _metadata_fields(extracted.read())
        if (name, artifact_version) != (EXPECTED_DISTRIBUTION_NAME, version):
            raise SystemExit(f"unexpected sdist metadata: Name={name!r} Version={artifact_version!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dist_dir", type=Path)
    parser.add_argument("--release-tag")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    name, version = _project_metadata(project_root)
    if name != EXPECTED_DISTRIBUTION_NAME:
        raise SystemExit(f"pyproject distribution name is {name!r}, expected {EXPECTED_DISTRIBUTION_NAME!r}")
    if args.release_tag and args.release_tag != f"{RELEASE_TAG_PREFIX}{version}":
        raise SystemExit(
            f"release tag {args.release_tag!r} does not match {RELEASE_TAG_PREFIX}{version!s}"
        )

    dist_dir = args.dist_dir.resolve()
    _verify_wheel(_single_artifact(dist_dir, "*.whl"), version=version)
    _verify_sdist(_single_artifact(dist_dir, "*.tar.gz"), version=version)
    print(f"verified {EXPECTED_DISTRIBUTION_NAME} {version} release artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
