from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "verify_image_tag.py"
VERSION = "1.2.3b4"
RELEASE_TAG = f"memforge-ai-v{VERSION}"


def _load_script():
    spec = importlib.util.spec_from_file_location("verify_image_tag", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


verify_image_tag = _load_script()
UNPUBLISHED_TAG = verify_image_tag.UNPUBLISHED_TAG


def _write_checkout(root: Path, *, api_tag: str, admin_ui_tag: str | None = None) -> None:
    (root / "pyproject.toml").write_text(f'[project]\nname = "memforge-ai"\nversion = "{VERSION}"\n', encoding="utf-8")
    (root / "compose.yaml").write_text(
        "# image: comments/memforge-api:ignored\n"
        "services:\n"
        "  api:\n"
        f"    image: ${{MEMFORGE_IMAGE_REGISTRY:-ghcr.io/acme}}/memforge-api:{api_tag}\n"
        "  admin-ui:\n"
        f"    image: ${{MEMFORGE_IMAGE_REGISTRY:-ghcr.io/acme}}/memforge-admin-ui:{admin_ui_tag or api_tag}\n",
        encoding="utf-8",
    )


def _git(root: Path, *arguments: str) -> None:
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "-c",
            "commit.gpgsign=false",
            *arguments,
        ],
        check=True,
        capture_output=True,
    )


def _commit(root: Path, message: str) -> None:
    _git(root, "add", ".")
    _git(root, "commit", "-m", message)


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "--quiet")
    return tmp_path


def test_this_repository_names_an_allowed_image_tag():
    tag = verify_image_tag.compose_image_tag(REPOSITORY_ROOT)

    assert tag in {UNPUBLISHED_TAG, verify_image_tag.project_version(REPOSITORY_ROOT)}


def test_a_commit_between_releases_names_the_unpublished_tag(checkout: Path):
    _write_checkout(checkout, api_tag=VERSION)
    _commit(checkout, "release")
    _git(checkout, "tag", RELEASE_TAG)
    _write_checkout(checkout, api_tag=UNPUBLISHED_TAG)
    _commit(checkout, "between releases")

    assert verify_image_tag.verify_checkout(checkout) == UNPUBLISHED_TAG


def test_a_raised_version_that_compose_does_not_name_is_refused(checkout: Path):
    _write_checkout(checkout, api_tag=UNPUBLISHED_TAG)
    _commit(checkout, "version raised, compose left behind")

    with pytest.raises(SystemExit, match=f"no release tag {RELEASE_TAG}"):
        verify_image_tag.verify_checkout(checkout)


def test_the_same_commit_would_be_refused_again_once_it_is_released(checkout: Path):
    _write_checkout(checkout, api_tag=UNPUBLISHED_TAG)
    _commit(checkout, "version raised, compose left behind")
    _git(checkout, "tag", RELEASE_TAG)

    with pytest.raises(SystemExit, match="must name its own version"):
        verify_image_tag.verify_release(checkout, RELEASE_TAG)


def test_a_release_commit_passes_before_and_after_it_is_tagged(checkout: Path):
    _write_checkout(checkout, api_tag=VERSION)
    _commit(checkout, "release")

    assert verify_image_tag.verify_checkout(checkout) == VERSION

    _git(checkout, "tag", RELEASE_TAG)

    assert verify_image_tag.verify_checkout(checkout) == VERSION


def test_a_later_commit_may_not_keep_a_released_tag(checkout: Path):
    _write_checkout(checkout, api_tag=VERSION)
    _commit(checkout, "release")
    _git(checkout, "tag", RELEASE_TAG)
    (checkout / "later.txt").write_text("later\n", encoding="utf-8")
    _commit(checkout, "later")

    with pytest.raises(SystemExit, match="set it back to"):
        verify_image_tag.verify_checkout(checkout)


@pytest.mark.parametrize("tag", ["latest", "dev"])
def test_a_tag_other_than_the_version_or_the_unpublished_tag_is_refused(checkout: Path, tag: str):
    _write_checkout(checkout, api_tag=tag)
    _commit(checkout, "floating")

    with pytest.raises(SystemExit, match=f"names image tag '{tag}'"):
        verify_image_tag.verify_checkout(checkout)


def test_both_images_must_name_one_tag(tmp_path: Path):
    _write_checkout(tmp_path, api_tag=VERSION, admin_ui_tag=UNPUBLISHED_TAG)

    with pytest.raises(SystemExit, match="different image tags"):
        verify_image_tag.compose_image_tag(tmp_path)


def test_a_missing_image_is_refused(tmp_path: Path):
    (tmp_path / "compose.yaml").write_text(
        "services:\n  api:\n    image: ghcr.io/acme/memforge-api:1.0\n", encoding="utf-8"
    )

    with pytest.raises(SystemExit, match="expected"):
        verify_image_tag.compose_image_tag(tmp_path)


def test_a_release_publishes_the_version_its_commit_names(tmp_path: Path):
    _write_checkout(tmp_path, api_tag=VERSION)

    assert verify_image_tag.verify_release(tmp_path, RELEASE_TAG) == VERSION


@pytest.mark.parametrize(
    "release_tag",
    [f"memforge-memory-v{VERSION}", f"v{VERSION}", "memforge-ai-v9.9.9"],
)
def test_a_tag_that_is_not_this_python_release_publishes_nothing(tmp_path: Path, release_tag: str):
    _write_checkout(tmp_path, api_tag=VERSION)

    with pytest.raises(SystemExit, match="does not match"):
        verify_image_tag.verify_release(tmp_path, release_tag)


def test_a_release_whose_commit_still_names_the_unpublished_tag_publishes_nothing(tmp_path: Path):
    _write_checkout(tmp_path, api_tag=UNPUBLISHED_TAG)

    with pytest.raises(SystemExit, match="must name its own version"):
        verify_image_tag.verify_release(tmp_path, RELEASE_TAG)
