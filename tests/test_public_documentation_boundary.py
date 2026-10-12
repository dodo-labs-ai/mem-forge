"""The public repository carries user documentation; design records are kept elsewhere."""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
DESIGN_RECORD_PATHS = (
    "CONTEXT.md",
    "docs/adr",
    "docs/design",
    "docs/integrations/agent-clients.md",
    "docs/research",
    "docs/superpowers",
)


def _tracked_files(relative_path: str) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--", relative_path],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip("not a Git checkout")
    return result.stdout.split()


@pytest.mark.parametrize("relative_path", DESIGN_RECORD_PATHS)
def test_design_records_are_not_tracked(relative_path: str) -> None:
    assert _tracked_files(relative_path) == []
