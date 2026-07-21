#!/usr/bin/env python3
"""Generate packaged plugin runtime copies from canonical source files."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "src" / "memforge" / "plugin_mcp_proxy.py"
GENERATED_COPIES = (
    ROOT / "integrations" / "codex" / "memforge-memory" / "scripts" / "memforge_mcp.py",
    ROOT / "integrations" / "claude-code" / "memforge-memory" / "scripts" / "memforge_mcp.py",
)
HOOK_CANONICAL = ROOT / "src" / "memforge" / "hook_adapter.py"
HOOK_GENERATED_COPIES = (
    ROOT / "integrations" / "codex" / "memforge-memory" / "scripts" / "memforge_hook_adapter.py",
    ROOT / "integrations" / "claude-code" / "memforge-memory" / "scripts" / "memforge_hook_adapter.py",
)
REPO_IDENTITY_CANONICAL = ROOT / "src" / "memforge" / "repo_identity.py"
REPO_IDENTITY_GENERATED_COPIES = (
    ROOT / "integrations" / "codex" / "memforge-memory" / "scripts" / "memforge_repo_identity.py",
    ROOT / "integrations" / "claude-code" / "memforge-memory" / "scripts" / "memforge_repo_identity.py",
)
PLUGIN_RUNTIME_FILES = (
    (CANONICAL, GENERATED_COPIES),
    (HOOK_CANONICAL, HOOK_GENERATED_COPIES),
    (REPO_IDENTITY_CANONICAL, REPO_IDENTITY_GENERATED_COPIES),
)


def synchronize_plugin_copies(
    canonical: Path,
    generated_copies: Sequence[Path],
    *,
    check: bool,
) -> tuple[Path, ...]:
    """Return stale copies, updating them from canonical unless check is true."""
    canonical_content = canonical.read_bytes()
    stale = tuple(
        path
        for path in generated_copies
        if not path.exists() or path.read_bytes() != canonical_content
    )
    if check:
        return stale

    for path in stale:
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(canonical, path)
        shutil.copymode(canonical, path)
    return stale


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate packaged plugin runtime copies from canonical MemForge sources.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if a generated copy differs instead of updating it",
    )
    args = parser.parse_args(argv)

    stale = tuple(
        path
        for canonical, generated_copies in PLUGIN_RUNTIME_FILES
        for path in synchronize_plugin_copies(canonical, generated_copies, check=args.check)
    )
    if not stale:
        print("Packaged plugin runtime copies are already in sync.")
        return 0
    if args.check:
        print("Packaged plugin runtime copies are stale:", file=sys.stderr)
        for path in stale:
            print(f"- {_display_path(path)}", file=sys.stderr)
        print("Run: uv run python scripts/sync_plugin_mcp_proxy.py", file=sys.stderr)
        return 1

    for path in stale:
        print(f"Synchronized {_display_path(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
