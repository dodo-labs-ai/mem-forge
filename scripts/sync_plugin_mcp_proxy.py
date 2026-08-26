#!/usr/bin/env python3
"""Generate packaged plugin runtime copies from canonical source files."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
from typing import Callable, Sequence


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
REPOSITORY_CONTEXT_CANONICAL = ROOT / "src" / "memforge" / "repository_context.py"
REPOSITORY_CONTEXT_GENERATED_COPIES = (
    ROOT / "integrations" / "codex" / "memforge-memory" / "scripts" / "memforge_repository_context.py",
    ROOT / "integrations" / "claude-code" / "memforge-memory" / "scripts" / "memforge_repository_context.py",
)
API_TARGET_CANONICAL = ROOT / "src" / "memforge" / "api_target.py"
API_TARGET_GENERATED_COPIES = (
    ROOT / "integrations" / "codex" / "memforge-memory" / "scripts" / "memforge_api_target.py",
    ROOT / "integrations" / "claude-code" / "memforge-memory" / "scripts" / "memforge_api_target.py",
)
CAPABILITY_DISCOVERY_CANONICAL = ROOT / "src" / "memforge" / "capability_discovery.py"
CAPABILITY_DISCOVERY_GENERATED_COPIES = (
    ROOT / "integrations" / "codex" / "memforge-memory" / "scripts" / "memforge_capability_discovery.py",
    ROOT / "integrations" / "claude-code" / "memforge-memory" / "scripts" / "memforge_capability_discovery.py",
)
PLUGIN_CONFIG_CANONICAL = ROOT / "src" / "memforge" / "plugin_config.py"
PLUGIN_CONFIG_GENERATED_COPIES = (
    ROOT / "integrations" / "codex" / "memforge-memory" / "scripts" / "memforge_plugin_config.py",
    ROOT / "integrations" / "claude-code" / "memforge-memory" / "scripts" / "memforge_plugin_config.py",
)
WORKSPACE_BINDINGS_CANONICAL = ROOT / "src" / "memforge" / "workspace_bindings.py"
WORKSPACE_BINDINGS_GENERATED_COPIES = (
    ROOT / "integrations" / "codex" / "memforge-memory" / "scripts" / "memforge_workspace_bindings.py",
    ROOT / "integrations" / "claude-code" / "memforge-memory" / "scripts" / "memforge_workspace_bindings.py",
)
SETUP_SKILL_CANONICAL = (
    ROOT / "integrations" / "codex" / "memforge-memory" / "skills" / "memforge-setup" / "SKILL.md"
)
SETUP_SKILL_GENERATED_COPIES = (
    ROOT / "integrations" / "claude-code" / "memforge-memory" / "skills" / "memforge-setup" / "SKILL.md",
)
SETUP_HELPER_CANONICAL = (
    ROOT
    / "integrations"
    / "codex"
    / "memforge-memory"
    / "skills"
    / "memforge-setup"
    / "scripts"
    / "workspace_config.py"
)
SETUP_HELPER_GENERATED_COPIES = (
    ROOT
    / "integrations"
    / "claude-code"
    / "memforge-memory"
    / "skills"
    / "memforge-setup"
    / "scripts"
    / "workspace_config.py",
)
DAEMON_SETUP_SKILL_CANONICAL = (
    ROOT
    / "integrations"
    / "codex"
    / "memforge-memory"
    / "skills"
    / "memforge-daemon-setup"
    / "SKILL.md"
)
DAEMON_SETUP_SKILL_GENERATED_COPIES = (
    ROOT
    / "integrations"
    / "claude-code"
    / "memforge-memory"
    / "skills"
    / "memforge-daemon-setup"
    / "SKILL.md",
)
DAEMON_SETUP_OPENAI_CANONICAL = (
    ROOT
    / "integrations"
    / "codex"
    / "memforge-memory"
    / "skills"
    / "memforge-daemon-setup"
    / "agents"
    / "openai.yaml"
)
DAEMON_SETUP_OPENAI_GENERATED_COPIES = (
    ROOT
    / "integrations"
    / "claude-code"
    / "memforge-memory"
    / "skills"
    / "memforge-daemon-setup"
    / "agents"
    / "openai.yaml",
)
DAEMON_SETUP_OPERATIONS_CANONICAL = (
    ROOT
    / "integrations"
    / "codex"
    / "memforge-memory"
    / "skills"
    / "memforge-daemon-setup"
    / "references"
    / "operations.md"
)
DAEMON_SETUP_OPERATIONS_GENERATED_COPIES = (
    ROOT
    / "integrations"
    / "claude-code"
    / "memforge-memory"
    / "skills"
    / "memforge-daemon-setup"
    / "references"
    / "operations.md",
)
CANONICAL_PLUGIN_CONFIG_TARGET_IMPORT = b"""if __package__:
    from .api_target import MemForgeTarget, build_target
    from .capability_discovery import discover_target
else:  # pragma: no cover - direct file load used by packaged integrations
    from memforge.api_target import MemForgeTarget, build_target
    from memforge.capability_discovery import discover_target
"""
PACKAGED_PLUGIN_CONFIG_TARGET_IMPORT = b"""if __package__:
    from .memforge_api_target import MemForgeTarget, build_target
    from .memforge_capability_discovery import discover_target
else:  # pragma: no cover - direct file load used by packaged integrations
    from memforge_api_target import MemForgeTarget, build_target
    from memforge_capability_discovery import discover_target
"""
CANONICAL_CAPABILITY_TARGET_IMPORT = b"from memforge.api_target import (\n"
PACKAGED_CAPABILITY_TARGET_IMPORT = b"from memforge_api_target import (\n"


def packaged_plugin_config_content(canonical_content: bytes) -> bytes:
    """Rewrite the canonical package import for the standalone plugin layout."""
    if canonical_content.count(CANONICAL_PLUGIN_CONFIG_TARGET_IMPORT) != 1:
        raise ValueError("canonical plugin config target import block is missing or ambiguous")
    return canonical_content.replace(
        CANONICAL_PLUGIN_CONFIG_TARGET_IMPORT,
        PACKAGED_PLUGIN_CONFIG_TARGET_IMPORT,
        1,
    )


def packaged_capability_discovery_content(canonical_content: bytes) -> bytes:
    """Rewrite the capability parser import for the standalone plugin layout."""
    if canonical_content.count(CANONICAL_CAPABILITY_TARGET_IMPORT) != 1:
        raise ValueError("canonical capability target import is missing or ambiguous")
    return canonical_content.replace(
        CANONICAL_CAPABILITY_TARGET_IMPORT,
        PACKAGED_CAPABILITY_TARGET_IMPORT,
        1,
    )


PLUGIN_RUNTIME_FILES = (
    (CANONICAL, GENERATED_COPIES, None),
    (HOOK_CANONICAL, HOOK_GENERATED_COPIES, None),
    (REPO_IDENTITY_CANONICAL, REPO_IDENTITY_GENERATED_COPIES, None),
    (REPOSITORY_CONTEXT_CANONICAL, REPOSITORY_CONTEXT_GENERATED_COPIES, None),
    (API_TARGET_CANONICAL, API_TARGET_GENERATED_COPIES, None),
    (
        CAPABILITY_DISCOVERY_CANONICAL,
        CAPABILITY_DISCOVERY_GENERATED_COPIES,
        packaged_capability_discovery_content,
    ),
    (
        PLUGIN_CONFIG_CANONICAL,
        PLUGIN_CONFIG_GENERATED_COPIES,
        packaged_plugin_config_content,
    ),
    (WORKSPACE_BINDINGS_CANONICAL, WORKSPACE_BINDINGS_GENERATED_COPIES, None),
    (SETUP_SKILL_CANONICAL, SETUP_SKILL_GENERATED_COPIES, None),
    (SETUP_HELPER_CANONICAL, SETUP_HELPER_GENERATED_COPIES, None),
    (DAEMON_SETUP_SKILL_CANONICAL, DAEMON_SETUP_SKILL_GENERATED_COPIES, None),
    (DAEMON_SETUP_OPENAI_CANONICAL, DAEMON_SETUP_OPENAI_GENERATED_COPIES, None),
    (DAEMON_SETUP_OPERATIONS_CANONICAL, DAEMON_SETUP_OPERATIONS_GENERATED_COPIES, None),
)


def synchronize_plugin_copies(
    canonical: Path,
    generated_copies: Sequence[Path],
    *,
    check: bool,
    transform: Callable[[bytes], bytes] | None = None,
) -> tuple[Path, ...]:
    """Return stale copies, updating them from canonical unless check is true."""
    canonical_content = canonical.read_bytes()
    generated_content = transform(canonical_content) if transform else canonical_content
    stale = tuple(
        path
        for path in generated_copies
        if not path.exists() or path.read_bytes() != generated_content
    )
    if check:
        return stale

    for path in stale:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(generated_content)
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
        for canonical, generated_copies, transform in PLUGIN_RUNTIME_FILES
        for path in synchronize_plugin_copies(
            canonical,
            generated_copies,
            check=args.check,
            transform=transform,
        )
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
