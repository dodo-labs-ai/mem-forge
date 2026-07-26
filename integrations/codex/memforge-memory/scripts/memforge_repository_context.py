"""Resolve local coding-workspace context into portable repository identity."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Literal, Sequence
from urllib.parse import unquote, urlparse

try:
    from .repo_identity import normalize_repo_identifier
except ImportError:  # pragma: no cover - copied plugin package or direct file load
    try:
        from memforge_repo_identity import normalize_repo_identifier
    except ImportError:
        import importlib.util

        _repo_identity_path = Path(__file__).with_name("memforge_repo_identity.py")
        if not _repo_identity_path.exists():
            _repo_identity_path = Path(__file__).with_name("repo_identity.py")
        _repo_identity_spec = importlib.util.spec_from_file_location("memforge_repo_identity", _repo_identity_path)
        if _repo_identity_spec is None or _repo_identity_spec.loader is None:
            raise
        _repo_identity_module = importlib.util.module_from_spec(_repo_identity_spec)
        _repo_identity_spec.loader.exec_module(_repo_identity_module)
        normalize_repo_identifier = _repo_identity_module.normalize_repo_identifier


RepositoryContextState = Literal["exact", "absent", "ambiguous"]
RepositoryContextSource = Literal["tool_argument", "mcp_roots", "none"]


@dataclass(frozen=True)
class RepositoryContext:
    """Portable repository context resolved on the agent host."""

    state: RepositoryContextState
    repo_identifier: str | None
    source: RepositoryContextSource


def resolve_repository_context(
    *,
    working_directory: str | None = None,
    root_paths: Sequence[str] = (),
) -> RepositoryContext:
    """Resolve explicit per-call context first, then compatible MCP roots."""
    if working_directory is not None:
        repo_identifier = _repo_identifier_from_location(working_directory)
        if repo_identifier:
            return RepositoryContext("exact", repo_identifier, "tool_argument")
        return RepositoryContext("absent", None, "tool_argument")

    identifiers = {
        repo_identifier
        for root_path in root_paths
        if (repo_identifier := _repo_identifier_from_location(root_path))
    }
    if len(identifiers) == 1:
        return RepositoryContext("exact", next(iter(identifiers)), "mcp_roots")
    if len(identifiers) > 1:
        return RepositoryContext("ambiguous", None, "mcp_roots")
    return RepositoryContext("absent", None, "mcp_roots" if root_paths else "none")


def _repo_identifier_from_location(location: str) -> str | None:
    path = _local_path(location)
    if path is None:
        return None
    remote = _git_value(["git", "remote", "get-url", "origin"], cwd=path)
    return normalize_repo_identifier(remote)


def _local_path(location: str) -> Path | None:
    value = str(location or "").strip()
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    parsed = urlparse(value)
    if parsed.scheme:
        if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
            return None
        value = unquote(parsed.path)
    path = Path(value)
    if not path.is_absolute():
        return None
    return path


def _git_value(command: list[str], *, cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        )
    except Exception:
        return None
    value = result.stdout.strip()
    return value or None
