"""Shared configuration helpers for installable MemForge agent plugins.

Hooks are not launched as MCP servers, so they do not automatically inherit MCP
stdio process environment. Keep a tiny stdlib resolver here so hooks and MCP
tools agree on the same endpoint, token, and workspace without copying secrets
into hook command strings or registering a second manual MCP server.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Mapping

if __package__:
    from .memforge_api_target import MemForgeTarget, build_target
else:  # pragma: no cover - direct file load used by packaged integrations
    from memforge_api_target import MemForgeTarget, build_target


_CONFIG_CACHE: dict[str, str] | None = None
_REPOSITORY_CONFIG_PATH = Path(".memforge") / "config.toml"
_BASIC_TOML_STRING = re.compile(
    r'(?:[^"\\\x00-\x1f]|\\(?:["\\btnfr]|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8}))*\Z'
)


def configured_target(repository_paths: Iterable[str | os.PathLike[str]] = ()) -> MemForgeTarget:
    origin = _configured_value("MEMFORGE_API_URL", "").strip() or None
    workspace = _configured_workspace_id(repository_paths)
    return build_target(origin=origin, workspace_id=workspace)


def configured_api_token() -> str:
    return _configured_value("MEMFORGE_API_TOKEN", "").strip()


def _configured_workspace_id(repository_paths: Iterable[str | os.PathLike[str]]) -> str | None:
    repository_value = _repository_workspace_id(repository_paths)
    if repository_value is not None:
        return repository_value
    process_value = os.getenv("MEMFORGE_WORKSPACE_ID")
    if process_value:
        return process_value.strip() or None
    return _codex_memforge_config().get("MEMFORGE_WORKSPACE_ID", "").strip() or None


def _configured_value(name: str, default: str) -> str:
    value = os.getenv(name)
    if value:
        return value
    return _codex_memforge_config().get(name, default)


def _repository_workspace_id(repository_paths: Iterable[str | os.PathLike[str]]) -> str | None:
    workspace_ids: set[str] = set()
    repository_roots: set[Path] = set()
    for raw_path in repository_paths:
        repository_root = _repository_root(raw_path)
        if repository_root is None:
            return None
        if repository_root in repository_roots:
            continue
        repository_roots.add(repository_root)
        config_path = repository_root / _REPOSITORY_CONFIG_PATH
        try:
            text = config_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            return None
        configured_workspace = _parse_toml_string_table(text, "memforge").get("workspace_id")
        if configured_workspace is None:
            return None
        workspace_id = configured_workspace.strip()
        if not workspace_id:
            return None
        workspace_ids.add(workspace_id)
    if repository_roots and len(workspace_ids) == 1:
        return next(iter(workspace_ids))
    return None


def _repository_root(raw_path: str | os.PathLike[str]) -> Path | None:
    try:
        candidate = Path(raw_path).expanduser().resolve(strict=False)
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    if candidate.is_file():
        candidate = candidate.parent
    for directory in (candidate, *candidate.parents):
        if (directory / ".git").exists():
            return directory
    return None


def _codex_memforge_config() -> Mapping[str, str]:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    config_path = _codex_config_path()
    if config_path is None:
        _CONFIG_CACHE = {}
        return _CONFIG_CACHE

    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError:
        _CONFIG_CACHE = {}
        return _CONFIG_CACHE

    _CONFIG_CACHE = _parse_toml_string_table(text, "memforge")
    return _CONFIG_CACHE


def _codex_config_path() -> Path | None:
    explicit = os.getenv("MEMFORGE_CODEX_CONFIG")
    if explicit:
        return Path(explicit).expanduser()
    codex_home = os.getenv("CODEX_HOME")
    if codex_home:
        return Path(codex_home).expanduser() / "config.toml"
    default = Path.home() / ".codex" / "config.toml"
    return default if default.exists() else None


def _parse_toml_string_table(text: str, table_name: str) -> dict[str, str]:
    table_header = f"[{table_name}]"
    current: str | None = None
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line
            continue
        if current != table_header:
            continue
        match = re.match(r"([A-Za-z0-9_]+)\s*=\s*\"(.*)\"\s*$", line)
        if match:
            try:
                values[match.group(1)] = _unescape_basic_toml_string(match.group(2))
            except (UnicodeDecodeError, ValueError):
                continue
    return values


def _unescape_basic_toml_string(value: str) -> str:
    if _BASIC_TOML_STRING.fullmatch(value) is None:
        raise ValueError("invalid TOML basic string")
    return bytes(value, "utf-8").decode("unicode_escape")
