"""Shared configuration helpers for installable MemForge agent plugins.

Hooks are not launched as MCP servers, so they do not automatically inherit MCP
stdio process environment. Keep a tiny stdlib resolver here so hooks and MCP
tools agree on the same endpoint, token, and workspace without copying secrets
into hook command strings or registering a second manual MCP server.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Mapping

if __package__:
    from .memforge_api_target import MemForgeTarget, build_target
else:  # pragma: no cover - direct file load used by packaged integrations
    from memforge_api_target import MemForgeTarget, build_target


_CONFIG_CACHE: dict[str, str] | None = None


def configured_target() -> MemForgeTarget:
    origin = _configured_value("MEMFORGE_API_URL", "").strip() or None
    workspace = _configured_value("MEMFORGE_WORKSPACE_ID", "").strip() or None
    return build_target(origin=origin, workspace_id=workspace)


def configured_api_token() -> str:
    return _configured_value("MEMFORGE_API_TOKEN", "").strip()


def _configured_value(name: str, default: str) -> str:
    value = os.getenv(name)
    if value:
        return value
    return _codex_memforge_config().get(name, default)


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
            values[match.group(1)] = _unescape_basic_toml_string(match.group(2))
    return values


def _unescape_basic_toml_string(value: str) -> str:
    return bytes(value, "utf-8").decode("unicode_escape")
