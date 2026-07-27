#!/usr/bin/env python3
"""Plan and apply safe MemForge workspace-selection changes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import tomllib
from typing import Any


PLUGIN_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PLUGIN_ROOT / "scripts"))

from memforge_api_target import TargetConfigurationError, build_target  # noqa: E402


class SetupError(ValueError):
    """A safe, user-actionable setup failure."""


def _codex_config_path() -> Path:
    explicit = os.getenv("MEMFORGE_CODEX_CONFIG")
    if explicit:
        return Path(explicit).expanduser()
    codex_home = os.getenv("CODEX_HOME")
    if codex_home:
        return Path(codex_home).expanduser() / "config.toml"
    return Path.home() / ".codex" / "config.toml"


def _global_config_path(client: str) -> Path:
    if client == "codex":
        return _codex_config_path()
    return Path.home() / ".claude" / "settings.json"


def _load_codex_values(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise SetupError(f"invalid Codex config: {path}") from exc
    table = document.get("memforge")
    if not isinstance(table, dict):
        return {}
    return {key: value for key, value in table.items() if isinstance(value, str)}


def _load_claude_document(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SetupError(f"invalid Claude Code settings: {path}") from exc
    if not isinstance(document, dict):
        raise SetupError(f"Claude Code settings must contain a JSON object: {path}")
    return document


def _global_values(client: str) -> dict[str, str]:
    path = _global_config_path(client)
    if client == "codex":
        return _load_codex_values(path)
    environment = _load_claude_document(path).get("env")
    if not isinstance(environment, dict):
        return {}
    return {key: value for key, value in environment.items() if isinstance(value, str)}


def _git_root(repository_path: str) -> Path:
    try:
        result = subprocess.run(
            ["git", "-C", repository_path, "rev-parse", "--show-toplevel"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SetupError(f"not a Git repository: {repository_path}") from exc
    return Path(result.stdout.strip()).resolve()


def _repository_config(root: Path) -> Path:
    return root / ".memforge" / "config.toml"


def _repository_workspace(path: Path) -> tuple[str, str | None]:
    if not path.exists():
        return "missing", None
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError):
        return "invalid", None
    table = document.get("memforge")
    if not isinstance(table, dict):
        return "invalid", None
    value = table.get("workspace_id")
    if not isinstance(value, str) or not value.strip():
        return "invalid", None
    return "valid", value.strip()


def _contains_credential_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).casefold().replace("-", "_")
            if any(marker in normalized for marker in ("token", "password", "secret", "api_key", "credential")):
                return True
            if _contains_credential_key(item):
                return True
    elif isinstance(value, list):
        return any(_contains_credential_key(item) for item in value)
    return False


def _assert_safe_repository_config(path: Path) -> None:
    if not path.exists():
        return
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise SetupError(f"repository config is invalid and will not be overwritten: {path}") from exc
    if _contains_credential_key(document):
        raise SetupError(f"repository config contains credential-like keys and will not be changed: {path}")


def _target(origin: str | None, workspace: str | None) -> dict[str, str | None]:
    try:
        target = build_target(origin=origin, workspace_id=workspace)
    except TargetConfigurationError as exc:
        raise SetupError(f"invalid MemForge target: {exc.code}") from exc
    return {
        "edition": target.edition.value,
        "origin": target.origin,
        "workspace_id": target.workspace_id,
        "api_base": target.workspace_api_base,
    }


def _origin(client: str) -> str | None:
    process_value = os.getenv("MEMFORGE_API_URL")
    if process_value:
        return process_value.strip() or None
    return _global_values(client).get("MEMFORGE_API_URL", "").strip() or None


def _fallback_workspace(client: str) -> tuple[str, str | None]:
    process_value = os.getenv("MEMFORGE_WORKSPACE_ID")
    if process_value:
        return "process", process_value.strip() or None
    return "global", _global_values(client).get("MEMFORGE_WORKSPACE_ID", "").strip() or None


def _effective(client: str, root: Path | None) -> tuple[str, str | None, str]:
    if root is not None:
        status, repository_value = _repository_workspace(_repository_config(root))
        if repository_value is not None:
            return "repository", repository_value, status
    source, workspace = _fallback_workspace(client)
    return source, workspace, "missing" if root is None else _repository_workspace(_repository_config(root))[0]


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _updated_toml_value(text: str, key: str, value: str | None) -> str:
    lines = text.splitlines()
    table_start: int | None = None
    table_end = len(lines)
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "[memforge]":
            table_start = index
            continue
        if table_start is not None and stripped.startswith("[") and stripped.endswith("]"):
            table_end = index
            break
    if table_start is None:
        if value is None:
            return text
        prefix = "\n" if text and not text.endswith("\n") else ""
        return f"{text}{prefix}\n[memforge]\n{key} = {_toml_string(value)}\n"

    key_pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
    for index in range(table_start + 1, table_end):
        if key_pattern.match(lines[index]):
            if value is None:
                del lines[index]
            else:
                lines[index] = f"{key} = {_toml_string(value)}"
            return "\n".join(lines) + ("\n" if text.endswith("\n") else "")
    if value is not None:
        lines.insert(table_end, f"{key} = {_toml_string(value)}")
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode if path.exists() else None
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    if mode is not None:
        temporary.chmod(mode)
    os.replace(temporary, path)


def _set_codex_global(path: Path, workspace: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if path.exists():
        _load_codex_values(path)
    _atomic_write(path, _updated_toml_value(text, "MEMFORGE_WORKSPACE_ID", workspace))


def _set_claude_global(path: Path, workspace: str) -> None:
    document = _load_claude_document(path)
    environment = document.get("env")
    if environment is None:
        environment = {}
        document["env"] = environment
    if not isinstance(environment, dict):
        raise SetupError(f"Claude Code env must contain a JSON object: {path}")
    environment["MEMFORGE_WORKSPACE_ID"] = workspace
    _atomic_write(path, json.dumps(document, indent=2, ensure_ascii=False) + "\n")


def _is_tracked(root: Path, path: Path) -> bool:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--error-unmatch", "--", str(path.relative_to(root))],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _is_ignored(root: Path, path: Path) -> bool:
    result = subprocess.run(
        ["git", "-C", str(root), "check-ignore", "-q", "--", str(path.relative_to(root))],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _git_exclude_path(root: Path) -> Path:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--git-path", "info/exclude"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=5,
    )
    path = Path(result.stdout.strip())
    return path if path.is_absolute() else root / path


def _ensure_locally_ignored(root: Path, config_path: Path) -> Path | None:
    if _is_ignored(root, config_path):
        return None
    exclude_path = _git_exclude_path(root)
    pattern = "/.memforge/config.toml"
    text = exclude_path.read_text(encoding="utf-8") if exclude_path.exists() else ""
    if pattern not in {line.strip() for line in text.splitlines()}:
        prefix = "" if not text or text.endswith("\n") else "\n"
        _atomic_write(exclude_path, f"{text}{prefix}{pattern}\n")
    return exclude_path


def _set_repository_workspace(root: Path, workspace: str) -> tuple[Path, Path | None]:
    config_path = _repository_config(root)
    if _is_tracked(root, config_path):
        raise SetupError(f"repository config is tracked and will not be changed: {config_path}")
    _assert_safe_repository_config(config_path)
    exclude_path = _ensure_locally_ignored(root, config_path)
    text = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
    _atomic_write(config_path, _updated_toml_value(text, "workspace_id", workspace))
    return config_path, exclude_path


def _remove_repository_workspace(root: Path) -> Path:
    config_path = _repository_config(root)
    if _is_tracked(root, config_path):
        raise SetupError(f"repository config is tracked and will not be changed: {config_path}")
    _assert_safe_repository_config(config_path)
    if not config_path.exists():
        raise SetupError(f"repository workspace override is not configured: {config_path}")
    text = config_path.read_text(encoding="utf-8")
    updated = _updated_toml_value(text, "workspace_id", None)
    try:
        document = tomllib.loads(updated)
    except tomllib.TOMLDecodeError as exc:
        raise SetupError(f"repository config could not be updated safely: {config_path}") from exc
    if document == {"memforge": {}}:
        config_path.unlink()
    else:
        _atomic_write(config_path, updated)
    return config_path


def _base_result(operation: str, client: str, apply: bool) -> dict[str, Any]:
    return {"operation": operation, "client": client, "mode": "applied" if apply else "preview"}


def _inspect(args: argparse.Namespace) -> dict[str, Any]:
    root = _git_root(args.repo) if args.repo else None
    source, workspace, repository_status = _effective(args.client, root)
    result = _base_result("inspect", args.client, False)
    result.update(
        {
            "global_config": str(_global_config_path(args.client)),
            "repository_root": str(root) if root else None,
            "repository_config": str(_repository_config(root)) if root else None,
            "repository_status": repository_status,
            "effective_source": source,
            "effective_workspace": workspace,
            "target": _target(_origin(args.client), workspace),
            "credentials_reported": False,
            "mcp_configuration_changed": False,
        }
    )
    return result


def _set_global(args: argparse.Namespace) -> dict[str, Any]:
    workspace = args.workspace.strip()
    if not workspace:
        raise SetupError("workspace must not be empty")
    path = _global_config_path(args.client)
    if args.client == "codex":
        _load_codex_values(path)
    else:
        _load_claude_document(path)
    result = _base_result("set-global", args.client, args.apply)
    result.update(
        {
            "files": [str(path)],
            "workspace": workspace,
            "target": _target(_origin(args.client), workspace),
            "shadowed_by_process_workspace": bool(os.getenv("MEMFORGE_WORKSPACE_ID")),
            "new_session_required": True,
        }
    )
    if args.apply:
        if args.client == "codex":
            _set_codex_global(path, workspace)
        else:
            _set_claude_global(path, workspace)
    return result


def _set_repo(args: argparse.Namespace) -> dict[str, Any]:
    workspace = args.workspace.strip()
    if not workspace:
        raise SetupError("workspace must not be empty")
    root = _git_root(args.repo)
    config_path = _repository_config(root)
    if _is_tracked(root, config_path):
        raise SetupError(f"repository config is tracked and will not be changed: {config_path}")
    _assert_safe_repository_config(config_path)
    exclude_required = not _is_ignored(root, config_path)
    files = [str(config_path)]
    if exclude_required:
        files.append(str(_git_exclude_path(root)))
    result = _base_result("set-repo", args.client, args.apply)
    result.update(
        {
            "repository_root": str(root),
            "files": files,
            "workspace": workspace,
            "target": _target(_origin(args.client), workspace),
            "local_git_exclude_update": exclude_required,
            "new_session_required": False,
        }
    )
    if args.apply:
        _set_repository_workspace(root, workspace)
    return result


def _remove_repo(args: argparse.Namespace) -> dict[str, Any]:
    root = _git_root(args.repo)
    config_path = _repository_config(root)
    if _is_tracked(root, config_path):
        raise SetupError(f"repository config is tracked and will not be changed: {config_path}")
    _assert_safe_repository_config(config_path)
    status, repository_value = _repository_workspace(config_path)
    if status != "valid" or repository_value is None:
        raise SetupError(f"repository workspace override is not configured: {config_path}")
    fallback_source, fallback_workspace = _fallback_workspace(args.client)
    result = _base_result("remove-repo", args.client, args.apply)
    result.update(
        {
            "repository_root": str(root),
            "files": [str(config_path)],
            "removed_workspace": repository_value,
            "fallback_source": fallback_source,
            "fallback_workspace": fallback_workspace,
            "target": _target(_origin(args.client), fallback_workspace),
            "new_session_required": False,
        }
    )
    if args.apply:
        _remove_repository_workspace(root)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    for operation in ("inspect", "set-global", "set-repo", "remove-repo"):
        subparser = subparsers.add_parser(operation)
        subparser.add_argument("--client", choices=("codex", "claude-code"), required=True)
        if operation in {"inspect", "set-repo", "remove-repo"}:
            subparser.add_argument("--repo")
        if operation in {"set-global", "set-repo"}:
            subparser.add_argument("--workspace", required=True)
        if operation != "inspect":
            subparser.add_argument("--apply", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.operation == "inspect":
            result = _inspect(args)
        elif args.operation == "set-global":
            result = _set_global(args)
        elif args.operation == "set-repo":
            if not args.repo:
                raise SetupError("--repo is required for repository configuration")
            result = _set_repo(args)
        else:
            if not args.repo:
                raise SetupError("--repo is required for repository configuration")
            result = _remove_repo(args)
    except (OSError, SetupError, subprocess.SubprocessError) as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
