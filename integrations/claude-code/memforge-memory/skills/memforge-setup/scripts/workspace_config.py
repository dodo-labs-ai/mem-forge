#!/usr/bin/env python3
"""Preview and atomically apply user-confirmed MemForge workspace bindings."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any
import urllib.error
import urllib.request


PLUGIN_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PLUGIN_ROOT / "scripts"))

from memforge_plugin_config import configured_api_token, configured_target  # noqa: E402
from memforge_repo_identity import normalize_repo_identifier  # noqa: E402
from memforge_workspace_bindings import (  # noqa: E402
    WorkspaceBindingError,
    load_workspace_bindings,
    resolve_workspace_binding,
    workspace_bindings_path,
)


class SetupError(ValueError):
    """A safe, user-actionable setup failure."""


def _current_bytes(path: Path) -> bytes:
    if path.is_symlink():
        raise SetupError(f"refusing symlinked binding file: {path}")
    try:
        return path.read_bytes() if path.exists() else b""
    except OSError as exc:
        raise SetupError(f"cannot read binding file: {path}") from exc


def _digest(path: Path) -> str:
    return hashlib.sha256(_current_bytes(path)).hexdigest()


def _load_document(path: Path) -> dict[str, Any]:
    raw = _current_bytes(path)
    if not raw:
        return {"version": 1, "targets": {}}
    try:
        document = json.loads(raw)
        load_workspace_bindings(path)
    except (json.JSONDecodeError, UnicodeError, WorkspaceBindingError) as exc:
        raise SetupError(f"binding file is invalid and will not be overwritten: {path}") from exc
    return document


def _target_document(document: dict[str, Any], origin: str) -> dict[str, Any]:
    targets = document.setdefault("targets", {})
    return targets.setdefault(origin, {})


def _path_context(value: str) -> tuple[Path, str | None]:
    path = Path(value).expanduser()
    if not path.is_absolute() or not path.exists() or not path.is_dir():
        raise SetupError("path must be an existing absolute directory")
    path = path.resolve(strict=True)
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=path,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        repository = normalize_repo_identifier(result.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        repository = None
    return path, repository


def _workspace_catalog() -> tuple[Any, set[str]]:
    target = configured_target()
    headers: dict[str, str] = {"Accept": "application/json"}
    token = configured_api_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(target.resource_url("/workspaces"), headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError, UnicodeError) as exc:
        raise SetupError("could not validate workspace against the configured MemForge target") from exc
    workspaces = payload.get("workspaces") if isinstance(payload, dict) else None
    if not isinstance(workspaces, list):
        raise SetupError("workspace directory returned an invalid response")
    ids = {
        item["workspace_id"].strip()
        for item in workspaces
        if isinstance(item, dict)
        and isinstance(item.get("workspace_id"), str)
        and item["workspace_id"].strip()
        and item.get("selectable", True)
    }
    return target, ids


def _validate_workspace(workspace_id: str) -> Any:
    target, workspace_ids = _workspace_catalog()
    if workspace_id not in workspace_ids:
        raise SetupError(f"workspace is not selectable on the configured target: {workspace_id}")
    return target


def _cleanup(document: dict[str, Any], origin: str) -> None:
    target = document["targets"].get(origin)
    if not isinstance(target, dict):
        return
    for key in ("repository_bindings", "directory_bindings"):
        if target.get(key) == {}:
            target.pop(key)
    if not target:
        document["targets"].pop(origin)


def _atomic_write(path: Path, document: dict[str, Any]) -> None:
    if path.is_symlink():
        raise SetupError(f"refusing symlinked binding file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        temporary.chmod(0o600)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _base(args: argparse.Namespace, path: Path, target: Any) -> dict[str, Any]:
    return {
        "operation": args.operation,
        "mode": "applied" if getattr(args, "apply", False) else "preview",
        "client": args.client,
        "binding_file": str(path),
        "origin": target.origin,
        "document_digest": _digest(path),
        "credentials_reported": False,
        "mcp_configuration_changed": False,
    }


def _mutate(args: argparse.Namespace) -> dict[str, Any]:
    path = workspace_bindings_path()
    before_digest = _digest(path)
    document = _load_document(path)
    target = configured_target()
    result = _base(args, path, target)

    if args.operation in {"bind", "set-hook-workspace"}:
        workspace_id = args.workspace.strip()
        if not workspace_id:
            raise SetupError("workspace must not be empty")
        target = _validate_workspace(workspace_id)
        result["origin"] = target.origin
    else:
        workspace_id = None

    target_document = _target_document(document, target.origin)
    if args.operation in {"bind", "unbind"}:
        local_path, repository = _path_context(args.path)
        directory_bindings = target_document.setdefault("directory_bindings", {})
        repository_bindings = target_document.setdefault("repository_bindings", {})
        if args.operation == "bind":
            if repository is not None:
                repository_bindings[repository] = workspace_id
                binding_kind, binding_key = "repository", repository
            else:
                directory_bindings[str(local_path)] = workspace_id
                binding_kind, binding_key = "directory", str(local_path)
        elif str(local_path) in directory_bindings:
            directory_bindings.pop(str(local_path))
            binding_kind, binding_key = "directory", str(local_path)
        elif repository is not None and repository in repository_bindings:
            repository_bindings.pop(repository)
            binding_kind, binding_key = "repository", repository
        else:
            raise SetupError("no exact directory or repository binding exists for this path")
        result.update(binding_kind=binding_kind, binding_key=binding_key, workspace_id=workspace_id)
    elif args.operation == "set-hook-workspace":
        target_document["hook_workspace_id"] = workspace_id
        result["workspace_id"] = workspace_id
    elif args.operation == "clear-hook-workspace":
        if "hook_workspace_id" not in target_document:
            raise SetupError("hook workspace fallback is not configured for this target")
        result["workspace_id"] = target_document.pop("hook_workspace_id")

    _cleanup(document, target.origin)
    result["planned_document"] = document
    if args.apply:
        if args.expect_digest != before_digest:
            raise SetupError("binding file changed after preview; preview and confirm again")
        _atomic_write(path, document)
        result["document_digest"] = _digest(path)
    return result


def _inspect(args: argparse.Namespace) -> dict[str, Any]:
    path = workspace_bindings_path()
    target = configured_target()
    local_path, repository = _path_context(args.path)
    resolution = resolve_workspace_binding(
        origin=target.origin,
        working_directory=str(local_path),
        allow_hook_default=False,
    )
    document = _load_document(path)
    target_document = document.get("targets", {}).get(target.origin, {})
    return {
        **_base(args, path, target),
        "path": str(local_path),
        "repository": repository,
        "effective_workspace_id": resolution.workspace_id,
        "effective_source": resolution.source,
        "hook_workspace_id": target_document.get("hook_workspace_id"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client", choices=("codex", "claude-code"), required=True)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--path", required=True)
    for name in ("bind", "unbind"):
        command = subparsers.add_parser(name)
        command.add_argument("--path", required=True)
        if name == "bind":
            command.add_argument("--workspace", required=True)
        command.add_argument("--apply", action="store_true")
        command.add_argument("--expect-digest")
    hook = subparsers.add_parser("set-hook-workspace")
    hook.add_argument("--workspace", required=True)
    hook.add_argument("--apply", action="store_true")
    hook.add_argument("--expect-digest")
    clear = subparsers.add_parser("clear-hook-workspace")
    clear.add_argument("--apply", action="store_true")
    clear.add_argument("--expect-digest")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    os.environ["MEMFORGE_MCP_CLIENT"] = args.client
    try:
        result = _inspect(args) if args.operation == "inspect" else _mutate(args)
    except (SetupError, WorkspaceBindingError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
