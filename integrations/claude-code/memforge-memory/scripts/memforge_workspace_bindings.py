"""Resolve user-confirmed local context into an explicit workspace selector."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
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
        _repo_identity_spec = importlib.util.spec_from_file_location(
            "memforge_repo_identity",
            _repo_identity_path,
        )
        if _repo_identity_spec is None or _repo_identity_spec.loader is None:
            raise
        _repo_identity_module = importlib.util.module_from_spec(_repo_identity_spec)
        _repo_identity_spec.loader.exec_module(_repo_identity_module)
        normalize_repo_identifier = _repo_identity_module.normalize_repo_identifier


DEFAULT_WORKSPACE_BINDINGS_FILE = Path.home() / ".memforge" / "workspace-bindings.json"
WorkspaceBindingSource = Literal["directory", "repository", "hook_default", "none"]


class WorkspaceBindingError(ValueError):
    """A local workspace-binding document or context is unsafe to use."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class TargetWorkspaceBindings:
    repository_bindings: dict[str, str]
    directory_bindings: dict[Path, str]
    hook_workspace_id: str | None


@dataclass(frozen=True)
class WorkspaceBindings:
    targets: dict[str, TargetWorkspaceBindings]


@dataclass(frozen=True)
class WorkspaceBindingResolution:
    workspace_id: str | None
    source: WorkspaceBindingSource
    repo_identifier: str | None
    working_directory: str | None


def workspace_bindings_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path).expanduser()
    configured = os.getenv("MEMFORGE_WORKSPACE_BINDINGS_FILE")
    return Path(configured).expanduser() if configured else DEFAULT_WORKSPACE_BINDINGS_FILE


def load_workspace_bindings(
    path: str | Path | None = None,
) -> WorkspaceBindings:
    config_path = workspace_bindings_path(path)
    if not config_path.exists():
        return WorkspaceBindings(targets={})
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WorkspaceBindingError("workspace_bindings_invalid") from exc
    if not isinstance(document, dict) or set(document) != {"version", "targets"}:
        raise WorkspaceBindingError("workspace_bindings_invalid")
    if document.get("version") != 1 or not isinstance(document.get("targets"), dict):
        raise WorkspaceBindingError("workspace_bindings_version_unsupported")

    targets: dict[str, TargetWorkspaceBindings] = {}
    for raw_origin, raw_target in document["targets"].items():
        origin = _required_string(raw_origin)
        if origin is None or not isinstance(raw_target, dict):
            raise WorkspaceBindingError("workspace_bindings_invalid")
        unknown = set(raw_target) - {
            "repository_bindings",
            "directory_bindings",
            "hook_workspace_id",
        }
        if unknown:
            raise WorkspaceBindingError("workspace_bindings_invalid")
        repositories = _repository_bindings(raw_target.get("repository_bindings", {}))
        directories = _directory_bindings(raw_target.get("directory_bindings", {}))
        hook_workspace_id = _optional_string(raw_target.get("hook_workspace_id"))
        if "hook_workspace_id" in raw_target and hook_workspace_id is None:
            raise WorkspaceBindingError("workspace_bindings_invalid")
        targets[origin] = TargetWorkspaceBindings(
            repository_bindings=repositories,
            directory_bindings=directories,
            hook_workspace_id=hook_workspace_id,
        )
    return WorkspaceBindings(targets=targets)


def resolve_workspace_binding(
    *,
    origin: str,
    working_directory: str | None = None,
    root_paths: Sequence[str] = (),
    allow_hook_default: bool = False,
    config_path: str | Path | None = None,
) -> WorkspaceBindingResolution:
    """Resolve one local selector without treating the hook fallback as search scope."""
    bindings = load_workspace_bindings(config_path)
    target = bindings.targets.get(origin)
    locations = (working_directory,) if working_directory is not None else tuple(root_paths)
    resolved_locations = tuple(_local_context(location) for location in locations)

    if target is None:
        return _unbound_resolution(resolved_locations)

    candidates = tuple(_resolve_location(target, path, repo) for path, repo in resolved_locations)
    workspace_ids = {candidate.workspace_id for candidate in candidates if candidate.workspace_id is not None}
    if len(workspace_ids) > 1:
        raise WorkspaceBindingError("workspace_binding_ambiguous")
    if workspace_ids:
        workspace_id = next(iter(workspace_ids))
        selected = next(candidate for candidate in candidates if candidate.workspace_id == workspace_id)
        repo_identifiers = {
            candidate.repo_identifier for candidate in candidates if candidate.repo_identifier is not None
        }
        return WorkspaceBindingResolution(
            workspace_id=workspace_id,
            source=selected.source,
            repo_identifier=(next(iter(repo_identifiers)) if len(repo_identifiers) == 1 else None),
            working_directory=selected.working_directory,
        )

    unbound = _unbound_resolution(resolved_locations)
    if allow_hook_default and target.hook_workspace_id is not None:
        return WorkspaceBindingResolution(
            workspace_id=target.hook_workspace_id,
            source="hook_default",
            repo_identifier=unbound.repo_identifier,
            working_directory=unbound.working_directory,
        )
    return unbound


def _repository_bindings(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        raise WorkspaceBindingError("workspace_bindings_invalid")
    result: dict[str, str] = {}
    for raw_repo, raw_workspace in value.items():
        repo = normalize_repo_identifier(_required_string(raw_repo))
        workspace = _required_string(raw_workspace)
        if repo is None or workspace is None:
            raise WorkspaceBindingError("workspace_bindings_invalid")
        if repo in result and result[repo] != workspace:
            raise WorkspaceBindingError("workspace_bindings_invalid")
        result[repo] = workspace
    return result


def _directory_bindings(value: object) -> dict[Path, str]:
    if not isinstance(value, dict):
        raise WorkspaceBindingError("workspace_bindings_invalid")
    result: dict[Path, str] = {}
    for raw_directory, raw_workspace in value.items():
        directory_value = _required_string(raw_directory)
        workspace = _required_string(raw_workspace)
        if directory_value is None or workspace is None:
            raise WorkspaceBindingError("workspace_bindings_invalid")
        directory = Path(directory_value).expanduser()
        if not directory.is_absolute():
            raise WorkspaceBindingError("workspace_bindings_directory_must_be_absolute")
        normalized = directory.resolve(strict=False)
        if normalized in result and result[normalized] != workspace:
            raise WorkspaceBindingError("workspace_bindings_invalid")
        result[normalized] = workspace
    return result


def _resolve_location(
    target: TargetWorkspaceBindings,
    path: Path,
    repo_identifier: str | None,
) -> WorkspaceBindingResolution:
    directory_matches = [
        (directory, workspace_id)
        for directory, workspace_id in target.directory_bindings.items()
        if path == directory or directory in path.parents
    ]
    if directory_matches:
        directory, workspace_id = max(directory_matches, key=lambda item: len(item[0].parts))
        return WorkspaceBindingResolution(
            workspace_id=workspace_id,
            source="directory",
            repo_identifier=repo_identifier,
            working_directory=str(path),
        )
    if repo_identifier is not None:
        workspace_id = target.repository_bindings.get(repo_identifier)
        if workspace_id is not None:
            return WorkspaceBindingResolution(
                workspace_id=workspace_id,
                source="repository",
                repo_identifier=repo_identifier,
                working_directory=str(path),
            )
    return WorkspaceBindingResolution(
        workspace_id=None,
        source="none",
        repo_identifier=repo_identifier,
        working_directory=str(path),
    )


def _unbound_resolution(
    contexts: Sequence[tuple[Path, str | None]],
) -> WorkspaceBindingResolution:
    repo_identifiers = {repo for _path, repo in contexts if repo is not None}
    working_directory = str(contexts[0][0]) if len(contexts) == 1 else None
    return WorkspaceBindingResolution(
        workspace_id=None,
        source="none",
        repo_identifier=(next(iter(repo_identifiers)) if len(repo_identifiers) == 1 else None),
        working_directory=working_directory,
    )


def _local_context(location: str) -> tuple[Path, str | None]:
    path = _local_path(location)
    if path is None or not path.exists() or not path.is_dir():
        raise WorkspaceBindingError("workspace_context_invalid")
    path = path.resolve(strict=True)
    remote = _git_value(["git", "remote", "get-url", "origin"], cwd=path)
    return path, normalize_repo_identifier(remote)


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
        path = Path(unquote(parsed.path))
    return path if path.is_absolute() else None


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


def _required_string(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    return _required_string(value)


__all__ = [
    "DEFAULT_WORKSPACE_BINDINGS_FILE",
    "TargetWorkspaceBindings",
    "WorkspaceBindingError",
    "WorkspaceBindingResolution",
    "WorkspaceBindings",
    "load_workspace_bindings",
    "resolve_workspace_binding",
    "workspace_bindings_path",
]
