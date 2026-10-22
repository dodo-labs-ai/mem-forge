"""CLI entry point for MemForge."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import selectors
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from importlib import metadata
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any, Callable
from urllib.parse import quote

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 compatibility
    import tomli as tomllib

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from memforge.api_target import Edition, MemForgeTarget, build_target
from memforge.auth import browser_session
from memforge.capability_discovery import discover_target
from memforge.config import DEFAULT_SEARCH_TOP_K, AppConfig, load_config
from memforge.github_repo_utils import (
    DEFAULT_INCLUDE_EXTENSION_LIST,
    GITHUB_SYMLINK_MODE,
    GitHubBlobBuffer,
    GitHubResolvedTreeEntry,
    GitHubTreeEntryResolver,
    build_github_repo_doc_id,
    decode_github_base64_content,
    decode_github_text,
    github_content_paths,
    github_content_type,
    github_content_type_is_binary,
    github_exclude_paths,
    github_extension,
    github_include_extensions,
    github_include_paths,
    github_latest_commit_time,
    github_path_commits_query,
    github_path_in_scope,
    normalize_github_relative_path,
    parse_github_repo_url,
    validate_github_file_mode,
    validate_github_tree_payload,
)
from memforge.local_agent.folder_picker import FolderPickerCancelled, FolderPickerUnavailable, pick_folder
from memforge.local_agent.document_identity import (
    build_jira_doc_id,
    build_local_markdown_doc_id,
)
from memforge.local_agent.source_contract import (
    GitHubProviderConnectionError,
    local_agent_sync_snapshot_id,
    source_processing_receipt,
)
from memforge.retrieval.intents import RANKED_RETRIEVAL_INTENTS
from memforge.source_time import latest_source_time, source_time_iso
from memforge.source_artifacts import (
    MAX_SOURCE_ARTIFACT_STORAGE_BYTES,
    MAX_SOURCE_ARTIFACT_STORAGE_BYTES_PER_UNIT,
    SUPPORTED_SOURCE_ARTIFACT_MEDIA_TYPES,
    SOURCE_ARTIFACT_STREAM_CHUNK_BYTES,
    read_source_artifact_bytes,
)
from memforge.sync_progress import normalize_sync_progress_snapshot
from memforge.storage.admin_source import (
    SOURCE_SYNC_SCHEDULE_MAX_INTERVAL_MINUTES,
    SOURCE_SYNC_SCHEDULE_MIN_INTERVAL_MINUTES,
)
from memforge.tool_client import ToolClient
from memforge.workspace_bindings import (
    WORKSPACE_SELECTION_REQUIRED_CODE,
    WorkspaceBindingError,
    resolve_workspace_binding,
)

console = Console()
log_console = Console(stderr=True)
logger = logging.getLogger(__name__)

DEFAULT_CLI_CONFIG_PATH = Path.home() / ".memforge" / "cli.toml"
DEFAULT_LOCAL_AGENT_STATE_PATH = Path.home() / ".memforge" / "local-agent-state.json"
DEFAULT_LOCAL_AGENT_LOCK_PATH = Path.home() / ".memforge" / "local-agent-daemon.lock"
DEFAULT_KB_INCLUDE = [
    "*.md",
    "**/*.md",
    "*.markdown",
    "**/*.markdown",
    "*.txt",
    "**/*.txt",
    "*.json",
    "**/*.json",
    "*.html",
    "**/*.html",
    "*.htm",
    "**/*.htm",
]
DEFAULT_KB_EXCLUDE = [".obsidian/**", ".trash/**", ".git/**", "**/.git/**"]
LOCAL_MARKDOWN_SOURCE_TYPE = "local_markdown"
GITHUB_REPO_SOURCE_TYPE = "github_repo"
GITHUB_RAW_TRANSFER_TIMEOUT_SECONDS = 30.0
JIRA_SOURCE_TYPE = "jira"
DEFAULT_GITHUB_INCLUDE_EXTENSIONS = DEFAULT_INCLUDE_EXTENSION_LIST
# Watch defaults. The tick interval is deliberately shorter than a typical Jira
# idle-session timeout so the stored copy is renewed while it is still valid.
WATCH_DEFAULT_INTERVAL_SECONDS = 1800  # 30 minutes
WATCH_BACKOFF_BASE_SECONDS = 5
WATCH_BACKOFF_MAX_SECONDS = 300  # 5 minutes
LOCAL_AGENT_LEASE_NOT_CURRENT = "local_agent_lease_not_current"
INTERACTIVE_DISABLE_ENV = "MEMFORGE_NO_INTERACTIVE"
INTERACTIVE_SCRIPT_ENV = "MEMFORGE_INTERACTIVE_SCRIPT"
INTERACTIVE_BIN_ENV = "MEMFORGE_CLI_BIN"
INTERACTIVE_CACHE_ENV = "MEMFORGE_INTERACTIVE_CACHE"
INTERACTIVE_RESOURCE_DIR = "interactive_cli"
INTERACTIVE_INSTALL_LOCK = ".install.lock"
INTERACTIVE_INSTALL_LOCK_TIMEOUT_SECONDS = 120
INTERACTIVE_INSTALL_LOCK_STALE_SECONDS = 600
INTERACTIVE_DEPENDENCY_SENTINEL = Path("node_modules") / "@clack" / "prompts"
PYPI_DISTRIBUTION_NAME = "memforge-ai"


@dataclass(frozen=True)
class _ResolvedCliTarget:
    target: MemForgeTarget
    api_token: str | None
    active_target: str
    token_env: str


def setup_logging(verbose: bool = False) -> None:
    """Log to stderr: formatted for a terminal, one plain line per record otherwise.

    A container or service manager collects stderr as text without a terminal
    width, where boxed tracebacks wrap into unreadable columns.
    """
    level = logging.DEBUG if verbose else logging.INFO
    if log_console.is_terminal:
        logging.basicConfig(
            level=level,
            format="%(message)s",
            handlers=[RichHandler(console=log_console, rich_tracebacks=True)],
            force=True,
        )
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )


async def _get_db(config: AppConfig):
    """Open the local database for a command that runs with ``config``.

    Opening applies the migrations of this release, after taking a recovery
    point of an existing database. A database this release cannot open, or
    an upgrade that fails, ends the command with the reason.
    """
    from memforge.storage.database import Database

    from memforge.storage.document_store import LocalDocumentStore
    from memforge.storage.recovery_points import RecoveryPoints, UpgradeError

    db = Database(
        config.storage.db_path,
        document_store=LocalDocumentStore(config.storage.docs_path),
        recovery_points=RecoveryPoints.for_config(config),
    )
    try:
        await db.connect()
    except UpgradeError as exc:
        raise click.ClickException(str(exc)) from exc
    try:
        await db.translate_llm_config(config.llm)
    except BaseException:
        await db.close()
        raise
    return db


def _cli_config_path() -> Path:
    configured = os.getenv("MEMFORGE_CLI_CONFIG", "").strip()
    return Path(configured).expanduser() if configured else DEFAULT_CLI_CONFIG_PATH


def _read_cli_config() -> dict[str, Any]:
    path = _cli_config_path()
    if not path.exists():
        return {"active": "", "targets": {}}
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    targets = data.get("targets")
    if not isinstance(targets, dict):
        targets = {}
    active = data.get("active") if isinstance(data.get("active"), str) else ""
    return {"active": active, "targets": targets}


def _write_cli_config(data: dict[str, Any]) -> None:
    path = _cli_config_path()
    targets = data.get("targets") if isinstance(data.get("targets"), dict) else {}
    lines = [f"active = {_toml_string(str(data.get('active') or ''))}", ""]
    for name, target in sorted(targets.items()):
        if not isinstance(target, dict):
            continue
        lines.append(f"[targets.{_toml_key(str(name))}]")
        lines.append(f"api_url = {_toml_string(str(target.get('api_url') or ''))}")
        if target.get("edition"):
            lines.append(f"edition = {_toml_string(str(target['edition']))}")
        if target.get("token_env"):
            lines.append(f"token_env = {_toml_string(str(target['token_env']))}")
        lines.append("")
    _write_cli_config_bytes(path, "\n".join(lines).encode("utf-8"))


def _write_cli_config_bytes(path: Path, payload: bytes) -> None:
    """Atomically replace the private CLI config, including exact rollback snapshots."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(payload)
        temporary.chmod(0o600)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _toml_string(value: str) -> str:
    return json.dumps(value)


def _toml_key(value: str) -> str:
    return json.dumps(value)


def _daemon_keyring_api_token(api_url: str) -> str | None:
    """Reuse the daemon credential for ordinary CLI calls to the exact same target."""
    from memforge.local_agent.service import DaemonCredentialStore, DaemonServiceError, DaemonServicePaths

    store = DaemonCredentialStore(DaemonServicePaths.for_home(Path.home()).config)
    try:
        environment = store.environment()
    except DaemonServiceError:
        return None
    if environment.get("MEMFORGE_API_URL", "").rstrip("/") != api_url.rstrip("/"):
        return None
    return environment.get("MEMFORGE_API_TOKEN")


def _resolve_api_target(
    _config: AppConfig,
) -> _ResolvedCliTarget:
    target_env_names = ("MEMFORGE_API_URL",)
    if any(os.getenv(name, "").strip() for name in target_env_names):
        target = _build_cli_target(
            api_url=os.getenv("MEMFORGE_API_URL"),
            edition=os.getenv("MEMFORGE_EDITION"),
        )
        return _ResolvedCliTarget(
            target=target,
            api_token=os.getenv("MEMFORGE_API_TOKEN") or _daemon_keyring_api_token(target.origin),
            active_target="",
            token_env="MEMFORGE_API_TOKEN",
        )

    cli_config = _read_cli_config()
    active = str(cli_config.get("active") or "")
    profiles = cli_config.get("targets") if isinstance(cli_config.get("targets"), dict) else {}
    profile = profiles.get(active)
    if isinstance(profile, dict):
        target = _build_cli_target(
            api_url=profile.get("api_url"),
            edition=profile.get("edition"),
        )
        token_env = str(profile.get("token_env") or "")
        return _ResolvedCliTarget(
            target=target,
            api_token=(os.getenv(token_env) if token_env else None) or _daemon_keyring_api_token(target.origin),
            active_target=active,
            token_env=token_env,
        )

    return _ResolvedCliTarget(
        target=build_target(origin=None),
        api_token=os.getenv("MEMFORGE_API_TOKEN"),
        active_target="",
        token_env="",
    )


def _build_cli_target(
    *,
    api_url: object,
    edition: object | None = None,
) -> MemForgeTarget:
    origin = str(api_url) if api_url is not None else None
    if origin is None:
        return build_target(origin=None)
    if edition is not None and str(edition).strip():
        return build_target(origin=origin, edition=str(edition).strip())
    try:
        return discover_target(origin)
    except ValueError as exc:
        raise click.ClickException(f"Could not discover MemForge capabilities: {exc}") from exc


def _workspace_id_option(command):
    """Add the request-scoped workspace selector to a workspace data-plane command."""
    return click.option(
        "--workspace-id",
        default=None,
        help="Workspace for this request. Required when the account can select more than one workspace.",
    )(command)


def _tool_client(ctx, workspace_id: str | None) -> ToolClient:
    """Return a client for the active target, scoped to an explicit workspace when one is given."""
    resolved = _resolve_api_target(ctx.obj["config"])
    return ToolClient(target=resolved.target, api_token=resolved.api_token, workspace_id=workspace_id)


def _project_tool_client(ctx, workspace_id: str | None) -> ToolClient:
    """Return a client for a project command.

    An explicit ``--workspace-id`` wins. Otherwise the current directory's
    confirmed directory or repository binding selects the workspace. The hook
    fallback is never used for interactive commands.
    """
    resolved = _resolve_api_target(ctx.obj["config"])
    return ToolClient(
        target=resolved.target,
        api_token=resolved.api_token,
        workspace_id=workspace_id or _current_directory_workspace_id(resolved.target.origin),
    )


def _current_directory_workspace_id(origin: str) -> str | None:
    try:
        resolution = resolve_workspace_binding(origin=origin, working_directory=os.getcwd())
    except WorkspaceBindingError as exc:
        raise click.ClickException(
            f"The workspace binding for this directory cannot be used ({exc.code}). "
            "Fix it with the memforge-setup skill or pass --workspace-id."
        ) from exc
    return resolution.workspace_id


def _local_agent_tool_client(ctx) -> ToolClient:
    resolved = _resolve_api_target(ctx.obj["config"])
    return ToolClient(target=resolved.target, api_token=resolved.api_token)


def _emit_tool_payload(ctx, payload: dict) -> None:
    if payload.get("code") == WORKSPACE_SELECTION_REQUIRED_CODE:
        workspace_ids = [str(workspace_id) for workspace_id in payload.get("workspace_ids") or []]
        payload = {
            **payload,
            "hint": f"Pass --workspace-id with one of: {', '.join(workspace_ids)}.",
        }
    click.echo(json.dumps(payload, indent=2, ensure_ascii=False))
    if payload.get("error"):
        ctx.exit(1)


def _local_agent_state_path() -> Path:
    configured = os.getenv("MEMFORGE_LOCAL_AGENT_STATE", "").strip()
    return Path(configured).expanduser() if configured else DEFAULT_LOCAL_AGENT_STATE_PATH


def _local_agent_lock_path() -> Path:
    configured = os.getenv("MEMFORGE_LOCAL_AGENT_LOCK", "").strip()
    return Path(configured).expanduser() if configured else DEFAULT_LOCAL_AGENT_LOCK_PATH


def _local_agent_state_store():
    from memforge.local_agent.state import LocalAgentStateStore

    return LocalAgentStateStore(_local_agent_state_path())


class _LocalAgentDaemonLock:
    def __init__(self, path: Path, fd: int) -> None:
        self.path = path
        self.fd = fd
        self.closed = False

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            os.close(self.fd)
        finally:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass


def _pid_is_running(pid: Any) -> bool:
    try:
        normalized = int(pid)
    except (TypeError, ValueError):
        return False
    if normalized <= 0:
        return False
    try:
        os.kill(normalized, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_local_agent_lock(lock_path: Path | None = None) -> dict[str, Any] | None:
    path = (lock_path or _local_agent_lock_path()).expanduser()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _acquire_local_agent_daemon_lock(lock_path: Path | None = None) -> _LocalAgentDaemonLock | None:
    path = (lock_path or _local_agent_lock_path()).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError:
        existing = _read_local_agent_lock(path)
        if existing is not None and _pid_is_running(existing.get("pid")):
            return None
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        try:
            fd = os.open(path, flags, 0o600)
        except FileExistsError:
            return None
    payload = {
        "pid": os.getpid(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
    }
    with os.fdopen(os.dup(fd), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return _LocalAgentDaemonLock(path, fd)


def _build_local_agent_runner(
    ctx,
    *,
    browser: str | None,
):
    from memforge.local_agent.runner import LocalAgentRunner

    client = _local_agent_tool_client(ctx)
    return LocalAgentRunner(
        state_store=_local_agent_state_store(),
        cloud_job_handler=lambda job, report_progress=None: _run_cloud_local_agent_job(
            job, client, browser=browser, report_progress=report_progress
        ),
        cloud_jobs_provider=lambda limit=1, wait_seconds=0, lease_seconds=60: client.lease_local_agent_jobs(
            limit=limit,
            wait_seconds=wait_seconds,
            lease_seconds=lease_seconds,
        ),
        cloud_job_completer=lambda job_id, attempt_count, status, result, error=None: client.complete_local_agent_job(
            job_id,
            attempt_count=attempt_count,
            status=status,
            result=result,
            error=error,
        ),
        cloud_job_heartbeat=lambda job_id, attempt_count, lease_seconds, progress=None: (
            client.heartbeat_local_agent_job(
                job_id,
                attempt_count=attempt_count,
                lease_seconds=lease_seconds,
                progress=progress,
            )
        ),
    )


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    return []


def _merge_default_excludes(excludes: tuple[str, ...] | list[str]) -> list[str]:
    merged: list[str] = []
    for pattern in [*DEFAULT_KB_EXCLUDE, *[str(item) for item in excludes]]:
        if pattern and pattern not in merged:
            merged.append(pattern)
    return merged


def _parse_github_repo_url(repo_url: str) -> dict[str, str]:
    try:
        parsed = parse_github_repo_url(repo_url)
    except ValueError as exc:
        raise click.ClickException(str(exc).replace("repo_url", "Repository URL")) from exc
    return {"repo_url": parsed["repo_url"], "host": parsed["host"], "owner": parsed["owner"], "repo": parsed["repo"]}


def _github_gh_env(host: str) -> dict[str, str]:
    env = dict(os.environ)
    if host != "github.com":
        env["GH_HOST"] = host
    else:
        env.pop("GH_HOST", None)
    return env


def _gh_api_json(repo: dict[str, str], endpoint: str) -> dict[str, Any]:
    payload = _gh_api_payload(repo, endpoint)
    return payload if isinstance(payload, dict) else {}


def _gh_api_payload(repo: dict[str, str], endpoint: str) -> object:
    try:
        result = subprocess.run(
            ["gh", "api", endpoint],
            capture_output=True,
            text=True,
            env=_github_gh_env(repo["host"]),
            check=False,
        )
    except FileNotFoundError as exc:
        raise click.ClickException("GitHub CLI `gh` is required for local GitHub repository sync.") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "gh api failed").strip()
        _raise_github_cli_error(detail)
    try:
        return json.loads(result.stdout or "{}")
    except ValueError as exc:
        raise click.ClickException("GitHub CLI returned invalid JSON.") from exc


def _github_last_commit_at(
    repo: dict[str, str],
    *,
    commit_sha: str,
    relative_path: str,
    resolved_relative_path: object,
) -> str | None:
    """The file's latest commit time at the collection commit, its symlink target included.

    A refused request leaves the time unknown; the service then asks for the
    file again on the next collection (``LOCAL_PACKAGE_SOURCE_TIME_CONTRACT_VERSION``).
    """

    commit_times = []
    for path in github_content_paths(relative_path, resolved_relative_path):
        try:
            payload = _gh_api_payload(
                repo,
                f"repos/{repo['owner']}/{repo['repo']}/commits?"
                + github_path_commits_query(relative_path=path, ref=commit_sha),
            )
        except click.ClickException as exc:
            logger.warning("GitHub commit time for %s is unknown: %s", path, exc.message)
            return None
        commit_times.append(github_latest_commit_time(payload))
    return latest_source_time(commit_times)


def _raise_github_cli_error(detail: str) -> None:
    normalized_detail = detail.lower()
    if any(marker in normalized_detail for marker in (
        "error connecting to", "check your internet connection", "could not resolve host",
        "no such host", "temporary failure in name resolution", "network is unreachable",
        "connection refused", "connection timed out", "i/o timeout", "tls handshake timeout",
    )):
        raise GitHubProviderConnectionError(f"GitHub CLI request failed: {detail}")
    raise click.ClickException(f"GitHub CLI request failed: {detail}")


@dataclass(frozen=True)
class _GitHubCollectionSnapshot:
    commit_sha: str
    root_tree_sha: str


def _resolve_github_collection_snapshot(
    repo: dict[str, str],
    ref: str,
) -> _GitHubCollectionSnapshot:
    payload = _gh_api_json(
        repo,
        f"repos/{repo['owner']}/{repo['repo']}/commits/{quote(ref, safe='')}",
    )
    commit_sha = str(payload.get("sha") or "").strip()
    commit = payload.get("commit") if isinstance(payload.get("commit"), dict) else {}
    tree = commit.get("tree") if isinstance(commit.get("tree"), dict) else {}
    root_tree_sha = str(tree.get("sha") or "").strip()
    if not commit_sha or not root_tree_sha:
        raise click.ClickException(
            "GitHub did not return an immutable commit and root tree for this collection."
        )
    return _GitHubCollectionSnapshot(
        commit_sha=commit_sha,
        root_tree_sha=root_tree_sha,
    )


def _github_tree(repo: dict[str, str], tree_sha: str) -> list[dict[str, Any]]:
    payload = _gh_api_json(
        repo,
        f"repos/{repo['owner']}/{repo['repo']}/git/trees/{quote(tree_sha, safe='')}?recursive=1",
    )
    try:
        return validate_github_tree_payload(payload, label="GitHub repository")
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


def _github_blob(repo: dict[str, str], blob_sha: str, relative_path: str) -> bytes:
    payload = _gh_api_json(
        repo,
        f"repos/{repo['owner']}/{repo['repo']}/git/blobs/{quote(blob_sha, safe='')}",
    )
    if str(payload.get("sha") or "").strip() != blob_sha:
        raise click.ClickException(
            f"GitHub returned a different blob revision for {relative_path}."
        )
    try:
        return decode_github_base64_content(
            content=payload.get("content"),
            encoding=payload.get("encoding"),
            size=payload.get("size"),
            label=relative_path,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


def _github_text_blob(
    repo: dict[str, str], blob_sha: str, relative_path: str, *, size: object,
) -> bytes:
    """Read one immutable raw body with bounded pipes and a transport deadline."""
    try:
        buffer = GitHubBlobBuffer(sha=blob_sha, size=size, label=relative_path)
        command = ["gh", "api", f"repos/{repo['owner']}/{repo['repo']}/git/blobs/{blob_sha}",
                   "-H", "Accept: application/vnd.github.raw+json", "-H", "Accept-Encoding: identity"]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   env=_github_gh_env(repo["host"]))
    except FileNotFoundError as exc:
        raise click.ClickException("GitHub CLI `gh` is required for local GitHub repository sync.") from exc
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    stderr = bytearray()
    deadline = time.monotonic() + GITHUB_RAW_TRANSFER_TIMEOUT_SECONDS
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ, "body")
            selector.register(process.stderr, selectors.EVENT_READ, "error")
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise GitHubProviderConnectionError("GitHub raw blob request timed out")
                for key, _ in selector.select(remaining):
                    chunk = os.read(key.fd, SOURCE_ARTIFACT_STREAM_CHUNK_BYTES)
                    if not chunk:
                        selector.unregister(key.fileobj)
                    elif key.data == "body":
                        buffer.write(chunk)
                    else:
                        stderr.extend(chunk[:max(0, 8192 - len(stderr))])
            process.wait(timeout=max(0.001, deadline - time.monotonic()))
        if process.returncode != 0:
            detail = stderr.decode("utf-8", errors="replace").strip() or "gh api failed"
            _raise_github_cli_error(detail)
        return buffer.finish()
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    except subprocess.TimeoutExpired as exc:
        raise GitHubProviderConnectionError("GitHub raw blob request timed out") from exc
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        process.stdout.close()
        process.stderr.close()


def _resolve_github_profile(
    name: str,
    profile: dict[str, Any],
) -> tuple[dict[str, str], str, list[str], list[str], list[str]]:
    repo = _parse_github_repo_url(str(profile.get("repo_url") or ""))
    ref = str(profile.get("ref") or "main").strip() or "main"
    if ref.startswith("-"):
        raise click.ClickException("GitHub repository ref must not start with '-'.")
    try:
        include_paths = github_include_paths(profile)
        exclude_paths = github_exclude_paths(profile)
    except ValueError as exc:
        raise click.ClickException(str(exc).replace("relative_path", "GitHub include path")) from exc
    include_extensions = sorted(github_include_extensions(profile))
    return repo, ref, include_paths, exclude_paths, include_extensions


def _github_title(markdown_body: str, fallback: str) -> str:
    for line in markdown_body.splitlines():
        stripped = line.strip()
        if stripped.startswith("# ") and stripped[2:].strip():
            return stripped[2:].strip()
    return fallback


def _resolve_github_profile_entry(
    repo: dict[str, str],
    entry: dict[str, Any],
    *,
    entries_by_path: dict[str, dict[str, Any]],
    exclude_paths: list[str],
) -> GitHubResolvedTreeEntry:
    """Bind one logical GitHub path to its immutable regular content blob."""

    resolver = GitHubTreeEntryResolver(
        entry,
        entries_by_path=entries_by_path,
        exclude_paths=exclude_paths,
    )
    try:
        while link := resolver.next_symlink_read():
            raw_target = _github_text_blob(
                repo,
                link.blob_sha,
                link.path,
                size=link.blob_size,
            )
            resolver.accept_symlink_target(raw_target)
        return resolver.result()
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


def _preview_github_profile(name: str, profile: dict[str, Any], *, limit: int | None) -> dict[str, Any]:
    repo, ref, include_paths, exclude_paths, include_extensions = _resolve_github_profile(name, profile)
    snapshot = _resolve_github_collection_snapshot(repo, ref)
    counts = {"included": 0, "ignored": 0}
    extension_counts: dict[str, int] = {}
    items: list[dict[str, Any]] = []
    tree = _github_tree(repo, snapshot.root_tree_sha)
    entries_by_path = {
        normalize_github_relative_path(str(entry.get("path") or "")).rstrip("/"): entry
        for entry in tree
    }
    for entry in tree:
        if (
            entry.get("type") not in {"blob", "commit"}
            and entry.get("mode") != GITHUB_SYMLINK_MODE
        ):
            continue
        relative_path = str(entry.get("path") or "")
        if not github_path_in_scope(relative_path, include_paths, exclude_paths):
            counts["ignored"] += 1
            continue
        extension = github_extension(relative_path)
        if extension:
            extension_counts[extension] = extension_counts.get(extension, 0) + 1
        if include_extensions and extension not in include_extensions:
            counts["ignored"] += 1
            continue
        counts["included"] += 1
        if limit is None or len(items) < limit:
            resolved = _resolve_github_profile_entry(
                repo,
                entry,
                entries_by_path=entries_by_path,
                exclude_paths=exclude_paths,
            )
            content_entry = resolved.content_entry
            content_type = github_content_type(relative_path)
            resolved_content_type = github_content_type(
                resolved.resolved_relative_path or relative_path
            )
            if resolved.symlink_chain and (
                github_content_type_is_binary(content_type)
                or github_content_type_is_binary(resolved_content_type)
            ):
                raise click.ClickException(
                    f"GitHub symlink {relative_path} selects a binary Artifact; "
                    "symlinked Artifacts are unsupported"
                )
            items.append(
                {
                    "relative_path": resolved.logical_path,
                    "blob_sha": content_entry.get("sha"),
                    "bytes": content_entry.get("size"),
                    "file_mode": content_entry.get("mode"),
                    "logical_file_mode": resolved.logical_file_mode,
                    "symlink_chain": [dict(item) for item in resolved.symlink_chain],
                    "resolved_relative_path": resolved.resolved_relative_path,
                    "content_type": content_type,
                }
            )
    payload = {
        "profile": name,
        "repo_url": repo["repo_url"],
        "ref": ref,
        "commit_sha": snapshot.commit_sha,
        "root_tree_sha": snapshot.root_tree_sha,
        "include_paths": include_paths,
        "exclude_paths": exclude_paths,
        "include_extensions": include_extensions,
        "counts": counts,
        "extension_counts": dict(sorted(extension_counts.items())),
        "items": items,
    }
    return payload


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------


@click.group(invoke_without_command=True)
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to config.toml",
)
@click.pass_context
def cli(ctx, verbose: bool, config_path: Path | None):
    """MemForge -- Auto-evolutionary agent memory layer for development teams."""
    setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config_path=config_path)

    if ctx.invoked_subcommand is None:
        ctx.exit(_dispatch_interactive())


def _dispatch_interactive() -> int:
    """Run the Clack-based interactive menu when memforge is called bare.

    Returns the exit code the launcher should use. The dispatcher is a single
    adapters: tests monkey-patch this function so they can verify routing without
    spawning Node, and the production implementation forwards to the bundled
    Node script via :func:`_run_interactive_script`.
    """
    if os.environ.get(INTERACTIVE_DISABLE_ENV):
        click.echo(cli.get_help(click.Context(cli)))
        return 0
    return _run_interactive_script()


@cli.group("eval")
def eval_group() -> None:
    """Run deterministic MemForge evaluations."""


@eval_group.command("retrieval")
@click.option("--case-set", default="retrieval-core-v1", show_default=True, help="Packaged retrieval case set id.")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "text"]),
    default="json",
    show_default=True,
    help="Output format.",
)
@click.option(
    "--db-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Base SQLite path for temporary per-case databases.",
)
@click.option("--keep-databases", is_flag=True, help="Keep per-case SQLite databases for debugging.")
@click.option("--fail-on-hard-failure", is_flag=True, help="Exit non-zero when hard failures are present.")
def eval_retrieval(
    case_set: str,
    output_format: str,
    db_path: Path | None,
    keep_databases: bool,
    fail_on_hard_failure: bool,
) -> None:
    """Run the packaged deterministic retrieval golden eval."""

    logging.getLogger().setLevel(logging.WARNING)
    if keep_databases and db_path is None:
        raise click.ClickException("--keep-databases requires --db-path so artifacts have a durable location.")
    exit_code = asyncio.run(
        _run_retrieval_eval_cli(
            case_set=case_set,
            output_format=output_format,
            db_path=db_path,
            keep_databases=keep_databases,
            fail_on_hard_failure=fail_on_hard_failure,
        )
    )
    raise click.exceptions.Exit(exit_code)


async def _run_retrieval_eval_cli(
    *,
    case_set: str,
    output_format: str,
    db_path: Path | None,
    keep_databases: bool,
    fail_on_hard_failure: bool,
) -> int:
    from memforge.evals.retrieval import load_case_set
    from memforge.evals.retrieval.runner import run_sqlite_case_set

    if db_path is not None:
        report = await run_sqlite_case_set(
            load_case_set(case_set),
            db_path=db_path,
            keep_databases=keep_databases,
        )
    else:
        with TemporaryDirectory() as tmp:
            report = await run_sqlite_case_set(
                load_case_set(case_set),
                db_path=Path(tmp) / "retrieval-eval.db",
            )

    payload = report.to_json()
    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        summary = payload["summary"]
        click.echo(
            f"retrieval eval {case_set}: {summary['case_count']} cases, {summary['hard_failures']} hard failures"
        )
        for failure in payload["hard_failures"]:
            click.echo(f"- {failure['case_id']}: {failure['message']}")
    return 1 if fail_on_hard_failure and report.hard_failures else 0


@eval_group.command("online-report")
@click.option("--from", "occurred_from", required=True, help="Inclusive RFC 3339 timestamp.")
@click.option("--to", "occurred_to", required=True, help="Exclusive RFC 3339 timestamp.")
@click.option("--event-id")
@click.option("--source-id")
@click.option("--source-type")
@click.option("--user-id", "requesting_user_id")
@click.option("--include-private", is_flag=True)
@click.option("--event-name")
@click.option("--outcome", type=click.Choice(["expected", "degraded", "rejected", "failed"]))
@click.option("--reason-code")
@click.option("--model")
@click.option("--provider")
@click.option("--deployment-revision")
@click.option("--trace-id")
@click.option("--contract-version", "extraction_contract_version")
@click.option("--limit", type=click.IntRange(1, 1000), default=1000, show_default=True)
@click.pass_context
def eval_online_report(
    ctx: click.Context,
    occurred_from: str,
    occurred_to: str,
    event_id: str | None,
    source_id: str | None,
    source_type: str | None,
    requesting_user_id: str | None,
    include_private: bool,
    event_name: str | None,
    outcome: str | None,
    reason_code: str | None,
    model: str | None,
    provider: str | None,
    deployment_revision: str | None,
    trace_id: str | None,
    extraction_contract_version: str | None,
    limit: int,
) -> None:
    """Report a bounded local online agent-evaluation cohort."""

    config: AppConfig = ctx.obj["config"]
    try:
        time_from = datetime.fromisoformat(occurred_from.replace("Z", "+00:00"))
        time_to = datetime.fromisoformat(occurred_to.replace("Z", "+00:00"))
    except ValueError as exc:
        raise click.ClickException("--from and --to must be RFC 3339 timestamps") from exc
    payload = asyncio.run(
        _run_online_evaluation_report(
            config=config,
            occurred_from=time_from,
            occurred_to=time_to,
            event_id=event_id,
            source_id=source_id,
            source_type=source_type,
            requesting_user_id=requesting_user_id,
            include_private=include_private,
            event_name=event_name,
            outcome=outcome,
            reason_code=reason_code,
            model=model,
            provider=provider,
            deployment_revision=deployment_revision,
            trace_id=trace_id,
            extraction_contract_version=extraction_contract_version,
            limit=limit,
        )
    )
    click.echo(json.dumps(payload, indent=2, sort_keys=True))


async def _run_online_evaluation_report(
    *,
    config: AppConfig,
    occurred_from: datetime,
    occurred_to: datetime,
    event_id: str | None,
    source_id: str | None,
    source_type: str | None,
    requesting_user_id: str | None,
    include_private: bool,
    event_name: str | None,
    outcome: str | None,
    reason_code: str | None,
    model: str | None,
    provider: str | None,
    deployment_revision: str | None,
    trace_id: str | None,
    extraction_contract_version: str | None,
    limit: int,
) -> dict[str, object]:
    from memforge.evals.agent_evaluation import (
        AgentRuntimeEventQuery,
        event_public_payload,
        summarize_agent_runtime_events,
    )

    db = await _get_db(config)
    try:
        events = await db.list_agent_runtime_events(
            AgentRuntimeEventQuery(
                occurred_from=occurred_from,
                occurred_to=occurred_to,
                event_id=event_id,
                source_id=source_id,
                source_type=source_type,
                requesting_user_id=requesting_user_id,
                include_private=include_private,
                event_name=event_name,
                outcome=outcome,
                reason_code=reason_code,
                model=model,
                provider=provider,
                deployment_revision=deployment_revision,
                trace_id=trace_id,
                extraction_contract_version=extraction_contract_version,
                limit=limit,
            )
        )
    finally:
        await db.close()
    return {
        "window": {
            "from": occurred_from.isoformat(),
            "to": occurred_to.isoformat(),
            "kind": "half_open",
        },
        "cohort": summarize_agent_runtime_events(events).to_payload(),
        "events": [event_public_payload(event) for event in events],
    }


@eval_group.command("offline-report")
@click.option("--run-id", required=True)
@_workspace_id_option
@click.pass_context
def eval_offline_report(
    ctx: click.Context,
    run_id: str,
    workspace_id: str | None,
) -> None:
    """Read one authorized immutable offline evaluation report via the service."""

    _emit_tool_payload(ctx, _tool_client(ctx, workspace_id).get_agent_evaluation_run(run_id))


@eval_group.command("offline-submit")
@click.option(
    "--spec",
    "spec_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@_workspace_id_option
@click.pass_context
def eval_offline_submit(ctx: click.Context, spec_path: Path, workspace_id: str | None) -> None:
    """Submit one pinned offline evaluation JSON spec via the service."""

    try:
        payload = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise click.ClickException(f"Cannot read evaluation spec: {exc}") from exc
    if not isinstance(payload, dict):
        raise click.ClickException("Evaluation spec must be a JSON object")
    _emit_tool_payload(ctx, _tool_client(ctx, workspace_id).admit_agent_evaluation_run(payload))


def _read_json_object(path: Path, what: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise click.ClickException(f"Cannot read {what}: {exc}") from exc
    if not isinstance(payload, dict):
        raise click.ClickException(f"{what[0].upper()}{what[1:]} must be a JSON object")
    return payload


@eval_group.command("offline-run")
@click.option(
    "--spec",
    "spec_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--timeout", "timeout_s", default=1800, show_default=True, help="Seconds to wait for the run.")
@click.option("--poll", "poll_s", default=10, show_default=True, help="Seconds between report reads.")
@_workspace_id_option
@click.pass_context
def eval_offline_run(
    ctx: click.Context,
    spec_path: Path,
    timeout_s: int,
    poll_s: int,
    workspace_id: str | None,
) -> None:
    """Submit one offline evaluation spec, wait for its run and print a Markdown summary.

    Exits 0 when the run's verdict is pass, 1 when it is fail or unknown, and 2
    when the run does not finish within --timeout.
    """

    from memforge.evals.offline_run import render_report_markdown, report_verdict, wait_for_report

    client = _tool_client(ctx, workspace_id)
    admitted = client.admit_agent_evaluation_run(_read_json_object(spec_path, "evaluation spec"))
    if admitted.get("error") or admitted.get("code"):
        _emit_tool_payload(ctx, admitted)
        ctx.exit(1)
    run_id = str(admitted["run"]["run_id"])
    click.echo(f"Run {run_id} admitted; waiting for the worker.", err=True)
    report = wait_for_report(lambda: client.get_agent_evaluation_run(run_id), timeout_s=timeout_s, poll_s=poll_s)
    if report is None:
        click.echo(f"Run {run_id} did not finish within {timeout_s} s.", err=True)
        ctx.exit(2)
    if report.get("error"):
        _emit_tool_payload(ctx, dict(report))
        ctx.exit(1)
    click.echo(render_report_markdown(report))
    ctx.exit(0 if report_verdict(report) == "pass" else 1)


@eval_group.command("seed-derivation-cases")
@click.option(
    "--spec",
    "spec_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@_workspace_id_option
@click.pass_context
def eval_seed_derivation_cases(ctx: click.Context, spec_path: Path, workspace_id: str | None) -> None:
    """Pin synced Documents as derivation cases with their rubrics and freeze their cohort.

    The spec names ``selection_policy_version``, an optional ``role`` and the
    ``cases``, each a ``doc_id`` with its ``rubric`` and, to pin a derivation
    other than the Document's first, its ``derivation_id``. Requires a
    maintenance operator.
    """

    _emit_tool_payload(
        ctx,
        _tool_client(ctx, workspace_id).seed_derivation_evaluation_cases(
            _read_json_object(spec_path, "derivation case spec")
        ),
    )


@eval_group.command("seed-support-cases")
@click.option(
    "--spec",
    "spec_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@_workspace_id_option
@click.pass_context
def eval_seed_support_cases(ctx: click.Context, spec_path: Path, workspace_id: str | None) -> None:
    """Pin applied updates of synced Documents as Support cases and freeze their cohort.

    The spec names ``selection_policy_version``, an optional ``role`` and the
    ``cases``, each a ``doc_id`` with its ``rubric`` (``expected_support``:
    Memory id to ``supported`` or ``unsupported``) and, to pin an update other
    than the latest one the model judged Support in, its ``derivation_id``.
    Requires a maintenance operator.
    """

    _emit_tool_payload(
        ctx,
        _tool_client(ctx, workspace_id).seed_support_evaluation_cases(
            _read_json_object(spec_path, "Support case spec")
        ),
    )


@eval_group.command("langfuse-policy-approve")
@click.option("--source-id", required=True)
@click.option("--policy-version", required=True)
@click.option("--queue-id", required=True)
@_workspace_id_option
@click.pass_context
def eval_langfuse_policy_approve(
    ctx: click.Context,
    source_id: str,
    policy_version: str,
    queue_id: str,
    workspace_id: str | None,
) -> None:
    """Approve one Source for one preconfigured Langfuse reviewer queue."""

    _emit_tool_payload(
        ctx,
        _tool_client(ctx, workspace_id).approve_langfuse_annotation_policy(
            {
                "source_id": source_id,
                "policy_version": policy_version,
                "queue_id": queue_id,
            }
        ),
    )


@eval_group.command("langfuse-annotation-export")
@click.option("--result-id", required=True)
@click.option("--content-policy-id", required=True)
@click.option("--criterion", required=True)
@click.option("--rubric-version", required=True)
@click.option("--reviewer-id", "provider_reviewer_id", required=True)
@click.option("--queue-id", required=True)
@click.option("--score-config-id", required=True)
@_workspace_id_option
@click.pass_context
def eval_langfuse_annotation_export(
    ctx: click.Context,
    result_id: str,
    content_policy_id: str,
    criterion: str,
    rubric_version: str,
    provider_reviewer_id: str,
    queue_id: str,
    score_config_id: str,
    workspace_id: str | None,
) -> None:
    """Create or resume one blinded Langfuse reviewer task."""

    _emit_tool_payload(
        ctx,
        _tool_client(ctx, workspace_id).export_langfuse_annotation(
            {
                "result_id": result_id,
                "content_policy_id": content_policy_id,
                "criterion": criterion,
                "rubric_version": rubric_version,
                "provider_reviewer_id": provider_reviewer_id,
                "queue_id": queue_id,
                "score_config_id": score_config_id,
            }
        ),
    )


@eval_group.command("langfuse-annotation-status")
@click.option("--task-id", required=True)
@_workspace_id_option
@click.pass_context
def eval_langfuse_annotation_status(ctx: click.Context, task_id: str, workspace_id: str | None) -> None:
    """Read one content-free Langfuse annotation task status."""

    _emit_tool_payload(ctx, _tool_client(ctx, workspace_id).get_langfuse_annotation_task(task_id))


@eval_group.command("langfuse-annotation-import")
@click.option("--task-id", required=True)
@_workspace_id_option
@click.pass_context
def eval_langfuse_annotation_import(ctx: click.Context, task_id: str, workspace_id: str | None) -> None:
    """Import one completed Langfuse annotation Score."""

    _emit_tool_payload(ctx, _tool_client(ctx, workspace_id).import_langfuse_annotation(task_id))


@eval_group.command("purge-runtime-events")
@click.option("--before", "occurred_before", help="Exclusive RFC 3339 cutoff; defaults to configured retention.")
@click.option("--limit", type=click.IntRange(1, 10_000), help="Bounded delete count.")
@click.pass_context
def eval_purge_runtime_events(
    ctx: click.Context,
    occurred_before: str | None,
    limit: int | None,
) -> None:
    """Apply one bounded AgentRuntimeEvent retention batch."""

    config: AppConfig = ctx.obj["config"]
    if occurred_before:
        try:
            cutoff = datetime.fromisoformat(occurred_before.replace("Z", "+00:00"))
        except ValueError as exc:
            raise click.ClickException("--before must be an RFC 3339 timestamp") from exc
    else:
        cutoff = datetime.now(timezone.utc) - timedelta(
            days=config.agent_evaluation.runtime_event_retention_days
        )
    if cutoff.tzinfo is None or cutoff.utcoffset() is None:
        raise click.ClickException("--before must include an explicit timezone")
    deleted = asyncio.run(
        _purge_runtime_events(
            config=config,
            occurred_before=cutoff,
            limit=limit or config.agent_evaluation.runtime_event_purge_batch_size,
        )
    )
    click.echo(json.dumps({"deleted": deleted, "before": cutoff.isoformat()}, sort_keys=True))


async def _purge_runtime_events(
    *,
    config: AppConfig,
    occurred_before: datetime,
    limit: int,
) -> int:
    db = await _get_db(config)
    try:
        return await db.purge_agent_runtime_events(
            occurred_before=occurred_before,
            limit=limit,
        )
    finally:
        await db.close()


def _interactive_resource_dir() -> Path:
    return Path(__file__).resolve().parent / INTERACTIVE_RESOURCE_DIR


def _interactive_script_path(resource_dir: Path | None = None) -> Path | None:
    override = os.environ.get(INTERACTIVE_SCRIPT_ENV, "").strip()
    if override:
        candidate = Path(override).expanduser()
        return candidate if candidate.exists() else None
    resource_dir = resource_dir or _interactive_resource_dir()
    candidate = resource_dir / "index.mjs"
    return candidate if candidate.exists() else None


def _run_interactive_script() -> int:
    resource_dir = _interactive_resource_dir()
    script = _interactive_script_path(resource_dir)
    if script is None:
        log_console.print(
            "[yellow]Interactive UI not available: packaged interactive assets are missing.[/]\n"
            "Run scriptable subcommands directly. See [bold]memforge --help[/]."
        )
        return 2

    node_bin = shutil.which("node")
    if node_bin is None:
        log_console.print(
            "[yellow]Interactive UI requires Node.js (>=18) on PATH.[/]\nInstall Node, then re-run [bold]memforge[/]."
        )
        return 2

    if not os.environ.get(INTERACTIVE_SCRIPT_ENV, "").strip():
        try:
            workspace = _prepare_interactive_workspace(resource_dir)
        except RuntimeError as exc:
            log_console.print(
                f"[yellow]{exc}[/]\n"
                "Run scriptable subcommands directly, or retry after npm can install the interactive UI."
            )
            return 2
        script = workspace / "index.mjs"

    env = os.environ.copy()
    env.setdefault(INTERACTIVE_BIN_ENV, sys.argv[0] or "memforge")
    env[INTERACTIVE_DISABLE_ENV] = "1"

    completed = subprocess.run([node_bin, str(script)], env=env)
    return completed.returncode


def _prepare_interactive_workspace(resource_dir: Path | None = None) -> Path:
    resource_dir = resource_dir or _interactive_resource_dir()
    _validate_interactive_resources(resource_dir)
    workspace = _interactive_cache_root() / _interactive_cache_key(resource_dir)
    if (workspace / INTERACTIVE_DEPENDENCY_SENTINEL).exists():
        return workspace

    with _interactive_install_lock(workspace):
        _copy_interactive_resources(resource_dir, workspace)
        if not (workspace / INTERACTIVE_DEPENDENCY_SENTINEL).exists():
            _install_interactive_dependencies(workspace)
    return workspace


def _validate_interactive_resources(resource_dir: Path) -> None:
    missing = [
        name for name in ("index.mjs", "package.json", "package-lock.json") if not (resource_dir / name).exists()
    ]
    if missing:
        raise RuntimeError(
            "Interactive UI package is incomplete: missing " + ", ".join(missing) + f" under {resource_dir}."
        )


def _interactive_cache_root() -> Path:
    configured = os.environ.get(INTERACTIVE_CACHE_ENV, "").strip()
    if configured:
        return Path(configured).expanduser()
    xdg_cache = os.environ.get("XDG_CACHE_HOME", "").strip()
    base = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
    return base / "memforge" / "interactive-cli"


def _interactive_cache_key(resource_dir: Path) -> str:
    try:
        package_version = metadata.version(PYPI_DISTRIBUTION_NAME)
    except metadata.PackageNotFoundError:
        package_version = "0.0.0"
    digest = hashlib.sha256()
    for name in ("package-lock.json", "package.json", "index.mjs"):
        digest.update(name.encode("utf-8"))
        digest.update((resource_dir / name).read_bytes())
    return f"{package_version}-{digest.hexdigest()[:12]}"


def _copy_interactive_resources(resource_dir: Path, workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    for name in ("index.mjs", "package.json", "package-lock.json"):
        source = resource_dir / name
        target = workspace / name
        if not target.exists() or target.read_bytes() != source.read_bytes():
            shutil.copy2(source, target)


class _interactive_install_lock:
    def __init__(self, workspace: Path) -> None:
        self.path = workspace / INTERACTIVE_INSTALL_LOCK
        self.fd: int | None = None

    def __enter__(self) -> "_interactive_install_lock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + INTERACTIVE_INSTALL_LOCK_TIMEOUT_SECONDS
        while True:
            try:
                self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self.fd, str(os.getpid()).encode("utf-8"))
                return self
            except FileExistsError:
                if self._is_stale():
                    self.path.unlink(missing_ok=True)
                    continue
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"Timed out waiting for interactive UI install lock at {self.path}.")
                time.sleep(0.1)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.fd is not None:
            os.close(self.fd)
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    def _is_stale(self) -> bool:
        try:
            age = time.time() - self.path.stat().st_mtime
        except FileNotFoundError:
            return False
        return age > INTERACTIVE_INSTALL_LOCK_STALE_SECONDS


def _install_interactive_dependencies(workspace: Path) -> None:
    npm_bin = shutil.which("npm")
    if npm_bin is None:
        raise RuntimeError("Interactive UI requires npm on PATH to prepare its first-run dependencies.")
    try:
        subprocess.run(
            [npm_bin, "ci", "--omit=dev", "--no-audit", "--no-fund"],
            cwd=workspace,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Interactive UI dependency installation failed in {workspace}. "
            f"Retry manually with: cd {workspace} && npm ci --omit=dev --no-audit --no-fund"
        ) from exc


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


@cli.command()
@click.pass_context
def init(ctx):
    """Create directories, initialise the database, and create an admin user."""
    config: AppConfig = ctx.obj["config"]
    base = config.base_dir

    # Create directory structure
    dirs = [
        base / "db",
        base / "vectors" / "chroma",
        base / "documents",
        base / "logs",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        console.print(f"  [dim]Created {d}[/]")

    async def _seed():
        import bcrypt

        db = await _get_db(config)

        # Check for existing admin user
        existing = await db.get_user_by_username("admin")
        if existing:
            console.print("Admin user already exists.")
        else:
            password = click.prompt(
                "Set admin password",
                hide_input=True,
                confirmation_prompt=True,
            )
            pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
            await db.create_user(
                username="admin",
                display_name="Admin",
                password_hash=pw_hash,
                role="admin",
            )
            console.print("[green]Created admin user.[/]")

        await db.close()

    asyncio.run(_seed())

    console.print(f"\n[bold green]Initialised MemForge at {base}[/]")
    console.print("\nNext steps:")
    console.print("  1. Configure API keys in config.toml or via environment variables")
    console.print("  2. Run: memforge api     (start the Admin API)")
    console.print("  3. Run: cd admin-ui && npm run dev")
    console.print("  4. Add sources in the admin UI")
    console.print("  5. Run sync from the UI, schedule sync, or run: memforge sync")


# ---------------------------------------------------------------------------
# target group
# ---------------------------------------------------------------------------


@cli.group("target")
def target():
    """Manage MemForge API targets."""
    pass


@target.command("list")
@click.pass_context
def target_list(ctx):
    """List configured API targets."""
    _emit_tool_payload(ctx, _read_cli_config())


@target.command("add")
@click.argument("name")
@click.option("--api-url", required=True, help="MemForge API URL for this target.")
@click.option(
    "--token-env", default="MEMFORGE_API_TOKEN", show_default=True, help="Environment variable for the token."
)
@click.pass_context
def target_add(ctx, name: str, api_url: str, token_env: str):
    """Add or update an API target and make it active."""
    payload = _set_cli_target(
        name=name,
        target=_build_cli_target(api_url=api_url),
        token_env=token_env,
    )
    _emit_tool_payload(ctx, payload)


def _set_cli_target(*, name: str, target: MemForgeTarget, token_env: str) -> dict[str, Any]:
    """Persist one active CLI target for explicit and guided setup flows."""
    name = name.strip()
    if not name:
        raise click.ClickException("Target name is required.")
    data = _read_cli_config()
    targets = data.setdefault("targets", {})
    targets[name] = {
        "api_url": target.origin,
        "edition": target.edition.value,
        "token_env": token_env.strip(),
    }
    data["active"] = name
    _write_cli_config(data)
    return {
        "ok": True,
        "active": name,
        "edition": target.edition.value,
        "api_url": target.origin,
    }


@target.command("use")
@click.argument("name")
@click.pass_context
def target_use(ctx, name: str):
    """Set the active API target."""
    data = _read_cli_config()
    targets = data.get("targets") if isinstance(data.get("targets"), dict) else {}
    if name not in targets:
        raise click.ClickException(f"Unknown target: {name}")
    data["active"] = name
    _write_cli_config(data)
    _emit_tool_payload(ctx, {"ok": True, "active": name})


@target.command("check")
@click.pass_context
def target_check(ctx):
    """Check the active API target health."""
    payload = _tool_client(ctx, None).health()
    _emit_tool_payload(ctx, payload)


# ---------------------------------------------------------------------------
# daemon service and guided setup
# ---------------------------------------------------------------------------


def _daemon_service_manager():
    from memforge.local_agent.service import DaemonServiceManager

    return DaemonServiceManager.current()


def _current_cli_executable() -> str:
    invoked = Path(sys.argv[0]).expanduser()
    if invoked.is_absolute() and invoked.exists() and os.access(invoked, os.X_OK):
        return str(invoked)
    discovered = shutil.which(sys.argv[0]) or shutil.which("memforge")
    if discovered:
        return discovered
    raise click.ClickException("Could not locate the installed `memforge` executable.")


def _daemon_service_action(action: str) -> dict[str, Any]:
    from memforge.local_agent.service import DaemonServiceError

    manager = _daemon_service_manager()
    try:
        return getattr(manager, action)()
    except DaemonServiceError as exc:
        raise click.ClickException(str(exc)) from exc


def _require_safe_daemon_service_install(manager) -> None:
    from memforge.local_agent.service import DaemonServiceError

    try:
        native_status = manager.status()
    except DaemonServiceError as exc:
        raise click.ClickException(str(exc)) from exc
    runtime_status = _local_agent_daemon_status(_local_agent_state_store().load())
    if runtime_status["status"] == "running" and not native_status.get("loaded"):
        raise click.ClickException(
            "A foreground or unmanaged MemForge daemon is already running. Stop it before installing the user service."
        )


@cli.group("daemon")
def daemon_service():
    """Install and manage the MemForge local daemon user service."""


@daemon_service.command("install")
@click.pass_context
def daemon_service_install(ctx):
    """Install and start the daemon as a login user service."""
    from memforge.local_agent.service import DaemonServiceError

    resolved = _resolve_api_target(ctx.obj["config"])
    manager = _daemon_service_manager()
    _require_safe_daemon_service_install(manager)
    api_token = resolved.api_token
    if resolved.target.authentication_required and not api_token:
        token_name = resolved.token_env or "MEMFORGE_API_TOKEN"
        raise click.ClickException(f"Set {token_name} before installing the daemon service.")
    try:
        payload = manager.install(
            executable=_current_cli_executable(),
            api_url=resolved.target.origin,
            api_token=api_token,
            edition=resolved.target.edition.value,
        )
    except DaemonServiceError as exc:
        raise click.ClickException(str(exc)) from exc
    _emit_tool_payload(ctx, payload)


@daemon_service.command("status")
@click.option("--verbose", is_flag=True, help="Include the raw local daemon state file.")
@click.pass_context
def daemon_service_status(ctx, verbose: bool):
    """Show native, local runtime, and server-observed connection status."""
    _emit_tool_payload(ctx, _daemon_health_payload(ctx, verbose=verbose))


def _daemon_health_payload(ctx, *, verbose: bool) -> dict[str, Any]:
    from memforge.local_agent.service import DaemonServiceError

    manager = _daemon_service_manager()
    try:
        service = manager.status()
    except DaemonServiceError as exc:
        raise click.ClickException(str(exc)) from exc
    runtime = _local_agent_status_payload(ctx, verbose=verbose)
    try:
        environment = manager.environment()
        target = _build_cli_target(
            api_url=environment["MEMFORGE_API_URL"],
            edition=environment.get("MEMFORGE_EDITION"),
        )
        connection = ToolClient(
            target=target,
            api_token=environment.get("MEMFORGE_API_TOKEN"),
        ).get_local_agent_status()
    except (DaemonServiceError, click.ClickException) as exc:
        connection = {
            "status": "unconfigured",
            "error": str(exc),
        }
    healthy = (
        service.get("running") is True
        and runtime.get("status") == "running"
        and connection.get("status") == "online"
    )
    return {
        "status": "healthy" if healthy else "unhealthy",
        "service": service,
        "runtime": runtime,
        "connection": connection,
    }


@daemon_service.command("check")
@click.pass_context
def daemon_service_check(ctx):
    """Exit nonzero unless the native service is running and connected to MemForge."""
    payload = _daemon_health_payload(ctx, verbose=False)
    if payload["status"] != "healthy":
        payload["error"] = "MemForge daemon health check failed."
    _emit_tool_payload(ctx, payload)


@daemon_service.command("start")
@click.pass_context
def daemon_service_start(ctx):
    """Start the installed daemon user service."""
    _emit_tool_payload(ctx, _daemon_service_action("start"))


@daemon_service.command("stop")
@click.pass_context
def daemon_service_stop(ctx):
    """Stop the installed daemon user service until the next start or login."""
    _emit_tool_payload(ctx, _daemon_service_action("stop"))


@daemon_service.command("restart")
@click.pass_context
def daemon_service_restart(ctx):
    """Restart the installed daemon user service."""
    _emit_tool_payload(ctx, _daemon_service_action("restart"))


@daemon_service.command("uninstall")
@click.option("--yes", is_flag=True, help="Remove the service without prompting.")
@click.pass_context
def daemon_service_uninstall(ctx, yes: bool):
    """Stop the daemon and remove its user-service configuration and credential."""
    if not yes and not click.confirm("Uninstall the MemForge daemon service?", default=False):
        _emit_tool_payload(ctx, {"ok": True, "cancelled": True})
        return
    _emit_tool_payload(ctx, _daemon_service_action("uninstall"))


@daemon_service.command("logs")
@click.option("--lines", default=100, show_default=True, type=click.IntRange(min=1))
@click.option("--follow", "follow_logs", is_flag=True, help="Keep streaming new daemon log entries.")
def daemon_service_logs(lines: int, follow_logs: bool):
    """Show native daemon-service logs."""
    from memforge.local_agent.service import DaemonServiceError

    try:
        exit_code = _daemon_service_manager().logs(lines=lines, follow=follow_logs)
    except DaemonServiceError as exc:
        raise click.ClickException(str(exc)) from exc
    raise click.exceptions.Exit(exit_code)


@daemon_service.command("_run", hidden=True)
@click.pass_context
def daemon_service_run(ctx):
    """Run from the native user-service adapter with its stored credential."""
    from memforge.local_agent.service import DaemonServiceError

    try:
        environment = _daemon_service_manager().environment()
    except DaemonServiceError as exc:
        raise click.ClickException(str(exc)) from exc
    os.environ.update(environment)
    if "MEMFORGE_API_TOKEN" not in environment:
        os.environ.pop("MEMFORGE_API_TOKEN", None)
    _run_local_agent_daemon(ctx, browser=None, poll_interval_seconds=10, cloud_job_wait_seconds=25)


def _status_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _wait_for_local_agent_online(
    client: ToolClient,
    *,
    after_last_seen_at: str | None,
    timeout_seconds: float = 35.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    baseline = _status_timestamp(after_last_seen_at)
    latest: dict[str, Any] = {"status": "offline"}
    while True:
        latest = client.get_local_agent_status()
        observed = _status_timestamp(latest.get("last_seen_at"))
        fresh = observed is not None and (baseline is None or observed > baseline)
        if latest.get("status") == "online" and fresh:
            return {**latest, "fresh": True}
        if latest.get("error"):
            return latest
        if time.monotonic() >= deadline:
            return {**latest, "fresh": False}
        time.sleep(0.5)


@cli.command("setup")
@click.option("--api-url", default=None, help="MemForge API URL. Uses the active target or prompts when absent.")
@click.option("--target-name", default=None, help="Name used when saving a newly configured target.")
@click.option(
    "--token-env", default="MEMFORGE_API_TOKEN", show_default=True, help="Environment variable holding the token."
)
@click.pass_context
def setup_command(ctx, api_url: str | None, target_name: str | None, token_env: str):
    """Configure a target and install the local daemon user service."""
    from memforge.api_target import Edition
    from memforge.local_agent.service import DaemonServiceError

    cli_config_path = _cli_config_path()
    cli_config_snapshot = cli_config_path.read_bytes() if cli_config_path.exists() else None
    cli_config = _read_cli_config()
    active_name = str(cli_config.get("active") or "")
    targets = cli_config.get("targets") if isinstance(cli_config.get("targets"), dict) else {}
    has_configured_target = bool(os.getenv("MEMFORGE_API_URL", "").strip()) or active_name in targets
    candidate_url = api_url
    if candidate_url is None and not has_configured_target:
        candidate_url = click.prompt(
            "MemForge API URL",
            default="http://127.0.0.1:8765",
            show_default=True,
        )
    if candidate_url:
        target_value = _build_cli_target(api_url=candidate_url)
        candidate_name = (target_name or ("cloud" if target_value.edition is Edition.CLOUD else "local")).strip()
        if not candidate_name:
            raise click.ClickException("Target name is required.")
        resolved = _ResolvedCliTarget(
            target=target_value,
            api_token=(os.getenv(token_env) if token_env else None) or _daemon_keyring_api_token(target_value.origin),
            active_target=candidate_name,
            token_env=token_env.strip(),
        )
    else:
        resolved = _resolve_api_target(ctx.obj["config"])
    manager = _daemon_service_manager()
    _require_safe_daemon_service_install(manager)
    api_token = resolved.api_token
    if resolved.target.authentication_required and not api_token:
        api_token = click.prompt("MemForge API token", hide_input=True, confirmation_prompt=True)

    client = ToolClient(target=resolved.target, api_token=api_token)
    health = client.health()
    if health.get("error"):
        _emit_tool_payload(ctx, health)
        return
    try:
        replacement = manager.replace(
            executable=_current_cli_executable(),
            api_url=resolved.target.origin,
            api_token=api_token,
            edition=resolved.target.edition.value,
        )
    except DaemonServiceError as exc:
        raise click.ClickException(str(exc)) from exc

    baseline_status = client.get_local_agent_status()
    if baseline_status.get("error"):
        try:
            replacement.rollback()
        except DaemonServiceError as rollback_error:
            raise click.ClickException(
                f"Daemon status baseline failed and the previous service could not be restored: {rollback_error}"
            ) from rollback_error
        _emit_tool_payload(ctx, baseline_status)
        return

    daemon_status = _wait_for_local_agent_online(
        client,
        after_last_seen_at=baseline_status.get("last_seen_at"),
    )
    if daemon_status.get("status") != "online" or daemon_status.get("fresh") is not True:
        try:
            replacement.rollback()
        except DaemonServiceError as rollback_error:
            raise click.ClickException(
                "Daemon heartbeat verification failed and the previous service could not be restored: "
                f"{rollback_error}"
            ) from rollback_error
        payload = dict(daemon_status)
        payload["error"] = "Daemon service did not produce a fresh server heartbeat."
        payload["recommendation"] = "Run `memforge daemon logs` and retry setup."
        _emit_tool_payload(ctx, payload)
        return
    try:
        _set_cli_target(
            name=resolved.active_target or ("cloud" if resolved.target.edition is Edition.CLOUD else "local"),
            target=resolved.target,
            token_env=resolved.token_env or token_env,
        )
    except Exception as config_error:
        try:
            replacement.rollback()
        except DaemonServiceError as rollback_error:
            raise click.ClickException(
                f"Could not save the CLI target ({config_error}); service rollback also failed: {rollback_error}"
            ) from config_error
        raise click.ClickException(f"Could not save the CLI target: {config_error}") from config_error
    try:
        replacement.commit()
    except DaemonServiceError as commit_error:
        recovery_errors: list[str] = []
        try:
            if cli_config_snapshot is not None:
                _write_cli_config_bytes(cli_config_path, cli_config_snapshot)
            else:
                cli_config_path.unlink(missing_ok=True)
        except Exception as config_restore_error:
            recovery_errors.append(f"CLI target restore failed: {config_restore_error}")
        try:
            replacement.rollback()
        except DaemonServiceError as service_restore_error:
            recovery_errors.append(f"service rollback failed: {service_restore_error}")
        detail = f"Could not commit daemon setup: {commit_error}"
        if recovery_errors:
            detail += "; " + "; ".join(recovery_errors)
        raise click.ClickException(detail) from commit_error
    payload: dict[str, Any] = {
        "ok": True,
        "target": {
            "edition": resolved.target.edition.value,
            "api_url": resolved.target.origin,
        },
        "service": manager.status(),
    }
    payload["daemon"] = daemon_status
    _emit_tool_payload(ctx, payload)


# ---------------------------------------------------------------------------
# sync
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--source", "-s", default=None, help="Sync only this source name")
@click.pass_context
def sync(ctx, source: str | None):
    """Run the sync pipeline for all sources (or a specific one)."""

    async def _run():
        config: AppConfig = ctx.obj["config"]
        from memforge.genes.conversation_gene import ConversationGene
        from memforge.genes import GENE_REGISTRY
        from memforge.runtime import build_sync_runtime, run_source_sync

        db = await _get_db(config)
        try:
            runtime = await build_sync_runtime(db, config)
        except Exception as e:
            console.print(f"[red]Sync startup failed: {e}[/]")
            await db.close()
            return

        # Load sources from DB
        db_sources = await db.list_sources()
        if not db_sources:
            console.print("[yellow]No sources configured. Add sources first.[/]")
            await db.close()
            return

        for src in db_sources:
            if source and src["name"] != source:
                continue
            if src.get("status") != "active":
                continue

            source_type = src["type"]
            gene_class = GENE_REGISTRY.get(source_type)
            if gene_class is not None and issubclass(gene_class, ConversationGene):
                # Its local agent collects each attempt; the server processes what was uploaded.
                console.print(
                    f"\n[yellow]Skipping {src['name']}[/] (type={source_type}): "
                    "a conversation Source is collected by its local agent; sync it from the Admin UI."
                )
                continue

            console.print(f"\n[bold]Syncing: {src['name']}[/] (type={source_type})")

            try:
                state = await run_source_sync(
                    db=db,
                    config=config,
                    source=src,
                    runtime=runtime,
                    progress_callback=lambda p: (
                        console.print(f"  [dim]{p.get('status', '')}[/]") if p.get("status") else None
                    ),
                )
                status_color = (
                    "green"
                    if state.last_sync_status == "success"
                    else "yellow"
                    if state.last_sync_status == "partial"
                    else "red"
                )
                summary = (
                    f"  [{status_color}]Done:[/] {state.docs_processed} discovered, "
                    f"{state.docs_updated} updated, {state.docs_failed} failed, "
                    f"{state.memories_extracted} memories extracted"
                )
                if state.skipped_reading_group_count:
                    summary += (
                        f", {state.skipped_reading_group_count} document part(s) "
                        "unread by the generation model"
                    )
                console.print(summary)
            except Exception as e:
                console.print(f"  [red]Sync failed: {e}[/]")

        await db.close()
        console.print("\n[bold green]Sync complete![/]")

    asyncio.run(_run())


@cli.command("search")
@click.argument("query", required=False, default="")
@click.option("--top-k", default=DEFAULT_SEARCH_TOP_K, show_default=True, type=int, help="Maximum number of results.")
@click.option(
    "--intent",
    type=click.Choice(RANKED_RETRIEVAL_INTENTS),
    default=None,
    help="Optional ranked retrieval goal hint.",
)
@click.option(
    "--type",
    "memory_types",
    multiple=True,
    type=click.Choice(["fact", "decision", "convention", "procedure"]),
    help="Filter by memory type. Repeat for multiple types.",
)
@click.option(
    "--source-id",
    "source_ids",
    multiple=True,
    help="Exact source ID from `memforge sources searchable`. Repeat for multiple sources.",
)
@click.option("--entity", "entities", multiple=True, help="Entity hint. Repeat for multiple entities.")
@click.option("--start-date", default=None, help="Optional YYYY-MM-DD lower bound for date filtering.")
@click.option("--end-date", default=None, help="Optional YYYY-MM-DD upper bound for date filtering.")
@click.option(
    "--date-type",
    default="source_updated_at",
    show_default=True,
    type=click.Choice(["source_updated_at", "memory_updated_at"]),
    help="Date field to filter when --start-date or --end-date is provided.",
)
@click.option("--include-superseded", is_flag=True, help="Include superseded memories.")
@_workspace_id_option
@click.pass_context
def search(
    ctx,
    query: str,
    top_k: int,
    intent: str | None,
    memory_types: tuple[str, ...],
    source_ids: tuple[str, ...],
    entities: tuple[str, ...],
    start_date: str | None,
    end_date: str | None,
    date_type: str,
    include_superseded: bool,
    workspace_id: str | None,
):
    """Search MemForge using the same service path as the MCP search tool."""
    time_range = (
        {k: v for k, v in {"date_type": date_type, "start_date": start_date, "end_date": end_date}.items() if v}
        if start_date or end_date
        else {}
    )
    kwargs: dict = {
        "query": query,
        "top_k": top_k,
        "include_private": True,
        "include_superseded": include_superseded,
    }
    if intent is not None:
        kwargs["intent"] = intent
    if memory_types:
        kwargs["memory_types"] = list(memory_types)
    if source_ids:
        kwargs["source_filter"] = {"source_ids": list(source_ids)}
    if entities:
        kwargs["entities"] = list(entities)
    if time_range:
        kwargs["time_range"] = time_range
    payload = _project_tool_client(ctx, workspace_id).search(**kwargs)
    _emit_tool_payload(ctx, payload)


@cli.command("get-memory")
@click.argument("memory_id")
@_workspace_id_option
@click.pass_context
def get_memory(ctx, memory_id: str, workspace_id: str | None):
    """Fetch full memory detail and provenance by memory ID."""
    payload = _project_tool_client(ctx, workspace_id).get_memory(memory_id)
    _emit_tool_payload(ctx, payload)


@cli.command("get-resource")
@click.argument("url")
@click.option("--mode", default="text", show_default=True, type=click.Choice(["text", "file", "base64"]))
@click.option("--max-chars", default=120_000, show_default=True, type=int, help="Maximum text characters to print.")
@click.option("--max-bytes", default=2_000_000, show_default=True, type=int, help="Maximum bytes for inline modes.")
@_workspace_id_option
@click.pass_context
def get_resource(ctx, url: str, mode: str, max_chars: int, max_bytes: int, workspace_id: str | None):
    """Fetch a source artifact URL returned by get-memory."""
    payload = _project_tool_client(ctx, workspace_id).get_resource(
        url=url,
        mode=mode,
        max_chars=max_chars,
        max_bytes=max_bytes,
    )
    _emit_tool_payload(ctx, payload)


@cli.group("memory")
def memory():
    """Search memories and fetch provenance-backed artifacts."""
    pass


@memory.command("search")
@click.argument("query", required=False, default="")
@click.option("--top-k", default=DEFAULT_SEARCH_TOP_K, show_default=True, type=int, help="Maximum number of results.")
@click.option(
    "--type",
    "memory_types",
    multiple=True,
    type=click.Choice(["fact", "decision", "convention", "procedure"]),
    help="Filter by memory type. Repeat for multiple types.",
)
@click.option(
    "--source-id",
    "source_ids",
    multiple=True,
    help="Exact source ID from `memforge sources searchable`. Repeat for multiple sources.",
)
@click.option("--entity", "entities", multiple=True, help="Entity hint. Repeat for multiple entities.")
@click.option("--start-date", default=None, help="Optional YYYY-MM-DD lower bound for date filtering.")
@click.option("--end-date", default=None, help="Optional YYYY-MM-DD upper bound for date filtering.")
@click.option(
    "--date-type",
    default="source_updated_at",
    show_default=True,
    type=click.Choice(["source_updated_at", "memory_updated_at"]),
    help="Date field to filter when --start-date or --end-date is provided.",
)
@click.option("--include-superseded", is_flag=True, help="Include superseded memories.")
@_workspace_id_option
@click.pass_context
def memory_search(
    ctx,
    query: str,
    top_k: int,
    memory_types: tuple[str, ...],
    source_ids: tuple[str, ...],
    entities: tuple[str, ...],
    start_date: str | None,
    end_date: str | None,
    date_type: str,
    include_superseded: bool,
    workspace_id: str | None,
):
    """Search MemForge memories."""
    time_range = (
        {k: v for k, v in {"date_type": date_type, "start_date": start_date, "end_date": end_date}.items() if v}
        if start_date or end_date
        else {}
    )
    kwargs: dict = {
        "query": query,
        "top_k": top_k,
        "include_private": True,
        "include_superseded": include_superseded,
    }
    if memory_types:
        kwargs["memory_types"] = list(memory_types)
    if source_ids:
        kwargs["source_filter"] = {"source_ids": list(source_ids)}
    if entities:
        kwargs["entities"] = list(entities)
    if time_range:
        kwargs["time_range"] = time_range
    payload = _project_tool_client(ctx, workspace_id).search(**kwargs)
    _emit_tool_payload(ctx, payload)


@memory.command("get")
@click.argument("memory_id")
@_workspace_id_option
@click.pass_context
def memory_get(ctx, memory_id: str, workspace_id: str | None):
    """Fetch memory detail and provenance."""
    payload = _project_tool_client(ctx, workspace_id).get_memory(memory_id)
    _emit_tool_payload(ctx, payload)


@memory.command("resource")
@click.argument("url")
@click.option("--mode", default="text", show_default=True, type=click.Choice(["text", "file", "base64"]))
@click.option("--max-chars", default=120_000, show_default=True, type=int, help="Maximum text characters to print.")
@click.option("--max-bytes", default=2_000_000, show_default=True, type=int, help="Maximum bytes for inline modes.")
@_workspace_id_option
@click.pass_context
def memory_resource(ctx, url: str, mode: str, max_chars: int, max_bytes: int, workspace_id: str | None):
    """Fetch a source artifact URL returned by get-memory."""
    payload = _project_tool_client(ctx, workspace_id).get_resource(
        url=url,
        mode=mode,
        max_chars=max_chars,
        max_bytes=max_bytes,
    )
    _emit_tool_payload(ctx, payload)


@cli.command()
@click.option("--port", default=None, type=int, help="Port to listen on (default: from config)")
@click.option("--host", default="127.0.0.1", show_default=True, help="Host interface to bind")
@click.pass_context
def api(ctx, port: int | None, host: str):
    """Start the Admin REST API server."""
    import uvicorn

    config: AppConfig = ctx.obj["config"]
    listen_port = port or config.server.admin_api_port

    from memforge.server.admin_api import create_admin_app

    app = create_admin_app(config=config)
    click.echo(f"Starting Admin API on http://{host}:{listen_port}")
    uvicorn.run(app, host=host, port=listen_port, log_level="info")


# ---------------------------------------------------------------------------
# sources group
# ---------------------------------------------------------------------------


@cli.group()
def sources():
    """Manage configured data sources."""
    pass


@sources.command("list")
@_workspace_id_option
@click.pass_context
def sources_list(ctx, workspace_id: str | None):
    """List all configured sources for the active API target."""
    payload = _tool_client(ctx, workspace_id).list_sources()
    if ctx.obj.get("json") or payload.get("error"):
        _emit_tool_payload(ctx, payload)
        return

    src_list = payload.get("data") or []
    if not src_list:
        console.print("[dim]No sources configured.[/]")
        return

    table = Table(title="Configured Sources")
    table.add_column("ID", style="dim", max_width=18)
    table.add_column("Name")
    table.add_column("Type")
    table.add_column("Status")
    table.add_column("Docs", justify="right")
    table.add_column("Memories", justify="right")
    table.add_column("Last Sync")

    for src in src_list:
        table.add_row(
            src.get("source_id") or src.get("id", ""),
            src.get("name", ""),
            src.get("type", ""),
            src.get("status", ""),
            str(src.get("doc_count", 0)),
            str(src.get("memory_count", 0)),
            src.get("last_synced_at") or "never",
        )
    console.print(table)


@sources.command("searchable")
@_workspace_id_option
@click.pass_context
def sources_searchable(ctx, workspace_id: str | None):
    """List source IDs available for memory search filtering."""
    payload = _tool_client(ctx, workspace_id).list_searchable_sources()
    if ctx.obj.get("json") or payload.get("error"):
        _emit_tool_payload(ctx, payload)
        return

    src_list = payload.get("data") or []
    if not src_list:
        console.print("[dim]No searchable sources configured.[/]")
        return

    table = Table(title="Searchable Sources")
    table.add_column("Source ID", style="dim", max_width=18)
    table.add_column("Name")
    table.add_column("Type")
    table.add_column("Status")
    table.add_column("Docs", justify="right")
    table.add_column("Memories", justify="right")
    table.add_column("Last Sync")

    for src in src_list:
        table.add_row(
            src.get("source_id", ""),
            src.get("name", ""),
            src.get("type", ""),
            src.get("status", ""),
            str(src.get("doc_count", 0)),
            str(src.get("memory_count", 0)),
            src.get("last_synced_at") or "never",
        )
    console.print(table)


@sources.command("schedule")
@click.argument("source_id")
@click.option("--every-minutes", type=int, default=1440, show_default=True, help="Automatic sync interval.")
@click.option("--disable", is_flag=True, help="Disable automatic sync for this source.")
@_workspace_id_option
@click.pass_context
def sources_schedule(ctx, source_id: str, every_minutes: int, disable: bool, workspace_id: str | None):
    """Configure automatic sync for one source over the active API target."""
    if every_minutes < SOURCE_SYNC_SCHEDULE_MIN_INTERVAL_MINUTES:
        raise click.ClickException(f"--every-minutes must be at least {SOURCE_SYNC_SCHEDULE_MIN_INTERVAL_MINUTES}.")
    if every_minutes > SOURCE_SYNC_SCHEDULE_MAX_INTERVAL_MINUTES:
        raise click.ClickException(f"--every-minutes must be at most {SOURCE_SYNC_SCHEDULE_MAX_INTERVAL_MINUTES}.")
    client = _tool_client(ctx, workspace_id)
    payload = client.update_source_schedule(
        source_id=source_id,
        enabled=not disable,
        interval_minutes=every_minutes,
    )
    _emit_tool_payload(ctx, payload)


@sources.command("schedule-show")
@click.argument("source_id")
@_workspace_id_option
@click.pass_context
def sources_schedule_show(ctx, source_id: str, workspace_id: str | None):
    """Show automatic sync schedule for one source over the active API target."""
    payload = _tool_client(ctx, workspace_id).get_source_schedule(source_id)
    _emit_tool_payload(ctx, payload)


@sources.command("stop")
@click.argument("source_id")
@_workspace_id_option
@click.pass_context
def sources_stop(ctx, source_id: str, workspace_id: str | None):
    """Stop the sync in progress of one source over the active API target.

    The source stays active and later scheduled syncs run. A running server
    sync stops at its worker's next heartbeat, within about 30 seconds; what it
    has committed stays. To also keep the source from syncing, pause it.
    """
    payload = _tool_client(ctx, workspace_id).stop_source_sync(source_id=source_id)
    _emit_tool_payload(ctx, payload)


@sources.command("reprocess")
@click.argument("source_id")
@click.option(
    "--document",
    "document_ids",
    multiple=True,
    help="Document to reprocess at its current revision (repeatable).",
)
@click.option("--all", "all_documents", is_flag=True, help="Reprocess every current Document of the source.")
@click.option("--dry-run", is_flag=True, help="Report what would be read and the model calls, without writing.")
@_workspace_id_option
@click.pass_context
def sources_reprocess(
    ctx,
    source_id: str,
    document_ids: tuple[str, ...],
    all_documents: bool,
    dry_run: bool,
    workspace_id: str | None,
):
    """Reprocess Documents of one source with the current adapter and compiler.

    Name the Documents with --document, or take every current one with --all.
    Reads each Document's current state from the provider when the source can
    ask for one Document by id, and its stored input otherwise: every
    ReadingGroup is extracted and every Support is read over the whole Unit.
    """
    if bool(document_ids) == all_documents:
        raise click.UsageError("Name the Documents with --document, or pass --all.")
    payload = _tool_client(ctx, workspace_id).reprocess_source_documents(
        source_id=source_id,
        document_ids=list(document_ids),
        all_documents=all_documents,
        dry_run=dry_run,
    )
    _emit_tool_payload(ctx, payload)


# ---------------------------------------------------------------------------
# memories group
# ---------------------------------------------------------------------------


@cli.group()
def memories():
    """Browse and inspect memories."""
    pass


@memories.command("list")
@click.option(
    "--type", "memory_type", default=None, help="Filter by memory type (fact, decision, convention, procedure)"
)
@click.option("--entity", default=None, help="Filter by entity name")
@click.option("--source", default=None, help="Filter by source name/ID")
@click.option("--status", default="active", help="Filter by status (default: active)")
@click.option("--limit", "-n", default=20, type=int, help="Max results (default: 20)")
@click.pass_context
def memories_list(ctx, memory_type: str | None, entity: str | None, source: str | None, status: str | None, limit: int):
    """List memories with optional filters."""

    async def _run():
        config: AppConfig = ctx.obj["config"]
        db = await _get_db(config)

        # If entity filter, look up entity and get memories by entity
        if entity:
            from memforge.models import canonicalize_entity_name

            canonical = canonicalize_entity_name(entity)
            ent = await db.get_entity_by_canonical(canonical)
            if not ent:
                # Try alias
                alias = await db.get_entity_by_alias(canonical)
                if alias:
                    ent_obj = await db.get_entity_by_canonical(canonical)
                    if not ent_obj:
                        console.print(f"[yellow]Entity '{entity}' not found.[/]")
                        await db.close()
                        return
                    ent = ent_obj
                else:
                    console.print(f"[yellow]Entity '{entity}' not found.[/]")
                    await db.close()
                    return
            mems = await db.get_memories_by_entity(ent.id)
            # Apply client-side filters
            if memory_type:
                mems = [m for m in mems if m.memory_type == memory_type]
            if status:
                mems = [m for m in mems if m.status == status]
            mems = mems[:limit]
        else:
            mems = await db.list_memories(
                type=memory_type,
                status=status,
                source=source,
                limit=limit,
            )

        await db.close()

        if not mems:
            console.print("[dim]No memories found.[/]")
            return

        table = Table(title=f"Memories ({len(mems)} results)")
        table.add_column("ID", style="dim", max_width=12)
        table.add_column("Type", max_width=12)
        table.add_column("Content", max_width=60)
        table.add_column("Corr.", justify="right", max_width=5)
        table.add_column("Status", max_width=10)
        table.add_column("Updated", max_width=20)

        for m in mems:
            content_preview = m.content[:80] + "..." if len(m.content) > 80 else m.content
            table.add_row(
                m.id,
                m.memory_type,
                content_preview,
                str(m.corroboration_count),
                m.status,
                m.updated_at.isoformat()[:19] if m.updated_at else "",
            )
        console.print(table)

    asyncio.run(_run())


@memories.command("stats")
@click.pass_context
def memories_stats(ctx):
    """Show memory counts by type, status, and source."""

    async def _run():
        config: AppConfig = ctx.obj["config"]
        db = await _get_db(config)

        from memforge.models import MemoryType

        console.print("\n[bold]Memory Statistics[/]\n")

        # By type
        type_table = Table(title="By Type")
        type_table.add_column("Type")
        type_table.add_column("Count", justify="right")
        total = 0
        for mt in MemoryType:
            count = await db.count_memories(type=mt.value)
            type_table.add_row(mt.value, str(count))
            total += count
        type_table.add_row("[bold]Total[/]", f"[bold]{total}[/]")
        console.print(type_table)

        console.print()

        # By status
        status_table = Table(title="By Status")
        status_table.add_column("Status")
        status_table.add_column("Count", justify="right")
        for status in ("active", "superseded", "retired", "pending_review"):
            count = await db.count_memories(status=status)
            status_table.add_row(status, str(count))
        console.print(status_table)

        console.print()

        # By source
        src_list = await db.list_sources()
        if src_list:
            source_table = Table(title="By Source")
            source_table.add_column("Source")
            source_table.add_column("Type")
            source_table.add_column("Documents", justify="right")
            for src in src_list:
                source_table.add_row(
                    src["name"],
                    src["type"],
                    str(src.get("doc_count", 0)),
                )
            console.print(source_table)

        await db.close()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# adapter group
# ---------------------------------------------------------------------------


@cli.group("adapter")
def adapter():
    """Run local adapter operations for sources that need local capabilities."""
    pass


@adapter.command("list")
@click.pass_context
def adapter_list(ctx):
    """List local adapter capabilities."""
    _emit_tool_payload(
        ctx,
        {
            "data": [
                {"type": "jira", "auth": "browser_session"},
                {"type": "kb", "kind": "markdown"},
                {"type": "github", "kind": "repository"},
                {"type": "teams", "auth": "browser_session", "kind": "conversation"},
            ]
        },
    )


@adapter.command("status")
@click.pass_context
def adapter_status(ctx):
    """Show local adapter status."""
    _emit_tool_payload(
        ctx,
        {
            "status": "available",
            "capabilities": [
                "jira.browser_session",
                "kb.markdown_preview",
                "kb.markdown_push",
                "github.repo_preview",
                "github.repo_push",
                "teams.auth",
                "teams.browse",
                "teams.sync",
            ],
        },
    )


@adapter.group("daemon")
def adapter_daemon():
    """Run the MemForge local agent daemon."""
    pass


@adapter_daemon.command("status")
@click.option("--verbose", is_flag=True, help="Include the raw local daemon state file.")
@click.pass_context
def adapter_daemon_status(ctx, verbose: bool):
    """Show local agent daemon state."""
    _emit_tool_payload(ctx, _local_agent_status_payload(ctx, verbose=verbose))


def _local_agent_status_payload(ctx, *, verbose: bool) -> dict[str, Any]:
    """Build the process-level daemon status shared by legacy and service CLI surfaces."""
    state = _local_agent_state_store().load()
    state_tasks = state.get("tasks") if isinstance(state.get("tasks"), dict) else {}
    daemon_status = _local_agent_daemon_status(state)
    target_summary = _local_agent_status_target(
        state,
        _local_agent_target_summary(ctx),
        daemon_running=daemon_status["status"] == "running",
    )
    payload = {
        "status": daemon_status["status"],
        "state_path": str(_local_agent_state_path()),
        "lock_path": str(_local_agent_lock_path()),
        "target": target_summary,
        "daemon": daemon_status["daemon"],
        "summary": _summarize_local_agent_state_tasks(state_tasks),
        "recent_tasks": _recent_local_agent_tasks(state_tasks, limit=5),
    }
    recommendations = _local_agent_status_recommendations(target_summary)
    if recommendations:
        payload["recommendations"] = recommendations
    if verbose:
        payload["state"] = state
    return payload


def _local_agent_daemon_status(state: dict[str, Any]) -> dict[str, Any]:
    lock_payload = _read_local_agent_lock()
    lock_pid = lock_payload.get("pid") if isinstance(lock_payload, dict) else None
    lock_held = _pid_is_running(lock_pid)
    daemon = state.get("daemon") if isinstance(state.get("daemon"), dict) else {}
    daemon_pid = daemon.get("pid") if isinstance(daemon, dict) else None
    heartbeat_alive = _pid_is_running(daemon_pid)
    running = lock_held or heartbeat_alive
    status_payload: dict[str, Any] = {
        "lock_held": lock_held,
        "lock_pid": lock_pid if lock_held else None,
        "pid": daemon_pid,
        "started_at": daemon.get("started_at") if isinstance(daemon, dict) else None,
        "updated_at": daemon.get("updated_at") if isinstance(daemon, dict) else None,
        "command": daemon.get("command") if isinstance(daemon, dict) else None,
    }
    return {"status": "running" if running else "stopped", "daemon": status_payload}


def _daemon_recorded_target(state: dict[str, Any]) -> dict[str, Any] | None:
    daemon = state.get("daemon") if isinstance(state.get("daemon"), dict) else {}
    target = daemon.get("target") if isinstance(daemon, dict) else None
    if not isinstance(target, dict):
        return None
    return _local_agent_clean_target_summary(target)


def _local_agent_clean_target_summary(target: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in target.items() if key != "workspace_id_configured"}


def _local_agent_target_summary(ctx) -> dict[str, Any]:
    config: AppConfig = ctx.obj["config"]
    resolved = _resolve_api_target(config)
    return {
        "edition": resolved.target.edition.value,
        "api_url": resolved.target.origin,
        "active_target": resolved.active_target,
        "token_env": resolved.token_env,
        "api_token_configured": bool(resolved.api_token),
    }


def _local_agent_status_recommendations(target: dict[str, Any]) -> list[str]:
    recommendations: list[str] = []
    token_env = str(target.get("token_env") or "MEMFORGE_API_TOKEN")
    if target.get("edition") == "cloud" and not target.get("api_token_configured"):
        recommendations.append(f"Set {token_env} before starting the daemon.")
    return recommendations


def _local_agent_status_target(
    state: dict[str, Any],
    current_target: dict[str, Any],
    *,
    daemon_running: bool,
) -> dict[str, Any]:
    recorded = _daemon_recorded_target(state)
    if recorded is None or not daemon_running:
        return current_target
    return {**recorded, "source": "running_daemon"}


def _summarize_local_agent_state_tasks(tasks: dict[str, Any]) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    last_cloud_job_lease: dict[str, Any] | None = None
    for task_id, task in tasks.items():
        if not isinstance(task, dict):
            continue
        status = str(task.get("last_status") or "unknown")
        statuses[status] = statuses.get(status, 0) + 1
        if task_id == "cloud-jobs:lease":
            last_cloud_job_lease = _compact_local_agent_task(task_id, task)
    return {
        "total_recorded_tasks": sum(statuses.values()),
        "statuses": statuses,
        "last_cloud_job_lease": last_cloud_job_lease,
    }


def _recent_local_agent_tasks(tasks: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    compact_tasks = [
        _compact_local_agent_task(task_id, task) for task_id, task in tasks.items() if isinstance(task, dict)
    ]
    compact_tasks.sort(
        key=lambda task: (
            _local_agent_task_timestamp(task.get("updated_at")),
            _local_agent_task_timestamp(task.get("last_finished_at")),
            str(task.get("task_id") or ""),
        ),
        reverse=True,
    )
    return compact_tasks[:limit]


def _local_agent_task_timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or not value.strip():
        return datetime.min.replace(tzinfo=timezone.utc)
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _compact_local_agent_task(task_id: str, task: dict[str, Any]) -> dict[str, Any]:
    error = task.get("last_error")
    compact = {
        "task_id": task_id,
        "status": task.get("last_status"),
        "last_finished_at": task.get("last_finished_at"),
        "updated_at": task.get("updated_at"),
        "run_count": task.get("run_count"),
        "error": str(error) if error else None,
    }
    last_result = task.get("last_result")
    payload = last_result.get("payload") if isinstance(last_result, dict) else None
    if isinstance(payload, dict) and payload:
        compact["payload"] = payload
    return compact


@adapter_daemon.command("once")
@click.option("--browser", default=None, help="Browser to read Jira cookies from, for example chrome or edge.")
@click.pass_context
def adapter_daemon_once(
    ctx,
    browser: str | None,
):
    """Lease and execute one batch of server-owned jobs, then exit."""
    runner = _build_local_agent_runner(
        ctx,
        browser=browser,
    )
    report = runner.run_once()
    if int(report.get("counts", {}).get("failed") or 0) > 0:
        report = dict(report)
        report["error"] = "one or more local agent tasks failed"
    _emit_tool_payload(ctx, report)


@adapter_daemon.command("run")
@click.option("--browser", default=None, help="Browser to read Jira cookies from, for example chrome or edge.")
@click.option(
    "--interval-seconds",
    "poll_interval_seconds",
    default=10,
    show_default=True,
    type=int,
    help="Seconds between server-job lease polls.",
)
@click.option(
    "--cloud-job-wait-seconds",
    default=25,
    show_default=True,
    type=int,
    help="Long-poll wait for Cloud-triggered local-agent jobs.",
)
@click.pass_context
def adapter_daemon_run(
    ctx,
    browser: str | None,
    poll_interval_seconds: int,
    cloud_job_wait_seconds: int,
):
    """Run the local agent daemon until interrupted."""
    _run_local_agent_daemon(
        ctx,
        browser=browser,
        poll_interval_seconds=poll_interval_seconds,
        cloud_job_wait_seconds=cloud_job_wait_seconds,
    )


def _run_local_agent_daemon(
    ctx,
    *,
    browser: str | None,
    poll_interval_seconds: int,
    cloud_job_wait_seconds: int,
) -> None:
    """Own the single daemon process loop for foreground and native-service entrypoints."""
    daemon_lock = _acquire_local_agent_daemon_lock()
    if daemon_lock is None:
        _emit_tool_payload(
            ctx,
            {
                "status": "already_running",
                "error": "local agent daemon is already running",
                "lock_path": str(_local_agent_lock_path()),
            },
        )
        return
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        _local_agent_state_store().record_daemon_heartbeat(
            pid=os.getpid(),
            started_at=started_at,
            command=sys.argv,
            target=_local_agent_target_summary(ctx),
        )
        runner = _build_local_agent_runner(
            ctx,
            browser=browser,
        )
        click.echo(
            json.dumps(
                {
                    "status": "running",
                    "state_path": str(_local_agent_state_path()),
                    "poll_interval_seconds": max(int(poll_interval_seconds), 1),
                    "cloud_job_wait_seconds": max(int(cloud_job_wait_seconds), 0),
                },
                ensure_ascii=False,
            )
        )
        runner.run_forever(
            poll_interval_seconds=poll_interval_seconds,
            cloud_job_wait_seconds=cloud_job_wait_seconds,
            log=lambda message: click.echo(message, err=True),
        )
    except KeyboardInterrupt:
        click.echo(json.dumps({"status": "stopped"}, ensure_ascii=False))
    finally:
        daemon_lock.close()


@dataclass(frozen=True)
class _GitHubTransferResult:
    """Body-free receipt and byte accounting for one document transfer."""

    receipt: dict[str, Any]
    prepared_bytes: int | None = None
    submitted_bytes: int | None = None


def _push_github_source_entry(
    entry: dict[str, Any],
    *,
    repo: dict[str, str],
    ref: str,
    commit_sha: str,
    source_id: str,
    client: ToolClient,
    sync_snapshot_id: str | None,
    local_agent_job_id: str | None,
    local_agent_attempt_count: int | None,
    submitted_by: str | None,
    on_body_ready: Callable[[], None],
) -> _GitHubTransferResult:
    """Own a single body's lifetime through validation and synchronous upload."""

    relative_path = str(entry["relative_path"])
    content_type = str(entry.get("content_type") or "text/plain")
    blob_sha = str(entry.get("blob_sha") or "")
    is_source_artifact = content_type in SUPPORTED_SOURCE_ARTIFACT_MEDIA_TYPES
    try:
        if github_content_type_is_binary(content_type) and not is_source_artifact:
            raise click.ClickException(f"unsupported binary media type: {content_type}")
        if is_source_artifact:
            declared_size = entry.get("bytes")
            if not isinstance(declared_size, int) or isinstance(declared_size, bool) or declared_size < 0:
                raise click.ClickException("GitHub tree did not return a valid blob size")
            if declared_size > MAX_SOURCE_ARTIFACT_STORAGE_BYTES:
                raise click.ClickException("Source Artifact exceeds storage limit")
        raw = (
            _github_blob(repo, blob_sha, relative_path)
            if is_source_artifact
            else _github_text_blob(repo, blob_sha, relative_path, size=entry.get("bytes"))
        )
        if is_source_artifact and len(raw) != declared_size:
            raise click.ClickException("GitHub blob size changed after tree discovery")
        text_body = "" if is_source_artifact else decode_github_text(raw, label=relative_path)
        last_commit_at = _github_last_commit_at(
            repo,
            commit_sha=commit_sha,
            relative_path=relative_path,
            resolved_relative_path=entry.get("resolved_relative_path"),
        )
    except ValueError as exc:
        return _GitHubTransferResult({"relative_path": relative_path, "error": str(exc)})
    except click.ClickException as exc:
        return _GitHubTransferResult({"relative_path": relative_path, "error": str(exc)})

    body_bytes = len(raw)
    raw_hash = hashlib.sha256(raw).hexdigest()
    on_body_ready()
    artifact_input_hashes: list[str] = []
    if is_source_artifact:
        artifact_response = client.push_source_artifact(
            source_id=source_id,
            source_unit_key=relative_path,
            provider_key=relative_path,
            provider_revision=blob_sha,
            parent_observation_type="file_content",
            parent_provider_key="content",
            filename=Path(relative_path).name,
            media_type=content_type,
            content=raw,
            local_agent_job_id=local_agent_job_id,
            local_agent_attempt_count=local_agent_attempt_count,
        )
        _raise_if_local_agent_lease_not_current(artifact_response)
        input_hash = str(artifact_response.get("input_sha256") or "").strip()
        if artifact_response.get("error") or len(input_hash) != 64:
            return _GitHubTransferResult(
                {
                    "relative_path": relative_path,
                    "error": artifact_response.get("error") or "Source Artifact upload failed",
                    "detail": artifact_response.get("detail"),
                    "status_code": artifact_response.get("status_code"),
                },
                prepared_bytes=body_bytes,
            )
        artifact_input_hashes.append(input_hash)
    response = client.push_github_repo_document(
        source_id=source_id,
        repo_url=repo["repo_url"],
        repo_ref=ref,
        relative_path=relative_path,
        markdown_body=text_body,
        content_type=content_type,
        title=relative_path if is_source_artifact else _github_title(text_body, relative_path),
        raw_hash=raw_hash,
        blob_sha=blob_sha,
        symlink_chain=(
            [dict(item) for item in entry.get("symlink_chain", [])]
            if isinstance(entry.get("symlink_chain"), list)
            else None
        ),
        resolved_relative_path=(
            str(entry.get("resolved_relative_path") or "").strip() or None
        ),
        source_updated_at=last_commit_at,
        sync_snapshot_id=sync_snapshot_id,
        local_agent_job_id=local_agent_job_id,
        local_agent_attempt_count=local_agent_attempt_count,
        artifact_source_unit_key=relative_path if artifact_input_hashes else None,
        artifact_input_hashes=artifact_input_hashes,
        submitted_by=submitted_by,
    )
    _raise_if_local_agent_lease_not_current(response)
    if response.get("error"):
        receipt = {
            "relative_path": relative_path,
            "error": response["error"],
            "detail": response.get("detail"),
            "status_code": response.get("status_code"),
        }
    else:
        receipt = {
            "relative_path": relative_path,
            "doc_id": response.get("doc_id"),
            "document_hash": response.get("document_hash"),
        }
    return _GitHubTransferResult(receipt, prepared_bytes=body_bytes, submitted_bytes=body_bytes)


def _push_github_profile_to_source(
    name: str,
    profile: dict[str, Any],
    *,
    source_id: str,
    limit: int,
    force_full_sync: bool,
    submitted_by: str | None,
    client: ToolClient,
    sync_snapshot_id: str | None = None,
    local_agent_job_id: str | None = None,
    local_agent_attempt_count: int | None = None,
    report_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    source_id = source_id.strip()
    if not source_id:
        raise click.ClickException("source_id is required")
    if limit and sync_snapshot_id:
        raise ValueError("limited collection cannot finalize a fenced source snapshot")

    repo, ref, _include_paths, _exclude_paths, _include_extensions = _resolve_github_profile(name, profile)
    preview = _preview_github_profile(name, profile, limit=None if limit == 0 else max(limit, 0))
    selected_entries = list(preview["items"])
    _report_local_agent_progress(
        report_progress,
        _sync_progress_snapshot(
            phase="discovering",
            completed=len(selected_entries),
            unit="file",
        ),
    )

    manifest_items: list[dict[str, str]] = []
    entries_by_doc_id: dict[str, dict[str, Any]] = {}
    revisions_complete = True
    for entry in selected_entries:
        relative_path = str(entry["relative_path"])
        try:
            validate_github_file_mode(entry.get("file_mode"), label=relative_path)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        revision = str(entry.get("blob_sha") or "").strip()
        if not revision:
            revisions_complete = False
        doc_id = build_github_repo_doc_id(
            source_id=source_id,
            repo_url=repo["repo_url"],
            repo_ref=ref,
            relative_path=relative_path,
        )
        manifest_items.append({"doc_id": doc_id, "revision": revision, "change_kind": "upsert"})
        entries_by_doc_id[doc_id] = entry

    required_doc_ids = set(entries_by_doc_id)
    comparison_started_at = time.perf_counter()
    fallback_reason: str | None = None
    if (
        not limit
        and revisions_complete
        and sync_snapshot_id
        and local_agent_job_id
        and local_agent_attempt_count is not None
    ):
        try:
            required_doc_ids = _required_local_source_doc_ids(
                client,
                source_id=source_id,
                items=manifest_items,
                coverage="complete_snapshot",
                known_doc_ids=set(entries_by_doc_id),
                sync_snapshot_id=sync_snapshot_id,
                local_agent_job_id=local_agent_job_id,
                local_agent_attempt_count=local_agent_attempt_count,
            )
        except RuntimeError as exc:
            return {
                "profile": name,
                "repo_url": repo["repo_url"],
                "ref": ref,
                "source_id": source_id,
                "counts": {
                    "selected": len(selected_entries),
                    "reused": 0,
                    "fetched": 0,
                    "pushed": 0,
                    "failed": 0,
                },
                "error": str(exc),
                "retryable": True,
            }
    elif not revisions_complete:
        fallback_reason = "missing_provider_revision"
    comparison_latency_ms = round(
        (time.perf_counter() - comparison_started_at) * 1000,
        3,
    )

    entries_to_fetch = [entry for doc_id, entry in entries_by_doc_id.items() if doc_id in required_doc_ids]
    pushed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    fetched_count = submitted_count = 0
    read_body_bytes = uploaded_body_bytes = 0

    def report_transfer_phase(phase: str, completed: int) -> None:
        _report_local_agent_progress(
            report_progress,
            _sync_progress_snapshot(
                phase=phase,
                completed=completed,
                total=len(entries_to_fetch),
                unit="file",
                failed=len(failed),
            ),
        )

    report_transfer_phase("fetching", 0)
    for index, entry in enumerate(entries_to_fetch, start=1):
        if index > 1:
            report_transfer_phase("fetching", index - 1)

        def body_ready() -> None:
            report_transfer_phase("fetching", index)
            report_transfer_phase("uploading", index - 1)

        transfer = _push_github_source_entry(
            entry,
            repo=repo,
            ref=ref,
            commit_sha=str(preview["commit_sha"]),
            source_id=source_id,
            client=client,
            sync_snapshot_id=sync_snapshot_id,
            local_agent_job_id=local_agent_job_id,
            local_agent_attempt_count=local_agent_attempt_count,
            submitted_by=submitted_by,
            on_body_ready=body_ready,
        )
        if transfer.prepared_bytes is not None:
            fetched_count += 1
            read_body_bytes += transfer.prepared_bytes
        if transfer.submitted_bytes is not None:
            submitted_count += 1
            uploaded_body_bytes += transfer.submitted_bytes
        if transfer.receipt.get("error"):
            failed.append(transfer.receipt)
        else:
            pushed.append(transfer.receipt)
        report_transfer_phase(
            "uploading" if transfer.prepared_bytes is not None else "fetching",
            index,
        )

    payload = {
        "profile": name,
        "repo_url": repo["repo_url"],
        "ref": ref,
        "source_id": source_id,
        "counts": {
            "selected": len(selected_entries),
            "reused": len(selected_entries) - len(entries_to_fetch),
            "fetched": fetched_count,
            "pushed": len(pushed),
            "failed": len(failed),
        },
        "pushed": pushed,
        "failed": failed,
        "metrics": {
            "manifest_items": len(manifest_items),
            "full_bodies_read": fetched_count,
            "full_bodies_uploaded": submitted_count,
            "bytes_read": read_body_bytes,
            "bytes_uploaded": uploaded_body_bytes,
            "comparison_latency_ms": comparison_latency_ms,
            "end_to_end_latency_ms": round(
                (time.perf_counter() - started_at) * 1000,
                3,
            ),
            "fallback_reason": fallback_reason,
        },
    }
    if failed:
        examples = "; ".join(f"{item.get('relative_path')}: {item.get('error')}" for item in failed[:3])
        payload["error"] = f"{len(failed)} document(s) failed to push" + (f": {examples}" if examples else "")
    if not failed:
        sync_result = client.start_source_processing(
            source_id=source_id,
            force_full_sync=force_full_sync,
            sync_snapshot_id=sync_snapshot_id,
            local_agent_job_id=local_agent_job_id,
            local_agent_attempt_count=local_agent_attempt_count,
        )
        payload["sync_started"] = not bool(sync_result.get("error"))
        payload.update(source_processing_receipt(sync_result))
        if sync_result.get("error"):
            payload["error"] = "source processing failed to start"
            payload["sync_error"] = sync_result
    if payload.get("error"):
        payload["retryable"] = True
    return payload


def _run_cloud_local_agent_job(
    job: dict[str, Any],
    client: ToolClient,
    *,
    browser: str | None = None,
    report_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    operation = str(job.get("operation") or "").strip()
    handlers = {
        "github_repo_preview_tree": lambda: _run_cloud_github_preview_job(job),
        "github_repo_sync": lambda: _run_cloud_github_sync_job(job, client, report_progress=report_progress),
        "local_markdown_pick_root": lambda: _run_cloud_pick_root_job(job),
        "local_markdown_preview_tree": lambda: _run_cloud_local_markdown_preview_job(job),
        "local_markdown_sync": lambda: _run_cloud_local_markdown_sync_job(job, client, report_progress=report_progress),
        "jira_auth": lambda: _run_cloud_jira_auth_job(job, client, browser=browser),
        "jira_sync": lambda: _run_cloud_jira_sync_job(job, client, browser=browser, report_progress=report_progress),
        "teams_auth_check": lambda: _run_cloud_teams_auth_check_job(job),
        "teams_auth": lambda: _run_cloud_teams_auth_job(job),
        "teams_browse": lambda: _run_cloud_teams_browse_job(job),
        "teams_sync": lambda: _run_cloud_teams_sync_job(
            job,
            client,
            report_progress=report_progress,
        ),
    }
    handler = handlers.get(operation)
    if handler is None:
        return {
            "operation": operation,
            "error": f"unsupported cloud local-agent job operation: {operation or '<empty>'}",
        }
    return handler()


def _run_cloud_jira_auth_job(
    job: dict[str, Any],
    client: ToolClient,
    *,
    browser: str | None,
) -> dict[str, Any]:
    operation = str(job.get("operation") or "").strip()
    payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
    source_id, client, error = _local_agent_job_source_client(
        job,
        payload,
        operation=operation,
        client=client,
    )
    if error:
        return {**error, "retryable": False}

    try:
        from memforge.auth.jira_capture import capture_and_prevalidate

        session_config = _jira_cloud_config_from_job_payload(payload)
        captured = asyncio.run(
            capture_and_prevalidate(
                str(session_config["base_url"]),
                browser=browser,
                tls_config=session_config,
                interactive=True,
            )
        )
        uploaded_session = client.upload_jira_session(
            base_url=captured.origin,
            cookie_header=captured.cookie_header,
            browser=captured.browser,
        )
        if _is_jira_principal_change(uploaded_session):
            return {
                "operation": operation,
                "source_id": source_id,
                **_principal_change_payload(uploaded_session),
                "error_type": "JiraPrincipalChangedError",
                "retryable": False,
            }
        if uploaded_session.get("error"):
            return {
                "operation": operation,
                "source_id": source_id,
                "error": "Unable to store the renewed Jira browser session",
                "error_type": "JiraSessionUploadError",
                "retryable": False,
            }
    except Exception as exc:
        return {
            "operation": operation,
            "source_id": source_id,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "retryable": False,
        }

    return {
        "operation": operation,
        "source_id": source_id,
        "authenticated": True,
        "origin": captured.origin,
    }


def _local_agent_lease_not_current(response: object) -> bool:
    return (
        isinstance(response, dict)
        and response.get("status_code") == HTTPStatus.CONFLICT
        and response.get("detail") == LOCAL_AGENT_LEASE_NOT_CURRENT
    )


def _raise_if_local_agent_lease_not_current(response: object) -> None:
    if not _local_agent_lease_not_current(response):
        return
    from memforge.local_agent.runner import CloudJobLeaseLost

    raise CloudJobLeaseLost(LOCAL_AGENT_LEASE_NOT_CURRENT)


def _required_local_source_doc_ids(
    client: ToolClient,
    *,
    source_id: str,
    items: list[dict[str, str]],
    coverage: str,
    known_doc_ids: set[str],
    sync_snapshot_id: str,
    local_agent_job_id: str,
    local_agent_attempt_count: int,
) -> set[str]:
    """Plan one fenced collection and validate the returned materialization set."""
    plan = client.prepare_local_source_snapshot(
        source_id=source_id,
        items=items,
        coverage=coverage,
        sync_snapshot_id=sync_snapshot_id,
        local_agent_job_id=local_agent_job_id,
        local_agent_attempt_count=local_agent_attempt_count,
    )
    _raise_if_local_agent_lease_not_current(plan)
    if plan.get("error"):
        raise RuntimeError("source manifest comparison failed")
    required_doc_ids = {
        str(doc_id)
        for doc_id in plan.get("required_doc_ids", [])
        if str(doc_id)
    }
    if required_doc_ids - known_doc_ids:
        raise RuntimeError("source manifest comparison returned unknown document identities")
    return required_doc_ids


def _run_cloud_github_preview_job(job: dict[str, Any]) -> dict[str, Any]:
    operation = str(job.get("operation") or "").strip()
    payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
    profile = _github_profile_from_cloud_job(job)
    source_id = str(job.get("source_id") or payload.get("source_id") or "").strip()
    profile_name = f"cloud-job:{job.get('job_id') or operation or 'unknown'}"

    limit = _cloud_job_limit(payload.get("limit"), default=200)
    tree_profile = {**profile, "include_paths": [], "exclude_paths": []}
    preview = _preview_github_profile(profile_name, tree_profile, limit=limit)
    return {"operation": operation, "source_id": source_id, **preview}


def _run_cloud_github_sync_job(
    job: dict[str, Any],
    client: ToolClient,
    *,
    report_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    operation = str(job.get("operation") or "").strip()
    payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
    profile = _github_profile_from_cloud_job(job)
    source_id, client, error = _local_agent_job_source_client(
        job,
        payload,
        operation=operation,
        client=client,
    )
    if error:
        return error
    profile_name = f"cloud-job:{job.get('job_id') or operation or 'unknown'}"
    result = _push_github_profile_to_source(
        profile_name,
        profile,
        source_id=source_id,
        limit=0,
        force_full_sync=bool(payload.get("force_full_sync", False)),
        submitted_by=str(payload.get("submitted_by") or "memforge-local-agent"),
        client=client,
        sync_snapshot_id=local_agent_sync_snapshot_id(job.get("job_id"), job.get("attempt_count")),
        local_agent_job_id=str(job["job_id"]),
        local_agent_attempt_count=int(job["attempt_count"]),
        report_progress=report_progress,
    )
    return {"operation": operation, **result}


def _run_cloud_local_markdown_preview_job(job: dict[str, Any]) -> dict[str, Any]:
    operation = str(job.get("operation") or "").strip()
    payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
    source_id = str(job.get("source_id") or payload.get("source_id") or "").strip()
    profile_name = f"cloud-job:{job.get('job_id') or operation or 'unknown'}"
    profile = _kb_profile_from_cloud_job(job)
    preview = _preview_kb_profile(profile_name, profile, limit=_cloud_job_limit(payload.get("limit"), default=200))
    return {"operation": operation, "source_id": source_id, **preview}


def _run_cloud_pick_root_job(job: dict[str, Any]) -> dict[str, Any]:
    operation = str(job.get("operation") or "").strip()
    payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
    try:
        root = pick_folder(
            title=str(payload.get("title") or "Choose folder to sync"),
            initial_directory=str(payload.get("initial_directory") or "").strip() or None,
        )
    except FolderPickerCancelled:
        return {"operation": operation, "cancelled": True}
    except FolderPickerUnavailable as exc:
        return {"operation": operation, "error": str(exc)}
    return {"operation": operation, "root": root}


def _run_cloud_local_markdown_sync_job(
    job: dict[str, Any],
    client: ToolClient,
    *,
    report_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    operation = str(job.get("operation") or "").strip()
    payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
    source_id, client, error = _local_agent_job_source_client(
        job,
        payload,
        operation=operation,
        client=client,
    )
    if error:
        return error
    profile_name = f"cloud-job:{job.get('job_id') or operation or 'unknown'}"
    result = _push_kb_profile_to_source(
        profile_name,
        _kb_profile_from_cloud_job(job),
        source_id=source_id,
        limit=0,
        force_full_sync=bool(payload.get("force_full_sync", False)),
        submitted_by=str(payload.get("submitted_by") or "memforge-local-agent"),
        client=client,
        sync_snapshot_id=local_agent_sync_snapshot_id(job.get("job_id"), job.get("attempt_count")),
        local_agent_job_id=str(job["job_id"]),
        local_agent_attempt_count=int(job["attempt_count"]),
        report_progress=report_progress,
    )
    return {"operation": operation, **result}


def _run_cloud_jira_sync_job(
    job: dict[str, Any],
    client: ToolClient,
    *,
    browser: str | None,
    report_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    operation = str(job.get("operation") or "").strip()
    payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
    source_id, client, error = _local_agent_job_source_client(
        job,
        payload,
        operation=operation,
        client=client,
    )
    if error:
        return error
    _report_local_agent_progress(
        report_progress,
        _sync_progress_snapshot(phase="connecting"),
    )
    try:
        from memforge.auth.jira_capture import capture_and_prevalidate

        session_config = _jira_cloud_config_from_job_payload(payload)
        base_url = str(session_config["base_url"])
        captured = asyncio.run(
            capture_and_prevalidate(
                base_url,
                browser=browser,
                tls_config=session_config,
            )
        )
        uploaded_session = client.upload_jira_session(
            base_url=captured.origin,
            cookie_header=captured.cookie_header,
            browser=captured.browser,
        )
        if _is_jira_principal_change(uploaded_session):
            principal_change = _principal_change_payload(uploaded_session)
            return {
                "operation": operation,
                "source_id": source_id,
                **principal_change,
                "error_type": "JiraPrincipalChangedError",
                "retryable": False,
            }
        if uploaded_session.get("error"):
            return {
                "operation": operation,
                "source_id": source_id,
                "error": "Unable to store the renewed Jira browser session",
                "error_type": "JiraSessionUploadError",
                "retryable": True,
            }
        sync_snapshot_id = local_agent_sync_snapshot_id(job.get("job_id"), job.get("attempt_count"))
        collection = asyncio.run(
            _collect_jira_documents_from_cloud_job(
                job,
                source_id=source_id,
                jira_cookie=captured.cookie_header,
                client=client,
                sync_snapshot_id=sync_snapshot_id,
                local_agent_job_id=str(job["job_id"]),
                local_agent_attempt_count=int(job["attempt_count"]),
                limit=0,
                report_progress=report_progress,
            )
        )
    except Exception as exc:
        from memforge.auth.jira_auth import JiraAuthSessionMissingError

        return {
            "operation": operation,
            "source_id": source_id,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "retryable": not isinstance(exc, JiraAuthSessionMissingError),
        }

    pushed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    scoped_client = client
    documents = collection["documents"]
    sync_result: dict[str, Any] | None = None
    _report_local_agent_progress(
        report_progress,
        _sync_progress_snapshot(
            phase="uploading",
            completed=0,
            total=len(documents),
            unit="issue",
        ),
    )
    for index, doc in enumerate(documents, start=1):
        artifact_input_hashes = [str(value) for value in doc.get("artifact_input_hashes", []) if str(value).strip()]
        response = scoped_client.push_jira_package(
            source_id=source_id,
            base_url=str(doc["base_url"]),
            issue_key=str(doc["issue_key"]),
            source_url=str(doc["source_url"]),
            raw_payload=doc.get("raw_payload") if isinstance(doc.get("raw_payload"), dict) else {},
            title=str(doc["title"]),
            raw_hash=str(doc["raw_hash"]),
            provider_revision=str(doc["provider_revision"]),
            submitted_by=str(payload.get("submitted_by") or "memforge-local-agent"),
            sync_snapshot_id=sync_snapshot_id,
            local_agent_job_id=str(job["job_id"]),
            local_agent_attempt_count=int(job["attempt_count"]),
            artifact_source_unit_key=(str(doc["issue_key"]) if artifact_input_hashes else None),
            artifact_input_hashes=artifact_input_hashes,
        )
        _raise_if_local_agent_lease_not_current(response)
        if isinstance(response, dict) and response.get("error"):
            failed.append(
                {
                    "issue_key": doc["issue_key"],
                    "error": response.get("error"),
                    "detail": response.get("detail"),
                    "status_code": response.get("status_code"),
                }
            )
        else:
            pushed.append(
                {
                    "issue_key": doc["issue_key"],
                    "doc_id": response.get("doc_id"),
                    "document_hash": response.get("document_hash"),
                }
            )
        _report_local_agent_progress(
            report_progress,
            _sync_progress_snapshot(
                phase="uploading",
                completed=index,
                total=len(documents),
                unit="issue",
                failed=len(failed),
            ),
        )
    if not failed:
        sync_result = scoped_client.start_source_processing(
            source_id=source_id,
            force_full_sync=bool(payload.get("force_full_sync", False)),
            sync_snapshot_id=sync_snapshot_id,
            local_agent_job_id=str(job["job_id"]),
            local_agent_attempt_count=int(job["attempt_count"]),
        )
    result = {
        "operation": operation,
        "source_id": source_id,
        "base_url": str(payload.get("base_url") or ""),
        "counts": {
            "selected": int(collection["inventory_count"]),
            "reused": int(collection["reused_count"]),
            "pushed": len(pushed),
            "failed": len(failed),
        },
        "pushed": pushed,
        "failed": failed,
        "sync_started": bool(sync_result and not sync_result.get("error")),
        **(source_processing_receipt(sync_result) if sync_result else {}),
    }
    sync_failed = isinstance(sync_result, dict) and bool(sync_result.get("error"))
    if sync_failed:
        result["sync_error"] = sync_result
    if failed and sync_failed:
        result["error"] = "one or more documents failed to push; source sync failed to start"
    elif sync_failed:
        result["error"] = "source sync failed to start"
    elif failed:
        result["error"] = "one or more documents failed to push"
    if result.get("error"):
        result["retryable"] = True
    return result


def _run_cloud_teams_auth_job(job: dict[str, Any]) -> dict[str, Any]:
    from memforge.auth import teams_auth

    operation = str(job.get("operation") or "").strip()
    payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
    region = _teams_region_from_payload(payload)
    try:
        token_data = teams_auth.TeamsAuthenticator().authenticate(
            region=region,
            wait_seconds=_cloud_job_limit(payload.get("wait_seconds"), default=90),
            poll_interval_seconds=2.0,
            rejected_token_hashes=_teams_rejected_token_hashes_from_payload(payload),
        )
    except RuntimeError as exc:
        return {
            "operation": operation,
            "authenticated": False,
            "region": region,
            "error": str(exc),
        }
    except Exception as exc:
        return {
            "operation": operation,
            "authenticated": False,
            "region": region,
            "error": str(exc),
            "error_type": type(exc).__name__,
        }

    tokens = token_data.get("tokens") if isinstance(token_data, dict) else {}
    return {
        "operation": operation,
        "authenticated": True,
        "region": region,
        "token_count": len(tokens) if isinstance(tokens, dict) else 0,
    }


def _run_cloud_teams_auth_check_job(job: dict[str, Any]) -> dict[str, Any]:
    from memforge.local_agent.teams_browse import teams_auth_status

    operation = str(job.get("operation") or "").strip()
    payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
    return {
        "operation": operation,
        "region": _teams_region_from_payload(payload),
        **teams_auth_status(),
    }


def _run_cloud_teams_browse_job(job: dict[str, Any]) -> dict[str, Any]:
    from memforge.local_agent.teams_browse import browse_teams_conversations

    operation = str(job.get("operation") or "").strip()
    payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
    region = _teams_region_from_payload(payload)
    try:
        data = asyncio.run(browse_teams_conversations(region=region))
    except Exception as exc:
        if _teams_auth_exception(exc):
            reauth = _run_cloud_teams_auth_job(
                {
                    **job,
                    "operation": "teams_auth",
                    "payload": {
                        **payload,
                        "wait_seconds": _cloud_job_limit(payload.get("wait_seconds"), default=90),
                        "rejected_token_hashes": sorted(_current_teams_chat_token_hashes()),
                    },
                }
            )
            if not reauth.get("error"):
                try:
                    data = asyncio.run(browse_teams_conversations(region=region))
                except Exception as retry_exc:
                    return {
                        "operation": operation,
                        "region": region,
                        "error": str(retry_exc),
                        "error_type": type(retry_exc).__name__,
                    }
            else:
                return {"operation": operation, "region": region, **reauth}
        else:
            return {"operation": operation, "region": region, "error": str(exc), "error_type": type(exc).__name__}
    return {"operation": operation, "region": region, **data}


def _teams_auth_exception(exc: Exception) -> bool:
    from memforge.genes.teams_client import AuthenticationError

    if isinstance(exc, AuthenticationError):
        return True
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "401",
            "authentication failed",
            "session expired",
            "no teams session found",
            "missing chat api token",
            "no active teams session",
        )
    )


def _current_teams_chat_token_hashes() -> set[str]:
    from memforge.auth.teams_auth import CHAT_API_AUDIENCE, TeamsAuthenticator

    try:
        tokens = TeamsAuthenticator.load_tokens()
    except Exception:
        return set()
    if not isinstance(tokens, dict):
        return set()
    token = TeamsAuthenticator.get_token_for_audience(tokens, CHAT_API_AUDIENCE)
    if not token:
        return set()
    return {hashlib.sha256(token.encode("utf-8")).hexdigest()}


def _teams_rejected_token_hashes_from_payload(payload: dict[str, Any]) -> set[str]:
    value = payload.get("rejected_token_hashes")
    if not isinstance(value, (list, tuple, set)):
        return set()
    return {str(item) for item in value if str(item).strip()}


def _teams_region_from_payload(payload: dict[str, Any]) -> str:
    region = str(payload.get("region") or "emea").strip().lower()
    return region if region in {"emea", "amer", "apac"} else "emea"


def _run_cloud_teams_sync_job(
    job: dict[str, Any],
    client: ToolClient,
    *,
    report_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Collect one job attempt of a Teams Source on this device and hand it over to the server.

    An expired Teams session is renewed once through the browser before the
    attempt fails. The shared device executor uploads the attempt and starts
    its processing, or for a dry run asks for its plan.
    """

    from memforge.local_agent.conversation_executor import ConversationUploadError

    operation = str(job.get("operation") or "").strip()
    payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
    source_id, client, error = _local_agent_job_source_client(job, payload, operation=operation, client=client)
    if error:
        return error
    _report_local_agent_progress(report_progress, _sync_progress_snapshot(phase="connecting"))
    try:
        try:
            result = asyncio.run(_collect_teams_on_device(job, payload, source_id, client, report_progress))
        except ConversationUploadError:
            # The server refused an upload; a new Teams session would not change its answer.
            raise
        except Exception as exc:
            if not _teams_auth_exception(exc):
                raise
            reauth = _run_cloud_teams_auth_job(
                {
                    **job,
                    "operation": "teams_auth",
                    "payload": {
                        **payload,
                        "wait_seconds": _cloud_job_limit(payload.get("wait_seconds"), default=90),
                        "rejected_token_hashes": sorted(_current_teams_chat_token_hashes()),
                    },
                }
            )
            if reauth.get("error"):
                raise RuntimeError(str(reauth.get("error") or "Teams authentication failed.")) from exc
            result = asyncio.run(_collect_teams_on_device(job, payload, source_id, client, report_progress))
    except ConversationUploadError as exc:
        _raise_if_local_agent_lease_not_current(exc.response)
        return {"operation": operation, "source_id": source_id, "error": str(exc), "retryable": True}
    except Exception as exc:
        return {
            "operation": operation,
            "source_id": source_id,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "retryable": True,
        }
    outcome: dict[str, Any] = {"operation": operation, "source_id": source_id, "counts": result.counts}
    if result.plan is not None:
        return {**outcome, "dry_run": True, "plan": result.plan}
    return {**outcome, "sync_started": True, **source_processing_receipt(result.processing)}


async def _collect_teams_on_device(
    job: dict[str, Any],
    payload: dict[str, Any],
    source_id: str,
    client: ToolClient,
    report_progress: Callable[[dict[str, Any]], None] | None,
):
    from memforge.genes.teams_client import TeamsChatClient
    from memforge.genes.teams_provider import TeamsConversationProvider
    from memforge.local_agent.conversation_executor import run_device_collection

    def report(phase: str, completed: int, total: int) -> None:
        unit = "conversation" if phase == "fetching" else "message"
        _report_local_agent_progress(
            report_progress,
            _sync_progress_snapshot(phase=phase, completed=completed, total=total, unit=unit),
        )

    teams = TeamsChatClient(region=_teams_region_from_payload(payload))
    try:
        await teams.validate()
        return await run_device_collection(
            TeamsConversationProvider(teams),
            client,
            source_id=source_id,
            local_agent_job_id=str(job["job_id"]),
            local_agent_attempt_count=int(job["attempt_count"]),
            config_revision=str(payload.get("source_config_revision") or ""),
            dry_run=bool(payload.get("dry_run")),
            force_full_sync=bool(payload.get("force_full_sync", False)),
            report=report,
        )
    finally:
        await teams.close()


def _github_profile_from_cloud_job(job: dict[str, Any]) -> dict[str, Any]:
    payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
    profile: dict[str, Any] = {
        "repo_url": str(payload.get("repo_url") or "").strip(),
        "ref": str(payload.get("ref") or "main").strip() or "main",
    }
    for key in ("include_paths", "exclude_paths", "include_extensions"):
        if key in payload:
            profile[key] = payload[key]
    return profile


def _kb_profile_from_cloud_job(job: dict[str, Any]) -> dict[str, Any]:
    payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
    root = str(payload.get("root") or "").strip()
    vault_id = str(payload.get("vault_id") or "").strip()
    if not root:
        raise click.ClickException("local_markdown root is required")
    if not vault_id:
        raise click.ClickException("local_markdown vault_id is required")
    profile: dict[str, Any] = {
        "root": root,
        "vault_id": vault_id,
    }
    if "include" in payload:
        profile["include"] = payload["include"]
    if "exclude" in payload:
        profile["exclude"] = payload["exclude"]
    return profile


def _local_agent_job_source_client(
    job: dict[str, Any],
    payload: dict[str, Any],
    *,
    operation: str,
    client: ToolClient,
) -> tuple[str, ToolClient, dict[str, Any] | None]:
    source_id = str(job.get("source_id") or payload.get("source_id") or "").strip()
    if not source_id:
        return "", client, {"operation": operation, "error": "source_id is required"}
    if client.target.edition is Edition.OSS:
        return source_id, client, None
    workspace_id = str(job.get("workspace_id") or payload.get("workspace_id") or "").strip()
    if not workspace_id:
        return source_id, client, {
            "operation": operation,
            "source_id": source_id,
            "error": "workspace_id is required",
        }
    return source_id, client.for_workspace(workspace_id), None


async def _collect_jira_documents_from_cloud_job(
    job: dict[str, Any],
    *,
    source_id: str,
    limit: int,
    jira_cookie: str,
    client: ToolClient,
    sync_snapshot_id: str,
    local_agent_job_id: str,
    local_agent_attempt_count: int,
    report_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    from memforge.genes.jira_gene import JiraGene

    payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
    if limit:
        raise ValueError("limited Jira collection cannot finalize a fenced source snapshot")
    config = _jira_cloud_config_from_job_payload(payload)
    base_url = str(config["base_url"])
    config["jira_cookie"] = jira_cookie

    gene = JiraGene(config, source_id)
    await gene.authenticate()
    documents: list[dict[str, Any]] = []
    _report_local_agent_progress(
        report_progress,
        _sync_progress_snapshot(phase="discovering", completed=0, unit="issue"),
    )
    try:
        inventory_items = []
        async for item in gene.discover_inventory():
            if limit and len(inventory_items) >= limit:
                break
            issue_key = str(item.extra.get("issue_key") or item.item_id.replace("jira-", "")).strip()
            revision = str(item.version or "").strip()
            if not issue_key or not revision:
                raise RuntimeError("Jira inventory item is missing issue identity or provider revision")
            inventory_items.append(item)
            _report_local_agent_progress(
                report_progress,
                _sync_progress_snapshot(
                    phase="discovering",
                    completed=len(inventory_items),
                    unit="issue",
                ),
            )

        items_by_doc_id = {
            build_jira_doc_id(
                source_id=source_id,
                issue_key=str(item.extra.get("issue_key") or item.item_id.replace("jira-", "")),
            ): item
            for item in inventory_items
        }
        required_doc_ids = set(items_by_doc_id)
        if not limit:
            required_doc_ids = _required_local_source_doc_ids(
                client,
                source_id=source_id,
                items=[
                    {
                        "doc_id": doc_id,
                        "revision": str(item.version),
                        "change_kind": "upsert",
                    }
                    for doc_id, item in items_by_doc_id.items()
                ],
                coverage="complete_snapshot",
                known_doc_ids=set(items_by_doc_id),
                sync_snapshot_id=sync_snapshot_id,
                local_agent_job_id=local_agent_job_id,
                local_agent_attempt_count=local_agent_attempt_count,
            )
        items_to_fetch = [
            item
            for doc_id, item in items_by_doc_id.items()
            if doc_id in required_doc_ids
        ]
        _report_local_agent_progress(
            report_progress,
            _sync_progress_snapshot(
                phase="fetching",
                completed=0,
                total=len(items_to_fetch),
                unit="issue",
            ),
        )
        for index, item in enumerate(items_to_fetch, start=1):
            raw = await gene.fetch(item)
            issue_key = str(item.extra.get("issue_key") or item.item_id.replace("jira-", "")).strip()
            raw_payload = json.loads(raw.body.decode("utf-8"))
            hydrated_fields = raw_payload.get("fields")
            hydrated_revision = (
                str(hydrated_fields.get("updated") or "").strip()
                if isinstance(hydrated_fields, dict)
                else ""
            )
            inventory_revision = str(item.version or "").strip()
            if not hydrated_revision or hydrated_revision != inventory_revision:
                raise RuntimeError(
                    f"Jira issue {issue_key} changed during materialization; retry inventory"
                )
            artifact_input_hashes: list[str] = []
            remaining_artifact_bytes = MAX_SOURCE_ARTIFACT_STORAGE_BYTES_PER_UNIT
            for artifact in raw.artifacts:
                content = await read_source_artifact_bytes(
                    artifact=artifact,
                    remaining_unit_bytes=remaining_artifact_bytes,
                    open_artifact=gene.open_source_artifact,
                )
                remaining_artifact_bytes -= len(content)
                artifact_response = client.push_source_artifact(
                    source_id=source_id,
                    source_unit_key=issue_key,
                    provider_key=artifact.provider_key,
                    provider_revision=artifact.provider_revision,
                    parent_observation_type=artifact.parent_observation_type,
                    parent_provider_key=artifact.parent_provider_key,
                    filename=artifact.filename,
                    media_type=artifact.media_type,
                    content=content,
                    local_agent_job_id=local_agent_job_id,
                    local_agent_attempt_count=local_agent_attempt_count,
                )
                _raise_if_local_agent_lease_not_current(artifact_response)
                input_hash = str(artifact_response.get("input_sha256") or "").strip()
                if artifact_response.get("error") or len(input_hash) != 64:
                    raise RuntimeError(
                        str(
                            artifact_response.get("detail")
                            or artifact_response.get("error")
                            or "Jira Source Artifact upload failed"
                        )
                    )
                artifact_input_hashes.append(input_hash)
            raw_hash = hashlib.sha256(raw.body).hexdigest()
            documents.append(
                {
                    "base_url": base_url,
                    "issue_key": issue_key,
                    "source_url": item.source_url,
                    "title": item.title,
                    "provider_revision": str(item.version),
                    "raw_payload": raw_payload,
                    "raw_hash": raw_hash,
                    "artifact_input_hashes": artifact_input_hashes,
                }
            )
            _report_local_agent_progress(
                report_progress,
                _sync_progress_snapshot(
                    phase="fetching",
                    completed=index,
                    total=len(items_to_fetch),
                    unit="issue",
                ),
            )
    finally:
        client = getattr(gene, "_client", None)
        if client is not None:
            await client.aclose()
    return {
        "documents": documents,
        "inventory_count": len(inventory_items),
        "reused_count": len(inventory_items) - len(items_to_fetch),
    }


def _jira_cloud_config_from_job_payload(payload: dict[str, Any]) -> dict[str, Any]:
    from memforge.genes.jira_gene import JIRA_AUTH_MODE_COOKIE

    config = dict(payload)
    config.pop("local_agent_documents_dir", None)
    config.pop("pat", None)
    config.pop("pat_encrypted", None)
    config.pop("pat_configured", None)
    config["sync_mode"] = "cloud"
    base_url = str(config.get("base_url") or "").strip().rstrip("/")
    if not base_url:
        raise click.ClickException("Jira base_url is required")
    auth_mode = str(config.get("auth_mode") or JIRA_AUTH_MODE_COOKIE).strip().lower()
    if auth_mode != JIRA_AUTH_MODE_COOKIE:
        raise click.ClickException("Jira local sync requires browser-session authentication")
    config["base_url"] = base_url
    config["auth_mode"] = auth_mode
    return config


def _report_local_agent_progress(
    reporter: Callable[[dict[str, Any]], None] | None,
    progress: dict[str, Any],
) -> None:
    if reporter is not None:
        reporter(progress)


def _sync_progress_snapshot(
    *,
    phase: str,
    completed: int | None = None,
    total: int | None = None,
    unit: str | None = None,
    source_time_start: str | None = None,
    source_time_end: str | None = None,
    changed: int | None = None,
    failed: int | None = None,
    memories_created: int | None = None,
) -> dict[str, Any]:
    snapshot: dict[str, Any] = {"schema_version": 1, "phase": phase}
    if completed is not None and unit is not None:
        snapshot["progress"] = {"completed": completed, "unit": unit}
        if total is not None:
            snapshot["progress"]["total"] = total
    source_time_range = {key: value for key, value in (("start", source_time_start), ("end", source_time_end)) if value}
    if source_time_range:
        snapshot["source_time_range"] = source_time_range
    counts = {
        key: value
        for key, value in (
            ("changed", changed),
            ("failed", failed),
            ("memories_created", memories_created),
        )
        if value is not None
    }
    if counts:
        snapshot["counts"] = counts
    return normalize_sync_progress_snapshot(snapshot)


def _int_or_zero(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _cloud_job_limit(value: Any, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(parsed, 0)


def _push_kb_profile_to_source(
    name: str,
    profile: dict[str, Any],
    *,
    source_id: str,
    limit: int,
    force_full_sync: bool,
    submitted_by: str | None,
    client: ToolClient,
    sync_snapshot_id: str | None = None,
    local_agent_job_id: str | None = None,
    local_agent_attempt_count: int | None = None,
    report_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    if limit and sync_snapshot_id:
        raise ValueError("limited collection cannot finalize a fenced source snapshot")
    root, include, exclude, vault_id = _resolve_kb_profile(name, profile)
    counts = {"included": 0, "ignored": 0, "too_large": 0, "invalid_utf8": 0, "unreadable": 0}
    pushed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    entries = list(_scan_kb_profile(root, include=include, exclude=exclude, counts=counts))
    selected_entries = entries[:limit] if limit else entries
    _report_local_agent_progress(
        report_progress,
        _sync_progress_snapshot(
            phase="discovering",
            completed=len(selected_entries),
            unit="file",
        ),
    )

    incomplete_reasons = {
        key: counts[key]
        for key in ("too_large", "invalid_utf8", "unreadable")
        if counts[key]
    }
    if incomplete_reasons:
        return {
            "profile": name,
            "root": str(root),
            "vault_id": vault_id,
            "source_id": source_id,
            "counts": {**counts, "pushed": 0, "failed": sum(incomplete_reasons.values())},
            "error": "local collection was incomplete",
            "retryable": True,
        }
    try:
        _attest_kb_scan_stable(
            root,
            include=include,
            exclude=exclude,
            entries=entries,
        )
    except ValueError as exc:
        return {
            "profile": name,
            "root": str(root),
            "vault_id": vault_id,
            "source_id": source_id,
            "counts": {**counts, "pushed": 0, "failed": 1},
            "error": "local collection was incomplete",
            "detail": str(exc),
            "retryable": True,
        }

    entries_by_doc_id: dict[str, dict[str, Any]] = {}
    manifest_items: list[dict[str, str]] = []
    for entry in selected_entries:
        doc_id = build_local_markdown_doc_id(
            source_id=source_id,
            vault_id=vault_id,
            relative_path=str(entry["relative_path"]),
        )
        entries_by_doc_id[doc_id] = entry
        manifest_items.append(
            {
                "doc_id": doc_id,
                "revision": str(entry["raw_hash"]),
                "change_kind": "upsert",
            }
        )

    required_doc_ids = set(entries_by_doc_id)
    comparison_started_at = time.perf_counter()
    fallback_reason: str | None = None
    if not limit and sync_snapshot_id and local_agent_job_id and local_agent_attempt_count is not None:
        try:
            required_doc_ids = _required_local_source_doc_ids(
                client,
                source_id=source_id,
                items=manifest_items,
                coverage="complete_snapshot",
                known_doc_ids=set(entries_by_doc_id),
                sync_snapshot_id=sync_snapshot_id,
                local_agent_job_id=local_agent_job_id,
                local_agent_attempt_count=local_agent_attempt_count,
            )
        except RuntimeError as exc:
            return {
                "profile": name,
                "root": str(root),
                "vault_id": vault_id,
                "source_id": source_id,
                "counts": {**counts, "pushed": 0, "failed": 0},
                "error": str(exc),
                "retryable": True,
            }
    else:
        fallback_reason = "missing_fenced_snapshot"
    comparison_latency_ms = round((time.perf_counter() - comparison_started_at) * 1000, 3)
    entries_to_upload = [
        entry
        for doc_id, entry in entries_by_doc_id.items()
        if doc_id in required_doc_ids
    ]

    _report_local_agent_progress(
        report_progress,
        _sync_progress_snapshot(
            phase="uploading",
            completed=0,
            total=len(entries_to_upload),
            unit="file",
        ),
    )

    uploaded_body_bytes = 0
    worktree = _git_worktree(root) if entries_to_upload else None
    for index, entry in enumerate(entries_to_upload, start=1):
        uploaded_body_bytes += int(entry["bytes"])
        response = client.push_local_markdown_document(
            source_id=source_id,
            vault_id=vault_id,
            relative_path=entry["relative_path"],
            markdown_body=entry["text"],
            content_type=entry["content_type"],
            title=entry["title"],
            raw_hash=entry["raw_hash"],
            source_updated_at=_local_file_source_time(root, entry, worktree),
            sync_snapshot_id=sync_snapshot_id,
            local_agent_job_id=local_agent_job_id,
            local_agent_attempt_count=local_agent_attempt_count,
            submitted_by=submitted_by,
        )
        _raise_if_local_agent_lease_not_current(response)
        if isinstance(response, dict) and response.get("error"):
            failed.append(
                {
                    "relative_path": entry["relative_path"],
                    "error": response.get("error"),
                    "detail": response.get("detail"),
                    "status_code": response.get("status_code"),
                }
            )
        else:
            pushed.append(
                {
                    "relative_path": entry["relative_path"],
                    "doc_id": response.get("doc_id"),
                    "document_hash": response.get("document_hash"),
                }
            )
        _report_local_agent_progress(
            report_progress,
            _sync_progress_snapshot(
                phase="uploading",
                completed=index,
                total=len(entries_to_upload),
                unit="file",
                failed=len(failed),
            ),
        )

    payload = {
        "profile": name,
        "root": str(root),
        "vault_id": vault_id,
        "source_id": source_id,
        "counts": {
            **counts,
            "reused": len(selected_entries) - len(entries_to_upload),
            "pushed": len(pushed),
            "failed": len(failed),
        },
        "pushed": pushed,
        "failed": failed,
        "metrics": {
            "manifest_items": len(manifest_items),
            "full_bodies_read": len(selected_entries),
            "full_bodies_uploaded": len(pushed),
            "bytes_uploaded": uploaded_body_bytes,
            "comparison_latency_ms": comparison_latency_ms,
            "end_to_end_latency_ms": round((time.perf_counter() - started_at) * 1000, 3),
            "fallback_reason": fallback_reason,
        },
    }
    if failed:
        payload["error"] = "one or more documents failed to push"
    if not failed:
        sync_result = client.start_source_processing(
            source_id=source_id,
            force_full_sync=force_full_sync,
            sync_snapshot_id=sync_snapshot_id,
            local_agent_job_id=local_agent_job_id,
            local_agent_attempt_count=local_agent_attempt_count,
        )
        payload["sync_started"] = not bool(sync_result.get("error"))
        payload.update(source_processing_receipt(sync_result))
        if sync_result.get("error"):
            payload["error"] = "source processing failed to start"
            payload["sync_error"] = sync_result
    if payload.get("error"):
        payload["retryable"] = True
    return payload


def _preview_kb_profile(name: str, profile: dict[str, Any], *, limit: int) -> dict[str, Any]:
    root, include, exclude, vault_id = _resolve_kb_profile(name, profile)
    counts = {"included": 0, "ignored": 0, "too_large": 0, "invalid_utf8": 0, "unreadable": 0}
    items: list[dict[str, Any]] = []
    for entry in _scan_kb_profile(root, include=include, exclude=exclude, counts=counts):
        if len(items) < limit:
            items.append(
                {
                    "relative_path": entry["relative_path"],
                    "title": entry["title"],
                    "content_type": entry["content_type"],
                    "raw_hash": entry["raw_hash"],
                    "bytes": entry["bytes"],
                }
            )
    return {
        "profile": name,
        "root": str(root),
        "vault_id": vault_id,
        "counts": counts,
        "items": items,
    }


def _resolve_kb_profile(name: str, profile: dict[str, Any]) -> tuple[Path, list[str], list[str], str]:
    root_value = str(profile.get("root") or "").strip()
    if not root_value:
        raise click.ClickException("Knowledge-base root is required.")
    root = Path(root_value).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise click.ClickException(f"Knowledge-base root is not a directory: {root}")
    include = _string_list(profile.get("include")) or DEFAULT_KB_INCLUDE
    exclude = _string_list(profile.get("exclude")) or DEFAULT_KB_EXCLUDE
    vault_id = str(profile.get("vault_id") or name)
    return root, include, exclude, vault_id


def _scan_kb_profile(
    root: Path,
    *,
    include: list[str],
    exclude: list[str],
    counts: dict[str, int],
):
    """Yield one entry per included file, updating ``counts`` in place.

    The CLI is a thin bridge: it reads each file's raw UTF-8 text and tags it
    with a ``content_type`` derived from the extension. All conversion to
    markdown happens server-side during sync.
    """
    max_bytes = 1_000_000
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            counts["ignored"] += 1
            continue
        if not path.is_file():
            continue
        rel_path = path.relative_to(root).as_posix()
        if _glob_match(rel_path, exclude) or not _glob_match(rel_path, include):
            counts["ignored"] += 1
            continue
        try:
            stat_before = path.stat()
            size = stat_before.st_size
        except OSError:
            counts["unreadable"] += 1
            continue
        if size > max_bytes:
            counts["too_large"] += 1
            continue
        try:
            raw = path.read_bytes()
            text = raw.decode("utf-8")
            stat_after = path.stat()
        except UnicodeDecodeError:
            counts["invalid_utf8"] += 1
            continue
        except OSError:
            counts["unreadable"] += 1
            continue
        fingerprint_before = (
            stat_before.st_dev,
            stat_before.st_ino,
            stat_before.st_size,
            stat_before.st_mtime_ns,
        )
        fingerprint_after = (
            stat_after.st_dev,
            stat_after.st_ino,
            stat_after.st_size,
            stat_after.st_mtime_ns,
        )
        if fingerprint_before != fingerprint_after or len(raw) != stat_after.st_size:
            counts["unreadable"] += 1
            continue
        content_type = _content_type_for_path(path)
        raw_hash = hashlib.sha256(raw).hexdigest()
        title = (_markdown_title(text) if content_type == "text/markdown" else None) or rel_path
        counts["included"] += 1
        yield {
            "relative_path": rel_path,
            "title": title,
            "content_type": content_type,
            "raw_hash": raw_hash,
            "text": text,
            "bytes": size,
            "stat_fingerprint": fingerprint_after,
            "modified_at": datetime.fromtimestamp(stat_after.st_mtime, timezone.utc).isoformat(),
        }


@dataclass(frozen=True)
class _GitWorktree:
    """The Git checkout that holds a local collection root."""

    # Root-relative paths Git reports as modified, staged, renamed or untracked.
    changed_paths: frozenset[str]


def _git_output(root: Path, *args: str) -> str | None:
    """Run a read-only ``git`` command in ``root``; None when Git is absent or refuses."""

    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            check=False,
            # Reading status must not take the index lock from the user's own Git work.
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except FileNotFoundError:
        return None
    return result.stdout if result.returncode == 0 else None


def _git_worktree(root: Path) -> _GitWorktree | None:
    """The Git checkout holding ``root`` and its uncommitted paths; None outside Git."""

    prefix = _git_output(root, "rev-parse", "--show-prefix")
    status = (
        _git_output(root, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--", ".")
        if prefix is not None
        else None
    )
    if prefix is None or status is None:
        return None
    prefix = prefix.strip()
    changed: set[str] = set()
    records = iter(status.split("\0"))
    for record in records:
        if not record:
            continue
        state, path = record[:2], record[3:]
        changed.add(path)
        if "R" in state or "C" in state:
            # A rename or copy record is followed by its original path.
            changed.add(next(records, ""))
    return _GitWorktree(
        changed_paths=frozenset(path.removeprefix(prefix) for path in changed if path.startswith(prefix)),
    )


def _local_file_source_time(root: Path, entry: dict[str, Any], worktree: _GitWorktree | None) -> str:
    """A local file's own time for its current content.

    A committed file without local changes takes its latest commit time,
    because a checkout or clone sets the modification time to when it ran.
    Any other file takes its modification time, the only time the file
    system records for its content.
    """

    relative_path = str(entry["relative_path"])
    if worktree is not None and relative_path not in worktree.changed_paths:
        # Empty for a file Git does not track, such as an ignored one.
        committed_at = source_time_iso((_git_output(root, "log", "-1", "--format=%cI", "--", relative_path) or "").strip())
        if committed_at is not None:
            return committed_at
    return str(entry["modified_at"])


def _attest_kb_scan_stable(
    root: Path,
    *,
    include: list[str],
    exclude: list[str],
    entries: list[dict[str, Any]],
) -> None:
    """Prove that one body scan still describes the current configured scope."""
    expected = {
        str(entry["relative_path"]): tuple(entry["stat_fingerprint"])
        for entry in entries
    }
    observed: dict[str, tuple[int, int, int, int]] = {}
    try:
        paths = sorted(root.rglob("*"))
        for path in paths:
            if path.is_symlink() or not path.is_file():
                continue
            relative_path = path.relative_to(root).as_posix()
            if _glob_match(relative_path, exclude) or not _glob_match(relative_path, include):
                continue
            stat = path.stat()
            observed[relative_path] = (
                stat.st_dev,
                stat.st_ino,
                stat.st_size,
                stat.st_mtime_ns,
            )
    except OSError as exc:
        raise ValueError("local collection could not be revalidated") from exc
    if set(observed) != set(expected):
        raise ValueError("local collection membership changed during collection")
    changed = sorted(
        relative_path
        for relative_path, fingerprint in observed.items()
        if fingerprint != expected[relative_path]
    )
    if changed:
        raise ValueError(f"local file changed during collection: {changed[0]}")


def _glob_match(relative_path: str, patterns: list[str]) -> bool:
    path = PurePosixPath(relative_path)
    return any(path.match(pattern) for pattern in patterns)


_CONTENT_TYPE_BY_SUFFIX = {
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
    ".json": "application/json",
    ".html": "text/html",
    ".htm": "text/html",
}


def _content_type_for_path(path: Path) -> str:
    """Map a file extension to the content type the service converter expects."""
    return _CONTENT_TYPE_BY_SUFFIX.get(path.suffix.lower(), "text/plain")


def _markdown_title(markdown_body: str) -> str | None:
    for line in markdown_body.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip() or None
    return None


@adapter.group("auth")
def adapter_auth():
    """Refresh local authentication used by adapter-backed sources."""
    pass


def _is_jira_principal_change(upload_result: dict) -> bool:
    from memforge.auth.jira_auth import JIRA_PRINCIPAL_CHANGED_CODE

    return upload_result.get("code") == JIRA_PRINCIPAL_CHANGED_CODE


def _principal_change_payload(upload_result: dict) -> dict:
    """Translate a Jira principal-change conflict into the signal the CLI emits."""
    detail = upload_result.get("detail")
    principal = detail if isinstance(detail, dict) else {}
    return {
        "error": "principal_changed",
        "origin": principal.get("origin"),
        "old_principal_id": principal.get("old_principal_id"),
        "new_principal_id": principal.get("new_principal_id"),
    }


def _cookie_hash(cookie_header: str) -> str:
    return hashlib.sha256(cookie_header.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _JiraSessionFanOut:
    """Workspaces that receive one captured Jira session.

    ``clients`` maps each receiving workspace to its scoped client. ``failures``
    are per-workspace results for workspaces whose Jira Sources could not be
    read. ``error`` is set when no workspace can receive the session.
    """

    clients: dict[str, Any]
    failures: tuple[dict, ...] = ()
    error: dict | None = None


def _jira_session_fan_out(client: Any, *, origin: str, workspace_id: str | None) -> _JiraSessionFanOut:
    """Select the workspaces that receive a captured Jira session for one origin.

    An explicit workspace receives it directly. Otherwise an account with one
    selectable workspace uploads there, including before its first Jira Source
    exists, and an account with several uploads to every workspace that has a
    Jira Source for this origin.
    """
    if workspace_id:
        return _JiraSessionFanOut(clients={workspace_id: client})
    listing = client.list_workspaces()
    if listing.get("error"):
        return _JiraSessionFanOut(clients={}, error=listing)
    selectable = [
        str(workspace["workspace_id"])
        for workspace in listing.get("workspaces") or []
        if workspace.get("selectable") and workspace.get("workspace_id")
    ]
    if not selectable:
        return _JiraSessionFanOut(
            clients={},
            error={"error": "no_selectable_workspace", "detail": "No workspace can receive the Jira session."},
        )
    if len(selectable) == 1:
        return _JiraSessionFanOut(clients={selectable[0]: client.for_workspace(selectable[0])})

    clients: dict[str, Any] = {}
    failures: list[dict] = []
    for selectable_id in selectable:
        scoped = client.for_workspace(selectable_id)
        origins = scoped.list_jira_origins()
        if origins.get("error"):
            failures.append(_jira_workspace_failure(selectable_id, origins))
            continue
        if any(
            entry.get("origin") == origin and entry.get("configured")
            for entry in origins.get("origins") or []
        ):
            clients[selectable_id] = scoped
    if not clients and not failures:
        return _JiraSessionFanOut(
            clients={},
            error={
                "error": "jira_source_not_found",
                "detail": f"No workspace has a Jira Source for {origin}. Pass --workspace-id to choose one.",
                "workspace_ids": selectable,
            },
        )
    return _JiraSessionFanOut(clients=clients, failures=tuple(failures))


def _jira_workspace_failure(workspace_id: str, response: dict) -> dict:
    return {
        "workspace_id": workspace_id,
        "status": "failed",
        "status_code": response.get("status_code"),
        "code": response.get("code"),
        "detail": response.get("detail") or response.get("error"),
    }


def _jira_upload_result(workspace_id: str, uploaded: dict) -> dict:
    """Describe one workspace's Jira session upload."""
    if _is_jira_principal_change(uploaded):
        principal = _principal_change_payload(uploaded)
        return {
            "workspace_id": workspace_id,
            "status": "principal_changed",
            "old_principal_id": principal["old_principal_id"],
            "new_principal_id": principal["new_principal_id"],
        }
    if uploaded.get("error"):
        return _jira_workspace_failure(workspace_id, uploaded)
    return {"workspace_id": workspace_id, "status": "uploaded", "session": uploaded}


def _jira_refresh_payload(origin: str, results: list[dict]) -> dict:
    payload: dict[str, Any] = {"origin": origin, "results": results}
    if any(result["status"] != "uploaded" for result in results):
        payload["error"] = "jira_session_upload_incomplete"
    return payload


async def run_watch_tick(*, base_url, browser, clients, last_hash, capture, log):
    """One watch iteration over every receiving workspace. Returns (action, new_last_hash).

    action is one of: uploaded, unchanged, expired, principal_conflict, transport_error.
    ``clients`` maps each receiving workspace ID to its scoped client. The new
    cookie is recorded only when every workspace accepted it, so the next tick
    retries a partial upload. Pure over its injected collaborators (capture,
    clients, log) so it is unit-testable.
    """
    from memforge.auth.jira_auth import JiraAuthSessionMissingError

    try:
        result = await capture(base_url, browser=browser)
    except JiraAuthSessionMissingError as exc:
        for client in clients.values():
            client.mark_jira_session_expired(base_url=base_url, error=str(exc))
        log(f"Jira session for {base_url} is not active; sign back into Jira in your browser. ({exc})")
        return "expired", None

    new_hash = _cookie_hash(result.cookie_header)
    if new_hash == last_hash:
        return "unchanged", last_hash

    statuses = set()
    for workspace_id, client in clients.items():
        outcome = _jira_upload_result(
            workspace_id,
            client.upload_jira_session(
                base_url=base_url,
                cookie_header=result.cookie_header,
                browser=result.browser,
            ),
        )
        statuses.add(outcome["status"])
        if outcome["status"] == "principal_changed":
            log(
                f"[{workspace_id}] A different Jira user is signed in for {base_url}; "
                "re-run refresh with --confirm-principal-change."
            )
        elif outcome["status"] == "failed":
            log(f"[{workspace_id}] Upload to MemForge failed: {outcome['code'] or outcome['detail']}")
        else:
            log(f"[{workspace_id}] Refreshed Jira session for {base_url} (cookie {new_hash[:8]}).")
    if "failed" in statuses:
        return "transport_error", last_hash
    if "principal_changed" in statuses:
        return "principal_conflict", last_hash
    return "uploaded", new_hash


def _make_browser_session_group(descriptor):
    """Build an ``adapter auth <provider>`` group (status/list/forget/refresh).

    These commands manage the browser session stored on the remote MemForge
    server over the active target. ``status``, ``list``, and ``forget`` ask the
    server about its stored session, and ``refresh`` captures the cookie from
    the local browser and uploads it, so the server owns the durable session.
    """
    provider = descriptor.provider
    if provider != "jira":
        raise NotImplementedError(
            f"Browser-session CLI is implemented for Jira only; {provider} needs its own ToolClient methods."
        )

    @click.group(name=provider, help=f"Manage the {descriptor.label} browser session on the server.")
    def group():
        """Manage the server's stored browser session for this provider.

        ``status``, ``list``, and ``forget`` talk to the remote MemForge server
        for one workspace. ``refresh`` and ``watch`` capture the cookie from the
        local browser once and upload it to every workspace with a Jira Source
        for the origin, or to the workspace given by ``--workspace-id``.
        """

    def fan_out_or_emit(ctx, client, *, base_url: str, workspace_id: str | None) -> _JiraSessionFanOut | None:
        from memforge.auth.jira_auth import canonical_jira_origin

        try:
            origin = canonical_jira_origin(base_url)
        except ValueError as exc:
            _emit_tool_payload(ctx, {"error": "invalid_base_url", "detail": str(exc)})
            return None
        fan_out = _jira_session_fan_out(client, origin=origin, workspace_id=workspace_id)
        if fan_out.error is not None:
            _emit_tool_payload(ctx, fan_out.error)
            return None
        if not fan_out.clients:
            _emit_tool_payload(ctx, _jira_refresh_payload(origin, list(fan_out.failures)))
            return None
        return fan_out

    @group.command("status")
    @click.option("--base-url", required=True, help=f"{descriptor.label} base URL.")
    @_workspace_id_option
    @click.pass_context
    def status_cmd(ctx, base_url, workspace_id):
        """Show the server's stored session status for an origin."""
        _emit_tool_payload(ctx, _tool_client(ctx, workspace_id).get_jira_session(base_url))

    @group.command("list")
    @_workspace_id_option
    @click.pass_context
    def list_cmd(ctx, workspace_id):
        """List known origins from the server."""
        _emit_tool_payload(ctx, _tool_client(ctx, workspace_id).list_jira_origins())

    @group.command("forget")
    @click.option("--base-url", required=True, help="Origin whose stored session to delete.")
    @_workspace_id_option
    @click.pass_context
    def forget_cmd(ctx, base_url, workspace_id):
        """Forget the local and server-side stored session for an origin."""
        from memforge.auth.jira_auth import canonical_jira_origin
        from memforge.auth.jira_browser_session import JiraBrowserSession

        origin = canonical_jira_origin(base_url)
        JiraBrowserSession().forget(origin=origin)
        _emit_tool_payload(ctx, _tool_client(ctx, workspace_id).forget_jira_session(origin))

    @group.command("refresh")
    @click.option("--base-url", required=True, help=f"{descriptor.label} base URL.")
    @click.option("--browser", default=None, help="Browser to read cookies from, for example chrome or edge.")
    @click.option("--confirm-principal-change", is_flag=True, help="Allow this session to replace a different user.")
    @_workspace_id_option
    @click.pass_context
    def refresh_cmd(ctx, base_url, browser, confirm_principal_change, workspace_id):
        """Capture the local browser session and upload it to each receiving workspace."""
        from memforge.auth import jira_capture
        from memforge.auth.jira_auth import JiraAuthSessionError, JiraAuthSessionMissingError

        fan_out = fan_out_or_emit(ctx, _tool_client(ctx, workspace_id), base_url=base_url, workspace_id=workspace_id)
        if fan_out is None:
            return
        try:
            result = asyncio.run(
                jira_capture.capture_and_prevalidate(
                    base_url,
                    browser=browser,
                    interactive=True,
                )
            )
        except JiraAuthSessionMissingError as exc:
            _emit_tool_payload(ctx, {"error": "no_session", "detail": str(exc)})
            return
        except (JiraAuthSessionError, ValueError) as exc:
            _emit_tool_payload(ctx, {"error": "auth_failed", "detail": str(exc)})
            return
        results = [
            _jira_upload_result(
                receiving_id,
                client.upload_jira_session(
                    base_url=result.origin,
                    cookie_header=result.cookie_header,
                    browser=result.browser,
                    confirm_principal_change=confirm_principal_change,
                ),
            )
            for receiving_id, client in fan_out.clients.items()
        ]
        _emit_tool_payload(ctx, _jira_refresh_payload(result.origin, [*results, *fan_out.failures]))

    @group.command("watch")
    @click.option("--base-url", required=True, help=f"{descriptor.label} base URL.")
    @click.option("--browser", default=None, help="Browser to read cookies from, for example chrome or edge.")
    @click.option(
        "--interval-seconds",
        type=int,
        default=WATCH_DEFAULT_INTERVAL_SECONDS,
        show_default=True,
        help="Seconds between re-capture attempts. Keep it under your Jira idle timeout.",
    )
    @_workspace_id_option
    @click.pass_context
    def watch_cmd(ctx, base_url, browser, interval_seconds, workspace_id):
        """Keep each receiving workspace's Jira session fresh by re-capturing on an interval."""
        from memforge.auth import jira_capture

        fan_out = fan_out_or_emit(ctx, _tool_client(ctx, workspace_id), base_url=base_url, workspace_id=workspace_id)
        if fan_out is None:
            return
        for failure in fan_out.failures:
            click.echo(
                f"[{failure['workspace_id']}] Could not read Jira Sources; this workspace is not refreshed: "
                f"{failure['code'] or failure['detail']}"
            )

        async def _capture(url, *, browser=None):
            return await jira_capture.capture_and_prevalidate(url, browser=browser)

        async def _loop():
            last_hash = None
            backoff = WATCH_BACKOFF_BASE_SECONDS
            while True:
                try:
                    action, last_hash = await run_watch_tick(
                        base_url=base_url,
                        browser=browser,
                        clients=fan_out.clients,
                        last_hash=last_hash,
                        capture=_capture,
                        log=click.echo,
                    )
                except Exception as exc:  # a daemon must survive any single-tick failure
                    click.echo(f"Jira watch tick failed: {exc}")
                    action = "transport_error"
                if action == "transport_error":
                    await asyncio.sleep(min(backoff, WATCH_BACKOFF_MAX_SECONDS))
                    backoff = min(backoff * 2, WATCH_BACKOFF_MAX_SECONDS)
                    continue
                backoff = WATCH_BACKOFF_BASE_SECONDS
                await asyncio.sleep(interval_seconds)

        try:
            asyncio.run(_loop())
        except KeyboardInterrupt:
            click.echo("Stopped Jira session watch.")

    return group


browser_session.ensure_builtin_providers()
for _descriptor in browser_session.registered_providers():
    adapter_auth.add_command(_make_browser_session_group(_descriptor), name=_descriptor.provider)


# ---------------------------------------------------------------------------
# auth group
# ---------------------------------------------------------------------------


@cli.group()
def auth():
    """Manage browser-session authentication helpers."""
    pass


@auth.command("teams")
@click.option("--region", default="emea", type=click.Choice(["emea", "amer", "apac"]), help="Teams API region")
def auth_teams(region: str):
    """Authenticate with Microsoft Teams through Keychain and Teams Web."""

    from memforge.auth.teams_auth import TeamsAuthenticator

    authenticator = TeamsAuthenticator()
    console.print("[bold]Loading Teams session...[/]\n")

    try:
        token_data = authenticator.authenticate(region=region, wait_seconds=300)
    except RuntimeError as e:
        console.print(f"[yellow]{e}[/]")
        return

    tokens = token_data.get("tokens", {})
    console.print(f"[green]Captured {len(tokens)} tokens:[/]")

    from datetime import datetime, timezone

    table = Table()
    table.add_column("Audience")
    table.add_column("Scopes")
    table.add_column("Expires")

    for audience, info in tokens.items():
        expires_at = info.get("expiresAt", 0)
        if expires_at:
            exp_str = datetime.fromtimestamp(expires_at, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        else:
            exp_str = "unknown"
        scopes = info.get("scopes", "")[:60]
        table.add_row(audience, scopes, exp_str)
    console.print(table)

    # Verify with a test API call
    console.print("\n[dim]Verifying token with Teams API...[/]")
    chat_token = authenticator.get_token_for_audience(tokens, "https://ic3.teams.office.com")
    if chat_token:
        import httpx

        try:
            resp = httpx.get(
                f"https://teams.cloud.microsoft/api/chatsvc/{region}/v1/users/ME/conversations",
                headers={"Authorization": f"Bearer {chat_token}"},
                params={"view": "msnp24Equivalent", "pageSize": 3},
            )
            if resp.status_code == 200:
                data = resp.json()
                convs = data.get("conversations", [])
                console.print(f"[green]API verified: {len(convs)} conversations returned[/]")
                for c in convs[:3]:
                    topic = c.get("threadProperties", {}).get("topic", "(untitled)")
                    console.print(f"  [dim]{topic[:60]}[/]")
            else:
                console.print(f"[yellow]API returned {resp.status_code}[/]")
        except Exception as e:
            console.print(f"[yellow]API verification failed: {e}[/]")
    else:
        console.print("[yellow]No Chat API token found — skipping verification[/]")

    if authenticator.keychain_session_available:
        console.print("\n[bold green]Done! Teams session saved to the OS keychain.[/]")
    else:
        console.print(
            "\n[bold yellow]Teams is connected for this command, but the access token could not be saved "
            "to the OS keychain.[/]"
        )


@auth.command("status")
def auth_status():
    """Show authentication status for all configured sources."""
    from memforge.auth.teams_auth import TeamsAuthenticator

    tokens = TeamsAuthenticator.load_tokens()
    if not tokens:
        console.print("[yellow]No Teams tokens found.[/]")
        console.print("Run: [bold]memforge auth teams[/] to authenticate.")
        return

    expiry = TeamsAuthenticator.check_token_expiry(tokens)

    from rich.table import Table as RichTable

    table = RichTable(title="Teams Authentication")
    table.add_column("Audience")
    table.add_column("Status")
    table.add_column("Expires")

    from datetime import datetime, timezone

    for audience, valid in expiry.items():
        info = tokens.get(audience, {})
        expires_at = info.get("expiresAt", 0) if isinstance(info, dict) else 0
        if expires_at:
            exp_str = datetime.fromtimestamp(expires_at, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        else:
            exp_str = "unknown"
        status = "[green]valid[/]" if valid else "[red]expired[/]"
        table.add_row(audience, status, exp_str)

    console.print(table)


# ---------------------------------------------------------------------------
# maintenance group
# ---------------------------------------------------------------------------


@cli.group()
def maintenance():
    """Inspect and repair local storage/index consistency."""
    pass


@maintenance.command("repair-indexes")
@click.pass_context
def maintenance_repair_indexes(ctx):
    """Repair FTS5 and Chroma indexes from SQLite.

    Vectors are repaired in the index in use, with the embedding model it was
    built with. A change of embedding model is not a repair: see
    ``maintenance reindex``.
    """

    async def _run():
        from memforge.memory.health import MemoryIndexHealthChecker
        from memforge.memory.repair import MemoryIndexRepairer
        from memforge.model_settings import ModelSettings
        from memforge.runtime import (
            DefaultRuntimeProvider,
            build_active_embedding_runtime,
            memory_vector_indexes,
        )
        from memforge.vector_index_errors import ReindexInProgress

        config: AppConfig = ctx.obj["config"]
        provider = DefaultRuntimeProvider()
        db = await _get_db(config)
        try:
            # An installation with no index in use gets its first one here,
            # when its embedding model is known to answer.
            await ModelSettings(config, provider).vector_index(db)
            indexes = memory_vector_indexes(provider, db, config)
            index = None if indexes is None else await indexes.in_use()
            memory_collection = None if index is None else index.collection
            try:
                result = await MemoryIndexRepairer(
                    db=db,
                    memory_collection=memory_collection,
                    embedding=await build_active_embedding_runtime(db, config, provider),
                    index_fingerprint=None if index is None else index.fingerprint,
                ).repair()
            except ReindexInProgress as exc:
                console.print(f"[red]{exc.message}[/]")
                raise SystemExit(1) from exc
            report = await MemoryIndexHealthChecker(
                db=db,
                memory_collection=memory_collection,
            ).check()
        finally:
            await db.close()

        table = Table(title="Index Repair")
        table.add_column("Action")
        table.add_column("Count", justify="right")
        table.add_row("FTS rows rebuilt", str(result.fts_rows_rebuilt))
        table.add_row("FTS rows deleted", str(result.fts_rows_deleted))
        table.add_row("Memory vectors repaired", str(result.memory_vectors_repaired))
        table.add_row("Memory vectors deleted", str(result.memory_vectors_deleted))
        table.add_row("Unrepaired memories", str(len(result.unrepaired_memories)))
        console.print(table)
        if index is None:
            console.print(
                "[yellow]No Memory vector index is in use, so no vector was repaired. "
                "Run `memforge maintenance reindex --check`.[/]"
            )
        if report.ok:
            console.print("[green]Index health is clean.[/]")
        else:
            console.print(f"[yellow]{len(report.issues)} consistency issue(s) remain.[/]")
            for memory_id in result.unrepaired_memories[:10]:
                console.print(f"  [dim]Unrepaired memory: {memory_id}[/]")

    asyncio.run(_run())


class _Termination:
    """Whether the process was asked to terminate while a command ran."""

    requested = False


@contextlib.contextmanager
def _cancelling_on_sigterm() -> Iterator[_Termination]:
    """Cancel the running task when the process is asked to terminate, as Ctrl-C does.

    What the task is doing then ends the way cancelled work ends, with its
    clean-up, instead of being cut off with the process. Signals reach a
    command only in the main thread of a POSIX process; anywhere else the
    block runs without this.
    """
    termination = _Termination()
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    assert task is not None

    def cancel() -> None:
        termination.requested = True
        task.cancel()

    try:
        loop.add_signal_handler(signal.SIGTERM, cancel)
    except (NotImplementedError, RuntimeError, ValueError):
        yield termination
        return
    try:
        yield termination
    finally:
        loop.remove_signal_handler(signal.SIGTERM)


_VECTOR_INDEX_STATUS_TEXT = {
    "ready": "[green]ready[/]: built with the embedding model in use",
    "reindex_required": "[red]reindex required[/]: vector search is off until the index is rebuilt",
    "reindexing": "[yellow]a rebuild is running[/]",
    "awaiting_validation": "[yellow]waiting[/]: the embedding model has not passed a check yet",
    "not_configured": "[dim]no embedding model is configured[/]",
}


@maintenance.command("reindex")
@click.option("--check", "mode", flag_value="check", default=True, help="Report the index and what a rebuild would do.")
@click.option("--apply", "mode", flag_value="apply", help="Rebuild the index and switch to it.")
@click.pass_context
def maintenance_reindex(ctx, mode: str):
    """Rebuild the Memory vector index for the embedding model.

    A rebuild embeds every searchable Memory into a new index, verifies it and
    switches to it; the previous index is deleted afterwards. It is for an
    embedding candidate that passed its check and waits for the index, or for
    the embedding model in use when the index was built with another one.
    While it runs no Source syncs and no Memory is changed; search keeps
    answering from the previous index.

    --check, the default, changes nothing and asks no model. --apply checks
    an embedding model in use that has not passed a check yet before it
    rebuilds for it. --apply exits with status 1 when the rebuild is refused,
    fails or is interrupted; the index in use is then the one it was.
    """

    async def _run() -> int:
        from memforge.llm.providers import ModelConfigError
        from memforge.model_settings import ModelSettings
        from memforge.runtime import DefaultRuntimeProvider

        config: AppConfig = ctx.obj["config"]
        settings = ModelSettings(config, DefaultRuntimeProvider())
        db = await _get_db(config)
        try:
            view = await settings.vector_index(db)
            searchable = await db.count_searchable_memories()
            in_use = view.in_use
            table = Table(title="Memory vector index", show_header=False)
            table.add_column("Setting", style="bold")
            table.add_column("Value")
            table.add_row("Status", _VECTOR_INDEX_STATUS_TEXT[view.status])
            if in_use is None:
                table.add_row("Index in use", "[dim]none[/]")
            else:
                model = (in_use["profile"] or {}).get("model_id", "unknown")
                table.add_row(
                    "Index in use",
                    f"{in_use['collection_name']} ({model}, {in_use['dimension']} dimensions"
                    + (", adopted" if in_use["adopted"] else "")
                    + ")",
                )
            table.add_row("Searchable memories", str(searchable))
            table.add_row(
                "A rebuild would be for",
                {
                    "candidate": "the embedding candidate that waits for the index",
                    "active": "the embedding model in use",
                    None: "[dim]nothing: no rebuild can start[/]",
                }[view.reindex_target],
            )
            console.print(table)
            if view.status == "ready" and view.reindex_target == "active":
                console.print(
                    "The index in use was adopted: its vectors were in place before it was recorded, "
                    "and the model they came from is assumed. A rebuild is possible, not required."
                )
            if mode == "check":
                if view.reindex_target is not None:
                    console.print("Run `memforge maintenance reindex --apply` to rebuild the index.")
                if view.status == "awaiting_validation" and view.reindex_target == "active":
                    console.print(
                        "--apply checks the embedding model first: stored vectors of its length "
                        "can be adopted as they are, and are rebuilt on request."
                    )
                return 0

            def progress(indexed: int, total: int) -> None:
                console.print(f"  [dim]{indexed}/{total} memories embedded[/]")

            with _cancelling_on_sigterm() as termination:
                try:
                    finished = await settings.run_reindex(db, progress=progress)
                except ModelConfigError as exc:
                    console.print(f"[red]The rebuild did not start ({exc.code}): {exc.message}[/]")
                    return 1
                except asyncio.CancelledError:
                    if not termination.requested:
                        raise
                    console.print(
                        "[red]The rebuild was interrupted. "
                        "The index in use is unchanged; run the command again.[/]"
                    )
                    return 1
            if finished["state"] != "active":
                console.print(
                    f"[red]The rebuild failed ({finished['error_code']}). "
                    "The index in use is unchanged; run the command again.[/]"
                )
                return 1
            console.print(
                f"[green]The index was rebuilt: {finished['vector_count']} vectors "
                f"in {finished['collection_name']}.[/]"
            )
            return 0
        finally:
            await db.close()

    raise SystemExit(asyncio.run(_run()))


# ---------------------------------------------------------------------------
# backup group
# ---------------------------------------------------------------------------


@cli.group("backup")
def backup():
    """List and restore the recovery points of the local data.

    MemForge takes a recovery point of the database and the vector index
    before a new release upgrades them, and keeps the newest one of each of
    the three most recent schema versions. Restoring one puts the data back
    as it was before that upgrade.
    """
    pass


def _byte_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    value = size / 1024
    for unit in ("KB", "MB"):
        if value < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


@backup.command("list")
@click.pass_context
def backup_list(ctx):
    """List the recovery points, newest first."""
    from memforge.storage.database import latest_schema_version
    from memforge.storage.recovery_points import RecoveryPoints

    points = RecoveryPoints.for_config(ctx.obj["config"])
    found = points.list()
    if not found:
        console.print(f"No recovery points in {points.directory}.")
        return
    readable = latest_schema_version()
    table = Table(title=f"Recovery points in {points.directory}")
    table.add_column("ID", style="bold")
    table.add_column("Taken")
    table.add_column("Schema", justify="right")
    table.add_column("Taken by")
    table.add_column("Vectors")
    table.add_column("Size", justify="right")
    for point in found:
        schema = str(point.schema_version)
        if point.schema_version > readable:
            schema += " [dim](newer release)[/]"
        table.add_row(
            point.id,
            point.created_at.astimezone().strftime("%Y-%m-%d %H:%M"),
            schema,
            point.taken_by_release,
            "yes" if point.includes_vectors else "no",
            _byte_size(point.size_bytes()),
        )
    console.print(table)
    console.print(f"This release reads schema {readable} and older. Restore one with `memforge backup restore ID`.")


@backup.command("restore")
@click.argument("point_id", metavar="ID")
@click.option("--yes", is_flag=True, help="Restore without asking for confirmation.")
@click.pass_context
def backup_restore(ctx, point_id: str, yes: bool):
    """Put the database and vector index of recovery point ID back in place.

    The data being replaced is first kept as a new recovery point, so a
    restore can itself be undone. Stop the MemForge server first: the restore
    is refused while any MemForge process has the data open. Source documents
    are left as they are. The next start of a release newer than the restored
    data upgrades it again.
    """
    from memforge.storage.database import latest_schema_version
    from memforge.storage.recovery_points import RecoveryPoints, UpgradeError

    points = RecoveryPoints.for_config(ctx.obj["config"])
    try:
        point = points.get(point_id)
        if not yes:
            click.confirm(
                f"Replace the database and vector index in {ctx.obj['config'].base_dir} with recovery point "
                f"{point.id} (schema {point.schema_version}, taken {point.created_at.astimezone():%Y-%m-%d %H:%M})?",
                abort=True,
            )
        result = points.restore(point.id, readable_schema_version=latest_schema_version())
    except UpgradeError as exc:
        raise click.ClickException(str(exc)) from exc
    console.print(f"[green]Restored recovery point {result.restored.id}.[/]")
    if result.replaced is not None:
        console.print(f"The data it replaced is kept as recovery point {result.replaced.id}.")
    if result.restored.schema_version < latest_schema_version():
        console.print(
            f"The restored data has schema {result.restored.schema_version}; this release upgrades it to "
            f"schema {latest_schema_version()} when it next starts. Start the release you want to run."
        )


# ---------------------------------------------------------------------------
# config group
# ---------------------------------------------------------------------------


@cli.group("config")
def config_group():
    """View and manage configuration."""
    pass


async def _active_model_rows(config: AppConfig) -> dict[str, str]:
    """Describe the model each kind uses with the local database, without any key.

    The database holds the models saved in Settings. Before it exists, the
    models are the ones the MEMFORGE_* variables describe.
    """
    from memforge.llm.bindings import environment_binding, key_source, resolve_active_binding
    from memforge.llm.providers import MODEL_KINDS, provider_descriptor

    bindings = {}
    if Path(config.storage.db_path).exists():
        db = await _get_db(config)
        try:
            for kind in MODEL_KINDS:
                bindings[kind] = await resolve_active_binding(db, config, kind)
        finally:
            await db.close()
    else:
        source = "environment" if config.server.llm_config_writable else "deployment"
        for kind in MODEL_KINDS:
            bindings[kind] = environment_binding(config, kind, config_source=source)

    rows: dict[str, str] = {}
    for kind, binding in bindings.items():
        title = kind.capitalize()
        if binding is None:
            rows[f"{title} model"] = "[red]NOT CONFIGURED[/]"
            continue
        credential = {
            "api_key": "API key from the environment"
            if key_source(binding) == "environment"
            else "API key saved in Settings",
            "environment": "provider credentials from the environment",
            "none": "no credential",
        }[binding.auth_strategy]
        rows[f"{title} model"] = f"{binding.model_id} ({provider_descriptor(binding.provider).label})"
        rows[f"{title} endpoint"] = binding.endpoint or "[dim]set by the route[/]"
        rows[f"{title} credential"] = credential
        rows[f"{title} status"] = f"{binding.state.replace('_', ' ')}, from {binding.config_source}"
    return rows


@config_group.command("show")
@click.pass_context
def config_show(ctx):
    """Display current configuration."""
    config: AppConfig = ctx.obj["config"]

    console.print("\n[bold]MemForge Configuration[/]\n")

    info = {
        "Base directory": str(config.base_dir),
        "Database": config.storage.db_path,
        "ChromaDB": config.storage.chroma_path,
        "Documents": config.storage.docs_path,
        **asyncio.run(_active_model_rows(config)),
        "Decision model": config.llm.decision_model or "[dim]generation model[/]",
        "Dedup threshold": str(config.memory.dedup_cosine_threshold),
        "Default top_k": str(config.retrieval.default_top_k),
        "RRF k": str(config.retrieval.rrf_k),
        "Recency half-life": f"{config.retrieval.recency_half_life_days} days",
    }

    table = Table(show_header=False)
    table.add_column("Setting", style="bold")
    table.add_column("Value")
    for key, val in info.items():
        table.add_row(key, val)
    console.print(table)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    cli()
