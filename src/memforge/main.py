"""CLI entry point for MemForge."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from itertools import islice
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

from memforge.api_target import MemForgeTarget, TargetConfigurationError, build_host_target, build_target
from memforge.auth import browser_session
from memforge.config import AppConfig, load_config
from memforge.github_repo_utils import (
    DEFAULT_INCLUDE_EXTENSION_LIST,
    decode_github_base64_content,
    github_content_type,
    github_exclude_paths,
    github_extension,
    github_include_extensions,
    github_include_paths,
    github_path_in_scope,
    parse_github_repo_url,
)
from memforge.local_agent.folder_picker import FolderPickerCancelled, FolderPickerUnavailable, pick_folder
from memforge.local_agent.source_contract import local_agent_sync_snapshot_id
from memforge.sync_progress import normalize_sync_progress_snapshot
from memforge.storage.admin_source import (
    SOURCE_SYNC_SCHEDULE_MAX_INTERVAL_MINUTES,
    SOURCE_SYNC_SCHEDULE_MIN_INTERVAL_MINUTES,
)
from memforge.tool_client import ToolClient

console = Console()
log_console = Console(stderr=True)
DEFAULT_CLI_CONFIG_PATH = Path.home() / ".memforge" / "cli.toml"
DEFAULT_LOCAL_AGENT_STATE_PATH = Path.home() / ".memforge" / "local-agent-state.json"
DEFAULT_LOCAL_AGENT_LOCK_PATH = Path.home() / ".memforge" / "local-agent-daemon.lock"
DEFAULT_TEAMS_AUDIT_LOG_PATH = Path.home() / ".memforge" / "teams-sync-audit.jsonl"
DEFAULT_TEAMS_LEDGER_STATE_PATH = Path.home() / ".memforge" / "teams-ledger-state.json"
DEFAULT_KB_INCLUDE = [
    "*.md", "**/*.md",
    "*.markdown", "**/*.markdown",
    "*.txt", "**/*.txt",
    "*.json", "**/*.json",
    "*.html", "**/*.html",
    "*.htm", "**/*.htm",
]
DEFAULT_KB_EXCLUDE = [".obsidian/**", ".trash/**", ".git/**", "**/.git/**"]
LOCAL_MARKDOWN_SOURCE_TYPE = "local_markdown"
GITHUB_REPO_SOURCE_TYPE = "github_repo"
JIRA_SOURCE_TYPE = "jira"
DEFAULT_GITHUB_INCLUDE_EXTENSIONS = DEFAULT_INCLUDE_EXTENSION_LIST
# Watch defaults. The tick interval is deliberately shorter than a typical Jira
# idle-session timeout so the stored copy is renewed while it is still valid.
WATCH_DEFAULT_INTERVAL_SECONDS = 1800  # 30 minutes
WATCH_BACKOFF_BASE_SECONDS = 5
WATCH_BACKOFF_MAX_SECONDS = 300  # 5 minutes
INTERACTIVE_DISABLE_ENV = "MEMFORGE_NO_INTERACTIVE"
INTERACTIVE_SCRIPT_ENV = "MEMFORGE_INTERACTIVE_SCRIPT"
INTERACTIVE_BIN_ENV = "MEMFORGE_CLI_BIN"
INTERACTIVE_CACHE_ENV = "MEMFORGE_INTERACTIVE_CACHE"
INTERACTIVE_RESOURCE_DIR = "interactive_cli"
INTERACTIVE_INSTALL_LOCK = ".install.lock"
INTERACTIVE_INSTALL_LOCK_TIMEOUT_SECONDS = 120
INTERACTIVE_INSTALL_LOCK_STALE_SECONDS = 600
INTERACTIVE_DEPENDENCY_SENTINEL = Path("node_modules") / "@clack" / "prompts"


@dataclass(frozen=True)
class _ResolvedCliTarget:
    target: MemForgeTarget
    api_token: str | None
    active_target: str
    token_env: str


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=log_console, rich_tracebacks=True)],
        force=True,
    )


async def _get_db(config: AppConfig):
    from memforge.storage.database import Database

    db = Database(config.storage.db_path)
    await db.connect()
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
    path.parent.mkdir(parents=True, exist_ok=True)
    targets = data.get("targets") if isinstance(data.get("targets"), dict) else {}
    lines = [f"active = {_toml_string(str(data.get('active') or ''))}", ""]
    for name, target in sorted(targets.items()):
        if not isinstance(target, dict):
            continue
        lines.append(f"[targets.{_toml_key(str(name))}]")
        lines.append(f"api_url = {_toml_string(str(target.get('api_url') or ''))}")
        if target.get("workspace_id"):
            lines.append(f"workspace_id = {_toml_string(str(target['workspace_id']))}")
        if target.get("token_env"):
            lines.append(f"token_env = {_toml_string(str(target['token_env']))}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o600)


def _toml_string(value: str) -> str:
    return json.dumps(value)


def _toml_key(value: str) -> str:
    return json.dumps(value)


def _resolve_api_target(
    _config: AppConfig,
    *,
    allow_cloud_without_workspace: bool = False,
) -> _ResolvedCliTarget:
    target_env_names = ("MEMFORGE_API_URL", "MEMFORGE_WORKSPACE_ID")
    if any(os.getenv(name, "").strip() for name in target_env_names):
        target = _build_cli_target(
            api_url=os.getenv("MEMFORGE_API_URL"),
            workspace_id=os.getenv("MEMFORGE_WORKSPACE_ID"),
            allow_cloud_without_workspace=allow_cloud_without_workspace,
        )
        return _ResolvedCliTarget(
            target=target,
            api_token=os.getenv("MEMFORGE_API_TOKEN"),
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
            workspace_id=profile.get("workspace_id"),
            allow_cloud_without_workspace=allow_cloud_without_workspace,
        )
        token_env = str(profile.get("token_env") or "")
        return _ResolvedCliTarget(
            target=target,
            api_token=os.getenv(token_env) if token_env else None,
            active_target=active,
            token_env=token_env,
        )

    return _ResolvedCliTarget(
        target=build_target(origin=None, workspace_id=None),
        api_token=os.getenv("MEMFORGE_API_TOKEN"),
        active_target="",
        token_env="",
    )


def _build_cli_target(
    *,
    api_url: object,
    workspace_id: object,
    allow_cloud_without_workspace: bool = False,
) -> MemForgeTarget:
    try:
        return build_target(
            origin=str(api_url) if api_url is not None else None,
            workspace_id=str(workspace_id) if workspace_id is not None else None,
        )
    except TargetConfigurationError as exc:
        if allow_cloud_without_workspace and exc.code == "cloud_workspace_required":
            return build_host_target(origin=str(api_url) if api_url is not None else None)
        raise click.ClickException(exc.code) from exc


def _tool_client(ctx) -> ToolClient:
    resolved = _resolve_api_target(ctx.obj["config"])
    return ToolClient(target=resolved.target, api_token=resolved.api_token)


def _local_agent_tool_client(ctx) -> ToolClient:
    resolved = _resolve_api_target(
        ctx.obj["config"],
        allow_cloud_without_workspace=True,
    )
    return ToolClient(target=resolved.target, api_token=resolved.api_token)


def _emit_tool_payload(ctx, payload: dict) -> None:
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
        cloud_jobs_provider=lambda wait_seconds=0, lease_seconds=60: client.lease_local_agent_jobs(
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
        cloud_job_heartbeat=lambda job_id, attempt_count, lease_seconds, progress=None: client.heartbeat_local_agent_job(
            job_id,
            attempt_count=attempt_count,
            lease_seconds=lease_seconds,
            progress=progress,
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
        raise click.ClickException(f"GitHub CLI request failed: {detail}")
    try:
        payload = json.loads(result.stdout or "{}")
    except ValueError as exc:
        raise click.ClickException("GitHub CLI returned invalid JSON.") from exc
    return payload if isinstance(payload, dict) else {}


def _github_tree(repo: dict[str, str], ref: str) -> list[dict[str, Any]]:
    payload = _gh_api_json(
        repo,
        f"repos/{repo['owner']}/{repo['repo']}/git/trees/{quote(ref, safe='')}?recursive=1",
    )
    if payload.get("truncated") is True:
        raise click.ClickException(
            "GitHub tree response was truncated; use Internal network / VPN access for this large repository."
        )
    tree = payload.get("tree")
    return tree if isinstance(tree, list) else []


def _github_content(repo: dict[str, str], ref: str, relative_path: str) -> bytes:
    payload = _gh_api_json(
        repo,
        f"repos/{repo['owner']}/{repo['repo']}/contents/{quote(relative_path, safe='/')}?ref={quote(ref, safe='')}",
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


def _preview_github_profile(name: str, profile: dict[str, Any], *, limit: int | None) -> dict[str, Any]:
    repo, ref, include_paths, exclude_paths, include_extensions = _resolve_github_profile(name, profile)
    counts = {"included": 0, "ignored": 0}
    extension_counts: dict[str, int] = {}
    items: list[dict[str, Any]] = []
    tree = _github_tree(repo, ref)
    for entry in tree:
        if entry.get("type") != "blob":
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
            items.append(
                {
                    "relative_path": relative_path,
                    "blob_sha": entry.get("sha"),
                    "bytes": entry.get("size", 0),
                    "content_type": github_content_type(relative_path),
                }
            )
    payload = {
        "profile": name,
        "repo_url": repo["repo_url"],
        "ref": ref,
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
    "--config", "config_path", type=click.Path(exists=True, path_type=Path), default=None,
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
            f"retrieval eval {case_set}: "
            f"{summary['case_count']} cases, {summary['hard_failures']} hard failures"
        )
        for failure in payload["hard_failures"]:
            click.echo(f"- {failure['case_id']}: {failure['message']}")
    return 1 if fail_on_hard_failure and report.hard_failures else 0


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
            "[yellow]Interactive UI requires Node.js (>=18) on PATH.[/]\n"
            "Install Node, then re-run [bold]memforge[/]."
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
        name
        for name in ("index.mjs", "package.json", "package-lock.json")
        if not (resource_dir / name).exists()
    ]
    if missing:
        raise RuntimeError(
            "Interactive UI package is incomplete: missing "
            + ", ".join(missing)
            + f" under {resource_dir}."
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
        package_version = metadata.version("memforge")
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
@click.option("--workspace-id", default=None, help="Required workspace ID for Cloud targets.")
@click.option("--token-env", default="MEMFORGE_API_TOKEN", show_default=True, help="Environment variable for the token.")
@click.pass_context
def target_add(ctx, name: str, api_url: str, workspace_id: str | None, token_env: str):
    """Add or update an API target and make it active."""
    name = name.strip()
    if not name:
        raise click.ClickException("Target name is required.")
    resolved = _build_cli_target(api_url=api_url, workspace_id=workspace_id)
    data = _read_cli_config()
    targets = data.setdefault("targets", {})
    targets[name] = {
        "api_url": resolved.origin,
        "workspace_id": resolved.workspace_id,
        "token_env": token_env.strip(),
    }
    data["active"] = name
    _write_cli_config(data)
    _emit_tool_payload(
        ctx,
        {
            "ok": True,
            "active": name,
            "edition": resolved.edition.value,
            "api_url": resolved.origin,
            "workspace_id": resolved.workspace_id,
        },
    )


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
    payload = _tool_client(ctx).health()
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
        from memforge.storage.database import Database
        from memforge.runtime import build_sync_runtime, run_source_sync

        db = Database(config.storage.db_path)
        await db.connect()
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

            console.print(f"\n[bold]Syncing: {src['name']}[/] (type={source_type})")

            try:
                state = await run_source_sync(
                    db=db,
                    config=config,
                    source=src,
                    runtime=runtime,
                    progress_callback=lambda p: (
                        console.print(f"  [dim]{p.get('status', '')}[/]")
                        if p.get("status") else None
                    ),
                )
                status_color = (
                    "green" if state.last_sync_status == "success"
                    else "yellow" if state.last_sync_status == "partial"
                    else "red"
                )
                console.print(
                    f"  [{status_color}]Done:[/] {state.docs_processed} discovered, "
                    f"{state.docs_updated} updated, {state.docs_failed} failed, "
                    f"{state.memories_extracted} memories extracted"
                )
            except Exception as e:
                console.print(f"  [red]Sync failed: {e}[/]")

        await db.close()
        console.print("\n[bold green]Sync complete![/]")

    asyncio.run(_run())


@cli.command("search")
@click.argument("query", required=False, default="")
@click.option("--top-k", default=10, show_default=True, type=int, help="Maximum number of results.")
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
@click.pass_context
def search(
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
):
    """Search MemForge using the same service path as the MCP search tool."""
    time_range = (
        {
            k: v
            for k, v in {"date_type": date_type, "start_date": start_date, "end_date": end_date}.items()
            if v
        }
        if start_date or end_date
        else {}
    )
    kwargs: dict = {
        "query": query,
        "top_k": top_k,
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
    payload = _tool_client(ctx).search(**kwargs)
    _emit_tool_payload(ctx, payload)


@cli.command("get-memory")
@click.argument("memory_id")
@click.pass_context
def get_memory(ctx, memory_id: str):
    """Fetch full memory detail and provenance by memory ID."""
    payload = _tool_client(ctx).get_memory(memory_id)
    _emit_tool_payload(ctx, payload)


@cli.command("get-resource")
@click.argument("url")
@click.option("--mode", default="text", show_default=True, type=click.Choice(["text", "file", "base64"]))
@click.option("--max-chars", default=120_000, show_default=True, type=int, help="Maximum text characters to print.")
@click.option("--max-bytes", default=2_000_000, show_default=True, type=int, help="Maximum bytes for inline modes.")
@click.pass_context
def get_resource(ctx, url: str, mode: str, max_chars: int, max_bytes: int):
    """Fetch a source artifact URL returned by get-memory."""
    payload = _tool_client(ctx).get_resource(
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
@click.option("--top-k", default=10, show_default=True, type=int, help="Maximum number of results.")
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
):
    """Search MemForge memories."""
    time_range = (
        {
            k: v
            for k, v in {"date_type": date_type, "start_date": start_date, "end_date": end_date}.items()
            if v
        }
        if start_date or end_date
        else {}
    )
    kwargs: dict = {
        "query": query,
        "top_k": top_k,
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
    payload = _tool_client(ctx).search(**kwargs)
    _emit_tool_payload(ctx, payload)


@memory.command("get")
@click.argument("memory_id")
@click.pass_context
def memory_get(ctx, memory_id: str):
    """Fetch memory detail and provenance."""
    payload = _tool_client(ctx).get_memory(memory_id)
    _emit_tool_payload(ctx, payload)


@memory.command("resource")
@click.argument("url")
@click.option("--mode", default="text", show_default=True, type=click.Choice(["text", "file", "base64"]))
@click.option("--max-chars", default=120_000, show_default=True, type=int, help="Maximum text characters to print.")
@click.option("--max-bytes", default=2_000_000, show_default=True, type=int, help="Maximum bytes for inline modes.")
@click.pass_context
def memory_resource(ctx, url: str, mode: str, max_chars: int, max_bytes: int):
    """Fetch a source artifact URL returned by get-memory."""
    payload = _tool_client(ctx).get_resource(
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
@click.pass_context
def sources_list(ctx):
    """List all configured sources for the active API target."""
    payload = _tool_client(ctx).list_sources()
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
@click.pass_context
def sources_searchable(ctx):
    """List source IDs available for memory search filtering."""
    payload = _tool_client(ctx).list_searchable_sources()
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
@click.pass_context
def sources_schedule(ctx, source_id: str, every_minutes: int, disable: bool):
    """Configure automatic sync for one source over the active API target."""
    if every_minutes < SOURCE_SYNC_SCHEDULE_MIN_INTERVAL_MINUTES:
        raise click.ClickException(
            f"--every-minutes must be at least {SOURCE_SYNC_SCHEDULE_MIN_INTERVAL_MINUTES}."
        )
    if every_minutes > SOURCE_SYNC_SCHEDULE_MAX_INTERVAL_MINUTES:
        raise click.ClickException(
            f"--every-minutes must be at most {SOURCE_SYNC_SCHEDULE_MAX_INTERVAL_MINUTES}."
        )
    client = _tool_client(ctx)
    payload = client.update_source_schedule(
        source_id=source_id,
        enabled=not disable,
        interval_minutes=every_minutes,
    )
    _emit_tool_payload(ctx, payload)


@sources.command("schedule-show")
@click.argument("source_id")
@click.pass_context
def sources_schedule_show(ctx, source_id: str):
    """Show automatic sync schedule for one source over the active API target."""
    payload = _tool_client(ctx).get_source_schedule(source_id)
    _emit_tool_payload(ctx, payload)


# ---------------------------------------------------------------------------
# memories group
# ---------------------------------------------------------------------------


@cli.group()
def memories():
    """Browse and inspect memories."""
    pass


@memories.command("list")
@click.option("--type", "memory_type", default=None, help="Filter by memory type (fact, decision, convention, procedure)")
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
        table.add_column("Confidence", justify="right", max_width=6)
        table.add_column("Corr.", justify="right", max_width=5)
        table.add_column("Status", max_width=10)
        table.add_column("Updated", max_width=20)

        for m in mems:
            content_preview = m.content[:80] + "..." if len(m.content) > 80 else m.content
            table.add_row(
                m.id,
                m.memory_type,
                content_preview,
                f"{m.confidence:.2f}",
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
    _emit_tool_payload(
        ctx,
        payload,
    )


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
    resolved = _resolve_api_target(config, allow_cloud_without_workspace=True)
    return {
        "edition": resolved.target.edition.value,
        "api_url": resolved.target.origin,
        "workspace_id": resolved.target.workspace_id,
        "active_target": resolved.active_target,
        "token_env": resolved.token_env,
        "api_token_configured": bool(resolved.api_token),
    }


def _local_agent_status_recommendations(target: dict[str, Any]) -> list[str]:
    recommendations: list[str] = []
    token_env = str(target.get("token_env") or "MEMFORGE_API_TOKEN")
    if not target.get("api_token_configured"):
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
        _compact_local_agent_task(task_id, task)
        for task_id, task in tasks.items()
        if isinstance(task, dict)
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
@click.option("--interval-seconds", "poll_interval_seconds", default=10, show_default=True, type=int,
              help="Seconds between server-job lease polls.")
@click.option("--cloud-job-wait-seconds", default=25, show_default=True, type=int,
              help="Long-poll wait for Cloud-triggered local-agent jobs.")
@click.pass_context
def adapter_daemon_run(
    ctx,
    browser: str | None,
    poll_interval_seconds: int,
    cloud_job_wait_seconds: int,
):
    """Run the local agent daemon until interrupted."""
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
    source_id = source_id.strip()
    if not source_id:
        raise click.ClickException("source_id is required")

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

    pushed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    prepared: list[dict[str, Any]] = []
    for entry in selected_entries:
        relative_path = str(entry["relative_path"])
        try:
            raw = _github_content(repo, ref, relative_path)
            text_body = raw.decode("utf-8")
        except UnicodeDecodeError:
            failed.append({"relative_path": relative_path, "error": "invalid utf-8"})
            continue
        except click.ClickException as exc:
            failed.append({"relative_path": relative_path, "error": str(exc)})
            continue
        raw_hash = hashlib.sha256(raw).hexdigest()
        prepared.append(
            {
                "relative_path": relative_path,
                "markdown_body": text_body,
                "content_type": str(entry.get("content_type") or "text/plain"),
                "title": _github_title(text_body, relative_path),
                "raw_hash": raw_hash,
                "blob_sha": str(entry.get("blob_sha") or ""),
            }
        )

    for index, doc in enumerate(prepared, start=1):
        response = client.push_github_repo_document(
            source_id=source_id,
            repo_url=repo["repo_url"],
            repo_ref=ref,
            relative_path=str(doc["relative_path"]),
            markdown_body=str(doc["markdown_body"]),
            content_type=str(doc["content_type"]),
            title=str(doc["title"]),
            raw_hash=str(doc["raw_hash"]),
            blob_sha=str(doc["blob_sha"]),
            sync_snapshot_id=sync_snapshot_id,
            local_agent_job_id=local_agent_job_id,
            local_agent_attempt_count=local_agent_attempt_count,
            submitted_by=submitted_by,
        )
        if isinstance(response, dict) and response.get("error"):
            failed.append({"relative_path": doc["relative_path"], "error": response.get("error"),
                           "detail": response.get("detail"), "status_code": response.get("status_code")})
        else:
            pushed.append({"relative_path": doc["relative_path"],
                           "doc_id": response.get("doc_id"),
                           "document_hash": response.get("document_hash")})
        _report_local_agent_progress(
            report_progress,
            _sync_progress_snapshot(
                phase="uploading",
                completed=index,
                total=len(prepared),
                unit="file",
                failed=len(failed),
            ),
        )

    payload = {
        "profile": name,
        "repo_url": repo["repo_url"],
        "ref": ref,
        "source_id": source_id,
        "counts": {"selected": len(selected_entries), "pushed": len(pushed), "failed": len(failed)},
        "pushed": pushed,
        "failed": failed,
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
        "local_markdown_sync": lambda: _run_cloud_local_markdown_sync_job(
            job, client, report_progress=report_progress
        ),
        "jira_sync": lambda: _run_cloud_jira_sync_job(
            job, client, browser=browser, report_progress=report_progress
        ),
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
        return {"operation": operation, "error": f"unsupported cloud local-agent job operation: {operation or '<empty>'}"}
    return handler()


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
    source_id, workspace_id, error = _cloud_job_source_scope(job, payload, operation=operation)
    if error:
        return error
    client = client.for_workspace(workspace_id)
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
    source_id, workspace_id, error = _cloud_job_source_scope(job, payload, operation=operation)
    if error:
        return error
    client = client.for_workspace(workspace_id)
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
    source_id, workspace_id, error = _cloud_job_source_scope(job, payload, operation=operation)
    if error:
        return error
    client = client.for_workspace(workspace_id)
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
        if uploaded_session.get("status_code") == 409:
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
        documents = asyncio.run(
            _collect_jira_documents_from_cloud_job(
                job,
                source_id=source_id,
                jira_cookie=captured.cookie_header,
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
    sync_snapshot_id = local_agent_sync_snapshot_id(job.get("job_id"), job.get("attempt_count"))
    sync_result: dict[str, Any] | None = None
    for index, doc in enumerate(documents, start=1):
        response = scoped_client.push_jira_package(
            source_id=source_id,
            base_url=str(doc["base_url"]),
            issue_key=str(doc["issue_key"]),
            source_url=str(doc["source_url"]),
            raw_payload=doc.get("raw_payload") if isinstance(doc.get("raw_payload"), dict) else {},
            title=str(doc["title"]),
            raw_hash=str(doc["raw_hash"]),
            submitted_by=str(payload.get("submitted_by") or "memforge-local-agent"),
            sync_snapshot_id=sync_snapshot_id,
            local_agent_job_id=str(job["job_id"]),
            local_agent_attempt_count=int(job["attempt_count"]),
        )
        if isinstance(response, dict) and response.get("error"):
            failed.append({"issue_key": doc["issue_key"], "error": response.get("error"),
                           "detail": response.get("detail"), "status_code": response.get("status_code")})
        else:
            pushed.append({"issue_key": doc["issue_key"],
                           "doc_id": response.get("doc_id"),
                           "document_hash": response.get("document_hash")})
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
        "counts": {"selected": len(documents), "pushed": len(pushed), "failed": len(failed)},
        "pushed": pushed,
        "failed": failed,
        "sync_started": bool(sync_result and not sync_result.get("error")),
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
    try:
        from memforge.genes.teams_gene import AuthenticationError
    except Exception:
        AuthenticationError = ()  # type: ignore[assignment]
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
    from memforge.local_agent.teams_audit import write_teams_audit_event
    from memforge.local_agent.teams_ledger import TeamsLedgerStateStore

    operation = str(job.get("operation") or "").strip()
    payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
    source_id, workspace_id, error = _cloud_job_source_scope(job, payload, operation=operation)
    if error:
        return error
    client = client.for_workspace(workspace_id)

    run_id = str(job.get("job_id") or f"teams-sync-{int(time.time())}")
    audit_path = Path(str(payload.get("audit_log_path") or DEFAULT_TEAMS_AUDIT_LOG_PATH))
    ledger_state_path = Path(str(payload.get("ledger_state_path") or DEFAULT_TEAMS_LEDGER_STATE_PATH))
    ledger_store = TeamsLedgerStateStore(ledger_state_path)
    limit = _cloud_job_limit(payload.get("limit"), default=0)
    _report_local_agent_progress(
        report_progress,
        _sync_progress_snapshot(phase="connecting"),
    )
    try:
        collection = asyncio.run(
            _collect_teams_documents_from_cloud_job(
                job,
                source_id=source_id,
                limit=limit,
                report_progress=report_progress,
            )
        )
    except Exception as exc:
        collect_error: Exception | None = exc
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
                    collection = asyncio.run(
                        _collect_teams_documents_from_cloud_job(
                            job,
                            source_id=source_id,
                            limit=limit,
                            report_progress=report_progress,
                        )
                    )
                    collect_error = None
                except Exception as retry_exc:
                    collect_error = retry_exc
            else:
                collect_error = RuntimeError(str(reauth.get("error") or "Teams authentication failed."))
        if collect_error is not None:
            write_teams_audit_event(
                audit_path,
                {
                    "event": "teams_sync_run",
                    "run_id": run_id,
                    "operation": operation,
                    "source_id": source_id,
                    "status": "collect_failed",
                    "error_type": type(collect_error).__name__,
                    "error": str(collect_error),
                },
            )
            return {
                "operation": operation,
                "source_id": source_id,
                "error": str(collect_error),
                "error_type": type(collect_error).__name__,
                "retryable": True,
            }

    documents, poll_audits = _teams_collection_documents_and_polls(collection)
    progress_summary = _teams_progress_summary(documents, poll_audits)
    for poll in poll_audits:
        event = {"event": "teams_conversation_poll", "run_id": run_id, "source_id": source_id, **poll}
        write_teams_audit_event(audit_path, event)

    pushed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    skipped_existing: list[dict[str, Any]] = []
    pending_receipts: list[dict[str, str | None]] = []
    scoped_client = client
    submitted_by = str(payload.get("submitted_by") or "memforge-local-agent")
    sync_started = False

    processed_messages = 0
    for doc in documents:
        current_source_time = doc.get("date_from") or doc.get("date_to") or doc.get("last_modified")
        _report_local_agent_progress(
            report_progress,
            _sync_progress_snapshot(
                phase="uploading",
                completed=processed_messages,
                total=progress_summary.get("messages", 0),
                unit="message",
                source_time_start=current_source_time,
                source_time_end=current_source_time,
                failed=len(failed),
            ),
        )
        processed_messages += _int_or_zero(doc.get("message_count"))
        window_id = str(doc["window_id"])
        revision_hash = str(doc["revision_hash"])
        window_id_hash = hashlib.sha256(window_id.encode("utf-8")).hexdigest()
        if ledger_store.has_receipt(source_id=source_id, window_id=window_id, revision_hash=revision_hash):
            skipped_existing.append({"window_id": window_id, "revision_hash": revision_hash})
            write_teams_audit_event(
                audit_path,
                {
                    "event": "teams_window_projection",
                    "run_id": run_id,
                    "source_id": source_id,
                    "raw_conversation_id": doc.get("conversation_id"),
                    "raw_root_message_id": doc.get("root_message_id"),
                    "window_id_hash": window_id_hash,
                    "window_type": doc.get("window_type"),
                    "revision_hash": revision_hash,
                    "receipt_status": "existing",
                    "receipt_skip_reason": "receipt_exists",
                    "message_count": doc.get("message_count"),
                },
            )
            continue
        write_teams_audit_event(
            audit_path,
            {
                "event": "teams_window_projection",
                "run_id": run_id,
                "source_id": source_id,
                "raw_conversation_id": doc.get("conversation_id"),
                "raw_root_message_id": doc.get("root_message_id"),
                "window_id_hash": window_id_hash,
                "window_type": doc.get("window_type"),
                "revision_hash": revision_hash,
                "receipt_status": "new",
                "message_count": doc.get("message_count"),
            },
        )
        response = scoped_client.push_teams_window_package(
            source_id=source_id,
            conversation_id=str(doc["conversation_id"]),
            root_message_id=str(doc.get("root_message_id") or ""),
            window_id=window_id,
            window_type=str(doc.get("window_type") or ""),
            revision_hash=revision_hash,
            raw_payload=doc.get("raw_payload") if isinstance(doc.get("raw_payload"), dict) else {},
            title=str(doc["title"]),
            source_url=str(doc.get("source_url") or ""),
            raw_hash=str(doc["raw_hash"]),
            submitted_by=submitted_by,
            local_agent_job_id=str(job["job_id"]),
            local_agent_attempt_count=int(job["attempt_count"]),
        )
        if isinstance(response, dict) and response.get("error"):
            failed.append({
                "window_id": window_id,
                "revision_hash": revision_hash,
                "error": response.get("error"),
                "detail": response.get("detail"),
                "status_code": response.get("status_code"),
            })
            write_teams_audit_event(
                audit_path,
                {
                    "event": "teams_memory_patch",
                    "run_id": run_id,
                    "source_id": source_id,
                    "window_id_hash": window_id_hash,
                    "revision_hash": revision_hash,
                    "patch_status": "failed",
                    "error": response.get("error"),
                    "status_code": response.get("status_code"),
                    "claim_add": 0,
                    "claim_update": 0,
                    "claim_supersede": 0,
                    "claim_noop": 0,
                    "claim_rejected_ambiguous": 0,
                },
            )
            continue
        pushed.append({
            "window_id": window_id,
            "revision_hash": revision_hash,
            "doc_id": response.get("doc_id") if isinstance(response, dict) else None,
            "document_hash": response.get("document_hash") if isinstance(response, dict) else None,
        })
        pending_receipts.append({
            "window_id": window_id,
            "revision_hash": revision_hash,
            "document_hash": response.get("document_hash") if isinstance(response, dict) else None,
        })
        write_teams_audit_event(
            audit_path,
            {
                "event": "teams_memory_patch",
                "run_id": run_id,
                "source_id": source_id,
                "window_id_hash": window_id_hash,
                "revision_hash": revision_hash,
                "patch_status": "pushed",
                "document_hash": response.get("document_hash") if isinstance(response, dict) else None,
                "claim_add": 0,
                "claim_update": 0,
                "claim_supersede": 0,
                "claim_noop": 0,
                "claim_rejected_ambiguous": 0,
            },
        )

    _report_local_agent_progress(
        report_progress,
        _sync_progress_snapshot(
            phase="uploading",
            completed=processed_messages,
            total=progress_summary.get("messages", 0),
            unit="message",
            source_time_start=progress_summary.get("date_from"),
            source_time_end=progress_summary.get("date_to"),
            failed=len(failed),
        ),
    )

    sync_result = None
    if pushed:
        # Teams is incremental by stable window id and revision. Processing the
        # historical input set lets the server collapse each window to its
        # latest revision; document-style authoritative snapshots do not apply.
        sync_result = scoped_client.start_source_processing(
            source_id=source_id,
            force_full_sync=bool(payload.get("force_full_sync", False)),
            local_agent_job_id=str(job["job_id"]),
            local_agent_attempt_count=int(job["attempt_count"]),
        )
        sync_started = not bool(sync_result.get("error"))
        if sync_started:
            for receipt in pending_receipts:
                ledger_store.record_receipt(
                    source_id=source_id,
                    window_id=str(receipt["window_id"]),
                    revision_hash=str(receipt["revision_hash"]),
                    document_hash=receipt["document_hash"],
                )

    result = {
        "operation": operation,
        "source_id": source_id,
        "counts": {
            "selected": len(documents),
            "pushed": len(pushed),
            "failed": len(failed),
            "skipped_existing": len(skipped_existing),
            "polls": len(poll_audits),
        },
        "messages": progress_summary.get("messages", 0),
        "conversations": progress_summary.get("conversations", 0),
        "date_from": progress_summary.get("date_from"),
        "date_to": progress_summary.get("date_to"),
        "pushed": pushed,
        "failed": failed,
        "skipped_existing": skipped_existing,
        "sync_started": sync_started,
        "audit_log_path": str(audit_path.expanduser()),
    }
    if failed:
        result["error"] = "one or more Teams windows failed to push"
    if sync_result and sync_result.get("error"):
        result["error"] = (
            "one or more Teams windows failed to push; source processing failed to start"
            if failed
            else "source processing failed to start"
        )
        result["sync_error"] = sync_result
    if result.get("error"):
        result["retryable"] = True

    write_teams_audit_event(
        audit_path,
        {
            "event": "teams_sync_run",
            "run_id": run_id,
            "operation": operation,
            "source_id": source_id,
            "status": "completed" if not result.get("error") else "completed_with_error",
            "selected_windows": len(documents),
            "pushed_windows": len(pushed),
            "failed_windows": len(failed),
            "skipped_existing_windows": len(skipped_existing),
            "polls": len(poll_audits),
            "sync_started": result["sync_started"],
            "sync_error": result.get("sync_error"),
            "claim_add": 0,
            "claim_update": 0,
            "claim_supersede": 0,
            "claim_noop": 0,
            "claim_rejected_ambiguous": 0,
        },
    )
    return result


def _teams_collection_documents_and_polls(collection: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if isinstance(collection, dict):
        documents = collection.get("documents")
        poll_audits = collection.get("poll_audits")
        return (
            documents if isinstance(documents, list) else [],
            poll_audits if isinstance(poll_audits, list) else [],
        )
    if isinstance(collection, list):
        return collection, []
    return [], []


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


def _cloud_job_source_scope(
    job: dict[str, Any],
    payload: dict[str, Any],
    *,
    operation: str,
) -> tuple[str, str, dict[str, Any] | None]:
    source_id = str(job.get("source_id") or payload.get("source_id") or "").strip()
    if not source_id:
        return "", "", {"operation": operation, "error": "source_id is required"}
    workspace_id = str(job.get("workspace_id") or payload.get("workspace_id") or "").strip()
    if not workspace_id:
        return source_id, "", {"operation": operation, "source_id": source_id, "error": "workspace_id is required"}
    return source_id, workspace_id, None


async def _collect_jira_documents_from_cloud_job(
    job: dict[str, Any],
    *,
    source_id: str,
    limit: int,
    jira_cookie: str,
    report_progress: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    from memforge.genes.jira_gene import JiraGene

    payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
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
        async for item in gene.discover(None):
            if limit and len(documents) >= limit:
                break
            raw = await gene.fetch(item)
            issue_key = str(item.extra.get("issue_key") or item.item_id.replace("jira-", "")).strip()
            raw_payload = json.loads(raw.body.decode("utf-8"))
            raw_hash = hashlib.sha256(raw.body).hexdigest()
            documents.append(
                {
                    "base_url": base_url,
                    "issue_key": issue_key,
                    "source_url": item.source_url,
                    "title": item.title,
                    "raw_payload": raw_payload,
                    "raw_hash": raw_hash,
                }
            )
            _report_local_agent_progress(
                report_progress,
                _sync_progress_snapshot(
                    phase="discovering",
                    completed=len(documents),
                    unit="issue",
                ),
            )
    finally:
        client = getattr(gene, "_client", None)
        if client is not None:
            await client.aclose()
    return documents


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


async def _collect_teams_documents_from_cloud_job(
    job: dict[str, Any],
    *,
    source_id: str,
    limit: int,
    report_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    from memforge.genes.teams_gene import TeamsGene

    payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
    config = _teams_direct_rest_config_from_cloud_payload(payload)
    config.pop("local_agent_documents_dir", None)
    config.pop("local_agent_package_manifest", None)
    config.pop("audit_log_path", None)
    config.setdefault("ledger_state_path", str(DEFAULT_TEAMS_LEDGER_STATE_PATH))
    gene = TeamsGene(config, source_id)
    await gene.authenticate()
    _report_local_agent_progress(
        report_progress,
        _sync_progress_snapshot(phase="discovering", completed=0, unit="message"),
    )
    documents: list[dict[str, Any]] = []
    poll_inputs: list[dict[str, Any]] = []
    try:
        async for item in gene.discover(None):
            if limit and len(documents) >= limit:
                break
            raw = await gene.fetch(item)
            raw_payload = json.loads(raw.body.decode("utf-8"))
            raw_hash = hashlib.sha256(raw.body).hexdigest()
            conversation_id = str(item.extra.get("conversation_id") or "")
            root_message_id = str(item.extra.get("root_message_id") or item.item_id)
            window_type = "thread" if item.extra.get("is_thread") else "time_block"
            window_id = str(item.extra.get("window_id") or _teams_window_id(
                source_id=source_id,
                conversation_id=conversation_id,
                root_message_id=root_message_id,
                window_type=window_type,
            ))
            revision_hash = str(item.version or hashlib.sha256(
                f"{window_id}\n{raw_hash}".encode("utf-8")
            ).hexdigest())
            messages = raw_payload.get("messages") if isinstance(raw_payload, dict) else []
            message_count = item.extra.get("message_count") or (len(messages) if isinstance(messages, list) else None)
            message_times = [
                str(message.get("time") or "")
                for message in messages
                if isinstance(message, dict) and message.get("time")
            ] if isinstance(messages, list) else []
            documents.append(
                {
                    "conversation_id": conversation_id,
                    "root_message_id": root_message_id,
                    "window_id": window_id,
                    "window_type": window_type,
                    "revision_hash": revision_hash,
                    "title": item.title,
                    "source_url": item.source_url,
                    "last_modified": item.last_modified.isoformat(),
                    "date_from": min(message_times) if message_times else item.last_modified.isoformat(),
                    "date_to": max(message_times) if message_times else item.last_modified.isoformat(),
                    "raw_payload": raw_payload,
                    "raw_hash": raw_hash,
                    "message_count": message_count,
                }
            )
            message_count_value = sum(_int_or_zero(doc.get("message_count")) for doc in documents)
            _report_local_agent_progress(
                report_progress,
                _sync_progress_snapshot(
                    phase="discovering",
                    completed=message_count_value,
                    unit="message",
                ),
            )
            poll_inputs.append({
                "conversation_id": conversation_id,
                "message_count": message_count,
                "last_modified": item.last_modified.isoformat(),
                "window_type": window_type,
            })
    finally:
        api_client = getattr(gene, "_client", None)
        if api_client is not None:
            await api_client.close()
    poll_audits = gene.get_poll_audits() if hasattr(gene, "get_poll_audits") else []
    if not poll_audits:
        poll_audits = _teams_poll_audits_from_documents(poll_inputs)
    return {"documents": documents, "poll_audits": poll_audits}


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
    source_time_range = {
        key: value
        for key, value in (("start", source_time_start), ("end", source_time_end))
        if value
    }
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


def _teams_progress_summary(
    documents: list[dict[str, Any]],
    poll_audits: list[dict[str, Any]],
) -> dict[str, Any]:
    date_from_values = [
        str(value)
        for value in (
            [poll.get("covered_created_from") for poll in poll_audits]
            + [doc.get("date_from") or doc.get("last_modified") for doc in documents]
        )
        if value
    ]
    date_to_values = [
        str(value)
        for value in (
            [poll.get("covered_created_to") for poll in poll_audits]
            + [doc.get("date_to") or doc.get("last_modified") for doc in documents]
        )
        if value
    ]
    messages = sum(
        _int_or_zero(
            poll.get("selected_message_keys_seen")
            or poll.get("unique_message_keys_seen")
            or poll.get("raw_messages_seen")
        )
        for poll in poll_audits
    )
    if messages == 0:
        messages = sum(_int_or_zero(doc.get("message_count")) for doc in documents)
    return {
        "date_from": min(date_from_values) if date_from_values else None,
        "date_to": max(date_to_values) if date_to_values else None,
        "messages": messages,
        "conversations": len(poll_audits) or len({
            str(doc.get("conversation_id")) for doc in documents if doc.get("conversation_id")
        }),
    }


def _teams_direct_rest_config_from_cloud_payload(payload: dict[str, Any]) -> dict[str, Any]:
    config = dict(payload)
    direct_ids: list[str] = []

    for key in ("conversation_ids", "channels", "group_chats", "individual_chats"):
        values = _teams_string_list(config.pop(key, []))
        for value in values:
            if not _is_direct_teams_conversation_id(value):
                raise ValueError("teams_sync_requires_direct_conversation_ids")
            direct_ids.append(value)

    if not direct_ids:
        raise ValueError("teams_sync_requires_direct_conversation_ids")

    config["conversation_ids"] = _dedupe_preserving_order(direct_ids)
    return config


def _is_direct_teams_conversation_id(value: str) -> bool:
    return bool(value and (":" in value or "@" in value))


def _teams_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _teams_poll_audits_from_documents(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_conversation: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        conversation_id = str(item.get("conversation_id") or "")
        if not conversation_id:
            continue
        by_conversation.setdefault(conversation_id, []).append(item)

    audits: list[dict[str, Any]] = []
    for conversation_id, conversation_items in sorted(by_conversation.items()):
        raw_messages = sum(_int_or_zero(item.get("message_count")) for item in conversation_items)
        timestamps = [str(item.get("last_modified") or "") for item in conversation_items if item.get("last_modified")]
        covered_from = min(timestamps) if timestamps else None
        covered_to = max(timestamps) if timestamps else None
        audits.append({
            "raw_conversation_id": conversation_id,
            "pagination_complete": True,
            "access_probe_status": "ok",
            "covered_created_from": covered_from,
            "covered_created_to": covered_to,
            "raw_messages_seen": raw_messages,
            "unique_message_keys_seen": raw_messages,
            "duplicate_raw_messages": 0,
            "upsert_new": raw_messages,
            "upsert_updated": 0,
            "upsert_unchanged": 0,
            "explicit_delete_markers": 0,
            "missing_once_candidates": 0,
            "field_contract_version": "teams_chatsvc_rest_v1",
        })
    return audits


def _int_or_zero(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _teams_window_id(
    *,
    source_id: str,
    conversation_id: str,
    root_message_id: str,
    window_type: str,
) -> str:
    from memforge.local_agent.teams_ledger import build_teams_window_id

    return build_teams_window_id(
        source_id=source_id,
        conversation_id=conversation_id,
        root_or_anchor_message_id=root_message_id,
        window_type=window_type,
    )


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
    root, include, exclude, vault_id = _resolve_kb_profile(name, profile)
    counts = {"included": 0, "ignored": 0, "too_large": 0, "invalid_utf8": 0, "unreadable": 0}
    pushed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    entries = _scan_kb_profile(root, include=include, exclude=exclude, counts=counts)
    selected_entries = list(islice(entries, limit)) if limit else list(entries)
    _report_local_agent_progress(
        report_progress,
        _sync_progress_snapshot(
            phase="discovering",
            completed=len(selected_entries),
            unit="file",
        ),
    )

    for index, entry in enumerate(selected_entries, start=1):
        response = client.push_local_markdown_document(
            source_id=source_id,
            vault_id=vault_id,
            relative_path=entry["relative_path"],
            markdown_body=entry["text"],
            content_type=entry["content_type"],
            title=entry["title"],
            raw_hash=entry["raw_hash"],
            sync_snapshot_id=sync_snapshot_id,
            local_agent_job_id=local_agent_job_id,
            local_agent_attempt_count=local_agent_attempt_count,
            submitted_by=submitted_by,
        )
        if isinstance(response, dict) and response.get("error"):
            failed.append({"relative_path": entry["relative_path"], "error": response.get("error"),
                           "detail": response.get("detail"), "status_code": response.get("status_code")})
        else:
            pushed.append({"relative_path": entry["relative_path"],
                           "doc_id": response.get("doc_id"),
                           "document_hash": response.get("document_hash")})
        _report_local_agent_progress(
            report_progress,
            _sync_progress_snapshot(
                phase="uploading",
                completed=index,
                total=len(selected_entries),
                unit="file",
                failed=len(failed),
            ),
        )

    payload = {
        "profile": name,
        "root": str(root),
        "vault_id": vault_id,
        "source_id": source_id,
        "counts": {**counts, "pushed": len(pushed), "failed": len(failed)},
        "pushed": pushed,
        "failed": failed,
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
            size = path.stat().st_size
        except OSError:
            counts["unreadable"] += 1
            continue
        if size > max_bytes:
            counts["too_large"] += 1
            continue
        try:
            raw = path.read_bytes()
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            counts["invalid_utf8"] += 1
            continue
        except OSError:
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
        }


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


def _principal_change_payload(upload_result: dict) -> dict:
    """Translate a 409 upload response into the principal-changed signal the CLI emits."""
    inner: dict = {}
    body = upload_result.get("detail")
    if isinstance(body, str):
        try:
            parsed = json.loads(body)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            candidate = parsed.get("detail", parsed)
            if isinstance(candidate, dict):
                inner = candidate
    return {
        "error": "principal_changed",
        "origin": inner.get("origin"),
        "old_principal_id": inner.get("old_principal_id"),
        "new_principal_id": inner.get("new_principal_id"),
    }


def _cookie_hash(cookie_header: str) -> str:
    return hashlib.sha256(cookie_header.encode("utf-8")).hexdigest()


async def run_watch_tick(*, base_url, browser, client, last_hash, capture, log):
    """One watch iteration. Returns (action, new_last_hash).

    action is one of: uploaded, unchanged, expired, principal_conflict, transport_error.
    Pure over its injected collaborators (capture, client, log) so it is unit-testable.
    """
    from memforge.auth.jira_auth import JiraAuthSessionMissingError

    try:
        result = await capture(base_url, browser=browser)
    except JiraAuthSessionMissingError as exc:
        client.mark_jira_session_expired(base_url=base_url, error=str(exc))
        log(f"Jira session for {base_url} is not active; sign back into Jira in your browser. ({exc})")
        return "expired", None

    new_hash = _cookie_hash(result.cookie_header)
    if new_hash == last_hash:
        return "unchanged", last_hash

    uploaded = client.upload_jira_session(
        base_url=base_url, cookie_header=result.cookie_header, browser=result.browser,
    )
    if uploaded.get("status_code") == 409:
        log(f"A different Jira user is signed in for {base_url}; re-run refresh with --confirm-principal-change.")
        return "principal_conflict", last_hash
    if uploaded.get("error"):
        log(f"Upload to MemForge failed: {uploaded.get('detail') or uploaded['error']}")
        return "transport_error", last_hash
    log(f"Refreshed Jira session for {base_url} (cookie {new_hash[:8]}).")
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

        ``status``, ``list``, and ``forget`` talk to the remote MemForge server.
        ``refresh`` captures the cookie from the local browser and uploads it.
        """

    @group.command("status")
    @click.option("--base-url", required=True, help=f"{descriptor.label} base URL.")
    @click.pass_context
    def status_cmd(ctx, base_url):
        """Show the server's stored session status for an origin."""
        _emit_tool_payload(ctx, _tool_client(ctx).get_jira_session(base_url))

    @group.command("list")
    @click.pass_context
    def list_cmd(ctx):
        """List known origins from the server."""
        _emit_tool_payload(ctx, _tool_client(ctx).list_jira_origins())

    @group.command("forget")
    @click.option("--base-url", required=True, help="Origin whose stored session to delete.")
    @click.pass_context
    def forget_cmd(ctx, base_url):
        """Forget the local and server-side stored session for an origin."""
        from memforge.auth.jira_auth import canonical_jira_origin
        from memforge.auth.jira_browser_session import JiraBrowserSession

        origin = canonical_jira_origin(base_url)
        JiraBrowserSession().forget(origin=origin)
        _emit_tool_payload(ctx, _tool_client(ctx).forget_jira_session(origin))

    @group.command("refresh")
    @click.option("--base-url", required=True, help=f"{descriptor.label} base URL.")
    @click.option("--browser", default=None, help="Browser to read cookies from, for example chrome or edge.")
    @click.option("--confirm-principal-change", is_flag=True, help="Allow this session to replace a different user.")
    @click.pass_context
    def refresh_cmd(ctx, base_url, browser, confirm_principal_change):
        """Capture the local browser session and upload it to the server."""
        from memforge.auth import jira_capture
        from memforge.auth.jira_auth import JiraAuthSessionError, JiraAuthSessionMissingError

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
        payload = _tool_client(ctx).upload_jira_session(
            base_url=result.origin,
            cookie_header=result.cookie_header,
            browser=result.browser,
            confirm_principal_change=confirm_principal_change,
        )
        if payload.get("status_code") == 409:
            _emit_tool_payload(ctx, _principal_change_payload(payload))
            return
        _emit_tool_payload(ctx, payload)

    @group.command("watch")
    @click.option("--base-url", required=True, help=f"{descriptor.label} base URL.")
    @click.option("--browser", default=None, help="Browser to read cookies from, for example chrome or edge.")
    @click.option("--interval-seconds", type=int, default=WATCH_DEFAULT_INTERVAL_SECONDS, show_default=True,
                  help="Seconds between re-capture attempts. Keep it under your Jira idle timeout.")
    @click.pass_context
    def watch_cmd(ctx, base_url, browser, interval_seconds):
        """Keep the server's Jira session fresh by re-capturing on an interval."""
        from memforge.auth import jira_capture

        client = _tool_client(ctx)

        async def _capture(url, *, browser=None):
            return await jira_capture.capture_and_prevalidate(url, browser=browser)

        async def _loop():
            last_hash = None
            backoff = WATCH_BACKOFF_BASE_SECONDS
            while True:
                try:
                    action, last_hash = await run_watch_tick(
                        base_url=base_url, browser=browser, client=client,
                        last_hash=last_hash, capture=_capture, log=click.echo,
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
@click.option("--region", default="emea", type=click.Choice(["emea", "amer", "apac"]),
              help="Teams API region")
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
    """Repair FTS5 and Chroma indexes from SQLite."""

    async def _run():
        from memforge.memory.health import MemoryIndexHealthChecker
        from memforge.memory.repair import MemoryIndexRepairer
        from memforge.retrieval.embeddings import get_chroma_collection
        from memforge.runtime import get_effective_llm_config

        config: AppConfig = ctx.obj["config"]
        db = await _get_db(config)
        memory_collection = get_chroma_collection(config.storage.chroma_path, name="memories")
        document_collection = get_chroma_collection(config.storage.chroma_path, name="documents")
        llm = await get_effective_llm_config(db, config)
        embed_cfg = {
            "base_url": llm.embedding_base_url,
            "api_key": llm.embedding_api_key,
            "model": llm.embedding_model,
        }
        try:
            result = await MemoryIndexRepairer(
                db=db,
                memory_collection=memory_collection,
                document_collection=document_collection,
                embed_cfg=embed_cfg,
            ).repair()
            report = await MemoryIndexHealthChecker(
                db=db,
                memory_collection=memory_collection,
                document_collection=document_collection,
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
        table.add_row("Document vectors repaired", str(result.document_vectors_repaired))
        table.add_row("Document vectors created", str(result.document_vectors_created))
        table.add_row("Unrepaired memories", str(len(result.unrepaired_memories)))
        table.add_row("Unrepaired documents", str(len(result.unrepaired_documents)))
        console.print(table)
        if report.ok:
            console.print("[green]Index health is clean.[/]")
        else:
            console.print(f"[yellow]{len(report.issues)} consistency issue(s) remain.[/]")
            if result.unrepaired_memories:
                for memory_id in result.unrepaired_memories[:10]:
                    console.print(f"  [dim]Unrepaired memory: {memory_id}[/]")
            if result.unrepaired_documents:
                for doc_id in result.unrepaired_documents[:10]:
                    console.print(f"  [dim]Unrepaired document: {doc_id}[/]")

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# config group
# ---------------------------------------------------------------------------


@cli.group("config")
def config_group():
    """View and manage configuration."""
    pass


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
        "Enrichment model": config.llm.enrichment_model,
        "Enrichment API": config.llm.enrichment_base_url,
        "Enrichment key": "***" if config.llm.enrichment_api_key else "[red]NOT SET[/]",
        "Embedding model": config.llm.embedding_model,
        "Embedding API": config.llm.embedding_base_url,
        "Embedding key": "***" if config.llm.embedding_api_key else "[red]NOT SET[/]",
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
