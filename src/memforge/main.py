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
from importlib import metadata
from itertools import islice
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import quote

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 compatibility
    import tomli as tomllib

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from memforge.auth import browser_session
from memforge.config import AppConfig, load_config
from memforge.github_repo_utils import (
    DEFAULT_INCLUDE_EXTENSION_LIST,
    decode_github_base64_content,
    github_content_type,
    github_extension,
    github_include_extensions,
    github_include_paths,
    github_path_in_scope,
    parse_github_repo_url,
)
from memforge.storage.admin_source import (
    SOURCE_SYNC_SCHEDULE_MAX_INTERVAL_MINUTES,
    SOURCE_SYNC_SCHEDULE_MIN_INTERVAL_MINUTES,
)
from memforge.tool_client import ToolClient

console = Console()
log_console = Console(stderr=True)
DEFAULT_CLI_CONFIG_PATH = Path.home() / ".memforge" / "cli.toml"
DEFAULT_ADAPTER_CONFIG_PATH = Path.home() / ".memforge" / "adapter.toml"
DEFAULT_KB_INCLUDE = [
    "*.md", "**/*.md",
    "*.markdown", "**/*.markdown",
    "*.txt", "**/*.txt",
    "*.json", "**/*.json",
    "*.html", "**/*.html",
    "*.htm", "**/*.htm",
]
DEFAULT_KB_EXCLUDE = [".obsidian/**", ".trash/**", ".git/**", "**/.git/**"]
KB_SCHEDULE_PRESETS = {
    "15m": "*/15 * * * *",
    "30m": "*/30 * * * *",
    "hourly": "0 * * * *",
    "2h": "0 */2 * * *",
    "4h": "0 */4 * * *",
    "6h": "0 */6 * * *",
    "12h": "0 */12 * * *",
    "daily": "0 9 * * *",
    "weekly": "0 9 * * 1",
}
KB_CRON_MARK_START = "# >>> memforge:kb:{name} >>>"
KB_CRON_MARK_END = "# <<< memforge:kb:{name} <<<"
LOCAL_MARKDOWN_SOURCE_TYPE = "local_markdown"
GITHUB_REPO_SOURCE_TYPE = "github_repo"
DEFAULT_GITHUB_INCLUDE_EXTENSIONS = DEFAULT_INCLUDE_EXTENSION_LIST
INTERACTIVE_DISABLE_ENV = "MEMFORGE_NO_INTERACTIVE"
INTERACTIVE_SCRIPT_ENV = "MEMFORGE_INTERACTIVE_SCRIPT"
INTERACTIVE_BIN_ENV = "MEMFORGE_CLI_BIN"
INTERACTIVE_CACHE_ENV = "MEMFORGE_INTERACTIVE_CACHE"
INTERACTIVE_RESOURCE_DIR = "interactive_cli"
INTERACTIVE_INSTALL_LOCK = ".install.lock"
INTERACTIVE_INSTALL_LOCK_TIMEOUT_SECONDS = 120
INTERACTIVE_INSTALL_LOCK_STALE_SECONDS = 600
INTERACTIVE_DEPENDENCY_SENTINEL = Path("node_modules") / "@clack" / "prompts"


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
        if target.get("token_env"):
            lines.append(f"token_env = {_toml_string(str(target['token_env']))}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o600)


def _toml_string(value: str) -> str:
    return json.dumps(value)


def _toml_key(value: str) -> str:
    return json.dumps(value)


def _active_target() -> dict[str, Any] | None:
    data = _read_cli_config()
    active = str(data.get("active") or "")
    targets = data.get("targets") if isinstance(data.get("targets"), dict) else {}
    target = targets.get(active)
    return target if isinstance(target, dict) else None


def _resolve_api_endpoint(config: AppConfig) -> tuple[str, str | None]:
    env_url = os.getenv("MEMFORGE_API_URL")
    if env_url:
        return env_url, os.getenv("MEMFORGE_API_TOKEN")
    target = _active_target()
    if target and target.get("api_url"):
        token_env = str(target.get("token_env") or "")
        return str(target["api_url"]), os.getenv(token_env) if token_env else None
    return f"http://127.0.0.1:{config.server.admin_api_port}", os.getenv("MEMFORGE_API_TOKEN")


def _tool_client(ctx) -> ToolClient:
    config: AppConfig = ctx.obj["config"]
    api_url, api_token = _resolve_api_endpoint(config)
    return ToolClient(
        api_url=api_url,
        api_token=api_token,
    )


def _emit_tool_payload(ctx, payload: dict) -> None:
    click.echo(json.dumps(payload, indent=2, ensure_ascii=False))
    if payload.get("error"):
        ctx.exit(1)


def _adapter_config_path() -> Path:
    configured = os.getenv("MEMFORGE_ADAPTER_CONFIG", "").strip()
    return Path(configured).expanduser() if configured else DEFAULT_ADAPTER_CONFIG_PATH


def _read_adapter_config() -> dict[str, Any]:
    path = _adapter_config_path()
    if not path.exists():
        return {"kb": {}, "github": {}}
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    kb = data.get("kb")
    if not isinstance(kb, dict):
        kb = {}
    github = data.get("github")
    if not isinstance(github, dict):
        github = {}
    return {"kb": kb, "github": github}


def _write_adapter_config(data: dict[str, Any]) -> None:
    path = _adapter_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    kb = data.get("kb") if isinstance(data.get("kb"), dict) else {}
    github = data.get("github") if isinstance(data.get("github"), dict) else {}
    lines: list[str] = []
    for name, profile in sorted(kb.items()):
        if not isinstance(profile, dict):
            continue
        lines.append(f"[kb.{_toml_key(str(name))}]")
        lines.append(f"root = {_toml_string(str(profile.get('root') or ''))}")
        lines.append(f"vault_id = {_toml_string(str(profile.get('vault_id') or name))}")
        lines.append(f"include = {_toml_string_list(_string_list(profile.get('include')) or DEFAULT_KB_INCLUDE)}")
        lines.append(f"exclude = {_toml_string_list(_string_list(profile.get('exclude')) or DEFAULT_KB_EXCLUDE)}")
        source_id = str(profile.get("source_id") or "").strip()
        if source_id:
            lines.append(f"source_id = {_toml_string(source_id)}")
        schedule = str(profile.get("schedule") or "").strip()
        if schedule:
            lines.append(f"schedule = {_toml_string(schedule)}")
        lines.append("")
    for name, profile in sorted(github.items()):
        if not isinstance(profile, dict):
            continue
        lines.append(f"[github.{_toml_key(str(name))}]")
        lines.append(f"repo_url = {_toml_string(str(profile.get('repo_url') or ''))}")
        repo_path = str(profile.get("repo_path") or "").strip()
        if repo_path:
            lines.append(f"repo_path = {_toml_string(repo_path)}")
        lines.append(f"ref = {_toml_string(str(profile.get('ref') or 'main'))}")
        lines.append(f"include_paths = {_toml_string_list(_string_list(profile.get('include_paths')))}")
        lines.append(
            f"include_extensions = {_toml_string_list(_string_list(profile.get('include_extensions')) or DEFAULT_GITHUB_INCLUDE_EXTENSIONS)}"
        )
        source_id = str(profile.get("source_id") or "").strip()
        if source_id:
            lines.append(f"source_id = {_toml_string(source_id)}")
        schedule = str(profile.get("schedule") or "").strip()
        if schedule:
            lines.append(f"schedule = {_toml_string(schedule)}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o600)


# A profile name becomes part of the source id, a TOML table key, and a line in
# the user crontab, so it is restricted to an identifier-safe character set.
_KB_PROFILE_NAME_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)


def _validate_kb_profile_name(name: str) -> str:
    """Return a safe profile name or raise.

    Restricting the character set blocks crontab injection (an embedded newline
    would otherwise add a live cron line) and multi-word names that cron would
    split into separate arguments.
    """
    cleaned = (name or "").strip()
    if not cleaned:
        raise click.ClickException("Profile name is required.")
    if any(ch not in _KB_PROFILE_NAME_CHARS for ch in cleaned):
        raise click.ClickException(
            "Profile name may contain only letters, digits, '.', '_', and '-' (no spaces or newlines)."
        )
    return cleaned


def _toml_string_list(values: list[str]) -> str:
    return "[" + ", ".join(_toml_string(value) for value in values) + "]"


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


def _link_local_markdown_source(
    ctx,
    *,
    vault_id: str,
    display_label: str | None,
    name: str,
) -> dict[str, Any]:
    """Reuse or create a ``local_markdown`` source for ``vault_id`` on the server.

    A list failure means we cannot tell whether a matching source already exists,
    so we stop rather than risk creating a duplicate. The caller keeps the local
    profile either way and surfaces ``error`` to the user.
    """
    client = _tool_client(ctx)
    listed = client.list_sources()
    if isinstance(listed, dict) and listed.get("error"):
        return {"error": str(listed.get("error")), "detail": listed.get("detail")}
    for source in listed.get("data") or []:
        config = source.get("config") or {}
        if source.get("type") == LOCAL_MARKDOWN_SOURCE_TYPE and str(config.get("vault_id") or "") == vault_id:
            return {"source_id": str(source["id"]), "reused": True}
    source_config: dict[str, Any] = {"vault_id": vault_id}
    if display_label:
        source_config["display_label"] = display_label
    created = client.create_source(
        source_type=LOCAL_MARKDOWN_SOURCE_TYPE,
        name=name,
        config=source_config,
    )
    if isinstance(created, dict) and created.get("error"):
        return {"error": str(created.get("error")), "detail": created.get("detail")}
    return {"source_id": str(created["id"]), "reused": False}


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


def _git_run(repo_path: Path, args: list[str], *, text: bool = True) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["git", "-C", str(repo_path), *args],
            capture_output=True,
            text=text,
            check=False,
        )
    except FileNotFoundError as exc:
        raise click.ClickException("Git is required for local GitHub repository clone sync.") from exc


def _git_checked(repo_path: Path, args: list[str], *, text: bool = True) -> subprocess.CompletedProcess:
    result = _git_run(repo_path, args, text=text)
    if result.returncode != 0:
        stderr = result.stderr if isinstance(result.stderr, str) else bytes(result.stderr or b"").decode("utf-8", "replace")
        stdout = result.stdout if isinstance(result.stdout, str) else bytes(result.stdout or b"").decode("utf-8", "replace")
        detail = (stderr or stdout or "git command failed").strip()
        raise click.ClickException(f"Local Git repository request failed: {detail}")
    return result


def _repo_path_from_profile(profile: dict[str, Any], repo: dict[str, str]) -> Path | None:
    raw_path = str(profile.get("repo_path") or "").strip()
    if not raw_path:
        return None
    candidate = Path(raw_path).expanduser()
    if not candidate.exists() or not candidate.is_dir():
        raise click.ClickException(f"GitHub repository path does not exist or is not a directory: {candidate}")
    top_level = _git_checked(candidate, ["rev-parse", "--show-toplevel"]).stdout.strip()
    repo_path = Path(top_level).resolve()
    remote = _git_checked(repo_path, ["remote", "get-url", "origin"]).stdout.strip()
    try:
        remote_repo = parse_github_repo_url(remote)
    except ValueError as exc:
        raise click.ClickException("GitHub repository path origin remote must be an https GitHub URL.") from exc
    if remote_repo["repo_url"] != repo["repo_url"]:
        raise click.ClickException(
            f"GitHub repository path origin {remote_repo['repo_url']} does not match configured repo_url {repo['repo_url']}."
        )
    return repo_path


def _github_local_tree(repo_path: Path, ref: str) -> list[dict[str, Any]]:
    result = _git_checked(repo_path, ["ls-tree", "-z", "-r", "-l", ref, "--"])
    entries: list[dict[str, Any]] = []
    for line in result.stdout.split("\0"):
        if not line:
            continue
        metadata, separator, path = line.partition("\t")
        if not separator:
            continue
        parts = metadata.split()
        if len(parts) < 4 or parts[1] != "blob":
            continue
        size_raw = parts[3]
        try:
            size = int(size_raw)
        except ValueError:
            size = 0
        entries.append({"path": path, "type": "blob", "sha": parts[2], "size": size})
    return entries


def _github_tree_for_profile(repo: dict[str, str], ref: str, profile: dict[str, Any]) -> tuple[list[dict[str, Any]], Path | None]:
    repo_path = _repo_path_from_profile(profile, repo)
    if repo_path is not None:
        return _github_local_tree(repo_path, ref), repo_path
    return _github_tree(repo, ref), None


def _github_local_content(repo_path: Path, ref: str, relative_path: str) -> bytes:
    result = _git_checked(repo_path, ["show", f"{ref}:{relative_path}"], text=False)
    return result.stdout


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


def _resolve_github_profile(name: str, profile: dict[str, Any]) -> tuple[dict[str, str], str, list[str], list[str]]:
    repo = _parse_github_repo_url(str(profile.get("repo_url") or ""))
    ref = str(profile.get("ref") or "main").strip() or "main"
    if ref.startswith("-"):
        raise click.ClickException("GitHub repository ref must not start with '-'.")
    try:
        include_paths = github_include_paths(profile)
    except ValueError as exc:
        raise click.ClickException(str(exc).replace("relative_path", "GitHub include path")) from exc
    include_extensions = sorted(github_include_extensions(profile))
    return repo, ref, include_paths, include_extensions


def _github_title(markdown_body: str, fallback: str) -> str:
    for line in markdown_body.splitlines():
        stripped = line.strip()
        if stripped.startswith("# ") and stripped[2:].strip():
            return stripped[2:].strip()
    return fallback


def _preview_github_profile(name: str, profile: dict[str, Any], *, limit: int | None) -> dict[str, Any]:
    repo, ref, include_paths, include_extensions = _resolve_github_profile(name, profile)
    counts = {"included": 0, "ignored": 0}
    extension_counts: dict[str, int] = {}
    items: list[dict[str, Any]] = []
    tree, repo_path = _github_tree_for_profile(repo, ref, profile)
    for entry in tree:
        if entry.get("type") != "blob":
            continue
        relative_path = str(entry.get("path") or "")
        if not github_path_in_scope(relative_path, include_paths):
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
        "include_extensions": include_extensions,
        "counts": counts,
        "extension_counts": dict(sorted(extension_counts.items())),
        "items": items,
    }
    if repo_path is not None:
        payload["repo_path"] = str(repo_path)
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
            "Install Node, then run [bold]cd cli && npm install[/], then re-run [bold]memforge[/]."
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
@click.option("--token-env", default="MEMFORGE_API_TOKEN", show_default=True, help="Environment variable for the token.")
@click.pass_context
def target_add(ctx, name: str, api_url: str, token_env: str):
    """Add or update an API target and make it active."""
    name = name.strip()
    if not name:
        raise click.ClickException("Target name is required.")
    data = _read_cli_config()
    targets = data.setdefault("targets", {})
    targets[name] = {"api_url": api_url.rstrip("/"), "token_env": token_env.strip()}
    data["active"] = name
    _write_cli_config(data)
    _emit_tool_payload(ctx, {"ok": True, "active": name, "api_url": targets[name]["api_url"]})


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
            ],
        },
    )


@adapter.group("kb")
def adapter_kb():
    """Manage local repository adapter profiles (Markdown, text, JSON, HTML)."""
    pass


@adapter.group("github")
def adapter_github():
    """Manage local GitHub repository adapter profiles."""
    pass


@adapter_github.command("add")
@click.argument("name")
@click.option("--repo-url", required=True, help="GitHub or GitHub Enterprise repository URL.")
@click.option("--repo-path", default=None, type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Local clone path for Internal network / VPN repositories.")
@click.option("--ref", "repo_ref", default="main", show_default=True, help="Branch, tag, or commit to sync.")
@click.option("--include-path", "include_paths", multiple=True,
              help="Repo-relative folder or file to include. Repeatable. Empty means all paths.")
@click.option("--include-extension", "include_extensions", multiple=True,
              help="File extension to include. Repeatable. Defaults to markdown/text-like files.")
@click.option("--source-id", default=None, help="Existing github_repo source id to link to this profile.")
@click.pass_context
def adapter_github_add(
    ctx,
    name: str,
    repo_url: str,
    repo_path: Path | None,
    repo_ref: str,
    include_paths: tuple[str, ...],
    include_extensions: tuple[str, ...],
    source_id: str | None,
):
    """Add or update a local GitHub repository profile."""
    name = _validate_kb_profile_name(name)
    repo = _parse_github_repo_url(repo_url)
    try:
        normalized_include_paths = github_include_paths({"include_paths": list(include_paths)})
    except ValueError as exc:
        raise click.ClickException(str(exc).replace("relative_path", "GitHub include path")) from exc
    profile = {
        "repo_url": repo["repo_url"],
        "ref": repo_ref.strip() or "main",
        "include_paths": normalized_include_paths,
        "include_extensions": [
            ext.lower().lstrip(".")
            for ext in (list(include_extensions) or DEFAULT_GITHUB_INCLUDE_EXTENSIONS)
            if ext.strip()
        ],
    }
    if repo_path is not None:
        profile["repo_path"] = str(_repo_path_from_profile({"repo_path": str(repo_path)}, repo))
    if source_id:
        profile["source_id"] = source_id.strip()
    data = _read_adapter_config()
    data.setdefault("github", {})[name] = profile
    _write_adapter_config(data)
    _emit_tool_payload(ctx, {"ok": True, "profile": name, "config": profile})


@adapter_github.command("list")
@click.pass_context
def adapter_github_list(ctx):
    """List local GitHub repository profiles."""
    _emit_tool_payload(ctx, {"profiles": _read_adapter_config().get("github", {})})


@adapter_github.command("remove")
@click.argument("name")
@click.pass_context
def adapter_github_remove(ctx, name: str):
    """Remove a local GitHub repository profile."""
    name = name.strip()
    data = _read_adapter_config()
    github = data.get("github", {})
    if name not in github:
        raise click.ClickException(f"Unknown GitHub repository profile: {name}")
    github.pop(name)
    _write_adapter_config(data)
    _emit_tool_payload(ctx, {"ok": True, "removed": name})


@adapter_github.command("preview")
@click.argument("name")
@click.option("--limit", default=20, show_default=True, type=int, help="Maximum included files to return.")
@click.pass_context
def adapter_github_preview(ctx, name: str, limit: int):
    """Preview the files selected by a GitHub repository profile."""
    profiles = _read_adapter_config().get("github", {})
    profile = profiles.get(name)
    if not isinstance(profile, dict):
        raise click.ClickException(f"Unknown GitHub repository profile: {name}")
    _emit_tool_payload(ctx, _preview_github_profile(name, profile, limit=limit))


@adapter_github.command("push")
@click.argument("name")
@click.option("--source-id", default=None,
              help="MemForge github_repo source id (defaults to the profile's linked source).")
@click.option("--limit", default=0, show_default=True, type=int,
              help="Maximum files to push. 0 means push every included file.")
@click.option("--process-now/--no-process-now", default=False, show_default=True,
              help="Trigger source sync after the push completes.")
@click.option("--submitted-by", default=None, help="Optional label recorded with each push.")
@click.pass_context
def adapter_github_push(
    ctx,
    name: str,
    source_id: str | None,
    limit: int,
    process_now: bool,
    submitted_by: str | None,
):
    """Push selected GitHub repository files into a configured github_repo source."""
    profiles = _read_adapter_config().get("github", {})
    profile = profiles.get(name)
    if not isinstance(profile, dict):
        raise click.ClickException(f"Unknown GitHub repository profile: {name}")
    source_id = (source_id or "").strip() or str(profile.get("source_id") or "").strip()
    if not source_id:
        raise click.ClickException(
            "No source id for this profile. Re-run with --source-id, or link one with `adapter github add ... --source-id`."
        )

    repo, ref, _include_paths, _include_extensions = _resolve_github_profile(name, profile)
    repo_path = _repo_path_from_profile(profile, repo)
    preview = _preview_github_profile(name, profile, limit=None if limit == 0 else max(limit, 0))
    selected_entries = list(preview["items"])

    client = _tool_client(ctx)
    pushed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    prepared: list[dict[str, Any]] = []
    for entry in selected_entries:
        relative_path = str(entry["relative_path"])
        try:
            raw = _github_local_content(repo_path, ref, relative_path) if repo_path is not None else _github_content(
                repo, ref, relative_path
            )
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

    for index, doc in enumerate(prepared):
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
            submitted_by=submitted_by,
            process_now=process_now and index == len(prepared) - 1,
        )
        if isinstance(response, dict) and response.get("error"):
            failed.append({"relative_path": doc["relative_path"], "error": response.get("error"),
                           "detail": response.get("detail"), "status_code": response.get("status_code")})
        else:
            pushed.append({"relative_path": doc["relative_path"],
                           "doc_id": response.get("doc_id"),
                           "document_hash": response.get("document_hash")})

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
    _emit_tool_payload(ctx, payload)


@adapter_kb.command("add")
@click.argument("name")
@click.option("--root", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--vault-id", default=None, help="Stable vault identifier used in local metadata.")
@click.option("--include", "includes", multiple=True,
              help="Glob to include, relative to root. Repeatable. Replaces the default md/txt/json/html set.")
@click.option("--exclude", "excludes", multiple=True,
              help="Glob to exclude, relative to root. Repeatable. Added to the .obsidian/.trash/.git safety excludes.")
@click.option("--display-label", default=None, help="Human-readable label for the linked source.")
@click.option("--create-source/--no-create-source", default=False, show_default=True,
              help="Reuse or create the matching local_markdown source and store its id in the profile.")
@click.pass_context
def adapter_kb_add(
    ctx,
    name: str,
    root: Path,
    vault_id: str | None,
    includes: tuple[str, ...],
    excludes: tuple[str, ...],
    display_label: str | None,
    create_source: bool,
):
    """Add or update a local repository profile.

    With ``--create-source`` the profile is also linked to its MemForge source:
    an existing ``local_markdown`` source with the same vault id is reused, or a
    new one is created, and its id is stored so ``push`` needs no source id.
    """
    name = _validate_kb_profile_name(name)
    resolved_vault = vault_id or name
    data = _read_adapter_config()
    kb = data.setdefault("kb", {})
    kb[name] = {
        "root": str(root.expanduser().resolve()),
        "vault_id": resolved_vault,
        "include": list(includes) or DEFAULT_KB_INCLUDE,
        "exclude": _merge_default_excludes(list(excludes)),
    }

    payload: dict[str, Any] = {"ok": True, "profile": name}
    if create_source:
        link = _link_local_markdown_source(
            ctx,
            vault_id=resolved_vault,
            display_label=display_label,
            name=display_label or name,
        )
        if link.get("error"):
            payload["source_link_error"] = link["error"]
            if link.get("detail"):
                payload["detail"] = link["detail"]
        else:
            kb[name]["source_id"] = link["source_id"]
            payload["source_id"] = link["source_id"]
            payload["source_reused"] = link["reused"]

    _write_adapter_config(data)
    payload["config"] = kb[name]
    _emit_tool_payload(ctx, payload)


@adapter_kb.command("list")
@click.pass_context
def adapter_kb_list(ctx):
    """List local repository profiles."""
    _emit_tool_payload(ctx, {"profiles": _read_adapter_config().get("kb", {})})


@adapter_kb.command("scan")
@click.option("--root", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--include", "includes", multiple=True,
              help="Glob to include, relative to root. Repeatable. Replaces the default md/txt/json/html set.")
@click.option("--exclude", "excludes", multiple=True,
              help="Glob to exclude, relative to root. Repeatable. Added to the .obsidian/.trash/.git safety excludes.")
@click.option("--limit", default=20, show_default=True, type=int, help="Maximum included files to return.")
@click.pass_context
def adapter_kb_scan(ctx, root: Path, includes: tuple[str, ...], excludes: tuple[str, ...], limit: int):
    """Dry-scan a folder for markdown files without saving a profile.

    Backs the setup wizard's instant-feedback step: it reports what would be
    included before a profile (and its vault id) exist.
    """
    profile = {
        "root": str(root.expanduser().resolve()),
        "vault_id": "scan",
        "include": list(includes) or DEFAULT_KB_INCLUDE,
        "exclude": _merge_default_excludes(list(excludes)),
    }
    _emit_tool_payload(ctx, _preview_kb_profile("scan", profile, limit=limit))


@adapter_kb.command("remove")
@click.argument("name")
@click.pass_context
def adapter_kb_remove(ctx, name: str):
    """Remove a local repository profile."""
    name = name.strip()
    data = _read_adapter_config()
    kb = data.get("kb", {})
    if name not in kb:
        raise click.ClickException(f"Unknown knowledge-base profile: {name}")
    kb.pop(name)
    _write_adapter_config(data)
    _emit_tool_payload(ctx, {"ok": True, "removed": name})


@adapter_kb.command("preview")
@click.argument("name")
@click.option("--limit", default=20, show_default=True, type=int, help="Maximum included files to return.")
@click.pass_context
def adapter_kb_preview(ctx, name: str, limit: int):
    """Preview local markdown files for a knowledge-base profile."""
    profiles = _read_adapter_config().get("kb", {})
    profile = profiles.get(name)
    if not isinstance(profile, dict):
        raise click.ClickException(f"Unknown knowledge-base profile: {name}")
    payload = _preview_kb_profile(name, profile, limit=limit)
    _emit_tool_payload(ctx, payload)


@adapter_kb.command("push")
@click.argument("name")
@click.option("--source-id", default=None,
              help="MemForge source id (defaults to the profile's linked source).")
@click.option("--limit", default=0, show_default=True, type=int,
              help="Maximum files to push. 0 means push every included file.")
@click.option("--process-now/--no-process-now", default=False, show_default=True,
              help="Trigger source sync after the push completes.")
@click.option("--submitted-by", default=None, help="Optional label recorded with each push.")
@click.pass_context
def adapter_kb_push(
    ctx,
    name: str,
    source_id: str | None,
    limit: int,
    process_now: bool,
    submitted_by: str | None,
):
    """Push local markdown files for a profile into a configured local_markdown source.

    The source id defaults to the one stored in the profile by
    ``adapter kb add --create-source``; pass ``--source-id`` to override it.
    """
    profiles = _read_adapter_config().get("kb", {})
    profile = profiles.get(name)
    if not isinstance(profile, dict):
        raise click.ClickException(f"Unknown knowledge-base profile: {name}")

    source_id = (source_id or "").strip() or str(profile.get("source_id") or "").strip()
    if not source_id:
        raise click.ClickException(
            "No source id for this profile. Re-run with --source-id, or link one "
            "with `adapter kb add " + name + " --root <path> --create-source`."
        )

    root, include, exclude, vault_id = _resolve_kb_profile(name, profile)
    counts = {"included": 0, "ignored": 0, "too_large": 0, "invalid_utf8": 0, "unreadable": 0}
    client = _tool_client(ctx)
    pushed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    entries = _scan_kb_profile(root, include=include, exclude=exclude, counts=counts)
    selected_entries = list(islice(entries, limit)) if limit else list(entries)

    for index, entry in enumerate(selected_entries):
        is_last = index == len(selected_entries) - 1
        response = client.push_local_markdown_document(
            source_id=source_id,
            vault_id=vault_id,
            relative_path=entry["relative_path"],
            markdown_body=entry["text"],
            content_type=entry["content_type"],
            title=entry["title"],
            raw_hash=entry["raw_hash"],
            submitted_by=submitted_by,
            process_now=process_now and is_last,
        )
        if isinstance(response, dict) and response.get("error"):
            failed.append({"relative_path": entry["relative_path"], "error": response.get("error"),
                           "detail": response.get("detail"), "status_code": response.get("status_code")})
        else:
            pushed.append({"relative_path": entry["relative_path"],
                           "doc_id": response.get("doc_id"),
                           "document_hash": response.get("document_hash")})

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
    _emit_tool_payload(ctx, payload)


@adapter_kb.command("schedule")
@click.argument("name")
@click.option("--every", default="daily", show_default=True, type=click.Choice(list(KB_SCHEDULE_PRESETS)),
              help="How often to sync. Ignored when --cron is given.")
@click.option("--at", "at_time", default=None,
              help="Time of day HH:MM for the daily and weekly presets (default 09:00).")
@click.option("--cron", "cron_expr", default=None,
              help="Raw 5-field cron expression. Overrides --every and --at.")
@click.pass_context
def adapter_kb_schedule(ctx, name: str, every: str, at_time: str | None, cron_expr: str | None):
    """Install an OS cron job that runs ``adapter kb push NAME --process-now``.

    Scheduling uses the user crontab today. A background watcher daemon is a
    planned alternative (see docs/local-repo-sync.md).
    """
    name = _validate_kb_profile_name(name)
    if not isinstance(_read_adapter_config().get("kb", {}).get(name), dict):
        raise click.ClickException(f"Unknown knowledge-base profile: {name}")
    try:
        expr = _render_cron_expr(every=every, at_time=at_time, cron_expr=cron_expr)
    except ValueError as exc:
        raise click.ClickException(str(exc))
    command = _kb_push_command(name)
    _write_crontab(_apply_crontab_block(_read_crontab(), name, _render_crontab_block(name, expr, command)))
    data = _read_adapter_config()
    data.setdefault("kb", {}).setdefault(name, {})["schedule"] = expr
    _write_adapter_config(data)
    _emit_tool_payload(ctx, {"ok": True, "profile": name, "cron": expr, "command": command})


@adapter_kb.command("schedule-list")
@click.pass_context
def adapter_kb_schedule_list(ctx):
    """List KB sync schedules and whether each cron job is installed."""
    profiles = _read_adapter_config().get("kb", {})
    crontab = _read_crontab()
    schedules = []
    for name, profile in profiles.items():
        if not isinstance(profile, dict):
            continue
        cron = str(profile.get("schedule") or "").strip()
        installed = _has_crontab_block(crontab, name)
        if cron or installed:
            schedules.append({"profile": name, "cron": cron or None, "installed": installed})
    _emit_tool_payload(ctx, {"schedules": schedules})


@adapter_kb.command("unschedule")
@click.argument("name")
@click.pass_context
def adapter_kb_unschedule(ctx, name: str):
    """Remove the OS cron job for a KB profile."""
    name = name.strip()
    crontab = _read_crontab()
    removed = _has_crontab_block(crontab, name)
    if removed:
        _write_crontab(_remove_crontab_block(crontab, name))
    data = _read_adapter_config()
    profile = data.get("kb", {}).get(name)
    if isinstance(profile, dict) and profile.pop("schedule", None) is not None:
        _write_adapter_config(data)
    _emit_tool_payload(ctx, {"ok": True, "profile": name, "removed": removed})


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


def _render_cron_expr(*, every: str, at_time: str | None = None, cron_expr: str | None = None) -> str:
    """Resolve a 5-field cron expression from a preset (+ optional time) or a raw expression."""
    if cron_expr:
        fields = cron_expr.split()
        if len(fields) != 5:
            raise ValueError("--cron must be a 5-field expression, for example '0 9 * * 1'")
        return " ".join(fields)
    base = KB_SCHEDULE_PRESETS.get(every)
    if base is None:
        raise ValueError(f"Unknown schedule preset: {every}")
    if at_time and every in ("daily", "weekly"):
        hour, minute = _parse_hh_mm(at_time)
        parts = base.split()
        parts[0], parts[1] = str(minute), str(hour)
        return " ".join(parts)
    return base


def _parse_hh_mm(value: str) -> tuple[int, int]:
    try:
        hh, mm = value.strip().split(":")
        hour, minute = int(hh), int(mm)
    except (ValueError, AttributeError):
        raise ValueError("--at must be a 24-hour time HH:MM, for example 08:30") from None
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError("--at must be a valid 24-hour time HH:MM")
    return hour, minute


def _kb_push_command(name: str) -> str:
    memforge = shutil.which("memforge") or "memforge"
    return f"{memforge} adapter kb push {name} --process-now >> $HOME/.memforge/kb-{name}.log 2>&1"


def _render_crontab_block(name: str, cron_expr: str, command: str) -> str:
    return "\n".join([
        KB_CRON_MARK_START.format(name=name),
        f"{cron_expr} {command}",
        KB_CRON_MARK_END.format(name=name),
    ])


def _has_crontab_block(crontab: str, name: str) -> bool:
    return KB_CRON_MARK_START.format(name=name) in crontab


def _strip_crontab_block(crontab: str, name: str) -> str:
    start = KB_CRON_MARK_START.format(name=name)
    end = KB_CRON_MARK_END.format(name=name)
    kept: list[str] = []
    skipping = False
    for line in crontab.splitlines():
        if line.strip() == start:
            skipping = True
            continue
        if skipping:
            if line.strip() == end:
                skipping = False
            continue
        kept.append(line)
    return "\n".join(kept).rstrip("\n")


def _apply_crontab_block(crontab: str, name: str, block: str) -> str:
    cleaned = _strip_crontab_block(crontab, name)
    body = "\n".join(part for part in (cleaned, block) if part)
    return body + "\n"


def _remove_crontab_block(crontab: str, name: str) -> str:
    cleaned = _strip_crontab_block(crontab, name)
    return cleaned + "\n" if cleaned else ""


def _read_crontab() -> str:
    try:
        result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise click.ClickException("`crontab` is not available on this system.") from exc
    # A missing crontab exits non-zero with a notice on stderr; treat that as empty.
    return result.stdout if result.returncode == 0 else ""


def _write_crontab(content: str) -> None:
    try:
        proc = subprocess.run(["crontab", "-"], input=content, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise click.ClickException("`crontab` is not available on this system.") from exc
    if proc.returncode != 0:
        raise click.ClickException(f"Failed to update crontab: {proc.stderr.strip()}")


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


# Watch defaults. The tick interval is deliberately shorter than a typical Jira
# idle-session timeout so the stored copy is renewed while it is still valid.
WATCH_DEFAULT_INTERVAL_SECONDS = 1800  # 30 minutes
WATCH_BACKOFF_BASE_SECONDS = 5
WATCH_BACKOFF_MAX_SECONDS = 300  # 5 minutes


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
        """Forget the server's stored session for an origin."""
        _emit_tool_payload(ctx, _tool_client(ctx).forget_jira_session(base_url))

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
            result = asyncio.run(jira_capture.capture_and_prevalidate(base_url, browser=browser))
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
    """Authenticate with Microsoft Teams by extracting Chrome session tokens."""

    from memforge.auth.teams_auth import TeamsAuthenticator

    authenticator = TeamsAuthenticator()
    console.print("[bold]Extracting Teams tokens from Chrome...[/]\n")

    try:
        token_data = authenticator.authenticate(region=region)
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

    console.print("\n[bold green]Done! Tokens saved to ~/.memforge/tokens/teams.json[/]")


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
