"""CLI entry point for MemForge."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from importlib import metadata
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Callable

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
from memforge.local_agent.folder_picker import FolderPickerCancelled, FolderPickerUnavailable, pick_folder
from memforge.local_agent.setup_results import (
    CredentialCheckResult,
    LocalMarkdownPickRootResult,
    LocalSetupJobResult,
    TeamsAuthCheckResult,
    TeamsAuthResult,
    TeamsBrowseResult,
)
from memforge.local_agent.source_contract import (
    source_processing_receipt,
)
from memforge.retrieval.intents import RANKED_RETRIEVAL_INTENTS
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

if TYPE_CHECKING:
    from memforge.local_agent.document_acquisition import (
        AcquiredInputPackage,
        AcquisitionAnswers,
        AcquisitionManifest,
        AcquisitionRequests,
    )
    from memforge.models import ContentItem
    from memforge.source_artifacts import RawSourceArtifact

console = Console()
log_console = Console(stderr=True)
logger = logging.getLogger(__name__)

DEFAULT_CLI_CONFIG_PATH = Path.home() / ".memforge" / "cli.toml"
DEFAULT_LOCAL_AGENT_STATE_PATH = Path.home() / ".memforge" / "local-agent-state.json"
DEFAULT_LOCAL_AGENT_LOCK_PATH = Path.home() / ".memforge" / "local-agent-daemon.lock"
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

    httpx logs only warnings: its INFO line names each request's full URL,
    query included, and a provider's signed download URL carries a live
    signature there. MemForge's own request logs name the address alone.
    """
    logging.getLogger("httpx").setLevel(logging.WARNING)
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


def _this_device_report(server: str, sources: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """This installation's device report to the server at API origin ``server``, as of now.

    It names each of ``sources``, the Sources the server's latest lease answer
    says this device executes, with whether this device is ready to run it,
    which can change at any time.
    """
    from memforge.local_agent.device_credentials import device_credentials
    from memforge.local_agent.device_identity import DeviceIdentityStore, device_identity_path
    from memforge.local_agent.device_report import this_device_report
    from memforge.local_agent.device_sources import readiness_of_sources

    identity = DeviceIdentityStore(device_identity_path()).ensure()
    readiness = readiness_of_sources(sources, device_credentials(server))
    return this_device_report(identity.device_id, identity.name, readiness).to_payload()


def _build_local_agent_runner(
    ctx,
    *,
    browser: str | None,
):
    from memforge.local_agent.runner import LocalAgentRunner

    from memforge.local_agent.device_credentials import device_credentials
    from memforge.local_agent.device_identity import DeviceIdentityStore, device_identity_path
    from memforge.local_agent.device_lease import DeviceLeaseAnswers

    client = _local_agent_tool_client(ctx)
    server = client.target.origin
    device_id = DeviceIdentityStore(device_identity_path()).ensure().device_id
    lease_answers = DeviceLeaseAnswers(device_credentials(server), log=lambda message: click.echo(message, err=True))

    def device() -> dict[str, Any]:
        return _this_device_report(server, lease_answers.sources)

    def lease(limit: int = 1, wait_seconds: int = 0, lease_seconds: int = 60) -> dict[str, Any]:
        answer = client.lease_local_agent_jobs(
            device=device(),
            limit=limit,
            wait_seconds=wait_seconds,
            lease_seconds=lease_seconds,
        )
        if not answer.get("error"):
            lease_answers.observe(answer)
        return answer

    return LocalAgentRunner(
        state_store=_local_agent_state_store(),
        cloud_job_handler=lambda job, report_progress=None: _run_cloud_local_agent_job(
            job, client, browser=browser, report_progress=report_progress
        ),
        cloud_jobs_provider=lease,
        cloud_job_completer=lambda job_id, attempt_count, status, result, error=None: client.complete_local_agent_job(
            job_id,
            device_id=device_id,
            attempt_count=attempt_count,
            status=status,
            result=result,
            error=error,
        ),
        cloud_job_heartbeat=lambda job_id, attempt_count, lease_seconds, progress=None: (
            client.heartbeat_local_agent_job(
                job_id,
                attempt_count=attempt_count,
                device=device(),
                lease_seconds=lease_seconds,
                progress=progress,
            )
        ),
    )


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


@eval_group.command("judge-calibrate")
@click.option(
    "--dir",
    "directory",
    default=Path("evals/quality/judge-calibration"),
    show_default=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="The folder of calibration cases.",
)
@click.option(
    "--split",
    type=click.Choice(["dev", "test", "all"]),
    default="all",
    show_default=True,
    help="Which calibration cases to judge.",
)
@click.pass_context
def eval_judge_calibrate(ctx: click.Context, directory: Path, split: str) -> None:
    """Judge labeled calibration cases with the configured expectation judge and print its agreement.

    The judge is the one MEMFORGE_AGENT_EVALUATION_JUDGE_* configures. Run it
    after changing how the judge reads a case kind or which model judges, and
    put the test split's summary in the pull request. Exits 0 once every case
    is judged, and 2 when no judge is configured or a calibration file is
    malformed.
    """

    from memforge.evals.judge_calibration import (
        CALIBRATION_SPLITS,
        calibrate,
        load_calibration,
        render_calibration_markdown,
    )
    from memforge.evals.offline_runtime import build_expectation_judge
    from memforge.llm.providers import ModelConfigError
    from memforge.runtime import DefaultRuntimeProvider

    logging.getLogger().setLevel(logging.WARNING)

    async def _run() -> str:
        built = await build_expectation_judge(ctx.obj["config"], DefaultRuntimeProvider())
        if built is None:
            raise ValueError("No expectation judge is configured: set MEMFORGE_AGENT_EVALUATION_JUDGE_*.")
        spec, judge = built
        splits = CALIBRATION_SPLITS if split == "all" else (split,)
        return render_calibration_markdown(spec, await calibrate(judge, cases, splits=splits))

    try:
        cases = load_calibration(directory)
        summary = asyncio.run(_run())
    except (ValueError, ModelConfigError) as exc:
        click.echo(str(exc), err=True)
        ctx.exit(2)
    click.echo(summary)


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
    ``cases``, each a ``doc_id`` with its ``rubric`` (``expectations``:
    sentences the extracted Memories should meet) and, to pin a derivation
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
    ``cases``, each a ``doc_id`` with its ``rubric`` (``expectations``:
    sentences about the Support verdicts) and, to pin an update other
    than the latest one the model judged Support in, its ``derivation_id``.
    Requires a maintenance operator.
    """

    _emit_tool_payload(
        ctx,
        _tool_client(ctx, workspace_id).seed_support_evaluation_cases(
            _read_json_object(spec_path, "Support case spec")
        ),
    )


@eval_group.command("seed-reconciliation-cases")
@click.option(
    "--spec",
    "spec_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@_workspace_id_option
@click.pass_context
def eval_seed_reconciliation_cases(ctx: click.Context, spec_path: Path, workspace_id: str | None) -> None:
    """Import hand-written reconciliation situations as cases and freeze their cohort.

    The spec names ``selection_policy_version``, an optional ``role`` and the
    ``cases``, each a synced Document (``doc_id``, or ``source_id`` and
    ``title``) with its ``input`` (``candidates``, ``incumbents`` and their
    ``support``) and ``rubric`` (``expectations``: sentences about the
    incumbents' outcomes). Requires a maintenance operator.
    """

    _emit_tool_payload(
        ctx,
        _tool_client(ctx, workspace_id).seed_reconciliation_evaluation_cases(
            _read_json_object(spec_path, "Reconciliation case spec")
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
    """Create directories and initialise the database."""
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

    async def _initialise_database():
        db = await _get_db(config)
        await db.close()

    asyncio.run(_initialise_database())

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
        daemon_client = _daemon_client(manager)
        connection = daemon_client.get_local_agent_status()
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
    payload: dict[str, Any] = {
        "status": "healthy" if healthy else "unhealthy",
        "service": service,
        "runtime": runtime,
        "connection": connection,
    }
    if not connection.get("error"):
        payload["sources"] = _device_source_states(connection, daemon_client.target.origin)
    return payload


def _daemon_client(manager) -> ToolClient:
    """A client for the server the installed daemon serves, as the daemon's user."""
    environment = manager.environment()
    target = _build_cli_target(
        api_url=environment["MEMFORGE_API_URL"],
        edition=environment.get("MEMFORGE_EDITION"),
    )
    return ToolClient(target=target, api_token=environment.get("MEMFORGE_API_TOKEN"))


def _device_source_states(status: Mapping[str, Any], server: str) -> list[dict[str, Any]]:
    """Each Source this device executes on ``server``: ready, or waiting for an update, a credential or its folder."""
    from memforge.local_agent.device_credentials import device_credentials
    from memforge.local_agent.device_identity import DeviceIdentityStore, device_identity_path
    from memforge.local_agent.device_sources import device_source_states

    try:
        identity = DeviceIdentityStore(device_identity_path()).load()
        if identity is None:
            return []
        return device_source_states(status, device_credentials(server), device_id=identity.device_id)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


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


def _reported_device(status: Mapping[str, Any], device_id: str) -> dict[str, Any] | None:
    """One device of a ``/local-agent/status`` answer, by id."""
    for device in status.get("devices") or ():
        if isinstance(device, Mapping) and device.get("device_id") == device_id:
            return dict(device)
    return None


def _wait_for_local_agent_online(
    client: ToolClient,
    *,
    device_id: str,
    after_last_seen_at: str | None,
    timeout_seconds: float = 35.0,
) -> dict[str, Any]:
    """Wait for a report from this installation's own device newer than ``after_last_seen_at``."""
    deadline = time.monotonic() + timeout_seconds
    baseline = _status_timestamp(after_last_seen_at)
    latest: dict[str, Any] = {"status": "offline"}
    while True:
        latest = client.get_local_agent_status()
        if latest.get("error"):
            return latest
        device = _reported_device(latest, device_id) or {
            "device_id": device_id,
            "status": "offline",
            "last_seen_at": None,
        }
        observed = _status_timestamp(device.get("last_seen_at"))
        fresh = observed is not None and (baseline is None or observed > baseline)
        if device.get("status") == "online" and fresh:
            return {**device, "fresh": True}
        if time.monotonic() >= deadline:
            return {**device, "fresh": False}
        time.sleep(0.5)


@cli.command("upgrade")
@click.pass_context
def upgrade_command(ctx):
    """Install exactly the MemForge version the server runs, and restart the daemon with it.

    A device syncs its document Sources only while its acquisition contracts
    equal the server's, so this also brings back a device that is ahead of
    the server. MemForge is installed the way it was installed before.
    """
    from memforge.local_agent.device_identity import DeviceIdentityStore, device_identity_path
    from memforge.local_agent.service import DaemonServiceError
    from memforge.local_agent.upgrade import UnknownInstallMethod, install_method
    from memforge.release import installed_version

    manager = _daemon_service_manager()
    try:
        environment = manager.environment()
        client = _daemon_client(manager)
    except DaemonServiceError as exc:
        raise click.ClickException(f"{exc} Run `memforge setup` first.") from exc
    status = client.get_local_agent_status()
    if status.get("error"):
        _emit_tool_payload(ctx, status)
        return
    version = str((status.get("server") or {}).get("version") or "")
    if not version:
        raise click.ClickException("The server did not name its MemForge version.")
    previous_version = installed_version()
    if version == previous_version:
        _emit_tool_payload(ctx, {"ok": True, "version": version, "upgraded": False})
        return
    executable = _current_cli_executable()
    try:
        method = install_method(Path(executable))
    except UnknownInstallMethod as exc:
        raise click.ClickException(
            f"{exc}. Install MemForge {version} the way you installed it, then run `memforge daemon restart`."
        ) from exc
    command = method.install_command(version)
    installed = subprocess.run(command, check=False)
    if installed.returncode != 0:
        raise click.ClickException(f"`{' '.join(command)}` failed with exit status {installed.returncode}.")
    device = DeviceIdentityStore(device_identity_path()).ensure()
    try:
        replacement = manager.replace(
            executable=executable,
            api_url=environment["MEMFORGE_API_URL"],
            api_token=environment.get("MEMFORGE_API_TOKEN"),
            edition=environment.get("MEMFORGE_EDITION"),
        )
    except DaemonServiceError as exc:
        raise click.ClickException(str(exc)) from exc
    daemon_status = _wait_for_local_agent_online(
        client,
        device_id=device.device_id,
        after_last_seen_at=(_reported_device(status, device.device_id) or {}).get("last_seen_at"),
    )
    if daemon_status.get("fresh") is not True or daemon_status.get("version") != version:
        restore_command = method.install_command(previous_version)
        reinstalled = subprocess.run(restore_command, check=False)
        if reinstalled.returncode == 0:
            reinstall_outcome = f"MemForge {previous_version} was installed again"
        else:
            reinstall_outcome = (
                f"`{' '.join(restore_command)}` failed with exit status {reinstalled.returncode}, "
                f"so MemForge {version} stays installed"
            )
        try:
            replacement.rollback()
        except DaemonServiceError as rollback_error:
            raise click.ClickException(
                f"The daemon did not report MemForge {version} after the restart; {reinstall_outcome}, "
                f"and its service could not be restored: {rollback_error}"
            ) from rollback_error
        raise click.ClickException(
            f"The daemon did not report MemForge {version} after the restart; {reinstall_outcome}, "
            "and its service was restored. Run `memforge daemon logs`."
        )
    try:
        replacement.commit()
    except DaemonServiceError as exc:
        raise click.ClickException(f"Could not commit the daemon restart: {exc}") from exc
    _emit_tool_payload(ctx, {"ok": True, "version": version, "upgraded": True, "install_method": method.name})


@cli.command("setup")
@click.option("--api-url", default=None, help="MemForge API URL. Uses the active target or prompts when absent.")
@click.option("--target-name", default=None, help="Name used when saving a newly configured target.")
@click.option(
    "--token-env", default="MEMFORGE_API_TOKEN", show_default=True, help="Environment variable holding the token."
)
@click.option("--device-name", default=None, help="Name of this computer in MemForge. Defaults to its host name.")
@click.pass_context
def setup_command(ctx, api_url: str | None, target_name: str | None, token_env: str, device_name: str | None):
    """Configure a target, name this device and install the local daemon user service.

    The device takes its new name once the setup is committed; a setup that
    is rolled back leaves the device as it was named.
    """
    from memforge.api_target import Edition
    from memforge.local_agent.device_identity import DeviceIdentityStore, checked_device_name, device_identity_path
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
    identities = DeviceIdentityStore(device_identity_path())
    try:
        new_device_name = checked_device_name(device_name)
        device = identities.ensure()
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
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
        device_id=device.device_id,
        after_last_seen_at=(_reported_device(baseline_status, device.device_id) or {}).get("last_seen_at"),
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
    identities.ensure(new_device_name)
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
        from memforge.genes.family import CollectionFamilyKind
        from memforge.runtime import build_sync_runtime, run_source_sync
        from memforge.source_types import source_type_descriptor

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
            descriptor = source_type_descriptor(source_type)
            if descriptor is not None and descriptor.family is CollectionFamilyKind.CONVERSATION:
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


@cli.group("source")
def source_group():
    """Manage one Source on this device."""


@source_group.group("credentials")
def source_credentials():
    """The provider credentials this device holds for a device Source."""


@source_credentials.command("set")
@click.argument("source_id")
@click.pass_context
def source_credentials_set(ctx, source_id: str):
    """Store a device Source's secrets in this device's keyring, prompting for each secret field of its form."""
    from memforge.local_agent.credential_check import signs_in_with_browser_session
    from memforge.local_agent.device_credentials import device_credentials
    from memforge.local_agent.device_identity import DeviceIdentityStore, device_identity_path
    from memforge.local_agent.device_sources import SIGN_IN_INSTRUCTION, credential_fields
    from memforge.local_agent.service import DaemonServiceError

    try:
        daemon_client = _daemon_client(_daemon_service_manager())
        status = daemon_client.get_local_agent_status()
    except DaemonServiceError as exc:
        raise click.ClickException(f"{exc} Run `memforge setup` first.") from exc
    if status.get("error"):
        _emit_tool_payload(ctx, status)
        return
    source = next((entry for entry in status.get("sources") or () if entry.get("id") == source_id), None)
    if source is None:
        raise click.ClickException(
            f"Source {source_id} is not one of your Sources that run on your computer. "
            "Choose My computer under Where it runs in its Configure page first."
        )
    identity = DeviceIdentityStore(device_identity_path()).load()
    if identity is None or source.get("execution_device_id") != identity.device_id:
        raise click.ClickException(
            f"Source {source_id} runs on another of your devices. Store its credentials there, or choose this "
            "device under Where it runs in its Configure page first."
        )
    source_type = str(source["type"])
    config = dict(source.get("config") or {})
    if signs_in_with_browser_session(source_type, config):
        raise click.ClickException(f"Source {source_id} signs in with a browser session. {SIGN_IN_INSTRUCTION}.")
    secrets: dict[str, str] = {}
    for field in credential_fields(source_type, config):
        value = click.prompt(field.label, hide_input=True, default="" if not field.required else None)
        if value:
            secrets[field.key] = value
    try:
        device_credentials(daemon_client.target.origin).set(source_id, secrets)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    _emit_tool_payload(ctx, {"ok": True, "source_id": source_id, "fields": sorted(secrets)})


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


def _run_cloud_local_agent_job(
    job: dict[str, Any],
    client: ToolClient,
    *,
    browser: str | None = None,
    report_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    from memforge.local_agent.document_acquisition import DOCUMENT_ACQUISITION_OPERATION

    operation = str(job.get("operation") or "").strip()
    handlers = {
        "local_markdown_pick_root": lambda: _run_cloud_pick_root_job(job),
        "credential_check": lambda: _run_cloud_credential_check_job(job, client, browser=browser),
        "teams_auth_check": lambda: _run_cloud_teams_auth_check_job(job),
        "teams_auth": lambda: _run_cloud_teams_auth_job(job),
        "teams_browse": lambda: _run_cloud_teams_browse_job(job),
        "teams_sync": lambda: _run_cloud_teams_sync_job(
            job,
            client,
            report_progress=report_progress,
        ),
        DOCUMENT_ACQUISITION_OPERATION: lambda: _run_cloud_document_acquisition_job(
            job, client, report_progress=report_progress
        ),
    }
    handler = handlers.get(operation)
    if handler is None:
        return {
            "operation": operation,
            "error": f"unsupported cloud local-agent job operation: {operation or '<empty>'}",
        }
    return handler()


class _ClientAcquisitionTransport:
    """Hands one job attempt's acquisition to the server through the API client."""

    def __init__(self, client: ToolClient, *, source_id: str, job_id: str, attempt_count: int) -> None:
        self._client = client
        self._source_id = source_id
        self._job_id = job_id
        self._attempt_count = attempt_count
        self._manifest_sha256 = ""

    def _checked(self, response: dict[str, Any], what: str) -> dict[str, Any]:
        _raise_if_local_agent_lease_not_current(response)
        if response.get("error"):
            raise RuntimeError(f"{what} failed: {response.get('detail') or response.get('error')}")
        return response

    async def submit_manifest(self, manifest: AcquisitionManifest) -> AcquisitionRequests:
        from memforge.local_agent.document_acquisition import AcquisitionRequests

        response = self._checked(
            await asyncio.to_thread(
                self._client.submit_acquisition_manifest,
                source_id=self._source_id,
                local_agent_job_id=self._job_id,
                local_agent_attempt_count=self._attempt_count,
                manifest=manifest.to_payload(),
            ),
            "submitting the acquisition manifest",
        )
        self._manifest_sha256 = str(response["manifest_sha256"])
        return AcquisitionRequests.from_payload(response.get("requests") or {})

    async def stored_extras(self, item_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        response = self._checked(
            await asyncio.to_thread(
                self._client.read_acquisition_stored_extras,
                source_id=self._source_id,
                local_agent_job_id=self._job_id,
                local_agent_attempt_count=self._attempt_count,
                item_ids=list(item_ids),
            ),
            "reading the stored extras",
        )
        return {str(item_id): dict(extra) for item_id, extra in (response.get("stored_extras") or {}).items()}

    async def upload_artifact(self, item: ContentItem, artifact: RawSourceArtifact, body: bytes) -> str:
        response = self._checked(
            await asyncio.to_thread(
                self._client.push_source_artifact,
                source_id=self._source_id,
                source_unit_key=item.item_id,
                provider_key=artifact.provider_key,
                provider_revision=artifact.provider_revision,
                parent_observation_type=artifact.parent_observation_type,
                parent_provider_key=artifact.parent_provider_key,
                filename=artifact.filename,
                media_type=artifact.media_type,
                content=body,
                local_agent_job_id=self._job_id,
                local_agent_attempt_count=self._attempt_count,
            ),
            f"uploading artifact {artifact.provider_key}",
        )
        return str(response["input_sha256"])

    async def held_bodies(self, body_sha256s: Sequence[str]) -> set[str]:
        response = self._checked(
            await asyncio.to_thread(
                self._client.read_held_acquired_bodies,
                source_id=self._source_id,
                local_agent_job_id=self._job_id,
                local_agent_attempt_count=self._attempt_count,
                body_sha256s=list(body_sha256s),
            ),
            "asking which bodies the server holds",
        )
        return {str(digest) for digest in response.get("held") or ()}

    async def upload_input(self, package: AcquiredInputPackage, body: bytes | None) -> None:
        from memforge.local_agent.document_acquisition import ACQUIRED_BODY_REQUIRED, AcquiredBodyRequired

        response = await asyncio.to_thread(
            self._client.upload_acquired_input,
            source_id=self._source_id,
            local_agent_job_id=self._job_id,
            local_agent_attempt_count=self._attempt_count,
            package=package.to_payload(),
            body=body,
        )
        if response.get("status_code") == HTTPStatus.CONFLICT and response.get("detail") == ACQUIRED_BODY_REQUIRED:
            raise AcquiredBodyRequired(package.body_sha256)
        self._checked(response, f"uploading the acquired input of {package.item.item_id}")

    async def submit_answers(self, answers: AcquisitionAnswers) -> None:
        self._checked(
            await asyncio.to_thread(
                self._client.submit_acquisition_answers,
                source_id=self._source_id,
                local_agent_job_id=self._job_id,
                local_agent_attempt_count=self._attempt_count,
                manifest_sha256=self._manifest_sha256,
                answers=answers.to_payload(),
            ),
            "submitting the acquisition answers",
        )


def _run_cloud_document_acquisition_job(
    job: dict[str, Any],
    client: ToolClient,
    *,
    report_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Acquire one document Source's run with its type's own Gene, then ask the server to process it.

    The Gene runs with the credentials this device holds for the Source, and
    a browser session is checked against the provider before anything is
    read. A missing credential or a refused session fails the job without a
    retry, because only the owner can renew it. A manifest above the bound
    every device manifest has fails the job with the count, and does not
    retry either: the same scope is as large next time.
    """

    from memforge.genes import gene_class
    from memforge.llm.structured import failure_retryable
    from memforge.local_agent.acquisition_daemon import acquire_document_source
    from memforge.local_agent.device_credentials import check_browser_session, device_credentials
    from memforge.local_agent.document_acquisition import CONFIG_PAYLOAD_FIELD
    from memforge.local_agent.runner import CloudJobLeaseLost
    from memforge.source_types import source_type_descriptor

    operation = str(job.get("operation") or "").strip()
    payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
    source_id, client, error = _local_agent_job_source_client(job, payload, operation=operation, client=client)
    if error:
        return error
    source_type = str(payload.get("source_type") or job.get("source_type") or "")
    job_id = str(job["job_id"])
    attempt_count = int(job["attempt_count"])
    gene_cls = gene_class(source_type)
    descriptor = source_type_descriptor(source_type)
    credentials = device_credentials(client.target.origin)

    def open_gene(config: Mapping[str, Any]) -> Any:
        return gene_cls(config=credentials.gene_config(source_id, source_type, config), source_id=source_id)

    async def acquire() -> Any:
        await check_browser_session(
            source_type,
            credentials.gene_config(source_id, source_type, payload.get(CONFIG_PAYLOAD_FIELD) or {}),
        )
        return await acquire_document_source(
            payload,
            open_gene=open_gene,
            transport=_ClientAcquisitionTransport(client, source_id=source_id, job_id=job_id, attempt_count=attempt_count),
            exports_pdf=bool(descriptor and descriptor.exports_pdf),
        )

    _report_local_agent_progress(report_progress, _sync_progress_snapshot(phase="connecting"))
    try:
        summary = asyncio.run(acquire())
    except CloudJobLeaseLost:
        raise
    except Exception as exc:
        return {
            "operation": operation,
            "source_id": source_id,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "retryable": failure_retryable(exc),
        }
    sync_result = client.start_source_processing(
        source_id=source_id,
        local_agent_job_id=job_id,
        local_agent_attempt_count=attempt_count,
    )
    _raise_if_local_agent_lease_not_current(sync_result)
    result: dict[str, Any] = {
        "operation": operation,
        "source_id": source_id,
        "counts": {"discovered": summary.discovered, "uploaded": summary.uploaded, "failed": summary.failed},
        "sync_started": not bool(sync_result.get("error")),
        **source_processing_receipt(sync_result),
    }
    if sync_result.get("error"):
        result.update({"error": "source sync failed to start", "sync_error": sync_result, "retryable": True})
    return result


def _setup_result(result: LocalSetupJobResult) -> dict[str, Any]:
    """Complete a setup job with every field its model declares, null where the handler set none."""
    return result.model_dump()


def _run_cloud_credential_check_job(
    job: dict[str, Any],
    client: ToolClient,
    *,
    browser: str | None,
) -> dict[str, Any]:
    from memforge.genes import gene_class
    from memforge.local_agent.credential_check import CREDENTIAL_CHECK_OPERATION, run_credential_check
    from memforge.local_agent.device_credentials import device_credentials

    payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
    source_type = str(payload.get("source_type") or "")
    _source_id, scoped_client, error = _local_agent_job_source_client(
        job, payload, operation=CREDENTIAL_CHECK_OPERATION, client=client
    )
    if error:
        return _setup_result(CredentialCheckResult(**error, retryable=False))
    result = asyncio.run(
        run_credential_check(
            payload,
            credentials=device_credentials(client.target.origin),
            open_gene=lambda source_id, config: gene_class(source_type)(config=dict(config), source_id=source_id),
            upload_session=lambda provider, captured: scoped_client.upload_browser_session(
                provider=provider,
                base_url=captured.origin,
                cookie_header=captured.cookie_header,
                browser=captured.browser,
            ),
            browser=browser,
        )
    )
    return _setup_result(result)


def _local_agent_lease_not_current(response: object) -> bool:
    """Whether the server answered that the attempt is no longer this device's.

    A submission under an acquisition contract the server no longer has
    queues the job again, so it ends the attempt as a lost lease does.
    """
    from memforge.local_agent.source_contract import ACQUISITION_CONTRACT_CHANGED

    return (
        isinstance(response, dict)
        and response.get("status_code") == HTTPStatus.CONFLICT
        and response.get("detail") in {LOCAL_AGENT_LEASE_NOT_CURRENT, ACQUISITION_CONTRACT_CHANGED}
    )


def _raise_if_local_agent_lease_not_current(response: object) -> None:
    if not _local_agent_lease_not_current(response):
        return
    from memforge.local_agent.runner import CloudJobLeaseLost

    raise CloudJobLeaseLost(str(response["detail"]))  # type: ignore[index]


def _run_cloud_pick_root_job(job: dict[str, Any]) -> dict[str, Any]:
    operation = str(job.get("operation") or "").strip()
    payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
    try:
        root = pick_folder(
            title=str(payload.get("title") or "Choose folder to sync"),
            initial_directory=str(payload.get("initial_directory") or "").strip() or None,
        )
    except FolderPickerCancelled:
        return _setup_result(LocalMarkdownPickRootResult(operation=operation, cancelled=True))
    except FolderPickerUnavailable as exc:
        return _setup_result(LocalMarkdownPickRootResult(operation=operation, error=str(exc)))
    return _setup_result(LocalMarkdownPickRootResult(operation=operation, root=root))


def _run_cloud_teams_auth_job(job: dict[str, Any]) -> dict[str, Any]:
    from memforge.auth import teams_auth
    from memforge.genes.teams_provider import teams_region

    operation = str(job.get("operation") or "").strip()
    payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
    region = teams_region(payload)
    try:
        token_data = teams_auth.TeamsAuthenticator().authenticate(
            region=region,
            wait_seconds=_cloud_job_limit(payload.get("wait_seconds"), default=90),
            poll_interval_seconds=2.0,
            rejected_token_hashes=_teams_rejected_token_hashes_from_payload(payload),
        )
    except RuntimeError as exc:
        return _setup_result(TeamsAuthResult(operation=operation, authenticated=False, region=region, error=str(exc)))
    except Exception as exc:
        return _setup_result(
            TeamsAuthResult(
                operation=operation,
                authenticated=False,
                region=region,
                error=str(exc),
                error_type=type(exc).__name__,
            )
        )

    tokens = token_data.get("tokens") if isinstance(token_data, dict) else {}
    return _setup_result(
        TeamsAuthResult(
            operation=operation,
            authenticated=True,
            region=region,
            token_count=len(tokens) if isinstance(tokens, dict) else 0,
        )
    )


def _run_cloud_teams_auth_check_job(job: dict[str, Any]) -> dict[str, Any]:
    from memforge.local_agent.teams_browse import teams_auth_status
    from memforge.genes.teams_provider import teams_region

    operation = str(job.get("operation") or "").strip()
    payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
    return _setup_result(TeamsAuthCheckResult(operation=operation, region=teams_region(payload), **teams_auth_status()))


def _run_cloud_teams_browse_job(job: dict[str, Any]) -> dict[str, Any]:
    from memforge.local_agent.teams_browse import browse_teams_conversations
    from memforge.genes.teams_provider import teams_region

    operation = str(job.get("operation") or "").strip()
    payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
    region = teams_region(payload)
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
                    return _setup_result(
                        TeamsBrowseResult(
                            operation=operation,
                            region=region,
                            error=str(retry_exc),
                            error_type=type(retry_exc).__name__,
                        )
                    )
            else:
                return _setup_result(
                    TeamsBrowseResult(
                        operation=operation,
                        region=region,
                        **{key: reauth[key] for key in ("error", "error_type") if key in reauth},
                    )
                )
        else:
            return _setup_result(
                TeamsBrowseResult(operation=operation, region=region, error=str(exc), error_type=type(exc).__name__)
            )
    return _setup_result(TeamsBrowseResult(operation=operation, region=region, **data))


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

    from memforge.local_agent.conversation_executor import ConversationUploadError, RecordSchemaMismatch

    operation = str(job.get("operation") or "").strip()
    payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
    source_id, client, error = _local_agent_job_source_client(job, payload, operation=operation, client=client)
    if error:
        return error
    _report_local_agent_progress(report_progress, _sync_progress_snapshot(phase="connecting"))
    try:
        try:
            result = asyncio.run(_collect_conversation_on_device(job, payload, source_id, client, report_progress))
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
            result = asyncio.run(_collect_conversation_on_device(job, payload, source_id, client, report_progress))
    except ConversationUploadError as exc:
        _raise_if_local_agent_lease_not_current(exc.response)
        return {"operation": operation, "source_id": source_id, "error": str(exc), "retryable": True}
    except RecordSchemaMismatch as exc:
        # No attempt succeeds until the daemon or the server is upgraded; the next scheduled sync asks again.
        return {"operation": operation, "source_id": source_id, "error": str(exc), "retryable": False}
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


async def _collect_conversation_on_device(
    job: dict[str, Any],
    payload: dict[str, Any],
    source_id: str,
    client: ToolClient,
    report_progress: Callable[[dict[str, Any]], None] | None,
):
    """Collect one attempt of a conversation Source through the provider its descriptor names."""

    from memforge.local_agent.conversation_executor import run_device_collection
    from memforge.source_types import source_type_descriptor

    descriptor = source_type_descriptor(str(job.get("source_type") or ""))
    if descriptor is None or descriptor.conversation_provider is None:
        raise ValueError(f"source type {job.get('source_type')!r} has no conversation provider")

    def report(phase: str, completed: int, total: int) -> None:
        unit = "conversation" if phase == "fetching" else "message"
        _report_local_agent_progress(
            report_progress,
            _sync_progress_snapshot(phase=phase, completed=completed, total=total, unit=unit),
        )

    async with descriptor.conversation_provider.open(payload) as provider:
        return await run_device_collection(
            provider,
            client,
            source_id=source_id,
            local_agent_job_id=str(job["job_id"]),
            local_agent_attempt_count=int(job["attempt_count"]),
            config_revision=str(payload.get("source_config_revision") or ""),
            dry_run=bool(payload.get("dry_run")),
            force_full_sync=bool(payload.get("force_full_sync", False)),
            report=report,
        )


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


def _cloud_job_limit(value: Any, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(parsed, 0)


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


@maintenance.command("migrate-device-packages")
@click.option("--check", "mode", flag_value="check", default=True, help="List the Sources and what applying would do.")
@click.option("--apply", "mode", flag_value="apply", help="Take a recovery point, then move each Source.")
@click.option("--source", "source_ids", multiple=True, help="Only this Source (repeatable).")
@click.option("--json", "as_json", is_flag=True, help="Print the result as JSON.")
@click.pass_context
def maintenance_migrate_device_packages(ctx, mode: str, source_ids: tuple[str, ...], as_json: bool):
    """Move Jira, GitHub repository and local Markdown Sources off their device package path onto device acquisition.

    A Jira Source set to local agent sync, a GitHub repository Source set to
    local push, or a local Markdown Source whose files the daemon uploaded to
    a server inbox, holds what the device submitted as packages. Each such
    Source moves once: every package its Units hold is rewritten to the input
    the type's Gene reads, replayed without forced extraction, so only Units
    whose content the rewrite changes are extracted, and the Source then runs
    on its owner's device. Jira issues move to their jira-<KEY> Documents with
    their Memories' Support.

    --check, the default, changes nothing. It names the Units whose content
    the rewrite changes, and refuses a Source with a package the Gene's form
    cannot be rebuilt from, with two Units that map to one Document, or with a
    GitHub file whose Document the configured repository names differently.
    --apply first takes a recovery point (`memforge backup list`), so stop
    the MemForge server before it. A Source whose replay fails keeps its
    package path and moves the next time --apply runs. Until a Source moves,
    it does not sync.
    """

    async def _run() -> int:
        from memforge.pipeline.package_migration import (
            apply_package_migration,
            package_path_sources,
            plan_package_migration,
        )
        from memforge.runtime import DefaultRuntimeProvider
        from memforge.server.admin_api import SELF_HOSTED_WORKSPACE_ID
        from memforge.storage.document_store import LocalDocumentStore
        from memforge.storage.recovery_points import PRE_MIGRATION, UpgradeError

        config: AppConfig = ctx.obj["config"]
        db = await _get_db(config)
        try:
            sources = [
                source for source in await package_path_sources(db) if not source_ids or source["id"] in source_ids
            ]
            if mode == "check":
                doc_store = LocalDocumentStore(config.storage.docs_path)
                plans = [await plan_package_migration(db, doc_store, source) for source in sources]
                _print_package_migration([plan.to_payload() for plan in plans], as_json=as_json)
                if plans and not as_json:
                    console.print("Run `memforge maintenance migrate-device-packages --apply` to move them.")
                return 1 if any(plan.refused for plan in plans) else 0
            if not sources:
                _print_package_migration([], as_json=as_json)
                return 0
            try:
                point = await db.take_recovery_point(PRE_MIGRATION)
            except UpgradeError as exc:
                raise click.ClickException(str(exc)) from exc
            if not as_json:
                console.print(f"Took recovery point {point.id}; `memforge backup restore {point.id}` undoes this run.")
            runtime = await DefaultRuntimeProvider().build_sync_runtime(db, config)
            results = [
                await apply_package_migration(
                    db, config, str(source["id"]), runtime=runtime, workspace_id=SELF_HOSTED_WORKSPACE_ID
                )
                for source in sources
            ]
            _print_package_migration([result.to_payload() for result in results], as_json=as_json)
            return 0 if all(result.status == "migrated" for result in results) else 1
        finally:
            await db.close()

    raise SystemExit(asyncio.run(_run()))


def _print_package_migration(rows: list[dict[str, Any]], *, as_json: bool) -> None:
    if as_json:
        click.echo(json.dumps({"sources": rows}, indent=2))
        return
    if not rows:
        console.print("No Source is on a device package path.")
        return
    table = Table(title="Sources on a device package path")
    table.add_column("Source", style="bold")
    table.add_column("Type")
    table.add_column("Packages", justify="right")
    table.add_column("Changed", justify="right")
    table.add_column("Moved", justify="right")
    table.add_column("Result")
    for row in rows:
        if row["refused"]:
            causes = [
                f"{len(row['unrebuildable'])} item(s) not rebuildable" if row["unrebuildable"] else "",
                f"{len(row['collisions'])} collision(s)" if row["collisions"] else "",
                f"{len(row['id_changes_refused'])} file(s) the configured repository names differently"
                if row["id_changes_refused"]
                else "",
            ]
            result = "[red]refused[/]: " + ", ".join(cause for cause in causes if cause)
        else:
            result = {
                None: "[green]ready[/]",
                "migrated": "[green]moved to the device[/]",
                "replay_failed": f"[red]replay {row.get('sync_status')}[/]: {row.get('error') or ''}",
            }.get(row.get("status"), str(row.get("status")))
        table.add_row(
            f"{row['source_name']} ({row['source_id']})",
            row["source_type"],
            str(row["units"]),
            str(len(row["changed_document_ids"])),
            str(len(row["moved_document_ids"])),
            result,
        )
    console.print(table)


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
