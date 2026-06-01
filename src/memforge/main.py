"""CLI entry point for MemForge."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from memforge.config import AppConfig, load_config
from memforge.tool_client import ToolClient

console = Console()
log_console = Console(stderr=True)


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


def _default_api_url(config: AppConfig) -> str:
    return os.getenv("MEMFORGE_API_URL") or f"http://127.0.0.1:{config.server.admin_api_port}"


def _tool_client(ctx) -> ToolClient:
    config: AppConfig = ctx.obj["config"]
    return ToolClient(
        api_url=_default_api_url(config),
        api_token=os.getenv("MEMFORGE_API_TOKEN"),
    )


def _emit_tool_payload(ctx, payload: dict) -> None:
    click.echo(json.dumps(payload, indent=2, ensure_ascii=False))
    if payload.get("error"):
        ctx.exit(1)


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------


@click.group()
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
@click.argument("query")
@click.option("--top-k", default=10, show_default=True, type=int, help="Maximum number of results.")
@click.option(
    "--type",
    "memory_types",
    multiple=True,
    type=click.Choice(["fact", "decision", "convention", "procedure"]),
    help="Filter by memory type. Repeat for multiple types.",
)
@click.option("--source", "sources", multiple=True, help="Filter by source name or ID. Repeat for multiple sources.")
@click.option("--entity", "entities", multiple=True, help="Entity hint. Repeat for multiple entities.")
@click.option("--after", default=None, help="Only include results after this ISO date.")
@click.option("--before", default=None, help="Only include results before this ISO date.")
@click.option("--include-superseded", is_flag=True, help="Include superseded memories.")
@click.pass_context
def search(
    ctx,
    query: str,
    top_k: int,
    memory_types: tuple[str, ...],
    sources: tuple[str, ...],
    entities: tuple[str, ...],
    after: str | None,
    before: str | None,
    include_superseded: bool,
):
    """Search MemForge using the same service path as the MCP search tool."""
    time_range = {k: v for k, v in {"after": after, "before": before}.items() if v}
    kwargs: dict = {
        "query": query,
        "top_k": top_k,
        "include_superseded": include_superseded,
    }
    if memory_types:
        kwargs["memory_types"] = list(memory_types)
    if sources:
        kwargs["sources"] = list(sources)
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
    """Fetch a source artifact URL returned by search or get-memory."""
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
    """List all configured sources."""

    async def _run():
        config: AppConfig = ctx.obj["config"]
        db = await _get_db(config)
        src_list = await db.list_sources()
        await db.close()

        if not src_list:
            console.print("[dim]No sources configured.[/]")
            return

        table = Table(title="Configured Sources")
        table.add_column("ID", style="dim")
        table.add_column("Name")
        table.add_column("Type")
        table.add_column("Status")
        table.add_column("Doc Count", justify="right")
        table.add_column("Last Sync")

        for src in src_list:
            table.add_row(
                src["id"],
                src["name"],
                src["type"],
                src.get("status", "unknown"),
                str(src.get("doc_count", 0)),
                src.get("last_sync") or "never",
            )
        console.print(table)

    asyncio.run(_run())


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
# auth group
# ---------------------------------------------------------------------------


@cli.group()
def auth():
    """Manage authentication for data sources."""
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


@auth.command("jira")
@click.option("--base-url", required=True, help="Jira base URL, for example https://jira.example.test")
@click.option("--browser", default=None, help="Browser to read cookies from, for example chrome or edge")
@click.option("--confirm-principal-change", is_flag=True, help="Allow this session to replace a different Jira user")
@click.pass_context
def auth_jira(ctx, base_url: str, browser: str | None, confirm_principal_change: bool):
    """Refresh the shared Jira browser session from local browser cookies."""

    async def _run():
        from memforge.auth.jira_auth import JiraAuthSessionError, JiraAuthSessionService, JiraPrincipalChangedError

        config: AppConfig = ctx.obj["config"]
        db = await _get_db(config)
        try:
            result = await JiraAuthSessionService(db).refresh_from_browser(
                base_url=base_url,
                browser=browser,
                confirm_principal_change=confirm_principal_change,
            )
        except JiraPrincipalChangedError as exc:
            console.print(f"[yellow]{exc}[/]")
            console.print("Re-run with [bold]--confirm-principal-change[/] to reset affected Jira sync cursors.")
            return
        except JiraAuthSessionError as exc:
            console.print(f"[red]{exc}[/]")
            return
        finally:
            await db.close()

        console.print(f"[green]Jira browser session active for {result['origin']}[/]")
        principal = result.get("principal_name") or result.get("principal_id") or "unknown user"
        console.print(f"Signed in as: [bold]{principal}[/]")
        if result.get("browser"):
            console.print(f"Browser: {result['browser']}")
        if result.get("sources_reset"):
            console.print(f"Reset sync cursor for {len(result['sources_reset'])} Jira source(s).")

    asyncio.run(_run())


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
