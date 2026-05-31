"""MCP server exposing memory retrieval tools to LLM agents.

Phase 1 implementation: basic vector search via ChromaDB, with SQLite for
full memory details and provenance. Hybrid retrieval (BM25 + vector + RRF)
is planned for Phase 2.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Resource, TextContent, Tool

from memforge.config import AppConfig
from memforge.models import MemoryType
from memforge.provenance import document_content_uri, document_content_url, document_pdf_url
from memforge.storage.database import Database

logger = logging.getLogger(__name__)


def create_mcp_server(
    db: Database,
    config: AppConfig,
) -> Server:
    """Build an MCP Server with MemForge tools and resources."""

    server = Server("memforge")

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="search",
                description=(
                    "Search the team memory layer for facts, decisions, conventions, and procedures. "
                    "Returns ranked results with provenance (source documents), confidence scores, "
                    "and entity links. Use this as the primary way to query team knowledge."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query",
                        },
                        "memory_types": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["fact", "decision", "convention", "procedure"],
                            },
                            "description": "Filter to specific memory types",
                        },
                        "sources": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter to specific source IDs",
                        },
                        "time_range": {
                            "type": "object",
                            "properties": {
                                "after": {
                                    "type": "string",
                                    "description": "ISO date — only memories created/updated after this date",
                                },
                                "before": {
                                    "type": "string",
                                    "description": "ISO date — only memories created/updated before this date",
                                },
                            },
                        },
                        "entities": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter to memories linked to these entities",
                        },
                        "include_superseded": {
                            "type": "boolean",
                            "default": False,
                            "description": "Include superseded memories in results (default false)",
                        },
                        "top_k": {
                            "type": "integer",
                            "default": 10,
                            "description": "Number of results to return",
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="get_memory",
                description=(
                    "Fetch the full detail of a single memory by its ID. "
                    "Returns the memory content, provenance (source documents with titles and URIs), "
                    "entity links, confidence, corroboration count, and all metadata fields."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "memory_id": {
                            "type": "string",
                            "description": "The memory ID (e.g. 'mem-a3f7b2c1')",
                        },
                    },
                    "required": ["memory_id"],
                },
            ),
            Tool(
                name="list_recent_changes",
                description=(
                    "List what has changed recently: new/updated source documents and optionally "
                    "the memories extracted from them. Useful for staying current on team knowledge."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "since": {
                            "type": "string",
                            "description": "ISO date. Defaults to 7 days ago.",
                        },
                        "source": {
                            "type": "string",
                            "description": "Filter to a specific source ID",
                        },
                        "include_memories": {
                            "type": "boolean",
                            "default": True,
                            "description": "Include new/updated memories in the response (default true)",
                        },
                    },
                },
            ),
            Tool(
                name="submit_agent_session_document",
                description=(
                    "Submit a client-generated agent session, task, or compaction-window summary as a "
                    "low-authority generated source document. The document is stored with receipt lineage "
                    "and can be routed through the normal source pipeline."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "client": {"type": "string", "description": "Client name, such as codex or claude-code"},
                        "session_id": {"type": "string", "description": "Client session identifier"},
                        "trigger": {"type": "string", "description": "Stop, TaskComplete, PreCompact, or similar"},
                        "workspace": {"type": "string", "description": "Workspace or repository path"},
                        "document_markdown": {
                            "type": "string",
                            "description": "Structured markdown summary generated from the client session history",
                        },
                        "repo": {"type": "string"},
                        "branch": {"type": "string"},
                        "commit_sha": {"type": "string"},
                        "history_window_kind": {"type": "string", "default": "session"},
                        "history_window_start": {"type": "string"},
                        "history_window_end": {"type": "string"},
                        "title": {"type": "string"},
                        "metadata": {"type": "object"},
                        "submitted_at": {"type": "string"},
                        "process_now": {
                            "type": "boolean",
                            "default": True,
                            "description": "Run the agent_session source through sync before returning",
                        },
                    },
                    "required": ["client", "session_id", "trigger", "workspace", "document_markdown"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        if name == "search":
            return await _handle_search(arguments)
        elif name == "get_memory":
            return await _handle_get_memory(arguments)
        elif name == "list_recent_changes":
            return await _handle_recent_changes(arguments)
        elif name == "submit_agent_session_document":
            return await _handle_submit_agent_session_document(arguments)
        else:
            return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]

    # ------------------------------------------------------------------
    # Lazy-init SearchEngine (created on first search call)
    # ------------------------------------------------------------------

    _search_engine = None

    async def _get_search_engine():
        nonlocal _search_engine
        if _search_engine is None:
            from memforge.runtime import get_effective_llm_config
            from memforge.llm.structured import LiteLlmStructuredClient, StructuredLlmConfig
            from memforge.retrieval.embeddings import get_chroma_collection
            from memforge.retrieval.search import SearchEngine

            memory_collection = get_chroma_collection(
                chroma_path=config.storage.chroma_path, name="memories",
            )
            llm = await get_effective_llm_config(db, config)
            embed_cfg = {
                "base_url": llm.embedding_base_url,
                "api_key": llm.embedding_api_key,
                "model": llm.embedding_model,
            }
            structured_llm_client = None
            if llm.enrichment_api_key:
                structured_llm_client = LiteLlmStructuredClient(
                    StructuredLlmConfig(
                        model=llm.enrichment_model,
                        base_url=llm.enrichment_base_url or None,
                        api_key=llm.enrichment_api_key,
                        timeout_s=llm.request_timeout_s,
                    )
                )
            _search_engine = SearchEngine(
                db=db,
                memory_collection=memory_collection,
                embed_cfg=embed_cfg,
                config=config.retrieval,
                structured_llm_client=structured_llm_client,
            )
        return _search_engine

    # ------------------------------------------------------------------
    # Tool handlers
    # ------------------------------------------------------------------

    async def _handle_search(args: dict) -> list[TextContent]:
        """Hybrid search via SearchEngine (vector + BM25 + entity-graph + temporal + RRF)."""
        query = args["query"]

        try:
            engine = await _get_search_engine()
            result = await engine.search(
                query=query,
                memory_types=args.get("memory_types"),
                sources=args.get("sources"),
                time_range=args.get("time_range"),
                entities=args.get("entities"),
                include_superseded=args.get("include_superseded", False),
                top_k=args.get("top_k", config.retrieval.default_top_k),
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

        except Exception as e:
            logger.warning("Search failed: %s", e, exc_info=True)
            return [TextContent(type="text", text=json.dumps({
                "error": f"Search unavailable: {e}",
                "hint": "Ensure embedding API keys are configured and ChromaDB is initialised.",
            }))]

    async def _handle_get_memory(args: dict) -> list[TextContent]:
        """Fetch full memory detail with provenance, entity links, and all fields."""
        memory_id = args["memory_id"]

        memory = await db.get_memory(memory_id)
        if not memory:
            return [TextContent(type="text", text=json.dumps({"error": f"Memory not found: {memory_id}"}))]

        # Get provenance (source documents)
        mem_sources = await db.get_memory_sources(memory_id)
        provenance = []
        for ms in mem_sources:
            entry: dict = {
                "doc_id": ms.doc_id,
                "source_type": ms.source_type,
                "excerpt": ms.excerpt,
                "added_at": ms.added_at.isoformat() if ms.added_at else None,
            }
            doc = await db.get_document(ms.doc_id)
            if doc:
                entry["title"] = doc.title
                entry["source_url"] = doc.source_url
                entry["file_uri"] = document_content_uri(doc)
                entry["pdf_uri"] = doc.pdf_content_uri
                entry["content_url"] = document_content_url(doc)
                entry["pdf_url"] = document_pdf_url(doc)
            provenance.append(entry)

        # Get linked entities
        entity_ids = await db.get_memory_entity_ids(memory_id)
        entities = []
        for eid in entity_ids:
            # Look up entity by scanning all entities (no get_entity_by_id in current DB)
            all_entities = await db.get_all_entities()
            for ent in all_entities:
                if ent.id == eid:
                    entities.append({
                        "id": ent.id,
                        "canonical_name": ent.canonical_name,
                        "tags": ent.tags,
                        "display_name": ent.display_name,
                    })
                    break

        result = {
            "id": memory.id,
            "memory_type": memory.memory_type,
            "content": memory.content,
            "content_hash": memory.content_hash,
            "scope": memory.scope,
            "project_key": memory.project_key,
            "tags": memory.tags,
            "entity_refs": memory.entity_refs,
            "confidence": memory.confidence,
            "corroboration_count": memory.corroboration_count,
            "contradiction_count": memory.contradiction_count,
            "valid_from": memory.valid_from.isoformat() if memory.valid_from else None,
            "valid_until": memory.valid_until.isoformat() if memory.valid_until else None,
            "status": memory.status,
            "superseded_by": memory.superseded_by,
            "extraction_context": memory.extraction_context,
            "created_at": memory.created_at.isoformat() if memory.created_at else None,
            "updated_at": memory.updated_at.isoformat() if memory.updated_at else None,
            "provenance": provenance,
            "entities": entities,
        }

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def _handle_submit_agent_session_document(args: dict) -> list[TextContent]:
        """Store a generated agent session document and optionally sync it."""
        try:
            from memforge.agent_sessions import submit_agent_session_document

            result = await submit_agent_session_document(
                db=db,
                config=config,
                client=args["client"],
                session_id=args["session_id"],
                trigger=args["trigger"],
                document_markdown=args["document_markdown"],
                workspace=args["workspace"],
                repo=args.get("repo"),
                branch=args.get("branch"),
                commit_sha=args.get("commit_sha"),
                history_window_kind=args.get("history_window_kind", "session"),
                history_window_start=args.get("history_window_start"),
                history_window_end=args.get("history_window_end"),
                title=args.get("title"),
                metadata=args.get("metadata") or {},
                submitted_at=args.get("submitted_at"),
            )

            sync_result = None
            if args.get("process_now", True):
                try:
                    from memforge.runtime import build_sync_runtime, run_source_sync

                    source = await db.get_source(result["source_id"])
                    if source:
                        runtime = await build_sync_runtime(db, config)
                        state = await run_source_sync(db, config, source, runtime=runtime)
                        sync_result = {
                            "status": state.last_sync_status,
                            "docs_processed": state.docs_processed,
                            "docs_updated": state.docs_updated,
                            "memories_extracted": state.memories_extracted,
                            "error_message": state.error_message,
                        }
                except Exception as e:
                    sync_result = {"status": "failed", "error_message": str(e)}

            return [TextContent(type="text", text=json.dumps({
                **result,
                "sync": sync_result,
            }, indent=2, default=str))]
        except ValueError as e:
            return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

    async def _handle_recent_changes(args: dict) -> list[TextContent]:
        """Return changelog entries and optionally new/updated memories."""
        since_str = args.get("since")
        source_filter = args.get("source")
        include_memories = args.get("include_memories", True)

        if since_str:
            since_dt = datetime.fromisoformat(since_str)
            if since_dt.tzinfo is None:
                since_dt = since_dt.replace(tzinfo=timezone.utc)
        else:
            since_dt = datetime.now(timezone.utc) - timedelta(days=7)

        since_iso = since_dt.isoformat()

        # Fetch changelog entries from the database
        changelog: list[dict] = []
        try:
            query = "SELECT * FROM changelog WHERE detected_at >= ?"
            params: list = [since_iso]
            if source_filter:
                query += " AND source = ?"
                params.append(source_filter)
            query += " ORDER BY detected_at DESC LIMIT 100"

            async with db.db.execute(query, params) as cursor:
                async for row in cursor:
                    d = dict(row)
                    entry = {
                        "id": d["id"],
                        "doc_id": d["doc_id"],
                        "change_type": d["change_type"],
                        "title": d.get("title"),
                        "source": d.get("source"),
                        "previous_version": d.get("previous_version"),
                        "current_version": d.get("current_version"),
                        "ai_change_summary": d.get("ai_change_summary"),
                        "detected_at": d["detected_at"],
                    }
                    changelog.append(entry)
        except Exception as e:
            logger.warning("Failed to query changelog: %s", e)

        # Optionally include new/updated memories in the time range
        recent_memories: list[dict] = []
        if include_memories:
            try:
                mem_query = "SELECT * FROM memories WHERE updated_at >= ?"
                mem_params: list = [since_iso]
                if source_filter:
                    mem_query = (
                        "SELECT DISTINCT m.* FROM memories m "
                        "JOIN memory_sources ms ON m.id = ms.memory_id "
                        "JOIN documents d ON ms.doc_id = d.doc_id "
                        "WHERE m.updated_at >= ? AND d.source = ?"
                    )
                    mem_params = [since_iso, source_filter]
                mem_query += " ORDER BY m.updated_at DESC LIMIT 50" if source_filter else " ORDER BY updated_at DESC LIMIT 50"

                async with db.db.execute(mem_query, mem_params) as cursor:
                    async for row in cursor:
                        d = dict(row)
                        recent_memories.append({
                            "id": d["id"],
                            "memory_type": d["memory_type"],
                            "content": d["content"],
                            "confidence": d["confidence"],
                            "status": d["status"],
                            "corroboration_count": d["corroboration_count"],
                            "updated_at": d.get("updated_at"),
                            "created_at": d.get("created_at"),
                        })
            except Exception as e:
                logger.warning("Failed to query recent memories: %s", e)

        result = {
            "since": since_iso,
            "changelog_entries": changelog,
            "total_changes": len(changelog),
        }
        if include_memories:
            result["recent_memories"] = recent_memories
            result["total_memories"] = len(recent_memories)

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    # ------------------------------------------------------------------
    # Resources
    # ------------------------------------------------------------------

    @server.list_resources()
    async def list_resources() -> list[Resource]:
        return [
            Resource(
                uri="memforge://stats",
                name="Memory layer statistics",
                description=(
                    "Current memory layer health: memory counts by type and status, "
                    "source counts, last sync times, entity statistics."
                ),
            ),
        ]

    @server.read_resource()
    async def read_resource(uri: str) -> str:
        if str(uri) == "memforge://stats":
            # Memory counts by type
            by_type = {}
            for mt in MemoryType:
                by_type[mt.value] = await db.count_memories(type=mt.value)

            # Memory counts by status
            by_status = {}
            for status in ("active", "superseded", "retired", "pending_review"):
                by_status[status] = await db.count_memories(status=status)

            total_memories = sum(by_type.values())

            # Source info
            sources_list = await db.list_sources()
            by_source = {}
            for src in sources_list:
                by_source[src["id"]] = {
                    "name": src["name"],
                    "type": src["type"],
                    "doc_count": src.get("doc_count", 0),
                    "status": src.get("status", "unknown"),
                    "last_sync": src.get("last_sync"),
                }

            # Sync states
            sync_states = []
            for src in sources_list:
                state = await db.get_sync_state(src["id"])
                if state:
                    sync_states.append({
                        "source": state.source,
                        "last_sync_at": state.last_sync_at.isoformat() if state.last_sync_at else None,
                        "status": state.last_sync_status,
                        "docs_processed": state.docs_processed,
                        "docs_updated": state.docs_updated,
                    })

            # Entity counts — aggregate by tag (multi-valued)
            all_entities = await db.get_all_entities()
            entity_tags: dict[str, int] = {}
            for ent in all_entities:
                for tag in ent.tags:
                    entity_tags[tag] = entity_tags.get(tag, 0) + 1

            stats = {
                "total_memories": total_memories,
                "memories_by_type": by_type,
                "memories_by_status": by_status,
                "total_sources": len(sources_list),
                "sources": by_source,
                "sync_states": sync_states,
                "total_entities": len(all_entities),
                "entities_by_tag": entity_tags,
            }

            # Add entity resolution stats if available
            if hasattr(db, "_entity_resolver_stats"):
                stats["entity_resolution"] = db._entity_resolver_stats

            return json.dumps(stats, indent=2)

        return json.dumps({"error": f"Unknown resource: {uri}"})

    return server


# ---------------------------------------------------------------------------
# Transport runners
# ---------------------------------------------------------------------------


async def run_mcp_stdio(db: Database, config: AppConfig) -> None:
    """Run the MCP server over stdio transport."""
    server = create_mcp_server(db, config)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())
