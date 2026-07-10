#!/usr/bin/env python3
"""Stdlib-only MCP proxy used by MemForge agent-client plugins."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import subprocess
import tempfile
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlencode, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

try:
    from .plugin_config import configured_api_token, configured_target
except ImportError:  # pragma: no cover - copied plugin package or direct file load
    try:
        from memforge_plugin_config import configured_api_token, configured_target
    except ImportError:
        import importlib.util

        _config_path = Path(__file__).with_name("memforge_plugin_config.py")
        if not _config_path.exists():
            _config_path = Path(__file__).with_name("plugin_config.py")
        _config_spec = importlib.util.spec_from_file_location("memforge_plugin_config", _config_path)
        if _config_spec is None or _config_spec.loader is None:
            raise
        _config_module = importlib.util.module_from_spec(_config_spec)
        _config_spec.loader.exec_module(_config_module)
        configured_api_token = _config_module.configured_api_token
        configured_target = _config_module.configured_target

DEFAULT_TIMEOUT_SECONDS = 60.0
SERVER_NAME = "memforge"
SERVER_VERSION = "0.1.26"
AGENT_CLIENT_VALUES = ["claude-code", "codex"]
ROOTS_LIST_REQUEST_ID = "memforge-roots-list-1"
WORKSPACE_ROOT_ENV_VARS = ("CODEX_WORKSPACE_ROOT",)
CURRENT_REPO_ONLY_DISABLED_ERROR = (
    "current_repo_only is disabled for MCP search because repo-scoped search depends on reliable "
    "workspace roots. Omit the filter to search all visible memories."
)
SEARCH_ALLOWED_KEYS = frozenset(
    {
        "query",
        "source_filter",
        "time_range",
        "top_k",
        "offset",
        "entities",
    }
)
SOURCE_FILTER_ALLOWED_KEYS = frozenset(
    {
        "source_ids",
        "clients",
        "current_repo_only",
    }
)
TIME_RANGE_ALLOWED_KEYS = frozenset({"date_type", "start_date", "end_date"})
DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SEARCH_TOP_K_MIN = 1
SEARCH_TOP_K_MAX = 50
_CLIENT_SUPPORTS_ROOTS = False
_PENDING_ROOTS_REQUEST_ID: str | None = None
_CLIENT_ROOT_PATHS: list[str] = []
_CLIENT_ROOTS_ERROR: str | None = None


TOOLS: list[dict[str, Any]] = [
    {
        "name": "search",
        "description": (
            "Search memories visible to the current principal. If the user names a configured "
            "source such as a Jira/Confluence source, call list_sources first and pass exact "
            "source_ids here; never guess source ids from the name. Query may be omitted only "
            "for deterministic listing with source_filter or time_range. Convert phrases like "
            "'last week' into explicit YYYY-MM-DD start_date/end_date before calling. For "
            "complete-list, enumeration, or inventory requests, use list_sources, then queryless deterministic "
            "listing; if total_candidates is greater than returned results, the answer is not a "
            "complete list until you request the next page with offset. Set each next offset to "
            "the previous offset plus the number of results returned. Stop when results is empty "
            "or the next offset is greater than or equal to total_candidates. For complete-list "
            "inventory tasks, choose a larger top_k page size up to 50 instead of assuming the default 10 is "
            "enough. Queried/ranked search uses candidate_count_kind=windowed; in that mode "
            "has_more means more results inside the current ranking window, not an exhaustive "
            "corpus count. A result may include follow_up with suggested_tool and reason. Search results "
            "do not include source links or artifact URLs; call get_memory for provenance, source "
            "titles, exact links, quotes, and lifecycle details before relying on source evidence."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Natural language search query. Optional when source_filter or time_range "
                        "is provided and the user wants a deterministic list rather than semantic ranking."
                    ),
                },
                "source_filter": {
                    "type": "object",
                    "description": (
                        "Optional provenance facets. Omit this object when unsure; MemForge "
                        "searches all visible memories when no facet is provided. Do not "
                        "invent source ids, repo ids, or fuzzy source names."
                    ),
                    "properties": {
                        "source_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Exact source IDs returned by list_sources. Use source_ids when "
                                "the user names a configured source; do not guess IDs."
                            ),
                        },
                        "clients": {
                            "type": "array",
                            "items": {"type": "string", "enum": AGENT_CLIENT_VALUES},
                            "description": (
                                "Restrict agent-session memories by producer. Use only when "
                                "the user explicitly names Codex or Claude Code."
                            ),
                        },
                    },
                    "additionalProperties": False,
                },
                "time_range": {
                    "type": "object",
                    "description": (
                        "Optional explicit date-only filter. Omit time_range when the "
                        "user did not ask for a date window. start_date and end_date "
                        "are individually optional; provide at least one if this object "
                        "is sent. Convert phrases like 'last week' into YYYY-MM-DD "
                        "dates before calling. date_type defaults to source_updated_at."
                    ),
                    "properties": {
                        "date_type": {
                            "type": "string",
                            "enum": ["source_updated_at", "memory_updated_at"],
                            "description": (
                                "source_updated_at filters by source/provenance update date; "
                                "memory_updated_at filters by MemForge memory lifecycle update date."
                            ),
                        },
                        "start_date": {
                            "type": "string",
                            "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                            "description": "Optional inclusive start date in YYYY-MM-DD format.",
                        },
                        "end_date": {
                            "type": "string",
                            "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                            "description": "Optional inclusive end date in YYYY-MM-DD format.",
                        },
                    },
                    "anyOf": [{"required": ["start_date"]}, {"required": ["end_date"]}],
                    "additionalProperties": False,
                },
                "top_k": {
                    "type": "integer",
                    "default": 10,
                    "minimum": SEARCH_TOP_K_MIN,
                    "maximum": SEARCH_TOP_K_MAX,
                    "description": (
                        "Page size for search results. Default is 10 for ordinary semantic search, "
                        "not a hard cap; use a larger value up to 50 for complete list, enumeration, or inventory "
                        "tasks, then continue with offset when total_candidates is still larger."
                    ),
                },
                "offset": {
                    "type": "integer",
                    "default": 0,
                    "minimum": 0,
                    "description": (
                        "Zero-based result offset for the next page. Use when total_candidates "
                        "is greater than the number of returned results and the user asked for a "
                        "complete list or inventory."
                    ),
                },
                "entities": {
                    "type": "array",
                    "maxItems": 8,
                    "items": {"type": "string", "minLength": 1, "maxLength": 128},
                    "description": (
                        "Optional agent-selected entity hints for graph-linking recall. When "
                        "the user's query clearly names a domain concept, pass a small list "
                        "of exact phrases while keeping query unchanged. Omit when unsure; "
                        "do not pass every noun, source IDs, broad action words, or hidden "
                        "guesses. Entity hints are not filters or authority."
                    ),
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "list_sources",
        "description": (
            "List search-eligible memory sources in the current workspace for the current "
            "principal. Use this before search when the user names a source, system, project, "
            "or configured connector such as a Jira or Confluence source. Returns safe metadata "
            "only: source_id, name, type, status, counts, and last_synced_at."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "get_memory",
        "description": (
            "Fetch full memory detail by ID when complete provenance, supporting sources, "
            "entity links, lifecycle metadata, or a search result follow_up hint indicates "
            "that summary-only context may be insufficient."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "The memory ID"},
            },
            "required": ["memory_id"],
        },
    },
    {
        "name": "get_resource",
        "description": (
            "Fetch a MemForge source artifact from get_memory.sources[].content_url or "
            "get_memory.sources[].pdf_url. In file mode this local proxy writes the "
            "artifact to the agent host cache and returns a real local_path. Use "
            "search -> get_memory -> get_resource "
            "when exact source text, quotes, or document evidence is needed."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": (
                        "A MemForge artifact URL such as /api/documents/{doc_id}/content, "
                        "/api/documents/{doc_id}/pdf, or /api/documents/{doc_id}/artifacts/{kind}."
                    ),
                },
                "mode": {
                    "type": "string",
                    "enum": ["text", "file", "base64"],
                    "default": "text",
                },
                "max_chars": {"type": "integer", "default": 120000},
                "max_bytes": {"type": "integer", "default": 2000000},
            },
            "required": ["url"],
        },
    },
    {
        "name": "create_memory",
        "description": (
            "Create a new memory when the user asks to remember or record durable knowledge. "
            "Users need not name this tool. First search for similar memories to avoid duplicates, "
            "show a readable preview with the new durable claim, provenance/evidence, scope, and type/tags, then get "
            "explicit confirmation via request_user_input if available, else a concise text question. "
            "Generate durable memory content from the confirmed preview without unapproved semantic changes. "
            "Keep provenance, confirmation details, test/deploy notes, and why-the-tool-was-called "
            "out of content; put source details in provenance. Never create memory silently."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": (
                        "Canonical durable memory content generated from the user-confirmed readable "
                        "preview. Preserve its meaning without unapproved semantic changes. Do not "
                        "put confirmation details, provenance, test/deploy notes, or why-the-tool-was-called "
                        "into content; those belong in provenance or stay out of the memory."
                    ),
                },
                "provenance": {
                    "type": "string",
                    "description": (
                        "Required evidence or source context for the provenance card. Use this "
                        "for details that explain where the memory came from but should not be "
                        "used as RAG memory content."
                    ),
                },
                "memory_type": {
                    "type": "string",
                    "enum": ["fact", "decision", "convention", "procedure"],
                    "default": "fact",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Short topical tags from the confirmed preview.",
                },
                "confidence": {"type": "number"},
                "idempotency_key": {
                    "type": "string",
                    "description": "Optional stable key for retrying the same user-confirmed create action.",
                },
            },
            "required": ["content", "provenance"],
            "additionalProperties": False,
        },
    },
    {
        "name": "retire_memory",
        "description": (
            "Retire a memory when conversation context shows it is wrong, obsolete, "
            "or should no longer be used. Users need not name this tool. First fetch "
            "the memory for hash/provenance, show a readable retire preview and reason, "
            "then get explicit confirmation via request_user_input if available, else "
            "a concise text question. Never retire silently or use this for arbitrary "
            "status changes."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "The memory ID to retire."},
                "reason": {
                    "type": "string",
                    "description": "User-facing reason for retiring this memory.",
                },
                "expected_content_hash": {
                    "type": "string",
                    "description": "Content hash from get_memory used as a stale guard.",
                },
            },
            "required": ["memory_id", "reason", "expected_content_hash"],
            "additionalProperties": False,
        },
    },
    {
        "name": "replace_memory",
        "description": (
            "Replace a memory when conversation context shows a claim should be corrected, "
            "narrowed, broadened, or superseded. Users need not name this tool. First fetch "
            "the memory for hash/provenance, show a readable preview with old claim, new "
            "claim, provenance/evidence, scope, and replacement reason, then get explicit "
            "confirmation via request_user_input if available, else a concise text question. "
            "Generate replacement_content from the confirmed preview without unapproved semantic "
            "changes. Keep provenance, confirmation details, test/deploy notes, and "
            "why-the-tool-was-called out of replacement_content; put source details in "
            "provenance. Never replace silently."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "The memory ID to replace."},
                "replacement_content": {
                    "type": "string",
                    "description": (
                        "Canonical memory text generated from the user-confirmed readable preview; "
                        "preserve its meaning without unapproved semantic changes. Do not put "
                        "confirmation details, provenance, test/deploy notes, or why-the-tool-was-called "
                        "into replacement_content; those belong in provenance or stay out of the memory."
                    ),
                },
                "provenance": {
                    "type": "string",
                    "description": (
                        "Required evidence or source context for the correction provenance card. "
                        "Use this for details that explain where the replacement came from but "
                        "should not be used as RAG memory content."
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": "User-facing reason for replacing this memory.",
                },
                "expected_content_hash": {
                    "type": "string",
                    "description": "Content hash from get_memory used as a stale guard.",
                },
                "replacement_kind": {
                    "type": "string",
                    "enum": ["revision", "supersession"],
                    "default": "supersession",
                    "description": (
                        "Use revision only when the user explicitly says this is the corrected "
                        "current version of the same knowledge. Otherwise use supersession."
                    ),
                },
            },
            "required": ["memory_id", "replacement_content", "provenance", "reason", "expected_content_hash"],
            "additionalProperties": False,
        },
    },
    {
        "name": "list_memory_reviews",
        "description": (
            "List pending memory-review decisions for interactive conflict or correction review."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["open", "pending", "stale", "approved", "rejected"],
                    "default": "open",
                },
                "limit": {"type": "integer", "default": 20},
                "offset": {"type": "integer", "default": 0},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "get_memory_review",
        "description": "Fetch full current/proposed memory details for a memory-review decision.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "review_id": {"type": "string", "description": "The memory review ID."},
            },
            "required": ["review_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "resolve_memory_review",
        "description": (
            "Resolve one memory review after explicit user confirmation. approve promotes "
            "the challenger, reject retires it, and refresh repins stale expectations. "
            "Show a readable current/proposed diff and confirm via request_user_input if "
            "available, else a concise text question. Never resolve silently."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "review_id": {"type": "string", "description": "The memory review ID."},
                "decision": {"type": "string", "enum": ["approve", "reject", "refresh"]},
                "note": {
                    "type": "string",
                    "description": "Reviewer note; required when decision is reject.",
                },
                "reviewer": {"type": "string", "description": "Optional reviewer display name or user id."},
            },
            "required": ["review_id", "decision"],
            "additionalProperties": False,
        },
    },
]


class NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


class ResourceTarget:
    def __init__(self, doc_id: str, kind: str, relative_url: str, request_url: str) -> None:
        self.doc_id = doc_id
        self.kind = kind
        self.relative_url = relative_url
        self.request_url = request_url


def main() -> int:
    while True:
        envelope = _read_message()
        if envelope is None:
            return 0
        message, transport = envelope
        response = _handle_rpc_message(message)
        if response is not None:
            _write_message(response, transport)


def _handle_rpc_message(message: dict[str, Any]) -> dict[str, Any] | None:
    method = message.get("method")
    request_id = message.get("id")
    try:
        if method is None:
            _handle_rpc_response(message)
            return None
        if method == "initialize":
            _record_client_capabilities(message)
            result = {
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            }
            return _rpc_result(request_id, result)
        if method == "notifications/initialized":
            if _CLIENT_SUPPORTS_ROOTS and _PENDING_ROOTS_REQUEST_ID is None and not _CLIENT_ROOT_PATHS:
                return _request_client_roots()
            return None
        if method == "notifications/roots/list_changed":
            _CLIENT_ROOT_PATHS.clear()
            if _CLIENT_SUPPORTS_ROOTS:
                return _request_client_roots()
            return None
        if method == "ping":
            return _rpc_result(request_id, {})
        if method == "tools/list":
            return _rpc_result(request_id, {"tools": TOOLS})
        if method == "tools/call":
            params = message.get("params") if isinstance(message.get("params"), dict) else {}
            name = str(params.get("name") or "")
            arguments = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
            payload = _call_tool(name, arguments)
            return _rpc_result(request_id, _tool_result(payload))
        if request_id is None:
            return None
        return _rpc_error(request_id, -32601, f"Method not found: {method}")
    except Exception as exc:  # pragma: no cover - defensive MCP boundary
        if request_id is None:
            return None
        return _rpc_error(request_id, -32603, f"Internal error: {exc}")


def _call_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    if name == "search":
        try:
            body = _search_args_with_context(args)
        except ValueError as exc:
            return {"error": str(exc)}
        return _compact_search_response(_http_json("POST", "/memories/search", body))
    if name == "list_sources":
        if args:
            return {"error": "list_sources does not accept parameters"}
        return _http_json("GET", "/sources/searchable", None)
    if name == "get_memory":
        memory_id = str(args.get("memory_id") or "").strip()
        if not memory_id:
            return {"error": "memory_id is required"}
        return _compact_memory_response(
            _http_json("GET", f"/memories/{quote(memory_id, safe='')}?include_private=true", None)
        )
    if name == "get_resource":
        return _handle_get_resource(args)
    if name == "create_memory":
        try:
            memory_type = str(args.get("memory_type") or "fact").strip()
            if memory_type not in {"fact", "decision", "convention", "procedure"}:
                raise ValueError("memory_type must be fact, decision, convention, or procedure")
            tags = args.get("tags") or []
            if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
                raise ValueError("tags must be an array of strings")
            body = {
                "content": _required_string_arg(args, "content"),
                "provenance": _required_string_arg(args, "provenance"),
                "memory_type": memory_type,
                "tags": tags,
                "client": _mcp_client(),
            }
            if "confidence" in args:
                confidence = args.get("confidence")
                if not isinstance(confidence, (int, float)):
                    raise ValueError("confidence must be a number")
                body["confidence"] = float(confidence)
            repo_identifier = _active_repo_identifier()
            if not repo_identifier:
                raise ValueError(_create_memory_repo_error())
            body["repo_identifier"] = repo_identifier
            idempotency_key = str(args.get("idempotency_key") or "").strip()
            if idempotency_key:
                body["idempotency_key"] = idempotency_key
        except ValueError as exc:
            return {"error": str(exc)}
        return _http_json("POST", "/memories/create", body)
    if name == "retire_memory":
        try:
            memory_id = _required_string_arg(args, "memory_id")
            body = {
                "reason": _required_string_arg(args, "reason"),
                "expected_content_hash": _required_string_arg(args, "expected_content_hash"),
            }
        except ValueError as exc:
            return {"error": str(exc)}
        return _http_json("POST", f"/memories/{quote(memory_id, safe='')}/retire", body)
    if name == "replace_memory":
        try:
            memory_id = _required_string_arg(args, "memory_id")
            replacement_kind = str(args.get("replacement_kind") or "supersession").strip()
            if replacement_kind not in {"revision", "supersession"}:
                raise ValueError("replacement_kind must be revision or supersession")
            body = {
                "replacement_content": _required_string_arg(args, "replacement_content"),
                "provenance": _required_string_arg(args, "provenance"),
                "reason": _required_string_arg(args, "reason"),
                "expected_content_hash": _required_string_arg(args, "expected_content_hash"),
                "replacement_kind": replacement_kind,
            }
        except ValueError as exc:
            return {"error": str(exc)}
        return _http_json("POST", f"/memories/{quote(memory_id, safe='')}/replace", body)
    if name == "list_memory_reviews":
        allowed = {"status", "limit", "offset"}
        unknown = sorted(set(args) - allowed)
        if unknown:
            return {"error": "Unsupported list_memory_reviews parameter(s): " + ", ".join(unknown)}
        try:
            query = {
                "status": str(args.get("status") or "open"),
                "limit": _optional_int_arg(args, "limit", 20),
                "offset": _optional_int_arg(args, "offset", 0),
            }
        except ValueError as exc:
            return {"error": str(exc)}
        return _http_json("GET", "/memory-reviews?" + urlencode(query), None)
    if name == "get_memory_review":
        try:
            review_id = _required_string_arg(args, "review_id")
        except ValueError as exc:
            return {"error": str(exc)}
        return _http_json("GET", f"/memory-reviews/{quote(review_id, safe='')}", None)
    if name == "resolve_memory_review":
        try:
            review_id = _required_string_arg(args, "review_id")
            decision = _required_string_arg(args, "decision")
            if decision not in {"approve", "reject", "refresh"}:
                raise ValueError("decision must be approve, reject, or refresh")
            note = str(args.get("note") or "").strip()
            if decision == "reject" and not note:
                raise ValueError("note is required when decision is reject")
            reviewer = str(args.get("reviewer") or "").strip()
            body = {}
            if note:
                body["note"] = note
            if reviewer:
                body["reviewer"] = reviewer
        except ValueError as exc:
            return {"error": str(exc)}
        return _http_json("POST", f"/memory-reviews/{quote(review_id, safe='')}/{decision}", body)
    return {"error": f"Unknown tool: {name}"}


def _required_string_arg(args: dict[str, Any], name: str) -> str:
    value = str(args.get(name) or "").strip()
    if not value:
        raise ValueError(f"{name} is required")
    return value


def _optional_string_arg(args: dict[str, Any], name: str) -> str | None:
    value = args.get(name)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int_arg(args: dict[str, Any], name: str, default: int) -> int:
    value = args.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _required_non_negative_int_arg(args: dict[str, Any], name: str) -> int:
    value = args.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _required_bounded_int_arg(args: dict[str, Any], name: str, *, minimum: int, maximum: int) -> int:
    value = args.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or value > maximum:
        raise ValueError(f"{name} must be an integer from {minimum} to {maximum}")
    return value


def _search_args_with_context(args: dict[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(args) - SEARCH_ALLOWED_KEYS)
    if unknown:
        raise ValueError(
            "Unsupported search parameter(s): " + ", ".join(unknown) + ". Omit unknown filters instead of guessing."
        )
    body = dict(args)
    query = str(body.get("query") or "").strip()
    has_deterministic_filter = False
    if "top_k" in body:
        body["top_k"] = _required_bounded_int_arg(
            body,
            "top_k",
            minimum=SEARCH_TOP_K_MIN,
            maximum=SEARCH_TOP_K_MAX,
        )
    if "offset" in body:
        body["offset"] = _required_non_negative_int_arg(body, "offset")
    if "entities" in body:
        entities = body["entities"]
        normalized_entities = _validate_search_entities(entities)
        if not normalized_entities:
            body.pop("entities")
        else:
            body["entities"] = normalized_entities
    body["include_private"] = True
    body["include_superseded"] = False
    repo_identifier = _active_repo_identifier()
    if repo_identifier:
        body["active_repo_identifier"] = repo_identifier
    source_filter = body.get("source_filter")
    if isinstance(source_filter, dict):
        unknown_filter_keys = sorted(set(source_filter) - SOURCE_FILTER_ALLOWED_KEYS)
        if unknown_filter_keys:
            raise ValueError(
                "Unsupported source_filter parameter(s): "
                + ", ".join(unknown_filter_keys)
                + ". Omit repo-scoped facets until MCP roots are available."
            )
        source_ids = source_filter.get("source_ids")
        if source_ids is not None:
            if not isinstance(source_ids, list) or not source_ids or not all(
                isinstance(item, str) and item.strip() for item in source_ids
            ):
                raise ValueError("source_filter.source_ids must be a non-empty array of source IDs from list_sources")
        if "current_repo_only" in source_filter:
            raise ValueError(CURRENT_REPO_ONLY_DISABLED_ERROR)
        has_deterministic_filter = bool(source_filter)
        body["source_filter"] = source_filter
    time_range = body.get("time_range")
    if time_range is not None:
        body["time_range"] = _validate_time_range(time_range)
        has_deterministic_filter = True
    if not query and not has_deterministic_filter:
        raise ValueError("search.query may be omitted only when source_filter or time_range is provided")
    return body


def _validate_search_entities(entities: Any) -> list[str]:
    error = "entities must be an array of 1-8 strings, each 1-128 characters after trimming"
    if not isinstance(entities, list):
        raise ValueError(error)
    if len(entities) > 8:
        raise ValueError(error)

    normalized: list[str] = []
    seen: set[str] = set()
    for entity in entities:
        if not isinstance(entity, str):
            raise ValueError(error)
        value = entity.strip()
        if not value or len(value) > 128:
            raise ValueError(error)
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(value)
    return normalized


def _validate_time_range(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("time_range must be an object with YYYY-MM-DD date bounds")
    unknown = sorted(set(value) - TIME_RANGE_ALLOWED_KEYS)
    if unknown:
        raise ValueError("Unsupported time_range parameter(s): " + ", ".join(unknown))
    date_type = value.get("date_type", "source_updated_at")
    if date_type not in {"source_updated_at", "memory_updated_at"}:
        raise ValueError("time_range.date_type must be source_updated_at or memory_updated_at")
    start_date = value.get("start_date")
    end_date = value.get("end_date")
    if not start_date and not end_date:
        raise ValueError("time_range requires start_date or end_date; omit time_range for no date filter")
    normalized: dict[str, str] = {"date_type": date_type}
    for key, item in (("start_date", start_date), ("end_date", end_date)):
        if item is None:
            continue
        if not isinstance(item, str) or not DATE_ONLY_RE.fullmatch(item):
            raise ValueError(f"time_range.{key} must be a YYYY-MM-DD date")
        normalized[key] = item
    if start_date and end_date and str(start_date) > str(end_date):
        raise ValueError("time_range.start_date must be on or before end_date")
    return normalized


def _active_repo_identifier() -> str | None:
    identifiers = _client_root_repo_identifiers()
    if len(identifiers) == 1:
        return next(iter(identifiers))
    if not _CLIENT_SUPPORTS_ROOTS:
        env_identifier = _env_workspace_repo_identifier()
        if env_identifier:
            return env_identifier
    return None


def _create_memory_repo_error() -> str:
    reason = _repo_roots_error_reason()
    return f"create_memory requires exactly one git remote from MCP workspace roots; {reason}. Refusing to create an unscoped memory."


def _repo_roots_error_reason() -> str:
    if not _CLIENT_SUPPORTS_ROOTS:
        if _env_workspace_root():
            return "the MCP client did not advertise roots support and CODEX_WORKSPACE_ROOT does not resolve to a git remote"
        return "the MCP client did not advertise roots support"
    elif _PENDING_ROOTS_REQUEST_ID is not None:
        return "the MCP client has not returned workspace roots yet"
    elif _CLIENT_ROOTS_ERROR:
        return f"the MCP client returned an error for roots/list: {_CLIENT_ROOTS_ERROR}"
    elif not _CLIENT_ROOT_PATHS:
        return "the MCP client did not provide workspace roots"
    identifiers = _client_root_repo_identifiers()
    if len(identifiers) > 1:
        return "workspace roots resolve to multiple git remotes"
    return "workspace roots do not resolve to a git remote"


def _client_root_repo_identifiers() -> set[str]:
    return {
        repo_identifier
        for root_path in _CLIENT_ROOT_PATHS
        if (repo_identifier := _repo_identifier_from_cwd(root_path))
    }


def _env_workspace_repo_identifier() -> str | None:
    if env_root := _env_workspace_root():
        return _repo_identifier_from_cwd(env_root)
    return None


def _env_workspace_root() -> str | None:
    for name in WORKSPACE_ROOT_ENV_VARS:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return None


def _repo_identifier_from_cwd(cwd: str | Path) -> str | None:
    remote = _git_value(["git", "remote", "get-url", "origin"], cwd=cwd)
    return _normalize_repo_identifier(remote)


def _mcp_client() -> str:
    value = os.getenv("MEMFORGE_MCP_CLIENT", "").strip()
    if value in AGENT_CLIENT_VALUES:
        return value
    return "codex"


def _normalize_repo_identifier(repo: str | None) -> str | None:
    if repo is None:
        return None
    value = repo.strip()
    if not value:
        return None

    ssh_match = re.match(r"^[^/@]+@([^:/]+):(.+)$", value)
    if ssh_match:
        host, path = ssh_match.groups()
        value = f"{host}/{path}"
    else:
        value = re.sub(r"^[a-z][a-z0-9+.-]*://", "", value, flags=re.IGNORECASE)
        value = re.sub(r"^[^@/]+@", "", value)

    value = value.split("?", 1)[0].split("#", 1)[0].rstrip("/")
    if value.endswith(".git"):
        value = value[:-4]
    value = re.sub(r"/+", "/", value)
    return value.lower() or None


def _git_value(command: list[str], *, cwd: str | Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd or os.getcwd()),
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


def _record_client_capabilities(message: dict[str, Any]) -> None:
    global _CLIENT_SUPPORTS_ROOTS, _PENDING_ROOTS_REQUEST_ID, _CLIENT_ROOT_PATHS, _CLIENT_ROOTS_ERROR
    params = message.get("params") if isinstance(message.get("params"), dict) else {}
    capabilities = params.get("capabilities") if isinstance(params.get("capabilities"), dict) else {}
    _CLIENT_SUPPORTS_ROOTS = isinstance(capabilities.get("roots"), dict)
    _PENDING_ROOTS_REQUEST_ID = None
    _CLIENT_ROOT_PATHS = []
    _CLIENT_ROOTS_ERROR = None


def _request_client_roots() -> dict[str, Any]:
    global _PENDING_ROOTS_REQUEST_ID, _CLIENT_ROOTS_ERROR
    _PENDING_ROOTS_REQUEST_ID = ROOTS_LIST_REQUEST_ID
    _CLIENT_ROOTS_ERROR = None
    return {"jsonrpc": "2.0", "id": ROOTS_LIST_REQUEST_ID, "method": "roots/list"}


def _handle_rpc_response(message: dict[str, Any]) -> None:
    global _PENDING_ROOTS_REQUEST_ID, _CLIENT_ROOT_PATHS, _CLIENT_ROOTS_ERROR
    if message.get("id") != _PENDING_ROOTS_REQUEST_ID:
        return
    _PENDING_ROOTS_REQUEST_ID = None
    error = message.get("error")
    if isinstance(error, dict):
        _CLIENT_ROOTS_ERROR = str(error.get("message") or error).strip()
        _CLIENT_ROOT_PATHS = []
        return
    _CLIENT_ROOTS_ERROR = None
    result = message.get("result") if isinstance(message.get("result"), dict) else {}
    roots = result.get("roots") if isinstance(result.get("roots"), list) else []
    _CLIENT_ROOT_PATHS = [path for item in roots if (path := _root_path(item))]


def _root_path(item: Any) -> str | None:
    if not isinstance(item, dict):
        return None
    uri = str(item.get("uri") or "").strip()
    if not uri:
        return None
    parsed = urlparse(uri)
    if parsed.scheme and parsed.scheme != "file":
        return None
    if parsed.scheme == "file":
        if parsed.netloc and parsed.netloc not in {"localhost", "127.0.0.1"}:
            path = f"//{parsed.netloc}{parsed.path}"
        else:
            path = parsed.path
        return unquote(path) or None
    return uri


def _tool_result(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": json.dumps(payload, indent=2)}],
        "isError": False,
    }


def _compact_search_response(payload: dict[str, Any]) -> dict[str, Any]:
    if "error" in payload:
        return payload

    compact: dict[str, Any] = {}
    results = payload.get("results")
    if isinstance(results, list):
        compact["results"] = [
            _compact_search_result(result)
            for result in results
            if isinstance(result, dict)
        ]

    for key in (
        "total_candidates",
        "candidate_count_kind",
        "ranking_window_size",
        "limit",
        "offset",
        "has_more",
    ):
        if key in payload:
            compact[key] = payload[key]

    if "has_more" not in compact and compact.get("candidate_count_kind") != "windowed":
        total = compact.get("total_candidates")
        limit = compact.get("limit")
        offset = compact.get("offset", 0)
        if isinstance(total, int) and isinstance(limit, int) and isinstance(offset, int):
            compact["has_more"] = offset + limit < total

    return compact


def _compact_search_result(result: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in (
        "memory_id",
        "memory_type",
        "summary",
        "confidence",
        "relevance_score",
        "tags",
        "freshness",
        "status",
        "follow_up",
    ):
        if key in result:
            compact[key] = result[key]
    return compact


def _compact_memory_response(payload: dict[str, Any]) -> dict[str, Any]:
    if "error" in payload:
        return payload

    compact: dict[str, Any] = {}
    for key in (
        "id",
        "memory_type",
        "content",
        "content_hash",
        "confidence",
        "tags",
        "status",
        "entity_refs",
    ):
        if key in payload:
            compact[key] = payload[key]

    sources = payload.get("sources")
    if isinstance(sources, list):
        compact["sources"] = [
            _compact_memory_source(source)
            for source in sources
            if isinstance(source, dict)
        ]
    return compact


def _compact_memory_source(source: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in (
        "doc_id",
        "source_type",
        "support_kind",
        "doc_title",
        "source_url",
        "content_url",
        "pdf_url",
        "source_updated_at",
    ):
        if key in source:
            compact[key] = source[key]
    return compact


def _resource_url(path: str) -> str:
    return configured_target().resource_url(path)


def _api_headers(*, json_body: bool = False) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if json_body:
        headers["Content-Type"] = "application/json"
    token = configured_api_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _http_json(method: str, path: str, body: dict[str, Any] | None) -> dict[str, Any]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = Request(
        _resource_url(path),
        data=data,
        headers=_api_headers(json_body=body is not None),
        method=method,
    )
    try:
        with build_opener(NoRedirectHandler).open(request, timeout=DEFAULT_TIMEOUT_SECONDS) as response:
            raw = response.read()
            if not raw:
                return {}
            return json.loads(raw.decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        return {"error": "MemForge API request failed", "status_code": exc.code, "detail": detail}
    except (OSError, URLError, json.JSONDecodeError) as exc:
        return {
            "error": "MemForge API unavailable",
            "api_url": configured_target().workspace_api_base,
            "detail": str(exc),
        }


def _handle_get_resource(args: dict[str, Any]) -> dict[str, Any]:
    mode = str(args.get("mode") or "text").strip().lower()
    if mode not in {"text", "file", "base64"}:
        return {"error": f"unsupported mode: {mode}", "supported_modes": ["text", "file", "base64"]}

    max_bytes = _positive_int_arg(args, "max_bytes", 2_000_000)
    if isinstance(max_bytes, dict):
        return max_bytes
    max_chars = _positive_int_arg(args, "max_chars", 120_000)
    if isinstance(max_chars, dict):
        return max_chars

    target = _parse_resource_url(str(args.get("url") or "").strip(), configured_target().origin)
    if target is None:
        return {
            "error": "unsupported resource URL",
            "hint": (
                "Use a relative MemForge /api/documents/{doc_id}/content, /pdf, "
                "or /artifacts/{kind} URL, or an absolute URL under MEMFORGE_API_URL."
            ),
        }

    try:
        if mode == "file":
            return _fetch_resource_file(target)
        return _fetch_resource_inline(target, mode=mode, max_bytes=max_bytes, max_chars=max_chars)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        return {
            "error": "resource fetch failed",
            "status_code": exc.code,
            "url": target.relative_url,
            "detail": detail,
        }
    except (OSError, URLError) as exc:
        return {"error": "resource fetch failed", "url": target.relative_url, "detail": str(exc)}


def _fetch_resource_inline(
    target: ResourceTarget,
    *,
    mode: str,
    max_bytes: int,
    max_chars: int,
) -> dict[str, Any]:
    request = Request(target.request_url, headers=_api_headers(), method="GET")
    with build_opener(NoRedirectHandler).open(request, timeout=DEFAULT_TIMEOUT_SECONDS) as response:
        data = response.read(max_bytes + 1)
        headers = _lower_headers(response.headers)
        content_type = headers.get("content-type", "application/octet-stream")
        metadata = _resource_metadata(target, headers, len(data), mode)
        if len(data) > max_bytes:
            return {
                **metadata,
                "error": "artifact exceeds max_bytes",
                "hint": "Use mode=file for large or binary artifacts.",
                "max_bytes": max_bytes,
            }
        if mode == "base64":
            return {**metadata, "data_base64": base64.b64encode(data).decode("ascii")}
        if not _is_text_content_type(content_type):
            return {
                **metadata,
                "error": "artifact is not text",
                "hint": "Use mode=file or mode=base64 for binary artifacts.",
            }
        text = data.decode("utf-8", errors="replace")
        return {**metadata, "text": text[:max_chars], "truncated": len(text) > max_chars}


def _fetch_resource_file(target: ResourceTarget) -> dict[str, Any]:
    request = Request(target.request_url, headers=_api_headers(), method="GET")
    tmp_path: Path | None = None
    try:
        with build_opener(NoRedirectHandler).open(request, timeout=DEFAULT_TIMEOUT_SECONDS) as response:
            headers = _lower_headers(response.headers)
            filename = _resource_filename(headers, target)
            digest = hashlib.sha256()
            observed_size = 0
            cache_root = _artifact_cache_root()
            safe_doc = _safe_cache_component(target.doc_id) or "document"
            safe_kind = _safe_cache_component(target.kind) or "artifact"
            with tempfile.NamedTemporaryFile(
                "wb",
                dir=cache_root,
                prefix=f".{safe_doc}-{safe_kind}-",
                suffix=".tmp",
                delete=False,
            ) as handle:
                tmp_path = Path(handle.name)
                while True:
                    chunk = response.read(1024 * 256)
                    if not chunk:
                        break
                    digest.update(chunk)
                    observed_size += len(chunk)
                    handle.write(chunk)
            final_path = _cache_artifact_path(target.doc_id, target.kind, filename, digest.hexdigest()[:16])
            if final_path.exists():
                tmp_path.unlink(missing_ok=True)
            else:
                tmp_path.chmod(0o600)
                tmp_path.replace(final_path)
            return {
                **_resource_metadata(target, headers, observed_size, "file"),
                "local_path": str(final_path),
                "cleanup": "temporary-cache",
            }
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def _resource_metadata(
    target: ResourceTarget,
    headers: dict[str, str],
    observed_size: int,
    mode: str,
) -> dict[str, Any]:
    return {
        "doc_id": target.doc_id,
        "kind": target.kind,
        "content_type": headers.get("content-type", "application/octet-stream"),
        "filename": _resource_filename(headers, target),
        "size_bytes": _response_size_bytes(headers, observed_size),
        "url": target.relative_url,
        "mode": mode,
    }


def _parse_resource_url(url: str, origin: str) -> ResourceTarget | None:
    parsed = urlparse(url)
    base = urlparse(origin)
    if parsed.query or parsed.fragment:
        return None

    if parsed.scheme or parsed.netloc:
        if parsed.scheme not in {"http", "https"}:
            return None
        if parsed.scheme != base.scheme or parsed.netloc != base.netloc:
            return None
        path = parsed.path
    else:
        path = url

    if not path.startswith("/"):
        path = f"/{path}"

    parts = [unquote(part) for part in path.strip("/").split("/") if part]
    if any(part in {".", ".."} or "/" in part or "\\" in part for part in parts):
        return None
    if len(parts) == 4 and parts[:2] == ["api", "documents"] and parts[3] == "content":
        return ResourceTarget(parts[2], "content", path, _resource_url(path[len("/api") :]))
    if len(parts) == 4 and parts[:2] == ["api", "documents"] and parts[3] == "pdf":
        return ResourceTarget(parts[2], "pdf", path, _resource_url(path[len("/api") :]))
    if len(parts) == 5 and parts[:2] == ["api", "documents"] and parts[3] == "artifacts":
        return ResourceTarget(parts[2], parts[4], path, _resource_url(path[len("/api") :]))
    return None


def _positive_int_arg(args: dict[str, Any], name: str, default: int) -> int | dict[str, Any]:
    raw_value = args.get(name, default)
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return {"error": f"invalid {name}", "detail": f"{name} must be a positive integer."}
    if value <= 0:
        return {"error": f"invalid {name}", "detail": f"{name} must be a positive integer."}
    return value


def _lower_headers(headers: Any) -> dict[str, str]:
    return {str(k).lower(): str(v) for k, v in headers.items()}


def _response_size_bytes(headers: dict[str, str], fallback: int) -> int:
    try:
        return int(headers.get("content-length") or fallback)
    except ValueError:
        return fallback


def _resource_filename(headers: dict[str, str], target: ResourceTarget) -> str:
    disposition = headers.get("content-disposition", "")
    match = re.search(r'filename="?([^";]+)"?', disposition)
    if match:
        return Path(match.group(1)).name
    suffix = ".pdf" if target.kind == "pdf" else ".md" if target.kind == "content" else ".bin"
    return f"{target.doc_id}-{target.kind}{suffix}"


def _is_text_content_type(media_type: str) -> bool:
    normalized = media_type.split(";", 1)[0].strip().lower()
    return normalized.startswith("text/") or normalized in {
        "application/json",
        "application/xml",
        "application/xhtml+xml",
    }


def _artifact_cache_root() -> Path:
    cache_root = Path(
        os.getenv("MEMFORGE_ARTIFACT_CACHE_DIR") or (Path.home() / ".memforge-agent" / "artifacts")
    ).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root


def _safe_cache_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def _cache_artifact_path(doc_id: str, kind: str, filename: str, digest: str) -> Path:
    safe_doc = _safe_cache_component(doc_id) or "document"
    safe_kind = _safe_cache_component(kind) or "artifact"
    suffix = Path(filename).suffix or ".bin"
    return _artifact_cache_root() / f"{safe_doc}-{safe_kind}-{digest}{suffix}"


def _read_message() -> tuple[dict[str, Any], str] | None:
    while True:
        line = sys.stdin.buffer.readline()
        if line == b"":
            return None
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(b"{"):
            return json.loads(stripped.decode("utf-8")), "line"
        key, _, value = stripped.decode("ascii", errors="replace").partition(":")
        if key.lower() != "content-length":
            raise ValueError(f"unsupported MCP stdio header: {key}")
        length = int(value.strip())
        while True:
            header_line = sys.stdin.buffer.readline()
            if header_line == b"":
                return None
            if not header_line.strip():
                break
        return json.loads(sys.stdin.buffer.read(length).decode("utf-8")), "framed"


def _write_message(message: dict[str, Any], transport: str) -> None:
    payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
    if transport == "line":
        sys.stdout.buffer.write(payload + b"\n")
        sys.stdout.buffer.flush()
        return
    sys.stdout.buffer.write(f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii"))
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


def _rpc_result(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _rpc_error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


if __name__ == "__main__":
    raise SystemExit(main())
