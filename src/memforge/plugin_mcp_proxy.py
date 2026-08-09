#!/usr/bin/env python3
"""Stdlib-only MCP proxy used by MemForge agent-client plugins.

The canonical source is ``src/memforge/plugin_mcp_proxy.py``. Packaged
integration copies are generated with ``scripts/sync_plugin_mcp_proxy.py`` and
must not be edited independently.
"""

from __future__ import annotations

import base64
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

try:
    from .repository_context import resolve_repository_context
except ImportError:  # pragma: no cover - copied plugin package or direct file load
    try:
        from memforge_repository_context import resolve_repository_context
    except ImportError:
        import importlib.util

        _repository_context_path = Path(__file__).with_name("memforge_repository_context.py")
        if not _repository_context_path.exists():
            _repository_context_path = Path(__file__).with_name("repository_context.py")
        _repository_context_spec = importlib.util.spec_from_file_location(
            "memforge_repository_context",
            _repository_context_path,
        )
        if _repository_context_spec is None or _repository_context_spec.loader is None:
            raise
        _repository_context_module = importlib.util.module_from_spec(_repository_context_spec)
        sys.modules[_repository_context_spec.name] = _repository_context_module
        _repository_context_spec.loader.exec_module(_repository_context_module)
        resolve_repository_context = _repository_context_module.resolve_repository_context

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
SERVER_VERSION = "0.1.52"
CODEX_SANDBOX_STATE_META_CAPABILITY = "codex/sandbox-state-meta"
SERVER_INSTRUCTIONS = (
    "Repository context is optional. MemForge uses negotiated request-scoped host context when "
    "available. Otherwise, when the coding host exposes an exact current working directory, pass "
    "it as repository_context.working_directory. Never guess or use a plugin/install directory. "
    "Omit it when unavailable; the operation must continue. Explicit repository_context takes "
    "precedence. MemForge derives repository attribution locally and never sends the local path "
    "to the service. Workspace selection is request-scoped through the optional workspace_id "
    "tool parameter."
)
AGENT_CLIENT_VALUES = ["claude-code", "codex"]
RANKED_RETRIEVAL_INTENTS = ["general_hybrid", "known_item", "relationship"]
ROOTS_LIST_REQUEST_ID = "memforge-roots-list-1"
CURRENT_REPO_ONLY_DISABLED_ERROR = (
    "current_repo_only is not exposed by this MCP search tool. Omit the filter to search all visible memories."
)
SEARCH_ALLOWED_KEYS = frozenset(
    {
        "query",
        "intent",
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
RECENT_MEMORY_ALLOWED_KEYS = frozenset(
    {
        "source_ids",
        "time_range",
        "memory_types",
        "page_size",
        "cursor",
    }
)
RECENT_MEMORY_TYPES = frozenset({"fact", "decision", "convention", "procedure"})
MAX_DEFERRED_TOOL_CALLS = 32
_CLIENT_SUPPORTS_ROOTS = False
_PENDING_ROOTS_REQUEST_ID: str | None = None
_CLIENT_ROOT_PATHS: list[str] = []
_DEFERRED_TOOL_CALLS: list[tuple[Any, str, dict[str, Any], Any]] = []

REPOSITORY_CONTEXT_SCHEMA = {
    "type": "object",
    "description": (
        "Optional agent-host context used locally to identify the active repository and derive "
        "repository attribution. It never selects a MemForge workspace, and the local path is "
        "never forwarded to MemForge."
    ),
    "properties": {
        "working_directory": {
            "type": "string",
            "minLength": 1,
            "description": (
                "Absolute current working directory or file:// URI supplied by the coding client. "
                "Do not guess or reuse an installation/plugin directory; omit repository_context "
                "when the current working directory is unavailable."
            ),
        },
    },
    "required": ["working_directory"],
    "additionalProperties": False,
}

WORKSPACE_ID_SCHEMA = {
    "type": "string",
    "minLength": 1,
    "description": (
        "Optional workspace selected for this call. Omit it to use the server-side default or "
        "the caller's only accessible workspace."
    ),
}

REVIEW_MANIFEST_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "review_id": {"type": "string", "minLength": 1},
        "decision": {"type": "string", "enum": ["approve", "reject"]},
        "expected_fingerprint": {
            "type": "string",
            "minLength": 1,
            "description": "decision_fingerprint returned by list_memory_reviews or get_memory_review.",
        },
        "note": {
            "type": "string",
            "description": "Audit note; required when the selected action says requires_note=true.",
        },
        "rationale": {
            "type": "string",
            "maxLength": 2000,
            "description": "Agent explanation for the proposed decision; never grants authority.",
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "risk": {"type": "string", "enum": ["low", "medium", "high"], "default": "medium"},
    },
    "required": ["review_id", "decision", "expected_fingerprint"],
    "additionalProperties": False,
}

REVIEW_MANIFEST_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "mode": {"type": "string", "enum": ["validate", "apply"]},
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "review_id": {"type": "string"},
                    "decision": {"type": "string", "enum": ["approve", "reject"]},
                    "outcome": {
                        "type": "string",
                        "enum": [
                            "ready",
                            "applied",
                            "already_applied",
                            "stale",
                            "forbidden",
                            "not_found",
                            "invalid",
                            "failed",
                        ],
                    },
                    "status": {"type": ["string", "null"]},
                    "message": {"type": ["string", "null"]},
                    "decision_label": {"type": ["string", "null"]},
                    "consequence": {"type": ["string", "null"]},
                },
                "required": [
                    "review_id",
                    "decision",
                    "outcome",
                    "status",
                    "message",
                    "decision_label",
                    "consequence",
                ],
                "additionalProperties": False,
            },
        },
        "ready": {"type": "integer"},
        "applied": {"type": "integer"},
        "already_applied": {"type": "integer"},
        "stale": {"type": "integer"},
        "forbidden": {"type": "integer"},
        "failed": {"type": "integer"},
        "validation_receipt": {"type": ["string", "null"]},
    },
    "required": ["mode", "results", "ready", "applied", "already_applied", "stale", "forbidden", "failed"],
}


TOOLS: list[dict[str, Any]] = [
    {
        "name": "list_workspaces",
        "description": (
            "List workspaces accessible to the current principal and identify the optional "
            "server-side default. Discovery is optional; other tools resolve omitted workspace_id "
            "without requiring this tool to be called first. If multiple workspaces exist and no "
            "default is set, ask which workspace to use for the current request. Separately offer "
            "to set it as the default for automatic hooks; never change the default silently."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "set_default_workspace",
        "description": (
            "Persist the workspace used by automatic MemForge hooks and by requests that omit "
            "workspace_id. Call only after the user explicitly confirms making this their default. "
            "A workspace_id supplied to another tool is a one-request override and must never "
            "change this preference implicitly."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "workspace_id": {
                    "type": "string",
                    "minLength": 1,
                    "description": (
                        "Workspace to use by default for automatic hooks and requests that omit workspace_id."
                    ),
                }
            },
            "required": ["workspace_id"],
            "additionalProperties": False,
        },
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
        },
    },
    {
        "name": "search",
        "description": (
            "Search visible Memories. For source-specific requests, call list_sources first and pass "
            "exact source_ids. For broad or cross-source requests, omit source_filter; use time_range only "
            "when explicitly requested. Use list_recent_memories for deterministic source/time listings. "
            "Send a self-contained query in the user's language; preserve identifiers and domain terms, "
            "without retrieval-only translation or keyword stuffing. Use total_candidates and offset only "
            "within the ranked window; ranked queries are not exhaustive. In conflict_contexts, confirmed "
            "means reviewed contradiction and dismissed means reviewed non-conflict; neither retires a claim. "
            "Call get_memory for provenance."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "description": (
                        "Required self-contained natural-language query in the user's language. "
                        "Resolve conversational ellipsis before calling while preserving exact identifiers "
                        "and domain terms."
                    ),
                },
                "intent": {
                    "type": "string",
                    "enum": RANKED_RETRIEVAL_INTENTS,
                    "description": (
                        "Optional goal hint from conversation context: general_hybrid for open-ended recall, "
                        "known_item for a specifically named item, or relationship for connected-memory "
                        "exploration. For known_item, keep the query self-contained and quote the exact title, "
                        "name, or path when available. Omit when unsure; this never sets source or time facets."
                    ),
                },
                "repository_context": REPOSITORY_CONTEXT_SCHEMA,
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
                        "Page size within the bounded ranked window. Default is 10; increasing it up to 50 "
                        "does not make ranked retrieval exhaustive. Use list_recent_memories for deterministic "
                        "time/source enumeration."
                    ),
                },
                "offset": {
                    "type": "integer",
                    "default": 0,
                    "minimum": 0,
                    "description": (
                        "Zero-based offset within the ranked window. It paginates ranked candidates only "
                        "and must not be treated as exhaustive enumeration."
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
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "list_recent_memories",
        "description": (
            "List current Memories deterministically within an explicit half-open time window. "
            "This is a current-state view, not a source changelog: it does not report deleted or "
            "superseded historical states. Do not supply or invent a query. Resolve named sources "
            "with list_sources, pass timezone-qualified timestamps, and follow next_cursor."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "source_ids": {
                    "type": "array",
                    "minItems": 1,
                    "uniqueItems": True,
                    "items": {"type": "string", "minLength": 1},
                    "description": (
                        "Optional exact source IDs returned by list_sources. Omit for all visible active sources."
                    ),
                },
                "time_range": {
                    "type": "object",
                    "description": (
                        "Required half-open [start_at, end_at) window. Timestamps must include "
                        "an explicit UTC offset; the response reports the resolved UTC window."
                    ),
                    "properties": {
                        "date_type": {
                            "type": "string",
                            "enum": ["source_updated_at", "memory_updated_at"],
                            "default": "source_updated_at",
                        },
                        "start_at": {
                            "type": "string",
                            "format": "date-time",
                            "description": "Inclusive RFC 3339 timestamp with an explicit UTC offset.",
                        },
                        "end_at": {
                            "type": "string",
                            "format": "date-time",
                            "description": "Exclusive RFC 3339 timestamp with an explicit UTC offset.",
                        },
                    },
                    "required": ["start_at", "end_at"],
                    "additionalProperties": False,
                },
                "memory_types": {
                    "type": "array",
                    "minItems": 1,
                    "uniqueItems": True,
                    "items": {
                        "type": "string",
                        "enum": ["fact", "decision", "convention", "procedure"],
                    },
                    "description": "Optional exact current Memory types.",
                },
                "page_size": {
                    "type": "integer",
                    "default": 50,
                    "minimum": 1,
                    "maximum": 50,
                },
                "cursor": {
                    "type": "string",
                    "minLength": 1,
                    "description": (
                        "Opaque continuation cursor from the preceding page. Reuse the same filters and window."
                    ),
                },
            },
            "required": ["time_range"],
            "additionalProperties": False,
        },
    },
    {
        "name": "list_sources",
        "description": (
            "List search-eligible memory sources visible to the current principal. Use before "
            "source-specific search to resolve exact source_ids; skip for broad or cross-source "
            "requests. Returns source_id, name, type, status, counts, and last_synced_at."
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
            "Fetch full memory detail by ID when a search result is insufficient. Returns "
            "canonical content together with claim-local provenance in sources[].excerpt, "
            "supporting source and artifact locators, entity links, lifecycle metadata, and "
            "visibility-safe cross-source Review dispositions in conflict_contexts."
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
            "Fetch a MemForge source artifact from get_memory.sources[].content_url, "
            "get_memory.sources[].pdf_url, or get_memory.evidence_artifacts[].url. "
            "In file mode this local proxy writes the "
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
                        "A MemForge artifact URL from get_memory.sources[] or "
                        "get_memory.evidence_artifacts[].url, such as "
                        "/api/v1/documents/{doc_id}/content, /api/v1/documents/{doc_id}/pdf, "
                        "/api/v1/documents/{doc_id}/artifacts/{kind}, or "
                        "/api/v1/source-artifacts/{observation_revision_id}."
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
            "show a readable preview with the new durable claim, provenance/evidence, scope, and type, then get "
            "explicit confirmation via request_user_input if available, else a concise text question. "
            "Distill durable memory content as one self-contained claim or bounded procedure "
            "from the confirmed preview without unapproved semantic changes; do not copy the "
            "raw conversation into content. "
            "Keep provenance, confirmation details, test/deploy notes, and why-the-tool-was-called "
            "out of content; put source details in provenance. Never create memory silently."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": (
                        "One self-contained canonical durable claim or bounded procedure, distilled from "
                        "the user-confirmed readable preview. Preserve its meaning, scope, preconditions, "
                        "and required ordering without unapproved semantic changes. Do not "
                        "put confirmation details, provenance, test/deploy notes, or why-the-tool-was-called "
                        "into content; those belong in provenance or stay out of the memory."
                    ),
                },
                "provenance": {
                    "type": "string",
                    "description": (
                        "Required claim-local evidence or source context. It is stored separately from "
                        "canonical content and returned as get_memory.sources[].excerpt. Use it for "
                        "details that explain where the memory came from or how it was verified, but "
                        "exclude secrets and raw oversized logs."
                    ),
                },
                "memory_type": {
                    "type": "string",
                    "enum": ["fact", "decision", "convention", "procedure"],
                    "default": "fact",
                },
                "confidence": {"type": "number"},
                "repository_context": REPOSITORY_CONTEXT_SCHEMA,
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
            "List an exact caller-visible Review queue for agent analysis. Use filters to build "
            "a bounded cohort, then inspect each presentation.actions entry and decision_fingerprint."
        ),
        "annotations": {"readOnlyHint": True},
        "inputSchema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["open", "pending", "stale", "approved", "rejected"],
                    "default": "open",
                },
                "origin": {"type": "string", "enum": ["memory", "lifecycle"]},
                "kind": {"type": "string", "enum": ["supersede", "cross_source_conflict"]},
                "source_id": {"type": "string", "minLength": 1},
                "limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 20},
                "offset": {"type": "integer", "minimum": 0, "default": 0},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "get_memory_review",
        "description": "Fetch full current/proposed memory details for a memory-review decision.",
        "annotations": {"readOnlyHint": True},
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
            "Resolve one Review after explicit user confirmation using the exact action shown "
            "in presentation.actions. Consequences vary by Review kind: lifecycle decisions may "
            "change Memory state, while cross-source conflict decisions keep both Memories active. "
            "Never resolve silently and never reuse a stale decision_fingerprint."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "review_id": {"type": "string", "description": "The memory review ID."},
                "decision": {"type": "string", "enum": ["approve", "reject"]},
                "expected_fingerprint": {
                    "type": "string",
                    "description": "Current decision_fingerprint from the Review response.",
                },
                "note": {
                    "type": "string",
                    "description": "Audit note; required when the selected action says requires_note=true.",
                },
            },
            "required": ["review_id", "decision", "expected_fingerprint"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": False, "destructiveHint": True},
    },
    {
        "name": "validate_memory_review_decisions",
        "description": (
            "Read-only validation of a bounded Review Decision Manifest. Use this after agent "
            "analysis to detect stale fingerprints, missing notes, and authorization failures "
            "before asking the user to confirm the exact cohort."
        ),
        "annotations": {"readOnlyHint": True},
        "outputSchema": REVIEW_MANIFEST_RESPONSE_SCHEMA,
        "inputSchema": {
            "type": "object",
            "properties": {
                "decisions": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 50,
                    "items": REVIEW_MANIFEST_DECISION_SCHEMA,
                },
            },
            "required": ["decisions"],
            "additionalProperties": False,
        },
    },
    {
        "name": "apply_memory_review_decisions",
        "description": (
            "Apply one previously validated Decision Manifest only after one explicit user "
            "confirmation of the displayed cohort. Each item keeps its own authorization, stale "
            "guard, atomic lifecycle path, audit, and result; this tool creates no batch state."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "decisions": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 50,
                    "items": REVIEW_MANIFEST_DECISION_SCHEMA,
                },
                "validation_receipt": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Receipt returned by validating this exact manifest as the same principal.",
                },
            },
            "required": ["decisions", "validation_receipt"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": False, "destructiveHint": True},
        "outputSchema": REVIEW_MANIFEST_RESPONSE_SCHEMA,
    },
]

for _tool in TOOLS:
    if _tool["name"] in {"list_workspaces", "set_default_workspace"}:
        continue
    _tool["inputSchema"]["properties"]["repository_context"] = REPOSITORY_CONTEXT_SCHEMA
    _tool["inputSchema"]["properties"]["workspace_id"] = WORKSPACE_ID_SCHEMA


class NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


class ResourceTarget:
    def __init__(
        self,
        resource_id: str,
        kind: str,
        relative_url: str,
        request_url: str,
        *,
        identity_key: str = "doc_id",
    ) -> None:
        self.resource_id = resource_id
        self.kind = kind
        self.relative_url = relative_url
        self.request_url = request_url
        self.identity_key = identity_key


class ToolCallContext:
    __slots__ = ("target", "repo_identifier")

    def __init__(self, target: Any, repo_identifier: str | None) -> None:
        self.target = target
        self.repo_identifier = repo_identifier


def main() -> int:
    while True:
        envelope = _read_message()
        if envelope is None:
            return 0
        message, transport = envelope
        response = _handle_rpc_message(message)
        responses = response if isinstance(response, list) else [response]
        for item in responses:
            if item is not None:
                _write_message(item, transport)


def _handle_rpc_message(message: dict[str, Any]) -> dict[str, Any] | list[dict[str, Any]] | None:
    method = message.get("method")
    request_id = message.get("id")
    try:
        if method is None:
            return _handle_rpc_response(message)
        if method == "initialize":
            _record_client_capabilities(message)
            result = {
                "protocolVersion": "2025-03-26",
                "capabilities": {
                    "tools": {"listChanged": False},
                    "experimental": {CODEX_SANDBOX_STATE_META_CAPABILITY: {}},
                },
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                "instructions": SERVER_INSTRUCTIONS,
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
            request_meta = params.get("_meta")
            if _PENDING_ROOTS_REQUEST_ID is not None and not _has_request_repository_context(arguments, request_meta):
                if len(_DEFERRED_TOOL_CALLS) >= MAX_DEFERRED_TOOL_CALLS:
                    return _rpc_error(
                        request_id,
                        -32001,
                        "Workspace context is still resolving; retry the tool call.",
                    )
                _DEFERRED_TOOL_CALLS.append((request_id, name, arguments, request_meta))
                return None
            return _tool_call_response(request_id, name, arguments, request_meta)
        if request_id is None:
            return None
        return _rpc_error(request_id, -32601, f"Method not found: {method}")
    except Exception as exc:  # pragma: no cover - defensive MCP boundary
        if request_id is None:
            return None
        return _rpc_error(request_id, -32603, f"Internal error: {exc}")


def _tool_call_response(
    request_id: Any,
    name: str,
    arguments: dict[str, Any],
    request_meta: Any = None,
) -> dict[str, Any]:
    try:
        payload = _call_tool(name, arguments, request_meta=request_meta)
        return _rpc_result(
            request_id,
            _tool_result(
                payload,
                structured=name
                in {
                    "validate_memory_review_decisions",
                    "apply_memory_review_decisions",
                },
            ),
        )
    except Exception as exc:  # pragma: no cover - defensive MCP boundary
        return _rpc_error(request_id, -32603, f"Internal error: {exc}")


def _call_tool(name: str, args: dict[str, Any], *, request_meta: Any = None) -> dict[str, Any]:
    if name == "list_workspaces":
        if args:
            return {"error": "list_workspaces does not accept parameters"}
        return _http_json(
            "GET",
            "/workspaces",
            None,
            target=configured_target(),
            workspace_id=None,
        )
    if name == "set_default_workspace":
        if any(key != "workspace_id" for key in args):
            return {"error": "set_default_workspace accepts only workspace_id"}
        try:
            workspace_id = _required_string_arg(args, "workspace_id")
        except ValueError as exc:
            return {"error": str(exc)}
        return _http_json(
            "PUT",
            "/me/default-workspace",
            {"workspace_id": workspace_id},
            target=configured_target(),
            workspace_id=None,
        )
    try:
        call_context = _tool_call_context(
            args.get("repository_context"),
            request_meta=request_meta,
        )
    except ValueError as exc:
        return {"error": str(exc)}
    try:
        workspace_id = _optional_string_arg(args, "workspace_id")
    except ValueError as exc:
        return {"error": str(exc)}
    args = {key: value for key, value in args.items() if key not in {"repository_context", "workspace_id"}}
    if name == "search":
        try:
            body = _search_args_with_context(args, repo_identifier=call_context.repo_identifier)
        except ValueError as exc:
            return {"error": str(exc)}
        return _compact_search_response(
            _http_json(
                "POST",
                "/memories/search",
                body,
                target=call_context.target,
                workspace_id=workspace_id,
            )
        )
    if name == "list_recent_memories":
        try:
            body = _recent_memory_args(args)
        except ValueError as exc:
            return {"error": str(exc)}
        return _compact_recent_memories_response(
            _http_json(
                "POST",
                "/memories/recent",
                body,
                target=call_context.target,
                workspace_id=workspace_id,
            )
        )
    if name == "list_sources":
        if args:
            return {"error": "list_sources does not accept parameters"}
        return _http_json(
            "GET",
            "/sources/searchable",
            None,
            target=call_context.target,
            workspace_id=workspace_id,
        )
    if name == "get_memory":
        memory_id = str(args.get("memory_id") or "").strip()
        if not memory_id:
            return {"error": "memory_id is required"}
        return _compact_memory_response(
            _http_json(
                "GET",
                f"/memories/{quote(memory_id, safe='')}?include_private=true",
                None,
                target=call_context.target,
                workspace_id=workspace_id,
            )
        )
    if name == "get_resource":
        return _handle_get_resource(
            args,
            target=call_context.target,
            workspace_id=workspace_id,
        )
    if name == "create_memory":
        try:
            memory_type = str(args.get("memory_type") or "fact").strip()
            if memory_type not in {"fact", "decision", "convention", "procedure"}:
                raise ValueError("memory_type must be fact, decision, convention, or procedure")
            body = {
                "content": _required_string_arg(args, "content"),
                "provenance": _required_string_arg(args, "provenance"),
                "memory_type": memory_type,
                "client": _mcp_client(),
            }
            if "confidence" in args:
                confidence = args.get("confidence")
                if not isinstance(confidence, (int, float)):
                    raise ValueError("confidence must be a number")
                body["confidence"] = float(confidence)
            if call_context.repo_identifier:
                body["repo_identifier"] = call_context.repo_identifier
            idempotency_key = str(args.get("idempotency_key") or "").strip()
            if idempotency_key:
                body["idempotency_key"] = idempotency_key
        except ValueError as exc:
            return {"error": str(exc)}
        return _http_json(
            "POST",
            "/memories/create",
            body,
            target=call_context.target,
            workspace_id=workspace_id,
        )
    if name == "retire_memory":
        try:
            memory_id = _required_string_arg(args, "memory_id")
            body = {
                "reason": _required_string_arg(args, "reason"),
                "expected_content_hash": _required_string_arg(args, "expected_content_hash"),
            }
        except ValueError as exc:
            return {"error": str(exc)}
        return _http_json(
            "POST",
            f"/memories/{quote(memory_id, safe='')}/retire",
            body,
            target=call_context.target,
            workspace_id=workspace_id,
        )
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
        return _http_json(
            "POST",
            f"/memories/{quote(memory_id, safe='')}/replace",
            body,
            target=call_context.target,
            workspace_id=workspace_id,
        )
    if name == "list_memory_reviews":
        allowed = {"status", "origin", "kind", "source_id", "limit", "offset"}
        unknown = sorted(set(args) - allowed)
        if unknown:
            return {"error": "Unsupported list_memory_reviews parameter(s): " + ", ".join(unknown)}
        try:
            limit = _optional_int_arg(args, "limit", 20)
            offset = _optional_int_arg(args, "offset", 0)
            if not 1 <= limit <= 500:
                raise ValueError("limit must be between 1 and 500")
            if offset < 0:
                raise ValueError("offset must not be negative")
            query = {
                key: value
                for key, value in {
                    "status": str(args.get("status") or "open"),
                    "origin": _optional_string_arg(args, "origin"),
                    "kind": _optional_string_arg(args, "kind"),
                    "source_id": _optional_string_arg(args, "source_id"),
                    "limit": limit,
                    "offset": offset,
                }.items()
                if value is not None
            }
        except ValueError as exc:
            return {"error": str(exc)}
        return _http_json(
            "GET",
            "/memory-reviews?" + urlencode(query),
            None,
            target=call_context.target,
            workspace_id=workspace_id,
        )
    if name == "get_memory_review":
        try:
            review_id = _required_string_arg(args, "review_id")
        except ValueError as exc:
            return {"error": str(exc)}
        return _http_json(
            "GET",
            f"/memory-reviews/{quote(review_id, safe='')}",
            None,
            target=call_context.target,
            workspace_id=workspace_id,
        )
    if name == "resolve_memory_review":
        unknown = sorted(set(args) - {"review_id", "decision", "expected_fingerprint", "note"})
        if unknown:
            return {"error": "Unsupported resolve_memory_review parameter(s): " + ", ".join(unknown)}
        try:
            review_id = _required_string_arg(args, "review_id")
            decision = _required_string_arg(args, "decision")
            if decision not in {"approve", "reject"}:
                raise ValueError("decision must be approve or reject")
            expected_fingerprint = _required_string_arg(args, "expected_fingerprint")
            note = str(args.get("note") or "").strip()
            if decision == "reject" and not note:
                raise ValueError("note is required when decision is reject")
            body = {"expected_fingerprint": expected_fingerprint}
            if note:
                body["note"] = note
        except ValueError as exc:
            return {"error": str(exc)}
        return _http_json(
            "POST",
            f"/memory-reviews/{quote(review_id, safe='')}/{decision}",
            body,
            target=call_context.target,
            workspace_id=workspace_id,
        )
    if name in {"validate_memory_review_decisions", "apply_memory_review_decisions"}:
        try:
            body = _review_manifest_body(
                args,
                require_receipt=name == "apply_memory_review_decisions",
            )
        except ValueError as exc:
            return {"error": str(exc)}
        operation = "validate" if name == "validate_memory_review_decisions" else "apply"
        return _http_json(
            "POST",
            f"/memory-reviews/decisions/{operation}",
            body,
            target=call_context.target,
            workspace_id=workspace_id,
        )
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


def _review_manifest_body(
    args: dict[str, Any],
    *,
    require_receipt: bool,
) -> dict[str, Any]:
    unknown = sorted(set(args) - {"decisions", "validation_receipt"})
    if unknown:
        raise ValueError("Unsupported Review manifest parameter(s): " + ", ".join(unknown))
    raw_decisions = args.get("decisions")
    if not isinstance(raw_decisions, list) or not 1 <= len(raw_decisions) <= 50:
        raise ValueError("decisions must contain between 1 and 50 items")
    allowed = {
        "review_id",
        "decision",
        "expected_fingerprint",
        "note",
        "rationale",
        "confidence",
        "risk",
    }
    normalized: list[dict[str, Any]] = []
    review_ids: set[str] = set()
    for index, raw in enumerate(raw_decisions):
        if not isinstance(raw, dict):
            raise ValueError(f"decisions[{index}] must be an object")
        item_unknown = sorted(set(raw) - allowed)
        if item_unknown:
            raise ValueError(f"Unsupported decisions[{index}] parameter(s): " + ", ".join(item_unknown))
        review_id = _required_string_arg(raw, "review_id")
        if review_id in review_ids:
            raise ValueError(f"duplicate review_id: {review_id}")
        review_ids.add(review_id)
        decision = _required_string_arg(raw, "decision")
        if decision not in {"approve", "reject"}:
            raise ValueError(f"decisions[{index}].decision must be approve or reject")
        fingerprint = _required_string_arg(raw, "expected_fingerprint")
        note = _optional_string_arg(raw, "note")
        if decision == "reject" and not note:
            raise ValueError(f"decisions[{index}].note is required when decision is reject")
        risk = str(raw.get("risk") or "medium").strip()
        if risk not in {"low", "medium", "high"}:
            raise ValueError(f"decisions[{index}].risk must be low, medium, or high")
        confidence = raw.get("confidence")
        if confidence is not None and (
            isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1
        ):
            raise ValueError(f"decisions[{index}].confidence must be between 0 and 1")
        item: dict[str, Any] = {
            "review_id": review_id,
            "decision": decision,
            "expected_fingerprint": fingerprint,
            "risk": risk,
        }
        for optional in ("note", "rationale"):
            value = _optional_string_arg(raw, optional)
            if value is not None:
                item[optional] = value
        if confidence is not None:
            item["confidence"] = float(confidence)
        normalized.append(item)
    body: dict[str, Any] = {"decisions": normalized}
    receipt = _optional_string_arg(args, "validation_receipt")
    if require_receipt and receipt is None:
        raise ValueError("validation_receipt is required before apply")
    if receipt is not None:
        body["validation_receipt"] = receipt
    return body


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


def _search_args_with_context(args: dict[str, Any], *, repo_identifier: str | None = None) -> dict[str, Any]:
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
    intent = body.get("intent")
    if intent is not None and intent not in RANKED_RETRIEVAL_INTENTS:
        raise ValueError("intent must be general_hybrid, known_item, or relationship")
    body["include_private"] = True
    body["include_superseded"] = False
    if repo_identifier:
        body["active_repo_identifier"] = repo_identifier
    source_filter = body.get("source_filter")
    if isinstance(source_filter, dict):
        unknown_filter_keys = sorted(set(source_filter) - SOURCE_FILTER_ALLOWED_KEYS)
        if unknown_filter_keys:
            raise ValueError(
                "Unsupported source_filter parameter(s): "
                + ", ".join(unknown_filter_keys)
                + ". Omit repo-scoped facets instead of guessing."
            )
        source_ids = source_filter.get("source_ids")
        if source_ids is not None:
            if (
                not isinstance(source_ids, list)
                or not source_ids
                or not all(isinstance(item, str) and item.strip() for item in source_ids)
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
    if not query:
        if has_deterministic_filter:
            raise ValueError("search.query is required; use list_recent_memories for deterministic listings")
        raise ValueError("search.query is required")
    return body


def _recent_memory_args(args: dict[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(args) - RECENT_MEMORY_ALLOWED_KEYS)
    if unknown:
        raise ValueError("Unsupported list_recent_memories parameter(s): " + ", ".join(unknown))
    body = dict(args)
    source_ids = body.get("source_ids")
    if source_ids is not None:
        if (
            not isinstance(source_ids, list)
            or not source_ids
            or not all(isinstance(source_id, str) and source_id.strip() for source_id in source_ids)
        ):
            raise ValueError("source_ids must be a non-empty array of source IDs from list_sources")
        body["source_ids"] = list(dict.fromkeys(source_id.strip() for source_id in source_ids))
    memory_types = body.get("memory_types")
    if memory_types is not None:
        if (
            not isinstance(memory_types, list)
            or not memory_types
            or not all(
                isinstance(memory_type, str) and memory_type in RECENT_MEMORY_TYPES for memory_type in memory_types
            )
        ):
            raise ValueError("memory_types must contain fact, decision, convention, or procedure")
        body["memory_types"] = list(dict.fromkeys(memory_types))
    if "page_size" in body:
        body["page_size"] = _required_bounded_int_arg(
            body,
            "page_size",
            minimum=SEARCH_TOP_K_MIN,
            maximum=SEARCH_TOP_K_MAX,
        )
    if "cursor" in body:
        body["cursor"] = _required_string_arg(body, "cursor")
    body["time_range"] = _validate_recent_time_range(body.get("time_range"))
    body["include_private"] = True
    return body


def _validate_recent_time_range(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("time_range is required")
    unknown = sorted(set(value) - {"date_type", "start_at", "end_at"})
    if unknown:
        raise ValueError("Unsupported time_range parameter(s): " + ", ".join(unknown))
    date_type = value.get("date_type", "source_updated_at")
    if date_type not in {"source_updated_at", "memory_updated_at"}:
        raise ValueError("time_range.date_type must be source_updated_at or memory_updated_at")
    parsed: dict[str, datetime] = {}
    normalized: dict[str, str] = {"date_type": date_type}
    for key in ("start_at", "end_at"):
        item = value.get(key)
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"time_range.{key} is required")
        text = item.strip()
        try:
            instant = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"time_range.{key} must be an RFC 3339 timestamp") from exc
        if instant.tzinfo is None or instant.utcoffset() is None:
            raise ValueError(f"time_range.{key} must include an explicit UTC offset")
        parsed[key] = instant
        normalized[key] = text
    if parsed["start_at"] >= parsed["end_at"]:
        raise ValueError("time_range.start_at must be before end_at")
    return normalized


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
    return resolve_repository_context(root_paths=_CLIENT_ROOT_PATHS).repo_identifier


def _tool_call_context(value: Any, *, request_meta: Any = None) -> ToolCallContext:
    if value is not None:
        return _tool_call_context_from_working_directory(
            value,
            field_name="repository_context.working_directory",
            require_object=True,
        )

    sandbox_cwd = _codex_sandbox_cwd(request_meta)
    if sandbox_cwd is None:
        return ToolCallContext(
            target=configured_target(),
            repo_identifier=_active_repo_identifier(),
        )

    return _tool_call_context_from_working_directory(
        sandbox_cwd,
        field_name=f"{CODEX_SANDBOX_STATE_META_CAPABILITY}.sandboxCwd",
    )


def _tool_call_context_from_working_directory(
    value: Any,
    *,
    field_name: str,
    require_object: bool = False,
) -> ToolCallContext:
    if require_object and not isinstance(value, dict):
        raise ValueError("repository_context must be an object with working_directory")
    context = value if isinstance(value, dict) else {"working_directory": value}
    unknown = sorted(set(context) - {"working_directory"})
    if unknown:
        raise ValueError("Unsupported repository_context parameter(s): " + ", ".join(unknown))
    working_directory = context.get("working_directory")
    if not isinstance(working_directory, str) or not working_directory.strip():
        raise ValueError(f"{field_name} must be an absolute path or file:// URI")
    repository_path = _root_path({"uri": working_directory})
    if repository_path is None or not Path(repository_path).is_absolute():
        raise ValueError(f"{field_name} must be an absolute path or file:// URI")
    repository_context = resolve_repository_context(
        working_directory=working_directory,
    )
    if repository_context.state != "exact" or not repository_context.repo_identifier:
        raise ValueError(f"{field_name} must resolve to a Git repository with an origin remote")
    return ToolCallContext(
        target=configured_target(),
        repo_identifier=repository_context.repo_identifier,
    )


def _repository_identifier_for_tool_context(value: Any) -> str | None:
    return _tool_call_context(value).repo_identifier


def _codex_sandbox_cwd(request_meta: Any) -> str | None:
    if not isinstance(request_meta, dict) or CODEX_SANDBOX_STATE_META_CAPABILITY not in request_meta:
        return None
    sandbox_state = request_meta.get(CODEX_SANDBOX_STATE_META_CAPABILITY)
    if not isinstance(sandbox_state, dict):
        raise ValueError(f"{CODEX_SANDBOX_STATE_META_CAPABILITY}.sandboxCwd must be an absolute path or file:// URI")
    sandbox_cwd = sandbox_state.get("sandboxCwd")
    if not isinstance(sandbox_cwd, str) or not sandbox_cwd.strip():
        raise ValueError(f"{CODEX_SANDBOX_STATE_META_CAPABILITY}.sandboxCwd must be an absolute path or file:// URI")
    return sandbox_cwd


def _has_request_repository_context(arguments: dict[str, Any], request_meta: Any) -> bool:
    return arguments.get("repository_context") is not None or (
        isinstance(request_meta, dict) and CODEX_SANDBOX_STATE_META_CAPABILITY in request_meta
    )


def _mcp_client() -> str:
    value = os.getenv("MEMFORGE_MCP_CLIENT", "").strip()
    if value in AGENT_CLIENT_VALUES:
        return value
    return "codex"


def _record_client_capabilities(message: dict[str, Any]) -> None:
    global _CLIENT_SUPPORTS_ROOTS, _PENDING_ROOTS_REQUEST_ID, _CLIENT_ROOT_PATHS
    params = message.get("params") if isinstance(message.get("params"), dict) else {}
    capabilities = params.get("capabilities") if isinstance(params.get("capabilities"), dict) else {}
    _CLIENT_SUPPORTS_ROOTS = isinstance(capabilities.get("roots"), dict)
    _PENDING_ROOTS_REQUEST_ID = None
    _CLIENT_ROOT_PATHS = []
    _DEFERRED_TOOL_CALLS.clear()


def _request_client_roots() -> dict[str, Any]:
    global _PENDING_ROOTS_REQUEST_ID
    _PENDING_ROOTS_REQUEST_ID = ROOTS_LIST_REQUEST_ID
    return {"jsonrpc": "2.0", "id": ROOTS_LIST_REQUEST_ID, "method": "roots/list"}


def _handle_rpc_response(message: dict[str, Any]) -> list[dict[str, Any]] | None:
    global _PENDING_ROOTS_REQUEST_ID, _CLIENT_ROOT_PATHS
    if message.get("id") != _PENDING_ROOTS_REQUEST_ID:
        return None
    _PENDING_ROOTS_REQUEST_ID = None
    error = message.get("error")
    if isinstance(error, dict):
        _CLIENT_ROOT_PATHS = []
    else:
        result = message.get("result") if isinstance(message.get("result"), dict) else {}
        roots = result.get("roots") if isinstance(result.get("roots"), list) else []
        _CLIENT_ROOT_PATHS = [path for item in roots if (path := _root_path(item))]
    deferred = tuple(_DEFERRED_TOOL_CALLS)
    _DEFERRED_TOOL_CALLS.clear()
    return [
        _tool_call_response(request_id, name, arguments, request_meta)
        for request_id, name, arguments, request_meta in deferred
    ] or None


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


def _tool_result(payload: dict[str, Any], *, structured: bool = False) -> dict[str, Any]:
    content_type = str(payload.get("content_type") or "").split(";", 1)[0].strip().lower()
    encoded = payload.get("data_base64")
    if content_type.startswith("image/") and isinstance(encoded, str):
        metadata = {key: value for key, value in payload.items() if key != "data_base64"}
        return {
            "content": [
                {"type": "text", "text": json.dumps(metadata, indent=2)},
                {"type": "image", "data": encoded, "mimeType": content_type},
            ],
            "isError": False,
        }
    result = {
        "content": [{"type": "text", "text": json.dumps(payload, indent=2)}],
        "isError": False,
    }
    if structured and "error" not in payload:
        result["structuredContent"] = payload
    return result


def _compact_search_response(payload: dict[str, Any]) -> dict[str, Any]:
    if "error" in payload:
        return payload

    compact: dict[str, Any] = {}
    results = payload.get("results")
    if isinstance(results, list):
        compact["results"] = [_compact_search_result(result) for result in results if isinstance(result, dict)]

    retrieval_intent = payload.get("retrieval_intent")
    if isinstance(retrieval_intent, dict):
        compact["retrieval_intent"] = {
            key: retrieval_intent.get(key)
            for key in (
                "requested_intent",
                "resolved_intent",
                "intent_source",
                "fallback_reason",
            )
        }

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


def _compact_recent_memories_response(payload: dict[str, Any]) -> dict[str, Any]:
    if "error" in payload:
        return payload
    compact: dict[str, Any] = {}
    results = payload.get("results")
    if isinstance(results, list):
        compact["results"] = [_compact_recent_memory_result(result) for result in results if isinstance(result, dict)]
    for key in (
        "result_kind",
        "is_changelog",
        "time_field",
        "resolved_window",
        "listing_watermark",
        "cursor_kind",
        "consistency",
        "total_candidates",
        "candidate_count_kind",
        "count_scope",
        "limit",
        "has_more",
        "next_cursor",
    ):
        if key in payload:
            compact[key] = payload[key]
    return compact


def _compact_recent_memory_result(result: dict[str, Any]) -> dict[str, Any]:
    compact = _compact_search_result(result)
    compact.pop("relevance_score", None)
    if "matched_at" in result:
        compact["matched_at"] = result["matched_at"]
    return compact


def _compact_search_result(result: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in (
        "memory_id",
        "memory_type",
        "summary",
        "confidence",
        "relevance_score",
        "freshness",
        "status",
        "contradiction_warning",
        "conflict_contexts",
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
        "status",
        "entity_refs",
        "conflict_contexts",
    ):
        if key in payload:
            compact[key] = payload[key]

    sources = payload.get("sources")
    if isinstance(sources, list):
        compact["sources"] = [_compact_memory_source(source) for source in sources if isinstance(source, dict)]
    evidence_artifacts = payload.get("evidence_artifacts")
    if isinstance(evidence_artifacts, list):
        compact["evidence_artifacts"] = [
            _compact_memory_evidence_artifact(artifact) for artifact in evidence_artifacts if isinstance(artifact, dict)
        ]
    return compact


def _compact_memory_evidence_artifact(
    artifact: dict[str, Any],
) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in (
        "summary",
        "evidence_role",
        "filename",
        "content_type",
        "size_bytes",
        "url",
    ):
        if key in artifact:
            compact[key] = artifact[key]
    return compact


def _compact_memory_source(source: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in (
        "doc_id",
        "source_type",
        "support_kind",
        "doc_title",
        "excerpt",
        "source_url",
        "content_url",
        "pdf_url",
        "source_updated_at",
    ):
        if key in source:
            compact[key] = source[key]
    return compact


def _resource_url(
    path: str,
    *,
    target: Any | None = None,
    workspace_id: str | None = None,
) -> str:
    return (target or _configured_target()).resource_url(
        path,
        workspace_id=workspace_id,
    )


def _configured_target():
    return configured_target()


def _api_headers(*, json_body: bool = False) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if json_body:
        headers["Content-Type"] = "application/json"
    token = configured_api_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _http_json(
    method: str,
    path: str,
    body: dict[str, Any] | None,
    *,
    target: Any | None = None,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    target = target or _configured_target()
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = Request(
        _resource_url(path, target=target, workspace_id=workspace_id),
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
            "api_url": target.api_base,
            "detail": str(exc),
        }


def _handle_get_resource(
    args: dict[str, Any],
    *,
    target: Any | None = None,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    target = target or _configured_target()
    mode = str(args.get("mode") or "text").strip().lower()
    if mode not in {"text", "file", "base64"}:
        return {"error": f"unsupported mode: {mode}", "supported_modes": ["text", "file", "base64"]}

    max_bytes = _positive_int_arg(args, "max_bytes", 2_000_000)
    if isinstance(max_bytes, dict):
        return max_bytes
    max_chars = _positive_int_arg(args, "max_chars", 120_000)
    if isinstance(max_chars, dict):
        return max_chars

    resource_target = _parse_resource_url(
        str(args.get("url") or "").strip(),
        target,
        workspace_id=workspace_id,
    )
    if resource_target is None:
        return {
            "error": "unsupported resource URL",
            "hint": (
                "Use a relative MemForge /api/v1/documents/{doc_id}/content, /pdf, "
                "/artifacts/{kind}, or /api/v1/source-artifacts/{observation_revision_id} "
                "URL, or an absolute URL under MEMFORGE_API_URL."
            ),
        }

    try:
        if mode == "file":
            return _fetch_resource_file(resource_target)
        return _fetch_resource_inline(
            resource_target,
            mode=mode,
            max_bytes=max_bytes,
            max_chars=max_chars,
        )
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        return {
            "error": "resource fetch failed",
            "status_code": exc.code,
            "url": resource_target.relative_url,
            "detail": detail,
        }
    except (OSError, URLError) as exc:
        return {
            "error": "resource fetch failed",
            "url": resource_target.relative_url,
            "detail": str(exc),
        }


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
            safe_resource = _safe_cache_component(target.resource_id) or "resource"
            safe_kind = _safe_cache_component(target.kind) or "artifact"
            with tempfile.NamedTemporaryFile(
                "wb",
                dir=cache_root,
                prefix=f".{safe_resource}-{safe_kind}-",
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
            observed_sha256 = digest.hexdigest()
            _verify_resource_integrity(
                headers,
                observed_size=observed_size,
                observed_sha256=observed_sha256,
            )
            final_path = _cache_artifact_path(
                target.resource_id,
                target.kind,
                filename,
                observed_sha256[:16],
            )
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
    metadata = {
        target.identity_key: target.resource_id,
        "kind": target.kind,
        "content_type": headers.get("content-type", "application/octet-stream"),
        "filename": _resource_filename(headers, target),
        "size_bytes": _response_size_bytes(headers, observed_size),
        "url": target.relative_url,
        "mode": mode,
    }
    if sha256 := headers.get("x-content-sha256"):
        metadata["sha256"] = sha256
    return metadata


def _verify_resource_integrity(
    headers: dict[str, str],
    *,
    observed_size: int,
    observed_sha256: str,
) -> None:
    """Fail closed when authoritative resource headers disagree with the stream."""

    expected_size = headers.get("content-length")
    if expected_size:
        try:
            parsed_size = int(expected_size)
        except ValueError as exc:
            raise OSError("resource response has an invalid Content-Length") from exc
        if parsed_size != observed_size:
            raise OSError("resource byte count does not match Content-Length")
    expected_sha256 = headers.get("x-content-sha256")
    if expected_sha256 and expected_sha256.strip().lower() != observed_sha256:
        raise OSError("resource SHA-256 does not match X-Content-SHA256")


def _parse_resource_url(
    url: str,
    target: Any,
    *,
    workspace_id: str | None = None,
) -> ResourceTarget | None:
    parsed = urlparse(url)
    base = urlparse(target.origin)
    if parsed.fragment:
        return None
    query = parse_qsl(parsed.query, keep_blank_values=True)
    if any(key != "workspace_id" for key, _value in query):
        return None
    locator_workspace_ids = [value.strip() for key, value in query if key == "workspace_id"]
    if len(locator_workspace_ids) > 1 or any(not value for value in locator_workspace_ids):
        return None
    locator_workspace_id = locator_workspace_ids[0] if locator_workspace_ids else None
    if workspace_id and locator_workspace_id and workspace_id != locator_workspace_id:
        return None
    effective_workspace_id = workspace_id or locator_workspace_id

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
    relative_url = path
    if locator_workspace_id:
        relative_url += "?" + urlencode({"workspace_id": locator_workspace_id})
    if len(parts) == 5 and parts[:3] == ["api", "v1", "documents"] and parts[4] == "content":
        return ResourceTarget(
            parts[3],
            "content",
            relative_url,
            _resource_url(
                path[len("/api/v1") :],
                target=target,
                workspace_id=effective_workspace_id,
            ),
        )
    if len(parts) == 5 and parts[:3] == ["api", "v1", "documents"] and parts[4] == "pdf":
        return ResourceTarget(
            parts[3],
            "pdf",
            relative_url,
            _resource_url(
                path[len("/api/v1") :],
                target=target,
                workspace_id=effective_workspace_id,
            ),
        )
    if len(parts) == 6 and parts[:3] == ["api", "v1", "documents"] and parts[4] == "artifacts":
        return ResourceTarget(
            parts[3],
            parts[5],
            relative_url,
            _resource_url(
                path[len("/api/v1") :],
                target=target,
                workspace_id=effective_workspace_id,
            ),
        )
    if len(parts) == 4 and parts[:3] == ["api", "v1", "source-artifacts"]:
        return ResourceTarget(
            parts[3],
            "source_artifact",
            relative_url,
            _resource_url(
                path[len("/api/v1") :],
                target=target,
                workspace_id=effective_workspace_id,
            ),
            identity_key="observation_revision_id",
        )
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
    return f"{target.resource_id}-{target.kind}{suffix}"


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


def _cache_artifact_path(resource_id: str, kind: str, filename: str, digest: str) -> Path:
    safe_resource = _safe_cache_component(resource_id) or "resource"
    safe_kind = _safe_cache_component(kind) or "artifact"
    suffix = Path(filename).suffix or ".bin"
    return _artifact_cache_root() / f"{safe_resource}-{safe_kind}-{digest}{suffix}"


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
