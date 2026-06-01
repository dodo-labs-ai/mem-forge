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
import tempfile
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, unquote, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

DEFAULT_API_URL = "http://127.0.0.1:8765"
DEFAULT_TIMEOUT_SECONDS = 60.0
SERVER_NAME = "memforge"
SERVER_VERSION = "0.1.1"


TOOLS: list[dict[str, Any]] = [
    {
        "name": "search",
        "description": (
            "Search the team memory layer for facts, decisions, conventions, and procedures. "
            "Returns ranked results with provenance and source artifact URLs when available."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language search query"},
                "memory_types": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["fact", "decision", "convention", "procedure"]},
                },
                "sources": {"type": "array", "items": {"type": "string"}},
                "time_range": {
                    "type": "object",
                    "properties": {
                        "after": {"type": "string"},
                        "before": {"type": "string"},
                    },
                },
                "entities": {"type": "array", "items": {"type": "string"}},
                "include_superseded": {"type": "boolean", "default": False},
                "top_k": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_memory",
        "description": (
            "Fetch full memory detail by ID when complete provenance, supporting sources, "
            "entity links, or lifecycle metadata are needed."
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
            "Fetch a MemForge source artifact from a content_url or pdf_url returned by search "
            "or get_memory. In file mode this local proxy writes the artifact to the agent host "
            "cache and returns a real local_path."
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
        "name": "list_recent_changes",
        "description": "List recent source-document changes and optionally new or updated memories.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "since": {"type": "string", "description": "ISO date. Defaults to 7 days ago."},
                "source": {"type": "string", "description": "Filter to a source ID"},
                "include_memories": {"type": "boolean", "default": True},
            },
        },
    },
    {
        "name": "submit_agent_session_document",
        "description": (
            "Submit a client-generated agent session, task, or compaction-window summary as a "
            "low-authority generated source document."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "client": {"type": "string"},
                "session_id": {"type": "string"},
                "trigger": {"type": "string"},
                "workspace": {"type": "string"},
                "document_markdown": {"type": "string"},
                "repo": {"type": "string"},
                "branch": {"type": "string"},
                "commit_sha": {"type": "string"},
                "history_window_kind": {"type": "string", "default": "session"},
                "history_window_start": {"type": "string"},
                "history_window_end": {"type": "string"},
                "title": {"type": "string"},
                "metadata": {"type": "object"},
                "submitted_at": {"type": "string"},
                "process_now": {"type": "boolean", "default": True},
            },
            "required": ["client", "session_id", "trigger", "workspace", "document_markdown"],
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
        if method == "initialize":
            result = {
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            }
            return _rpc_result(request_id, result)
        if method == "notifications/initialized":
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
        return _http_json("POST", "/api/memories/search", args)
    if name == "get_memory":
        memory_id = str(args.get("memory_id") or "").strip()
        if not memory_id:
            return {"error": "memory_id is required"}
        return _http_json("GET", f"/api/memories/{quote(memory_id, safe='')}", None)
    if name == "list_recent_changes":
        query: dict[str, Any] = {}
        for key in ("since", "source", "include_memories"):
            if key in args and args[key] is not None:
                query[key] = args[key]
        suffix = f"?{urlencode(query)}" if query else ""
        return _http_json("GET", f"/api/recent-changes{suffix}", None)
    if name == "submit_agent_session_document":
        return _http_json("POST", "/api/agent-sessions/documents", args)
    if name == "get_resource":
        return _handle_get_resource(args)
    return {"error": f"Unknown tool: {name}"}


def _tool_result(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": json.dumps(payload, indent=2)}],
        "isError": False,
    }


def _api_base_url() -> str:
    return os.getenv("MEMFORGE_API_URL", DEFAULT_API_URL).rstrip("/")


def _api_headers(*, json_body: bool = False) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if json_body:
        headers["Content-Type"] = "application/json"
    token = os.getenv("MEMFORGE_API_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _http_json(method: str, path: str, body: dict[str, Any] | None) -> dict[str, Any]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = Request(
        f"{_api_base_url()}{path}",
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
        return {"error": "MemForge API unavailable", "api_url": _api_base_url(), "detail": str(exc)}


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

    target = _parse_resource_url(str(args.get("url") or "").strip(), _api_base_url())
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


def _parse_resource_url(url: str, api_base_url: str) -> ResourceTarget | None:
    parsed = urlparse(url)
    base = urlparse(api_base_url)
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
        return ResourceTarget(parts[2], "content", path, f"{api_base_url}{path}")
    if len(parts) == 4 and parts[:2] == ["api", "documents"] and parts[3] == "pdf":
        return ResourceTarget(parts[2], "pdf", path, f"{api_base_url}{path}")
    if len(parts) == 5 and parts[:2] == ["api", "documents"] and parts[3] == "artifacts":
        return ResourceTarget(parts[2], parts[4], path, f"{api_base_url}{path}")
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
        os.getenv("MEMFORGE_ARTIFACT_CACHE_DIR")
        or (Path.home() / ".memforge-agent" / "artifacts")
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
