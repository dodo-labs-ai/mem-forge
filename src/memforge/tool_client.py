"""HTTP client for MemForge read-tool CLI commands."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlencode, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

DEFAULT_API_URL = "http://127.0.0.1:8765"
DEFAULT_TIMEOUT_SECONDS = 60.0


class NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


class ResourceTarget:
    def __init__(self, doc_id: str, kind: str, relative_url: str, request_url: str) -> None:
        self.doc_id = doc_id
        self.kind = kind
        self.relative_url = relative_url
        self.request_url = request_url


class ToolClient:
    """HTTP-backed implementation of MCP-aligned CLI read tools."""

    def __init__(
        self,
        *,
        api_url: str | None = None,
        api_token: str | None = None,
        workspace_id: str | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.api_url = (api_url or os.getenv("MEMFORGE_API_URL") or DEFAULT_API_URL).rstrip("/")
        self.api_token = api_token if api_token is not None else os.getenv("MEMFORGE_API_TOKEN")
        self.workspace_id = workspace_id if workspace_id is not None else os.getenv("MEMFORGE_WORKSPACE_ID")
        self.workspace_id = (self.workspace_id or "").strip()
        self.timeout_seconds = timeout_seconds

    def search(
        self,
        *,
        query: str = "",
        top_k: int = 10,
        memory_types: list[str] | tuple[str, ...] | None = None,
        time_range: dict[str, Any] | None = None,
        entities: list[str] | tuple[str, ...] | None = None,
        source_filter: dict[str, Any] | None = None,
        include_private: bool = False,
        include_superseded: bool = False,
        active_repo_identifier: str | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "top_k": top_k,
            "include_superseded": include_superseded,
        }
        if query:
            body["query"] = query
        if memory_types:
            body["memory_types"] = list(memory_types)
        if time_range is not None:
            body["time_range"] = time_range
        if entities:
            body["entities"] = list(entities)
        if source_filter:
            body["source_filter"] = source_filter
        if include_private:
            body["include_private"] = True
        if active_repo_identifier:
            body["active_repo_identifier"] = active_repo_identifier
        if status:
            body["status"] = status
        return self._http_json("POST", "/api/memories/search", body)

    def get_memory(self, memory_id: str) -> dict[str, Any]:
        memory_id = memory_id.strip()
        if not memory_id:
            return {"error": "memory_id is required"}
        return self._http_json("GET", f"/api/memories/{quote(memory_id, safe='')}?include_private=true", None)

    def start_source_sync(self, source_id: str, *, force_full_sync: bool = False) -> dict[str, Any]:
        source_id = source_id.strip()
        if not source_id:
            return {"error": "source_id is required"}
        return self._http_json(
            "POST",
            f"/api/sources/{quote(source_id, safe='')}/sync",
            {"force_full_sync": force_full_sync},
        )

    def create_memory(
        self,
        *,
        content: str,
        provenance: str,
        memory_type: str = "fact",
        tags: list[str] | tuple[str, ...] | None = None,
        confidence: float | None = None,
        client: str = "codex",
        repo_identifier: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "content": content,
            "memory_type": memory_type,
            "tags": list(tags or []),
            "client": client,
            "provenance": provenance,
        }
        if confidence is not None:
            body["confidence"] = confidence
        if repo_identifier:
            body["repo_identifier"] = repo_identifier
        if idempotency_key:
            body["idempotency_key"] = idempotency_key
        return self._http_json("POST", "/api/memories/create", body)

    def retire_memory(
        self,
        memory_id: str,
        *,
        reason: str,
        expected_content_hash: str,
    ) -> dict[str, Any]:
        memory_id = memory_id.strip()
        if not memory_id:
            return {"error": "memory_id is required"}
        return self._http_json(
            "POST",
            f"/api/memories/{quote(memory_id, safe='')}/retire",
            {
                "reason": reason,
                "expected_content_hash": expected_content_hash,
            },
        )

    def replace_memory(
        self,
        memory_id: str,
        *,
        replacement_content: str,
        provenance: str,
        reason: str,
        expected_content_hash: str,
        replacement_kind: str = "supersession",
    ) -> dict[str, Any]:
        memory_id = memory_id.strip()
        if not memory_id:
            return {"error": "memory_id is required"}
        body = {
            "replacement_content": replacement_content,
            "provenance": provenance,
            "reason": reason,
            "expected_content_hash": expected_content_hash,
            "replacement_kind": replacement_kind,
        }
        return self._http_json(
            "POST",
            f"/api/memories/{quote(memory_id, safe='')}/replace",
            body,
        )

    def push_local_markdown_document(
        self,
        *,
        source_id: str,
        vault_id: str,
        relative_path: str,
        markdown_body: str,
        content_type: str = "text/markdown",
        title: str | None = None,
        raw_hash: str | None = None,
        submitted_by: str | None = None,
        submitted_at: str | None = None,
        process_now: bool = False,
    ) -> dict[str, Any]:
        """Push one file's raw text into a configured local repository source.

        ``content_type`` tells the service how to convert ``markdown_body``
        (the raw file text) during sync: Markdown/text pass through, HTML and
        JSON are converted server-side.
        """
        source_id = source_id.strip()
        if not source_id:
            return {"error": "source_id is required"}
        body: dict[str, Any] = {
            "vault_id": vault_id,
            "relative_path": relative_path,
            "markdown_body": markdown_body,
            "content_type": content_type,
            "process_now": process_now,
        }
        if title is not None:
            body["title"] = title
        if raw_hash is not None:
            body["raw_hash"] = raw_hash
        if submitted_by is not None:
            body["submitted_by"] = submitted_by
        if submitted_at is not None:
            body["submitted_at"] = submitted_at
        return self._http_json(
            "POST",
            f"/api/sources/{quote(source_id, safe='')}/adapter/documents",
            body,
        )

    def push_github_repo_document(
        self,
        *,
        source_id: str,
        repo_url: str,
        repo_ref: str,
        relative_path: str,
        markdown_body: str,
        content_type: str = "text/markdown",
        title: str | None = None,
        raw_hash: str | None = None,
        blob_sha: str | None = None,
        submitted_by: str | None = None,
        submitted_at: str | None = None,
        process_now: bool = False,
    ) -> dict[str, Any]:
        """Push one GitHub repository file into a configured github_repo source."""
        source_id = source_id.strip()
        if not source_id:
            return {"error": "source_id is required"}
        body: dict[str, Any] = {
            "repo_url": repo_url,
            "repo_ref": repo_ref,
            "relative_path": relative_path,
            "markdown_body": markdown_body,
            "content_type": content_type,
            "process_now": process_now,
        }
        if title is not None:
            body["title"] = title
        if raw_hash is not None:
            body["raw_hash"] = raw_hash
        if blob_sha is not None:
            body["blob_sha"] = blob_sha
        if submitted_by is not None:
            body["submitted_by"] = submitted_by
        if submitted_at is not None:
            body["submitted_at"] = submitted_at
        return self._http_json(
            "POST",
            f"/api/sources/{quote(source_id, safe='')}/adapter/documents",
            body,
        )

    def push_jira_document(
        self,
        *,
        source_id: str,
        base_url: str,
        issue_key: str,
        source_url: str,
        markdown_body: str,
        title: str | None = None,
        raw_hash: str | None = None,
        source_semantics: dict[str, Any] | None = None,
        submitted_by: str | None = None,
        submitted_at: str | None = None,
        process_now: bool = False,
    ) -> dict[str, Any]:
        """Push one locally fetched Jira issue into a configured Jira source."""
        source_id = source_id.strip()
        if not source_id:
            return {"error": "source_id is required"}
        body: dict[str, Any] = {
            "base_url": base_url,
            "issue_key": issue_key,
            "source_url": source_url,
            "markdown_body": markdown_body,
            "content_type": "text/markdown",
            "process_now": process_now,
        }
        if title is not None:
            body["title"] = title
        if raw_hash is not None:
            body["raw_hash"] = raw_hash
        if source_semantics is not None:
            body["source_semantics"] = source_semantics
        if submitted_by is not None:
            body["submitted_by"] = submitted_by
        if submitted_at is not None:
            body["submitted_at"] = submitted_at
        return self._http_json(
            "POST",
            f"/api/sources/{quote(source_id, safe='')}/adapter/documents",
            body,
        )

    def list_sources(self) -> dict[str, Any]:
        """List configured sources. Returns the API ``{"data": [...]}`` envelope."""
        return self._http_json("GET", "/api/sources", None)

    def list_searchable_sources(self) -> dict[str, Any]:
        """List search-eligible sources for MCP/source-id discovery."""
        return self._http_json("GET", "/api/sources/searchable", None)

    def list_memory_reviews(
        self,
        *,
        status: str = "open",
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        query = urlencode({"status": status, "limit": limit, "offset": offset})
        return self._http_json("GET", f"/api/memory-reviews?{query}", None)

    def get_memory_review(self, review_id: str) -> dict[str, Any]:
        review_id = review_id.strip()
        if not review_id:
            return {"error": "review_id is required"}
        return self._http_json("GET", f"/api/memory-reviews/{quote(review_id, safe='')}", None)

    def resolve_memory_review(
        self,
        review_id: str,
        *,
        decision: str,
        note: str | None = None,
        reviewer: str | None = None,
    ) -> dict[str, Any]:
        review_id = review_id.strip()
        if not review_id:
            return {"error": "review_id is required"}
        body: dict[str, Any] = {}
        if note is not None:
            body["note"] = note
        if reviewer is not None:
            body["reviewer"] = reviewer
        return self._http_json("POST", f"/api/memory-reviews/{quote(review_id, safe='')}/{decision}", body)

    def create_source(self, *, source_type: str, name: str, config: dict[str, Any]) -> dict[str, Any]:
        """Create a source (gene instance) of ``source_type`` with the given config."""
        return self._http_json(
            "POST",
            "/api/sources",
            {"type": source_type, "name": name, "config": config},
        )

    def get_source_schedule(self, source_id: str) -> dict[str, Any]:
        source_id = source_id.strip()
        if not source_id:
            return {"error": "source_id is required"}
        return self._http_json("GET", f"/api/sources/{quote(source_id, safe='')}/schedule", None)

    def update_source_schedule(
        self,
        *,
        source_id: str,
        enabled: bool,
        interval_minutes: int,
    ) -> dict[str, Any]:
        source_id = source_id.strip()
        if not source_id:
            return {"error": "source_id is required"}
        return self._http_json(
            "PUT",
            f"/api/sources/{quote(source_id, safe='')}/schedule",
            {"enabled": enabled, "interval_minutes": interval_minutes},
        )

    def lease_local_agent_jobs(
        self,
        *,
        limit: int = 5,
        lease_seconds: int = 3600,
    ) -> dict[str, Any]:
        return self._http_json(
            "POST",
            "/api/cloud/local-agent/jobs/lease",
            {"limit": limit, "lease_seconds": lease_seconds},
        )

    def complete_local_agent_job(
        self,
        job_id: str,
        *,
        attempt_count: int,
        status: str,
        result: dict[str, Any],
        error: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"status": status, "attempt_count": attempt_count, "result": result}
        if error is not None:
            body["error"] = error
        return self._http_json(
            "POST",
            f"/api/cloud/local-agent/jobs/{quote(job_id, safe='')}/complete",
            body,
        )

    def get_jira_session(self, base_url: str) -> dict[str, Any]:
        return self._http_json("GET", f"/api/auth/jira-session?base_url={quote(base_url, safe='')}", None)

    def list_jira_origins(self) -> dict[str, Any]:
        return self._http_json("GET", "/api/auth/jira-origins", None)

    def upload_jira_session(
        self, *, base_url: str, cookie_header: str, browser: str | None = None,
        confirm_principal_change: bool = False,
    ) -> dict[str, Any]:
        return self._http_json(
            "POST",
            "/api/auth/jira-session",
            {
                "base_url": base_url,
                "cookie_header": cookie_header,
                "browser": browser,
                "confirm_principal_change": confirm_principal_change,
            },
        )

    def forget_jira_session(self, base_url: str) -> dict[str, Any]:
        return self._http_json("DELETE", f"/api/auth/jira-session?base_url={quote(base_url, safe='')}", None)

    def mark_jira_session_expired(self, *, base_url: str, error: str) -> dict[str, Any]:
        return self._http_json("POST", "/api/auth/jira-session/expire", {"base_url": base_url, "error": error})

    def health(self) -> dict[str, Any]:
        return self._http_json("GET", "/api/health", None)

    def get_resource(
        self,
        *,
        url: str,
        mode: str = "text",
        max_chars: int = 120_000,
        max_bytes: int = 2_000_000,
    ) -> dict[str, Any]:
        mode = str(mode or "text").strip().lower()
        if mode not in {"text", "file", "base64"}:
            return {"error": f"unsupported mode: {mode}", "supported_modes": ["text", "file", "base64"]}

        parsed_max_bytes = _positive_int(max_bytes, "max_bytes")
        if isinstance(parsed_max_bytes, dict):
            return parsed_max_bytes
        parsed_max_chars = _positive_int(max_chars, "max_chars")
        if isinstance(parsed_max_chars, dict):
            return parsed_max_chars

        target = _parse_resource_url(
            str(url or "").strip(),
            self.api_url,
            self._request_url,
        )
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
                return self._fetch_resource_file(target)
            return self._fetch_resource_inline(
                target,
                mode=mode,
                max_bytes=parsed_max_bytes,
                max_chars=parsed_max_chars,
            )
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

    def _headers(self, *, json_body: bool = False) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if json_body:
            headers["Content-Type"] = "application/json"
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        return headers

    def _http_json(self, method: str, path: str, body: dict[str, Any] | None) -> dict[str, Any]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = Request(
            self._request_url(path),
            data=data,
            headers=self._headers(json_body=body is not None),
            method=method,
        )
        try:
            with build_opener(NoRedirectHandler).open(request, timeout=self.timeout_seconds) as response:
                raw = response.read()
                if not raw:
                    return {}
                return json.loads(raw.decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            return {"error": "MemForge API request failed", "status_code": exc.code, "detail": detail}
        except (OSError, URLError, json.JSONDecodeError) as exc:
            return {"error": "MemForge API unavailable", "api_url": self.api_url, "detail": str(exc)}

    def _fetch_resource_inline(
        self,
        target: ResourceTarget,
        *,
        mode: str,
        max_bytes: int,
        max_chars: int,
    ) -> dict[str, Any]:
        request = Request(target.request_url, headers=self._headers(), method="GET")
        with build_opener(NoRedirectHandler).open(request, timeout=self.timeout_seconds) as response:
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

    def _fetch_resource_file(self, target: ResourceTarget) -> dict[str, Any]:
        request = Request(target.request_url, headers=self._headers(), method="GET")
        tmp_path: Path | None = None
        try:
            with build_opener(NoRedirectHandler).open(request, timeout=self.timeout_seconds) as response:
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

    def _request_url(self, path: str) -> str:
        if not self.workspace_id or not path.startswith("/api"):
            return f"{self.api_url}{path}"
        if path.startswith(("/api/cloud/", "/api/auth/")):
            return f"{self.api_url}{path}"
        quoted_workspace = quote(self.workspace_id, safe="")
        if path == "/api":
            return f"{self.api_url}/api/workspaces/{quoted_workspace}/api"
        if path.startswith("/api/"):
            return f"{self.api_url}/api/workspaces/{quoted_workspace}/api/{path[len('/api/'):]}"
        return f"{self.api_url}{path}"


def _parse_resource_url(
    url: str,
    api_base_url: str,
    request_url_for_path,
) -> ResourceTarget | None:
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
        return ResourceTarget(parts[2], "content", path, request_url_for_path(path))
    if len(parts) == 4 and parts[:2] == ["api", "documents"] and parts[3] == "pdf":
        return ResourceTarget(parts[2], "pdf", path, request_url_for_path(path))
    if len(parts) == 5 and parts[:2] == ["api", "documents"] and parts[3] == "artifacts":
        return ResourceTarget(parts[2], parts[4], path, request_url_for_path(path))
    return None


def _positive_int(raw_value: object, name: str) -> int | dict[str, Any]:
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return {"error": f"invalid {name}", "detail": f"{name} must be a positive integer."}
    if value <= 0:
        return {"error": f"invalid {name}", "detail": f"{name} must be a positive integer."}
    return value


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
