"""Shared helpers for GitHub repository sources."""

from __future__ import annotations

import hashlib
import base64
import binascii
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from memforge.models import slugify

DEFAULT_INCLUDE_EXTENSIONS = "md, markdown, txt, adoc, rst"
DEFAULT_INCLUDE_EXTENSION_LIST = ["md", "markdown", "txt", "adoc", "rst"]


def build_github_repo_doc_id(*, source_id: str, repo_url: str, repo_ref: str, relative_path: str) -> str:
    """Stable document id for one file in a configured GitHub repository source."""
    identity = "|".join([source_id.strip(), repo_url.strip(), repo_ref.strip(), relative_path.strip()])
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return "-".join([
        "github-repo",
        slugify(relative_path)[:50] or "doc",
        digest,
    ])


def list_config(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def parse_github_repo_url(repo_url: str) -> dict[str, str]:
    parts = urlsplit(str(repo_url or "").strip())
    if parts.scheme.lower() != "https" or not parts.hostname:
        raise ValueError("repo_url must be an https GitHub repository URL")
    path_parts = [part for part in parts.path.split("/") if part]
    if len(path_parts) < 2:
        raise ValueError("repo_url must include owner and repository")
    owner = path_parts[0]
    repo = path_parts[1][:-4] if path_parts[1].endswith(".git") else path_parts[1]
    host = parts.hostname.lower()
    if parts.port:
        host = f"{host}:{parts.port}"
    origin = urlunsplit(("https", host, "", "", ""))
    normalized_url = urlunsplit(("https", host, f"/{owner}/{repo}", "", ""))
    return {"repo_url": normalized_url, "origin": origin, "host": host, "owner": owner, "repo": repo}


def normalize_github_relative_path(value: str) -> str:
    candidate = str(value or "").strip().replace("\\", "/").lstrip("/")
    if not candidate:
        raise ValueError("relative_path is required")
    parts = [part for part in candidate.split("/") if part not in ("", ".")]
    if any(part == ".." for part in parts):
        raise ValueError("relative_path must not contain '..' segments")
    normalized = "/".join(parts)
    if not normalized:
        raise ValueError("relative_path is required")
    return normalized + "/" if candidate.endswith("/") and not normalized.endswith("/") else normalized


def github_include_paths(config: dict) -> list[str]:
    return normalize_github_scope_paths(list_config(config.get("include_paths")))


def github_exclude_paths(config: dict) -> list[str]:
    return normalize_github_scope_paths(list_config(config.get("exclude_paths")))


def normalize_github_scope_paths(paths: list[str]) -> list[str]:
    """Canonicalize repository paths and remove selections covered by an ancestor."""
    normalized = sorted({normalize_github_relative_path(path).rstrip("/") for path in paths})
    collapsed: list[str] = []
    for path in normalized:
        if any(path == ancestor or path.startswith(ancestor + "/") for ancestor in collapsed):
            continue
        collapsed.append(path)
    return collapsed


def github_include_extensions(config: dict) -> set[str]:
    value = config.get("include_extensions")
    if value is None:
        values = DEFAULT_INCLUDE_EXTENSION_LIST
    else:
        values = list_config(value)
    return {item.lower().lstrip(".") for item in values if item.strip()}


def github_path_in_scope(
    relative_path: str,
    include_paths: list[str],
    exclude_paths: list[str],
) -> bool:
    try:
        path = normalize_github_relative_path(relative_path)
    except ValueError:
        return False
    included = not include_paths or any(
        path == scope or path.startswith(scope.rstrip("/") + "/")
        for scope in include_paths
    )
    excluded = any(
        path == scope or path.startswith(scope.rstrip("/") + "/")
        for scope in exclude_paths
    )
    return included and not excluded


def github_extension(relative_path: str) -> str:
    name = relative_path.rsplit("/", 1)[-1]
    return name.rsplit(".", 1)[-1].lower() if "." in name else ""


def github_extension_allowed(relative_path: str, extensions: set[str]) -> bool:
    if not extensions:
        return True
    extension = github_extension(relative_path)
    return bool(extension and extension in extensions)


def github_content_type(relative_path: str) -> str:
    extension = github_extension(relative_path)
    if extension in {"md", "markdown"}:
        return "text/markdown"
    if extension in {"html", "htm"}:
        return "text/html"
    if extension == "json":
        return "application/json"
    return "text/plain"


def decode_github_base64_content(*, content: object, encoding: object, size: object, label: str) -> bytes:
    if not isinstance(content, str) or encoding != "base64":
        raise ValueError(f"GitHub contents API did not return base64 content for {label}")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise ValueError(f"GitHub contents API did not return a valid size for {label}")
    text = content
    try:
        decoded = base64.b64decode(text.replace("\n", ""), validate=True)
    except binascii.Error as exc:
        raise ValueError(f"GitHub contents API returned invalid base64 content for {label}") from exc
    if len(decoded) != size:
        raise ValueError(f"GitHub contents API content size mismatch for {label}")
    return decoded


def validate_github_tree_payload(payload: object, *, label: str) -> list[dict[str, Any]]:
    """Return one complete, stable Git tree or fail closed."""

    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} tree response must be an object")
    if payload.get("truncated") is not False:
        raise ValueError(f"{label} tree response did not attest truncated=false")
    tree = payload.get("tree")
    if not isinstance(tree, list):
        raise ValueError(f"{label} tree response is missing a tree list")
    result: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for entry in tree:
        if not isinstance(entry, Mapping):
            raise ValueError(f"{label} tree response contains an invalid entry")
        entry_type = str(entry.get("type") or "").strip()
        if entry_type not in {"blob", "tree", "commit"}:
            raise ValueError(f"{label} tree entry has an invalid type")
        raw_path = str(entry.get("path") or "").strip()
        try:
            canonical_path = normalize_github_relative_path(raw_path).rstrip("/")
        except ValueError as exc:
            raise ValueError(f"{label} tree entry has an invalid path") from exc
        object_sha = str(entry.get("sha") or "").strip()
        if not object_sha:
            raise ValueError(f"{label} tree entry is missing an object sha")
        if canonical_path in seen_paths:
            raise ValueError(f"{label} tree response contains duplicate path {canonical_path!r}")
        seen_paths.add(canonical_path)
        result.append(dict(entry))
    return result


def decode_github_contents_payload(
    payload: object,
    *,
    expected_sha: str,
    label: str,
) -> bytes:
    """Decode a Contents response bound to the blob discovered in the tree."""

    if not isinstance(payload, Mapping):
        raise ValueError(f"GitHub contents API returned an invalid object for {label}")
    actual_sha = str(payload.get("sha") or "").strip()
    if not expected_sha or actual_sha != expected_sha:
        raise ValueError(f"GitHub contents API blob identity mismatch for {label}")
    return decode_github_base64_content(
        content=payload.get("content"),
        encoding=payload.get("encoding"),
        size=payload.get("size"),
        label=label,
    )
