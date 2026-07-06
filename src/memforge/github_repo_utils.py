"""Shared helpers for GitHub repository sources."""

from __future__ import annotations

import hashlib
import base64
import binascii
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
    return [normalize_github_relative_path(path) for path in list_config(config.get("include_paths"))]


def github_include_extensions(config: dict) -> set[str]:
    value = config.get("include_extensions")
    if value is None:
        values = DEFAULT_INCLUDE_EXTENSION_LIST
    else:
        values = list_config(value)
    return {item.lower().lstrip(".") for item in values if item.strip()}


def github_path_in_scope(relative_path: str, include_paths: list[str]) -> bool:
    try:
        path = normalize_github_relative_path(relative_path)
    except ValueError:
        return False
    if not include_paths:
        return True
    return any(path == scope or path.startswith(scope.rstrip("/") + "/") for scope in include_paths)


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
    text = str(content or "")
    if encoding not in {None, "base64"} or (not text and _int_value(size) > 0):
        raise ValueError(f"GitHub contents API did not return base64 content for {label}")
    try:
        return base64.b64decode(text.replace("\n", ""), validate=True)
    except binascii.Error as exc:
        raise ValueError(f"GitHub contents API returned invalid base64 content for {label}") from exc


def _int_value(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
