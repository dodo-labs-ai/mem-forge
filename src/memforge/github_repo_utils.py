"""Shared helpers for GitHub repository sources."""

from __future__ import annotations

import hashlib
import base64
import binascii
import mimetypes
import re
from io import BytesIO
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from memforge.models import slugify

DEFAULT_INCLUDE_EXTENSIONS = "md, markdown, txt, adoc, rst"
DEFAULT_INCLUDE_EXTENSION_LIST = ["md", "markdown", "txt", "adoc", "rst"]

# Provider envelope, not a promise that arbitrary documents fit extraction memory.
GITHUB_BLOB_MAX_BYTES = 100 * 1024 * 1024


class GitHubBlobBuffer:
    """Bound one raw transfer and verify its immutable identity before use."""

    def __init__(
        self, *, sha: str, size: object, label: str, content_length: object = None,
    ) -> None:
        if not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise ValueError(f"GitHub blob identity is invalid for {label}")
        self.sha = sha
        self.label = label
        self.lengths: list[int] = []
        for value in (size, content_length):
            if value is None:
                continue
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"GitHub blob size is invalid for {label}")
            if value > GITHUB_BLOB_MAX_BYTES:
                raise ValueError(f"GitHub blob exceeds provider byte limit for {label}")
            self.lengths.append(value)
        if len(set(self.lengths)) > 1:
            raise ValueError(f"GitHub blob size mismatch for {label}")
        self.limit = min([GITHUB_BLOB_MAX_BYTES, *self.lengths])
        self.buffer = BytesIO()

    def write(self, chunk: bytes) -> None:
        if self.buffer.tell() + len(chunk) > self.limit:
            raise ValueError(f"GitHub blob exceeds declared size or provider byte limit for {self.label}")
        self.buffer.write(chunk)

    def finish(self) -> bytes:
        size = self.buffer.tell()
        if any(size != expected for expected in self.lengths):
            raise ValueError(f"GitHub blob size mismatch for {self.label}")
        digest = hashlib.sha1(f"blob {size}\0".encode("ascii"), usedforsecurity=False)
        with self.buffer.getbuffer() as view:
            digest.update(view)
        if digest.hexdigest() != self.sha:
            raise ValueError(f"GitHub blob hash mismatch for {self.label}")
        return self.buffer.getvalue()


def decode_github_text(raw: bytes, *, label: str) -> str:
    """Decode supported source text without altering the provider's Evidence."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"GitHub file {label} is invalid UTF-8 at byte {exc.start}; "
            "correct the source file encoding and commit a new revision."
        ) from exc


def validate_github_file_mode(mode: object, *, label: str) -> None:
    if mode not in ("100644", "100755"):
        raise ValueError(f"GitHub selected file {label} has unsupported file mode {mode!r}")


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
    if extension == "png":
        return "image/png"
    if extension in {"jpg", "jpeg"}:
        return "image/jpeg"
    if extension == "gif":
        return "image/gif"
    if extension == "webp":
        return "image/webp"
    if extension == "pdf":
        return "application/pdf"
    guessed_media_type = mimetypes.guess_type(relative_path)[0]
    if guessed_media_type and guessed_media_type.startswith("image/"):
        return guessed_media_type
    return "text/plain"


def github_content_type_is_binary(content_type: str) -> bool:
    """Return whether a selected repository file requires Artifact handling."""

    normalized = str(content_type or "").strip().lower()
    return normalized == "application/pdf" or normalized.startswith("image/")


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
