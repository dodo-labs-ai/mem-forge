"""Shared helpers for service-owned local-adapter package manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from memforge.models import ContentItem


def package_manifest(config: dict[str, Any]) -> list[dict[str, Any]]:
    manifest = config.get("local_agent_package_manifest")
    if not isinstance(manifest, list):
        return []
    return [entry for entry in manifest if isinstance(entry, dict)]


def read_package_body(gene: Any, item: ContentItem, *, source_label: str) -> bytes:
    package_uri = item.extra.get("package_uri")
    if package_uri:
        document_store = getattr(gene, "_document_store", None)
        if document_store is not None:
            return document_store.read_artifact(str(package_uri))
        if item.extra.get("package_path"):
            return Path(str(item.extra["package_path"])).read_bytes()
        raise FileNotFoundError(f"document store is required for {source_label} package {item.item_id}")

    package_path = item.extra.get("package_path")
    if not package_path:
        raise FileNotFoundError(f"{source_label} package {item.item_id} has no package_uri or package_path")
    return Path(str(package_path)).read_bytes()
