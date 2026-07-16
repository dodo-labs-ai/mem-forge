"""Shared helpers for service-owned local-adapter package manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from memforge.models import ContentItem


def has_package_manifest(config: dict[str, Any]) -> bool:
    """Return whether the server supplied an authoritative package snapshot."""
    return isinstance(config.get("local_agent_package_manifest"), list)


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
            body = document_store.read_artifact(str(package_uri))
        elif item.extra.get("package_path"):
            body = Path(str(item.extra["package_path"])).read_bytes()
        else:
            raise FileNotFoundError(
                f"document store is required for {source_label} package {item.item_id}"
            )

        from memforge.local_agent.replay_adapter import get_local_source_replay_adapter

        source_type = str(gene.metadata().name)
        adapter = get_local_source_replay_adapter(source_type)
        package = adapter.validate(
            body,
            expected_doc_id=item.item_id,
            expected_version=item.version,
            expected_input_sha256=str(item.extra.get("input_sha256") or ""),
            expected_package_sha256=str(item.extra.get("package_sha256") or ""),
        )
        if adapter.derive_document_id(
            source_id=str(getattr(gene, "source_id", "")),
            package=package,
        ) != item.item_id:
            raise ValueError("source_lifecycle_local_replay_artifact_invalid")
        return body

    package_path = item.extra.get("package_path")
    if not package_path:
        raise FileNotFoundError(f"{source_label} package {item.item_id} has no package_uri or package_path")
    return Path(str(package_path)).read_bytes()
