"""Shared helpers for service-owned local-adapter package manifests."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from hashlib import sha256
from pathlib import Path
from typing import Any

from memforge.models import ContentItem
from memforge.source_artifacts import (
    SOURCE_ARTIFACT_STREAM_CHUNK_BYTES,
    RawSourceArtifact,
    SourceArtifactContractError,
    SourceArtifactDownload,
)


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


def source_artifacts_from_package(package: dict[str, Any]) -> tuple[RawSourceArtifact, ...]:
    """Reconstruct validated descriptors for service-owned binary inputs."""

    raw_artifacts = package.get("source_artifacts")
    if raw_artifacts is None:
        return ()
    if not isinstance(raw_artifacts, list):
        raise SourceArtifactContractError("local source package Artifacts must be a list")
    artifacts: list[RawSourceArtifact] = []
    for raw in raw_artifacts:
        if not isinstance(raw, dict):
            raise SourceArtifactContractError("local source package Artifact is invalid")
        uri = str(raw.get("uri") or "").strip()
        sha256 = str(raw.get("sha256") or "").strip()
        if not uri or len(sha256) != 64:
            raise SourceArtifactContractError("local source package Artifact bytes are not attested")
        artifacts.append(
            RawSourceArtifact(
                provider_key=str(raw.get("provider_key") or ""),
                parent_observation_type=str(raw.get("parent_observation_type") or ""),
                parent_provider_key=str(raw.get("parent_provider_key") or ""),
                provider_revision=str(raw.get("provider_revision") or ""),
                filename=str(raw.get("filename") or ""),
                media_type=str(raw.get("media_type") or ""),
                declared_size_bytes=raw.get("size_bytes"),
                locator={
                    "input_uri": uri,
                    "content_sha256": sha256,
                },
            )
        )
    return tuple(artifacts)


@asynccontextmanager
async def open_packaged_source_artifact(gene: Any, artifact: RawSourceArtifact):
    """Stream one service-owned local input through the shared Artifact path."""

    uri = str(artifact.locator.get("input_uri") or "").strip()
    if not uri:
        raise SourceArtifactContractError("local source package Artifact URI is missing")
    document_store = getattr(gene, "_document_store", None)
    if document_store is None:
        raise SourceArtifactContractError("document store is required for local source package Artifacts")
    expected_sha256 = str(artifact.locator.get("content_sha256") or "").strip().lower()
    if len(expected_sha256) != 64:
        raise SourceArtifactContractError("local source package Artifact byte hash is missing")
    context = document_store.open_artifact(uri)
    handle = await asyncio.to_thread(context.__enter__)

    async def chunks():
        digest = sha256()
        while True:
            chunk = await asyncio.to_thread(handle.read, SOURCE_ARTIFACT_STREAM_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            yield chunk
        if digest.hexdigest() != expected_sha256:
            raise SourceArtifactContractError("local source package Artifact byte hash does not match its attestation")

    try:
        yield SourceArtifactDownload(
            chunks=chunks(),
            media_type=artifact.media_type,
            content_length=artifact.declared_size_bytes,
        )
    finally:
        await asyncio.to_thread(context.__exit__, None, None, None)
