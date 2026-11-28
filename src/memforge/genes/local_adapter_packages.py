"""Stream an Artifact a device uploaded for a local-agent Source from the server's store."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from hashlib import sha256
from typing import Any

from memforge.source_artifacts import (
    SOURCE_ARTIFACT_STREAM_CHUNK_BYTES,
    RawSourceArtifact,
    SourceArtifactContractError,
    SourceArtifactDownload,
)


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
