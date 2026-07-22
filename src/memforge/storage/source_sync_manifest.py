"""Shared storage contract for bounded Source Sync manifest reuse."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from memforge.models import SourceSyncInput


MANIFEST_ATTESTATION_LOOKUP_CHUNK_SIZE = 200
"""Maximum exact manifest identities resolved by one relational query."""


@runtime_checkable
class SourceSyncManifestStore(Protocol):
    """Resolve reusable immutable inputs without scanning Source history."""

    async def find_source_sync_input_attestations(
        self,
        *,
        source_id: str,
        workspace_id: str,
        manifest_items: Sequence[tuple[str, str, str]],
    ) -> list[SourceSyncInput]: ...
