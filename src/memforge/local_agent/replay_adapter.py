"""Registered package/replay contracts for local-agent source types.

The lifecycle orchestration layer depends only on this provider-neutral
interface.  A new local-agent source type must register an adapter instead of
falling through to another provider's identity or validation rules.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from memforge.local_agent.document_identity import build_local_markdown_doc_id
from memforge.local_agent.source_contract import (
    local_agent_semantic_input_sha256,
    local_file_package_semantic_hash,
)
from memforge.source_time import SOURCE_UPDATED_AT_KEY, source_time_iso
from memforge.source_types import registered_source_types, source_type_descriptor


INVALID_REPLAY_ARTIFACT = "source_lifecycle_local_replay_artifact_invalid"
MISSING_REPLAY_ATTESTATION = "source_lifecycle_local_replay_attestation_required"


class LocalSourceReplayAdapter(ABC):
    """Provider contract for validating and identifying one durable package."""

    source_type: str
    package_kind: str
    # A file package carries the file's source time beside its content.
    package_source_time: bool = False

    def validate(
        self,
        body: bytes,
        *,
        expected_doc_id: str,
        expected_version: str,
        expected_input_sha256: str,
        expected_package_sha256: str,
    ) -> Mapping[str, Any]:
        package_hash = str(expected_package_sha256 or "").strip()
        input_hash = str(expected_input_sha256 or "").strip()
        if not package_hash or not input_hash:
            raise ValueError(MISSING_REPLAY_ATTESTATION)
        if hashlib.sha256(body).hexdigest() != package_hash:
            raise ValueError(INVALID_REPLAY_ARTIFACT)
        try:
            package = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(INVALID_REPLAY_ARTIFACT) from exc
        if not isinstance(package, Mapping):
            raise ValueError(INVALID_REPLAY_ARTIFACT)
        package_version = str(package.get("version") or package.get("revision_hash") or "")
        if (
            str(package.get("package_kind") or "") != self.package_kind
            or str(package.get("doc_id") or "") != expected_doc_id
            or package_version != expected_version
        ):
            raise ValueError(INVALID_REPLAY_ARTIFACT)
        semantic_hash = self._validate_semantics(package, package_version)
        if local_agent_semantic_input_sha256(expected_doc_id, semantic_hash) != input_hash:
            raise ValueError(INVALID_REPLAY_ARTIFACT)
        return package

    @abstractmethod
    def derive_document_id(
        self,
        *,
        source_id: str,
        package: Mapping[str, Any],
    ) -> str:
        """Recompute canonical document identity from provider locators."""

    @abstractmethod
    def _validate_semantics(
        self,
        package: Mapping[str, Any],
        package_version: str,
    ) -> str:
        """Validate provider payload and return its canonical semantic hash."""


class _MarkdownPackageAdapter(LocalSourceReplayAdapter):
    package_source_time = True

    def _validate_semantics(
        self,
        package: Mapping[str, Any],
        package_version: str,
    ) -> str:
        content_hash = self._content_hash(package)
        if not self._version_matches(package, package_version, content_hash):
            raise ValueError(INVALID_REPLAY_ARTIFACT)
        try:
            source_time = source_time_iso(package.get(SOURCE_UPDATED_AT_KEY))
        except ValueError as exc:
            raise ValueError(INVALID_REPLAY_ARTIFACT) from exc
        return local_file_package_semantic_hash(content_hash, source_time)

    def _content_hash(self, package: Mapping[str, Any]) -> str:
        markdown = package.get("markdown")
        if not isinstance(markdown, str):
            raise ValueError(INVALID_REPLAY_ARTIFACT)
        return hashlib.sha256(markdown.encode("utf-8")).hexdigest()

    @abstractmethod
    def _version_matches(
        self,
        package: Mapping[str, Any],
        package_version: str,
        content_hash: str,
    ) -> bool:
        pass


class LocalMarkdownReplayAdapter(_MarkdownPackageAdapter):
    source_type = "local_markdown"
    package_kind = "local_markdown_document"

    def derive_document_id(
        self,
        *,
        source_id: str,
        package: Mapping[str, Any],
    ) -> str:
        return build_local_markdown_doc_id(
            source_id=source_id,
            vault_id=str(package.get("vault_id") or ""),
            relative_path=str(package.get("relative_path") or ""),
        )

    def _version_matches(
        self,
        package: Mapping[str, Any],
        package_version: str,
        content_hash: str,
    ) -> bool:
        return package_version == content_hash


def get_local_source_replay_adapter(source_type: str) -> LocalSourceReplayAdapter:
    """The replay adapter a source type's descriptor names for its local-agent packages."""

    normalized = str(source_type or "").strip().lower()
    descriptor = source_type_descriptor(normalized)
    collection = descriptor.local_agent if descriptor is not None else None
    adapter = collection.replay_adapter_instance() if collection is not None else None
    if adapter is None:
        raise ValueError(f"local source replay adapter is not registered: {normalized}")
    return adapter


def registered_local_source_types() -> frozenset[str]:
    return frozenset(
        descriptor.source_type
        for descriptor in registered_source_types()
        if descriptor.local_agent is not None and descriptor.local_agent.replay_adapter is not None
    )
