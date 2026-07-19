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

from memforge.github_repo_utils import build_github_repo_doc_id
from memforge.local_agent.document_identity import (
    build_jira_doc_id,
    build_local_markdown_doc_id,
    build_teams_doc_id,
)
from memforge.local_agent.source_contract import (
    TEAMS_TOMBSTONE_REASONS,
    local_agent_semantic_input_sha256,
)


INVALID_REPLAY_ARTIFACT = "source_lifecycle_local_replay_artifact_invalid"
MISSING_REPLAY_ATTESTATION = "source_lifecycle_local_replay_attestation_required"


def _semantic_json_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(value),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


class LocalSourceReplayAdapter(ABC):
    """Provider contract for validating and identifying one durable package."""

    source_type: str
    package_kind: str
    authoritative_collection: bool = False

    def rebaseline_snapshot_is_authoritative(
        self,
        *,
        force_full_sync: bool,
        input_snapshot_id: str | None,
    ) -> bool:
        """Return whether one immutable attempt defines the replay corpus."""

        del force_full_sync
        return bool(self.authoritative_collection and input_snapshot_id)

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
    def _validate_semantics(
        self,
        package: Mapping[str, Any],
        package_version: str,
    ) -> str:
        markdown = package.get("markdown")
        if not isinstance(markdown, str):
            raise ValueError(INVALID_REPLAY_ARTIFACT)
        semantic_hash = hashlib.sha256(markdown.encode("utf-8")).hexdigest()
        if not self._version_matches(package, package_version, semantic_hash):
            raise ValueError(INVALID_REPLAY_ARTIFACT)
        return semantic_hash

    @abstractmethod
    def _version_matches(
        self,
        package: Mapping[str, Any],
        package_version: str,
        semantic_hash: str,
    ) -> bool:
        pass


class GitHubRepoReplayAdapter(_MarkdownPackageAdapter):
    source_type = "github_repo"
    package_kind = "github_repo_document"
    authoritative_collection = True

    def derive_document_id(
        self,
        *,
        source_id: str,
        package: Mapping[str, Any],
    ) -> str:
        return build_github_repo_doc_id(
            source_id=source_id,
            repo_url=str(package.get("repo_url") or ""),
            repo_ref=str(package.get("repo_ref") or ""),
            relative_path=str(package.get("relative_path") or ""),
        )

    def _version_matches(
        self,
        package: Mapping[str, Any],
        package_version: str,
        semantic_hash: str,
    ) -> bool:
        return package_version == str(package.get("blob_sha") or package.get("raw_hash") or semantic_hash).strip()


class LocalMarkdownReplayAdapter(_MarkdownPackageAdapter):
    source_type = "local_markdown"
    package_kind = "local_markdown_document"
    authoritative_collection = True

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
        semantic_hash: str,
    ) -> bool:
        return package_version == semantic_hash


class JiraReplayAdapter(LocalSourceReplayAdapter):
    source_type = "jira"
    package_kind = "jira_document"
    authoritative_collection = True

    def derive_document_id(
        self,
        *,
        source_id: str,
        package: Mapping[str, Any],
    ) -> str:
        return build_jira_doc_id(
            source_id=source_id,
            issue_key=str(package.get("issue_key") or ""),
        )

    def _validate_semantics(
        self,
        package: Mapping[str, Any],
        package_version: str,
    ) -> str:
        raw_payload = package.get("raw_payload")
        if not isinstance(raw_payload, Mapping) or not raw_payload:
            raise ValueError(INVALID_REPLAY_ARTIFACT)
        from memforge.local_agent.jira_contract import validate_jira_observation_identities

        try:
            validate_jira_observation_identities(raw_payload)
        except ValueError as exc:
            raise ValueError(INVALID_REPLAY_ARTIFACT) from exc
        issue_key = str(package.get("issue_key") or "").strip().upper()
        semantic_hash = _semantic_json_hash(raw_payload)
        if (
            not issue_key
            or str(raw_payload.get("key") or "").strip().upper() != issue_key
            or not isinstance(raw_payload.get("fields"), Mapping)
            or not str(package.get("raw_hash") or "").strip()
            or str(package.get("semantic_hash") or "").strip() != semantic_hash
            or package_version != semantic_hash
        ):
            raise ValueError(INVALID_REPLAY_ARTIFACT)
        return semantic_hash


class TeamsReplayAdapter(LocalSourceReplayAdapter):
    source_type = "teams"
    package_kind = "teams_window_document"

    def rebaseline_snapshot_is_authoritative(
        self,
        *,
        force_full_sync: bool,
        input_snapshot_id: str | None,
    ) -> bool:
        """Use a force-full attempt for cutover without changing normal sync."""

        return bool(force_full_sync and input_snapshot_id)

    def derive_document_id(
        self,
        *,
        source_id: str,
        package: Mapping[str, Any],
    ) -> str:
        raw_payload = package.get("raw_payload")
        if not isinstance(raw_payload, Mapping):
            raise ValueError(INVALID_REPLAY_ARTIFACT)
        from memforge.local_agent.teams_contract import (
            TeamsMessageEvidenceError,
            validate_teams_window_payload,
        )

        try:
            validate_teams_window_payload(
                raw_payload,
                conversation_id=str(package.get("conversation_id") or "").strip(),
                window_id=str(package.get("window_id") or "").strip(),
                source_id=source_id,
                root_message_id=str(package.get("root_message_id") or "").strip(),
                window_type=str(package.get("window_type") or "").strip(),
                tombstone_reasons=TEAMS_TOMBSTONE_REASONS,
            )
        except TeamsMessageEvidenceError as exc:
            raise ValueError(INVALID_REPLAY_ARTIFACT) from exc
        return build_teams_doc_id(
            source_id=source_id,
            window_id=str(package.get("window_id") or ""),
        )

    def _validate_semantics(
        self,
        package: Mapping[str, Any],
        package_version: str,
    ) -> str:
        raw_payload = package.get("raw_payload")
        if not isinstance(raw_payload, Mapping) or not raw_payload:
            raise ValueError(INVALID_REPLAY_ARTIFACT)
        from memforge.local_agent.teams_contract import (
            TeamsMessageEvidenceError,
            validate_teams_window_payload,
        )

        try:
            validate_teams_window_payload(
                raw_payload,
                conversation_id=str(package.get("conversation_id") or "").strip(),
                window_id=str(package.get("window_id") or "").strip(),
                source_id=None,
                root_message_id=str(package.get("root_message_id") or "").strip(),
                window_type=str(package.get("window_type") or "").strip(),
                tombstone_reasons=TEAMS_TOMBSTONE_REASONS,
            )
        except TeamsMessageEvidenceError as exc:
            raise ValueError(INVALID_REPLAY_ARTIFACT) from exc
        semantic_hash = _semantic_json_hash(raw_payload)
        if (
            not str(package.get("conversation_id") or "").strip()
            or not str(package.get("window_id") or "").strip()
            or str(package.get("revision_hash") or "").strip() != package_version
            or not str(package.get("raw_hash") or "").strip()
            or str(package.get("semantic_hash") or "").strip() != semantic_hash
        ):
            raise ValueError(INVALID_REPLAY_ARTIFACT)
        return semantic_hash


_ADAPTERS: dict[str, LocalSourceReplayAdapter] = {}


def register_local_source_replay_adapter(
    adapter: LocalSourceReplayAdapter,
    *,
    replace: bool = False,
) -> None:
    source_type = str(adapter.source_type or "").strip().lower()
    if not source_type:
        raise ValueError("local source replay adapter source_type is required")
    if source_type in _ADAPTERS and not replace:
        raise ValueError(f"local source replay adapter already registered: {source_type}")
    _ADAPTERS[source_type] = adapter


for _builtin_adapter in (
    GitHubRepoReplayAdapter(),
    JiraReplayAdapter(),
    LocalMarkdownReplayAdapter(),
    TeamsReplayAdapter(),
):
    register_local_source_replay_adapter(_builtin_adapter)


def get_local_source_replay_adapter(source_type: str) -> LocalSourceReplayAdapter:
    normalized = str(source_type or "").strip().lower()
    adapter = _ADAPTERS.get(normalized)
    if adapter is None:
        raise ValueError(f"local source replay adapter is not registered: {normalized}")
    return adapter


def registered_local_source_types() -> frozenset[str]:
    return frozenset(_ADAPTERS)
