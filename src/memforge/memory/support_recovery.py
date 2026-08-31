"""Read-only compatibility for persisted legacy Support recovery reports.

The one-time report/apply workflow was removed after every deployed workspace
converged. Historical reports remain immutable and parseable for audit; this
module contains no candidate enumeration, model call, plan construction, or
Memory mutation capability.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence


LEGACY_SUPPORT_SELECTOR_CONTRACT_VERSIONS = frozenset({1, 2, 3})


class LegacySupportRecoveryDisposition(str, Enum):
    MECHANICALLY_RECOVERABLE = "mechanically_recoverable"
    SUPPORTED = "supported"
    NOT_SUPPORTED = "not_supported"
    INCONCLUSIVE = "inconclusive"
    ALREADY_SUPPORTED = "already_supported"
    HISTORICAL_ONLY = "historical_only"


@dataclass(frozen=True, slots=True)
class LegacySupportRecoveryReportEntry:
    memory_id: str
    memory_version: str
    support_set_hash: str
    source_unit_id: str
    target_unit_revision_id: str
    doc_id: str
    access_context_hash: str
    legacy_evidence_unit_ids: tuple[str, ...]
    disposition: LegacySupportRecoveryDisposition
    reason: str
    catalog_digest: str | None = None
    primary_ref: str | None = None
    required_refs: tuple[str, ...] = ()

    def to_payload(self) -> Mapping[str, object]:
        return {
            "memory_id": self.memory_id,
            "memory_version": self.memory_version,
            "support_set_hash": self.support_set_hash,
            "source_unit_id": self.source_unit_id,
            "target_unit_revision_id": self.target_unit_revision_id,
            "doc_id": self.doc_id,
            "access_context_hash": self.access_context_hash,
            "legacy_evidence_unit_ids": list(self.legacy_evidence_unit_ids),
            "disposition": self.disposition.value,
            "reason": self.reason,
            "catalog_digest": self.catalog_digest,
            "primary_ref": self.primary_ref,
            "required_refs": list(self.required_refs),
        }

    def identity_payload(self) -> Mapping[str, object]:
        payload = dict(self.to_payload())
        payload.pop("reason")
        return payload


@dataclass(frozen=True, slots=True)
class LegacySupportRecoveryReport:
    id: str
    source_id: str
    llm_model: str | None
    entries: tuple[LegacySupportRecoveryReportEntry, ...]
    created_at: str
    memory_ids: tuple[str, ...] = ()
    selector_contract_version: int = 1

    def __post_init__(self) -> None:
        if self.memory_ids != tuple(sorted(set(self.memory_ids))) or any(
            not value for value in self.memory_ids
        ):
            raise ValueError("legacy Support recovery report Memory cohort is invalid")
        if self.selector_contract_version not in LEGACY_SUPPORT_SELECTOR_CONTRACT_VERSIONS:
            raise ValueError("legacy Support selector contract version is unsupported")

    @property
    def ready_count(self) -> int:
        return sum(
            item.disposition
            in {
                LegacySupportRecoveryDisposition.MECHANICALLY_RECOVERABLE,
                LegacySupportRecoveryDisposition.SUPPORTED,
            }
            for item in self.entries
        )

    @property
    def legacy_group_count(self) -> int:
        return sum(len(item.legacy_evidence_unit_ids) for item in self.entries)

    @property
    def ready_group_count(self) -> int:
        return sum(
            len(item.legacy_evidence_unit_ids)
            for item in self.entries
            if item.disposition
            in {
                LegacySupportRecoveryDisposition.MECHANICALLY_RECOVERABLE,
                LegacySupportRecoveryDisposition.SUPPORTED,
            }
        )

    def to_payload(self) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "kind": "legacy_support_recovery",
            "source_id": self.source_id,
            "llm_model": self.llm_model,
            "selector_contract_version": self.selector_contract_version,
            "entries": [item.to_payload() for item in self.entries],
        }
        if self.memory_ids:
            payload["memory_ids"] = list(self.memory_ids)
        return payload


def legacy_support_recovery_report_id(
    *,
    source_id: str,
    llm_model: str | None,
    entries: Sequence[LegacySupportRecoveryReportEntry],
    memory_ids: Sequence[str] = (),
    selector_contract_version: int = 1,
) -> str:
    """Return the immutable identity used by stored historical reports."""

    identity: dict[str, object] = {
        "kind": "legacy_support_recovery",
        "source_id": source_id,
        "llm_model": llm_model,
        "entries": [entry.identity_payload() for entry in entries],
    }
    if selector_contract_version != 1:
        identity["selector_contract_version"] = selector_contract_version
    if memory_ids:
        identity["memory_ids"] = list(memory_ids)
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    return "legacy-support-recovery-" + digest


def legacy_support_recovery_report_from_payload(
    *,
    report_id: str,
    payload: Mapping[str, object],
    created_at: str,
) -> LegacySupportRecoveryReport:
    """Parse and verify one persisted immutable recovery report."""

    def required_text(item: Mapping[str, object], field: str) -> str:
        value = item.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"legacy Support recovery report has invalid {field}")
        return value

    def optional_text(item: Mapping[str, object], field: str) -> str | None:
        value = item.get(field)
        if value is None:
            return None
        if not isinstance(value, str) or not value:
            raise ValueError(f"legacy Support recovery report has invalid {field}")
        return value

    def text_tuple(item: Mapping[str, object], field: str) -> tuple[str, ...]:
        value = item.get(field)
        if not isinstance(value, list) or any(
            not isinstance(member, str) or not member for member in value
        ):
            raise ValueError(f"legacy Support recovery report has invalid {field}")
        return tuple(value)

    if payload.get("kind") != "legacy_support_recovery":
        raise ValueError("persisted report is not a legacy Support recovery report")
    source_id = required_text(payload, "source_id")
    selector_contract_version = payload.get("selector_contract_version", 1)
    if not isinstance(selector_contract_version, int) or isinstance(
        selector_contract_version, bool
    ):
        raise ValueError(
            "legacy Support recovery report has invalid selector_contract_version"
        )
    llm_model = payload.get("llm_model")
    if llm_model is not None and (
        not isinstance(llm_model, str) or not llm_model
    ):
        raise ValueError("legacy Support recovery report has invalid llm_model")
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list):
        raise ValueError("legacy Support recovery report has invalid entries")
    raw_memory_ids = payload.get("memory_ids", [])
    if not isinstance(raw_memory_ids, list) or any(
        not isinstance(value, str) for value in raw_memory_ids
    ):
        raise ValueError("legacy Support recovery report has invalid memory_ids")

    entries: list[LegacySupportRecoveryReportEntry] = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, Mapping):
            raise ValueError("legacy Support recovery report entry is invalid")
        reason = raw_entry.get("reason")
        if not isinstance(reason, str):
            raise ValueError("legacy Support recovery report has invalid reason")
        target_revision = raw_entry.get("target_unit_revision_id")
        if not isinstance(target_revision, str):
            raise ValueError(
                "legacy Support recovery report has invalid target_unit_revision_id"
            )
        try:
            disposition = LegacySupportRecoveryDisposition(
                required_text(raw_entry, "disposition")
            )
        except ValueError as exc:
            raise ValueError(
                "legacy Support recovery report has invalid disposition"
            ) from exc
        entries.append(
            LegacySupportRecoveryReportEntry(
                memory_id=required_text(raw_entry, "memory_id"),
                memory_version=required_text(raw_entry, "memory_version"),
                support_set_hash=required_text(raw_entry, "support_set_hash"),
                source_unit_id=required_text(raw_entry, "source_unit_id"),
                target_unit_revision_id=target_revision,
                doc_id=required_text(raw_entry, "doc_id"),
                access_context_hash=required_text(raw_entry, "access_context_hash"),
                legacy_evidence_unit_ids=text_tuple(
                    raw_entry, "legacy_evidence_unit_ids"
                ),
                disposition=disposition,
                reason=reason,
                catalog_digest=optional_text(raw_entry, "catalog_digest"),
                primary_ref=optional_text(raw_entry, "primary_ref"),
                required_refs=text_tuple(raw_entry, "required_refs"),
            )
        )

    report = LegacySupportRecoveryReport(
        id=report_id,
        source_id=source_id,
        llm_model=llm_model,
        entries=tuple(entries),
        created_at=created_at,
        memory_ids=tuple(raw_memory_ids),
        selector_contract_version=selector_contract_version,
    )
    expected_id = legacy_support_recovery_report_id(
        source_id=report.source_id,
        llm_model=report.llm_model,
        entries=report.entries,
        memory_ids=report.memory_ids,
        selector_contract_version=report.selector_contract_version,
    )
    if report.id != expected_id:
        raise ValueError("legacy Support recovery report identity is invalid")
    return report
