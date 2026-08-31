"""Historical compatibility after the one-time recovery executable removal."""

from __future__ import annotations

from dataclasses import replace

import pytest
from click.testing import CliRunner

from memforge.main import cli
from memforge.memory import support_recovery
from memforge.memory.support_recovery import (
    LegacySupportRecoveryDisposition,
    LegacySupportRecoveryReport,
    LegacySupportRecoveryReportEntry,
    legacy_support_recovery_report_from_payload,
    legacy_support_recovery_report_id,
)


def _entry() -> LegacySupportRecoveryReportEntry:
    return LegacySupportRecoveryReportEntry(
        memory_id="mem-history",
        memory_version="memory-version",
        support_set_hash="support-hash",
        source_unit_id="unit-history",
        target_unit_revision_id="unitrev-history",
        doc_id="doc-history",
        access_context_hash="access-history",
        legacy_evidence_unit_ids=("eu-history",),
        disposition=LegacySupportRecoveryDisposition.SUPPORTED,
        reason="historical result",
        catalog_digest="catalog-history",
        primary_ref="f000001",
        required_refs=("f000002",),
    )


def _report() -> LegacySupportRecoveryReport:
    entry = _entry()
    report_id = legacy_support_recovery_report_id(
        source_id="src-history",
        llm_model="historical-model",
        entries=(entry,),
        memory_ids=(entry.memory_id,),
        selector_contract_version=3,
    )
    return LegacySupportRecoveryReport(
        id=report_id,
        source_id="src-history",
        llm_model="historical-model",
        entries=(entry,),
        created_at="2026-08-31T00:00:00+00:00",
        memory_ids=(entry.memory_id,),
        selector_contract_version=3,
    )


def test_historical_recovery_report_remains_readable_and_identity_checked() -> None:
    report = _report()

    parsed = legacy_support_recovery_report_from_payload(
        report_id=report.id,
        payload=report.to_payload(),
        created_at=report.created_at,
    )

    assert parsed == report
    assert parsed.ready_count == 1
    assert parsed.legacy_group_count == 1
    with pytest.raises(ValueError, match="identity is invalid"):
        legacy_support_recovery_report_from_payload(
            report_id="legacy-support-recovery-wrong",
            payload=report.to_payload(),
            created_at=report.created_at,
        )


@pytest.mark.parametrize("selector_contract_version", [1, 2, 3])
def test_every_persisted_report_contract_version_remains_readable(
    selector_contract_version: int,
) -> None:
    entry = _entry()
    memory_ids = () if selector_contract_version == 1 else (entry.memory_id,)
    report_id = legacy_support_recovery_report_id(
        source_id="src-history",
        llm_model="historical-model",
        entries=(entry,),
        memory_ids=memory_ids,
        selector_contract_version=selector_contract_version,
    )
    report = LegacySupportRecoveryReport(
        id=report_id,
        source_id="src-history",
        llm_model="historical-model",
        entries=(entry,),
        created_at="2026-08-31T00:00:00+00:00",
        memory_ids=memory_ids,
        selector_contract_version=selector_contract_version,
    )
    payload = dict(report.to_payload())
    if selector_contract_version == 1:
        payload.pop("selector_contract_version")
        payload.pop("memory_ids", None)

    parsed = legacy_support_recovery_report_from_payload(
        report_id=report.id,
        payload=payload,
        created_at=report.created_at,
    )

    assert parsed == report


def test_report_reason_is_not_part_of_exact_historical_identity() -> None:
    report = _report()
    changed_reason = replace(report.entries[0], reason="updated explanation")

    assert legacy_support_recovery_report_id(
        source_id=report.source_id,
        llm_model=report.llm_model,
        entries=(changed_reason,),
        memory_ids=report.memory_ids,
        selector_contract_version=report.selector_contract_version,
    ) == report.id


def test_one_time_recovery_execution_surface_is_absent() -> None:
    result = CliRunner().invoke(cli, ["maintenance", "--help"])

    assert result.exit_code == 0
    assert "recover-legacy-support" not in result.output
    for name in (
        "LegacySupportRecoveryCandidate",
        "PreparedLegacySupportRecovery",
        "prepare_legacy_support_recovery",
        "prepare_legacy_support_recovery_from_report",
        "apply_prepared_legacy_support_recovery",
        "build_legacy_support_recovery_plan",
    ):
        assert not hasattr(support_recovery, name)
