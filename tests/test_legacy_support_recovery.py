from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
import pytest_asyncio
from click.testing import CliRunner
from pydantic import ValidationError

from memforge.memory import support_recovery as support_recovery_module
from memforge.llm.structured import (
    LegacyNotSupportedRevalidationDecision,
    LegacySupportFragmentCandidatesDecision,
    LegacySupportFragmentNoneDecision,
    LegacySupportFragmentScanResponse,
    LegacySupportRevalidationResponse,
    LegacySupportedRevalidationDecision,
    StructuredLlmError,
)
from memforge.main import cli
from memforge.memory.evidence import (
    EvidenceContentProvenance,
    EvidenceReference,
    EvidenceRole,
    EvidenceUnit,
    MemorySupportAssertion,
    SupportScopeVersion,
)
from memforge.memory.lifecycle_plan import LifecycleGateState, LifecycleMutationType
from memforge.memory.support_recovery import (
    LegacySupportRecoveryCandidate,
    LegacySupportRecoveryDisposition,
    LegacySupportRecoveryReport,
    LegacySupportRecoveryReportEntry,
    build_legacy_support_recovery_plan,
    compile_legacy_support_revalidation_catalog,
    legacy_support_revalidation_prompt,
    legacy_limited_recovery_reason_codes,
    legacy_recovery_candidate_key,
    legacy_recovery_preserves_group_identity,
    prepare_legacy_support_recovery,
    resolve_legacy_support_revalidation_response,
    resolve_mechanical_legacy_support,
)
from memforge.models import Memory, content_hash
from memforge.source_projection import (
    AnchorKind,
    EvidenceCoordinateSpace,
    EvidenceRepresentationProfile,
    ProjectionCoverage,
    RevisionDelta,
    SourceAnchor,
    SourceObservation,
    SourceObservationRevision,
    SourceProjection,
    SourceUnit,
    SourceUnitRevision,
)
from memforge.storage.database import Database


MARKDOWN_PROFILE = EvidenceRepresentationProfile(
    name="markdown-structural",
    version=1,
    coordinate_space=EvidenceCoordinateSpace.UNICODE_SCALAR,
)
BINARY_PROFILE = EvidenceRepresentationProfile(
    name="binary-artifact",
    version=1,
    coordinate_space=EvidenceCoordinateSpace.WHOLE_ARTIFACT,
)


@pytest_asyncio.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "legacy-support-recovery.db"))
    await database.connect()
    try:
        yield database
    finally:
        await database.close()


def _projection(*, include_artifact: bool = False) -> SourceProjection:
    body = "# Release\n\nProduction release requires approval.\n"
    revisions = [
        SourceObservationRevision(
            id="rev-body",
            observation_id="obs-body",
            semantic_hash=hashlib.sha256(body.encode()).hexdigest(),
            content=body,
            evidence_profile=MARKDOWN_PROFILE,
        )
    ]
    observations = [
        SourceObservation(
            id="obs-body",
            source_id="source-1",
            source_unit_id="unit-1",
            observation_type="file_content",
            provider_key="body",
        )
    ]
    if include_artifact:
        revisions.append(
            SourceObservationRevision(
                id="rev-artifact",
                observation_id="obs-artifact",
                semantic_hash="artifact-semantic",
                content="",
                metadata={"source_artifact": {"sha256": "a" * 64, "filename": "proof.png"}},
                evidence_profile=BINARY_PROFILE,
            )
        )
        observations.append(
            SourceObservation(
                id="obs-artifact",
                source_id="source-1",
                source_unit_id="unit-1",
                observation_type="binary_artifact",
                provider_key="attachment:proof.png",
            )
        )
    unit = SourceUnit(
        id="unit-1",
        source_id="source-1",
        unit_type="document",
        provider_key="doc-1",
    )
    unit_revision = SourceUnitRevision(
        id="unitrev-current",
        source_unit_id=unit.id,
        semantic_hash="unit-semantic",
        observation_revision_ids=tuple(item.id for item in revisions),
        access_hash="workspace-access",
    )
    return SourceProjection(
        run_id="stored-current-unitrev-current",
        source_id="source-1",
        source_type="github_repo",
        scope={"stored_current_revalidation": True},
        coverage=ProjectionCoverage.COMPLETE_SNAPSHOT,
        observations=tuple(observations),
        observation_revisions=tuple(revisions),
        source_units=(unit,),
        source_unit_revisions=(unit_revision,),
        relations=(),
        deltas=(
            RevisionDelta(
                source_unit_id=unit.id,
                previous_unit_revision_id=unit_revision.id,
                current_unit_revision_id=unit_revision.id,
                axes=frozenset(),
                coverage=ProjectionCoverage.COMPLETE_SNAPSHOT,
            ),
        ),
        checkpoint={"stored_current_revalidation": True},
    )


def _candidate(*, include_artifact: bool = False) -> LegacySupportRecoveryCandidate:
    claim = "Production release requires approval."
    projection = _projection(include_artifact=include_artifact)
    references = (
        EvidenceReference(
            id="legacy-primary",
            evidence_unit_id="legacy-unit",
            role=EvidenceRole.PRIMARY,
            anchor=SourceAnchor(
                kind=AnchorKind.WHOLE_OBSERVATION,
                observation_id="obs-body",
                observation_revision_id="rev-body",
            ),
        ),
    )
    if include_artifact:
        references += (
            EvidenceReference(
                id="legacy-required",
                evidence_unit_id="legacy-unit",
                role=EvidenceRole.REQUIRED,
                anchor=SourceAnchor(
                    kind=AnchorKind.WHOLE_OBSERVATION,
                    observation_id="obs-artifact",
                    observation_revision_id="rev-artifact",
                ),
            ),
        )
    return LegacySupportRecoveryCandidate(
        memory=Memory(
            id="memory-1",
            memory_type="decision",
            content=claim,
            content_hash=content_hash(claim),
        ),
        memory_version="memory-version-current",
        support_set_hash="support-set-current",
        source_id="source-1",
        source_type="github_repo",
        source_unit_id="unit-1",
        doc_id="doc-1",
        access_context_hash="workspace-access",
        projection=projection,
        legacy_evidence_unit_ids=("legacy-unit",),
        legacy_references=references,
        reason_codes=("part_unresolvable" if include_artifact else "unit_revision_lineage_invalid",),
    )


def _candidate_with_body(
    body: str,
    *,
    claim: str | None = None,
) -> LegacySupportRecoveryCandidate:
    projection = _projection()
    revision = replace(
        projection.observation_revisions[0],
        semantic_hash=hashlib.sha256(body.encode()).hexdigest(),
        content=body,
    )
    candidate = replace(
        _candidate(),
        projection=replace(projection, observation_revisions=(revision,)),
    )
    if claim is None:
        return candidate
    return replace(
        candidate,
        memory=replace(
            candidate.memory,
            content=claim,
            content_hash=content_hash(claim),
        ),
    )


class _RecoveryFakeDb:
    def __init__(self, candidate: LegacySupportRecoveryCandidate) -> None:
        self.candidate = candidate
        self.persisted = None

    async def list_legacy_support_recovery_candidates(self, source_id: str):
        assert source_id == "source-1"
        return (self.candidate,)

    async def get_lifecycle_gate(self, source_id: str):
        assert source_id == "source-1"
        return type("Gate", (), {"state": LifecycleGateState.GATED})()

    async def persist_legacy_support_recovery_report(self, report):
        self.persisted = report

    async def get_legacy_support_recovery_report(self, report_id: str):
        assert self.persisted is not None and self.persisted.id == report_id
        return self.persisted


def test_revalidation_schema_requires_evidence_only_for_supported() -> None:
    with pytest.raises(ValidationError, match="primary_ref"):
        LegacySupportedRevalidationDecision(
            request_position=0,
            decision="supported",
            reason="missing selector",
        )
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        LegacyNotSupportedRevalidationDecision(
            request_position=0,
            decision="not_supported",
            primary_ref="f000001",
        )

    schema = LegacySupportRevalidationResponse.model_json_schema()
    decision_items = schema["properties"]["decisions"]["items"]
    decision_items = schema["$defs"][decision_items["$ref"].rsplit("/", 1)[-1]]
    assert decision_items["discriminator"]["propertyName"] == "decision"
    supported_ref = decision_items["discriminator"]["mapping"]["supported"]
    supported_schema = schema["$defs"][supported_ref.rsplit("/", 1)[-1]]
    assert "primary_ref" in supported_schema["required"]

    with pytest.raises(ValidationError, match="refs"):
        LegacySupportFragmentCandidatesDecision(
            request_position=0,
            outcome="candidates",
            refs=[],
        )
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        LegacySupportFragmentNoneDecision(
            request_position=0,
            outcome="none",
            refs=["f000001"],
        )
    scan_schema = LegacySupportFragmentScanResponse.model_json_schema()
    scan_items = scan_schema["properties"]["decisions"]["items"]
    scan_items = scan_schema["$defs"][scan_items["$ref"].rsplit("/", 1)[-1]]
    assert scan_items["discriminator"]["propertyName"] == "outcome"


def test_eligible_legacy_limited_group_retains_mechanical_recovery_reason() -> None:
    assert legacy_limited_recovery_reason_codes(()) == ("part_unresolvable",)
    assert legacy_limited_recovery_reason_codes(("unit_revision_lineage_invalid",)) == (
        "unit_revision_lineage_invalid",
    )
    assert legacy_recovery_preserves_group_identity(("part_unresolvable",))
    assert not legacy_recovery_preserves_group_identity(("unit_revision_lineage_invalid",))
    common = {
        "memory_id": "mem-1",
        "source_unit_id": "unit-1",
        "access_context_hash": "access-1",
        "doc_id": "doc-1",
        "legacy_support_active": True,
    }
    assert legacy_recovery_candidate_key(
        **common,
        legacy_evidence_unit_id="eu-1",
        reason_codes=("part_unresolvable",),
    ) != legacy_recovery_candidate_key(
        **common,
        legacy_evidence_unit_id="eu-2",
        reason_codes=("part_unresolvable",),
    )
    assert legacy_recovery_candidate_key(
        **common,
        legacy_evidence_unit_id="eu-1",
        reason_codes=("unit_revision_lineage_invalid",),
    ) == legacy_recovery_candidate_key(
        **common,
        legacy_evidence_unit_id="eu-2",
        reason_codes=("unit_revision_lineage_invalid",),
    )


def test_recovery_cli_requires_exact_report_for_apply() -> None:
    help_result = CliRunner().invoke(
        cli,
        ["maintenance", "recover-legacy-support", "--help"],
    )
    assert help_result.exit_code == 0
    assert "--expected-report-id" in help_result.output

    result = CliRunner().invoke(
        cli,
        [
            "maintenance",
            "recover-legacy-support",
            "--source-id",
            "src-1",
            "--apply",
        ],
    )
    assert result.exit_code == 2
    assert "requires exactly one" in result.output


def test_revalidation_resolves_current_catalog_and_builds_support_only_plan() -> None:
    candidate = _candidate()
    catalog = compile_legacy_support_revalidation_catalog(candidate)
    assert catalog.usable
    primary_ref = next(
        fragment.reference
        for fragment in catalog.fragments
        if EvidenceRole.PRIMARY in fragment.eligible_roles
        and "Production release requires approval" in fragment.presentation_text
    )
    response = LegacySupportRevalidationResponse(
        decisions=[
            LegacySupportedRevalidationDecision(
                request_position=0,
                decision="supported",
                primary_ref=primary_ref,
                reason="The current release rule states the claim.",
            )
        ]
    )
    decisions = resolve_legacy_support_revalidation_response(
        catalog=catalog,
        candidates=(candidate,),
        response=response,
    )
    plan = build_legacy_support_recovery_plan(
        decisions=decisions,
        gate_state=LifecycleGateState.GATED,
        report_id="legacy-recovery-report-1",
    )
    assert plan is not None
    plan.validate()
    assert decisions[0].disposition is LegacySupportRecoveryDisposition.SUPPORTED
    assert {item.mutation_type for item in plan.mutations} == {
        LifecycleMutationType.ATTACH_SUPPORT,
        LifecycleMutationType.REFRESH_MEMORY_INDEX,
    }
    assert all(
        item.mutation_type
        not in {
            LifecycleMutationType.REMOVE_SUPPORT,
            LifecycleMutationType.SUPERSEDE_MEMORY,
            LifecycleMutationType.RETIRE_MEMORY,
        }
        for item in plan.mutations
    )
    assert "legacy-unit" not in {item.id for item in plan.evidence_units}


def test_recovery_plan_coalesces_duplicate_memory_coverage_and_mutations() -> None:
    decision = resolve_mechanical_legacy_support(_candidate(include_artifact=True))

    plan = build_legacy_support_recovery_plan(
        decisions=(decision, decision),
        gate_state=LifecycleGateState.GATED,
        report_id="legacy-recovery-duplicate-memory",
    )

    assert plan is not None
    plan.validate()
    assert plan.coverage_proof.mandatory_incumbent_ids == ("memory-1",)
    assert [item.memory_id for item in plan.coverage_proof.incumbent_decisions] == ["memory-1"]
    assert [item.mutation_type for item in plan.mutations].count(LifecycleMutationType.ATTACH_SUPPORT) == 1
    assert [item.mutation_type for item in plan.mutations].count(LifecycleMutationType.REFRESH_MEMORY_INDEX) == 1


def test_revalidation_requires_complete_position_coverage() -> None:
    candidate = _candidate()
    catalog = compile_legacy_support_revalidation_catalog(candidate)
    with pytest.raises(ValueError, match="coverage is incomplete"):
        resolve_legacy_support_revalidation_response(
            catalog=catalog,
            candidates=(candidate,),
            response=LegacySupportRevalidationResponse(decisions=[]),
        )


def test_revalidation_prompt_exposes_fragment_refs_not_durable_ids() -> None:
    candidate = _candidate()
    catalog = compile_legacy_support_revalidation_catalog(candidate)
    prompt = legacy_support_revalidation_prompt(catalog, (candidate,))
    assert '"ref":"f000001"' in prompt
    assert "legacy-unit" not in prompt
    assert "memory-1" not in prompt


@pytest.mark.asyncio
async def test_large_catalog_scans_every_window_before_cross_window_adjudication() -> None:
    filler = "\n\n".join(f"Filler paragraph {index} " + ("x" * 980) for index in range(130))
    body = f"# Large policy\n\nAlpha clause.\n\n{filler}\n\nOmega clause."
    claim = "Alpha requires Omega."
    candidate = _candidate_with_body(body, claim=claim)

    def prompt_catalog(prompt: str, marker: str) -> list[dict[str, object]]:
        raw = prompt.split(marker + "\n", 1)[1].split("\n\nMEMORIES\n", 1)[0]
        return json.loads(raw)

    class WindowedClient:
        scan_calls = 0
        adjudication_calls = 0

        async def screen_legacy_support_fragments(self, prompt: str, **kwargs):
            del kwargs
            self.scan_calls += 1
            fragments = prompt_catalog(prompt, "AUTHORIZED_FRAGMENT_WINDOW")
            selected = next(item for item in fragments if "Alpha" in str(item["text"]) or "Omega" in str(item["text"]))
            return SimpleNamespace(
                decisions=[
                    SimpleNamespace(
                        request_position=0,
                        outcome="candidates",
                        refs=[selected["ref"]],
                    )
                ]
            )

        async def revalidate_legacy_support(self, prompt: str, **kwargs):
            del kwargs
            self.adjudication_calls += 1
            fragments = prompt_catalog(prompt, "AUTHORIZED_FRAGMENT_CATALOG")
            primary = next(item["ref"] for item in fragments if "Alpha" in str(item["text"]))
            required = next(item["ref"] for item in fragments if "Omega" in str(item["text"]))
            return LegacySupportRevalidationResponse(
                decisions=[
                    LegacySupportedRevalidationDecision(
                        request_position=0,
                        decision="supported",
                        primary_ref=primary,
                        required_refs=[required],
                        reason="Both policy clauses support the complete claim.",
                    )
                ]
            )

    client = WindowedClient()
    db = _RecoveryFakeDb(candidate)
    prepared = await prepare_legacy_support_recovery(
        db,
        source_id="source-1",
        structured_llm_client=client,
        llm_model="test-model",
    )

    assert client.scan_calls == 2
    assert client.adjudication_calls == 1
    assert prepared.report.entries[0].disposition is LegacySupportRecoveryDisposition.SUPPORTED
    assert prepared.decisions[0].selection is not None
    assert [part.role for part in prepared.decisions[0].selection.parts] == [
        EvidenceRole.PRIMARY,
        EvidenceRole.REQUIRED,
    ]
    assert db.persisted == prepared.report

    replayed = await support_recovery_module.prepare_legacy_support_recovery_from_report(
        db,
        source_id="source-1",
        report_id=prepared.report.id,
    )
    assert client.scan_calls == 2
    assert client.adjudication_calls == 1
    assert replayed.decisions[0].selection == prepared.decisions[0].selection


@pytest.mark.asyncio
async def test_fragment_count_over_one_call_budget_is_exhaustively_scanned() -> None:
    body = "\n\n".join(f"Paragraph {index}." for index in range(2_049))
    candidate = _candidate_with_body(body)

    class CompleteNoneClient:
        scan_calls = 0
        seen_refs: list[str] = []

        async def screen_legacy_support_fragments(self, prompt: str, **kwargs):
            del kwargs
            self.scan_calls += 1
            raw = prompt.split("AUTHORIZED_FRAGMENT_WINDOW\n", 1)[1].split("\n\nMEMORIES\n", 1)[0]
            self.seen_refs.extend(item["ref"] for item in json.loads(raw))
            return SimpleNamespace(decisions=[SimpleNamespace(request_position=0, outcome="none")])

        async def revalidate_legacy_support(self, prompt: str, **kwargs):
            del prompt, kwargs
            raise AssertionError("complete no-candidate coverage needs no adjudication")

    client = CompleteNoneClient()
    prepared = await prepare_legacy_support_recovery(
        _RecoveryFakeDb(candidate),
        source_id="source-1",
        structured_llm_client=client,
        llm_model="test-model",
    )

    assert client.scan_calls == 2
    assert len(client.seen_refs) == 2_049
    assert len(set(client.seen_refs)) == 2_049
    assert client.seen_refs[0] == "f000001"
    assert client.seen_refs[-1] == "f002049"
    assert prepared.report.entries[0].disposition is LegacySupportRecoveryDisposition.NOT_SUPPORTED
    assert prepared.report.entries[0].reason == ("complete_window_scan_found_no_candidate_evidence")


@pytest.mark.asyncio
async def test_incomplete_window_coverage_retries_the_same_exact_batch_once() -> None:
    filler = "\n\n".join(f"Filler paragraph {index} " + ("x" * 980) for index in range(130))
    candidate = _candidate_with_body(f"# Large policy\n\n{filler}")

    class CorrectedClient:
        prompts: list[str] = []

        async def screen_legacy_support_fragments(self, prompt: str, **kwargs):
            del kwargs
            self.prompts.append(prompt)
            if len(self.prompts) == 2:
                return SimpleNamespace(decisions=[])
            return SimpleNamespace(
                decisions=[SimpleNamespace(request_position=0, outcome="none")]
            )

        async def revalidate_legacy_support(self, prompt: str, **kwargs):
            del prompt, kwargs
            raise AssertionError("complete no-candidate coverage needs no adjudication")

    client = CorrectedClient()
    prepared = await prepare_legacy_support_recovery(
        _RecoveryFakeDb(candidate),
        source_id="source-1",
        structured_llm_client=client,
        llm_model="test-model",
    )

    assert len(client.prompts) == 3
    assert client.prompts[2].startswith(client.prompts[1])
    assert "<coverage_correction>" in client.prompts[2]
    assert prepared.report.entries[0].disposition is LegacySupportRecoveryDisposition.NOT_SUPPORTED
    assert prepared.report.entries[0].reason == (
        "complete_window_scan_found_no_candidate_evidence"
    )


@pytest.mark.asyncio
async def test_incomplete_window_coverage_cannot_authorize_a_negative_decision() -> None:
    filler = "\n\n".join(f"Filler paragraph {index} " + ("x" * 980) for index in range(130))
    body = f"# Large policy\n\nPotential support.\n\n{filler}"
    candidate = _candidate_with_body(body)

    class IncompleteClient:
        calls = 0

        async def screen_legacy_support_fragments(self, prompt: str, **kwargs):
            del prompt, kwargs
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(
                    decisions=[
                        SimpleNamespace(
                            request_position=0,
                            outcome="candidates",
                            refs=["f000001"],
                        )
                    ]
                )
            return SimpleNamespace(decisions=[])

        async def revalidate_legacy_support(self, prompt: str, **kwargs):
            del prompt, kwargs
            raise AssertionError("incomplete screening cannot reach adjudication")

    prepared = await prepare_legacy_support_recovery(
        _RecoveryFakeDb(candidate),
        source_id="source-1",
        structured_llm_client=IncompleteClient(),
        llm_model="test-model",
    )

    assert prepared.report.entries[0].disposition is LegacySupportRecoveryDisposition.INCONCLUSIVE
    assert prepared.report.entries[0].reason == ("llm_revalidation_failed:invalid_response:scan_coverage_incomplete")
    assert prepared.plans == ()


@pytest.mark.asyncio
async def test_one_unpresentable_fragment_fails_closed_without_splitting() -> None:
    body = "One atomic paragraph " + ("x" * 130_000)
    candidate = _candidate_with_body(body)

    class NoCallClient:
        async def screen_legacy_support_fragments(self, prompt: str, **kwargs):
            del prompt, kwargs
            raise AssertionError("an unpresentable atomic Fragment cannot be split")

        async def revalidate_legacy_support(self, prompt: str, **kwargs):
            del prompt, kwargs
            raise AssertionError("an unpresentable atomic Fragment cannot be adjudicated")

    prepared = await prepare_legacy_support_recovery(
        _RecoveryFakeDb(candidate),
        source_id="source-1",
        structured_llm_client=NoCallClient(),
        llm_model="test-model",
    )

    assert prepared.report.entries[0].disposition is LegacySupportRecoveryDisposition.INCONCLUSIVE
    assert prepared.report.entries[0].reason == ("catalog_unusable:fragment_unpresentable")
    assert prepared.plans == ()


@pytest.mark.asyncio
async def test_window_candidate_overflow_cannot_authorize_support_or_negative() -> None:
    filler = "\n\n".join(f"Filler paragraph {index} " + ("x" * 980) for index in range(130))
    body = f"# Large policy\n\nPotential support.\n\n{filler}"
    candidate = _candidate_with_body(body)

    class OverflowClient:
        calls = 0

        async def screen_legacy_support_fragments(self, prompt: str, **kwargs):
            del prompt, kwargs
            self.calls += 1
            return SimpleNamespace(
                decisions=[
                    SimpleNamespace(
                        request_position=0,
                        outcome=("candidate_overflow" if self.calls == 1 else "none"),
                    )
                ]
            )

        async def revalidate_legacy_support(self, prompt: str, **kwargs):
            del prompt, kwargs
            raise AssertionError("overflowed screening cannot reach adjudication")

    prepared = await prepare_legacy_support_recovery(
        _RecoveryFakeDb(candidate),
        source_id="source-1",
        structured_llm_client=OverflowClient(),
        llm_model="test-model",
    )

    assert prepared.report.entries[0].disposition is LegacySupportRecoveryDisposition.INCONCLUSIVE
    assert prepared.report.entries[0].reason == "window_candidate_overflow"
    assert prepared.plans == ()


@pytest.mark.asyncio
async def test_window_scan_budget_exhaustion_fails_closed() -> None:
    body = "\n\n".join(f"Atomic paragraph {index} " + ("x" * 70_000) for index in range(17))
    candidate = _candidate_with_body(body)

    class NoCallClient:
        async def screen_legacy_support_fragments(self, prompt: str, **kwargs):
            del prompt, kwargs
            raise AssertionError("an exhausted scan budget cannot call the model")

        async def revalidate_legacy_support(self, prompt: str, **kwargs):
            del prompt, kwargs
            raise AssertionError("an exhausted scan budget cannot call the model")

    prepared = await prepare_legacy_support_recovery(
        _RecoveryFakeDb(candidate),
        source_id="source-1",
        structured_llm_client=NoCallClient(),
        llm_model="test-model",
    )

    assert prepared.report.entries[0].disposition is LegacySupportRecoveryDisposition.INCONCLUSIVE
    assert prepared.report.entries[0].reason == ("catalog_unusable:scan_budget_exhausted")
    assert prepared.plans == ()


@pytest.mark.asyncio
async def test_final_adjudication_cannot_select_an_unscreened_global_ref() -> None:
    filler = "\n\n".join(
        f"Filler paragraph {index} " + ("x" * 980) for index in range(130)
    )
    body = f"# Large policy\n\nAlpha clause.\n\n{filler}\n\nOmega clause."
    candidate = _candidate_with_body(body)
    corpus = compile_legacy_support_revalidation_catalog(
        candidate,
        selector_contract_version=2,
    )
    alpha_ref = next(
        item.reference for item in corpus.fragments if "Alpha" in item.presentation_text
    )
    omega_ref = next(
        item.reference for item in corpus.fragments if "Omega" in item.presentation_text
    )

    class UnscreenedSelectionClient:
        calls = 0

        async def screen_legacy_support_fragments(self, prompt: str, **kwargs):
            del prompt, kwargs
            self.calls += 1
            return SimpleNamespace(
                decisions=[
                    SimpleNamespace(
                        request_position=0,
                        outcome=("candidates" if self.calls == 1 else "none"),
                        refs=([alpha_ref] if self.calls == 1 else []),
                    )
                ]
            )

        async def revalidate_legacy_support(self, prompt: str, **kwargs):
            del prompt, kwargs
            return LegacySupportRevalidationResponse(
                decisions=[
                    LegacySupportedRevalidationDecision(
                        request_position=0,
                        decision="supported",
                        primary_ref=omega_ref,
                    )
                ]
            )

    prepared = await prepare_legacy_support_recovery(
        _RecoveryFakeDb(candidate),
        source_id="source-1",
        structured_llm_client=UnscreenedSelectionClient(),
        llm_model="test-model",
    )

    assert prepared.report.entries[0].disposition is LegacySupportRecoveryDisposition.INCONCLUSIVE
    assert prepared.report.entries[0].reason == (
        "llm_revalidation_failed:invalid_response:selection_outside_candidates"
    )
    assert prepared.plans == ()


def test_mechanical_recovery_classifies_each_reference_by_its_profile() -> None:
    decision = resolve_mechanical_legacy_support(_candidate(include_artifact=True))
    assert decision.disposition is LegacySupportRecoveryDisposition.MECHANICALLY_RECOVERABLE
    assert decision.selection is not None
    assert [(part.role.value, part.kind.value) for part in decision.selection.parts] == [
        ("primary", "text"),
        ("required", "artifact"),
    ]


@pytest.mark.asyncio
async def test_llm_failure_becomes_one_durable_inconclusive_report() -> None:
    candidate = _candidate()

    class FailingClient:
        calls = 0

        async def revalidate_legacy_support(self, prompt: str, **kwargs):
            del prompt, kwargs
            self.calls += 1
            raise TimeoutError("provider timeout")

    class FakeDb:
        persisted = None

        async def list_legacy_support_recovery_candidates(self, source_id: str):
            assert source_id == "source-1"
            return (candidate,)

        async def get_lifecycle_gate(self, source_id: str):
            assert source_id == "source-1"
            return type("Gate", (), {"state": LifecycleGateState.GATED})()

        async def persist_legacy_support_recovery_report(self, report):
            self.persisted = report

    client = FailingClient()
    db = FakeDb()
    prepared = await prepare_legacy_support_recovery(
        db,
        source_id="source-1",
        structured_llm_client=client,
        llm_model="test-model",
    )

    assert client.calls == 1
    assert prepared.report.entries[0].disposition is LegacySupportRecoveryDisposition.INCONCLUSIVE
    assert prepared.report.entries[0].reason == "llm_revalidation_failed:TimeoutError"
    assert db.persisted == prepared.report
    assert prepared.plans == ()


@pytest.mark.asyncio
async def test_llm_failure_does_not_skip_an_independent_revalidation_scope() -> None:
    first = _candidate()
    assert first.projection is not None
    second_unit_revision = replace(
        first.projection.source_unit_revisions[0],
        id="unitrev-current-second-access",
        access_hash="workspace-access-second",
    )
    second_projection = replace(
        first.projection,
        run_id="stored-current-unitrev-current-second-access",
        source_unit_revisions=(second_unit_revision,),
        deltas=(
            replace(
                first.projection.deltas[0],
                previous_unit_revision_id=second_unit_revision.id,
                current_unit_revision_id=second_unit_revision.id,
            ),
        ),
    )
    second = replace(
        first,
        memory=replace(first.memory, id="memory-2"),
        memory_version="memory-version-current-2",
        support_set_hash="support-set-current-2",
        access_context_hash="workspace-access-second",
        projection=second_projection,
        legacy_evidence_unit_ids=("legacy-unit-2",),
    )
    second_catalog = compile_legacy_support_revalidation_catalog(second)
    second_primary_ref = next(
        fragment.reference
        for fragment in second_catalog.fragments
        if EvidenceRole.PRIMARY in fragment.eligible_roles
        and "Production release requires approval" in fragment.presentation_text
    )

    class FailThenSucceedClient:
        calls = 0

        async def revalidate_legacy_support(self, prompt: str, **kwargs):
            del prompt, kwargs
            self.calls += 1
            if self.calls == 1:
                raise StructuredLlmError(
                    "invalid structured response",
                    terminal_category="invalid_response",
                    error_code="ValidationError",
                )
            return LegacySupportRevalidationResponse(
                decisions=[
                    LegacySupportedRevalidationDecision(
                        request_position=0,
                        decision="supported",
                        primary_ref=second_primary_ref,
                        reason="current Evidence supports the complete claim",
                    )
                ]
            )

    class FakeDb:
        persisted = None

        async def list_legacy_support_recovery_candidates(self, source_id: str):
            assert source_id == "source-1"
            return (first, second)

        async def get_lifecycle_gate(self, source_id: str):
            assert source_id == "source-1"
            return type("Gate", (), {"state": LifecycleGateState.GATED})()

        async def persist_legacy_support_recovery_report(self, report):
            self.persisted = report

    client = FailThenSucceedClient()
    db = FakeDb()
    prepared = await prepare_legacy_support_recovery(
        db,
        source_id="source-1",
        structured_llm_client=client,
        llm_model="test-model",
    )

    entries = {entry.memory_id: entry for entry in prepared.report.entries}
    assert client.calls == 2
    assert entries["memory-1"].disposition is LegacySupportRecoveryDisposition.INCONCLUSIVE
    assert entries["memory-1"].reason == ("llm_revalidation_failed:invalid_response:ValidationError")
    assert entries["memory-2"].disposition is LegacySupportRecoveryDisposition.SUPPORTED
    assert db.persisted == prepared.report


@pytest.mark.asyncio
async def test_unexpected_revalidation_bug_is_not_persisted_as_model_inconclusive() -> None:
    candidate = _candidate()

    class BrokenClient:
        async def revalidate_legacy_support(self, prompt: str, **kwargs):
            del prompt, kwargs
            raise RuntimeError("programming defect")

    class FakeDb:
        persisted = None

        async def list_legacy_support_recovery_candidates(self, source_id: str):
            assert source_id == "source-1"
            return (candidate,)

        async def persist_legacy_support_recovery_report(self, report):
            self.persisted = report

    db = FakeDb()
    with pytest.raises(RuntimeError, match="programming defect"):
        await prepare_legacy_support_recovery(
            db,
            source_id="source-1",
            structured_llm_client=BrokenClient(),
            llm_model="test-model",
        )
    assert db.persisted is None


@pytest.mark.asyncio
async def test_sqlite_legacy_inventory_binds_source_predicate() -> None:
    calls = []

    class Connection:
        async def execute_fetchall(self, sql, params):
            calls.append((" ".join(sql.split()), params))
            return []

    database = object.__new__(Database)
    database._db = Connection()

    assert (
        await database._legacy_support_group_rows_unlocked(
            source_id="src-1",
            legacy_limited_only=True,
        )
        == []
    )
    assert len(calls) == 1
    assert "WHERE msa.source_id = ?" in calls[0][0]
    assert "AND eu.evidence_provenance = 'legacy_limited'" in calls[0][0]
    assert calls[0][1] == ("src-1",)


@pytest.mark.asyncio
async def test_report_identity_ignores_explanatory_llm_wording() -> None:
    candidate = _candidate()
    catalog = compile_legacy_support_revalidation_catalog(candidate)
    primary_ref = next(
        fragment.reference
        for fragment in catalog.fragments
        if EvidenceRole.PRIMARY in fragment.eligible_roles
        and "Production release requires approval" in fragment.presentation_text
    )

    class Client:
        reason = "first wording"

        async def revalidate_legacy_support(self, prompt: str, **kwargs):
            del prompt, kwargs
            return LegacySupportRevalidationResponse(
                decisions=[
                    LegacySupportedRevalidationDecision(
                        request_position=0,
                        decision="supported",
                        primary_ref=primary_ref,
                        reason=self.reason,
                    )
                ]
            )

    class FakeDb:
        async def list_legacy_support_recovery_candidates(self, source_id: str):
            assert source_id == "source-1"
            return (candidate,)

        async def get_lifecycle_gate(self, source_id: str):
            return type("Gate", (), {"state": LifecycleGateState.GATED})()

        async def persist_legacy_support_recovery_report(self, report):
            pass

    client = Client()
    first = await prepare_legacy_support_recovery(
        FakeDb(),
        source_id="source-1",
        structured_llm_client=client,
        llm_model="test-model",
    )
    client.reason = "same proof with different explanatory wording"
    second = await prepare_legacy_support_recovery(
        FakeDb(),
        source_id="source-1",
        structured_llm_client=client,
        llm_model="test-model",
    )

    assert first.report.id == second.report.id
    assert first.report.entries[0].reason != second.report.entries[0].reason


@pytest.mark.asyncio
async def test_exact_report_replay_rebuilds_ready_selection_without_calling_llm() -> None:
    candidate = _candidate()
    catalog = compile_legacy_support_revalidation_catalog(candidate)
    primary_ref = next(
        fragment.reference
        for fragment in catalog.fragments
        if EvidenceRole.PRIMARY in fragment.eligible_roles
        and "Production release requires approval" in fragment.presentation_text
    )

    class Client:
        calls = 0

        async def revalidate_legacy_support(self, prompt: str, **kwargs):
            del prompt, kwargs
            self.calls += 1
            return LegacySupportRevalidationResponse(
                decisions=[
                    LegacySupportedRevalidationDecision(
                        request_position=0,
                        decision="supported",
                        primary_ref=primary_ref,
                    )
                ]
            )

    class FakeDb:
        def __init__(self):
            self.report = None
            self.candidate = candidate

        async def list_legacy_support_recovery_candidates(self, source_id: str):
            assert source_id == "source-1"
            return (self.candidate,)

        async def get_lifecycle_gate(self, source_id: str):
            assert source_id == "source-1"
            return type("Gate", (), {"state": LifecycleGateState.GATED})()

        async def persist_legacy_support_recovery_report(self, report):
            self.report = report

        async def get_legacy_support_recovery_report(self, report_id: str):
            assert self.report is not None and self.report.id == report_id
            return self.report

    client = Client()
    db = FakeDb()
    reported = await prepare_legacy_support_recovery(
        db,
        source_id="source-1",
        structured_llm_client=client,
        llm_model="test-model",
    )

    replayed = await support_recovery_module.prepare_legacy_support_recovery_from_report(
        db,
        source_id="source-1",
        report_id=reported.report.id,
    )

    assert client.calls == 1
    assert replayed.report == reported.report
    assert replayed.decisions[0].selection == reported.decisions[0].selection
    assert replayed.plans == reported.plans

    db.candidate = replace(candidate, support_set_hash="support-set-changed")
    with pytest.raises(ValueError, match="report is stale"):
        await support_recovery_module.prepare_legacy_support_recovery_from_report(
            db,
            source_id="source-1",
            report_id=reported.report.id,
        )


@pytest.mark.asyncio
async def test_pre_window_selector_report_remains_replayable() -> None:
    candidate = _candidate()
    catalog = compile_legacy_support_revalidation_catalog(candidate)
    primary_ref = next(
        fragment.reference
        for fragment in catalog.fragments
        if "Production release requires approval" in fragment.presentation_text
    )
    entry = LegacySupportRecoveryReportEntry(
        memory_id=candidate.memory.id,
        memory_version=candidate.memory_version,
        support_set_hash=candidate.support_set_hash,
        source_unit_id=candidate.source_unit_id,
        target_unit_revision_id=candidate.projection.source_unit_revisions[0].id,
        doc_id=candidate.doc_id,
        access_context_hash=candidate.access_context_hash,
        legacy_evidence_unit_ids=candidate.legacy_evidence_unit_ids,
        disposition=LegacySupportRecoveryDisposition.SUPPORTED,
        reason="current Evidence supports the claim",
        catalog_digest=catalog.digest,
        primary_ref=primary_ref,
    )
    report_id = support_recovery_module.legacy_support_recovery_report_id(
        source_id="source-1",
        llm_model="test-model",
        entries=(entry,),
    )
    report = LegacySupportRecoveryReport(
        id=report_id,
        source_id="source-1",
        llm_model="test-model",
        entries=(entry,),
        created_at="2026-08-29T00:00:00+00:00",
    )

    class FakeDb:
        async def get_legacy_support_recovery_report(self, value: str):
            assert value == report.id
            return report

        async def list_legacy_support_recovery_candidates(self, source_id: str):
            assert source_id == "source-1"
            return (candidate,)

        async def get_lifecycle_gate(self, source_id: str):
            assert source_id == "source-1"
            return type("Gate", (), {"state": LifecycleGateState.GATED})()

    replayed = await support_recovery_module.prepare_legacy_support_recovery_from_report(
        FakeDb(),
        source_id="source-1",
        report_id=report.id,
    )

    assert replayed.report.selector_contract_version == 1
    assert replayed.decisions[0].selection is not None
    assert replayed.decisions[0].selection.catalog_digest == catalog.digest


@pytest.mark.asyncio
async def test_sqlite_round_trips_and_verifies_exact_recovery_report(
    db: Database,
) -> None:
    entry = LegacySupportRecoveryReportEntry(
        memory_id="memory-1",
        memory_version="memory-version-current",
        support_set_hash="support-set-current",
        source_unit_id="unit-1",
        target_unit_revision_id="unitrev-current",
        doc_id="doc-1",
        access_context_hash="workspace-access",
        legacy_evidence_unit_ids=("legacy-unit",),
        disposition=LegacySupportRecoveryDisposition.SUPPORTED,
        reason="current Evidence supports the claim",
        catalog_digest="catalog-current",
        primary_ref="f000001",
    )
    report_id = support_recovery_module.legacy_support_recovery_report_id(
        source_id="source-1",
        llm_model="test-model",
        entries=(entry,),
    )
    report = LegacySupportRecoveryReport(
        id=report_id,
        source_id="source-1",
        llm_model="test-model",
        entries=(entry,),
        created_at="2026-08-29T00:00:00+00:00",
    )

    await db.persist_legacy_support_recovery_report(report)

    assert await db.get_legacy_support_recovery_report(report.id) == report
    tampered = dict(report.to_payload())
    tampered_entries = [dict(item) for item in tampered["entries"]]
    tampered_entries[0]["primary_ref"] = "f000002"
    tampered["entries"] = tampered_entries
    with pytest.raises(ValueError, match="identity is invalid"):
        support_recovery_module.legacy_support_recovery_report_from_payload(
            report_id=report.id,
            payload=tampered,
            created_at=report.created_at,
        )


@pytest.mark.asyncio
async def test_sqlite_inventory_rehydrates_current_projection_for_missing_history(
    db: Database,
) -> None:
    projection = _projection()
    await db.upsert_source(
        id="source-1",
        type="github_repo",
        name="Repository",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="owner-1",
    )
    await db.record_source_projection(projection)
    now = datetime.now(timezone.utc).isoformat()
    await db.db.execute(
        """INSERT INTO documents (
               doc_id, source, source_url, title, space_or_project,
               last_modified, version, content_hash, last_synced
           ) VALUES ('doc-1', 'source-1', 'https://example.test/doc-1',
                     'Release', 'ENG', ?, '1', 'doc-hash', ?)""",
        (now, now),
    )
    await db.db.commit()
    memory = _candidate().memory
    await db.insert_memory(memory)
    await db.db.execute(
        """INSERT INTO memory_sources (
               memory_id, doc_id, source_id, source_type, excerpt,
               support_kind, added_at
           ) VALUES (?, 'doc-1', 'source-1', 'github_repo', NULL,
                     'legacy_limited', ?)""",
        (memory.id, now),
    )
    await db.db.commit()
    legacy_unit = EvidenceUnit(
        id="legacy-unit",
        source_id="source-1",
        doc_id="doc-1",
        doc_revision_id="unitrev-missing-history",
        source_type="github_repo",
        source_anchor="obs-body",
        source_lineage_id="unit-1",
        project_key=None,
        visibility="workspace",
        owner_user_id=None,
        repo_identifier=None,
        access_context_hash="workspace-access",
        content="",
        excerpt=None,
        evidence_provenance=EvidenceContentProvenance.LEGACY_LIMITED,
    )
    await db.upsert_evidence_unit(legacy_unit)
    reference = EvidenceReference(
        id="legacy-reference",
        evidence_unit_id=legacy_unit.id,
        role=EvidenceRole.PRIMARY,
        anchor=SourceAnchor(
            kind=AnchorKind.WHOLE_OBSERVATION,
            observation_id="obs-body",
            observation_revision_id="rev-body",
        ),
    )
    (reference,) = await db.record_evidence_references(legacy_unit.id, (reference,))
    await db.upsert_memory_support_assertion(
        MemorySupportAssertion(
            id="legacy-support",
            memory_id=memory.id,
            evidence_reference_id=reference.id or "",
            source_id="source-1",
            access_context_hash="workspace-access",
        )
    )
    await db.db.execute(
        """UPDATE system_contract_markers
              SET marker_value = 'evidence-unit-set-v2'
            WHERE marker_key = 'support_scope_version'"""
    )
    await db.db.commit()

    candidates = await db.list_legacy_support_recovery_candidates("source-1")

    assert len(candidates) == 1
    assert candidates[0].reason_codes == ("unit_revision_lineage_invalid",)
    assert candidates[0].projection.source_unit_revisions[0].id == "unitrev-current"
    assert candidates[0].memory.id == memory.id

    class RevalidationClient:
        async def revalidate_legacy_support(self, prompt: str, **kwargs):
            del kwargs
            assert "Production release requires approval" in prompt
            return LegacySupportRevalidationResponse(
                decisions=[
                    LegacySupportedRevalidationDecision(
                        request_position=0,
                        decision="supported",
                        primary_ref="f000002",
                        reason="current paragraph entails the claim",
                    )
                ]
            )

    prepared = await prepare_legacy_support_recovery(
        db,
        source_id="source-1",
        structured_llm_client=RevalidationClient(),
        llm_model="test-model",
    )
    assert prepared.report.ready_count == 1
    assert len(prepared.plans) == 1
    await db.apply_lifecycle_plan(prepared.plans[0])
    support = await db.get_memory_evidence_units(memory.id)
    assert any(not unit.legacy_limited for unit in support)
    assert any(unit.legacy_limited for unit in support)

    await db.db.execute(
        """UPDATE memory_unit_support_assertions
              SET active = 0, removed_at = ?
            WHERE memory_id = ?""",
        (now, memory.id),
    )
    await db.db.commit()
    second = await prepare_legacy_support_recovery(
        db,
        source_id="source-1",
        structured_llm_client=RevalidationClient(),
        llm_model="test-model",
    )
    assert second.report.id != prepared.report.id
    assert second.report.entries[0].reason == "previously_removed_v2_support"
    assert second.plans == ()
    attach = next(
        mutation
        for mutation in prepared.plans[0].mutations
        if mutation.mutation_type is LifecycleMutationType.ATTACH_SUPPORT
    )
    with pytest.raises(ValueError, match="cannot reactivate removed v2 Support"):
        await db._apply_lifecycle_mutation_unlocked(
            "guard-test-plan",
            attach,
            source_unit_id="unit-1",
            support_scope_version=SupportScopeVersion.EVIDENCE_UNIT_SET_V2,
            now=now,
        )
