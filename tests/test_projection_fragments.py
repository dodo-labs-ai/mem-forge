from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from pydantic import ValidationError

from memforge.llm.structured import (
    ProjectionFragmentMemoryCandidate,
    ProjectionFragmentMemoryExtractionResponse,
)
from memforge.agent_knowledge import (
    AgentKnowledgePatchModelResponse,
    AgentKnowledgePatchProposal,
)
from memforge.memory.evidence import EvidencePartKind, EvidenceRole
from memforge.pipeline.memory_extractor import MemoryExtractor
from memforge.pipeline.projection_context import (
    ProjectionExtractionBatch,
    plan_projection_extraction_batches,
)
from memforge.pipeline.projection_fragments import (
    FragmentSelectionError,
    FragmentSelectionErrorCode,
    compile_projection_fragment_catalog,
    resolve_projected_agent_claim_fragment,
)
from memforge.source_derivation import (
    memory_extraction_output_payload,
    memory_extraction_result_from_output_payload,
)
from memforge.storage.database import Database
from memforge.source_projection import (
    DeltaAxis,
    EvidenceCoordinateSpace,
    EvidenceRepresentationProfile,
    ProjectionCoverage,
    RevisionDelta,
    SourceObservation,
    SourceObservationRevision,
    SourceProjection,
    SourceUnit,
    SourceUnitRevision,
)
from memforge.source_representation import (
    BINARY_ARTIFACT_PROFILE,
    PLAIN_TEXT_PROFILE,
    representation_profile_for_observation_contract,
)


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


def _canonical_projection(
    *,
    observation_type: str,
    content: str,
) -> SourceProjection:
    profile = representation_profile_for_observation_contract(
        source_type="jira",
        observation_type=observation_type,
    )
    assert profile is not None
    revision = SourceObservationRevision(
        id=f"rev-{observation_type}",
        observation_id=f"obs-{observation_type}",
        semantic_hash=hashlib.sha256(content.encode()).hexdigest(),
        content=content,
        evidence_profile=profile,
    )
    unit = SourceUnit(
        id="unit-canonical",
        source_id="source-jira",
        unit_type="jira_issue",
        provider_key="SFPAY-182601",
    )
    unit_revision = SourceUnitRevision(
        id="unit-revision-canonical",
        source_unit_id=unit.id,
        semantic_hash="unit-canonical-hash",
        observation_revision_ids=(revision.id,),
    )
    return SourceProjection(
        run_id="run-canonical",
        source_id="source-jira",
        source_type="jira",
        scope={},
        coverage=ProjectionCoverage.COMPLETE_SNAPSHOT,
        observations=(
            SourceObservation(
                id=revision.observation_id,
                source_id="source-jira",
                source_unit_id=unit.id,
                observation_type=observation_type,
                provider_key=f"provider-{observation_type}",
            ),
        ),
        observation_revisions=(revision,),
        source_units=(unit,),
        source_unit_revisions=(unit_revision,),
        relations=(),
        deltas=(
            RevisionDelta(
                source_unit_id=unit.id,
                previous_unit_revision_id=None,
                current_unit_revision_id=unit_revision.id,
                axes=frozenset({DeltaAxis.SEMANTIC}),
                coverage=ProjectionCoverage.COMPLETE_SNAPSHOT,
                added_observation_ids=(revision.observation_id,),
            ),
        ),
        checkpoint={},
    )


@pytest_asyncio.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "projection-fragments.db"))
    await database.connect()
    await database.upsert_source(
        id="source-1",
        type="github_repo",
        name="Repository",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="owner-1",
    )
    try:
        yield database
    finally:
        await database.close()


def _projection(
    *,
    primary_content: str = "# Release rule\n\nUse approval before release.\n",
    context_profile=MARKDOWN_PROFILE,
    context_content: str = "# Dependency\n\nApproval means two reviewers.\n",
    context_metadata=None,
) -> SourceProjection:
    revisions = (
        SourceObservationRevision(
            id="rev-primary",
            observation_id="obs-primary",
            semantic_hash=hashlib.sha256(primary_content.encode()).hexdigest(),
            content=primary_content,
            evidence_profile=MARKDOWN_PROFILE,
        ),
        SourceObservationRevision(
            id="rev-context",
            observation_id="obs-context",
            semantic_hash=hashlib.sha256(context_content.encode()).hexdigest(),
            content=context_content,
            metadata=context_metadata or {},
            evidence_profile=context_profile,
        ),
    )
    unit = SourceUnit(
        id="unit-1",
        source_id="source-1",
        unit_type="document",
        provider_key="doc-1",
    )
    unit_revision = SourceUnitRevision(
        id="unit-revision-1",
        source_unit_id=unit.id,
        semantic_hash="unit-hash",
        observation_revision_ids=tuple(item.id for item in revisions),
    )
    return SourceProjection(
        run_id="run-1",
        source_id="source-1",
        source_type="github_repo",
        scope={},
        coverage=ProjectionCoverage.COMPLETE_SNAPSHOT,
        observations=(
            SourceObservation(
                id="obs-primary",
                source_id="source-1",
                source_unit_id=unit.id,
                observation_type="document_body",
                provider_key="body",
            ),
            SourceObservation(
                id="obs-context",
                source_id="source-1",
                source_unit_id=unit.id,
                observation_type="document_context",
                provider_key="context",
            ),
        ),
        observation_revisions=revisions,
        source_units=(unit,),
        source_unit_revisions=(unit_revision,),
        relations=(),
        deltas=(
            RevisionDelta(
                source_unit_id=unit.id,
                previous_unit_revision_id=None,
                current_unit_revision_id=unit_revision.id,
                axes=frozenset({DeltaAxis.SEMANTIC}),
                coverage=ProjectionCoverage.COMPLETE_SNAPSHOT,
                added_observation_ids=("obs-primary", "obs-context"),
            ),
        ),
        checkpoint={},
    )


def _batch(projection: SourceProjection) -> ProjectionExtractionBatch:
    primary = projection.observation_revisions[0].content
    context = projection.observation_revisions[1].content
    return ProjectionExtractionBatch(
        id="batch-1",
        source_unit_id="unit-1",
        primary_image_bytes=0,
        primary_observation_ids=("obs-primary",),
        primary_content_by_observation_id=(("obs-primary", primary),),
        context_observation_ids=("obs-context",),
        context_observation_ids_by_primary=(("obs-primary", ("obs-context",)),),
        primary_markdown=primary,
        context_markdown=context,
        primary_authority_spans=(("obs-primary", 0, primary),),
    )


def test_v9_candidate_rejects_legacy_authority_and_duplicate_refs() -> None:
    with pytest.raises(ValidationError):
        ProjectionFragmentMemoryCandidate.model_validate(
            {
                "content": "Approval is required.",
                "memory_type": "fact",
                "primary_ref": "f000001",
                "required_refs": [],
                "evidence_quote": "Approval is required.",
            }
        )


def test_agent_patch_model_uses_one_primary_event_and_reads_one_legacy_id() -> None:
    response = AgentKnowledgePatchModelResponse(
        action="create_new_concept",
        primary_event_id="E1",
        required_event_ids=["E2"],
    )
    assert response.primary_event_id == "E1"
    assert response.required_event_ids == ["E2"]

    legacy = AgentKnowledgePatchProposal.model_validate(
        {
            "action": "create_new_concept",
            "primary_evidence_ids": ["E1"],
        }
    )
    assert legacy.primary_event_id == "E1"
    with pytest.raises(ValidationError):
        AgentKnowledgePatchProposal.model_validate(
            {
                "action": "create_new_concept",
                "primary_evidence_ids": ["E1", "E2"],
            }
        )
    with pytest.raises(ValidationError):
        AgentKnowledgePatchModelResponse(action="create_new_concept")
    with pytest.raises(ValidationError):
        ProjectionFragmentMemoryCandidate.model_validate(
            {
                "content": "Approval is required.",
                "memory_type": "fact",
                "primary_ref": "f000001",
                "required_refs": ["f000002", "f000002"],
            }
        )
    with pytest.raises(ValidationError):
        ProjectionFragmentMemoryCandidate.model_validate(
            {
                "content": "Approval is required.",
                "memory_type": "fact",
                "primary_ref": "f000001",
                "required_refs": ["f000001"],
            }
        )


def test_catalog_resolves_one_primary_and_canonical_required_order() -> None:
    projection = _projection()
    catalog = compile_projection_fragment_catalog(
        projection,
        _batch(projection),
        access_context_hash="access-1",
    )
    replay = compile_projection_fragment_catalog(
        projection,
        _batch(projection),
        access_context_hash="access-1",
    )
    assert catalog.usable
    assert replay.digest == catalog.digest
    assert replay.model_payload() == catalog.model_payload()

    primary = next(
        item
        for item in catalog.fragments
        if item.anchor.observation_id == "obs-primary"
        and item.primary_eligible
        and "approval" in item.presentation_text.lower()
    )
    required = next(
        item
        for item in catalog.fragments
        if item.anchor.observation_id == "obs-context"
        and "reviewers" in item.presentation_text.lower()
    )
    assert required.primary_eligible is False

    selection = catalog.resolve_selection(
        primary_ref=primary.reference,
        required_refs=[required.reference],
    )
    assert selection.source_id == "source-1"
    assert selection.target_unit_revision_id == "unit-revision-1"
    assert [part.role for part in selection.parts] == [
        EvidenceRole.PRIMARY,
        EvidenceRole.REQUIRED,
    ]
    assert all(part.kind is EvidencePartKind.TEXT for part in selection.parts)
    assert selection.parts[0].anchor.observation_revision_id == "rev-primary"
    assert selection.parts[1].anchor.observation_revision_id == "rev-context"


@pytest.mark.parametrize("observation_type", ("comment", "changelog"))
def test_large_canonical_observation_compiles_from_whole_authority(
    observation_type: str,
) -> None:
    content = (
        json.dumps(
            {
                "body": "\n\n".join(
                    f"Decision paragraph {index}: retain exact Jira authority."
                    for index in range(700)
                )
            },
            separators=(",", ":"),
        )
        if observation_type == "comment"
        else json.dumps(
            {
                "field": "description",
                "from": "old",
                "to": "x" * 31_000,
            },
            separators=(",", ":"),
        )
    )
    assert len(content) > 30_000
    projection = _canonical_projection(
        observation_type=observation_type,
        content=content,
    )

    [batch] = plan_projection_extraction_batches(
        projection,
        max_primary_chars=30_000,
    )
    catalog = compile_projection_fragment_catalog(
        projection,
        batch,
        access_context_hash="access-canonical",
    )

    assert batch.primary_authority_spans == (
        (f"obs-{observation_type}", 0, content),
    )
    assert catalog.usable
    assert catalog.fragments
    assert all(fragment.primary_eligible for fragment in catalog.fragments)
    assert not any(
        error.code.value == "invalid_authority_range"
        for error in catalog.errors
    )


def test_canonical_nested_markdown_preserves_escaped_raw_json_ranges() -> None:
    body = 'Decision: keep "quoted" values and C:\\temp.\n\nUnicode: 雪.'
    content = json.dumps({"body": body}, ensure_ascii=True, separators=(",", ":"))
    projection = _canonical_projection(observation_type="comment", content=content)

    [batch] = plan_projection_extraction_batches(projection)
    catalog = compile_projection_fragment_catalog(
        projection,
        batch,
        access_context_hash="access-escaped",
    )

    assert catalog.usable
    assert [fragment.presentation_text for fragment in catalog.fragments] == [
        'Decision: keep "quoted" values and C:\\temp.',
        "Unicode: 雪.",
    ]
    for fragment in catalog.fragments:
        assert fragment.anchor.range_start is not None
        assert fragment.anchor.range_end is not None
        raw = content[
            fragment.anchor.range_start : fragment.anchor.range_end
        ]
        assert json.loads(f'"{raw}"') == fragment.presentation_text


def test_representation_policy_keeps_binary_whole_and_plain_text_range_addressable() -> None:
    content = "paragraph text\n\n" * 2_500
    base = _canonical_projection(observation_type="comment", content=content)
    [observation] = base.observations
    [revision] = base.observation_revisions

    plain = replace(
        base,
        observation_revisions=(
            replace(revision, evidence_profile=PLAIN_TEXT_PROFILE),
        ),
    )
    plain_batches = plan_projection_extraction_batches(
        plain,
        max_primary_chars=5_000,
        primary_overlap_chars=0,
    )
    assert len(plain_batches) > 1
    assert all(
        len(span_text) < len(content)
        for batch in plain_batches
        for _, _, span_text in batch.primary_authority_spans
    )

    binary = replace(
        base,
        observations=(replace(observation, observation_type="binary_artifact"),),
        observation_revisions=(
            replace(
                revision,
                evidence_profile=BINARY_ARTIFACT_PROFILE,
                metadata={
                    "source_artifact": {
                        "inference_eligible": True,
                        "sha256": "a" * 64,
                        "media_type": "application/pdf",
                        "size_bytes": 1,
                    }
                },
            ),
        ),
    )
    [binary_batch] = plan_projection_extraction_batches(
        binary,
        max_primary_chars=5_000,
    )
    assert binary_batch.primary_authority_spans == (
        (observation.id, 0, content),
    )
    binary_catalog = compile_projection_fragment_catalog(
        binary,
        binary_batch,
        access_context_hash="access-binary",
        supplied_artifact_observation_ids=(observation.id,),
    )
    assert binary_catalog.usable
    assert [
        fragment.anchor.kind.value for fragment in binary_catalog.fragments
    ] == ["whole_observation"]


def test_large_canonical_fragment_fails_with_capacity_error_without_raw_slicing() -> None:
    content = json.dumps(
        {"field": "description", "to": "x" * 2_000},
        separators=(",", ":"),
    )
    projection = _canonical_projection(
        observation_type="changelog",
        content=content,
    )

    [batch] = plan_projection_extraction_batches(
        projection,
        max_primary_chars=200,
    )
    catalog = compile_projection_fragment_catalog(
        projection,
        batch,
        access_context_hash="access-capacity",
        max_presentation_chars=1_000,
    )

    assert batch.primary_authority_spans == (("obs-changelog", 0, content),)
    assert not catalog.usable
    assert catalog.fragments == ()
    assert {
        error.code.value for error in catalog.errors if error.fatal
    } == {"catalog_too_large"}


def test_catalog_rejects_unknown_duplicate_and_primary_from_required_only() -> None:
    projection = _projection()
    catalog = compile_projection_fragment_catalog(
        projection,
        _batch(projection),
        access_context_hash="access-1",
    )
    primary = next(
        item for item in catalog.fragments if item.anchor.observation_id == "obs-primary"
    )
    required_only = next(
        item for item in catalog.fragments if item.anchor.observation_id == "obs-context"
    )

    with pytest.raises(FragmentSelectionError) as unknown:
        catalog.resolve_selection(primary_ref="f999999")
    assert unknown.value.code is FragmentSelectionErrorCode.UNKNOWN_REF

    with pytest.raises(FragmentSelectionError) as duplicate:
        catalog.resolve_selection(
            primary_ref=primary.reference,
            required_refs=[primary.reference],
        )
    assert duplicate.value.code is FragmentSelectionErrorCode.DUPLICATE_REF

    with pytest.raises(FragmentSelectionError) as ineligible:
        catalog.resolve_selection(primary_ref=required_only.reference)
    assert ineligible.value.code is FragmentSelectionErrorCode.INELIGIBLE_ROLE


def test_selection_fingerprint_distinguishes_refs_without_exposing_them() -> None:
    projection = _projection()
    catalog = compile_projection_fragment_catalog(
        projection,
        _batch(projection),
        access_context_hash="access-1",
    )
    primary = next(fragment for fragment in catalog.fragments if fragment.primary_eligible)
    context = next(
        fragment for fragment in catalog.fragments if not fragment.primary_eligible
    )
    content_hash = hashlib.sha256(b"Release requires approval.").hexdigest()

    accepted = catalog.selection_fingerprint(
        candidate_content_hash=content_hash,
        primary_ref=primary.reference,
        required_refs=[context.reference],
    )
    invalid_role = catalog.selection_fingerprint(
        candidate_content_hash=content_hash,
        primary_ref=context.reference,
        required_refs=[primary.reference],
    )

    assert accepted != invalid_role
    assert len(accepted) == 64
    assert primary.reference not in accepted
    assert context.reference not in accepted


def test_bounded_context_is_required_selectable_but_never_primary_eligible() -> None:
    projection = _projection()
    projection = replace(
        projection,
        deltas=(
            replace(
                projection.deltas[0],
                added_observation_ids=("obs-primary",),
            ),
        ),
    )
    [batch] = plan_projection_extraction_batches(projection)

    assert batch.primary_observation_ids == ("obs-primary",)
    assert batch.context_observation_ids == ("obs-context",)

    catalog = compile_projection_fragment_catalog(
        projection,
        batch,
        access_context_hash="access-1",
    )
    payload_by_observation = {
        fragment.anchor.observation_id: payload
        for fragment, payload in zip(
            catalog.fragments,
            catalog.model_payload(),
            strict=True,
        )
    }

    assert payload_by_observation["obs-primary"]["primary_eligible"] is True
    assert payload_by_observation["obs-context"]["primary_eligible"] is False
    assert all("eligible_roles" not in payload for payload in catalog.model_payload())

    primary = next(
        fragment
        for fragment in catalog.fragments
        if fragment.anchor.observation_id == "obs-primary"
    )
    context = next(
        fragment
        for fragment in catalog.fragments
        if fragment.anchor.observation_id == "obs-context"
    )
    selection = catalog.resolve_selection(
        primary_ref=primary.reference,
        required_refs=[context.reference],
    )
    assert [part.role for part in selection.parts] == [
        EvidenceRole.PRIMARY,
        EvidenceRole.REQUIRED,
    ]

    with pytest.raises(FragmentSelectionError) as ineligible:
        catalog.resolve_selection(primary_ref=context.reference)
    assert ineligible.value.code is FragmentSelectionErrorCode.INELIGIBLE_ROLE


def test_truncated_context_is_display_only_and_not_selectable() -> None:
    projection = _projection(context_content="Context that does not fit. " * 20)
    projection = replace(
        projection,
        deltas=(
            replace(
                projection.deltas[0],
                added_observation_ids=("obs-primary",),
            ),
        ),
    )
    [batch] = plan_projection_extraction_batches(
        projection,
        max_context_chars=80,
    )

    assert batch.context_observation_ids == ("obs-context",)
    assert batch.candidate_context_observation_ids == ()

    catalog = compile_projection_fragment_catalog(
        projection,
        batch,
        access_context_hash="access-1",
    )
    assert {
        fragment.anchor.observation_id for fragment in catalog.fragments
    } == {"obs-primary"}


def test_agent_event_receipt_maps_authority_to_one_projected_markdown_fragment() -> None:
    projection = _projection()
    selection, receipt = resolve_projected_agent_claim_fragment(
        projection,
        claim_text="Use approval before release.",
        access_context_hash="access-1",
        primary_event_id="E1",
        required_event_ids=("E2",),
    )
    assert len(selection.parts) == 1
    assert selection.parts[0].role is EvidenceRole.PRIMARY
    assert receipt.claim_anchor == selection.parts[0].anchor
    assert [(item.event_id, item.role) for item in receipt.event_ranges] == [
        ("E1", EvidenceRole.PRIMARY),
        ("E2", EvidenceRole.REQUIRED),
    ]
    assert receipt.to_payload()["catalog_digest"] == selection.catalog_digest


def test_agent_claim_spanning_structural_blocks_uses_one_primary_required_unit() -> None:
    claim = (
        "MemForge Cloud uses two database tiers.\n\n"
        "**Control plane**\n\n"
        "- Uses native runtime DDL.\n"
        "- Does not use the retired HDI deployer.\n\n"
        "**Workspace plane**\n\n"
        "Uses app-managed runtime migrations."
    )
    projection = _projection(primary_content=f"# Database architecture\n\n{claim}\n")

    selection, receipt = resolve_projected_agent_claim_fragment(
        projection,
        claim_text=claim,
        access_context_hash="access-1",
        primary_event_id="E1",
    )

    assert len(selection.parts) > 1
    assert selection.parts[0].role is EvidenceRole.PRIMARY
    assert all(part.role is EvidenceRole.REQUIRED for part in selection.parts[1:])
    assert {part.anchor.observation_revision_id for part in selection.parts} == {
        "rev-primary"
    }
    assert receipt.claim_anchor.range_start == projection.observation_revisions[
        0
    ].content.index(claim)
    assert receipt.claim_anchor.range_end == receipt.claim_anchor.range_start + len(
        claim
    )


def test_agent_claim_fragment_set_rejects_uncovered_markdown_content() -> None:
    claim = "First paragraph.\n\n---\n\nSecond paragraph."
    projection = _projection(primary_content=f"# Rule\n\n{claim}\n")

    with pytest.raises(ValueError, match="content gap"):
        resolve_projected_agent_claim_fragment(
            projection,
            claim_text=claim,
            access_context_hash="access-1",
        )


def test_missing_profile_makes_complete_catalog_unusable_without_widening() -> None:
    projection = _projection(context_profile=None)
    catalog = compile_projection_fragment_catalog(
        projection,
        _batch(projection),
        access_context_hash="access-1",
    )
    assert not catalog.usable
    assert catalog.fragments == ()
    assert any(error.fatal for error in catalog.errors)
    with pytest.raises(FragmentSelectionError) as rejected:
        catalog.resolve_selection(primary_ref="f000001")
    assert rejected.value.code is FragmentSelectionErrorCode.CATALOG_UNUSABLE


def test_inspected_artifact_uses_same_ref_shape_as_text_required() -> None:
    artifact_digest = "a" * 64
    projection = _projection(
        context_profile=BINARY_PROFILE,
        context_content="",
        context_metadata={
            "source_artifact": {
                "inference_eligible": True,
                "sha256": artifact_digest,
                "media_type": "image/png",
                "size_bytes": 128,
                "filename": "diagram.png",
            }
        },
    )
    catalog = compile_projection_fragment_catalog(
        projection,
        _batch(projection),
        access_context_hash="access-1",
        supplied_artifact_observation_ids=("obs-context",),
    )
    primary = next(
        item
        for item in catalog.fragments
        if item.kind.value == "text" and item.primary_eligible
    )
    artifact = next(item for item in catalog.fragments if item.kind.value == "artifact")
    assert artifact.primary_eligible is False
    artifact_payload = next(
        item for item in catalog.model_payload() if item["kind"] == "artifact"
    )
    assert artifact_payload["ref"] == artifact.reference
    assert artifact_payload["image_source_observation_id"] == "obs-context"
    assert "diagram.png" not in str(artifact_payload)

    selection = catalog.resolve_selection(
        primary_ref=primary.reference,
        required_refs=[artifact.reference],
    )
    assert selection.parts[1].kind is EvidencePartKind.ARTIFACT
    assert selection.parts[1].raw_content_sha256 == artifact_digest
    assert selection.parts[1].artifact_metadata["media_type"] == "image/png"


def test_supplied_fieldless_legacy_artifact_uses_normalized_eligibility() -> None:
    artifact_digest = "b" * 64
    projection = _projection(
        context_profile=BINARY_PROFILE,
        context_content="",
        context_metadata={
            "source_artifact": {
                "sha256": artifact_digest,
                "media_type": "image/png",
                "size_bytes": 128,
                "filename": "legacy-diagram.png",
            }
        },
    )

    catalog = compile_projection_fragment_catalog(
        projection,
        _batch(projection),
        access_context_hash="access-1",
        supplied_artifact_observation_ids=("obs-context",),
    )

    assert catalog.usable
    artifact = next(item for item in catalog.fragments if item.kind.value == "artifact")
    assert artifact.raw_content_sha256 == artifact_digest


@pytest.mark.asyncio
async def test_extractor_persists_only_resolved_parts_and_never_falls_back() -> None:
    projection = _projection()
    batch = _batch(projection)
    catalog = compile_projection_fragment_catalog(
        projection,
        batch,
        access_context_hash="access-1",
    )
    primary = next(
        item
        for item in catalog.fragments
        if item.anchor.observation_id == "obs-primary"
        and item.primary_eligible
        and "approval" in item.presentation_text.lower()
    )
    required = next(
        item
        for item in catalog.fragments
        if item.anchor.observation_id == "obs-context"
        and "reviewers" in item.presentation_text.lower()
    )

    class Client:
        async def extract_projection_fragment_memories(self, prompt: str, **kwargs):
            assert "evidence_quote" not in prompt
            return ProjectionFragmentMemoryExtractionResponse(
                memories=[
                    ProjectionFragmentMemoryCandidate(
                        content="Release requires approval by two reviewers.",
                        memory_type="convention",
                        primary_ref=primary.reference,
                        required_refs=[required.reference],
                    )
                ]
            )

    result = await MemoryExtractor(
        structured_llm_client=Client(),
    ).extract_projection_fragment_memories(
        catalog,
        source_type="github_repo",
        context_markdown=batch.context_markdown,
    )
    assert result.error_type is None
    assert len(result.memories) == 1
    memory = result.memories[0]
    assert memory.evidence_quote is None
    assert memory.evidence_block_id is None
    assert memory.resolved_evidence_selection is not None
    assert len(memory.resolved_evidence_selection.parts) == 2
    assert "evidence_block_fallback_samples" not in result.metadata

    restored = memory_extraction_result_from_output_payload(
        memory_extraction_output_payload(result)
    )
    assert restored.memories[0].resolved_evidence_selection == (
        memory.resolved_evidence_selection
    )


@pytest.mark.asyncio
async def test_contract_cutover_supersedes_only_incomplete_v8_derivations(db) -> None:
    now = datetime.now(timezone.utc).isoformat()
    for derivation_id, status in (
        ("sdrv-pending", "pending"),
        ("sdrv-retry", "retryable_failure"),
        ("sdrv-complete", "completed"),
    ):
        await db.db.execute(
            """INSERT INTO source_derivation_attempts (
                   id, source_id, source_unit_id, target_unit_revision_id,
                   projection_payload_json, projection_payload_hash,
                   projection_identity_hash, context_payload_json,
                   context_payload_hash, context_identity_hash,
                   extraction_contract_version, status, created_at, updated_at
               ) VALUES (?, 'source-1', 'unit-1', 'unitrev-1',
                         '{}', 'projection-hash', 'projection-identity',
                         '{}', 'context-hash', 'context-identity',
                         'projection-extraction-v8', ?, ?, ?)""",
            (derivation_id, status, now, now),
        )
    await db.db.commit()

    superseded = await db.supersede_incomplete_source_derivations_for_contract(
        extraction_contract_version="projection-extraction-v8",
    )
    assert superseded == ("sdrv-pending", "sdrv-retry")
    rows = await db.db.execute_fetchall(
        """SELECT id, status, terminal_reason_code
           FROM source_derivation_attempts ORDER BY id"""
    )
    by_id = {str(row["id"]): row for row in rows}
    assert by_id["sdrv-pending"]["status"] == "superseded"
    assert by_id["sdrv-retry"]["status"] == "superseded"
    assert by_id["sdrv-pending"]["terminal_reason_code"] == "CONTRACT_SUPERSEDED"
    assert by_id["sdrv-complete"]["status"] == "completed"
    assert by_id["sdrv-complete"]["terminal_reason_code"] is None
