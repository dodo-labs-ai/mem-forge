from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from pydantic import ValidationError

from memforge.evals.agent_evaluation import QualitySignalCollector, quality_signal_scope
from memforge.llm.structured import (
    ProjectionFragmentMemoryCandidate,
    ProjectionFragmentMemoryExtractionResponse,
)
from memforge.agent_knowledge import (
    AgentKnowledgePatchModelResponse,
    AgentKnowledgePatchProposal,
)
from memforge.memory.evidence import (
    ActiveSupportEvidence,
    EvidencePartKind,
    EvidenceRole,
)
from memforge.models import DocumentRecord
from memforge.pipeline.memory_extractor import MemoryExtractor
from memforge.pipeline.extraction_contract import (
    PROJECTION_EXTRACTION_V8,
    PROJECTION_EXTRACTION_V9,
)
from memforge.pipeline.projection_context import (
    ProjectionExtractionBatch,
    plan_projection_extraction_batches,
)
from memforge.pipeline.projection_fragments import (
    FragmentSelectionError,
    FragmentSelectionErrorCode,
    RevalidatedSelectionError,
    RevalidatedSelectionErrorCode,
    SupportRevalidationLimitation,
    SupportRevalidationLimitationCode,
    compile_projection_fragment_catalog,
    prepare_support_revalidation_workset,
    resolve_revalidated_noop_selection,
    resolve_projected_agent_claim_fragment,
)
from memforge.source_derivation import (
    SourceUnitDerivationContext,
    memory_extraction_output_payload,
    memory_extraction_result_from_output_payload,
    source_derivation_manifest,
)
from memforge.storage.database import Database
from memforge.source_projection import (
    AnchorKind,
    DeltaAxis,
    EvidenceCoordinateSpace,
    EvidenceRepresentationProfile,
    FragmentMapping,
    ProjectionCoverage,
    RevisionDelta,
    SourceObservation,
    SourceObservationRevision,
    SourceProjection,
    SourceAnchor,
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


def test_v9_candidate_rejects_legacy_authority() -> None:
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


def test_v9_response_accepts_redundant_selectors_for_admission_normalization() -> None:
    payload = {
        "memories": [
            {
                "content": "Approval is required.",
                "memory_type": "fact",
                "primary_ref": "p000001",
                "required_refs": [
                    "r000003",
                    "p000001",
                    "p000002",
                    "r000003",
                    "p000002",
                ],
            }
        ]
    }

    response = ProjectionFragmentMemoryExtractionResponse.model_validate(payload)

    assert response.memories[0].primary_ref == "p000001"
    assert response.memories[0].required_refs == payload["memories"][0]["required_refs"]
    reconstructed = ProjectionFragmentMemoryExtractionResponse.model_validate_json(
        response.model_dump_json()
    )
    assert reconstructed == response


def test_v9_well_formed_response_remains_byte_stable() -> None:
    payload = {
        "memories": [
            {
                "content": "Approval is required.",
                "memory_type": "fact",
                "confidence": 0.7,
                "entity_refs": [],
                "valid_from": None,
                "valid_until": None,
                "primary_ref": "p000001",
                "required_refs": ["r000002"],
            }
        ]
    }

    response = ProjectionFragmentMemoryExtractionResponse.model_validate(payload)

    assert response.model_dump() == payload


def test_noop_revalidation_presents_only_the_existing_evidence_unit_from_a_large_revision() -> None:
    filler = tuple(
        f"Unrelated historical paragraph {index}: " + ("x" * 180)
        for index in range(700)
    )
    claim = "Approval requires two reviewers."
    content = "\n\n".join((*filler[:350], "<div></div>", claim, *filler[350:]))
    assert len(content) > 120_000
    projection = _projection(primary_content=content)
    current_revision = projection.observation_revisions[0]
    claim_start = content.index(claim)
    claim_end = claim_start + len(claim)
    support = ActiveSupportEvidence(
        memory_id="mem-approval",
        source_id=projection.source_id,
        reference_id="ref-primary",
        evidence_unit_id="eu-approval",
        role=EvidenceRole.PRIMARY,
        anchor=SourceAnchor(
            kind=AnchorKind.REVISION_RANGE,
            observation_id=current_revision.observation_id,
            observation_revision_id="rev-primary-previous",
            range_start=claim_start,
            range_end=claim_end,
        ),
        excerpt=claim,
    )

    selection = resolve_revalidated_noop_selection(
        projection,
        support=(support,),
        access_context_hash="access-1",
        current_primary_quote=claim,
        current_required_quotes_by_reference_id={},
    )

    assert len(selection.parts) == 1
    [primary] = selection.parts
    assert primary.role is EvidenceRole.PRIMARY
    assert primary.excerpt == claim
    assert primary.anchor == SourceAnchor(
        kind=AnchorKind.REVISION_RANGE,
        observation_id=current_revision.observation_id,
        observation_revision_id=current_revision.id,
        range_start=claim_start,
        range_end=claim_end,
    )


def test_support_revalidation_prefers_persisted_digest_over_stale_range_overlap() -> None:
    stale = "Approval requires one reviewer."
    current = "Approval requires two reviewers."
    content = f"{stale}\n\n{current}"
    base = _projection(primary_content=content)
    revision, context_revision = base.observation_revisions
    projection = replace(
        base,
        observation_revisions=(
            replace(revision, evidence_profile=PLAIN_TEXT_PROFILE),
            context_revision,
        ),
    )
    digest = hashlib.sha256(current.encode()).hexdigest()
    support = ActiveSupportEvidence(
        memory_id="mem-approval",
        source_id=projection.source_id,
        reference_id="ref-primary",
        evidence_unit_id="eu-approval",
        role=EvidenceRole.PRIMARY,
        anchor=SourceAnchor(
            kind=AnchorKind.REVISION_RANGE,
            observation_id=revision.observation_id,
            observation_revision_id="rev-primary-previous",
            range_start=0,
            range_end=len(stale),
        ),
        excerpt=None,
        raw_content_sha256=digest,
        presentation_sha256=digest,
    )

    workset = prepare_support_revalidation_workset(
        projection,
        support=(support,),
        required_selector_by_reference_id={},
        revision_indexes_by_id={},
        memory_claim="Two-person approval remains mandatory.",
    )

    assert len(workset.primary_refs) == 1
    selected = workset.fragments_by_ref[workset.primary_refs[0]]
    assert selected.presentation_text == current
    assert selected.anchor.range_start == content.index(current)


def test_stable_fragment_revalidation_requires_provider_correspondence() -> None:
    current = "Approval requires two reviewers."
    base = _projection(primary_content=current)
    revision, context_revision = base.observation_revisions
    [delta] = base.deltas
    projection = replace(
        base,
        observation_revisions=(
            replace(revision, evidence_profile=PLAIN_TEXT_PROFILE),
            context_revision,
        ),
        deltas=(
            replace(
                delta,
                previous_unit_revision_id="unit-revision-previous",
                added_observation_ids=(),
                fragment_mappings=(
                    FragmentMapping(
                        observation_id=revision.observation_id,
                        previous_revision_id="rev-primary-previous",
                        current_revision_id=revision.id,
                        previous_fragment_id="approval-old",
                        current_fragment_id="approval-current",
                    ),
                ),
            ),
        ),
    )
    digest = hashlib.sha256(current.encode()).hexdigest()
    support = ActiveSupportEvidence(
        memory_id="mem-approval",
        source_id=projection.source_id,
        reference_id="ref-primary",
        evidence_unit_id="eu-approval",
        role=EvidenceRole.PRIMARY,
        anchor=SourceAnchor(
            kind=AnchorKind.STABLE_FRAGMENT,
            observation_id=revision.observation_id,
            observation_revision_id="rev-primary-previous",
            fragment_id="approval-old",
        ),
        excerpt=current,
        raw_content_sha256=digest,
        presentation_sha256=digest,
    )

    workset = prepare_support_revalidation_workset(
        projection,
        support=(support,),
        required_selector_by_reference_id={},
        revision_indexes_by_id={},
        memory_claim=current,
    )
    model_selection = workset.resolve_model_selection(
        primary_ref=workset.primary_refs[0],
        required_refs_by_selector={},
    )
    resolved = resolve_revalidated_noop_selection(
        projection,
        support=(support,),
        access_context_hash="access-1",
        current_primary_quote=current,
        current_required_quotes_by_reference_id={},
        selected_fragments_by_reference_id=(
            model_selection.fragments_by_evidence_reference_id
        ),
    )
    assert resolved.parts[0].anchor.kind is AnchorKind.REVISION_RANGE
    assert resolved.parts[0].anchor.observation_revision_id == revision.id

    without_mapping = replace(
        projection,
        deltas=(replace(projection.deltas[0], fragment_mappings=()),),
    )
    with pytest.raises(RevalidatedSelectionError) as exc_info:
        prepare_support_revalidation_workset(
            without_mapping,
            support=(support,),
            required_selector_by_reference_id={},
            revision_indexes_by_id={},
            memory_claim=current,
        )
    assert exc_info.value.code is RevalidatedSelectionErrorCode.UNPRESENTABLE


def test_support_revalidation_preserves_more_than_32_required_parts() -> None:
    paragraphs = tuple(
        ["Approval requires two reviewers."]
        + [f"Required policy condition {index}." for index in range(1, 34)]
    )
    content = "\n\n".join(paragraphs)
    base = _projection(primary_content=content)
    revision, context_revision = base.observation_revisions
    projection = replace(
        base,
        observation_revisions=(
            replace(revision, evidence_profile=PLAIN_TEXT_PROFILE),
            context_revision,
        ),
    )
    support = []
    offset = 0
    for index, paragraph in enumerate(paragraphs):
        digest = hashlib.sha256(paragraph.encode()).hexdigest()
        support.append(
            ActiveSupportEvidence(
                memory_id="mem-complete-policy",
                source_id=projection.source_id,
                reference_id=f"ref-{index:02d}",
                evidence_unit_id="eu-complete-policy",
                role=EvidenceRole.PRIMARY if index == 0 else EvidenceRole.REQUIRED,
                anchor=SourceAnchor(
                    kind=AnchorKind.REVISION_RANGE,
                    observation_id=revision.observation_id,
                    observation_revision_id="rev-primary-previous",
                    range_start=offset,
                    range_end=offset + len(paragraph),
                ),
                excerpt=paragraph,
                raw_content_sha256=digest,
                presentation_sha256=digest,
            )
        )
        offset += len(paragraph) + 2
    selectors = {
        item.reference_id: f"r{index:06d}"
        for index, item in enumerate(support[1:], start=1)
    }

    workset = prepare_support_revalidation_workset(
        projection,
        support=tuple(support),
        required_selector_by_reference_id=selectors,
        revision_indexes_by_id={},
        memory_claim=paragraphs[0],
    )
    selection = workset.resolve_model_selection(
        primary_ref=workset.primary_refs[0],
        required_refs_by_selector={
            selector: references[0]
            for selector, references in workset.required_refs_by_selector.items()
        },
    )

    assert len(workset.required_refs_by_selector) == 33
    assert workset.model_max_output_tokens() > 512
    assert len(selection.fragments_by_evidence_reference_id) == 34


def test_noop_revalidation_keeps_the_presentation_limit_for_one_oversized_fragment() -> None:
    claim = "x" * 120_001
    projection = _projection(primary_content=claim)
    current_revision = projection.observation_revisions[0]
    support = ActiveSupportEvidence(
        memory_id="mem-oversized",
        source_id=projection.source_id,
        reference_id="ref-primary",
        evidence_unit_id="eu-oversized",
        role=EvidenceRole.PRIMARY,
        anchor=SourceAnchor(
            kind=AnchorKind.REVISION_RANGE,
            observation_id=current_revision.observation_id,
            observation_revision_id="rev-primary-previous",
            range_start=0,
            range_end=len(claim),
        ),
        excerpt=claim,
    )

    with pytest.raises(SupportRevalidationLimitation) as exc_info:
        resolve_revalidated_noop_selection(
            projection,
            support=(support,),
            access_context_hash="access-1",
            current_primary_quote=claim,
            current_required_quotes_by_reference_id={},
        )

    assert (
        exc_info.value.code
        is SupportRevalidationLimitationCode.CAPACITY_EXCEEDED
    )


def test_noop_revalidation_rejects_an_ambiguous_moved_fragment() -> None:
    claim = "Approval requires two reviewers."
    content = f"{claim}\n\nUnrelated paragraph.\n\n{claim}"
    projection = _projection(primary_content=content)
    current_revision = projection.observation_revisions[0]
    support = ActiveSupportEvidence(
        memory_id="mem-ambiguous",
        source_id=projection.source_id,
        reference_id="ref-primary",
        evidence_unit_id="eu-ambiguous",
        role=EvidenceRole.PRIMARY,
        anchor=SourceAnchor(
            kind=AnchorKind.REVISION_RANGE,
            observation_id=current_revision.observation_id,
            observation_revision_id="rev-primary-previous",
            range_start=1,
            range_end=len(claim) + 1,
        ),
        excerpt=claim,
    )

    with pytest.raises(RevalidatedSelectionError) as exc_info:
        resolve_revalidated_noop_selection(
            projection,
            support=(support,),
            access_context_hash="access-1",
            current_primary_quote=claim,
            current_required_quotes_by_reference_id={},
        )

    assert exc_info.value.code is RevalidatedSelectionErrorCode.AMBIGUOUS


@pytest.mark.parametrize("json_text", [False, True])
def test_v9_response_accepts_redundant_stringified_and_json_text_fallback_shapes(
    json_text: bool,
) -> None:
    payload = {
        "memories": "[{\"content\":\"Approval is required.\","
        "\"memory_type\":\"fact\",\"primary_ref\":\"p000001\","
        "\"required_refs\":[\"r000002\",\"r000002\"]}]"
    }

    response = (
        ProjectionFragmentMemoryExtractionResponse.model_validate_json(
            json.dumps(payload)
        )
        if json_text
        else ProjectionFragmentMemoryExtractionResponse.model_validate(payload)
    )

    assert response.memories[0].required_refs == ["r000002", "r000002"]


@pytest.mark.parametrize(
    ("primary_ref", "required_refs"),
    [
        ("not-a-fragment", []),
        ("p000001", ["not-a-fragment"]),
    ],
)
def test_v9_response_still_rejects_malformed_fragment_refs(
    primary_ref: str,
    required_refs: list[str],
) -> None:
    with pytest.raises(ValidationError):
        ProjectionFragmentMemoryExtractionResponse.model_validate(
            {
                "memories": [
                    {
                        "content": "Approval is required.",
                        "memory_type": "fact",
                        "primary_ref": primary_ref,
                        "required_refs": required_refs,
                    }
                ]
            }
        )


def test_v9_schema_rejects_required_only_ref_as_primary() -> None:
    with pytest.raises(ValidationError):
        ProjectionFragmentMemoryExtractionResponse.model_validate(
            {
                "memories": [
                    {
                        "content": "Historical context cannot authorize a new claim.",
                        "memory_type": "fact",
                        "primary_ref": "r000004",
                        "required_refs": [],
                    }
                ]
            }
        )

    accepted = ProjectionFragmentMemoryExtractionResponse.model_validate(
        {
            "memories": [
                {
                    "content": "Current work authorizes the claim.",
                    "memory_type": "fact",
                    "primary_ref": "p000001",
                    "required_refs": ["p000002", "r000004"],
                }
            ]
        }
    )
    assert accepted.memories[0].primary_ref == "p000001"
    assert accepted.memories[0].required_refs == ["p000002", "r000004"]


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
    changed_capability = compile_projection_fragment_catalog(
        projection,
        _batch(projection),
        access_context_hash="access-1",
        inference_capability_hash="b" * 64,
    )
    assert catalog.usable
    assert replay.digest == catalog.digest
    assert replay.model_payload() == catalog.model_payload()
    assert changed_capability.digest != catalog.digest

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
        extraction_contract_version=PROJECTION_EXTRACTION_V9,
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


def test_legacy_v8_large_canonical_observation_keeps_bounded_text_segments() -> None:
    content = json.dumps(
        {"body": "Legacy projection prompt. " * 2_000},
        separators=(",", ":"),
    )
    assert len(content) > 30_000
    projection = _canonical_projection(observation_type="comment", content=content)

    batches = plan_projection_extraction_batches(
        projection,
        max_primary_chars=30_000,
        extraction_contract_version=PROJECTION_EXTRACTION_V8,
    )

    assert len(batches) > 1
    assert all(len(batch.primary_markdown) <= 30_000 for batch in batches)
    assert all(
        len(span_text) < len(content)
        for batch in batches
        for _, _, span_text in batch.primary_authority_spans
    )


@pytest.mark.parametrize("future_profile_kind", ("canonical", "artifact"))
def test_v9_unknown_whole_authority_profile_fails_in_compiler_not_planner(
    future_profile_kind: str,
) -> None:
    content = json.dumps(
        {"body": "Future representation content. " * 2_000},
        separators=(",", ":"),
    )
    base = _canonical_projection(observation_type="comment", content=content)
    [observation] = base.observations
    [revision] = base.observation_revisions
    if future_profile_kind == "canonical":
        assert revision.evidence_profile is not None
        future = replace(
            base,
            observation_revisions=(
                replace(
                    revision,
                    evidence_profile=replace(
                        revision.evidence_profile,
                        version=99,
                    ),
                ),
            ),
        )
        supplied_artifacts: tuple[str, ...] = ()
    else:
        future = replace(
            base,
            observations=(replace(observation, observation_type="binary_artifact"),),
            observation_revisions=(
                replace(
                    revision,
                    evidence_profile=EvidenceRepresentationProfile(
                        name="future-artifact",
                        version=7,
                        coordinate_space=EvidenceCoordinateSpace.WHOLE_ARTIFACT,
                    ),
                    metadata={
                        "source_artifact": {
                            "inference_eligible": True,
                            "sha256": "b" * 64,
                            "media_type": "application/pdf",
                            "size_bytes": 1,
                        }
                    },
                ),
            ),
        )
        supplied_artifacts = (observation.id,)

    [batch] = plan_projection_extraction_batches(
        future,
        max_primary_chars=5_000,
        extraction_contract_version=PROJECTION_EXTRACTION_V9,
    )
    catalog = compile_projection_fragment_catalog(
        future,
        batch,
        access_context_hash="access-future-profile",
        supplied_artifact_observation_ids=supplied_artifacts,
    )

    assert batch.primary_authority_spans == ((observation.id, 0, content),)
    assert not catalog.usable
    assert {
        error.code.value for error in catalog.errors if error.fatal
    } == {"unsupported_profile"}


def test_canonical_nested_markdown_preserves_escaped_raw_json_ranges() -> None:
    body = 'Decision: keep "quoted" values and C:\\temp.\n\nUnicode: 雪.'
    content = json.dumps({"body": body}, ensure_ascii=True, separators=(",", ":"))
    projection = _canonical_projection(observation_type="comment", content=content)

    [batch] = plan_projection_extraction_batches(
        projection,
        extraction_contract_version=PROJECTION_EXTRACTION_V9,
    )
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
        extraction_contract_version=PROJECTION_EXTRACTION_V9,
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
        extraction_contract_version=PROJECTION_EXTRACTION_V9,
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


def test_support_revalidation_workset_reuses_the_shared_representation_compilers() -> None:
    canonical = _canonical_projection(
        observation_type="comment",
        content=json.dumps(
            {"body": "Decision: retain A7 for regular payroll."},
            separators=(",", ":"),
        ),
    )
    [canonical_revision] = canonical.observation_revisions
    plain_base = _canonical_projection(
        observation_type="comment",
        content="Decision: retain A7 for regular payroll.",
    )
    [plain_revision] = plain_base.observation_revisions
    plain = replace(
        plain_base,
        observation_revisions=(replace(plain_revision, evidence_profile=PLAIN_TEXT_PROFILE),),
    )
    [binary_observation] = plain_base.observations
    binary = replace(
        plain_base,
        observations=(replace(binary_observation, observation_type="binary_artifact"),),
        observation_revisions=(
            replace(
                plain_revision,
                content="",
                evidence_profile=BINARY_ARTIFACT_PROFILE,
                metadata={
                    "source_artifact": {
                        "inference_eligible": True,
                        "sha256": "a" * 64,
                        "media_type": "image/png",
                        "size_bytes": 1,
                    }
                },
            ),
        ),
    )

    for projection in (canonical, plain, binary):
        [revision] = projection.observation_revisions
        support = ActiveSupportEvidence(
            memory_id="mem-a7",
            source_id=projection.source_id,
            reference_id="ref-primary",
            evidence_unit_id="eu-a7",
            role=EvidenceRole.PRIMARY,
            anchor=SourceAnchor(
                kind=(
                    AnchorKind.WHOLE_OBSERVATION
                    if revision.evidence_profile
                    and revision.evidence_profile.coordinate_space is EvidenceCoordinateSpace.WHOLE_ARTIFACT
                    else AnchorKind.REVISION_RANGE
                ),
                observation_id=revision.observation_id,
                observation_revision_id=f"{revision.id}-previous",
                range_start=(
                    None
                    if revision.evidence_profile
                    and revision.evidence_profile.coordinate_space is EvidenceCoordinateSpace.WHOLE_ARTIFACT
                    else 0
                ),
                range_end=(
                    None
                    if revision.evidence_profile
                    and revision.evidence_profile.coordinate_space is EvidenceCoordinateSpace.WHOLE_ARTIFACT
                    else len(revision.content)
                ),
            ),
            excerpt=(revision.content or None),
        )

        workset = prepare_support_revalidation_workset(
            projection,
            support=(support,),
            required_selector_by_reference_id={},
            revision_indexes_by_id={},
            memory_claim="A7 remains a regular-payroll decision.",
        )

        assert workset.primary_refs
        assert all(reference.startswith("f") for reference in workset.primary_refs)


@pytest.mark.parametrize(
    ("label", "profile", "primary_content", "required_content", "primary_text", "required_text"),
    (
        (
            "markdown",
            MARKDOWN_PROFILE,
            "# Decision\n\nPrimary approval rule.",
            "# Constraint\n\nRequired payroll scope.",
            "Primary approval rule.",
            "Required payroll scope.",
        ),
        (
            "plain",
            PLAIN_TEXT_PROFILE,
            "Primary approval rule.",
            "Required payroll scope.",
            "Primary approval rule.",
            "Required payroll scope.",
        ),
        (
            "canonical",
            representation_profile_for_observation_contract(
                source_type="jira",
                observation_type="comment",
            ),
            json.dumps({"body": "Primary approval rule."}, separators=(",", ":")),
            json.dumps({"body": "Required payroll scope."}, separators=(",", ":")),
            "Primary approval rule.",
            "Required payroll scope.",
        ),
    ),
)
def test_text_revalidation_profiles_preserve_exact_complete_unit_coordinates(
    label,
    profile,
    primary_content,
    required_content,
    primary_text,
    required_text,
) -> None:
    assert profile is not None
    base = _projection(
        primary_content=primary_content,
        context_content=required_content,
    )
    primary_revision, required_revision = base.observation_revisions
    projection = replace(
        base,
        run_id=f"run-{label}",
        observation_revisions=(
            replace(primary_revision, evidence_profile=profile),
            replace(required_revision, evidence_profile=profile),
        ),
    )
    support = (
        ActiveSupportEvidence(
            memory_id=f"mem-{label}",
            source_id=projection.source_id,
            reference_id="ref-primary",
            evidence_unit_id=f"eu-{label}",
            role=EvidenceRole.PRIMARY,
            anchor=SourceAnchor(
                kind=AnchorKind.REVISION_RANGE,
                observation_id=primary_revision.observation_id,
                observation_revision_id="rev-primary-previous",
                range_start=0,
                range_end=len(primary_content),
            ),
            excerpt=primary_text,
        ),
        ActiveSupportEvidence(
            memory_id=f"mem-{label}",
            source_id=projection.source_id,
            reference_id="ref-required",
            evidence_unit_id=f"eu-{label}",
            role=EvidenceRole.REQUIRED,
            anchor=SourceAnchor(
                kind=AnchorKind.REVISION_RANGE,
                observation_id=required_revision.observation_id,
                observation_revision_id="rev-required-previous",
                range_start=0,
                range_end=len(required_content),
            ),
            excerpt=required_text,
        ),
    )
    workset = prepare_support_revalidation_workset(
        projection,
        support=support,
        required_selector_by_reference_id={"ref-required": "r000001"},
        revision_indexes_by_id={},
        memory_claim=primary_text,
    )
    model_selection = workset.resolve_model_selection(
        primary_ref=workset.primary_refs[0],
        required_refs_by_selector={
            "r000001": workset.required_refs_by_selector["r000001"][0]
        },
    )
    resolved = resolve_revalidated_noop_selection(
        projection,
        support=support,
        access_context_hash=f"access-{label}",
        current_primary_quote=primary_text,
        current_required_quotes_by_reference_id={"ref-required": required_text},
        selected_fragments_by_reference_id=(
            model_selection.fragments_by_evidence_reference_id
        ),
    )

    assert [part.role for part in resolved.parts] == [
        EvidenceRole.PRIMARY,
        EvidenceRole.REQUIRED,
    ]
    assert {part.anchor.observation_revision_id for part in resolved.parts} == {
        primary_revision.id,
        required_revision.id,
    }
    assert all(part.anchor.kind is AnchorKind.REVISION_RANGE for part in resolved.parts)
    assert {part.excerpt for part in resolved.parts} == {primary_text, required_text}
    assert all(len(part.raw_content_sha256) == 64 for part in resolved.parts)


def test_binary_revalidation_preserves_exact_complete_unit_coordinates() -> None:
    base = _projection(primary_content="", context_content="")
    revisions = tuple(
        replace(
            revision,
            evidence_profile=BINARY_ARTIFACT_PROFILE,
            metadata={
                "source_artifact": {
                    "inference_eligible": True,
                    "sha256": character * 64,
                    "media_type": "image/png",
                    "size_bytes": 1,
                }
            },
        )
        for revision, character in zip(
            base.observation_revisions,
            ("a", "b"),
            strict=True,
        )
    )
    projection = replace(base, observation_revisions=revisions)
    support = tuple(
        ActiveSupportEvidence(
            memory_id="mem-binary",
            source_id=projection.source_id,
            reference_id=f"ref-{role.value}",
            evidence_unit_id="eu-binary",
            role=role,
            anchor=SourceAnchor(
                kind=AnchorKind.WHOLE_OBSERVATION,
                observation_id=revision.observation_id,
                observation_revision_id=f"{revision.id}-previous",
            ),
            excerpt=None,
            raw_content_sha256=character * 64,
            presentation_sha256=hashlib.sha256(b"").hexdigest(),
        )
        for revision, character, role in zip(
            revisions,
            ("a", "b"),
            (EvidenceRole.PRIMARY, EvidenceRole.REQUIRED),
            strict=True,
        )
    )
    workset = prepare_support_revalidation_workset(
        projection,
        support=support,
        required_selector_by_reference_id={"ref-required": "r000001"},
        revision_indexes_by_id={},
        memory_claim="The current artifacts jointly support the claim.",
    )
    model_selection = workset.resolve_model_selection(
        primary_ref=workset.primary_refs[0],
        required_refs_by_selector={
            "r000001": workset.required_refs_by_selector["r000001"][0]
        },
    )
    resolved = resolve_revalidated_noop_selection(
        projection,
        support=support,
        access_context_hash="access-binary",
        current_primary_quote="",
        current_required_quotes_by_reference_id={"ref-required": ""},
        selected_fragments_by_reference_id=(
            model_selection.fragments_by_evidence_reference_id
        ),
    )

    assert [part.role for part in resolved.parts] == [
        EvidenceRole.PRIMARY,
        EvidenceRole.REQUIRED,
    ]
    assert all(part.anchor.kind is AnchorKind.WHOLE_OBSERVATION for part in resolved.parts)
    assert all(part.excerpt is None for part in resolved.parts)


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
        extraction_contract_version=PROJECTION_EXTRACTION_V9,
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


def test_catalog_rejects_duplicate_unknown_and_ineligible_selectors() -> None:
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
        catalog.resolve_selection(primary_ref="p999999")
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
    model_payload = catalog.model_payload()
    payload_by_ref = {
        item["ref"]: item
        for group in model_payload.values()
        for item in group
    }
    payload_by_observation = {
        fragment.anchor.observation_id: payload_by_ref[fragment.reference]
        for fragment in catalog.fragments
    }

    assert payload_by_observation["obs-primary"]["ref"].startswith("p")
    assert payload_by_observation["obs-context"]["ref"].startswith("r")
    assert all(
        "eligible_roles" not in payload
        for group in model_payload.values()
        for payload in group
    )

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


def test_model_catalog_separates_primary_capable_from_required_only_refs() -> None:
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
    catalog = compile_projection_fragment_catalog(
        projection,
        batch,
        access_context_hash="access-1",
    )

    payload = catalog.model_payload()

    assert set(payload) == {"primary_candidates", "required_only_candidates"}
    assert payload["primary_candidates"]
    assert payload["required_only_candidates"]
    assert all(
        item["ref"].startswith("p")
        for item in payload["primary_candidates"]
    )
    assert all(
        item["ref"].startswith("r")
        for item in payload["required_only_candidates"]
    )
    assert all(
        "primary_eligible" not in item
        for group in payload.values()
        for item in group
    )


@pytest.mark.asyncio
async def test_prompt_requires_empty_output_when_only_required_only_context_has_claim() -> None:
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
    catalog = compile_projection_fragment_catalog(
        projection,
        batch,
        access_context_hash="access-1",
    )
    prompts: list[str] = []

    class Client:
        async def extract_projection_fragment_memories(self, prompt: str, **kwargs):
            del kwargs
            prompts.append(prompt)
            return ProjectionFragmentMemoryExtractionResponse(memories=[])

    result = await MemoryExtractor(
        structured_llm_client=Client(),
    ).extract_projection_fragment_memories(
        catalog,
        source_type="jira",
        context_markdown="",
    )

    assert result.error_type is None
    assert len(prompts) == 1
    assert '"primary_candidates"' in prompts[0]
    assert '"required_only_candidates"' in prompts[0]
    assert (
        "If a durable claim is stated only by required_only_candidates, "
        "return an empty memories array."
    ) in prompts[0]


def test_v9_derivation_identity_includes_model_presentation_policy(
    monkeypatch,
) -> None:
    import memforge.source_derivation as source_derivation_module

    projection = _projection()
    now = datetime(2026, 9, 1, tzinfo=timezone.utc)
    context = SourceUnitDerivationContext(
        document=DocumentRecord(
            doc_id="doc-1",
            source="source-1",
            source_url="https://example.test/doc-1",
            title="Document",
            space_or_project="ENG",
            author=None,
            last_modified=now,
            labels=[],
            version="1",
            content_hash="document-hash",
            token_count=10,
            raw_content_uri=None,
            raw_content_type=None,
            normalized_content_uri=None,
            pdf_content_uri=None,
            last_synced=now,
        ),
        doc_type="document",
        project_key="ENG",
        repo_identifier=None,
        document_content=projection.observation_revisions[0].content,
        update_mode="diff_guided",
        changed_hunks="current work changed",
        update_plan_stats=None,
        source_updated_at=now.isoformat(),
        user_id=None,
        source_activity_epoch=None,
    )
    v9_batches = plan_projection_extraction_batches(
        projection,
        extraction_contract_version=PROJECTION_EXTRACTION_V9,
    )
    v8_batches = plan_projection_extraction_batches(
        projection,
        extraction_contract_version=PROJECTION_EXTRACTION_V8,
    )

    monkeypatch.setattr(
        source_derivation_module,
        "PROJECTION_FRAGMENT_MODEL_PRESENTATION_POLICY_VERSION",
        1,
        raising=False,
    )
    old_v9 = source_derivation_manifest(
        projection,
        v9_batches,
        context=context,
        extraction_contract_version=PROJECTION_EXTRACTION_V9,
    )
    old_v8 = source_derivation_manifest(
        projection,
        v8_batches,
        context=context,
        extraction_contract_version=PROJECTION_EXTRACTION_V8,
    )
    monkeypatch.setattr(
        source_derivation_module,
        "PROJECTION_FRAGMENT_MODEL_PRESENTATION_POLICY_VERSION",
        2,
        raising=False,
    )
    current_v9 = source_derivation_manifest(
        projection,
        v9_batches,
        context=context,
        extraction_contract_version=PROJECTION_EXTRACTION_V9,
    )
    current_v8 = source_derivation_manifest(
        projection,
        v8_batches,
        context=context,
        extraction_contract_version=PROJECTION_EXTRACTION_V8,
    )

    assert current_v9.id != old_v9.id
    assert current_v9.batches[0].input_payload_hash != (
        old_v9.batches[0].input_payload_hash
    )
    assert current_v8.id == old_v8.id
    assert current_v8.batches[0].input_payload_hash == (
        old_v8.batches[0].input_payload_hash
    )


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
        item
        for item in catalog.model_payload()["required_only_candidates"]
        if item["kind"] == "artifact"
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
async def test_extractor_admits_normalized_candidates_with_candidate_local_telemetry() -> None:
    projection = _projection(
        context_profile=BINARY_PROFILE,
        context_content="",
        context_metadata={
            "source_artifact": {
                "inference_eligible": True,
                "sha256": "a" * 64,
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
    primary = next(item for item in catalog.fragments if item.primary_eligible)
    artifact = next(item for item in catalog.fragments if item.kind.value == "artifact")

    class Client:
        async def extract_projection_fragment_memories(self, prompt: str, **kwargs):
            return ProjectionFragmentMemoryExtractionResponse.model_validate(
                {
                    "memories": [
                        {
                            "content": "Release requires approval.",
                            "memory_type": "convention",
                            "primary_ref": primary.reference,
                            "required_refs": [
                                artifact.reference,
                                primary.reference,
                                artifact.reference,
                            ],
                        },
                        {
                            "content": "Unknown evidence must fail closed.",
                            "memory_type": "fact",
                            "primary_ref": "p999999",
                            "required_refs": [artifact.reference, artifact.reference],
                        },
                        {
                            "content": "A stale selector must fail closed.",
                            "memory_type": "fact",
                            "primary_ref": "p900001",
                            "required_refs": [artifact.reference, artifact.reference],
                        },
                        {
                            "content": "A cross-catalog selector must fail closed.",
                            "memory_type": "fact",
                            "primary_ref": "p900002",
                            "required_refs": [artifact.reference, artifact.reference],
                        },
                        {
                            "content": "An inaccessible selector must fail closed.",
                            "memory_type": "fact",
                            "primary_ref": "p900003",
                            "required_refs": [artifact.reference, artifact.reference],
                        },
                    ]
                }
            )

    collector = QualitySignalCollector()
    with quality_signal_scope(collector):
        result = await MemoryExtractor(
            structured_llm_client=Client(),
        ).extract_projection_fragment_memories(
            catalog,
            source_type="github_repo",
            context_markdown="",
        )

    assert len(result.memories) == 1
    assert [part.kind for part in result.memories[0].resolved_evidence_selection.parts] == [
        EvidencePartKind.TEXT,
        EvidencePartKind.ARTIFACT,
    ]
    assert result.metadata["selector_normalized_candidate_count"] == 5
    assert result.metadata["selector_normalization_count"] == 6
    fingerprints = result.metadata["selector_normalization_fingerprints"]
    assert len(fingerprints) == 5
    assert all(len(value) == 64 for value in fingerprints)
    assert all(primary.reference not in value for value in fingerprints)
    assert all(artifact.reference not in value for value in fingerprints)
    assert result.metadata["fragment_selection_rejection_counts"] == {
        "unknown_ref": 4,
    }

    normalization_signals = [
        signal
        for signal in collector.snapshot()
        if signal.reason_code == "fragment_selector_normalized"
    ]
    assert len(normalization_signals) == 1
    strong_candidate_hash = catalog.selection_fingerprint(
        candidate_content_hash=hashlib.sha256(
            b"Release requires approval."
        ).hexdigest(),
        primary_ref=primary.reference,
        required_refs=[artifact.reference],
    )
    assert normalization_signals[0].candidate_hash == strong_candidate_hash
    assert normalization_signals[0].candidate_hash != fingerprints[0]

    restored = memory_extraction_result_from_output_payload(
        memory_extraction_output_payload(result)
    )
    assert restored.metadata["selector_normalization_count"] == 6
    assert restored.metadata["selector_normalization_fingerprints"] == fingerprints


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
    assert "selector_normalization_count" not in restored.metadata


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
