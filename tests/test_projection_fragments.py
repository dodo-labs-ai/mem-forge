from __future__ import annotations

from tests.evidence_display_fixture import evidence_displays

from tests.revision_client_fixture import RevisionClientFixture

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
    ProjectionFragmentSelectorCorrectionResponse,
    StructuredLlmImage,
)
from memforge.agent_knowledge import (
    AgentKnowledgePatchModelResponse,
    AgentKnowledgePatchProposal,
)
from memforge.memory.evidence import (
    EvidencePartKind,
    EvidenceRole,
)
from memforge.models import DocumentRecord
from memforge.pipeline.memory_extractor import ExtractionReading, MemoryExtractor
from memforge.pipeline.revision_assessment import RevisionAssessmentContext
from memforge.pipeline.fragment_selector_correction import correct_fragment_selectors_once
from memforge.pipeline.projection_context import (
    ExtractionAuthority,
    ExtractionRequest,
    plan_projection_evidence_work,
)
from memforge.pipeline.projection_fragments import (
    FragmentSelectionError,
    FragmentSelectionErrorCode,
    compile_projection_fragment_catalog,
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
    ProjectionCoverage,
    RevisionDelta,
    SourceObservation,
    SourceObservationRevision,
    SourceProjection,
    SourceRelation,
    SourceRelationType,
    SourceUnit,
    SourceUnitRevision,
)
from memforge.source_representation import (
    BINARY_ARTIFACT_PROFILE,
    PLAIN_TEXT_PROFILE,
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
    from memforge.source_representation import legacy_representation_profile_for_observation_contract

    # These retained JSON fixtures exercise the original Markdown declaration.
    # New provider HTML records are verified through the Jira adapter boundary.
    profile = legacy_representation_profile_for_observation_contract(
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


def _compile(projection: SourceProjection, *, access_context_hash: str):
    """The extraction catalog of obs-primary, read with obs-context as its declared predecessor.

    obs-context is reading context, so it is Required-only.
    """
    read = replace(
        projection,
        relations=(SourceRelation(SourceRelationType.PRECEDES, from_id="obs-context", to_id="obs-primary"),),
    )
    context = RevisionAssessmentContext(projection=read, base=None, access_context_hash=access_context_hash)
    return ExtractionReading.of_authority(
        context, ExtractionAuthority({"obs-primary": None}), source_type="github_repo", doc_type="markdown",
    ).catalog


def _whole_authority(projection: SourceProjection) -> ExtractionAuthority:
    authority = plan_projection_evidence_work(projection, reprocess_all_current_observations=False)
    assert isinstance(authority, ExtractionAuthority)
    return authority


def test_v9_candidate_rejects_legacy_authority() -> None:
    with pytest.raises(ValidationError):
        ProjectionFragmentMemoryCandidate.model_validate(
            {
                "content": "Approval is required.",
                "memory_type": "fact",
                "primary_ref": "f000001",
                "required_refs": [],
                "evidence_quote": "Approval is required.",
             "evidence_displays": evidence_displays("f000001", [])}
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


def test_agent_patch_model_drops_primary_repeated_as_required() -> None:
    response = AgentKnowledgePatchModelResponse.model_validate_json(
        '{"action":"create_new_concept","primary_event_id":"E1","required_event_ids":["E1","E2"]}'
    )

    assert response.primary_event_id == "E1"
    assert response.required_event_ids == ["E2"]
    with pytest.raises(ValidationError, match="primary_event_id cannot also be Required"):
        AgentKnowledgePatchProposal(
            action="create_new_concept",
            primary_event_id="E1",
            required_event_ids=["E1"],
        )


def test_agent_patch_model_no_output_with_events_still_fails() -> None:
    with pytest.raises(ValidationError, match="no_output cannot select Evidence events"):
        AgentKnowledgePatchModelResponse.model_validate(
            {"action": "no_output", "primary_event_id": "E1", "required_event_ids": ["E1"]}
        )
    with pytest.raises(ValidationError, match="no_output cannot select Evidence events"):
        AgentKnowledgePatchModelResponse.model_validate(
            {"action": "no_output", "required_event_ids": ["E2"]}
        )


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
             "evidence_displays": evidence_displays("p000001", [
                    "r000003",
                    "p000001",
                    "p000002",
                    "r000003",
                    "p000002",
                ])}
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
                "entity_refs": [],
                "valid_from": None,
                "valid_until": None,
                "primary_ref": "p000001",
                "required_refs": ["r000002"],
             "evidence_displays": evidence_displays("p000001", ["r000002"])}
        ]
    }

    response = ProjectionFragmentMemoryExtractionResponse.model_validate(payload)

    assert response.model_dump() == payload


@pytest.mark.parametrize("json_text", [False, True])
def test_v9_response_accepts_redundant_stringified_and_json_text_fallback_shapes(
    json_text: bool,
) -> None:
    payload = {"memories": json.dumps([{
        "content": "Approval is required.", "memory_type": "fact", "primary_ref": "p000001",
        "required_refs": ["r000002", "r000002"],
        "evidence_displays": evidence_displays("p000001", ["r000002", "r000002"]),
    }])}

    response = (
        ProjectionFragmentMemoryExtractionResponse.model_validate_json(
            json.dumps(payload)
        )
        if json_text
        else ProjectionFragmentMemoryExtractionResponse.model_validate(payload)
    )

    assert response.memories[0].required_refs == ["r000002", "r000002"]


def test_v9_response_defers_string_selector_membership_to_catalog_admission() -> None:
    response = ProjectionFragmentMemoryExtractionResponse.model_validate(
        {
            "memories": [
                {
                    "content": "Approval is required.",
                    "memory_type": "fact",
                    "primary_ref": "not-a-fragment",
                    "required_refs": ["also-not-a-fragment"],
                 "evidence_displays": evidence_displays("not-a-fragment", ["also-not-a-fragment"])}
            ]
        }
    )

    assert response.memories[0].primary_ref == "not-a-fragment"


def test_v9_schema_defers_primary_role_to_catalog_admission() -> None:
    deferred = ProjectionFragmentMemoryExtractionResponse.model_validate(
        {
            "memories": [
                {
                    "content": "Historical context cannot authorize a new claim.",
                    "memory_type": "fact",
                    "primary_ref": "r000004",
                    "required_refs": [],
                 "evidence_displays": evidence_displays("r000004", [])}
            ]
        }
    )
    assert deferred.memories[0].primary_ref == "r000004"

    accepted = ProjectionFragmentMemoryExtractionResponse.model_validate(
        {
            "memories": [
                {
                    "content": "Current work authorizes the claim.",
                    "memory_type": "fact",
                    "primary_ref": "p000001",
                    "required_refs": ["p000002", "r000004"],
                 "evidence_displays": evidence_displays("p000001", ["p000002", "r000004"])}
            ]
        }
    )
    assert accepted.memories[0].primary_ref == "p000001"
    assert accepted.memories[0].required_refs == ["p000002", "r000004"]


def test_catalog_resolves_one_primary_and_canonical_required_order() -> None:
    projection = _projection()
    catalog = _compile(
        projection,
        access_context_hash="access-1",
    )
    replay = _compile(
        projection,
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
                "created": "2026-09-08",
                "items": [{"field": "description", "fromString": "old", "toString": "x" * 31_000}],
            },
            separators=(",", ":"),
        )
    )
    assert len(content) > 30_000
    projection = _canonical_projection(
        observation_type=observation_type,
        content=content,
    )

    authority = _whole_authority(projection)
    catalog = compile_projection_fragment_catalog(
        projection,
        authority,
        catalog_id="catalog-canonical",
        access_context_hash="access-canonical",
    )

    assert authority.ranges_by_observation_id == {f"obs-{observation_type}": None}
    assert catalog.usable
    assert catalog.fragments
    assert all(fragment.primary_eligible for fragment in catalog.fragments)
    assert not any(
        error.code.value == "invalid_authority_range"
        for error in catalog.errors
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

    catalog = compile_projection_fragment_catalog(
        future,
        ExtractionAuthority({observation.id: None}),
        catalog_id="catalog-future-profile",
        access_context_hash="access-future-profile",
    )

    assert not catalog.usable
    assert {
        error.code.value for error in catalog.errors if error.fatal
    } == {"unsupported_profile"}


def test_canonical_nested_markdown_preserves_escaped_raw_json_ranges() -> None:
    body = 'Decision: keep "quoted" values and C:\\temp.\n\nUnicode: 雪.'
    content = json.dumps({"body": body}, ensure_ascii=True, separators=(",", ":"))
    projection = _canonical_projection(observation_type="comment", content=content)

    catalog = compile_projection_fragment_catalog(
        projection,
        _whole_authority(projection),
        catalog_id="catalog-escaped",
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
    content = "paragraph text\n\n" * 20
    base = _canonical_projection(observation_type="comment", content=content)
    [observation] = base.observations
    [revision] = base.observation_revisions

    plain = replace(
        base,
        observation_revisions=(
            replace(revision, evidence_profile=PLAIN_TEXT_PROFILE),
        ),
    )
    plain_catalog = compile_projection_fragment_catalog(
        plain,
        _whole_authority(plain),
        catalog_id="catalog-plain",
        access_context_hash="access-plain",
    )
    assert len(plain_catalog.fragments) > 1
    assert all(
        fragment.anchor.kind is AnchorKind.REVISION_RANGE
        and fragment.anchor.range_end - fragment.anchor.range_start < len(content)
        for fragment in plain_catalog.fragments
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
    binary_authority = _whole_authority(binary)
    assert binary_authority.ranges_by_observation_id == {observation.id: None}
    binary_catalog = compile_projection_fragment_catalog(
        binary,
        binary_authority,
        catalog_id="catalog-binary",
        access_context_hash="access-binary",
    )
    assert binary_catalog.usable
    assert [
        fragment.anchor.kind.value for fragment in binary_catalog.fragments
    ] == ["whole_observation"]


def test_large_canonical_fragment_fails_with_capacity_error_without_raw_slicing() -> None:
    content = json.dumps(
        {"items": [{"field": "description", "toString": "x" * 2_000}]},
        separators=(",", ":"),
    )
    projection = _canonical_projection(
        observation_type="changelog",
        content=content,
    )

    authority = _whole_authority(projection)
    catalog = compile_projection_fragment_catalog(
        projection,
        authority,
        catalog_id="catalog-capacity",
        access_context_hash="access-capacity",
        max_presentation_chars=1_000,
    )

    assert authority.ranges_by_observation_id == {"obs-changelog": None}
    assert not catalog.usable
    assert catalog.fragments == ()
    assert {
        error.code.value for error in catalog.errors if error.fatal
    } == {"catalog_too_large"}


def test_catalog_rejects_duplicate_unknown_and_ineligible_selectors() -> None:
    projection = _projection()
    catalog = _compile(
        projection,
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
    catalog = _compile(
        projection,
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
    assert _whole_authority(projection).ranges_by_observation_id == {"obs-primary": None}

    catalog = _compile(projection, access_context_hash="access-1")
    model_payload = catalog.model_payload()
    payload_by_ref = {
        item[0]: item
        for group in model_payload.values()
        for item in group
    }
    payload_by_observation = {
        fragment.anchor.observation_id: payload_by_ref[fragment.reference]
        for fragment in catalog.fragments
    }

    assert payload_by_observation["obs-primary"][0].startswith("p")
    assert payload_by_observation["obs-context"][0].startswith("r")
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
    catalog = _compile(projection, access_context_hash="access-1")

    payload = catalog.model_payload()

    assert set(payload) == {"primary_candidates", "required_only_candidates"}
    assert payload["primary_candidates"]
    assert payload["required_only_candidates"]
    assert all(
        item[0].startswith("p")
        for item in payload["primary_candidates"]
    )
    assert all(
        item[0].startswith("r")
        for item in payload["required_only_candidates"]
    )
    assert all(
        "primary_eligible" not in item
        for group in payload.values()
        for item in group
    )


@pytest.mark.asyncio
async def test_prompt_requires_owned_primary_without_suppressing_other_eligible_claims() -> None:
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
    catalog = _compile(projection, access_context_hash="access-1")
    prompts: list[str] = []

    class Client(RevisionClientFixture):
        def request_fits(self, prompt, **kwargs):
            return True

        async def extract_projection_fragment_memories(self, prompt: str, **kwargs):
            del kwargs
            prompts.append(prompt)
            return ProjectionFragmentMemoryExtractionResponse(memories=[])

    result = await MemoryExtractor(
        structured_llm_client=Client(),
    ).extract_projection_fragment_memories(
        catalog,
        source_type="jira",
        revision_context=RevisionAssessmentContext(projection=projection, base=None, access_context_hash="access-1"),
    )

    assert result.error_type is None
    assert len(prompts) == 1
    assert '"primary_candidates"' in prompts[0]
    assert '"required_only_candidates"' in prompts[0]
    assert (
        "Do not originate a Memory whose central assertion is stated only by required_only_candidates; "
        "preserve other useful claims with eligible Primary evidence."
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
    )
    batches = (
        ExtractionRequest(
            id="request-1",
            source_unit_id="unit-1",
            catalog=_compile(projection, access_context_hash="access-1"),
            prompt_sha256=hashlib.sha256(b"extraction prompt").hexdigest(),
        ),
    )

    monkeypatch.setattr(
        source_derivation_module,
        "PROJECTION_FRAGMENT_MODEL_PRESENTATION_POLICY_VERSION",
        1,
        raising=False,
    )
    old = source_derivation_manifest(projection, batches, context=context)
    monkeypatch.setattr(
        source_derivation_module,
        "PROJECTION_FRAGMENT_MODEL_PRESENTATION_POLICY_VERSION",
        2,
        raising=False,
    )
    current = source_derivation_manifest(projection, batches, context=context)

    assert current.id != old.id
    assert current.batches[0].input_payload_hash != (
        old.batches[0].input_payload_hash
    )


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
        ExtractionAuthority({"obs-context": None}),
        catalog_id="catalog-missing-profile",
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
    catalog = _compile(projection, access_context_hash="access-1")
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
        if len(item) == 3
    )
    assert artifact_payload[0] == artifact.reference
    assert artifact_payload[2]["image_source_observation_id"] == "obs-context"
    assert artifact_payload[2]["filename"] == "diagram.png"

    selection = catalog.resolve_selection(
        primary_ref=primary.reference,
        required_refs=[artifact.reference],
    )
    assert selection.parts[1].kind is EvidencePartKind.ARTIFACT
    assert selection.parts[1].raw_content_sha256 == artifact_digest
    assert selection.parts[1].artifact_metadata["media_type"] == "image/png"


@pytest.mark.asyncio
async def test_extractor_rejects_malformed_batch_without_normalizing_or_pruning() -> None:
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
    catalog = _compile(projection, access_context_hash="access-1")
    primary = next(item for item in catalog.fragments if item.primary_eligible)
    artifact = next(item for item in catalog.fragments if item.kind.value == "artifact")

    class Client(RevisionClientFixture):
        def request_fits(self, prompt, **kwargs):
            return True

        async def correct_projection_fragment_selectors(self, prompt: str, **kwargs):
            return ProjectionFragmentSelectorCorrectionResponse(corrections=[])

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
                         "evidence_displays": evidence_displays(primary.reference, [
                                artifact.reference,
                                primary.reference,
                                artifact.reference,
                            ])},
                        {
                            "content": "Unknown evidence must fail closed.",
                            "memory_type": "fact",
                            "primary_ref": "not-a-fragment",
                            "required_refs": [artifact.reference, artifact.reference],
                         "evidence_displays": evidence_displays("not-a-fragment", [artifact.reference, artifact.reference])},
                        {
                            "content": "A stale selector must fail closed.",
                            "memory_type": "fact",
                            "primary_ref": "p900001",
                            "required_refs": [artifact.reference, artifact.reference],
                         "evidence_displays": evidence_displays("p900001", [artifact.reference, artifact.reference])},
                        {
                            "content": "A cross-catalog selector must fail closed.",
                            "memory_type": "fact",
                            "primary_ref": "p900002",
                            "required_refs": [artifact.reference, artifact.reference],
                         "evidence_displays": evidence_displays("p900002", [artifact.reference, artifact.reference])},
                        {
                            "content": "An inaccessible selector must fail closed.",
                            "memory_type": "fact",
                            "primary_ref": "p900003",
                            "required_refs": [artifact.reference, artifact.reference],
                         "evidence_displays": evidence_displays("p900003", [artifact.reference, artifact.reference])},
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
            revision_context=RevisionAssessmentContext(
                projection=projection,
                base=None,
                access_context_hash="access-1",
                images=(
                    StructuredLlmImage(source_observation_id="obs-context", media_type="image/png", body=b"image"),
                ),
            ),
        )

    assert result.error_type == "projection_extraction_incomplete"
    assert result.metadata["safe_error_code"] == "EXTRACTION_EVIDENCE_UNRESOLVED"
    assert result.memories == []
    assert result.metadata["fragment_selection_rejection_counts"] == {"duplicate_ref": 5}
    assert not any(key.startswith("selector_normalization") for key in result.metadata)
    assert not any(signal.reason_code == "fragment_selector_normalized"
                   for signal in collector.snapshot())

    with pytest.raises(ValueError, match="failed derivation batches have no reusable output"):
        memory_extraction_output_payload(result)


@pytest.mark.asyncio
async def test_extractor_persists_only_resolved_parts_and_never_falls_back() -> None:
    projection = _projection()
    catalog = _compile(projection, access_context_hash="access-1")
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

    class Client(RevisionClientFixture):
        def request_fits(self, prompt, **kwargs):
            return True

        async def extract_projection_fragment_memories(self, prompt: str, **kwargs):
            assert "evidence_quote" not in prompt
            return ProjectionFragmentMemoryExtractionResponse(
                memories=[
                    ProjectionFragmentMemoryCandidate(
                        content="Release requires approval by two reviewers.",
                        memory_type="convention",
                        primary_ref=primary.reference,
                        required_refs=[required.reference],
                     evidence_displays=evidence_displays(primary.reference, [required.reference]))
                ]
            )

    result = await MemoryExtractor(
        structured_llm_client=Client(),
    ).extract_projection_fragment_memories(
        catalog,
        source_type="github_repo",
        revision_context=RevisionAssessmentContext(projection=projection, base=None, access_context_hash="access-1"),
    )
    assert result.error_type is None
    assert len(result.memories) == 1
    memory = result.memories[0]
    assert memory.evidence_quote is None
    assert memory.resolved_evidence_selection is not None
    assert len(memory.resolved_evidence_selection.parts) == 2

    restored = memory_extraction_result_from_output_payload(
        memory_extraction_output_payload(result)
    )
    assert restored.memories[0].resolved_evidence_selection == (
        memory.resolved_evidence_selection
    )
    assert "selector_normalization_count" not in restored.metadata


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["valid", "unknown", "duplicate", "required_only"])
async def test_extraction_binds_once_without_selector_repair_or_partial_success(mode) -> None:
    projection = _projection()
    catalog = _compile(projection, access_context_hash="access-1")
    primary = next(f.reference for f in catalog.fragments if f.primary_eligible)
    required = next(f.reference for f in catalog.fragments if not f.primary_eligible)
    stable = ProjectionFragmentMemoryCandidate(
        content="Release requires approval.", memory_type="convention", primary_ref=primary,
     evidence_displays=evidence_displays(primary, ()))
    extra = ProjectionFragmentMemoryCandidate(
        content="Two reviewers are required.", memory_type="convention",
        primary_ref=required if mode == "required_only" else "unknown" if mode == "unknown" else primary,
        required_refs=[primary] if mode == "duplicate" else [],
     evidence_displays=evidence_displays(required if mode == "required_only" else "unknown" if mode == "unknown" else primary, [primary] if mode == "duplicate" else []))

    class Client(RevisionClientFixture):
        extraction_calls = 0
        correction_calls = 0

        def request_fits(self, prompt, **kwargs):
            return True

        async def extract_projection_fragment_memories(self, prompt, **kwargs):
            self.extraction_calls += 1
            return ProjectionFragmentMemoryExtractionResponse(memories=[stable] if mode == "valid" else [stable, extra])

        async def correct_projection_fragment_selectors(self, prompt, **kwargs):
            self.correction_calls += 1
            raise AssertionError("the extraction contract has no selector repair stage")

    client = Client()
    result = await MemoryExtractor(structured_llm_client=client).extract_projection_fragment_memories(
        catalog, source_type="github_repo",
        revision_context=RevisionAssessmentContext(projection=projection, base=None, access_context_hash="access-1"),
    )
    assert client.extraction_calls == 1 and client.correction_calls == 0
    assert result.metadata["structured_llm_calls"] == 1
    if mode != "valid":
        assert result.error_type == "projection_extraction_incomplete"
        assert result.metadata["safe_error_code"] == "EXTRACTION_EVIDENCE_UNRESOLVED"
        assert result.memories == []
        with pytest.raises(ValueError, match="failed derivation batches have no reusable output"):
            memory_extraction_output_payload(result)
    else:
        assert result.error_type is None
        restored = memory_extraction_result_from_output_payload(memory_extraction_output_payload(result))
        assert restored.memories[0].resolved_evidence_selection == result.memories[0].resolved_evidence_selection
        assert not any(key.startswith("selector_correction") for key in result.metadata)


@pytest.mark.asyncio
async def test_selector_correction_groups_failures_and_reuses_artifact_images() -> None:
    projection = _projection(context_profile=BINARY_PROFILE, context_content="", context_metadata={
        "source_artifact": {"inference_eligible": True, "sha256": "a" * 64,
                            "media_type": "image/png", "size_bytes": 128, "filename": "diagram.png"},
    })
    catalog = _compile(projection, access_context_hash="access-1")
    primary = next(f.reference for f in catalog.fragments if f.primary_eligible)
    artifact = next(f.reference for f in catalog.fragments if f.kind.value == "artifact")
    candidates = [
        ProjectionFragmentMemoryCandidate(content="Fixed claim A", memory_type="fact", primary_ref="bad", evidence_displays=evidence_displays("bad", ())),
        ProjectionFragmentMemoryCandidate(content="Fixed claim B", memory_type="fact", primary_ref=primary,
                                          required_refs=["unknown-required"], evidence_displays=evidence_displays(primary, ["unknown-required"])),
    ]
    images = (StructuredLlmImage(source_observation_id="obs-context", media_type="image/png", body=b"image"),)

    class Client(RevisionClientFixture):
        calls = 0

        def request_fits(self, prompt, **kwargs):
            assert kwargs["images"] is images
            return True

        async def correct_projection_fragment_selectors(self, prompt, **kwargs):
            self.calls += 1
            assert kwargs["images"] is images
            assert all(candidate.content in prompt for candidate in candidates)
            return ProjectionFragmentSelectorCorrectionResponse.model_validate({"corrections": [
                {"candidate_index": 0, "primary_ref": primary, "required_refs": [artifact]},
                {"candidate_index": 1, "primary_ref": primary, "required_refs": ["still-unknown"]},
            ]})

    client = Client()
    corrected, metrics = await correct_fragment_selectors_once(
        candidates, catalog=catalog, client=client, extraction_prompt="supplied catalog",
        max_tokens=512, model="test-model", images=images,
    )
    assert client.calls == 1
    assert metrics["selector_correction_candidate_count"] == 2
    assert metrics["selector_correction_recovered_count"] == 1
    assert corrected[0].required_refs == [artifact]
    assert corrected[1] is candidates[1]


def test_compact_catalog_preserves_exact_text_authority_and_internal_provenance():
    import json
    projection = _projection()
    catalog = _compile(projection, access_context_hash="access-1")
    payload = catalog.model_payload()
    rows = [row for group in payload.values() for row in group]
    by_ref = {row[0]: row for row in rows}
    legacy = {role: [] for role in payload}
    for fragment in catalog.fragments:
        role = "primary_candidates" if fragment.primary_eligible else "required_only_candidates"
        assert by_ref[fragment.reference][1] == fragment.presentation_text
        assert by_ref[fragment.reference] in payload[role]
        legacy[role].append({"ref": fragment.reference, "kind": fragment.kind.value,
                             "type": fragment.fragment_type, "text": fragment.presentation_text})
    compact_text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    legacy_text = json.dumps(legacy, ensure_ascii=False, separators=(",", ":"))
    assert len(compact_text) < len(legacy_text)
    assert '"kind"' not in compact_text and '"type"' not in compact_text
    primary = next(f for f in catalog.fragments if f.primary_eligible)
    result = catalog.resolve_selection(primary_ref=primary.reference)
    assert result.parts[0].anchor == primary.anchor


@pytest.mark.parametrize("tag", ["h2", "pre", "blockquote"])
def test_compact_catalog_retains_html_semantics_after_tag_stripping(tag):
    projection = _projection()
    catalog = _compile(projection, access_context_hash="access-1")
    fragment = replace(catalog.fragments[0], fragment_type=f"html-{tag}", presentation_text="US payroll")
    catalog = replace(catalog, fragments=(fragment,))
    row = next(row for group in catalog.model_payload().values() for row in group)
    assert row == (fragment.reference, "US payroll", {"format": f"html-{tag}"})
    assert catalog.fragments[0].anchor == fragment.anchor
