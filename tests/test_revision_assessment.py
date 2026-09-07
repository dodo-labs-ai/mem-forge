"""Full/delta semantics and exact current Evidence selection contracts."""

import json
from dataclasses import replace

import pytest

from memforge.llm.structured import RevisionSupportResponse
from memforge.memory.evidence import ActiveSupportEvidence, EvidenceRole
from memforge.models import Memory, content_hash
from memforge.pipeline.revision_assessment import RevisionAssessmentContext
from memforge.pipeline.reconciler import ReconciliationContractError
from memforge.pipeline.projection_fragments import SupportRevalidationLimitation
from memforge.source_projection import ProjectionCoverage
from tests.test_projection_fragments import _projection


def revisions(old, new):
    base = _projection(primary_content=old, context_content="Country: US.\n")
    current = _projection(primary_content=new, context_content="Country: US.\n")
    primary = replace(current.observation_revisions[0], id="rev-primary-v2")
    unit = replace(current.source_unit_revisions[0], id="unit-v2", observation_revision_ids=(primary.id, "rev-context"))
    current = replace(
        current,
        run_id="run-v2",
        observation_revisions=(primary, current.observation_revisions[1]),
        source_unit_revisions=(unit,),
        deltas=(
            replace(
                current.deltas[0],
                previous_unit_revision_id=base.source_unit_revisions[0].id,
                current_unit_revision_id=unit.id,
            ),
        ),
    )
    return base, current


def old_support(base):
    ctx = RevisionAssessmentContext(projection=base, base=None, access_context_hash="scope")
    primary = next(f for f in ctx.full_fragments if f.presentation_text.startswith("Two reviewers"))
    return (
        ActiveSupportEvidence(
            memory_id="memory",
            source_id="source-1",
            reference_id="e1",
            evidence_unit_id="eu1",
            role=EvidenceRole.PRIMARY,
            anchor=primary.anchor,
            excerpt=primary.presentation_text,
        ),
    )


def memory():
    claim = "US releases require two reviewers."
    return Memory(id="memory", content=claim, content_hash=content_hash(claim), memory_type="fact")


class Client:
    def __init__(self, mode="full", status="supported", invalid=0):
        self.mode, self.status, self.invalid = mode, status, invalid
        self.prompts = []
        self.budgets = []

    def request_fits(self, prompt, **kwargs):
        return self.mode == "full" or '"input_mode":"delta"' in prompt

    async def assess_revision_support(self, prompt, **kwargs):
        self.prompts.append(prompt)
        self.budgets.append(kwargs["max_tokens"])
        if len(self.prompts) <= self.invalid:
            return RevisionSupportResponse(status="supported", primary_ref="invented")
        payload = json.loads(prompt.split("<assessment>", 1)[1].split("</assessment>", 1)[0])
        refs = payload["current"]["primary_candidates"]
        primary = next(item for item in refs if "reviewers" in item[1].lower())
        required = [item[0] for item in refs if item[1].startswith(("Country:", "Payroll type:"))]
        return RevisionSupportResponse(status=self.status, primary_ref=primary[0], required_refs=required)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["full", "delta"])
async def test_moved_rewritten_primary_and_split_required_select_current_coordinates(mode):
    base, current = revisions(
        "Two reviewers approve US regular payroll.\n",
        "# New section\n\nCountry: US.\n\nPayroll type: regular.\n\nRelease requires two reviewers.\n",
    )
    context = RevisionAssessmentContext(projection=current, base=base, access_context_hash="scope")
    client = Client(mode)
    result = await context.assess(memory=memory(), support=old_support(base), client=client, model="test")
    assert result.supported and result.input_mode == mode
    assert (
        len(result.memory.resolved_evidence_selection.parts) >= 3
    )  # primary plus complete country and payroll conditions
    assert result.memory.resolved_evidence_selection.parts[0].anchor.observation_revision_id == "rev-primary-v2"
    assert len(client.prompts) == 1


@pytest.mark.asyncio
async def test_complete_delta_includes_remote_exception_and_unchanged_heading():
    base, current = revisions(
        "Two reviewers approve US releases.\n\n# Europe\n\nOld note.\n",
        "Two reviewers approve US releases.\n\n# Europe\n\nOne reviewer is sufficient.\n",
    )
    ctx = RevisionAssessmentContext(projection=current, base=base, access_context_hash="scope")
    _, prompt, mode, _ = ctx.input_for(memory=memory(), support=old_support(base), client=Client("delta"), model="test")
    assert mode == "delta"
    payload = json.loads(prompt.split("<assessment>", 1)[1].split("</assessment>", 1)[0])
    exception = next(item for item in payload["current"]["primary_candidates"] if "One reviewer" in item[1])
    group = next(group for group in payload["current"]["structural_groups"] if exception[0] in group["refs"])
    assert any("Europe" in title for title in group["heading_context"])
    assert any("Old note" in item["text"] for item in payload["removed_historical"])


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", [1, 2])
async def test_current_selection_has_one_local_correction(invalid):
    base, current = revisions("Two reviewers approve US releases.\n", "Two reviewers approve US releases today.\n")
    ctx = RevisionAssessmentContext(projection=current, base=base, access_context_hash="scope")
    client = Client(invalid=invalid)
    if invalid == 2:
        with pytest.raises(ReconciliationContractError, match="correction exhausted"):
            await ctx.assess(memory=memory(), support=old_support(base), client=client, model="test")
    else:
        assert (await ctx.assess(memory=memory(), support=old_support(base), client=client, model="test")).supported
    assert len(client.prompts) == 2


def test_missing_baseline_never_becomes_empty_delta():
    base, current = revisions("Two reviewers approve US releases.\n", "Two reviewers approve US releases today.\n")
    ctx = RevisionAssessmentContext(projection=current, base=None, access_context_hash="scope")
    with pytest.raises(SupportRevalidationLimitation):
        ctx.input_for(memory=memory(), support=old_support(base), client=Client("delta"), model="test")


def test_partial_projection_keeps_exact_carried_member():
    base, current = revisions("Two reviewers approve US releases.\n", "Two reviewers approve US releases today.\n")
    current = replace(
        current,
        observations=current.observations[:1],
        coverage=ProjectionCoverage.PARTIAL_PROJECTION,
        carried_observation_revision_ids=("rev-context",),
        deltas=(replace(current.deltas[0], coverage=ProjectionCoverage.PARTIAL_PROJECTION, added_observation_ids=()),),
    )
    ctx = RevisionAssessmentContext(projection=current, base=base, access_context_hash="scope")
    assert "obs-context" in ctx.current
    assert not any(item["observation_id"] == "obs-context" for item in ctx.delta()[1])


def test_large_unchanged_artifacts_do_not_block_text_delta_or_get_read(monkeypatch):
    from types import SimpleNamespace
    from memforge.pipeline.projection_images import load_projection_images
    from tests.test_projected_lifecycle_integration import _projection_with_artifact

    base = _projection_with_artifact(
        run_id="image-base",
        payload=b"x" * 100,
        provider_revision="1",
        inference_eligible=True,
        body="Two reviewers approve US releases.",
    )
    target = _projection_with_artifact(
        run_id="image-target",
        payload=b"x" * 100,
        provider_revision="1",
        inference_eligible=True,
        body="Two reviewers approve US releases.\n\nNew unrelated note.",
        prior=base.source_unit_revisions[0],
        prior_observations={r.observation_id: r for r in base.observation_revisions},
    )
    # Total metadata admission rejects full image context before any read.
    monkeypatch.setattr("memforge.pipeline.projection_images.MAX_SOURCE_ARTIFACT_INFERENCE_BYTES_PER_BATCH", 50)
    reads = []
    ctx = RevisionAssessmentContext(
        projection=target,
        base=base,
        access_context_hash="scope",
        image_loader=lambda ids: load_projection_images(
            projection=target,
            observation_ids=ids,
            document_store=SimpleNamespace(read_artifact=lambda uri: reads.append(uri)),
        ),
    )
    catalog, prompt, mode, images = ctx.input_for(
        memory=memory(), support=old_support(base), client=Client(), model="test"
    )
    assert mode == "delta" and images == () and reads == []
    assert "New unrelated note." in prompt
    assert all(f.kind.value != "artifact" for f in catalog.fragments)


@pytest.mark.asyncio
async def test_changed_artifact_bytes_are_supplied_to_the_selected_delta():
    from types import SimpleNamespace
    from memforge.llm.structured_images import StructuredLlmImage
    from memforge.pipeline.projection_images import load_projection_images
    from tests.test_projected_lifecycle_integration import _projection_with_artifact

    base = _projection_with_artifact(
        run_id="image-base",
        payload=b"old",
        provider_revision="1",
        inference_eligible=True,
        body="Two reviewers approve US releases.",
    )
    target = _projection_with_artifact(
        run_id="image-target",
        payload=b"new",
        provider_revision="2",
        inference_eligible=True,
        body="Two reviewers approve US releases.",
        prior=base.source_unit_revisions[0],
        prior_observations={r.observation_id: r for r in base.observation_revisions},
    )
    reads = []

    def read(uri):
        reads.append(uri)
        return b"new"

    client = Client("delta")
    ctx = RevisionAssessmentContext(
        projection=target,
        base=base,
        access_context_hash="scope",
        image_loader=lambda ids: load_projection_images(
            projection=target, observation_ids=ids, document_store=SimpleNamespace(read_artifact=read)
        ),
    )
    _, _, mode, images = ctx.input_for(memory=memory(), support=old_support(base), client=client, model="test")
    assert mode == "delta" and len(reads) == 1
    assert len(images) == 1 and isinstance(images[0], StructuredLlmImage) and images[0].body == b"new"


def test_processing_limitation_cannot_retry_whole_document():
    from memforge.pipeline.projection_fragments import SupportRevalidationLimitationCode

    error = SupportRevalidationLimitation(SupportRevalidationLimitationCode.COMPILER_FAILURE, "bad representation")
    assert error.retryable is False


@pytest.mark.parametrize("mode", ["full", "delta"])
def test_extraction_context_expansion_preserves_exact_changed_primary_authority(mode):
    from dataclasses import replace

    base, target = revisions(
        "Two reviewers approve US releases.\n", "Two reviewers approve US releases.\n\nNew deployment owner: Alex.\n"
    )
    ctx = RevisionAssessmentContext(projection=target, base=base, access_context_hash="scope")
    changed = next(f for f in ctx.full_fragments if "New deployment owner" in f.presentation_text)
    authorized = ctx.catalog((replace(changed, primary_eligible=True),))
    expanded = ctx.extraction_catalog(authorized, mode)
    assert {f.anchor for f in expanded.fragments if f.primary_eligible} == {changed.anchor}
    if mode == "full":
        assert any("Two reviewers" in f.presentation_text and not f.primary_eligible for f in expanded.fragments)


def test_actual_extraction_allowance_is_part_of_reuse_identity():
    from memforge.pipeline.revision_assessment import revision_inference_capability_hash

    original = revision_inference_capability_hash(None, extraction_model="model-a", extraction_max_tokens=4096)
    assert original != revision_inference_capability_hash(None, extraction_model="model-a", extraction_max_tokens=8192)
    assert original != revision_inference_capability_hash(None, extraction_model="model-b", extraction_max_tokens=4096)
