from dataclasses import replace

from memforge.pipeline.extraction_requests import plan_fragment_requests
from memforge.pipeline.memory_extractor import MemoryExtractor
from memforge.pipeline.projection_fragments import compile_projection_fragment_catalog
from memforge.pipeline.revision_assessment import RevisionAssessmentContext
from tests.test_projection_fragments import _projection, _batch
from tests.test_revision_work import Client


def test_first_import_over_old_catalog_limit_batches_complete_requests_and_keeps_heading_refs():
    body = "# US payroll\n\n" + "\n\n".join(f"Rule {i}: " + "This is exact source content. " * 80 for i in range(75))
    assert len(body) > 120000
    projection = _projection(primary_content=body, context_content="Country: US.\n")
    batch = _batch(projection)
    context = RevisionAssessmentContext(projection=projection, base=None, access_context_hash="scope")
    catalog = compile_projection_fragment_catalog(
        projection,
        batch,
        access_context_hash="scope",
        max_fragments=len(context.full_fragments),
        max_presentation_chars=sum(len(f.presentation_text) for f in context.full_fragments),
    )
    client = Client(limit=10000)
    extractor = MemoryExtractor(model="fixture", max_tokens=8192, structured_llm_client=client)
    requests = plan_fragment_requests(
        batch, catalog, context=context, extractor=extractor, source_type="confluence", doc_type="document"
    )
    assert len(requests) > 2
    primary = [f.anchor for request in requests for f in request.prepared_catalog.fragments if f.primary_eligible]
    assert len(primary) == len(set(primary))
    assert set(primary) == {f.anchor for f in catalog.fragments if f.primary_eligible}
    for request in requests:
        assert client.request_fits(request.prepared_prompt, max_tokens=8192)
        heading = next(f for f in request.prepared_catalog.fragments if f.presentation_text.strip() == "# US payroll")
        if heading.anchor not in {f.anchor for f in request.prepared_catalog.fragments if f.primary_eligible}:
            assert not heading.primary_eligible
        assert request.prepared_prompt.count("Country: US.") == 1
    repeated = plan_fragment_requests(
        batch, catalog, context=context, extractor=extractor, source_type="confluence", doc_type="document"
    )
    assert [r.id for r in repeated] == [r.id for r in requests]


def test_current_full_read_never_authorizes_unchanged_fragments():
    projection = _projection(
        primary_content="# US payroll\n\nOld unchanged rule.\n\nNew approval rule.\n", context_content="Country: US.\n"
    )
    context = RevisionAssessmentContext(projection=projection, base=None, access_context_hash="scope")
    batch = _batch(projection)
    authorized = context.catalog(
        tuple(replace(f, primary_eligible="New approval" in f.presentation_text) for f in context.full_fragments)
    )
    extractor = MemoryExtractor(model="fixture", structured_llm_client=Client(limit=20000))
    [request] = plan_fragment_requests(
        batch, authorized, context=context, extractor=extractor, source_type="confluence", doc_type="document"
    )
    assert [f.presentation_text.strip() for f in request.prepared_catalog.fragments if f.primary_eligible] == [
        "New approval rule."
    ]
