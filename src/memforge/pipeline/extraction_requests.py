"""Budget exact L1 requests before their immutable derivation manifest is staged."""

from dataclasses import replace
import hashlib

from memforge.llm.structured import ProjectionFragmentMemoryExtractionResponse
from memforge.pipeline.memory_extractor import MemoryExtractor
from memforge.pipeline.projection_fragments import SupportRevalidationLimitation, SupportRevalidationLimitationCode
from memforge.pipeline.projection_images import ProjectionImageLoadError


def plan_fragment_requests(batch, catalog, *, context, extractor, source_type, doc_type):
    def prompt(selected):
        # Omit only context proven present by exact current Fragment identity.
        context_ids = set(batch.context_observation_ids)
        context_fragments = {f.anchor for f in context.full_fragments if f.anchor.observation_id in context_ids}
        represented_ids = {f.anchor.observation_id for f in context.full_fragments if f.anchor in context_fragments}
        covered = context_ids <= represented_ids and context_fragments <= {f.anchor for f in selected.fragments}
        return MemoryExtractor.projection_fragment_prompt(
            selected,
            source_type=source_type,
            doc_type=doc_type,
            context_markdown="" if covered else batch.context_markdown,
            revision_context=context,
            mode="authorized_work",
        )

    def fits(selected):
        text = prompt(selected)
        kwargs = dict(
            response_format=ProjectionFragmentMemoryExtractionResponse,
            max_tokens=extractor.fragment_output_tokens(selected),
            model=extractor.model,
        )
        if not extractor.structured_llm_client.request_fits(text, **kwargs):
            return False
        try:
            images = context.images_for(selected)
        except ProjectionImageLoadError as error:
            if error.error_code == "image_batch_too_large":
                return False
            raise
        return extractor.structured_llm_client.request_fits(text, images=images, **kwargs)

    def materialize(primary):
        selected = {f.anchor: f for f in catalog.fragments if not f.primary_eligible}
        for ancestor in context.ancestor_fragments(primary):
            selected[ancestor.anchor] = replace(ancestor, primary_eligible=False)
        selected.update({f.anchor: f for f in primary})
        return context.catalog(tuple(selected.values()))

    primary = [f for f in catalog.fragments if f.primary_eligible]
    selected = (context.extraction_catalog(catalog, "full") if context.base is None
                else materialize(primary))
    if fits(selected):
        return (replace(batch, prepared_catalog=selected, prepared_prompt=prompt(selected)),)
    if not primary:
        return (replace(batch, prepared_catalog=catalog, prepared_prompt=prompt(catalog)),)
    result = []
    while primary:
        low, high = 0, len(primary)
        while low < high:
            middle = (low + high + 1) // 2
            if fits(materialize(primary[:middle])):
                low = middle
            else:
                high = middle - 1
        if not low:
            raise SupportRevalidationLimitation(
                SupportRevalidationLimitationCode.CAPACITY_EXCEEDED,
                "one authorized structure and its required context exceed configured capability",
            )
        selected = materialize(primary[:low])
        text = prompt(selected)
        identity = hashlib.sha256((batch.id + selected.digest + text).encode()).hexdigest()
        result.append(replace(batch, id="request-" + identity, prepared_catalog=selected, prepared_prompt=text))
        primary = primary[low:]
    expected = {f.anchor for f in catalog.fragments if f.primary_eligible}
    actual = [f.anchor for request in result for f in request.prepared_catalog.fragments if f.primary_eligible]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("extraction request authority coverage mismatch")
    return tuple(result)
