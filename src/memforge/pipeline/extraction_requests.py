"""Budget exact L1 requests before their immutable derivation manifest is staged."""

from dataclasses import replace
import hashlib

from memforge.llm.batch_runner import ItemCapacityExceeded, LlmBatchRunner, LlmRequest
from memforge.llm.structured import ProjectionFragmentMemoryExtractionResponse
from memforge.pipeline.memory_extractor import MemoryExtractor
from memforge.pipeline.revision_input import (
    ExtractionInputTask,
    InputCandidate,
    InputCost,
    PlannedTransport,
    RevisionInputMode,
    RevisionInputPlanner,
)


class _ExtractionRequestPolicy:
    """Pack the authorized Primary Fragments into the fewest requests that fit."""

    def __init__(self, batch, catalog, *, context, extractor, source_type, doc_type):
        self.batch = batch
        self.authorized = catalog
        self.context = context
        self.extractor = extractor
        self.source_type = source_type
        self.doc_type = doc_type
        self.runner = LlmBatchRunner(extractor.structured_llm_client, model=extractor.model)

    def _prompt(self, selected, mode):
        return MemoryExtractor.projection_fragment_prompt(
            selected,
            source_type=self.source_type,
            doc_type=self.doc_type,
            context_markdown=self.batch.context_markdown,
            context_observation_ids=self.batch.context_observation_ids,
            revision_context=self.context,
            mode=mode.value,
        )

    def _request(self, selected, mode, *, load_images) -> LlmRequest:
        request = LlmRequest(
            self._prompt(selected, mode),
            ProjectionFragmentMemoryExtractionResponse,
            self.extractor.fragment_output_tokens(selected),
        )
        if not load_images:
            return request
        return self.context.attach_images(request, selected, fits=self.runner.fits)

    def _selected_catalog(self, candidate, selected_fragments):
        selected = {
            fragment.anchor: fragment
            for fragment in self.authorized.fragments
            if not fragment.primary_eligible
        }
        for index in candidate.reading_indexes:
            scoped = tuple(
                fragment
                for fragment in selected_fragments
                if fragment.anchor.observation_revision_id == index.observation_revision_id
            )
            if not scoped:
                continue
            expansion = index.expand(scoped)
            for fragment in expansion.fragments:
                if fragment.anchor in expansion.context_anchors:
                    selected[fragment.anchor] = replace(fragment, primary_eligible=False)
        selected.update({fragment.anchor: fragment for fragment in selected_fragments})
        return self.context.catalog(
            tuple(
                selected[fragment.anchor]
                for fragment in candidate.catalog.fragments
                if fragment.anchor in selected
            )
        )

    def _pack(self, candidate, *, load_images):
        primary = {fragment.reference: fragment for fragment in self.authorized.fragments if fragment.primary_eligible}
        # A full revision over a known baseline, or work without Primary authority, is one indivisible request.
        indivisible = not primary or (candidate.mode is RevisionInputMode.FULL and self.context.base is not None)

        def request_for(selected) -> LlmRequest:
            return self._request(selected, candidate.mode, load_images=load_images)

        if indivisible:
            request = self.runner.fit(lambda: request_for(candidate.catalog))
            if request is None:
                return None
            planned = ((candidate.catalog, request),)
        else:
            def selected_catalog(refs):
                return self._selected_catalog(candidate, [primary[ref] for ref in refs])

            try:
                packed = self.runner.plan_items(
                    tuple(primary), lambda refs, _context: request_for(selected_catalog(refs)),
                )
            except ItemCapacityExceeded:
                return None
            planned = tuple((selected_catalog(entry.item_ids), entry.request) for entry in packed)

        requests = []
        input_tokens = output_tokens = image_count = image_bytes = 0
        for selected, request in planned:
            input_tokens += self.extractor.structured_llm_client.request_tokens(
                request.prompt,
                response_format=request.response_format,
                model=self.extractor.model,
                images=request.images,
            )
            output_tokens += request.max_tokens
            image_count += len(request.images)
            image_bytes += sum(len(image.body) for image in request.images)
            identity = hashlib.sha256(
                (self.batch.id + selected.digest + request.prompt).encode()
            ).hexdigest()
            requests.append(
                replace(
                    self.batch,
                    id="request-" + identity,
                    prepared_catalog=selected,
                    prepared_prompt=request.prompt,
                )
            )

        expected = {f.anchor for f in self.authorized.fragments if f.primary_eligible}
        actual = [
            f.anchor
            for request in requests
            for f in request.prepared_catalog.fragments
            if f.primary_eligible
        ]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError("extraction request authority coverage mismatch")
        has_images = any(fragment.kind.value == "artifact" for fragment in candidate.catalog.fragments)
        return PlannedTransport(
            InputCost(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                request_count=len(requests),
                image_count=image_count,
                image_bytes=image_bytes,
                complete=load_images or not has_images,
            ),
            tuple(requests),
        )

    def lower_bound(self, candidate: InputCandidate):
        planned = self._pack(candidate, load_images=False)
        return planned.cost if planned is not None else None

    def materialize(self, candidate: InputCandidate):
        return self._pack(candidate, load_images=True)


def plan_fragment_requests(batch, catalog, *, context, extractor, source_type, doc_type):
    baseline = (
        context.projection.deltas[0].previous_unit_revision_id
        if context.projection.deltas
        else None
    )
    policy = _ExtractionRequestPolicy(
        batch,
        catalog,
        context=context,
        extractor=extractor,
        source_type=source_type,
        doc_type=doc_type,
    )
    plan = RevisionInputPlanner.plan(
        context=context,
        task=ExtractionInputTask(catalog, named_baseline_revision_id=baseline),
        request_policy=policy,
    )
    cost = plan.estimated_cost.as_payload()
    return tuple(
        replace(
            request,
            prepared_input_mode=plan.mode.value,
            prepared_selection_reason=plan.selection_reason,
            prepared_estimated_cost=cost,
        )
        for request in plan.transport
    )
