"""Budget exact L1 requests before their immutable derivation manifest is staged."""

from dataclasses import replace
import hashlib

from memforge.llm.structured import ProjectionFragmentMemoryExtractionResponse
from memforge.pipeline.memory_extractor import MemoryExtractor
from memforge.pipeline.projection_images import ProjectionImageLoadError
from memforge.pipeline.revision_input import (
    ExtractionInputTask,
    InputCandidate,
    InputCost,
    PlannedTransport,
    RevisionInputMode,
    RevisionInputPlanner,
)


class _ExtractionRequestPolicy:
    def __init__(self, batch, catalog, *, context, extractor, source_type, doc_type):
        self.batch = batch
        self.authorized = catalog
        self.context = context
        self.extractor = extractor
        self.source_type = source_type
        self.doc_type = doc_type

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

    def _request(self, selected, mode, *, load_images):
        text = self._prompt(selected, mode)
        kwargs = dict(
            response_format=ProjectionFragmentMemoryExtractionResponse,
            max_tokens=self.extractor.fragment_output_tokens(selected),
            model=self.extractor.model,
        )
        client = self.extractor.structured_llm_client
        if not client.request_fits(text, **kwargs):
            return None
        images = ()
        if not load_images:
            return text, images, kwargs
        try:
            images = self.context.images_for(selected)
        except ProjectionImageLoadError as error:
            if error.error_code == "image_batch_too_large":
                return None
            raise
        return (text, images, kwargs) if client.request_fits(text, images=images, **kwargs) else None

    def _selected_catalog(
        self,
        candidate,
        selected_fragments,
        *,
        authorize_primary=True,
        include_persistent_context=True,
    ):
        selected_by_anchor = {
            fragment.anchor: (
                fragment if authorize_primary else replace(fragment, primary_eligible=False)
            )
            for fragment in selected_fragments
        }
        selected = {
            fragment.anchor: fragment
            for fragment in self.authorized.fragments
            if not fragment.primary_eligible
        } if include_persistent_context else {}
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
        selected.update(selected_by_anchor)
        return self.context.catalog(
            tuple(
                selected[fragment.anchor]
                for fragment in candidate.catalog.fragments
                if fragment.anchor in selected
            )
        )

    def _pack(self, candidate, *, load_images):
        primary = [fragment for fragment in self.authorized.fragments if fragment.primary_eligible]
        if candidate.mode is RevisionInputMode.FULL and self.context.base is not None:
            selections = [candidate.catalog]
            remaining = []
        elif not primary:
            selections = [candidate.catalog]
            remaining = []
        else:
            selections = []
            remaining = list(primary)
        while remaining:
            low, high = 0, len(remaining)
            while low < high:
                middle = (low + high + 1) // 2
                selected = self._selected_catalog(candidate, remaining[:middle])
                if self._request(selected, candidate.mode, load_images=load_images) is not None:
                    low = middle
                else:
                    high = middle - 1
            if not low:
                return None
            selections.append(self._selected_catalog(candidate, remaining[:low]))
            remaining = remaining[low:]

        requests = []
        input_tokens = output_tokens = image_count = image_bytes = 0
        for selected in selections:
            request = self._request(selected, candidate.mode, load_images=load_images)
            if request is None:
                return None
            text, images, kwargs = request
            output = kwargs["max_tokens"]
            input_tokens += self.extractor.structured_llm_client.request_tokens(
                text,
                response_format=ProjectionFragmentMemoryExtractionResponse,
                model=self.extractor.model,
                images=images,
            )
            output_tokens += output
            image_count += len(images)
            image_bytes += sum(len(image.body) for image in images)
            identity = hashlib.sha256(
                (self.batch.id + selected.digest + text).encode()
            ).hexdigest()
            requests.append(
                replace(
                    self.batch,
                    id="request-" + identity,
                    prepared_catalog=selected,
                    prepared_prompt=text,
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
    cost = {
        "input_tokens": plan.estimated_cost.input_tokens,
        "output_tokens": plan.estimated_cost.output_tokens,
        "request_count": plan.estimated_cost.request_count,
        "image_count": plan.estimated_cost.image_count,
        "image_bytes": plan.estimated_cost.image_bytes,
        "total_tokens": plan.estimated_cost.total_tokens,
    }
    return tuple(
        replace(
            request,
            prepared_input_mode=plan.mode.value,
            prepared_selection_reason=plan.selection_reason,
            prepared_estimated_cost=cost,
        )
        for request in plan.transport
    )
