"""Plan one Source Unit revision's Support work: what each incumbent's Support is read against.

Each incumbent Memory gets one work item per Evidence Unit that supports it in
the Unit. An item is assessed against its Support's validation baseline: the
committed revision it was last validated on, or the whole current revision when
no usable baseline exists. Production lifecycle and offline Support replay plan
the same work through this one function, over the store or over a pinned case.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from memforge.memory.evidence import ActiveSupportEvidence
from memforge.models import Memory
from memforge.pipeline.revision_assessment import RevisionAssessmentContext
from memforge.pipeline.support_reading import SupportWorkItem
from memforge.pipeline.support_relation_coordinator import ReconciliationContractError
from memforge.source_projection import SourceObservationRevision, SourceProjection

logger = logging.getLogger(__name__)


class SupportBaselineStore(Protocol):
    """The stored revisions Support work reads besides the target and its committed base."""

    async def get_source_unit_revision_projection(
        self, source_unit_id: str, revision_id: str | None,
    ) -> SourceProjection | None: ...

    async def get_source_observation_revisions(
        self, revision_ids: Collection[str],
    ) -> Mapping[str, SourceObservationRevision]: ...


@dataclass(frozen=True)
class SupportWork:
    """The work items in assessment order, by Memory, and the contexts they read.

    ``context`` reads against the committed base; ``context_for`` builds one
    against another baseline that shares its Fragment indexes.
    """

    items: tuple[SupportWorkItem, ...]
    by_memory: Mapping[str, tuple[SupportWorkItem, ...]]
    context: RevisionAssessmentContext
    context_for: Callable[[SourceProjection | None], RevisionAssessmentContext]
    unusable_baseline_count: int
    reprocess_count: int


async def plan_support_work(
    store: SupportBaselineStore,
    *,
    projection: SourceProjection,
    base: SourceProjection | None,
    incumbents: Sequence[Memory],
    evidence_by_memory: Mapping[str, Sequence[ActiveSupportEvidence]],
    unit_support: Mapping[str, Collection[str]],
    access_context_hash: str,
    image_loader=None,
    reads_images: bool = True,
    without_baseline: bool = False,
) -> SupportWork:
    """Plan the work of ``incumbents`` in order, each Support in the order its Evidence is listed.

    ``unit_support`` names each incumbent's Evidence Units in this Source Unit;
    ``without_baseline`` reads every Support against the whole current
    revision, as an operator reprocess does.
    """

    source_unit_id = projection.source_unit_revisions[0].source_unit_id
    # Each part is classified by the revision it names, whichever baseline its Support has.
    evidence_revisions = tuple((await store.get_source_observation_revisions({
        item.anchor.observation_revision_id
        for items in evidence_by_memory.values() for item in items
    } - {revision.id for revision in projection.observation_revisions})).values())
    context = RevisionAssessmentContext(
        projection=projection, base=base, access_context_hash=access_context_hash,
        image_loader=image_loader, evidence_revisions=evidence_revisions, reads_images=reads_images,
    )

    def context_for(baseline: SourceProjection | None) -> RevisionAssessmentContext:
        """Assess against another baseline, sharing this operation's Fragment indexes.

        The committed revision still describes carried Observations the baseline may predate.
        """
        return RevisionAssessmentContext(
            projection=projection, base=baseline, access_context_hash=access_context_hash,
            image_loader=image_loader, indexes=context.indexes,
            known_observations=base.observations if base is not None else (),
            evidence_revisions=tuple(context.revisions.values()),
            reads_images=reads_images,
        )

    contexts_by_revision = {base.source_unit_revisions[0].id: context} if base else {}
    items: list[SupportWorkItem] = []
    by_memory: dict[str, list[SupportWorkItem]] = {}
    unusable_count = reprocess_count = 0
    for memory in incumbents:
        groups: dict[str, list[ActiveSupportEvidence]] = {}
        for item in evidence_by_memory.get(memory.id, ()):
            if item.evidence_unit_id in unit_support.get(memory.id, ()):
                groups.setdefault(item.evidence_unit_id, []).append(item)
        if not groups:
            raise ReconciliationContractError("revision_support_missing", "incumbent has no complete scoped support")
        by_memory[memory.id] = []
        for evidence_unit_id, support in groups.items():
            if without_baseline:
                baseline_id, unusable = None, None
                reprocess_count += 1
            else:
                baseline_id, unusable = support_validation_baseline(support)
            item_context = contexts_by_revision.get(baseline_id) if unusable is None else None
            if item_context is None and baseline_id is not None:
                historical = await store.get_source_unit_revision_projection(source_unit_id, baseline_id)
                if (
                    historical is None
                    or historical.source_id != projection.source_id
                    or len(historical.source_unit_revisions) != 1
                    or historical.source_unit_revisions[0].source_unit_id != source_unit_id
                    or historical.source_unit_revisions[0].id != baseline_id
                ):
                    unusable = "validation snapshot is unavailable or inconsistent"
                else:
                    item_context = context_for(historical)
                    contexts_by_revision[baseline_id] = item_context
            if item_context is None:
                # No usable baseline: the whole current revision is read without
                # assuming the old Support is still valid. A named but unusable
                # baseline is a data defect worth a diagnostic, not a failure.
                if unusable is not None:
                    logger.warning(
                        "support_baseline_unusable memory_id=%s evidence_unit_id=%s "
                        "baseline_unit_revision_ids=%s reason=%s",
                        memory.id, evidence_unit_id,
                        ",".join(sorted({str(item.validation_unit_revision_id) for item in support})),
                        unusable,
                    )
                    unusable_count += 1
                item_context = contexts_by_revision.get(None)
                if item_context is None:
                    item_context = context_for(None)
                    contexts_by_revision[None] = item_context
            work_item = SupportWorkItem(f"w{len(items):06d}", memory, tuple(support), item_context)
            items.append(work_item)
            by_memory[memory.id].append(work_item)
    return SupportWork(
        items=tuple(items),
        by_memory={memory_id: tuple(memory_items) for memory_id, memory_items in by_memory.items()},
        context=context,
        context_for=context_for,
        unusable_baseline_count=unusable_count,
        reprocess_count=reprocess_count,
    )


def support_validation_baseline(support: Sequence[ActiveSupportEvidence]) -> tuple[str | None, str | None]:
    """Return one Evidence Unit's recorded baseline Source Unit revision, or why it is unusable.

    ``(None, None)`` means no baseline was ever recorded for this Support.
    """
    plan_ids = {item.validation_plan_id for item in support}
    baseline_ids = {item.validation_unit_revision_id for item in support}
    if len(plan_ids) != 1 or len(baseline_ids) != 1:
        return None, "validation association is inconsistent"
    baseline_id = next(iter(baseline_ids))
    if next(iter(plan_ids)) is not None and baseline_id is None:
        return None, "validation Plan is not applied to this Source Unit"
    return baseline_id, None
