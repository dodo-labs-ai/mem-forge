"""Pin applied revision updates as Support Assessment cases and freeze their cohort.

When a Document's new revision is applied, every Support the model judged
leaves receipts in the derivation work journal: the target and baseline
revisions, the Fragment catalog digest, each work item's claim as the model saw
it and its Evidence parts, and the result. A case pins one applied update from
those receipts: the derivation's own projection as synced, its committed base,
the validation baselines the receipts used, the stored Observation revisions the
parts name, and every work item the model judged in the first pass, in recorded
order with its recorded result.

Program rebinds and Supports kept unresolved under partial coverage never reach
the model in the first pass. A coordinator re-check of one is read like a first
pass, under the same work id, so seeding replans every receipt item and keeps
only those the first pass sent to the model; a Change Impact result is the
first-pass result of the item it judged. A re-check against a Candidate's
Evidence names its reading in its scope and is not pinned.

Each part is read back with its stored columns, as production read it.
Seeding then rebuilds the work items through the shared Support work planner
and requires every item's baseline and Fragment catalog digest to match its
receipt, so a case holds exactly the input production read. Receipts written
under another Support or Change Impact contract are not pinned. A Document
whose update cannot be pinned is counted by reason.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from enum import Enum
from typing import Protocol

from memforge.derivation_work import DerivationWork
from memforge.evals.offline_evaluation import (
    AgentEvaluationCaseKind,
    AgentEvaluationCohortItem,
    AgentEvaluationPopulation,
    AgentEvaluationRole,
    OfflineAgentEvaluation,
    PinnedSupportInput,
    PinnedSupportWorkItem,
)
from memforge.evals.source_unit_derivation_cases import (
    DerivationCaseRequest,
    DerivationCaseSkip,
    DerivationCaseStore,
    find_case_document,
)
from memforge.memory.evidence import ActiveSupportEvidence, EvidenceReference
from memforge.models import Memory, content_hash
from memforge.pipeline.revision_work import CHANGE_IMPACT_CONTRACT, SUPPORT_ASSESSMENT_CONTRACT
from memforge.pipeline.support_reading import SupportRoute, plan_support_revision
from memforge.pipeline.support_relation_coordinator import ReconciliationContractError
from memforge.source_derivation import SOURCE_DERIVATION_APPLIED, SourceDerivationAttempt
from memforge.source_projection import SourceObservationRevision

SUPPORT_CASE_POLICY_VERSION = "source-unit-support-cases-v1"
SUPPORT_CASE_GROUP_KEY = "source_unit_support"
# The contract each journal kind that lists judged Support work items is written under.
_MODEL_WORK_CONTRACTS = {"support_assess": SUPPORT_ASSESSMENT_CONTRACT, "change_impact": CHANGE_IMPACT_CONTRACT}
_MODEL_ROUTES = frozenset({SupportRoute.CHANGE_IMPACT, SupportRoute.SUPPORT_ASSESSMENT})


class SupportCaseStore(DerivationCaseStore, Protocol):
    async def list_source_derivation_work(self, derivation_id: str) -> tuple[DerivationWork, ...]: ...

    async def get_evidence_references(self, reference_ids: Collection[str]) -> Mapping[str, EvidenceReference]: ...

    async def get_source_observation_revisions(
        self, revision_ids: Collection[str],
    ) -> Mapping[str, SourceObservationRevision]: ...


class SupportCaseSkip(str, Enum):
    DOCUMENT_NOT_FOUND = "document_not_found"
    TITLE_NOT_UNIQUE = "title_not_unique"
    NO_UPDATE = "no_update"
    DERIVATION_NOT_FOUND = "derivation_not_found"
    NO_SUPPORT_WORK = "no_support_work"
    REVISION_UNAVAILABLE = "revision_unavailable"
    EVIDENCE_UNAVAILABLE = "evidence_unavailable"
    CONTRACT_CHANGED = "contract_changed"
    CATALOG_MISMATCH = "catalog_mismatch"


@dataclass(frozen=True, slots=True)
class SupportCaseRequest:
    """One Document whose applied update to pin: ``doc_id``, or ``title`` within ``source_id``.

    ``derivation_id`` names the update; without it, the Document's latest
    applied update whose Support the model judged is pinned. ``rubric`` holds
    ``expectations``: sentences about the Support verdicts, for the expectation
    judge.
    """

    rubric: Mapping[str, object]
    doc_id: str | None = None
    source_id: str | None = None
    title: str | None = None
    derivation_id: str | None = None
    population: AgentEvaluationPopulation = AgentEvaluationPopulation.REPRESENTATIVE_CONTROL

    def __post_init__(self) -> None:
        if bool(self.doc_id) == bool(self.source_id and self.title):
            raise ValueError("a Support case names doc_id, or source_id and title")

    @property
    def label(self) -> str:
        return self.doc_id or f"{self.source_id}:{self.title}"


@dataclass(frozen=True, slots=True)
class PinnedSupportCase:
    doc_id: str
    derivation_id: str
    case_id: str
    ground_truth_revision_id: str
    population: AgentEvaluationPopulation


@dataclass(frozen=True, slots=True)
class SupportCaseSeedReport:
    cohort_id: str | None
    cases: tuple[PinnedSupportCase, ...]
    skipped: Mapping[str, tuple[str, ...]]


async def seed_source_unit_support_cases(
    store: SupportCaseStore,
    evaluation: OfflineAgentEvaluation,
    *,
    requests: Sequence[SupportCaseRequest],
    role: AgentEvaluationRole,
    selection_policy_version: str,
    actor: str,
) -> SupportCaseSeedReport:
    """Pin one case per named Document, accept its rubric, and freeze one cohort of ``role``.

    Seeding the same updates with the same rubrics again pins the same cases
    and returns the same cohort.
    """

    pinned: list[PinnedSupportCase] = []
    skipped: dict[str, list[str]] = {reason.value: [] for reason in SupportCaseSkip}
    for request in requests:
        outcome = await _pin_update_case(store, evaluation, request=request, actor=actor)
        if isinstance(outcome, SupportCaseSkip):
            skipped[outcome.value].append(request.label)
        else:
            pinned.append(outcome)
    cohort_id = None
    if pinned:
        cohort = await evaluation.freeze_cohort(
            items=[
                AgentEvaluationCohortItem(
                    case_id=case.case_id,
                    ground_truth_revision_id=case.ground_truth_revision_id,
                    population=case.population,
                    role=role,
                    group_key=SUPPORT_CASE_GROUP_KEY,
                )
                for case in pinned
            ],
            selection_policy_version=selection_policy_version,
            created_by=actor,
        )
        cohort_id = cohort.cohort_id
    return SupportCaseSeedReport(
        cohort_id=cohort_id,
        cases=tuple(pinned),
        skipped={reason: tuple(labels) for reason, labels in skipped.items()},
    )


async def _pin_update_case(
    store: SupportCaseStore,
    evaluation: OfflineAgentEvaluation,
    *,
    request: SupportCaseRequest,
    actor: str,
) -> PinnedSupportCase | SupportCaseSkip:
    document = await find_case_document(
        store, DerivationCaseRequest(rubric=request.rubric, doc_id=request.doc_id,
                                     source_id=request.source_id, title=request.title),
    )
    if isinstance(document, DerivationCaseSkip):
        return SupportCaseSkip(document.value)
    unit = await store.find_source_unit_by_document_id(document.source, document.doc_id)
    if unit is None:
        return SupportCaseSkip.NO_UPDATE
    applied = await store.list_source_derivation_attempts(
        source_id=document.source, statuses=(SOURCE_DERIVATION_APPLIED,), source_unit_id=unit.id,
    )
    if request.derivation_id:
        named = [attempt for attempt in applied if attempt.id == request.derivation_id]
        if not named:
            return SupportCaseSkip.DERIVATION_NOT_FOUND
        attempts = named
    else:
        attempts = [attempt for attempt in reversed(applied) if attempt.base_unit_revision_id is not None]
        if not attempts:
            return SupportCaseSkip.NO_UPDATE
    # The latest update whose Support the model judged, or the named one.
    for attempt in attempts:
        pinned = await _pinned_input(store, attempt)
        if pinned is not SupportCaseSkip.NO_SUPPORT_WORK:
            break
    if isinstance(pinned, SupportCaseSkip):
        return pinned
    case = await evaluation.curate_case(
        case_kind=AgentEvaluationCaseKind.SOURCE_UNIT_SUPPORT,
        source_id=attempt.source_id,
        doc_id=document.doc_id,
        source_unit_id=attempt.source_unit_id,
        manifest=pinned.to_manifest(),
        promotion_policy_version=SUPPORT_CASE_POLICY_VERSION,
        created_by=actor,
    )
    revision = await evaluation.accept_ground_truth(case_id=case.case_id, rubric=request.rubric, accepted_by=actor)
    return PinnedSupportCase(
        doc_id=document.doc_id,
        derivation_id=attempt.id,
        case_id=case.case_id,
        ground_truth_revision_id=revision.ground_truth_revision_id,
        population=request.population,
    )


@dataclass(frozen=True, slots=True)
class _ReceiptItem:
    identity: Mapping[str, object]
    baseline: str | None
    catalog: str


def _receipt_items(works: Sequence[DerivationWork]) -> dict[str, _ReceiptItem] | SupportCaseSkip:
    """Each numbered work item a model call judged, by its recorded work id.

    A single re-check of a claim is not numbered as a work item, and a re-check
    against Candidate Evidence names its reading in its scope.
    """

    items: dict[str, _ReceiptItem] = {}
    for work in works:
        scope = work.manifest.get("scope")
        if work.kind not in _MODEL_WORK_CONTRACTS or not isinstance(scope, Mapping) or "reading" in scope:
            continue
        if scope.get("contract") != _MODEL_WORK_CONTRACTS[work.kind]:
            return SupportCaseSkip.CONTRACT_CHANGED
        for identity in scope["work_items"]:
            work_id = str(identity["claim"]["work_id"])
            if work_id.startswith("w"):
                items[work_id] = _ReceiptItem(identity, scope["baseline"], str(scope["catalog"]))
    return items


def _observed_results(works: Sequence[DerivationWork]) -> dict[str, str]:
    """The recorded first-pass result of each work item.

    Change Impact judges an item only in the first pass, and an item it found
    unaffected is read again only by a re-check, so its result stands.
    """

    read: dict[str, str] = {}
    unaffected: set[str] = set()
    for work in works:
        scope = work.manifest.get("scope")
        if work.kind != "support_finalize" or not isinstance(scope, Mapping) or "reading" in scope:
            continue
        for result in (work.result or {}).get("results", ()):
            if work.manifest.get("contract") == CHANGE_IMPACT_CONTRACT:
                unaffected.add(str(result["work_id"]))
            elif work.manifest.get("contract") == SUPPORT_ASSESSMENT_CONTRACT:
                read[str(result["work_id"])] = str(result["verdict"])
    return {**read, **{work_id: "unaffected" for work_id in unaffected}}


async def _pinned_input(
    store: SupportCaseStore,
    attempt: SourceDerivationAttempt,
) -> PinnedSupportInput | SupportCaseSkip:
    works = await store.list_source_derivation_work(attempt.id)
    receipt_items = _receipt_items(works)
    if isinstance(receipt_items, SupportCaseSkip):
        return receipt_items
    if not receipt_items:
        return SupportCaseSkip.NO_SUPPORT_WORK
    observed = _observed_results(works)
    projection = attempt.projection
    unit_id = attempt.source_unit_id
    base = None
    if attempt.base_unit_revision_id is not None:
        base = await store.get_source_unit_revision_projection(unit_id, attempt.base_unit_revision_id)
        if base is None:
            return SupportCaseSkip.REVISION_UNAVAILABLE
    baselines = []
    for baseline_id in sorted({str(item.baseline) for item in receipt_items.values() if item.baseline}):
        if base is not None and baseline_id == base.source_unit_revisions[0].id:
            continue
        baseline = await store.get_source_unit_revision_projection(unit_id, baseline_id)
        if baseline is None:
            return SupportCaseSkip.REVISION_UNAVAILABLE
        baselines.append(baseline)

    references = await store.get_evidence_references({
        str(part[1]) for item in receipt_items.values() for part in item.identity["support"]
    })
    work_items = []
    for work_id in sorted(receipt_items):
        identity = receipt_items[work_id].identity
        memory_id = str(identity["memory_id"])
        support = []
        for evidence_unit_id, reference_id, revision_id, plan_id, baseline_id in identity["support"]:
            reference = references.get(reference_id)
            if (
                reference is None
                or reference.evidence_unit_id != evidence_unit_id
                or reference.anchor.observation_revision_id != revision_id
            ):
                return SupportCaseSkip.EVIDENCE_UNAVAILABLE
            support.append(ActiveSupportEvidence(
                memory_id=memory_id, source_id=attempt.source_id, reference_id=reference_id,
                evidence_unit_id=evidence_unit_id, role=reference.role, anchor=reference.anchor,
                excerpt=reference.excerpt, raw_content_sha256=reference.raw_content_sha256,
                presentation_sha256=reference.presentation_sha256, text_view=reference.text_view,
                validation_plan_id=plan_id, validation_unit_revision_id=baseline_id,
                display_text=reference.display_text,
            ))
        claim = identity["claim"]
        work_items.append(PinnedSupportWorkItem(
            recorded_work_id=work_id,
            memory=Memory(
                id=memory_id,
                memory_type=str(claim["memory_type"]),
                content=str(claim["claim"]),
                content_hash=content_hash(str(claim["claim"])),
                valid_from=date.fromisoformat(claim["valid_from"]) if claim.get("valid_from") else None,
                valid_until=date.fromisoformat(claim["valid_until"]) if claim.get("valid_until") else None,
            ),
            support=tuple(support),
            observed=observed.get(work_id, "unjudged"),
        ))
    named = {part.anchor.observation_revision_id for item in work_items for part in item.support}
    evidence_revisions = await store.get_source_observation_revisions(
        named - {revision.id for revision in projection.observation_revisions}
    )
    pinned = PinnedSupportInput(
        derivation_id=attempt.id,
        access_context_hash=str((attempt.authority_plan_identity or {}).get("access_context_hash") or ""),
        without_baseline=attempt.context.support_without_baseline,
        reads_images=None,
        projection=projection,
        base=base,
        baselines=tuple(baselines),
        evidence_revisions=tuple(evidence_revisions[revision_id] for revision_id in sorted(evidence_revisions)),
        work_items=tuple(work_items),
    )
    return await _first_pass_as_recorded(pinned, receipt_items)


async def _first_pass_as_recorded(
    pinned: PinnedSupportInput,
    receipt_items: Mapping[str, _ReceiptItem],
) -> PinnedSupportInput | SupportCaseSkip:
    """The pinned input restricted to the items its first pass sent to the model, or why it is not exact.

    Every item must be planned against the baseline and Fragment catalog its
    receipt names. The catalog depends on whether the reading model saw
    images, which receipts do not record: the reading that matches is pinned,
    and none is pinned when both match.
    """

    matching = []
    for reads_images in (True, False):
        try:
            work = await pinned.plan(projection=pinned.projection, reads_images=reads_images)
        except ValueError:
            return SupportCaseSkip.CATALOG_MISMATCH
        if all(
            (item.context.base.source_unit_revisions[0].id if item.context.base else None)
            == receipt_items[pinned_item.recorded_work_id].baseline
            and item.context.catalog(item.context.full_fragments).digest
            == receipt_items[pinned_item.recorded_work_id].catalog
            for pinned_item, item in zip(pinned.work_items, work.items, strict=True)
        ):
            matching.append((reads_images, work))
    if not matching:
        return SupportCaseSkip.CATALOG_MISMATCH
    reads_images, work = matching[0]
    routes = {}
    for item in work.items:
        routes.setdefault(id(item.context), (item.context, []))[1].append(item)
    try:
        judged = {
            support.item.id
            for context, items in routes.values()
            for support in plan_support_revision(context, items).supports
            if support.route in _MODEL_ROUTES
        }
    except ReconciliationContractError:
        return SupportCaseSkip.CATALOG_MISMATCH
    except ValueError:
        # A part's digest no longer matches the revision content it names.
        return SupportCaseSkip.EVIDENCE_UNAVAILABLE
    kept = tuple(
        pinned_item
        for pinned_item, item in zip(pinned.work_items, work.items, strict=True)
        if item.id in judged
    )
    if not kept:
        return SupportCaseSkip.NO_SUPPORT_WORK
    first_pass = replace(pinned, work_items=kept, reads_images=reads_images if len(matching) == 1 else None)
    return PinnedSupportInput.from_manifest(first_pass.to_manifest())
