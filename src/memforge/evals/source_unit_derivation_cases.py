"""Pin synced Documents as Source Unit derivation cases and freeze their cohort.

A maintainer names the Documents of a workspace that a derivation cohort should
hold, by id or by their title within a Source, each with a rubric: the number of
Memories its extraction should give and terms those Memories must hold. A case
pins one derivation of the Document's Source Unit as it was staged, whether or
not its lifecycle has been applied since: the projected revision, the Document
context and the access and inference capability hashes its Evidence work was
planned with. Replay therefore runs the current extraction planner and prompts
over exactly the input production read, with no live source content.

A Document's first derivation is pinned unless the request names another of its
derivations. A later revision is derived against the committed base it updates,
so its case also pins that base revision's projection, which replay plans the
same Evidence work against. A Document that cannot be pinned is counted by
reason: it was never synced, its title names more than one Document, it has no
first derivation, the named derivation is not one of its completed
derivations, or the named derivation's base revision can no longer be read.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from memforge.evals.offline_evaluation import (
    AgentEvaluationCaseKind,
    AgentEvaluationCohortItem,
    AgentEvaluationPopulation,
    AgentEvaluationRole,
    OfflineAgentEvaluation,
)
from memforge.models import DocumentRecord
from memforge.source_derivation import (
    SOURCE_DERIVATION_APPLIED,
    SOURCE_DERIVATION_COMPLETED,
    SourceDerivationAttempt,
    source_unit_derivation_context_to_payload,
)
from memforge.source_projection import SourceProjection, SourceUnit, source_projection_to_payload

DERIVATION_CASE_POLICY_VERSION = "source-unit-derivation-cases-v1"
DERIVATION_CASE_GROUP_KEY = "source_unit_derivation"


class DerivationCaseStore(Protocol):
    async def get_document(self, doc_id: str) -> DocumentRecord | None: ...

    async def list_documents(
        self,
        source: str | None = None,
        search: str | None = None,
        limit: int = 100,
    ) -> list[DocumentRecord]: ...

    async def find_source_unit_by_document_id(
        self,
        source_id: str,
        document_id: str,
        *,
        current_only: bool = False,
    ) -> SourceUnit | None: ...

    async def list_source_derivation_attempts(
        self,
        *,
        source_id: str,
        statuses: tuple[str, ...] | None = None,
        source_unit_id: str | None = None,
    ) -> tuple[SourceDerivationAttempt, ...]: ...

    async def get_source_unit_revision_projection(
        self, source_unit_id: str, revision_id: str | None,
    ) -> SourceProjection | None: ...


class DerivationCaseSkip(str, Enum):
    DOCUMENT_NOT_FOUND = "document_not_found"
    TITLE_NOT_UNIQUE = "title_not_unique"
    NO_FIRST_DERIVATION = "no_first_derivation"
    DERIVATION_NOT_FOUND = "derivation_not_found"
    BASE_UNAVAILABLE = "base_unavailable"


@dataclass(frozen=True, slots=True)
class DerivationCaseRequest:
    """One Document to pin: ``doc_id``, or ``title`` within ``source_id``.

    ``derivation_id`` names the derivation to pin; without it, the Document's
    first derivation is pinned.
    """

    rubric: Mapping[str, object]
    doc_id: str | None = None
    source_id: str | None = None
    title: str | None = None
    derivation_id: str | None = None
    population: AgentEvaluationPopulation = AgentEvaluationPopulation.REPRESENTATIVE_CONTROL

    def __post_init__(self) -> None:
        if bool(self.doc_id) == bool(self.source_id and self.title):
            raise ValueError("a derivation case names doc_id, or source_id and title")

    @property
    def label(self) -> str:
        return self.doc_id or f"{self.source_id}:{self.title}"


@dataclass(frozen=True, slots=True)
class PinnedDerivationCase:
    doc_id: str
    case_id: str
    ground_truth_revision_id: str
    population: AgentEvaluationPopulation


@dataclass(frozen=True, slots=True)
class DerivationCaseSeedReport:
    cohort_id: str | None
    cases: tuple[PinnedDerivationCase, ...]
    skipped: Mapping[str, tuple[str, ...]]


async def seed_source_unit_derivation_cases(
    store: DerivationCaseStore,
    evaluation: OfflineAgentEvaluation,
    *,
    requests: Sequence[DerivationCaseRequest],
    role: AgentEvaluationRole,
    selection_policy_version: str,
    actor: str,
) -> DerivationCaseSeedReport:
    """Pin one case per named Document, accept its rubric, and freeze one cohort of ``role``.

    Seeding the same Documents with the same rubrics again pins the same cases
    and returns the same cohort. ``skipped`` names each Document that could not
    be pinned under its reason.
    """

    pinned: list[PinnedDerivationCase] = []
    skipped: dict[str, list[str]] = {reason.value: [] for reason in DerivationCaseSkip}
    for request in requests:
        outcome = await _pin_document_case(store, evaluation, request=request, actor=actor)
        if isinstance(outcome, DerivationCaseSkip):
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
                    group_key=DERIVATION_CASE_GROUP_KEY,
                )
                for case in pinned
            ],
            selection_policy_version=selection_policy_version,
            created_by=actor,
        )
        cohort_id = cohort.cohort_id
    return DerivationCaseSeedReport(
        cohort_id=cohort_id,
        cases=tuple(pinned),
        skipped={reason: tuple(doc_ids) for reason, doc_ids in skipped.items()},
    )


async def _pin_document_case(
    store: DerivationCaseStore,
    evaluation: OfflineAgentEvaluation,
    *,
    request: DerivationCaseRequest,
    actor: str,
) -> PinnedDerivationCase | DerivationCaseSkip:
    document = await _find_document(store, request)
    if isinstance(document, DerivationCaseSkip):
        return document
    unit = await store.find_source_unit_by_document_id(document.source, document.doc_id)
    if unit is None:
        return DerivationCaseSkip.NO_FIRST_DERIVATION
    planned = [
        attempt
        for attempt in await store.list_source_derivation_attempts(
            source_id=document.source,
            statuses=(SOURCE_DERIVATION_COMPLETED, SOURCE_DERIVATION_APPLIED),
            source_unit_id=unit.id,
        )
        if attempt.authority_plan_identity is not None
    ]
    if request.derivation_id:
        named = [attempt for attempt in planned if attempt.id == request.derivation_id]
        if not named:
            return DerivationCaseSkip.DERIVATION_NOT_FOUND
        attempt = named[0]
    else:
        first = [attempt for attempt in planned if attempt.base_unit_revision_id is None]
        if not first:
            return DerivationCaseSkip.NO_FIRST_DERIVATION
        attempt = first[-1]
    manifest: dict[str, object] = {
        "projection": source_projection_to_payload(attempt.projection),
        "context": source_unit_derivation_context_to_payload(attempt.context),
        "access_context_hash": str(attempt.authority_plan_identity.get("access_context_hash") or ""),
        "inference_capability_hash": str(attempt.authority_plan_identity.get("inference_capability_hash") or ""),
        "derivation_id": attempt.id,
    }
    if attempt.base_unit_revision_id is not None:
        base = await store.get_source_unit_revision_projection(attempt.source_unit_id, attempt.base_unit_revision_id)
        if base is None:
            return DerivationCaseSkip.BASE_UNAVAILABLE
        manifest["base_projection"] = source_projection_to_payload(base)
    case = await evaluation.curate_case(
        case_kind=AgentEvaluationCaseKind.SOURCE_UNIT_DERIVATION,
        source_id=attempt.source_id,
        doc_id=document.doc_id,
        source_unit_id=attempt.source_unit_id,
        manifest=manifest,
        promotion_policy_version=DERIVATION_CASE_POLICY_VERSION,
        created_by=actor,
    )
    revision = await evaluation.accept_ground_truth(
        case_id=case.case_id,
        rubric=request.rubric,
        accepted_by=actor,
    )
    return PinnedDerivationCase(
        doc_id=document.doc_id,
        case_id=case.case_id,
        ground_truth_revision_id=revision.ground_truth_revision_id,
        population=request.population,
    )


async def _find_document(
    store: DerivationCaseStore,
    request: DerivationCaseRequest,
) -> DocumentRecord | DerivationCaseSkip:
    if request.doc_id:
        document = await store.get_document(request.doc_id)
        return document if document is not None else DerivationCaseSkip.DOCUMENT_NOT_FOUND
    titled = [
        document
        for document in await store.list_documents(source=request.source_id, search=request.title, limit=2_000)
        if document.title == request.title
    ]
    if not titled:
        return DerivationCaseSkip.DOCUMENT_NOT_FOUND
    if len(titled) > 1:
        return DerivationCaseSkip.TITLE_NOT_UNIQUE
    return titled[0]

