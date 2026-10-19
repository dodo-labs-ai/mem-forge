"""Pin synced Documents as Source Unit derivation cases and freeze their cohort.

A maintainer names the Documents of a workspace that a derivation cohort should
hold, by id or by their title within a Source, each with a rubric: the number of
Memories its extraction should give and terms those Memories must hold. A case
pins the first derivation of the Document's Source Unit as it was staged,
whether or not its lifecycle has been applied since: the projected revision, the
Document context and the access and inference capability hashes its Evidence
work was planned with. Replay therefore runs the current extraction planner and
prompts over exactly the input production read, with no live source content.

Only a first derivation is pinned. A later revision is derived against the
committed base it updates, which a case does not pin, so replay could not plan
its Evidence work. A Document that has none is counted by reason, as is one
that was never synced and a title that names more than one Document.
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
from memforge.source_projection import SourceUnit, source_projection_to_payload

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


class DerivationCaseSkip(str, Enum):
    DOCUMENT_NOT_FOUND = "document_not_found"
    TITLE_NOT_UNIQUE = "title_not_unique"
    NO_FIRST_DERIVATION = "no_first_derivation"


@dataclass(frozen=True, slots=True)
class DerivationCaseRequest:
    """One Document to pin: ``doc_id``, or ``title`` within ``source_id``."""

    rubric: Mapping[str, object]
    doc_id: str | None = None
    source_id: str | None = None
    title: str | None = None
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
    first = [
        attempt
        for attempt in await store.list_source_derivation_attempts(
            source_id=document.source,
            statuses=(SOURCE_DERIVATION_COMPLETED, SOURCE_DERIVATION_APPLIED),
            source_unit_id=unit.id,
        )
        if attempt.base_unit_revision_id is None and attempt.authority_plan_identity is not None
    ]
    if not first:
        return DerivationCaseSkip.NO_FIRST_DERIVATION
    attempt = first[-1]
    authority = attempt.authority_plan_identity
    case = await evaluation.curate_case(
        case_kind=AgentEvaluationCaseKind.SOURCE_UNIT_DERIVATION,
        source_id=attempt.source_id,
        doc_id=document.doc_id,
        source_unit_id=attempt.source_unit_id,
        manifest={
            "projection": source_projection_to_payload(attempt.projection),
            "context": source_unit_derivation_context_to_payload(attempt.context),
            "access_context_hash": str(authority.get("access_context_hash") or ""),
            "inference_capability_hash": str(authority.get("inference_capability_hash") or ""),
            "derivation_id": attempt.id,
        },
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
