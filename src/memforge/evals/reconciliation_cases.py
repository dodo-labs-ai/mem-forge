"""Import hand-written reconciliation cases and freeze their cohort.

A reconciliation case fixes one situation: the old Memories of a Source Unit,
the new Candidates an update extracted with their Evidence, and each old
Memory's Support result in that update. Production's receipts record these
inputs only as request hashes, so cases are written by hand rather than read
back from a sync, and only the model varies between runs.

A case is imported against a synced Document, which scopes it to that
Document's Source and Source Unit like any other case. Its Memory ids and its
Evidence observation and revision labels are local to the case: they are what
the model sees and are never resolved against the Source. Import turns each
Candidate's Evidence into the stored Candidate format, so replay renders it
exactly as production renders a Candidate's Evidence.

The input names ``candidates`` (claim, ``memory_type``, optional
``valid_from``/``valid_until`` and ``evidence``: text parts, each a ``role`` of
``primary`` or ``required``, an ``excerpt`` and ``observation``/``revision``
labels, exactly one ``primary``), ``incumbents`` (``id``, claim, ``memory_type``,
optional validity) and ``support`` (incumbent id to its Support result, or to
``{"result": ..., "complete_read": false}`` for an unsupported read that did not
cover the whole revision).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from enum import Enum

from memforge.derivation_work import payload_hash
from memforge.evals.offline_evaluation import (
    AgentEvaluationCaseKind,
    AgentEvaluationCohortItem,
    AgentEvaluationPopulation,
    AgentEvaluationRole,
    OfflineAgentEvaluation,
)
from memforge.evals.source_unit_derivation_cases import (
    DerivationCaseRequest,
    DerivationCaseSkip,
    DerivationCaseStore,
    find_case_document,
)
from memforge.memory.evidence import EvidencePartKind, EvidenceRole, ResolvedEvidencePart, ResolvedEvidenceSelection
from memforge.models import MemoryExtractionResult, MemoryType, RawMemory
from memforge.pipeline.evidence_fragments import COMPILER_CONTRACT_VERSION
from memforge.source_derivation import memory_extraction_output_payload
from memforge.source_projection import AnchorKind, SourceAnchor

RECONCILIATION_CASE_POLICY_VERSION = "source-unit-reconciliation-cases-v1"
RECONCILIATION_CASE_GROUP_KEY = "source_unit_reconciliation"
# Names the selection identity of hand-written Evidence, which no Source revision or catalog produced.
_HAND_WRITTEN = "hand-written"


class ReconciliationCaseSkip(str, Enum):
    DOCUMENT_NOT_FOUND = "document_not_found"
    TITLE_NOT_UNIQUE = "title_not_unique"
    NOT_DERIVED = "not_derived"


@dataclass(frozen=True, slots=True)
class ReconciliationCaseRequest:
    """One hand-written situation imported against a synced Document: ``doc_id``, or ``title`` within ``source_id``.

    ``input`` holds the situation (see the module docstring); ``rubric`` holds
    ``expected_incumbents``: incumbent id to ``kept``, ``review``, ``revised`` or
    ``retired``.
    """

    input: Mapping[str, object]
    rubric: Mapping[str, object]
    doc_id: str | None = None
    source_id: str | None = None
    title: str | None = None
    population: AgentEvaluationPopulation = AgentEvaluationPopulation.REPRESENTATIVE_CONTROL

    def __post_init__(self) -> None:
        if bool(self.doc_id) == bool(self.source_id and self.title):
            raise ValueError("a reconciliation case names doc_id, or source_id and title")

    @property
    def label(self) -> str:
        return self.doc_id or f"{self.source_id}:{self.title}"


@dataclass(frozen=True, slots=True)
class ImportedReconciliationCase:
    doc_id: str
    case_id: str
    ground_truth_revision_id: str
    population: AgentEvaluationPopulation


@dataclass(frozen=True, slots=True)
class ReconciliationCaseSeedReport:
    cohort_id: str | None
    cases: tuple[ImportedReconciliationCase, ...]
    skipped: Mapping[str, tuple[str, ...]]


async def seed_reconciliation_cases(
    store: DerivationCaseStore,
    evaluation: OfflineAgentEvaluation,
    *,
    requests: Sequence[ReconciliationCaseRequest],
    role: AgentEvaluationRole,
    selection_policy_version: str,
    actor: str,
) -> ReconciliationCaseSeedReport:
    """Import one case per request, accept its rubric, and freeze one cohort of ``role``.

    Importing the same situations with the same rubrics again pins the same
    cases and returns the same cohort.
    """

    imported: list[ImportedReconciliationCase] = []
    skipped: dict[str, list[str]] = {reason.value: [] for reason in ReconciliationCaseSkip}
    for request in requests:
        document = await find_case_document(store, DerivationCaseRequest(
            rubric=request.rubric, doc_id=request.doc_id, source_id=request.source_id, title=request.title,
        ))
        if isinstance(document, DerivationCaseSkip):
            skipped[document.value].append(request.label)
            continue
        unit = await store.find_source_unit_by_document_id(document.source, document.doc_id)
        if unit is None:
            skipped[ReconciliationCaseSkip.NOT_DERIVED.value].append(request.label)
            continue
        case = await evaluation.curate_case(
            case_kind=AgentEvaluationCaseKind.SOURCE_UNIT_RECONCILIATION,
            source_id=document.source,
            doc_id=document.doc_id,
            source_unit_id=unit.id,
            manifest=reconciliation_manifest(request.input, source_id=document.source, source_unit_id=unit.id),
            promotion_policy_version=RECONCILIATION_CASE_POLICY_VERSION,
            created_by=actor,
        )
        revision = await evaluation.accept_ground_truth(case_id=case.case_id, rubric=request.rubric, accepted_by=actor)
        imported.append(ImportedReconciliationCase(
            doc_id=document.doc_id,
            case_id=case.case_id,
            ground_truth_revision_id=revision.ground_truth_revision_id,
            population=request.population,
        ))
    cohort_id = None
    if imported:
        cohort = await evaluation.freeze_cohort(
            items=[
                AgentEvaluationCohortItem(
                    case_id=case.case_id,
                    ground_truth_revision_id=case.ground_truth_revision_id,
                    population=case.population,
                    role=role,
                    group_key=RECONCILIATION_CASE_GROUP_KEY,
                )
                for case in imported
            ],
            selection_policy_version=selection_policy_version,
            created_by=actor,
        )
        cohort_id = cohort.cohort_id
    return ReconciliationCaseSeedReport(
        cohort_id=cohort_id,
        cases=tuple(imported),
        skipped={reason: tuple(labels) for reason, labels in skipped.items()},
    )


def reconciliation_manifest(
    situation: Mapping[str, object], *, source_id: str, source_unit_id: str,
) -> dict[str, object]:
    """The replay manifest of one hand-written situation within the named Source Unit."""

    candidates = _list(situation, "candidates")
    incumbents = _list(situation, "incumbents")
    support = situation.get("support")
    if not isinstance(support, Mapping):
        raise ValueError("support must map each incumbent id to its Support result")
    return {
        "new_extractions": memory_extraction_output_payload(MemoryExtractionResult(memories=[
            _candidate(candidate, source_id=source_id, source_unit_id=source_unit_id) for candidate in candidates
        ]))["memories"],
        "incumbents": [
            {
                "id": _text(incumbent, "id"),
                "content": _text(incumbent, "claim"),
                "memory_type": _memory_type(incumbent),
                "valid_from": _date(incumbent, "valid_from"),
                "valid_until": _date(incumbent, "valid_until"),
            }
            for incumbent in incumbents
        ],
        "support_audits": [
            {"incumbent_id": str(incumbent_id), **(dict(result) if isinstance(result, Mapping) else {"result": result})}
            for incumbent_id, result in support.items()
        ],
    }


def _candidate(candidate: Mapping[str, object], *, source_id: str, source_unit_id: str) -> RawMemory:
    parts = []
    for part in _list(candidate, "evidence"):
        excerpt = _text(part, "excerpt")
        digest = hashlib.sha256(excerpt.encode()).hexdigest()
        parts.append(ResolvedEvidencePart(
            role=EvidenceRole(_text(part, "role")),
            kind=EvidencePartKind.TEXT,
            anchor=SourceAnchor(
                kind=AnchorKind.REVISION_RANGE,
                observation_id=_text(part, "observation"),
                observation_revision_id=_text(part, "revision"),
                range_start=0,
                range_end=len(excerpt),
            ),
            raw_content_sha256=digest,
            presentation_sha256=digest,
            excerpt=excerpt,
        ))
    return RawMemory(
        content=_text(candidate, "claim"),
        memory_type=_memory_type(candidate),
        valid_from=_date(candidate, "valid_from"),
        valid_until=_date(candidate, "valid_until"),
        resolved_evidence_selection=ResolvedEvidenceSelection(
            source_id=source_id,
            source_unit_id=source_unit_id,
            target_unit_revision_id=_HAND_WRITTEN,
            access_context_hash=_HAND_WRITTEN,
            catalog_digest=payload_hash(_list(candidate, "evidence")),
            compiler_contract_version=COMPILER_CONTRACT_VERSION,
            parts=tuple(parts),
        ),
    )


def _list(payload: Mapping[str, object], key: str) -> list[Mapping[str, object]]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise ValueError(f"{key} must be a list of objects")
    return value


def _memory_type(payload: Mapping[str, object]) -> str:
    value = _text(payload, "memory_type")
    if value not in {memory_type.value for memory_type in MemoryType}:
        raise ValueError("memory_type must be one of " + ", ".join(memory_type.value for memory_type in MemoryType))
    return value


def _date(payload: Mapping[str, object], key: str) -> str | None:
    """An optional ISO date, checked so that a typo cannot pass as an open validity."""

    value = payload.get(key)
    if value is None:
        return None
    try:
        return date.fromisoformat(str(value)).isoformat()
    except ValueError:
        raise ValueError(f"{key} must be an ISO date") from None


def _text(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be non-empty text")
    return value
