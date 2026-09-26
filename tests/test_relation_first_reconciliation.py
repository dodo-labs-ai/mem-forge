"""Relation-first reconciliation contracts."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json

from tests.revision_client_fixture import (
    RevisionClientFixture,
    RevisionProof,
    RevisionProofs,
)

import pytest

from memforge.llm.structured import (
    MemoryRelationDecision,
    MemoryRelationResponse,
    StructuredLlmError,
)
from memforge.memory.evidence import (
    EvidencePartKind,
    EvidenceRole,
    RelationDirection,
    ResolvedEvidencePart,
    ResolvedEvidenceSelection,
)
from memforge.source_projection import AnchorKind, SourceAnchor
from memforge.memory.relation_classifier import MemoryRelationType
from memforge.models import (
    CoordinatorProposal,
    Memory,
    RawMemory,
    ReconcileAction,
    content_hash,
)
from memforge.pipeline.reconciler import ReconciliationResult, reconcile_memories
from memforge.pipeline.support_relation_coordinator import (
    RelationLedgerEntry,
    RevisionCompositionProof,
    coordinate,
)
from tests.revision_client_fixture import pinned


def coordinated_operations(*, new_extractions, existing_memories, relations, supports, revision_proofs=()):
    """Final coordinator operations for completed Support reads."""
    return list(coordinate(
        candidates=new_extractions, incumbents=existing_memories, relations=relations,
        proofs=revision_proofs, supports=supports,
    ).operations)


def _memory(memory_id: str, content: str) -> Memory:
    now = datetime.now(timezone.utc)
    return Memory(
        id=memory_id,
        memory_type="fact",
        content=content,
        content_hash=content_hash(content),
        created_at=now,
        updated_at=now,
    )


def _with_selection(raw: RawMemory) -> RawMemory:
    """Give a quoted candidate the exact Evidence selection extraction resolves."""

    assert raw.source_observation_id and raw.evidence_quote
    digest = hashlib.sha256(raw.evidence_quote.encode("utf-8")).hexdigest()
    return replace(
        raw,
        resolved_evidence_selection=ResolvedEvidenceSelection(
            source_id="src-relation-first",
            source_unit_id="unit-relation-first",
            target_unit_revision_id="unitrev-relation-first",
            access_context_hash="access-relation-first",
            catalog_digest="catalog-relation-first",
            compiler_contract_version=1,
            parts=(
                ResolvedEvidencePart(
                    role=EvidenceRole.PRIMARY,
                    kind=EvidencePartKind.TEXT,
                    anchor=SourceAnchor(
                        kind=AnchorKind.REVISION_RANGE,
                        observation_id=raw.source_observation_id,
                        observation_revision_id=f"{raw.source_observation_id}-rev",
                        range_start=0,
                        range_end=len(raw.evidence_quote),
                    ),
                    raw_content_sha256=digest,
                    presentation_sha256=digest,
                    excerpt=raw.evidence_quote,
                ),
            ),
        ),
    )


def _relations_from_prompt(prompt: str, classification: str = "unrelated") -> MemoryRelationResponse:
    groups = json.loads(
        prompt.split("<memory_pair_groups>\n", 1)[1].split("\n</memory_pair_groups>", 1)[0]
    )
    return MemoryRelationResponse(
        decisions=[
            MemoryRelationDecision(
                pair_index=item["pair_index"],
                classification=classification,
                direction="symmetric",
                same_subject_and_scope=classification == "contradicts",
                incompatible_assertions=("incompatible assertions" if classification == "contradicts" else ""),
            )
            for group in groups
            for item in group["candidates"]
        ]
    )


def _single_refines_response() -> MemoryRelationResponse:
    return MemoryRelationResponse(
        decisions=[
            MemoryRelationDecision(
                pair_index=0,
                classification="refines",
                direction="challenger_to_candidate",
                same_subject_and_scope=True,
                incompatible_assertions="",
                reason="The challenger adds a detail to the same claim.",
            )
        ]
    )


@pytest.mark.asyncio
async def test_supported_incumbent_and_unrelated_case25_keep_and_add() -> None:
    incumbent = _memory(
        "mem-cases20-24",
        "Component tests cover batch-handling cases 20 through 24.",
    )
    case25 = _with_selection(RawMemory(
        content="Case 25 verifies that a mixed valid and invalid batch returns partial results.",
        memory_type="fact",
        source_observation_id="obs-case25",
        evidence_quote="Case 25 verifies that a mixed valid and invalid batch returns partial results.",
        evidence_anchor="projection_batch",
    ))

    class RelationFirstClient(RevisionClientFixture):
        async def classify_memory_relations(self, prompt: str, **kwargs):
            del prompt, kwargs
            return MemoryRelationResponse(
                decisions=[
                    MemoryRelationDecision(
                        pair_index=0,
                        classification="unrelated",
                        direction="symmetric",
                        same_subject_and_scope=False,
                        incompatible_assertions="",
                        reason="Case 25 is a sibling scenario, not part of the incumbent claim.",
                    )
                ]
            )

    result = await reconcile_memories(
        new_extractions=[case25],
        existing_memories=[incumbent],
        supports=dict([pinned(incumbent.id, True)]),
        llm_model="test-model",
        structured_llm_client=RelationFirstClient(),
    )

    assert isinstance(result, ReconciliationResult)
    assert result.failure is None
    assert [(operation.action, operation.memory_id) for operation in result.operations] == [
        (ReconcileAction.ADD, None),
        (ReconcileAction.NOOP, incumbent.id),
    ]
    assert result.operations[0].memory is case25


@pytest.mark.asyncio
async def test_additive_refinement_with_complete_current_evidence_is_revision() -> None:
    incumbent = _memory("mem-timeout", "The client timeout is 30 seconds.")
    refinement = _with_selection(RawMemory(
        content="The client timeout is 30 seconds and is configurable with CLIENT_TIMEOUT.",
        memory_type="fact",
        evidence_quote=(
            "The client timeout is 30 seconds and is configurable with CLIENT_TIMEOUT."
        ),
        source_observation_id="obs-timeout",
        evidence_anchor="projection_batch",
    ))

    class RevisionClient(RevisionClientFixture):
        async def classify_memory_relations(self, prompt: str, **kwargs):
            del prompt, kwargs
            return MemoryRelationResponse(
                decisions=[
                    MemoryRelationDecision(
                        pair_index=0,
                        classification="refines",
                        direction="challenger_to_candidate",
                        same_subject_and_scope=True,
                        incompatible_assertions="",
                        reason="The challenger preserves the timeout and adds its configuration key.",
                    )
                ]
            )


        async def prove_revisions(self, prompt: str, **kwargs):
            del kwargs
            assert '"type": "fact"' in prompt
            assert '"valid_from": null' in prompt
            return RevisionProofs(
                decisions=[
                    RevisionProof(
                        pair_index=0,
                        same_memory_identity=True,
                        preserves_incumbent_truth=True,
                        candidate_is_canonical_composite=True,
                        reason="The candidate is the complete current timeout claim.",
                    )
                ]
            )

    result = await reconcile_memories(
        new_extractions=[refinement],
        existing_memories=[incumbent],
        supports=dict([pinned(incumbent.id, True)]),
        llm_model="test-model",
        structured_llm_client=RevisionClient(),
    )

    assert isinstance(result, ReconciliationResult)
    assert result.failure is None
    assert result.metrics.revision_proof_count == 1
    [operation] = result.operations
    assert operation.action is ReconcileAction.UPDATE
    assert operation.memory_id == incumbent.id
    assert operation.memory is refinement


@pytest.mark.asyncio
async def test_revision_response_failure_cannot_fall_back_to_add() -> None:
    incumbent = _memory("mem-timeout", "The client timeout is 30 seconds.")
    refinement = _with_selection(RawMemory(
        content="The client timeout is 30 seconds and is configurable with CLIENT_TIMEOUT.",
        memory_type="fact",
        evidence_quote="The client timeout is 30 seconds and is configurable with CLIENT_TIMEOUT.",
        evidence_anchor="projection_batch",
        source_observation_id="obs-timeout",
    ))

    class ProofFailureClient(RevisionClientFixture):
        async def classify_memory_relations(self, prompt: str, **kwargs):
            del prompt, kwargs
            return _single_refines_response()


        async def prove_revisions(self, prompt: str, **kwargs):
            del prompt, kwargs
            raise StructuredLlmError("revision proof unavailable", terminal_category="provider_error")

    result = await reconcile_memories(
        new_extractions=[refinement],
        existing_memories=[incumbent],
        supports=dict([pinned(incumbent.id, True)]),
        llm_model="test-model",
        structured_llm_client=ProofFailureClient(),
    )

    assert isinstance(result, ReconciliationResult)
    assert result.failure is not None
    assert result.operations == []


@pytest.mark.asyncio
async def test_missing_conditional_assessment_preserves_incumbent() -> None:
    incumbent = _memory("mem-timeout", "The client timeout is 30 seconds.")
    refinement = _with_selection(RawMemory(
        content="The client timeout is 30 seconds and is configurable with CLIENT_TIMEOUT.",
        memory_type="fact",
        evidence_quote="The client timeout is 30 seconds and is configurable with CLIENT_TIMEOUT.",
        evidence_anchor="projection_batch",
        source_observation_id="obs-timeout",
    ))

    class IncompleteProofClient(RevisionClientFixture):
        def __init__(self) -> None:
            self.proof_calls = 0

        async def classify_memory_relations(self, prompt: str, **kwargs):
            del prompt, kwargs
            return _single_refines_response()


        async def prove_revisions(self, prompt: str, **kwargs):
            del prompt, kwargs
            self.proof_calls += 1
            return RevisionProofs(decisions=[])

    client = IncompleteProofClient()
    result = await reconcile_memories(
        new_extractions=[refinement],
        existing_memories=[incumbent],
        supports=dict([pinned(incumbent.id, True)]),
        llm_model="test-model",
        structured_llm_client=client,
    )

    assert isinstance(result, ReconciliationResult)
    assert result.failure is None
    [operation] = result.operations
    assert operation.action == ReconcileAction.NOOP
    assert operation.memory is None and operation.support_revalidation_skipped
    assert client.proof_calls == 1


def test_refinement_without_revision_proof_falls_back_to_keep_and_add() -> None:
    incumbent = _memory("mem-timeout", "The client timeout is 30 seconds.")
    narrower = RawMemory(
        content="Upload requests use a 30 second client timeout.",
        memory_type="fact",
        source_observation_id="obs-upload",
        evidence_anchor="projection_batch",
    )

    operations = coordinated_operations(
        new_extractions=[narrower],
        existing_memories=[incumbent],
        relations=[
            RelationLedgerEntry(
                candidate_index=0,
                incumbent_id=incumbent.id,
                relation_type=MemoryRelationType.REFINES,
                direction=RelationDirection.CHALLENGER_TO_CANDIDATE,
            )
        ],
        supports=dict([pinned(incumbent.id, True)]),
        revision_proofs=[
            RevisionCompositionProof(
                candidate_index=0,
                incumbent_id=incumbent.id,
                same_memory_identity=False,
                preserves_incumbent_truth=False,
                candidate_is_canonical_composite=True,
            )
        ],
    )

    assert [operation.action for operation in operations] == [ReconcileAction.ADD, ReconcileAction.NOOP]


@pytest.mark.parametrize(
    ("supported", "relation_type", "expected_actions", "review"),
    [
        (True, MemoryRelationType.EQUIVALENT, [ReconcileAction.NOOP], None),
        (True, MemoryRelationType.UNRELATED, [ReconcileAction.ADD, ReconcileAction.NOOP], None),
        (True, MemoryRelationType.CONTRADICTS, [ReconcileAction.NOOP], CoordinatorProposal.SUPERSEDE),
        (False, MemoryRelationType.UNRELATED, [ReconcileAction.ADD, ReconcileAction.DELETE], None),
        (False, MemoryRelationType.CONTRADICTS, [ReconcileAction.SUPERSEDE], None),
        # Still unsupported after its one re-check.
        (False, MemoryRelationType.EQUIVALENT, [ReconcileAction.NOOP], CoordinatorProposal.REBIND),
    ],
)
def test_relation_support_matrix(
    supported: bool,
    relation_type: MemoryRelationType,
    expected_actions: list[ReconcileAction],
    review: CoordinatorProposal | None,
) -> None:
    incumbent = _memory("mem-current", "The service uses PostgreSQL 15.")
    candidate = RawMemory(
        content="The service uses PostgreSQL 16.",
        memory_type="fact",
        source_observation_id="obs-db",
        evidence_anchor="projection_batch",
    )
    operations = coordinated_operations(
        new_extractions=[candidate],
        existing_memories=[incumbent],
        relations=[
            RelationLedgerEntry(
                candidate_index=0,
                incumbent_id=incumbent.id,
                relation_type=relation_type,
                direction=RelationDirection.SYMMETRIC,
            )
        ],
        supports=dict([pinned(incumbent.id, supported)]),
    )

    assert [operation.action for operation in operations] == expected_actions
    incumbent_operation = operations[-1]
    assert not incumbent_operation.flag_for_review
    assert [item.proposal for item in incumbent_operation.reviews] == ([review] if review else [])


@pytest.mark.parametrize("supported", [True, False])
def test_multiple_contradiction_candidates_never_guess_a_successor(supported: bool) -> None:
    incumbent = _memory("mem-current", "The service uses one legacy database configuration.")
    candidates = [
        RawMemory(content="The primary database uses PostgreSQL 16.", memory_type="fact"),
        RawMemory(content="The analytics database uses ClickHouse.", memory_type="fact"),
    ]

    operations = coordinated_operations(
        new_extractions=candidates,
        existing_memories=[incumbent],
        relations=[
            RelationLedgerEntry(
                candidate_index=index,
                incumbent_id=incumbent.id,
                relation_type=MemoryRelationType.CONTRADICTS,
                direction=RelationDirection.SYMMETRIC,
            )
            for index in range(len(candidates))
        ],
        supports=dict([pinned(incumbent.id, supported)]),
    )

    if supported:
        # Each contradicting Candidate is staged in its own Review of the kept old Memory.
        [operation] = operations
        assert operation.action is ReconcileAction.NOOP and operation.memory_id == incumbent.id
        assert [review.candidate for review in operation.reviews] == candidates
        return
    # A read without Support removes it; the Candidates stand on their own.
    assert [operation.action for operation in operations] == [
        ReconcileAction.ADD,
        ReconcileAction.ADD,
        ReconcileAction.DELETE,
    ]
    assert [operation.memory for operation in operations[:2]] == candidates
    assert operations[-1].memory_id == incumbent.id


def test_an_unsupported_equivalent_is_held_in_a_rebind_review() -> None:
    incumbent = _memory("mem-current", "The client timeout is 30 seconds.")
    equivalent = RawMemory(content="Client timeout: 30 seconds.", memory_type="fact")
    refinement = RawMemory(content="Upload timeout is 30 seconds.", memory_type="fact")

    operations = coordinated_operations(
        new_extractions=[equivalent, refinement],
        existing_memories=[incumbent],
        relations=[
            RelationLedgerEntry(
                candidate_index=0,
                incumbent_id=incumbent.id,
                relation_type=MemoryRelationType.EQUIVALENT,
                direction=RelationDirection.SYMMETRIC,
            ),
            RelationLedgerEntry(
                candidate_index=1,
                incumbent_id=incumbent.id,
                relation_type=MemoryRelationType.REFINES,
                direction=RelationDirection.CHALLENGER_TO_CANDIDATE,
            ),
        ],
        supports=dict([pinned(incumbent.id, False)]),
    )

    # The refinement proves nothing about the old Memory's truth, so it stands on its own.
    add, operation = operations
    assert add.action is ReconcileAction.ADD and add.memory is refinement
    assert operation.action is ReconcileAction.NOOP and operation.memory is None
    assert operation.support_revalidation_skipped and not operation.flag_for_review
    [review] = operation.reviews
    assert review.proposal is CoordinatorProposal.REBIND and review.candidate is equivalent


def test_equivalent_candidate_rebinds_each_supported_incumbent() -> None:
    first = _memory("mem-first", "Retries use exponential backoff.")
    second = _memory("mem-second", "Retry delays increase exponentially.")
    candidate = RawMemory(content="Retries back off exponentially.", memory_type="fact")

    operations = coordinated_operations(
        new_extractions=[candidate],
        existing_memories=[first, second],
        relations=[
            RelationLedgerEntry(
                candidate_index=0,
                incumbent_id=memory.id,
                relation_type=MemoryRelationType.EQUIVALENT,
                direction=RelationDirection.SYMMETRIC,
            )
            for memory in (first, second)
        ],
        supports=dict([pinned(memory.id, True) for memory in (first, second)]),
    )

    assert [operation.action for operation in operations] == [
        ReconcileAction.NOOP,
        ReconcileAction.NOOP,
    ]
    assert [operation.memory_id for operation in operations] == [first.id, second.id]
    assert all(not operation.reviews and not operation.support_revalidation_skipped for operation in operations)


def test_runbook_candidate_with_multiple_incumbents_falls_back_to_keep_and_add() -> None:
    incumbents = [
        _memory(
            "mem-http-404",
            "For HTTP 404, check service health, retrigger, then open a DwC issue if it persists.",
        ),
        _memory(
            "mem-http-502-503",
            "For HTTP 502 or 503, wait for service recovery and then retrigger the process.",
        ),
        _memory(
            "mem-other-errors",
            "For other invalid process map errors, create a design-time Jira defect.",
        ),
    ]
    current_procedure = RawMemory(
        content=(
            "Diagnose an invalid process map from its actual HTTP error, then follow the status-specific recovery path."
        ),
        memory_type="procedure",
    )

    operations = coordinated_operations(
        new_extractions=[current_procedure],
        existing_memories=incumbents,
        relations=[
            RelationLedgerEntry(
                candidate_index=0,
                incumbent_id=incumbent.id,
                relation_type=MemoryRelationType.REFINES,
                direction=RelationDirection.CHALLENGER_TO_CANDIDATE,
                reason="The current procedure is related but does not prove lossless replacement.",
            )
            for incumbent in incumbents
        ],
        supports=dict([
            pinned(
                incumbent.id,
                True,
                "The branch remains supported in the current runbook.",
            )
            for incumbent in incumbents
        ]),
    )

    assert [operation.action for operation in operations] == [
        ReconcileAction.ADD,
        ReconcileAction.NOOP,
        ReconcileAction.NOOP,
        ReconcileAction.NOOP,
    ]
    assert operations[0].memory is current_procedure
    assert [operation.memory_id for operation in operations[1:]] == [incumbent.id for incumbent in incumbents]


@pytest.mark.asyncio
async def test_a_candidate_whose_relation_row_stays_invalid_is_consumed_without_add() -> None:
    class IncompleteClient(RevisionClientFixture):
        """Never returns the row of the Candidate "Unjudged claim"."""

        def __init__(self) -> None:
            self.calls = 0

        async def assess_claim_revisions(self, prompt: str, **kwargs):
            self.calls += 1
            from memforge.llm.structured import ClaimRevisionWireResponse
            from tests.revision_client_fixture import catalog_payload
            return ClaimRevisionWireResponse.model_validate(dict(results=[
                dict(candidate_id=claim["id"], relations=[], uncertain_existing_ids=[])
                for claim in catalog_payload(prompt)["new_claims"] if claim["text"] != "Unjudged claim"
            ]))

    client = IncompleteClient()
    unjudged, judged = (
        _with_selection(RawMemory(content=text, memory_type="fact", evidence_quote=text, source_observation_id="obs"))
        for text in ("Unjudged claim", "Independent claim")
    )
    result = await reconcile_memories(
        new_extractions=[unjudged, judged],
        existing_memories=[_memory("mem-old", "Old claim")],
        llm_model="test-model",
        structured_llm_client=client,
        supports=dict([pinned("mem-old", True)]),
    )

    assert isinstance(result, ReconciliationResult)
    assert result.failure is None
    # The unjudged Candidate is consumed without ADD; the independent one is added and the old Memory kept.
    assert [op.memory for op in result.operations if op.action == ReconcileAction.ADD] == [judged]
    assert [(op.memory_id, op.action) for op in result.operations if op.memory_id] == [("mem-old", ReconcileAction.NOOP)]
    assert result.unresolved_candidate_count == 1
    # Both Candidates, then the unjudged one's re-ask.
    assert client.calls == 2


@pytest.mark.asyncio
async def test_relation_provider_failure_fails_closed_with_incumbents() -> None:
    class FailingClient(RevisionClientFixture):
        async def classify_memory_relations(self, prompt: str, **kwargs):
            del prompt, kwargs
            raise StructuredLlmError("structured unavailable", terminal_category="provider_error")

    result = await reconcile_memories(
        new_extractions=[RawMemory(content="New claim", memory_type="fact")],
        existing_memories=[_memory("mem-old", "Old claim")],
        llm_model="test-model",
        structured_llm_client=FailingClient(),
        supports=dict([pinned("mem-old", True)]),
    )

    assert isinstance(result, ReconciliationResult)
    assert result.operations == []
    assert result.failure is not None


def test_unresolved_pair_preserves_related_component_and_allows_independent_work():
    candidates = [RawMemory(content=f"Claim {n}", memory_type="fact") for n in range(3)]
    old = [_memory(f"mem-{n}", f"Old {n}") for n in range(3)]
    relations = [RelationLedgerEntry(c, m.id, MemoryRelationType.UNRELATED, RelationDirection.SYMMETRIC)
                 for c in range(3) for m in old]
    # Candidate 0 is unresolved against old0 and equivalent to old1. Candidate1
    # also touches old1. All four must stay together; candidate2/old2 may advance.
    from dataclasses import replace
    types = {(0, "mem-0"): None, (0, "mem-1"): MemoryRelationType.EQUIVALENT,
             (1, "mem-1"): MemoryRelationType.CONTRADICTS}
    relations = [replace(r, relation_type=types.get((r.candidate_index, r.incumbent_id), r.relation_type))
                 for r in relations]
    operations = coordinated_operations(
        new_extractions=candidates, existing_memories=old, relations=relations,
        supports=dict([pinned(m.id, False) for m in old]),
    )
    assert [op.memory for op in operations if op.action == ReconcileAction.ADD] == [candidates[2]]
    by_id = {op.memory_id: op for op in operations if op.memory_id}
    for mid in ("mem-0", "mem-1"):
        assert by_id[mid].action == ReconcileAction.NOOP
        assert by_id[mid].memory is None and by_id[mid].support_revalidation_skipped
    assert by_id["mem-2"].action == ReconcileAction.DELETE


@pytest.mark.asyncio
async def test_refinements_whose_comparison_stays_invalid_leave_their_edges_uncertain() -> None:
    from memforge.pipeline.reconciler import RelationLine, join_support_and_relation

    class InvalidComparisonClient(RevisionClientFixture):
        async def classify_memory_relations(self, prompt: str, **kwargs):
            return MemoryRelationResponse(decisions=[])

    incumbent = _memory("mem-timeout", "The client timeout is 30 seconds.")
    refinements = [
        RawMemory(content="The client timeout is 30 seconds for uploads.", memory_type="fact"),
        RawMemory(content="The client timeout is 30 seconds for downloads.", memory_type="fact"),
    ]
    relation = RelationLine(
        entries=tuple(
            RelationLedgerEntry(index, incumbent.id, MemoryRelationType.REFINES, RelationDirection.CHALLENGER_TO_CANDIDATE)
            for index in range(len(refinements))
        ),
        completed_candidate_count=len(refinements), incumbent_ids=frozenset({incumbent.id}),
    )

    result = await join_support_and_relation(
        relation, new_extractions=refinements, existing_memories=[incumbent],
        supports=dict([pinned(incumbent.id, True)]), structured_llm_client=InvalidComparisonClient(),
        llm_model="test-model",
    )

    # Which refinement may revise the old Memory cannot be judged: nothing revises it and nothing is added.
    assert result.failure is None
    [operation] = result.operations
    assert operation.action == ReconcileAction.NOOP and operation.support_revalidation_skipped
    assert result.unresolved_candidate_count == 2
