"""Relation-first reconciliation contracts."""

from __future__ import annotations

from datetime import datetime, timezone
import json

import pytest

from memforge.llm.structured import (
    IncumbentSupportAuditDecision,
    IncumbentSupportAuditResponse,
    MemoryRelationDecision,
    MemoryRelationResponse,
    RevisionCompositionDecision,
    RevisionCompositionResponse,
    StructuredLlmError,
)
from memforge.memory.evidence import RelationDirection
from memforge.memory.engine import MemoryEngine
from memforge.memory.relation_classifier import MemoryRelationType
from memforge.models import Memory, RawMemory, ReconcileAction, ReconcileOperation, content_hash
from memforge.pipeline.reconciler import (
    ReconciliationResult,
    RelationLedgerEntry,
    RevisionCompositionProof,
    SupportAuditEntry,
    reconcile_memories,
    reduce_relation_ledger,
)


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
    case25 = RawMemory(
        content="Case 25 verifies that a mixed valid and invalid batch returns partial results.",
        memory_type="fact",
        source_observation_id="obs-case25",
        evidence_anchor="projection_batch",
    )

    class RelationFirstClient:
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

        async def audit_incumbent_support(self, prompt: str, **kwargs):
            del prompt, kwargs
            return IncumbentSupportAuditResponse(
                decisions=[
                    IncumbentSupportAuditDecision(
                        supported=True,
                        reason="Cases 20 through 24 remain in the current Source Unit.",
                    )
                ]
            )

    result = await reconcile_memories(
        new_extractions=[case25],
        existing_memories=[incumbent],
        doc_type="component_test",
        structured_llm_client=RelationFirstClient(),
        updated_document="Cases 20 through 25 are documented independently.",
        include_metadata=True,
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
    refinement = RawMemory(
        content="The client timeout is 30 seconds and is configurable with CLIENT_TIMEOUT.",
        memory_type="fact",
        evidence_quote=(
            "The client timeout is 30 seconds and is configurable with CLIENT_TIMEOUT."
        ),
        source_observation_id="obs-timeout",
        evidence_anchor="projection_batch",
        evidence_resolved_from_block=True,
    )

    class RevisionClient:
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

        async def audit_incumbent_support(self, prompt: str, **kwargs):
            del prompt, kwargs
            return IncumbentSupportAuditResponse(
                decisions=[IncumbentSupportAuditDecision(supported=True, reason="The timeout remains current.")]
            )

        async def prove_revision_compositions(self, prompt: str, **kwargs):
            del kwargs
            assert '"memory_type": "fact"' in prompt
            assert '"valid_from": null' in prompt
            return RevisionCompositionResponse(
                decisions=[
                    RevisionCompositionDecision(
                        pair_index=0,
                        same_memory_identity=True,
                        preserves_incumbent_truth=True,
                        candidate_is_canonical_composite=True,
                        current_evidence_entails_candidate=True,
                        reason="The candidate is the complete current timeout claim.",
                    )
                ]
            )

    result = await reconcile_memories(
        new_extractions=[refinement],
        existing_memories=[incumbent],
        doc_type="design",
        structured_llm_client=RevisionClient(),
        updated_document=refinement.content,
        include_metadata=True,
    )

    assert isinstance(result, ReconciliationResult)
    assert result.failure is None
    assert result.metrics.revision_proof_count == 1
    [operation] = result.operations
    assert operation.action is ReconcileAction.UPDATE
    assert operation.memory_id == incumbent.id
    assert operation.memory is refinement


@pytest.mark.asyncio
async def test_revision_proof_failure_falls_back_to_keep_and_add() -> None:
    incumbent = _memory("mem-timeout", "The client timeout is 30 seconds.")
    refinement = RawMemory(
        content="The client timeout is 30 seconds and is configurable with CLIENT_TIMEOUT.",
        memory_type="fact",
        evidence_quote="The client timeout is 30 seconds and is configurable with CLIENT_TIMEOUT.",
        evidence_resolved_from_block=True,
        evidence_anchor="projection_batch",
        source_observation_id="obs-timeout",
    )

    class ProofFailureClient:
        async def classify_memory_relations(self, prompt: str, **kwargs):
            del prompt, kwargs
            return _single_refines_response()

        async def audit_incumbent_support(self, prompt: str, **kwargs):
            del prompt, kwargs
            return IncumbentSupportAuditResponse(
                decisions=[IncumbentSupportAuditDecision(supported=True)]
            )

        async def prove_revision_compositions(self, prompt: str, **kwargs):
            del prompt, kwargs
            raise StructuredLlmError("revision proof unavailable")

    result = await reconcile_memories(
        new_extractions=[refinement],
        existing_memories=[incumbent],
        doc_type="design",
        structured_llm_client=ProofFailureClient(),
        include_metadata=True,
    )

    assert isinstance(result, ReconciliationResult)
    assert result.failure is None
    assert result.metrics.revision_proof_count == 0
    assert result.metrics.revision_proof_failure_count == 1
    assert [operation.action for operation in result.operations] == [
        ReconcileAction.ADD,
        ReconcileAction.NOOP,
    ]


@pytest.mark.asyncio
async def test_revision_evidence_that_supports_only_added_detail_falls_back() -> None:
    incumbent = _memory("mem-timeout", "The client timeout is 30 seconds.")
    refinement = RawMemory(
        content="The client timeout is 30 seconds and is configurable with CLIENT_TIMEOUT.",
        memory_type="fact",
        evidence_quote="Configurable with CLIENT_TIMEOUT.",
        evidence_resolved_from_block=True,
        evidence_anchor="projection_batch",
        source_observation_id="obs-timeout",
    )

    class IncompleteEvidenceClient:
        async def classify_memory_relations(self, prompt: str, **kwargs):
            del prompt, kwargs
            return _single_refines_response()

        async def audit_incumbent_support(self, prompt: str, **kwargs):
            del prompt, kwargs
            return IncumbentSupportAuditResponse(
                decisions=[IncumbentSupportAuditDecision(supported=True)]
            )

        async def prove_revision_compositions(self, prompt: str, **kwargs):
            del kwargs
            assert '"current_primary_evidence_excerpt": "Configurable with CLIENT_TIMEOUT."' in prompt
            return RevisionCompositionResponse(
                decisions=[
                    RevisionCompositionDecision(
                        pair_index=0,
                        same_memory_identity=True,
                        preserves_incumbent_truth=True,
                        candidate_is_canonical_composite=True,
                        current_evidence_entails_candidate=False,
                        reason="The excerpt proves only the configuration key.",
                    )
                ]
            )

    result = await reconcile_memories(
        new_extractions=[refinement],
        existing_memories=[incumbent],
        doc_type="design",
        structured_llm_client=IncompleteEvidenceClient(),
        include_metadata=True,
    )

    assert isinstance(result, ReconciliationResult)
    assert result.failure is None
    assert [operation.action for operation in result.operations] == [
        ReconcileAction.ADD,
        ReconcileAction.NOOP,
    ]


@pytest.mark.asyncio
async def test_incomplete_revision_proof_coverage_retries_then_falls_back() -> None:
    incumbent = _memory("mem-timeout", "The client timeout is 30 seconds.")
    refinement = RawMemory(
        content="The client timeout is 30 seconds and is configurable with CLIENT_TIMEOUT.",
        memory_type="fact",
        evidence_quote="The client timeout is 30 seconds and is configurable with CLIENT_TIMEOUT.",
        evidence_resolved_from_block=True,
        evidence_anchor="projection_batch",
        source_observation_id="obs-timeout",
    )

    class IncompleteProofClient:
        def __init__(self) -> None:
            self.proof_calls = 0

        async def classify_memory_relations(self, prompt: str, **kwargs):
            del prompt, kwargs
            return _single_refines_response()

        async def audit_incumbent_support(self, prompt: str, **kwargs):
            del prompt, kwargs
            return IncumbentSupportAuditResponse(
                decisions=[IncumbentSupportAuditDecision(supported=True)]
            )

        async def prove_revision_compositions(self, prompt: str, **kwargs):
            del prompt, kwargs
            self.proof_calls += 1
            return RevisionCompositionResponse(decisions=[])

    client = IncompleteProofClient()
    result = await reconcile_memories(
        new_extractions=[refinement],
        existing_memories=[incumbent],
        doc_type="design",
        structured_llm_client=client,
        include_metadata=True,
    )

    assert isinstance(result, ReconciliationResult)
    assert result.failure is None
    assert client.proof_calls == 2
    assert result.metrics.revision_proof_failure_count == 1
    assert [operation.action for operation in result.operations] == [
        ReconcileAction.ADD,
        ReconcileAction.NOOP,
    ]


@pytest.mark.asyncio
async def test_unprovided_required_evidence_blocks_revision() -> None:
    incumbent = _memory("mem-timeout", "The client timeout is 30 seconds.")
    refinement = RawMemory(
        content="The client timeout is 30 seconds and is configurable with CLIENT_TIMEOUT.",
        memory_type="fact",
        evidence_quote="The client timeout is 30 seconds and is configurable with CLIENT_TIMEOUT.",
        evidence_resolved_from_block=True,
        evidence_anchor="projection_batch",
        source_observation_id="obs-timeout",
        required_source_observation_ids=["obs-config-scope"],
    )

    class RequiredEvidenceClient:
        async def classify_memory_relations(self, prompt: str, **kwargs):
            del prompt, kwargs
            return _single_refines_response()

        async def audit_incumbent_support(self, prompt: str, **kwargs):
            del prompt, kwargs
            return IncumbentSupportAuditResponse(
                decisions=[IncumbentSupportAuditDecision(supported=True)]
            )

        async def prove_revision_compositions(self, prompt: str, **kwargs):
            del kwargs
            assert '"required_evidence_count": 1' in prompt
            return RevisionCompositionResponse(
                decisions=[
                    RevisionCompositionDecision(
                        pair_index=0,
                        same_memory_identity=True,
                        preserves_incumbent_truth=True,
                        candidate_is_canonical_composite=True,
                        current_evidence_entails_candidate=True,
                    )
                ]
            )

    result = await reconcile_memories(
        new_extractions=[refinement],
        existing_memories=[incumbent],
        doc_type="design",
        structured_llm_client=RequiredEvidenceClient(),
        include_metadata=True,
    )

    assert isinstance(result, ReconciliationResult)
    assert result.failure is None
    assert [operation.action for operation in result.operations] == [
        ReconcileAction.ADD,
        ReconcileAction.NOOP,
    ]


def test_refinement_without_revision_proof_falls_back_to_keep_and_add() -> None:
    incumbent = _memory("mem-timeout", "The client timeout is 30 seconds.")
    narrower = RawMemory(
        content="Upload requests use a 30 second client timeout.",
        memory_type="fact",
        source_observation_id="obs-upload",
        evidence_anchor="projection_batch",
    )

    operations = reduce_relation_ledger(
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
        support_audits=[SupportAuditEntry(incumbent_id=incumbent.id, supported=True)],
        revision_proofs=[
            RevisionCompositionProof(
                candidate_index=0,
                incumbent_id=incumbent.id,
                same_memory_identity=False,
                preserves_incumbent_truth=False,
                candidate_is_canonical_composite=True,
                current_evidence_entails_candidate=True,
                complete_current_evidence=True,
            )
        ],
    )

    assert [operation.action for operation in operations] == [ReconcileAction.ADD, ReconcileAction.NOOP]


@pytest.mark.parametrize(
    ("supported", "relation_type", "expected_actions", "review"),
    [
        (True, MemoryRelationType.EQUIVALENT, [ReconcileAction.NOOP], False),
        (True, MemoryRelationType.UNRELATED, [ReconcileAction.ADD, ReconcileAction.NOOP], False),
        (True, MemoryRelationType.CONTRADICTS, [ReconcileAction.SUPERSEDE], True),
        (False, MemoryRelationType.UNRELATED, [ReconcileAction.ADD, ReconcileAction.DELETE], False),
        (False, MemoryRelationType.CONTRADICTS, [ReconcileAction.SUPERSEDE], False),
        (False, MemoryRelationType.EQUIVALENT, [ReconcileAction.DELETE], True),
    ],
)
def test_relation_support_matrix(
    supported: bool,
    relation_type: MemoryRelationType,
    expected_actions: list[ReconcileAction],
    review: bool,
) -> None:
    incumbent = _memory("mem-current", "The service uses PostgreSQL 15.")
    candidate = RawMemory(
        content="The service uses PostgreSQL 16.",
        memory_type="fact",
        source_observation_id="obs-db",
        evidence_anchor="projection_batch",
    )
    operations = reduce_relation_ledger(
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
        support_audits=[SupportAuditEntry(incumbent_id=incumbent.id, supported=supported)],
    )

    assert [operation.action for operation in operations] == expected_actions
    incumbent_operation = operations[-1]
    assert incumbent_operation.flag_for_review is review


@pytest.mark.parametrize(
    ("supported", "expected_incumbent_action"),
    [
        (True, ReconcileAction.NOOP),
        (False, ReconcileAction.DELETE),
    ],
)
def test_multiple_contradiction_candidates_remain_independent_without_guessing_a_successor(
    supported: bool,
    expected_incumbent_action: ReconcileAction,
) -> None:
    incumbent = _memory("mem-current", "The service uses one legacy database configuration.")
    candidates = [
        RawMemory(content="The primary database uses PostgreSQL 16.", memory_type="fact"),
        RawMemory(content="The analytics database uses ClickHouse.", memory_type="fact"),
    ]

    operations = reduce_relation_ledger(
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
        support_audits=[SupportAuditEntry(incumbent_id=incumbent.id, supported=supported)],
    )

    assert [operation.action for operation in operations] == [
        ReconcileAction.ADD,
        ReconcileAction.ADD,
        expected_incumbent_action,
    ]
    assert [operation.memory for operation in operations[:2]] == candidates
    assert operations[-1].memory_id == incumbent.id


def test_partial_projection_keep_does_not_drop_replacement_candidate() -> None:
    candidate = RawMemory(content="The service uses PostgreSQL 16.", memory_type="fact")

    operations = MemoryEngine._enforce_partial_projection_keep(
        (
            ReconcileOperation(
                action=ReconcileAction.UPDATE,
                memory_id="mem-current",
                memory=candidate,
            ),
        ),
        frozenset({"mem-current"}),
    )

    assert [operation.action for operation in operations] == [ReconcileAction.ADD, ReconcileAction.NOOP]
    assert operations[0].memory is candidate


def test_partial_projection_contradiction_stays_in_review() -> None:
    candidate = RawMemory(content="The service uses PostgreSQL 16.", memory_type="fact")

    [operation] = MemoryEngine._enforce_partial_projection_keep(
        (
            ReconcileOperation(
                action=ReconcileAction.SUPERSEDE,
                memory_id="mem-current",
                memory=candidate,
            ),
        ),
        frozenset({"mem-current"}),
    )

    assert operation.action is ReconcileAction.SUPERSEDE
    assert operation.memory is candidate
    assert operation.flag_for_review is True


def test_unsupported_equivalent_does_not_drop_sibling_refinement() -> None:
    incumbent = _memory("mem-current", "The client timeout is 30 seconds.")
    equivalent = RawMemory(content="Client timeout: 30 seconds.", memory_type="fact")
    refinement = RawMemory(content="Upload timeout is 30 seconds.", memory_type="fact")

    operations = reduce_relation_ledger(
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
        support_audits=[SupportAuditEntry(incumbent_id=incumbent.id, supported=False)],
    )

    assert [operation.action for operation in operations] == [ReconcileAction.ADD, ReconcileAction.DELETE]
    assert operations[0].memory is refinement
    assert operations[1].flag_for_review is True


def test_equivalent_candidate_rebinds_each_supported_incumbent() -> None:
    first = _memory("mem-first", "Retries use exponential backoff.")
    second = _memory("mem-second", "Retry delays increase exponentially.")
    candidate = RawMemory(content="Retries back off exponentially.", memory_type="fact")

    operations = reduce_relation_ledger(
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
        support_audits=[SupportAuditEntry(incumbent_id=memory.id, supported=True) for memory in (first, second)],
    )

    assert [operation.action for operation in operations] == [
        ReconcileAction.NOOP,
        ReconcileAction.NOOP,
    ]
    assert [operation.memory_id for operation in operations] == [first.id, second.id]
    assert all(operation.memory is candidate for operation in operations)


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

    operations = reduce_relation_ledger(
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
        support_audits=[
            SupportAuditEntry(
                incumbent_id=incumbent.id,
                supported=True,
                reason="The branch remains supported in the current runbook.",
            )
            for incumbent in incumbents
        ],
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
async def test_support_audit_batches_all_incumbents_without_candidates() -> None:
    incumbents = [_memory(f"mem-{index:08d}", f"Stable claim {index}") for index in range(65)]

    class AuditClient:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        async def classify_memory_relations(self, prompt: str, **kwargs):
            raise AssertionError(f"no candidate pairs expected: {prompt!r} {kwargs!r}")

        async def audit_incumbent_support(self, prompt: str, **kwargs):
            del kwargs
            incumbents_json = prompt.split("<incumbents>", 1)[1].split("</incumbents>", 1)[0]
            size = len(json.loads(incumbents_json))
            self.batch_sizes.append(size)
            return IncumbentSupportAuditResponse(
                decisions=[IncumbentSupportAuditDecision(supported=True) for _ in range(size)]
            )

    client = AuditClient()
    result = await reconcile_memories(
        new_extractions=[],
        existing_memories=incumbents,
        doc_type="design",
        structured_llm_client=client,
        include_metadata=True,
    )

    assert isinstance(result, ReconciliationResult)
    assert result.failure is None
    assert client.batch_sizes == [30, 30, 5]
    assert result.metrics.structured_llm_calls == 3
    assert len(result.operations) == 65


@pytest.mark.asyncio
async def test_incomplete_relation_ledger_retries_then_fails_closed() -> None:
    class IncompleteClient:
        def __init__(self) -> None:
            self.calls = 0

        async def classify_memory_relations(self, prompt: str, **kwargs):
            del prompt, kwargs
            self.calls += 1
            return MemoryRelationResponse(decisions=[])

    client = IncompleteClient()
    result = await reconcile_memories(
        new_extractions=[RawMemory(content="New claim", memory_type="fact")],
        existing_memories=[_memory("mem-old", "Old claim")],
        doc_type="design",
        structured_llm_client=client,
        include_metadata=True,
    )

    assert isinstance(result, ReconciliationResult)
    assert result.operations == []
    assert result.failure is not None
    assert client.calls == 2


@pytest.mark.asyncio
async def test_relation_provider_failure_fails_closed_with_incumbents() -> None:
    class FailingClient:
        async def classify_memory_relations(self, prompt: str, **kwargs):
            del prompt, kwargs
            raise StructuredLlmError("structured unavailable")

    result = await reconcile_memories(
        new_extractions=[RawMemory(content="New claim", memory_type="fact")],
        existing_memories=[_memory("mem-old", "Old claim")],
        doc_type="design",
        structured_llm_client=FailingClient(),
        include_metadata=True,
    )

    assert isinstance(result, ReconciliationResult)
    assert result.operations == []
    assert result.failure is not None
