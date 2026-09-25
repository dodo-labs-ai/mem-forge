from __future__ import annotations

import json
from dataclasses import dataclass, field, replace

import pytest

from memforge.llm.structured import (
    INPUT_CAPACITY_EXCEEDED,
    CrossDocumentRelationResponse,
    StructuredLlmError,
)
from memforge.memory.cross_document_relation import (
    CROSS_DOCUMENT_RELATION_RULES,
    CrossDocumentRelationLabel,
    CrossDocumentRelationPair,
    RELATION_SUBJECT_EXCERPT_CHARS,
    RelationSubject,
    StructuredCrossDocumentRelationClassifier,
    load_relation_subjects,
)
from memforge.memory.relation_classifier import MemoryPairClassificationError
from memforge.models import Memory, content_hash
from tests.llm_fixture import FIXTURE_MODEL, fixture_budget


def _subject(memory_id: str, statement: str) -> RelationSubject:
    return RelationSubject(
        memory_id=memory_id,
        content_hash=content_hash(statement),
        statement=statement,
        memory_type="decision",
        source_type="jira",
        document_title=f"Ticket for {memory_id}",
        primary_excerpt=f"Excerpt for {memory_id}",
    )


def _pairs(count: int, *, challenger: str = "mem-challenger") -> tuple[CrossDocumentRelationPair, ...]:
    return tuple(
        CrossDocumentRelationPair(
            challenger=_subject(challenger, "Payroll runs weekly."),
            candidate=_subject(f"mem-candidate-{index}", f"Payroll statement {index}."),
        )
        for index in range(count)
    )


def _pair_indexes(prompt: str) -> list[int]:
    groups = json.loads(prompt.split("<statement_pair_groups>\n", 1)[1].split("\n</statement_pair_groups>", 1)[0])
    return [item["pair_index"] for group in groups for item in group["compared_with"]]


@dataclass
class _Client:
    """Answers every requested pair with ``label`` unless ``respond`` overrides it."""

    label: str = "none"
    input_tokens: int = 100_000
    respond: object = None
    prompts: list[str] = field(default_factory=list)
    max_tokens: list[int] = field(default_factory=list)

    def request_budget(self, model=None):
        return fixture_budget(input_tokens=self.input_tokens, output_tokens=64_000, correction_reserve=0)

    def request_fits(self, prompt, *, response_format, max_tokens, model=None, images=(), reserve_correction=True):
        return self.request_budget(model).fits(len(prompt.split()), max_tokens, reserve_correction=reserve_correction)

    async def classify_cross_document_relations(self, prompt, *, max_tokens, model=None):
        self.prompts.append(prompt)
        self.max_tokens.append(max_tokens)
        if self.respond is not None:
            return self.respond(prompt)
        return CrossDocumentRelationResponse(
            decisions=[
                {"pair_index": index, "label": self.label, "reason": "fixture"}
                for index in _pair_indexes(prompt)
            ]
        )


def _classifier(client: _Client) -> StructuredCrossDocumentRelationClassifier:
    return StructuredCrossDocumentRelationClassifier(client=client, model=FIXTURE_MODEL)


@pytest.mark.asyncio
async def test_classifier_returns_one_judgment_per_pair_by_pair_index() -> None:
    pairs = _pairs(3)

    def respond(prompt):
        labels = ["contradicts", "none", "updates"]
        return CrossDocumentRelationResponse(
            decisions=[
                {"pair_index": index, "label": labels[index], "reason": f"reason {index}"}
                for index in reversed(_pair_indexes(prompt))
            ]
        )

    result = await _classifier(_Client(respond=respond)).classify(pairs)

    assert [(judgment.pair.key, judgment.label, judgment.reason) for judgment in result.judgments] == [
        (pairs[0].key, CrossDocumentRelationLabel.CONTRADICTS, "reason 0"),
        (pairs[1].key, CrossDocumentRelationLabel.NONE, "reason 1"),
        (pairs[2].key, CrossDocumentRelationLabel.UPDATES, "reason 2"),
    ]
    assert result.llm_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "decisions",
    [
        [{"pair_index": 0, "label": "none", "reason": ""}],
        [
            {"pair_index": 0, "label": "none", "reason": ""},
            {"pair_index": 1, "label": "none", "reason": ""},
            {"pair_index": 7, "label": "none", "reason": ""},
        ],
    ],
    ids=["missing_pair", "unknown_pair"],
)
async def test_classifier_fails_when_a_pair_is_left_without_exactly_one_label(decisions) -> None:
    client = _Client(respond=lambda _prompt: CrossDocumentRelationResponse(decisions=decisions))

    with pytest.raises(MemoryPairClassificationError) as error:
        await _classifier(client).classify(_pairs(2))

    assert error.value.error_code is not None
    assert error.value.pair_count == 2
    assert len(client.prompts) == 2


@pytest.mark.asyncio
async def test_classifier_reports_a_refused_reply_with_its_error_code() -> None:
    def refuse(_prompt):
        raise StructuredLlmError(
            "refused",
            terminal_category="invalid_response",
            error_code="cross_document_relation_response_incomplete",
        )

    with pytest.raises(MemoryPairClassificationError) as error:
        await _classifier(_Client(respond=refuse)).classify(_pairs(1))

    assert error.value.error_code == "cross_document_relation_response_incomplete"


@pytest.mark.asyncio
async def test_classifier_halves_a_request_that_exceeds_the_route_capacity() -> None:
    attempts: list[int] = []

    def respond(prompt):
        indexes = _pair_indexes(prompt)
        attempts.append(len(indexes))
        if len(indexes) > 2:
            raise StructuredLlmError("context window", terminal_category="provider_error", error_code=INPUT_CAPACITY_EXCEEDED)
        return CrossDocumentRelationResponse(
            decisions=[{"pair_index": index, "label": "none", "reason": ""} for index in indexes]
        )

    result = await _classifier(_Client(respond=respond)).classify(_pairs(4))

    assert len(result.judgments) == 4
    assert attempts[0] == 4
    assert all(count <= 2 for count in attempts[1:])


@pytest.mark.asyncio
async def test_prompt_shows_statements_without_identity_sources_or_dates() -> None:
    client = _Client()
    challenger = replace(
        _subject("mem-challenger-secret-id", "Payroll runs weekly."),
        document_title="Payroll design",
        primary_excerpt="Payroll runs weekly for all employees.",
    )
    candidate = replace(
        _subject("mem-candidate-secret-id", "Payroll runs monthly."),
        document_title="Payroll ticket",
        primary_excerpt=None,
    )

    await _classifier(client).classify((CrossDocumentRelationPair(challenger=challenger, candidate=candidate),))

    [prompt] = client.prompts
    for shown in ("Payroll runs weekly.", "Payroll runs monthly.", "Payroll design", "Payroll runs weekly for all"):
        assert shown in prompt
    for hidden in ("secret-id", "memory_id", "content_hash", "source_id", "primary_excerpt\": null", "2026-"):
        assert hidden not in prompt


def test_rules_are_domain_neutral_and_prefer_none_when_unsure() -> None:
    rules = CROSS_DOCUMENT_RELATION_RULES.lower()
    for label in CrossDocumentRelationLabel:
        assert f"- {label.value}:" in rules
    assert "when you are not certain, the label is none" in rules
    assert "more specific than the other" in rules
    for dimension in ("version", "country", "environment", "ticket"):
        assert dimension not in rules


def test_subject_manifest_round_trips() -> None:
    subject = _subject("mem-1", "Payroll runs weekly.")

    assert RelationSubject.from_manifest(subject.to_manifest()) == subject


class _SubjectStore:
    def __init__(self, units, documents) -> None:
        self.units = units
        self.documents = documents
        self.document_reads: list[str] = []

    async def get_memory_evidence_units(self, memory_id):
        return self.units.get(memory_id, ())

    async def get_document(self, doc_id):
        self.document_reads.append(doc_id)
        return self.documents.get(doc_id)


@pytest.mark.asyncio
async def test_subjects_come_from_the_current_primary_evidence_and_bound_the_excerpt() -> None:
    from tests.relation_evidence_fixture import primary_evidence_unit_fixture

    long_unit = primary_evidence_unit_fixture("mem-long")
    long_item = replace(long_unit.items[0], excerpt="x" * (RELATION_SUBJECT_EXCERPT_CHARS + 1))
    stale_unit = replace(primary_evidence_unit_fixture("mem-long"), evidence_unit_id="eu-old", current=False, doc_id="doc-old")
    store = _SubjectStore(
        units={
            "mem-long": (stale_unit, replace(long_unit, items=(long_item,))),
            "mem-plain": (),
        },
        documents={"doc-mem-long": type("Doc", (), {"title": "Current page"})()},
    )
    memories = (
        Memory(id="mem-long", memory_type="fact", content="Long", content_hash=content_hash("Long")),
        Memory(id="mem-plain", memory_type="fact", content="Plain", content_hash=content_hash("Plain")),
    )

    subjects = await load_relation_subjects(store, memories)

    assert subjects["mem-long"].document_title == "Current page"
    assert len(subjects["mem-long"].primary_excerpt) == RELATION_SUBJECT_EXCERPT_CHARS
    assert subjects["mem-plain"] == RelationSubject(
        memory_id="mem-plain", content_hash=content_hash("Plain"), statement="Plain", memory_type="fact",
    )
    assert store.document_reads == ["doc-mem-long"]
