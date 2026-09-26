from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

import pytest

from memforge.llm.structured import (
    INPUT_CAPACITY_EXCEEDED,
    CrossDocumentRelationResponse,
    StructuredLlmError,
)
from memforge.memory.cross_document_relation import (
    CROSS_DOCUMENT_RELATION_RULES,
    CrossDocumentRelationJudgment,
    CrossDocumentRelationLabel,
    CrossDocumentRelationOutcome,
    CrossDocumentRelationPair,
    CrossDocumentRelationRecord,
    RelationSubject,
    StructuredCrossDocumentRelationClassifier,
    load_relation_subjects,
    pair_key,
)
from memforge.memory.evidence import EvidencePartKind, EvidenceRole
from memforge.memory.relation_classifier import MemoryPairClassificationError
from memforge.models import Memory, content_hash
from memforge.pipeline.evidence_fragments import DEFAULT_MAX_PRESENTATION_CHARS
from memforge.source_projection import AnchorKind, SourceAnchor, SourceObservationRevision
from memforge.source_representation import (
    MARKDOWN_STRUCTURAL_PROFILE,
    representation_profile_for_observation_contract,
)
from tests.llm_fixture import FIXTURE_MODEL, fixture_budget
from tests.relation_evidence_fixture import (
    primary_evidence_unit_fixture,
    primary_observation_revision_fixture,
)


def _subject(memory_id: str, statement: str) -> RelationSubject:
    return RelationSubject(
        memory_id=memory_id,
        content_hash=content_hash(statement),
        statement=statement,
        memory_type="decision",
        source_type="jira",
        document_title=f"Ticket for {memory_id}",
        evidence_time="2026-03-25",
        evidence=(f"Evidence for {memory_id}",),
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
            {"pair_index": 0, "label": "none", "reason": ""},
        ],
    ],
    ids=["missing_pair", "duplicate_hides_missing"],
)
async def test_classifier_fails_when_a_pair_is_left_without_exactly_one_label(decisions) -> None:
    client = _Client(respond=lambda _prompt: CrossDocumentRelationResponse(decisions=decisions))

    with pytest.raises(MemoryPairClassificationError) as error:
        await _classifier(client).classify(_pairs(2))

    assert error.value.error_code is not None
    assert error.value.pair_count == 2
    # The accepted rows are kept and the rejected pairs re-asked once before the classifier fails.
    assert "<correction>" in client.prompts[1]
    assert error.value.llm_calls == len(client.prompts) == 2


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
async def test_prompt_shows_statement_context_without_identity() -> None:
    client = _Client()
    challenger = replace(
        _subject("mem-challenger-secret-id", "Payroll runs weekly."),
        document_title="Payroll design",
        evidence_time="2026-04-10",
        evidence=("Payroll runs weekly for all employees.",),
    )
    candidate = replace(
        _subject("mem-candidate-secret-id", "Payroll runs monthly."),
        document_title="Payroll ticket",
        evidence_time=None,
        evidence=(),
    )

    await _classifier(client).classify((CrossDocumentRelationPair(challenger=challenger, candidate=candidate),))

    [prompt] = client.prompts
    groups = json.loads(prompt.split("<statement_pair_groups>\n", 1)[1].split("\n</statement_pair_groups>", 1)[0])
    assert groups == [
        {
            "statement": {
                "statement": "Payroll runs weekly.",
                "memory_type": "decision",
                "source_type": "jira",
                "document_title": "Payroll design",
                "evidence_time": "2026-04-10",
                "evidence": ["Payroll runs weekly for all employees."],
            },
            "compared_with": [
                {
                    "pair_index": 0,
                    "statement": {
                        "statement": "Payroll runs monthly.",
                        "memory_type": "decision",
                        "source_type": "jira",
                        "document_title": "Payroll ticket",
                        "evidence_time": None,
                        "evidence": [],
                    },
                }
            ],
        }
    ]
    for hidden in ("secret-id", "memory_id", "content_hash", "source_id"):
        assert hidden not in prompt


def test_rules_define_the_same_situation_domain_neutrally_and_prefer_none() -> None:
    rules = " ".join(CROSS_DOCUMENT_RELATION_RULES.lower().split())
    for label in CrossDocumentRelationLabel:
        assert f"- {label.value}:" in rules
    for condition in (
        "the same object",
        "the same scope",
        "the same kind of statement",
        "the same occurrence",
        "the same lasting state or decision are about the same situation",
        "or the inputs do not establish it",
        "the later statement replaces the earlier one",
        "neither replaces the other over time",
    ):
        assert condition in rules
    assert "when you are not certain, the label is none" in rules
    assert "both can hold without stating the same knowledge" in rules
    assert "more specific than the other" in rules
    assert "judge the statements themselves" not in rules
    for dimension in ("version", "country", "environment", "ticket"):
        assert dimension not in rules


def test_record_orders_the_pair_and_binds_both_contents() -> None:
    challenger = _subject("mem-z", "Payroll runs weekly.")
    candidate = _subject("mem-a", "Payroll runs monthly.")
    judgment = CrossDocumentRelationJudgment(
        pair=CrossDocumentRelationPair(challenger=challenger, candidate=candidate),
        label=CrossDocumentRelationLabel.CONTRADICTS,
        reason="different schedules",
    )

    record = CrossDocumentRelationRecord.from_judgment(judgment, relation_run_id="run-1", discovery_work_id="work-1")

    assert (record.memory_low_id, record.memory_high_id) == ("mem-a", "mem-z")
    assert (record.low_content_hash, record.high_content_hash) == (candidate.content_hash, challenger.content_hash)
    assert pair_key("mem-z", "mem-a") == ("mem-a", "mem-z")
    with pytest.raises(ValueError):
        CrossDocumentRelationRecord.from_judgment(
            CrossDocumentRelationJudgment(pair=judgment.pair, label=CrossDocumentRelationLabel.NONE, reason=""),
            relation_run_id="run-1",
            discovery_work_id="work-1",
        )


def test_outcome_binds_every_judged_pair_to_the_contents_it_was_judged_on() -> None:
    challenger = _subject("mem-m", "Payroll runs weekly.")
    judged = (_subject("mem-a", "Payroll runs monthly."), _subject("mem-z", "Payroll runs on Fridays."))
    judgments = (
        CrossDocumentRelationJudgment(
            pair=CrossDocumentRelationPair(challenger=challenger, candidate=judged[0]),
            label=CrossDocumentRelationLabel.CONTRADICTS,
            reason="different schedules",
        ),
        CrossDocumentRelationJudgment(
            pair=CrossDocumentRelationPair(challenger=challenger, candidate=judged[1]),
            label=CrossDocumentRelationLabel.NONE,
            reason="compatible",
        ),
    )

    outcome = CrossDocumentRelationOutcome.from_judgments(
        challenger, judgments, relation_run_id="run-1", discovery_work_id="work-1"
    )

    assert outcome.judged_content_hashes == {subject.memory_id: subject.content_hash for subject in judged}
    [record] = outcome.relations
    assert (record.memory_low_id, record.memory_high_id) == ("mem-a", "mem-m")
    with pytest.raises(ValueError, match="judged pair"):
        CrossDocumentRelationOutcome(
            challenger_id=challenger.memory_id,
            challenger_content_hash=challenger.content_hash,
            judged_content_hashes={"mem-a": "older-content"},
            relations=(record,),
        )


def test_subject_manifest_round_trips_and_requires_the_evidence_fields() -> None:
    subject = _subject("mem-1", "Payroll runs weekly.")
    manifest = subject.to_manifest()

    assert RelationSubject.from_manifest(manifest) == subject
    assert RelationSubject.from_manifest({**manifest, "evidence_time": None}).evidence_time is None
    for field_name in ("evidence_time", "evidence"):
        with pytest.raises(KeyError):
            RelationSubject.from_manifest({key: value for key, value in manifest.items() if key != field_name})


class _Document:
    def __init__(self, title: str, last_modified: datetime) -> None:
        self.title = title
        self.last_modified = last_modified


class _SubjectStore:
    def __init__(self, units, documents, revisions=None) -> None:
        self.units = units
        self.documents = documents
        self.revisions = revisions or {}
        self.document_reads: list[str] = []
        self.revision_reads: list[str] = []

    async def get_memory_evidence_units(self, memory_id):
        return self.units.get(memory_id, ())

    async def get_document(self, doc_id):
        self.document_reads.append(doc_id)
        return self.documents.get(doc_id)

    async def get_current_source_observation_revisions(self, source_unit_id):
        self.revision_reads.append(source_unit_id)
        return self.revisions.get(source_unit_id, {})


def _canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _jira_revision(memory_id: str, observation_type: str, value, observed_at: str | None) -> SourceObservationRevision:
    return replace(
        primary_observation_revision_fixture(memory_id),
        content=_canonical_json(value),
        observed_at=observed_at,
        evidence_profile=representation_profile_for_observation_contract(
            source_type="jira",
            observation_type=observation_type,
        ),
    )


def _memory(memory_id: str) -> Memory:
    return Memory(id=memory_id, memory_type="fact", content=memory_id, content_hash=content_hash(memory_id))


SYNC_TIME = datetime(2026, 7, 17, 3, 0, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_subject_shows_readable_record_fields_and_the_observation_revision_time() -> None:
    history = {
        "author": {"displayName": "Dev", "avatarUrls": {"48x48": "https://avatars.example.test/dev.png"}},
        "created": "2026-03-25T23:30:00.000-0800",
        "items": [{"field": "status", "fromString": "Open", "toString": "Closed", "from": "1", "to": "6"}],
    }
    core = {
        "summary": "Rename the enumeration",
        "description": "The enumeration is **RETRO_CHAIN**.",
        "assignee": {"accountId": "acc-1", "displayName": "Dev", "avatarUrls": {"48x48": "https://avatars.example.test/a.png"}},
        "labels": ["payroll"],
    }
    unit = primary_evidence_unit_fixture("mem-jira")
    primary = unit.items[0]
    required = replace(
        primary,
        reference_id="ref-required",
        role=EvidenceRole.REQUIRED,
        anchor=replace(primary.anchor, observation_id="obs-core", observation_revision_id="rev-core"),
    )
    store = _SubjectStore(
        units={"mem-jira": (replace(unit, items=(primary, required)),)},
        documents={"doc-mem-jira": _Document("PAY-1: Rename", SYNC_TIME)},
        revisions={
            "unit-mem-jira": {
                "obs-mem-jira": _jira_revision("mem-jira", "changelog", history, history["created"]),
                "obs-core": replace(_jira_revision("mem-jira", "issue_core", core, None), id="rev-core", observation_id="obs-core"),
            }
        },
    )

    subject = (await load_relation_subjects(store, (_memory("mem-jira"),)))["mem-jira"]

    assert subject.document_title == "PAY-1: Rename"
    assert subject.evidence_time == "2026-03-26"
    assert subject.evidence == (
        "created: 2026-03-25T23:30:00.000-0800\n"
        "items/0/field: status\n"
        "items/0/fromString: Open\n"
        "items/0/toString: Closed",
        "assignee: accountId=acc-1, displayName=Dev\n"
        "description: The enumeration is **RETRO_CHAIN**.\n"
        "labels: payroll\n"
        "summary: Rename the enumeration",
    )
    assert "avatar" not in json.dumps(subject.to_manifest())


@pytest.mark.asyncio
async def test_record_fields_keep_each_array_item_together_and_skip_empty_values() -> None:
    history = {
        "created": "2026-04-27T04:22:39.861+0000",
        "items": [
            {"field": "Defect Review Status", "fromString": None, "toString": "Fixed"},
            {"field": "status", "fromString": "Work In Progress", "toString": "Resolved"},
        ],
    }
    core = {"summary": "Improper error message", "labels": []}
    unit = primary_evidence_unit_fixture("mem-jira")
    primary = unit.items[0]
    required = replace(
        primary,
        reference_id="ref-required",
        role=EvidenceRole.REQUIRED,
        anchor=replace(primary.anchor, observation_id="obs-core", observation_revision_id="rev-core"),
    )
    store = _SubjectStore(
        units={"mem-jira": (replace(unit, items=(primary, required)),)},
        documents={"doc-mem-jira": _Document("PAY-2: Error message", SYNC_TIME)},
        revisions={
            "unit-mem-jira": {
                "obs-mem-jira": _jira_revision("mem-jira", "changelog", history, history["created"]),
                "obs-core": replace(_jira_revision("mem-jira", "issue_core", core, None), id="rev-core", observation_id="obs-core"),
            }
        },
    )

    subject = (await load_relation_subjects(store, (_memory("mem-jira"),)))["mem-jira"]

    assert subject.evidence == (
        "created: 2026-04-27T04:22:39.861+0000\n"
        "items/0/field: Defect Review Status\n"
        "items/0/toString: Fixed\n"
        "items/1/field: status\n"
        "items/1/fromString: Work In Progress\n"
        "items/1/toString: Resolved",
        "summary: Improper error message",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("source_type", ["confluence", "github_repo", "local_markdown", "agent_session", "jira"])
@pytest.mark.parametrize(
    ("observed_at", "evidence_time"),
    [("2026-03-02T23:30:00-05:00", "2026-03-03"), (None, None)],
)
async def test_evidence_time_is_the_anchored_revision_source_time(source_type: str, observed_at, evidence_time) -> None:
    """Every Source is read alike; a document time never stands in for a missing revision time."""

    unit = replace(primary_evidence_unit_fixture("mem-page"), source_type=source_type)
    store = _SubjectStore(
        units={"mem-page": (unit,)},
        documents={"doc-mem-page": _Document("Page", datetime(2026, 4, 10, 9, 0, tzinfo=timezone.utc))},
        revisions={
            "unit-mem-page": {
                "obs-mem-page": replace(primary_observation_revision_fixture("mem-page"), observed_at=observed_at)
            }
        },
    )

    subject = (await load_relation_subjects(store, (_memory("mem-page"),)))["mem-page"]

    assert subject.evidence_time == evidence_time
    assert subject.evidence == ("Evidence for mem-page.",)


@pytest.mark.asyncio
async def test_evidence_time_needs_the_anchored_revision_to_be_current() -> None:
    unit = replace(primary_evidence_unit_fixture("mem-page"), source_type="confluence")
    newer = replace(
        primary_observation_revision_fixture("mem-page"), id="rev-newer", observed_at="2026-04-10T09:00:00+00:00"
    )
    store = _SubjectStore(
        units={"mem-page": (unit,)},
        documents={"doc-mem-page": _Document("Page", datetime(2026, 4, 10, 9, 0, tzinfo=timezone.utc))},
        revisions={"unit-mem-page": {"obs-mem-page": newer}},
    )

    subject = (await load_relation_subjects(store, (_memory("mem-page"),)))["mem-page"]

    assert subject.evidence_time is None
    assert subject.evidence == ()


@pytest.mark.asyncio
async def test_a_non_text_primary_or_an_observation_beyond_the_fragment_catalog_shows_no_evidence_text() -> None:
    image = primary_evidence_unit_fixture("mem-image")
    page = primary_evidence_unit_fixture("mem-page")
    oversized = "\n\n".join(
        f"Paragraph {index}." for index in range(DEFAULT_MAX_PRESENTATION_CHARS // len("Paragraph 0.") + 1)
    )
    store = _SubjectStore(
        units={
            "mem-image": (replace(image, items=(replace(image.items[0], kind=EvidencePartKind.ARTIFACT),)),),
            "mem-page": (page,),
        },
        documents={},
        revisions={
            "unit-mem-image": {"obs-mem-image": primary_observation_revision_fixture("mem-image")},
            "unit-mem-page": {"obs-mem-page": replace(primary_observation_revision_fixture("mem-page"), content=oversized)},
        },
    )

    subjects = await load_relation_subjects(store, (_memory("mem-image"), _memory("mem-page")))

    assert subjects["mem-image"].evidence == ()
    assert subjects["mem-page"].evidence == ()


@pytest.mark.asyncio
async def test_evidence_without_its_exact_revision_keeps_only_a_range_excerpt() -> None:
    whole = primary_evidence_unit_fixture("mem-old")
    ranged_item = replace(
        whole.items[0],
        anchor=SourceAnchor(
            kind=AnchorKind.REVISION_RANGE,
            observation_id="obs-mem-old",
            observation_revision_id="rev-mem-old",
            range_start=0,
            range_end=len("Raw observation for mem-old"),
        ),
    )
    changed = replace(primary_observation_revision_fixture("mem-old"), id="rev-newer")
    store = _SubjectStore(
        units={
            "mem-old": (replace(whole, current=False),),
            "mem-ranged": (replace(whole, current=False, items=(ranged_item,)),),
        },
        documents={"doc-mem-old": _Document("Page", SYNC_TIME)},
        revisions={"unit-mem-old": {"obs-mem-old": changed}},
    )

    subjects = await load_relation_subjects(store, (_memory("mem-old"), _memory("mem-ranged")))

    assert subjects["mem-old"].evidence == ()
    assert subjects["mem-old"].evidence_time is None
    assert subjects["mem-ranged"].evidence == ("Raw observation for mem-old",)
    assert store.document_reads == ["doc-mem-old"]
    assert store.revision_reads == ["unit-mem-old"]


@pytest.mark.asyncio
async def test_subjects_come_from_the_current_evidence_unit() -> None:
    current = primary_evidence_unit_fixture("mem-a")
    stale = replace(current, evidence_unit_id="eu-old", current=False, doc_id="doc-old")
    page = replace(primary_observation_revision_fixture("mem-a"), evidence_profile=MARKDOWN_STRUCTURAL_PROFILE)
    store = _SubjectStore(
        units={"mem-a": (stale, current), "mem-plain": ()},
        documents={"doc-mem-a": _Document("Current page", SYNC_TIME)},
        revisions={"unit-mem-a": {"obs-mem-a": page}},
    )

    subjects = await load_relation_subjects(store, (_memory("mem-a"), _memory("mem-plain")))

    assert subjects["mem-a"].document_title == "Current page"
    assert subjects["mem-a"].evidence == ("Evidence for mem-a.",)
    assert subjects["mem-a"].evidence_time == "2026-03-25"
    assert subjects["mem-plain"] == RelationSubject(
        memory_id="mem-plain", content_hash=content_hash("mem-plain"), statement="mem-plain", memory_type="fact",
    )
    assert store.document_reads == ["doc-mem-a"]


class _RelationClient:
    async def discover_memory_relations(self, prompt, **kwargs):
        raise AssertionError("not called")

    async def classify_cross_document_relations(self, prompt, **kwargs):
        raise AssertionError("not called")


class _IdentityOnlyClient:
    async def discover_memory_relations(self, prompt, **kwargs):
        raise AssertionError("not called")


@pytest.mark.parametrize(
    ("client", "expects_discovery"), [(_RelationClient(), True), (_IdentityOnlyClient(), False)]
)
def test_engine_gives_discovery_and_identity_their_own_classifiers(client, expects_discovery: bool) -> None:
    from memforge.memory.engine import MemoryEngine
    from memforge.memory.sparse_relation_classifier import SparseMemoryRelationClassifier

    engine = MemoryEngine(
        cross_document_candidates=None,  # type: ignore[arg-type]
        db=None,  # type: ignore[arg-type]
        memory_store=None,  # type: ignore[arg-type]
        structured_llm_client=client,
    )

    assert isinstance(engine.pair_classifier, StructuredCrossDocumentRelationClassifier) is expects_discovery
    assert (engine.pair_classifier is None) is not expects_discovery
    assert isinstance(engine.identity_resolver._pair_classifier, SparseMemoryRelationClassifier)  # noqa: SLF001
