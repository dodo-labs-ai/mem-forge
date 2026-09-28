"""Cross-document relation contract: one closed label per Memory pair.

Discovery compares a committed Memory with bounded candidates from other Source
Units after commit. Each pair receives exactly one label. A relation annotates
both Memories for readers; it never changes either Memory's lifecycle.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Protocol

from memforge.llm.batch_runner import ItemTask, LlmBatchRunner, LlmRequest, RejectedRow
from memforge.llm.decision_model import DecisionTask, decision_task_model
from memforge.llm.structured import CrossDocumentRelationResponse
from memforge.memory.evidence import (
    EvidencePartKind,
    EvidenceRole,
    MemoryEvidenceItemProjection,
    MemoryEvidenceUnitProjection,
)
from memforge.memory.relation_classifier import MemoryPairClassificationPolicy, run_pair_items
from memforge.models import Memory
from memforge.pipeline.evidence_fragments import (
    CanonicalFieldRange,
    EvidenceCandidateRange,
    EvidenceFragment,
    canonical_record_field_ranges,
    compile_fragments,
)
from memforge.source_projection import AnchorKind, SourceObservationRevision
from memforge.source_representation import representation_contract_for_profile


class CrossDocumentRelationLabel(str, Enum):
    NONE = "none"
    EQUIVALENT = "equivalent"
    UPDATES = "updates"
    CONTRADICTS = "contradicts"


class CrossDocumentRelationDecider(str, Enum):
    """Who decided a stored relation: the classifier or a person's confirmed Review."""

    CLASSIFIER = "classifier"
    REVIEW = "review"


CROSS_DOCUMENT_RELATION_CLASSIFIER_VERSION = "cross-document-relation-v4"
# Classifier contracts that read the current ``RelationSubject`` input. An
# evaluation case pinned under any of them holds the input this contract reads
# and is replayed under it; a contract that changes the input is evaluated on a
# set seeded again under it.
CROSS_DOCUMENT_RELATION_INPUT_VERSIONS = frozenset(
    {"cross-document-relation-v2", "cross-document-relation-v3", CROSS_DOCUMENT_RELATION_CLASSIFIER_VERSION}
)
CROSS_DOCUMENT_RELATION_TASK = DecisionTask("cross_document_relation", CROSS_DOCUMENT_RELATION_CLASSIFIER_VERSION)

# Requested output per request: a response envelope plus one label-only row per
# pair. A {"pair_index": ..., "label": ...} row is under 20 tokens with its JSON
# punctuation; the allowance rounds it up to cover large pair indexes.
_OUTPUT_BASE_TOKENS = 256
_OUTPUT_TOKENS_PER_PAIR = 32


@dataclass(frozen=True, slots=True)
class RelationSubject:
    """What the classifier sees of one Memory; identity stays out of the prompt.

    ``evidence_time`` is the UTC date on which the source recorded the Primary
    Evidence, or None when no source time is known. ``evidence`` holds the
    readable text of the Primary Evidence followed by any Required Evidence.
    """

    memory_id: str
    content_hash: str
    statement: str
    memory_type: str
    source_type: str | None = None
    document_title: str | None = None
    evidence_time: str | None = None
    evidence: tuple[str, ...] = ()

    def prompt_payload(self) -> dict[str, object]:
        return {
            "statement": self.statement,
            "memory_type": self.memory_type,
            "source_type": self.source_type,
            "document_title": self.document_title,
            "evidence_time": self.evidence_time,
            "evidence": list(self.evidence),
        }

    def to_manifest(self) -> dict[str, object]:
        return {
            "memory_id": self.memory_id,
            "content_hash": self.content_hash,
            **self.prompt_payload(),
        }

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> RelationSubject:
        def optional(key: str) -> str | None:
            value = payload[key]
            return str(value) if value else None

        evidence = payload["evidence"]
        if not isinstance(evidence, list):
            raise ValueError("relation subject evidence must be a list")
        return cls(
            memory_id=str(payload["memory_id"]),
            content_hash=str(payload["content_hash"]),
            statement=str(payload["statement"]),
            memory_type=str(payload["memory_type"]),
            source_type=optional("source_type"),
            document_title=optional("document_title"),
            evidence_time=optional("evidence_time"),
            evidence=tuple(str(text) for text in evidence),
        )


@dataclass(frozen=True, slots=True)
class CrossDocumentRelationPair:
    challenger: RelationSubject
    candidate: RelationSubject

    @property
    def key(self) -> tuple[str, str]:
        return self.challenger.memory_id, self.candidate.memory_id


@dataclass(frozen=True, slots=True)
class CrossDocumentRelationJudgment:
    pair: CrossDocumentRelationPair
    label: CrossDocumentRelationLabel

    @property
    def recorded_label(self) -> CrossDocumentRelationLabel:
        """The label this judgment is recorded with; see ``recorded_relation_label``."""

        times = {subject.memory_id: subject.evidence_time for subject in (self.pair.challenger, self.pair.candidate)}
        low_id, high_id = pair_key(*times)
        return recorded_relation_label(self.label, low_id, high_id, times[low_id], times[high_id])


@dataclass(frozen=True, slots=True)
class CrossDocumentRelationClassification:
    judgments: tuple[CrossDocumentRelationJudgment, ...]
    llm_calls: int
    prompt_chars: int


class CrossDocumentRelationClassifier(Protocol):
    async def classify(
        self,
        pairs: tuple[CrossDocumentRelationPair, ...],
    ) -> CrossDocumentRelationClassification: ...


def pair_key(first_memory_id: str, second_memory_id: str) -> tuple[str, str]:
    """The stored order of a pair: the lower Memory id first."""

    if first_memory_id == second_memory_id:
        raise ValueError("a relation needs two different Memories")
    return (
        (first_memory_id, second_memory_id)
        if first_memory_id < second_memory_id
        else (second_memory_id, first_memory_id)
    )


def pair_counterpart(memory_low_id: str, memory_high_id: str, memory_id: str) -> str:
    """The other Memory of a stored pair."""

    if memory_id == memory_low_id:
        return memory_high_id
    if memory_id == memory_high_id:
        return memory_low_id
    raise ValueError(f"{memory_id} is not part of this pair")


def _evidence_date(evidence_time: str | None) -> date | None:
    if not evidence_time:
        return None
    try:
        return date.fromisoformat(evidence_time)
    except ValueError:
        return None


def newer_memory_by_evidence_time(
    memory_low_id: str,
    memory_high_id: str,
    low_evidence_time: str | None,
    high_evidence_time: str | None,
) -> str | None:
    """The Memory whose Evidence the source recorded later, or None when the times do not order the pair."""

    low = _evidence_date(low_evidence_time)
    high = _evidence_date(high_evidence_time)
    if low is None or high is None or low == high:
        return None
    return memory_low_id if low > high else memory_high_id


def recorded_relation_label(
    label: CrossDocumentRelationLabel,
    memory_low_id: str,
    memory_high_id: str,
    low_evidence_time: str | None,
    high_evidence_time: str | None,
) -> CrossDocumentRelationLabel:
    """The label a relation is recorded with.

    The program, not the classifier, orders an ``updates`` pair, by the same
    Evidence times the classifier saw (``RelationSubject.evidence_time``). A
    pair those times do not order, because either is unknown or both fall on
    the same date, is recorded as ``contradicts``.
    """

    if label is CrossDocumentRelationLabel.UPDATES and (
        newer_memory_by_evidence_time(memory_low_id, memory_high_id, low_evidence_time, high_evidence_time) is None
    ):
        return CrossDocumentRelationLabel.CONTRADICTS
    return label


def _check_recorded_relation(
    label: CrossDocumentRelationLabel,
    memory_low_id: str,
    memory_high_id: str,
    low_evidence_time: str | None,
    high_evidence_time: str | None,
) -> None:
    if memory_low_id >= memory_high_id:
        raise ValueError("a relation pair must be stored lower Memory id first")
    if label is CrossDocumentRelationLabel.NONE:
        raise ValueError("a none judgment is not stored as a relation")
    if recorded_relation_label(label, memory_low_id, memory_high_id, low_evidence_time, high_evidence_time) is not label:
        raise ValueError("an updates relation needs Evidence times that order the pair")


@dataclass(frozen=True, slots=True)
class CrossDocumentRelationRecord:
    """One relation a discovery run found, bound to the two contents it judged.

    ``low_evidence_time`` and ``high_evidence_time`` are the Evidence times the
    classifier saw for each Memory; they order an ``updates`` pair.
    """

    memory_low_id: str
    memory_high_id: str
    label: CrossDocumentRelationLabel
    low_content_hash: str
    high_content_hash: str
    low_evidence_time: str | None
    high_evidence_time: str | None
    relation_run_id: str
    discovery_work_id: str
    classifier_version: str = CROSS_DOCUMENT_RELATION_CLASSIFIER_VERSION

    def __post_init__(self) -> None:
        _check_recorded_relation(
            self.label,
            self.memory_low_id,
            self.memory_high_id,
            self.low_evidence_time,
            self.high_evidence_time,
        )

    def counterpart_of(self, memory_id: str) -> str:
        return pair_counterpart(self.memory_low_id, self.memory_high_id, memory_id)

    @classmethod
    def from_judgment(
        cls,
        judgment: CrossDocumentRelationJudgment,
        *,
        relation_run_id: str,
        discovery_work_id: str,
    ) -> CrossDocumentRelationRecord:
        by_id = {
            judgment.pair.challenger.memory_id: judgment.pair.challenger,
            judgment.pair.candidate.memory_id: judgment.pair.candidate,
        }
        low_id, high_id = pair_key(*by_id)
        low, high = by_id[low_id], by_id[high_id]
        return cls(
            memory_low_id=low_id,
            memory_high_id=high_id,
            label=judgment.recorded_label,
            low_content_hash=low.content_hash,
            high_content_hash=high.content_hash,
            low_evidence_time=low.evidence_time,
            high_evidence_time=high.evidence_time,
            relation_run_id=relation_run_id,
            discovery_work_id=discovery_work_id,
        )


@dataclass(frozen=True, slots=True)
class CrossDocumentRelationOutcome:
    """What one discovery run judged for its challenger.

    ``judged_content_hashes`` maps every judged candidate to the content it was
    judged on; a judged candidate without a relation was judged none. Storage
    records the outcome only while the challenger and every judged candidate
    still hold those contents, and the outcome replaces every relation the
    classifier recorded for the same discovery work: a pair the run did not
    judge keeps no relation from that work.
    """

    challenger_id: str
    challenger_content_hash: str
    judged_content_hashes: Mapping[str, str]
    relations: tuple[CrossDocumentRelationRecord, ...] = ()

    def __post_init__(self) -> None:
        judged = {self.challenger_id: self.challenger_content_hash, **self.judged_content_hashes}
        for record in self.relations:
            counterpart = record.counterpart_of(self.challenger_id)
            if counterpart not in self.judged_content_hashes or (
                record.low_content_hash,
                record.high_content_hash,
            ) != (judged[record.memory_low_id], judged[record.memory_high_id]):
                raise ValueError("a relation must name a judged pair and the contents it was judged on")

    @classmethod
    def from_judgments(
        cls,
        challenger: RelationSubject,
        judgments: Sequence[CrossDocumentRelationJudgment],
        *,
        relation_run_id: str,
        discovery_work_id: str,
    ) -> CrossDocumentRelationOutcome:
        if any(judgment.pair.challenger != challenger for judgment in judgments):
            raise ValueError("a discovery outcome judges pairs of one challenger")
        return cls(
            challenger_id=challenger.memory_id,
            challenger_content_hash=challenger.content_hash,
            judged_content_hashes={
                judgment.pair.candidate.memory_id: judgment.pair.candidate.content_hash
                for judgment in judgments
            },
            relations=tuple(
                CrossDocumentRelationRecord.from_judgment(
                    judgment,
                    relation_run_id=relation_run_id,
                    discovery_work_id=discovery_work_id,
                )
                for judgment in judgments
                if judgment.label is not CrossDocumentRelationLabel.NONE
            ),
        )


# Relations shown per Memory, most consequential label first.
MAX_RELATIONS_PER_MEMORY = 10
RELATION_READ_ORDER = (
    CrossDocumentRelationLabel.CONTRADICTS,
    CrossDocumentRelationLabel.UPDATES,
    CrossDocumentRelationLabel.EQUIVALENT,
)


@dataclass(frozen=True, slots=True)
class CurrentCrossDocumentRelation:
    """A stored relation that is current for the caller.

    Current means both Memories are active and visible to the caller, both
    contents still equal the judged contents, and no active dismissal hides
    it. The Evidence times are the ones the relation was decided on; they
    order an ``updates`` pair and date each Memory for the reader.
    """

    memory_low_id: str
    memory_high_id: str
    label: CrossDocumentRelationLabel
    low_content_hash: str
    high_content_hash: str
    low_evidence_time: str | None
    high_evidence_time: str | None
    decided_by: CrossDocumentRelationDecider
    decided_at: str

    def __post_init__(self) -> None:
        _check_recorded_relation(
            self.label,
            self.memory_low_id,
            self.memory_high_id,
            self.low_evidence_time,
            self.high_evidence_time,
        )

    def counterpart_of(self, memory_id: str) -> str:
        return pair_counterpart(self.memory_low_id, self.memory_high_id, memory_id)

    def evidence_time_of(self, memory_id: str) -> str | None:
        if memory_id == self.memory_low_id:
            return self.low_evidence_time
        if memory_id == self.memory_high_id:
            return self.high_evidence_time
        raise ValueError(f"{memory_id} is not part of this pair")

    @property
    def newer_memory_id(self) -> str | None:
        """The newer Memory of an ``updates`` pair; None for every other label."""

        if self.label is not CrossDocumentRelationLabel.UPDATES:
            return None
        return newer_memory_by_evidence_time(
            self.memory_low_id,
            self.memory_high_id,
            self.low_evidence_time,
            self.high_evidence_time,
        )


@dataclass(frozen=True, slots=True)
class CrossDocumentRelationDismissal:
    """A person's record that one relation is wrong for both current contents."""

    id: str
    memory_low_id: str
    memory_high_id: str
    label: CrossDocumentRelationLabel
    low_content_hash: str
    high_content_hash: str
    dismissed_by: str
    dismissed_at: str
    note: str | None = None

    def counterpart_of(self, memory_id: str) -> str:
        return pair_counterpart(self.memory_low_id, self.memory_high_id, memory_id)


class RelationDismissalConflict(ValueError):
    """The dismissal names content or a relation that is no longer current."""


CROSS_DOCUMENT_RELATION_RULES = """Each pair holds two statements taken from different documents.
Each statement comes with its memory type, the source type and title of its
document, evidence_time (the date on which its source recorded the Evidence,
null when unknown), and the Evidence text it was taken from. Use all of them
to tell what each statement is about.

Two statements are about the same situation only if all of these hold:
- they concern the same object;
- they have the same scope: the conditions under which one applies do not
  exclude the other;
- they are the same kind of statement: both say what should be (a
  requirement, design, rule or expected behaviour), both say what actually
  happened (a reported defect, test result, incident or observed behaviour),
  or both say what was planned or decided. Read the kind from what each
  statement says, not from its memory type;
- they concern the same occurrence: statements about different events,
  tickets, incidents, test runs or cases are about different situations, while two statements about the same
  lasting state or decision are about the same situation even when their
  sources recorded them at different times.
If any of these differs, or the inputs do not establish it, the statements
are not about the same situation.

Decide whether the statements are about the same situation before deciding
whether both can hold. Compare only what the statements state: a conflict that
has to be inferred from either statement is not a relation. When the
statements overlap only in part, they conflict if the part they share cannot
hold for both; a shared part never makes them equivalent.

Give every pair exactly one label:
- none: the statements are not about the same situation, or they are about
  the same situation and both can hold without stating the same knowledge
  (for example, one is more specific than the other, or one adds a fact,
  condition, step or outcome the other does not state).
- equivalent: same situation, and each statement as a whole states everything
  the other states; they differ only in wording.
- updates: same situation, both cannot hold now, and the later statement
  replaces the earlier one.
- contradicts: same situation, the two statements directly state
  incompatible things about it, and neither replaces the other over time.

When you are not certain, the label is none: a false relation warns every
reader of both statements, while a missed one only omits a hint.
"""

# Every request states one challenger once, as the subject, and asks one
# question per candidate. Candidates of one subject often repeat one another,
# so the prompt says that such repetition tells nothing about the subject.
CROSS_DOCUMENT_RELATION_QUESTION = "What is the relation of this candidate to the subject?"
CROSS_DOCUMENT_RELATION_PROMPT = CROSS_DOCUMENT_RELATION_RULES + """
Every question below asks for the relation between one candidate statement and the subject statement. \
The subject is one statement; each candidate is one statement from another document.
Many candidates repeat one another. Judge each candidate only against the subject: \
two candidates that state the same thing tell nothing about the subject, \
and a candidate is equivalent only when it and the subject state the same knowledge.

<subject>
{subject_json}
</subject>

<questions>
{questions_json}
</questions>

For every question, answer with its pair_index and the label for that candidate's relation to the subject. \
Return exactly one decision for every pair_index and no other pair_index.
"""


def cross_document_relation_prompt(
    subject: RelationSubject,
    questions: Sequence[tuple[int, RelationSubject]],
) -> str:
    """One request: the subject stated once and one question per ``(pair_index, candidate)``."""

    return CROSS_DOCUMENT_RELATION_PROMPT.format(
        subject_json=json.dumps(subject.prompt_payload(), ensure_ascii=False),
        questions_json=json.dumps(
            [
                {
                    "pair_index": pair_index,
                    "question": CROSS_DOCUMENT_RELATION_QUESTION,
                    "candidate": candidate.prompt_payload(),
                }
                for pair_index, candidate in questions
            ],
            ensure_ascii=False,
        ),
    )


def cross_document_relation_output_tokens(policy: MemoryPairClassificationPolicy, pair_count: int) -> int:
    return min(policy.max_output_tokens, _OUTPUT_BASE_TOKENS + _OUTPUT_TOKENS_PER_PAIR * pair_count)


class StructuredCrossDocumentRelationClassifier:
    """Label exact pairs with the Structured LLM; any pair left without a label fails the run.

    Pairs are asked about per challenger: every request holds the pairs of one
    challenger, which it states once as the subject. The batch runner packs and
    splits each challenger's questions by capacity.
    """

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        policy: MemoryPairClassificationPolicy | None = None,
    ) -> None:
        self._client = client
        self._model = decision_task_model(client, CROSS_DOCUMENT_RELATION_TASK, model)
        self._policy = policy or MemoryPairClassificationPolicy()

    async def classify(
        self,
        pairs: tuple[CrossDocumentRelationPair, ...],
    ) -> CrossDocumentRelationClassification:
        runner = LlmBatchRunner(self._client, model=self._model)
        by_challenger: dict[str, list[int]] = {}
        for pair_index, pair in enumerate(pairs):
            by_challenger.setdefault(pair.challenger.memory_id, []).append(pair_index)
        judgments: dict[int, CrossDocumentRelationJudgment] = {}
        for pair_indexes in by_challenger.values():
            results = await run_pair_items(
                runner,
                self._challenger_task(pairs, pair_indexes),
                pair_count=len(pairs),
                label="cross-document relation classification",
            )
            judgments.update(zip(pair_indexes, results, strict=True))
        return CrossDocumentRelationClassification(
            judgments=tuple(judgments[pair_index] for pair_index in range(len(pairs))),
            llm_calls=runner.stats.calls,
            prompt_chars=runner.stats.prompt_chars,
        )

    def _challenger_task(
        self,
        pairs: tuple[CrossDocumentRelationPair, ...],
        pair_indexes: Sequence[int],
    ) -> ItemTask:
        """The questions about one challenger's pairs, with the challenger as their shared context."""

        asked = {pair_index: pairs[pair_index] for pair_index in pair_indexes}

        def render(item_ids: tuple[str, ...], context: tuple[RelationSubject, ...]) -> LlmRequest:
            [subject] = context
            prompt = cross_document_relation_prompt(
                subject,
                [(int(item_id), asked[int(item_id)].candidate) for item_id in item_ids],
            )
            return LlmRequest(
                prompt,
                CrossDocumentRelationResponse,
                cross_document_relation_output_tokens(self._policy, len(item_ids)),
            )

        def decode(response: CrossDocumentRelationResponse, _item_ids: tuple[str, ...], _context: tuple):
            """Each decision is its pair's row; the runner rejects an unrequested pair_index."""
            for decision in response.decisions:
                pair_index = int(decision.pair_index)
                if pair_index not in asked:
                    yield str(pair_index), RejectedRow(f"pair_index {pair_index} was not requested")
                    continue
                yield str(pair_index), CrossDocumentRelationJudgment(
                    pair=asked[pair_index],
                    label=CrossDocumentRelationLabel(decision.label),
                )

        return ItemTask(
            item_ids=tuple(str(pair_index) for pair_index in pair_indexes),
            render=render,
            decode=decode,
            call=self._client.classify_cross_document_relations,
            context=(pairs[pair_indexes[0]].challenger,),
            label=lambda item_id: f"pair_index {item_id}",
        )


class RelationSubjectStore(Protocol):
    async def get_memory_evidence_units(
        self,
        memory_id: str,
    ) -> tuple[MemoryEvidenceUnitProjection, ...]: ...

    async def get_document(self, doc_id: str) -> Any: ...

    async def get_current_source_observation_revisions(
        self,
        source_unit_id: str,
    ) -> Mapping[str, SourceObservationRevision]: ...


def source_evidence_time(value: object) -> str | None:
    """The Evidence time a stored source timestamp gives: its UTC calendar date, or None when unparsable."""

    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, str) and value.strip():
        try:
            moment = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if moment.tzinfo is not None:
        moment = moment.astimezone(timezone.utc)
    return moment.date().isoformat()


def _evidence_time(primary_revision: SourceObservationRevision | None) -> str | None:
    """When the source gave the Primary Evidence its content.

    ``primary_revision`` is the Observation Revision the Primary is anchored
    to, present only while it is the current revision. Its ``observed_at`` is
    the source's own time for that content (a page version, a commit, a
    comment, a message, an agent event), or None where the source records none.
    """

    if primary_revision is None:
        return None
    return source_evidence_time(primary_revision.observed_at)


def newest_evidence_unit_id(evidence_times: Mapping[str, str | None]) -> str | None:
    """The Evidence Unit a Memory's relation subject is read from.

    ``evidence_times`` maps each current Evidence Unit of the Memory to its
    Evidence time. The subject reads the Unit with the newest Evidence time, the
    time ``updates`` is ordered by; an unknown time ranks below every known one,
    and Units with the same time go to the lowest Evidence Unit id, so the
    choice is the same on every read. Storage applies this rule when discovery
    reads the challenger's Evidence in its Source Unit, and the subject builder
    applies it to every other Memory.
    """

    return max(
        sorted(evidence_times),
        key=lambda unit_id: (evidence_times[unit_id] is not None, evidence_times[unit_id] or ""),
        default=None,
    )


def _field_value_text(field: CanonicalFieldRange) -> str:
    """A structured record field by its schema-declared business keys, never its raw JSON."""

    value = field.comparison_value
    if isinstance(value, tuple):
        return ", ".join(f"{key}={item}" for key, item in value)
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _record_text(revision: SourceObservationRevision, fragments: Sequence[EvidenceFragment]) -> str:
    """Anchored Fragments of a canonical record, one line per field in source order.

    Each field is labeled by its full JSON pointer, so fields of one array item
    (a changelog item's field, previous value and new value) stay together and
    keep their item index. A text field shows its anchored Fragments; a
    structured field shows its business keys instead of the raw JSON value.
    Fields with nothing to show are left out.
    """

    remaining = list(fragments)
    sections: list[str] = []
    for field in sorted(canonical_record_field_ranges(revision), key=lambda field: field.start):
        inside = [
            fragment
            for fragment in remaining
            if field.start <= (fragment.anchor.range_start or 0)
            and (fragment.anchor.range_end or 0) <= field.end
        ]
        if not inside:
            continue
        remaining = [fragment for fragment in remaining if fragment not in inside]
        shown = (
            "\n\n".join(fragment.presentation_text for fragment in inside)
            if isinstance(field.value, str)
            else _field_value_text(field)
        ).strip()
        if shown:
            sections.append(f"{field.descriptor.json_pointer.lstrip('/')}: {shown}")
    return "\n".join(sections)


def _anchored_revision(
    item: MemoryEvidenceItemProjection,
    revisions: Mapping[str, SourceObservationRevision],
) -> SourceObservationRevision | None:
    """The exact Observation Revision an Evidence item is anchored to, when it is current."""

    revision = revisions.get(item.anchor.observation_id)
    return revision if revision is not None and revision.id == item.anchor.observation_revision_id else None


def _anchored_text(
    item: MemoryEvidenceItemProjection,
    revision: SourceObservationRevision | None,
) -> str | None:
    """The readable text of one text Evidence item within its exact Anchor.

    The anchored range is compiled into Evidence Fragments under the revision's
    representation profile, the presentation Support Assessment reads; a
    canonical record shows each anchored field by name. A whole-Observation
    Anchor therefore shows the whole Observation, bounded by the Fragment
    catalog's presentation limit. Without the exact revision, a range Anchor
    keeps its stored excerpt and a whole-Observation Anchor shows nothing,
    because its stored excerpt is the raw Observation content.
    """

    if item.kind is not EvidencePartKind.TEXT:
        return None
    if revision is not None:
        catalog = compile_fragments(
            revision,
            (EvidenceCandidateRange(anchor=item.anchor, primary_eligible=True),),
        )
        if catalog.usable:
            contract = representation_contract_for_profile(revision.evidence_profile)
            if contract is not None and contract.canonical_schema is not None:
                return _record_text(revision, catalog.fragments)
            return "\n\n".join(fragment.presentation_text for fragment in catalog.fragments)
    if item.anchor.kind is AnchorKind.REVISION_RANGE:
        return item.excerpt or None
    return None


class _RelationEvidenceReader:
    """Reads each Memory's relation Evidence, loading each Source Unit's current revisions once."""

    def __init__(self, store: RelationSubjectStore) -> None:
        self._store = store
        self._revisions: dict[str, Mapping[str, SourceObservationRevision]] = {}

    async def revisions(self, source_unit_id: str) -> Mapping[str, SourceObservationRevision]:
        if source_unit_id not in self._revisions:
            self._revisions[source_unit_id] = await self._store.get_current_source_observation_revisions(
                source_unit_id
            )
        return self._revisions[source_unit_id]

    async def evidence_time(self, unit: MemoryEvidenceUnitProjection) -> str | None:
        primary = _primary_item(unit)
        if primary is None:
            return None
        return _evidence_time(_anchored_revision(primary, await self.revisions(unit.source_unit_id)))

    async def evidence_unit(
        self,
        memory_id: str,
        *,
        evidence_unit_id: str | None = None,
    ) -> MemoryEvidenceUnitProjection | None:
        """The Evidence Unit a Memory is shown from.

        Among the Memory's current Evidence Units it is the one
        ``newest_evidence_unit_id`` chooses; a Memory without current Evidence
        is shown from its first Evidence Unit. ``evidence_unit_id`` names a Unit
        the caller already chose by the same rule, which must still support the
        Memory.
        """

        units = await self._store.get_memory_evidence_units(memory_id)
        if evidence_unit_id is not None:
            chosen = next((unit for unit in units if unit.evidence_unit_id == evidence_unit_id), None)
            if chosen is None:
                raise ValueError("relation evidence is no longer current")
            return chosen
        current = {unit.evidence_unit_id: unit for unit in units if unit.current}
        if not current:
            return units[0] if units else None
        times = {unit_id: await self.evidence_time(unit) for unit_id, unit in current.items()}
        return current[newest_evidence_unit_id(times)]


def _primary_item(unit: MemoryEvidenceUnitProjection) -> MemoryEvidenceItemProjection | None:
    return next(
        (item for item in unit.items if item.grants_support and item.role is EvidenceRole.PRIMARY),
        None,
    )


async def load_relation_evidence_units(
    store: RelationSubjectStore,
    memories: Sequence[Memory],
) -> Mapping[str, MemoryEvidenceUnitProjection | None]:
    """The Evidence Unit each Memory's relation subject is read from."""

    reader = _RelationEvidenceReader(store)
    return {memory.id: await reader.evidence_unit(memory.id) for memory in memories}


async def load_relation_subjects(
    store: RelationSubjectStore,
    memories: Sequence[Memory],
    *,
    evidence_unit_ids: Mapping[str, str] | None = None,
) -> Mapping[str, RelationSubject]:
    """Build the classifier input for each Memory from its current Evidence.

    Each Memory is shown from the Evidence Unit ``newest_evidence_unit_id``
    chooses among its current ones. ``evidence_unit_ids`` names the Unit for a
    Memory whose Evidence the caller already chose: discovery shows its
    challenger from the Evidence Unit it records the run against.

    Discovery and the evaluation set both build subjects here with the same
    rendering. Discovery chooses its challenger's Evidence Unit within the
    work's Source Unit; the evaluation set chooses among all of the
    challenger's current Units, so the two show the same Evidence unless the
    challenger has current Evidence in several Source Units. Discovery asks
    about all of a challenger's candidates in requests that state the
    challenger once; the evaluation sends each pinned pair on its own.
    """

    reader = _RelationEvidenceReader(store)
    chosen_units = evidence_unit_ids or {}
    documents: dict[str, Any] = {}
    subjects: dict[str, RelationSubject] = {}
    for memory in memories:
        if memory.id in subjects:
            continue
        unit = await reader.evidence_unit(memory.id, evidence_unit_id=chosen_units.get(memory.id))
        if unit is None:
            subjects[memory.id] = RelationSubject(
                memory_id=memory.id,
                content_hash=memory.content_hash,
                statement=memory.content,
                memory_type=memory.memory_type,
            )
            continue
        if unit.doc_id and unit.doc_id not in documents:
            documents[unit.doc_id] = await store.get_document(unit.doc_id)
        document = documents.get(unit.doc_id) if unit.doc_id else None
        unit_revisions = await reader.revisions(unit.source_unit_id)
        supporting = [item for item in unit.items if item.grants_support]
        evidence = tuple(
            text
            for item in supporting
            if (text := _anchored_text(item, _anchored_revision(item, unit_revisions)))
        )
        subjects[memory.id] = RelationSubject(
            memory_id=memory.id,
            content_hash=memory.content_hash,
            statement=memory.content,
            memory_type=memory.memory_type,
            source_type=unit.source_type,
            document_title=getattr(document, "title", None) or None,
            evidence_time=await reader.evidence_time(unit),
            evidence=evidence,
        )
    return subjects
