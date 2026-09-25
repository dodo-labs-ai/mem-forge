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

from memforge.llm.batch_runner import ItemTask, LlmBatchRunner, LlmRequest
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
from memforge.pipeline.source_projection_adapters import SOURCE_TYPES_WITH_DOCUMENT_REVISION_TIME
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


CROSS_DOCUMENT_RELATION_CLASSIFIER_VERSION = "cross-document-relation-v2"

# Requested output per request: a response envelope plus one label and one
# sentence of reasoning per pair.
_OUTPUT_BASE_TOKENS = 256
_OUTPUT_TOKENS_PER_PAIR = 192


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
    reason: str


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
    reason: str
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
            label=recorded_relation_label(judgment.label, low_id, high_id, low.evidence_time, high.evidence_time),
            low_content_hash=low.content_hash,
            high_content_hash=high.content_hash,
            low_evidence_time=low.evidence_time,
            high_evidence_time=high.evidence_time,
            reason=judgment.reason,
            relation_run_id=relation_run_id,
            discovery_work_id=discovery_work_id,
        )


@dataclass(frozen=True, slots=True)
class CrossDocumentRelationOutcome:
    """What one discovery run judged for its challenger.

    ``judged_content_hashes`` maps every judged candidate to the content it was
    judged on; a judged candidate without a relation was judged none. Storage
    records the outcome only while the challenger and every judged candidate
    still hold those contents.
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
    reason: str
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
- they are the same kind of statement: both say what should be, both say
  what actually happened, or both say what was planned or decided;
- they concern the same occurrence: statements about different events or
  runs are about different situations, while two statements about the same
  lasting state or decision are about the same situation even when their
  sources recorded them at different times.
If any of these differs, or the inputs do not establish it, the statements
are not about the same situation.

Give every pair exactly one label:
- none: the statements are not about the same situation, or they are about
  the same situation and both can hold without stating the same knowledge
  (for example, one is more specific than the other).
- equivalent: same situation, and both state the same knowledge.
- updates: same situation, both cannot hold now, and the later statement
  replaces the earlier one.
- contradicts: same situation, both cannot hold, and neither replaces the
  other over time.

When you are not certain, the label is none: a false relation warns every
reader of both statements, while a missed one only omits a hint.
"""

CROSS_DOCUMENT_RELATION_PROMPT = CROSS_DOCUMENT_RELATION_RULES + """
<statement_pair_groups>
{groups_json}
</statement_pair_groups>

Return exactly one decision for every pair_index and no other pair_index.
Keep each reason to one sentence.
"""


def _grouped_pairs_json(indexed_pairs: Sequence[tuple[int, CrossDocumentRelationPair]]) -> str:
    groups: dict[str, dict[str, Any]] = {}
    for pair_index, pair in indexed_pairs:
        group = groups.setdefault(
            pair.challenger.memory_id,
            {"statement": pair.challenger.prompt_payload(), "compared_with": []},
        )
        group["compared_with"].append(
            {"pair_index": pair_index, "statement": pair.candidate.prompt_payload()}
        )
    return json.dumps(list(groups.values()), ensure_ascii=False)


def cross_document_relation_output_tokens(policy: MemoryPairClassificationPolicy, pair_count: int) -> int:
    return min(policy.max_output_tokens, _OUTPUT_BASE_TOKENS + _OUTPUT_TOKENS_PER_PAIR * pair_count)


class StructuredCrossDocumentRelationClassifier:
    """Label exact pairs with the Structured LLM; any pair left without a label fails the run."""

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        policy: MemoryPairClassificationPolicy | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._policy = policy or MemoryPairClassificationPolicy()

    async def classify(
        self,
        pairs: tuple[CrossDocumentRelationPair, ...],
    ) -> CrossDocumentRelationClassification:
        if not pairs:
            return CrossDocumentRelationClassification(judgments=(), llm_calls=0, prompt_chars=0)
        runner = LlmBatchRunner(self._client, model=self._model)

        def render(item_ids: tuple[str, ...], _context: tuple) -> LlmRequest:
            indexed = tuple((int(item_id), pairs[int(item_id)]) for item_id in item_ids)
            prompt = CROSS_DOCUMENT_RELATION_PROMPT.format(groups_json=_grouped_pairs_json(indexed))
            return LlmRequest(
                prompt,
                CrossDocumentRelationResponse,
                cross_document_relation_output_tokens(self._policy, len(item_ids)),
            )

        def decode(response: CrossDocumentRelationResponse, _item_ids: tuple[str, ...], _context: tuple):
            for decision in response.decisions:
                pair_index = int(decision.pair_index)
                if not 0 <= pair_index < len(pairs):
                    raise ValueError(f"unknown pair_index {pair_index}")
                yield str(pair_index), CrossDocumentRelationJudgment(
                    pair=pairs[pair_index],
                    label=CrossDocumentRelationLabel(decision.label),
                    reason=decision.reason.strip(),
                )

        judgments = await run_pair_items(
            runner,
            ItemTask(
                item_ids=tuple(str(index) for index in range(len(pairs))),
                render=render,
                decode=decode,
                call=self._client.classify_cross_document_relations,
            ),
            pair_count=len(pairs),
            label="cross-document relation classification",
        )
        return CrossDocumentRelationClassification(
            judgments=tuple(judgments),
            llm_calls=runner.stats.calls,
            prompt_chars=runner.stats.prompt_chars,
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


def primary_evidence_unit(
    units: Sequence[MemoryEvidenceUnitProjection],
) -> MemoryEvidenceUnitProjection | None:
    """The Evidence Unit a relation subject is shown from: the first current one."""

    current = [unit for unit in units if unit.current]
    return (current or list(units) or [None])[0]


def _source_date(value: object) -> str | None:
    """The UTC calendar date of a stored source timestamp, or None when unparsable."""

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


def _evidence_time(
    unit: MemoryEvidenceUnitProjection,
    primary_revision: SourceObservationRevision | None,
    document: Any,
) -> str | None:
    """When the source recorded the Primary Evidence.

    ``primary_revision`` is the Observation Revision the Primary is anchored
    to, present only while it is the current revision. Its own time comes first
    (a comment, a changelog entry, a message). A document time counts only where
    it is the revision time of the document body, which then is the anchored
    revision; a sync or submission time is never shown as a source time.
    """

    if primary_revision is None:
        return None
    observed = _source_date(primary_revision.observed_at)
    if observed is not None:
        return observed
    if unit.source_type in SOURCE_TYPES_WITH_DOCUMENT_REVISION_TIME and document is not None:
        return _source_date(getattr(document, "last_modified", None))
    return None


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


async def load_relation_subjects(
    store: RelationSubjectStore,
    memories: Sequence[Memory],
) -> Mapping[str, RelationSubject]:
    """Build the classifier input for each Memory from its current Evidence.

    Discovery and the evaluation set both build subjects here, so an evaluated
    pair shows the model the same statements, titles, Evidence times and
    Evidence text that discovery shows it. Discovery groups a challenger's pairs
    into one request; the evaluation sends each pinned pair on its own.
    """

    documents: dict[str, Any] = {}
    revisions: dict[str, Mapping[str, SourceObservationRevision]] = {}
    subjects: dict[str, RelationSubject] = {}
    for memory in memories:
        if memory.id in subjects:
            continue
        unit = primary_evidence_unit(await store.get_memory_evidence_units(memory.id))
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
        if unit.source_unit_id not in revisions:
            revisions[unit.source_unit_id] = await store.get_current_source_observation_revisions(
                unit.source_unit_id
            )
        unit_revisions = revisions[unit.source_unit_id]
        supporting = [item for item in unit.items if item.grants_support]
        primary = next((item for item in supporting if item.role is EvidenceRole.PRIMARY), None)
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
            evidence_time=_evidence_time(
                unit,
                _anchored_revision(primary, unit_revisions) if primary is not None else None,
                document,
            ),
            evidence=evidence,
        )
    return subjects
