"""Cross-document relation contract: one closed label per Memory pair.

Discovery compares a committed Memory with bounded candidates from other Source
Units after commit. Each pair receives exactly one label. A relation annotates
both Memories for readers; it never changes either Memory's lifecycle.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from memforge.llm.batch_runner import ItemTask, LlmBatchRunner, LlmRequest
from memforge.llm.structured import CrossDocumentRelationResponse
from memforge.memory.evidence import EvidenceRole, MemoryEvidenceUnitProjection
from memforge.memory.relation_classifier import MemoryPairClassificationPolicy, run_pair_items
from memforge.models import Memory


class CrossDocumentRelationLabel(str, Enum):
    NONE = "none"
    EQUIVALENT = "equivalent"
    UPDATES = "updates"
    CONTRADICTS = "contradicts"


CROSS_DOCUMENT_RELATION_CLASSIFIER_VERSION = "cross-document-relation-v1"

# Enough of the Primary Evidence to show the statement's context, not the document.
RELATION_SUBJECT_EXCERPT_CHARS = 1200

# Requested output per request: a response envelope plus one label and one
# sentence of reasoning per pair.
_OUTPUT_BASE_TOKENS = 256
_OUTPUT_TOKENS_PER_PAIR = 192


@dataclass(frozen=True, slots=True)
class RelationSubject:
    """What the classifier sees of one Memory; identity and dates stay out of the prompt."""

    memory_id: str
    content_hash: str
    statement: str
    memory_type: str
    source_type: str | None = None
    document_title: str | None = None
    primary_excerpt: str | None = None

    def prompt_payload(self) -> dict[str, str]:
        payload = {
            "statement": self.statement,
            "memory_type": self.memory_type,
            "source_type": self.source_type,
            "document_title": self.document_title,
            "primary_excerpt": self.primary_excerpt,
        }
        return {key: value for key, value in payload.items() if value}

    def to_manifest(self) -> dict[str, str | None]:
        return {
            "memory_id": self.memory_id,
            "content_hash": self.content_hash,
            "statement": self.statement,
            "memory_type": self.memory_type,
            "source_type": self.source_type,
            "document_title": self.document_title,
            "primary_excerpt": self.primary_excerpt,
        }

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> RelationSubject:
        def optional(key: str) -> str | None:
            value = payload.get(key)
            return str(value) if value else None

        return cls(
            memory_id=str(payload["memory_id"]),
            content_hash=str(payload["content_hash"]),
            statement=str(payload["statement"]),
            memory_type=str(payload["memory_type"]),
            source_type=optional("source_type"),
            document_title=optional("document_title"),
            primary_excerpt=optional("primary_excerpt"),
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


CROSS_DOCUMENT_RELATION_RULES = """Each pair holds two statements taken from different documents.
Give every pair exactly one label:
- none: no relation a reader needs. This includes statements about different
  subjects, statements that can both hold under any reading, and one statement
  being more specific than the other.
- equivalent: both statements state the same knowledge.
- updates: both statements apply to the same situation and cannot both hold now,
  and the texts show that the knowledge changed over time. Do not decide which
  statement is newer.
- contradicts: both statements apply to the same situation and cannot both
  hold, and the texts show no change over time between them.

If two statements can both hold under any reading, the label is none.
When you are not certain, the label is none: a false conflict warns every
reader of both statements, while a missed one only omits a hint.
The document title and the Primary Evidence excerpt show where a statement
comes from; judge the statements themselves.
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


def primary_evidence_unit(
    units: Sequence[MemoryEvidenceUnitProjection],
) -> MemoryEvidenceUnitProjection | None:
    """The Evidence Unit a relation subject is shown from: the first current one."""

    current = [unit for unit in units if unit.current]
    return (current or list(units) or [None])[0]


def _primary_excerpt(unit: MemoryEvidenceUnitProjection) -> str | None:
    excerpt = next(
        (
            item.excerpt
            for item in unit.items
            if item.role is EvidenceRole.PRIMARY and item.excerpt
        ),
        None,
    )
    return excerpt[:RELATION_SUBJECT_EXCERPT_CHARS] if excerpt else None


async def load_relation_subjects(
    store: RelationSubjectStore,
    memories: Sequence[Memory],
) -> Mapping[str, RelationSubject]:
    """Build the classifier input for each Memory from its current Primary Evidence.

    Discovery and the evaluation set both build subjects here, so an evaluated
    pair shows the model the same statements, titles and excerpts that
    discovery shows it. Discovery groups a challenger's pairs into one request;
    the evaluation sends each pinned pair on its own.
    """

    titles: dict[str, str | None] = {}
    subjects: dict[str, RelationSubject] = {}
    for memory in memories:
        if memory.id in subjects:
            continue
        unit = primary_evidence_unit(await store.get_memory_evidence_units(memory.id))
        title: str | None = None
        if unit is not None and unit.doc_id:
            if unit.doc_id not in titles:
                document = await store.get_document(unit.doc_id)
                titles[unit.doc_id] = getattr(document, "title", None) or None
            title = titles[unit.doc_id]
        subjects[memory.id] = RelationSubject(
            memory_id=memory.id,
            content_hash=memory.content_hash,
            statement=memory.content,
            memory_type=memory.memory_type,
            source_type=unit.source_type if unit is not None else None,
            document_title=title,
            primary_excerpt=_primary_excerpt(unit) if unit is not None else None,
        )
    return subjects
