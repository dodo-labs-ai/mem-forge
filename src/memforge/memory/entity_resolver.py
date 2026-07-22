"""Bounded batch resolution of extracted entity mentions."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from math import sqrt
from time import perf_counter
from typing import Any

from memforge.models import Entity, EntityAlias, canonicalize_entity_name
from memforge.storage.adapters.protocols import (
    EntityResolutionContext,
    EntityResolutionScope,
    EntityUpsert,
    RelationalStore,
)

logger = logging.getLogger(__name__)

__all__ = [
    "EntityResolutionBatch",
    "EntityResolutionContext",
    "EntityResolutionMetrics",
    "EntityResolutionPolicy",
    "EntityResolver",
    "validate_alias",
]


@dataclass(frozen=True, slots=True)
class EntityResolutionMetrics:
    unique_mentions: int
    exact_hits: int
    alias_hits: int
    embedded_mentions: int
    ambiguous_mentions: int
    embedding_batches: int
    structured_llm_calls: int
    candidate_count: int
    new_entities: int
    elapsed_ms: int


@dataclass(frozen=True, slots=True)
class EntityResolutionBatch:
    """Resolved canonical IDs plus content-free batch metrics."""

    ids_by_canonical_name: Mapping[str, int]
    metrics: EntityResolutionMetrics

    def entity_id(self, mention: str) -> int | None:
        return self.ids_by_canonical_name.get(canonicalize_entity_name(mention))


@dataclass(frozen=True, slots=True)
class EntityResolutionPolicy:
    """Provider-neutral bounds for storage, embedding, and adjudication batches."""

    context_batch_size: int = 64
    embedding_batch_size: int = 256
    adjudication_batch_size: int = 32
    max_adjudication_prompt_chars: int = 32_000

    def __post_init__(self) -> None:
        for name, value in (
            ("context_batch_size", self.context_batch_size),
            ("embedding_batch_size", self.embedding_batch_size),
            ("adjudication_batch_size", self.adjudication_batch_size),
            ("max_adjudication_prompt_chars", self.max_adjudication_prompt_chars),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive")


def validate_alias(alias_name: str, canonical_name: str) -> bool:
    """Return whether two spellings have a deterministic lexical resemblance."""

    alias = canonicalize_entity_name(alias_name)
    canonical = canonicalize_entity_name(canonical_name)
    if not alias or not canonical:
        return False
    alias_tokens = set(alias.split())
    canonical_tokens = set(canonical.split())
    return bool(
        alias_tokens & canonical_tokens
        or alias in canonical
        or canonical in alias
        or SequenceMatcher(None, alias, canonical).ratio() >= 0.5
    )


_ENTITY_BATCH_PROMPT = """Resolve each entity mention against only its supplied candidates.

Mentions and candidate IDs:
{cases_json}

Document context:
{context}

Return one decision for every mention. matched_id must be one supplied candidate
ID or null. Related, parent/child, or same-category entities are not identical.
Return only the structured response."""


class EntityResolver:
    """Own bounded entity recall, ambiguity proof, alias learning, and creation."""

    def __init__(
        self,
        store: RelationalStore,
        embed_cfg: dict | None = None,
        structured_llm_client: Any = None,
        llm_model: str = "claude-sonnet-4-20250514",
        embedding_threshold: float = 0.6,
        candidate_limit: int = 10,
        confidence_threshold: float = 0.9,
        policy: EntityResolutionPolicy | None = None,
    ) -> None:
        self.store = store
        self.embed_cfg = embed_cfg
        self.structured_llm_client = structured_llm_client
        self.llm_model = llm_model
        self.embedding_threshold = embedding_threshold
        self.candidate_limit = candidate_limit
        self.confidence_threshold = confidence_threshold
        self.policy = policy or EntityResolutionPolicy()

    async def resolve_many(
        self,
        mentions: Sequence[str],
        *,
        scope: EntityResolutionScope,
        doc_context: str | None = None,
    ) -> EntityResolutionBatch:
        """Resolve distinct mentions with bounded storage and model batches."""

        started = perf_counter()
        display_by_canonical: dict[str, str] = {}
        for mention in mentions:
            canonical = canonicalize_entity_name(mention)
            if canonical:
                display_by_canonical.setdefault(canonical, mention.strip() or canonical)
        canonical_names = tuple(display_by_canonical)
        if not canonical_names:
            return self._finish_batch(started=started, resolved={})

        exact_matches: dict[str, Entity] = {}
        alias_matches: dict[str, EntityAlias] = {}
        recalled_candidates: dict[str, tuple[Entity, ...]] = {}
        for start in range(0, len(canonical_names), self.policy.context_batch_size):
            batch_names = canonical_names[start : start + self.policy.context_batch_size]
            context = await self.store.load_entity_resolution_context(
                batch_names,
                candidate_limit=self.candidate_limit,
                scope=scope,
            )
            exact_matches.update(
                (name, context.exact_matches[name])
                for name in batch_names
                if name in context.exact_matches
            )
            alias_matches.update(
                (name, context.alias_matches[name])
                for name in batch_names
                if name in context.alias_matches
            )
            recalled_candidates.update(
                (name, tuple(context.candidates.get(name, ())))
                for name in batch_names
            )
        resolved: dict[str, int] = {
            canonical: entity.id
            for canonical, entity in exact_matches.items()
            if canonical in display_by_canonical
        }
        exact_hits = len(resolved)
        alias_hits = 0
        for canonical, alias in alias_matches.items():
            if canonical in display_by_canonical and canonical not in resolved:
                resolved[canonical] = alias.canonical_id
                alias_hits += 1

        unresolved = tuple(canonical for canonical in canonical_names if canonical not in resolved)
        recalled: dict[str, tuple[Entity, ...]] = {
            canonical: recalled_candidates.get(canonical, ())
            for canonical in unresolved
        }
        candidate_count = sum(len(items) for items in recalled.values())
        embedded_mentions = 0
        embedding_batches = 0
        ambiguous: dict[str, tuple[Entity, ...]] = {}
        if unresolved and self.embed_cfg:
            unique_candidate_names = tuple(
                dict.fromkeys(
                    candidate.canonical_name
                    for canonical in unresolved
                    for candidate in recalled[canonical]
                )
            )
            texts = [*unresolved, *unique_candidate_names]
            if texts:
                from memforge.retrieval.embeddings import embed_texts

                vectors = []
                for start in range(0, len(texts), self.policy.embedding_batch_size):
                    vectors.extend(
                        await asyncio.to_thread(
                            embed_texts,
                            texts[start : start + self.policy.embedding_batch_size],
                            self.embed_cfg["base_url"],
                            self.embed_cfg["api_key"],
                            self.embed_cfg["model"],
                        )
                    )
                    embedding_batches += 1
                embedded_mentions = len(unresolved)
                vectors_by_text = dict(zip(texts, vectors, strict=True))
                for canonical in unresolved:
                    candidates = tuple(
                        candidate
                        for candidate in recalled[canonical]
                        if _cosine(
                            vectors_by_text[canonical],
                            vectors_by_text[candidate.canonical_name],
                        )
                        >= self.embedding_threshold
                    )
                    if candidates:
                        ambiguous[canonical] = candidates

        structured_llm_calls = 0
        learned_aliases: list[EntityAlias] = []
        if ambiguous and self.structured_llm_client is not None:
            cases = tuple(
                {
                    "mention": mention,
                    "candidates": [
                        {"id": candidate.id, "name": candidate.canonical_name}
                        for candidate in candidates
                    ],
                }
                for mention, candidates in ambiguous.items()
            )
            decisions: dict[str, object] = {}
            context_text = (doc_context or "")[:2000]
            for case_batch in self._adjudication_batches(cases, context=context_text):
                prompt = self._render_adjudication_prompt(case_batch, context=context_text)
                response = await self.structured_llm_client.validate_entity_batch(
                    prompt,
                    max_tokens=max(512, min(4096, len(case_batch) * 256)),
                    model=self.llm_model,
                )
                structured_llm_calls += 1
                batch_decisions: dict[str, object] = {}
                duplicates: set[str] = set()
                for decision in response.decisions:
                    mention = canonicalize_entity_name(decision.mention)
                    if mention in batch_decisions:
                        duplicates.add(mention)
                    batch_decisions[mention] = decision
                expected_mentions = {str(case["mention"]) for case in case_batch}
                actual_mentions = set(batch_decisions)
                if duplicates or actual_mentions != expected_mentions:
                    raise RuntimeError(
                        "entity adjudication coverage invalid: "
                        f"expected_count={len(expected_mentions)}, "
                        f"actual_count={len(actual_mentions)}, "
                        f"missing_count={len(expected_mentions - actual_mentions)}, "
                        f"duplicate_count={len(duplicates)}, "
                        f"unexpected_count={len(actual_mentions - expected_mentions)}"
                    )
                decisions.update(batch_decisions)
            for mention, candidates in ambiguous.items():
                decision = decisions.get(mention)
                assert decision is not None
                candidate_ids = {candidate.id for candidate in candidates}
                matched_id = getattr(decision, "matched_id", None)
                confidence = float(getattr(decision, "confidence", 0.0))
                if matched_id not in candidate_ids or confidence < self.confidence_threshold:
                    continue
                resolved[mention] = int(matched_id)
                learned_aliases.append(
                    EntityAlias(
                        alias=display_by_canonical[mention],
                        alias_normalized=mention,
                        canonical_id=int(matched_id),
                        source="resolver_confirmed",
                        access_context_hash=scope.access_context_hash,
                    )
                )

        new_names = tuple(canonical for canonical in unresolved if canonical not in resolved)
        if new_names:
            created = await self.store.upsert_entities(
                tuple(
                    EntityUpsert(canonical, display_by_canonical[canonical])
                    for canonical in new_names
                )
            )
            missing = set(new_names).difference(created)
            if missing:
                raise RuntimeError(f"entity batch upsert omitted canonical names: {sorted(missing)}")
            resolved.update({canonical: int(created[canonical]) for canonical in new_names})
        if learned_aliases:
            await self.store.insert_aliases(tuple(learned_aliases))

        return self._finish_batch(
            started=started,
            resolved=resolved,
            exact_hits=exact_hits,
            alias_hits=alias_hits,
            embedded_mentions=embedded_mentions,
            ambiguous_mentions=len(ambiguous),
            embedding_batches=embedding_batches,
            structured_llm_calls=structured_llm_calls,
            candidate_count=candidate_count,
            new_entities=len(new_names),
        )

    def _adjudication_batches(
        self,
        cases: tuple[dict[str, object], ...],
        *,
        context: str,
    ) -> tuple[tuple[dict[str, object], ...], ...]:
        batches: list[tuple[dict[str, object], ...]] = []
        current: list[dict[str, object]] = []
        for case in cases:
            candidate = (*current, case)
            candidate_chars = len(self._render_adjudication_prompt(candidate, context=context))
            if current and (
                len(candidate) > self.policy.adjudication_batch_size
                or candidate_chars > self.policy.max_adjudication_prompt_chars
            ):
                batches.append(tuple(current))
                current = [case]
            else:
                current.append(case)
            if len(self._render_adjudication_prompt(current, context=context)) > (
                self.policy.max_adjudication_prompt_chars
            ):
                raise RuntimeError("single entity adjudication case exceeds prompt character limit")
        if current:
            batches.append(tuple(current))
        return tuple(batches)

    @staticmethod
    def _render_adjudication_prompt(
        cases: Sequence[dict[str, object]],
        *,
        context: str,
    ) -> str:
        return _ENTITY_BATCH_PROMPT.format(
            cases_json=json.dumps(cases, ensure_ascii=False, sort_keys=True),
            context=context,
        )

    def _finish_batch(
        self,
        *,
        started: float,
        resolved: Mapping[str, int],
        exact_hits: int = 0,
        alias_hits: int = 0,
        embedded_mentions: int = 0,
        ambiguous_mentions: int = 0,
        embedding_batches: int = 0,
        structured_llm_calls: int = 0,
        candidate_count: int = 0,
        new_entities: int = 0,
    ) -> EntityResolutionBatch:
        metrics = EntityResolutionMetrics(
            unique_mentions=len(resolved),
            exact_hits=exact_hits,
            alias_hits=alias_hits,
            embedded_mentions=embedded_mentions,
            ambiguous_mentions=ambiguous_mentions,
            embedding_batches=embedding_batches,
            structured_llm_calls=structured_llm_calls,
            candidate_count=candidate_count,
            new_entities=new_entities,
            elapsed_ms=max(0, round((perf_counter() - started) * 1000)),
        )
        logger.info(
            "entity_resolution_batch unique=%d exact=%d alias=%d embedded=%d "
            "ambiguous=%d candidates=%d embedding_batches=%d llm_calls=%d new=%d elapsed_ms=%d",
            metrics.unique_mentions,
            metrics.exact_hits,
            metrics.alias_hits,
            metrics.embedded_mentions,
            metrics.ambiguous_mentions,
            metrics.candidate_count,
            metrics.embedding_batches,
            metrics.structured_llm_calls,
            metrics.new_entities,
            metrics.elapsed_ms,
        )
        return EntityResolutionBatch(dict(resolved), metrics)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(left, right, strict=True))
    left_norm = sqrt(sum(float(value) ** 2 for value in left))
    right_norm = sqrt(sum(float(value) ** 2 for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)
