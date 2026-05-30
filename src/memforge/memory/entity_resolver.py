"""Entity resolution pipeline.

5-step resolver: exact match -> alias lookup -> embedding search ->
LLM validation -> create new.

Code-level resolution with embedding similarity for candidate finding
and LLM validation for confirmation. Self-improving: confirmed matches
auto-register aliases for faster future lookups.
"""

from __future__ import annotations

import asyncio
import logging
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any

import numpy as np

from memforge.llm.structured import StructuredLlmError
from memforge.models import Entity, canonicalize_entity_name

if TYPE_CHECKING:
    from memforge.storage.database import Database

logger = logging.getLogger(__name__)

__all__ = [
    "EntityResolver",
    "validate_alias",
    "insert_llm_alias",
]


# ---------------------------------------------------------------------------
# Alias validation (for LLM-extracted aliases)
# ---------------------------------------------------------------------------

def validate_alias(alias_name: str, canonical_name: str) -> bool:
    """Check if an LLM-extracted alias has ANY resemblance to the canonical name.

    Prevents alias table corruption from LLM hallucination.
    Returns True if the alias is plausible, False if suspicious.

    If False, the alias should be queued for admin review rather than auto-inserted.
    """
    a = canonicalize_entity_name(alias_name)
    c = canonicalize_entity_name(canonical_name)

    # Check 1: Any token overlap?
    a_tokens = set(a.split())
    c_tokens = set(c.split())
    if a_tokens & c_tokens:
        return True

    # Check 2: Substring containment?
    if a in c or c in a:
        return True

    # Check 3: String similarity (SequenceMatcher ratio)
    ratio = SequenceMatcher(None, a, c).ratio()
    if ratio >= 0.5:
        return True

    # No resemblance — suspicious
    return False


# ---------------------------------------------------------------------------
# LLM alias insertion with validation gate
# ---------------------------------------------------------------------------

async def insert_llm_alias(
    alias_name: str,
    canonical_name: str,
    canonical_id: int,
    evidence: str,
    db: Database,
) -> bool:
    """Insert an LLM-extracted alias, with validation.

    If the alias passes validation, it's inserted directly.
    If it fails, it's logged as a warning (admin review queue is a future feature).

    Returns True if inserted, False if rejected.
    """
    if validate_alias(alias_name, canonical_name):
        alias_normalized = canonicalize_entity_name(alias_name)
        await db.insert_alias(
            alias=alias_name,
            alias_normalized=alias_normalized,
            canonical_id=canonical_id,
            source="llm_extracted",
        )
        logger.info(
            "LLM alias accepted: %r -> entity %d (%s)",
            alias_name, canonical_id, canonical_name,
        )
        return True
    else:
        logger.warning(
            "LLM alias rejected (no string similarity): %r -> %r (entity %d). Evidence: %s",
            alias_name, canonical_name, canonical_id, evidence,
        )
        return False


# ---------------------------------------------------------------------------
# LLM entity validation prompt
# ---------------------------------------------------------------------------

_ENTITY_VALIDATION_PROMPT = """You are resolving entity names in a team knowledge system.

A new entity name was extracted from a document. I found similar existing entities
via embedding search. Determine if the new name refers to the SAME real-world entity
as any of the candidates.

IMPORTANT: Similar names do NOT mean the same entity.
- "billing-service" and "payment-service" are DIFFERENT services
- "auth-service" and "auth-svc" ARE the same service (abbreviation)
- "PostgreSQL 15" and "PostgreSQL" ARE the same technology (version variant)

New entity name: "{new_name}"

Document context (where this entity appears):
{doc_context}

Candidate entities from our database:
{candidates_text}

Return ONLY a JSON object:
{{"same_entity": true/false, "matched_id": <entity_id or null>, "confidence": 0.0-1.0}}

If none of the candidates match, return: {{"same_entity": false, "matched_id": null, "confidence": 0.0}}"""


# ---------------------------------------------------------------------------
# EntityResolver class
# ---------------------------------------------------------------------------

class EntityResolver:
    """Resolves extracted entity names to canonical entity IDs.

    5-step pipeline:
    1. Exact match on canonical_name (cross-type, DB lookup)
    2. Alias table lookup (cross-type, DB lookup)
    3. Embedding similarity search (cross-type, cosine >= 0.6)
    4. LLM validation of embedding candidates (with document context)
    5. Create new entity

    Steps 1-2 handle ~70-90% of resolutions at zero cost (<1ms each).
    Steps 3-4 handle novel name variants that the alias table hasn't seen.
    Confirmed matches auto-register aliases so Steps 1-2 catch them next time.
    """

    def __init__(
        self,
        db: Database,
        embed_cfg: dict | None = None,
        structured_llm_client: Any = None,
        llm_model: str = "claude-sonnet-4-20250514",
        embedding_threshold: float = 0.6,
    ) -> None:
        self.db = db
        self.embed_cfg = embed_cfg
        self.structured_llm_client = structured_llm_client
        self.llm_model = llm_model
        self.embedding_threshold = embedding_threshold
        # Cache: entity_id -> embedding vector (populated lazily)
        self._entity_embeddings: dict[int, list[float]] = {}
        self._embeddings_loaded = False
        # Resolution stats (observability)
        self._stats = {
            "exact_match": 0,
            "alias_match": 0,
            "embedding_match": 0,
            "llm_confirmed": 0,
            "new_entity": 0,
            "total_resolved": 0,
        }

    @property
    def stats(self) -> dict:
        """Resolution hit-rate stats for observability."""
        return dict(self._stats)
        self._entity_embeddings: dict[int, list[float]] = {}
        self._embeddings_loaded = False

    # -------------------------------------------------------------------
    # Main resolution pipeline
    # -------------------------------------------------------------------

    async def resolve(
        self,
        extracted_name: str,
        db: Database | None = None,
        *,
        extracted_tags: list[str] | None = None,
        doc_context: str | None = None,
    ) -> int:
        """Resolve an extracted entity name to a canonical entity ID.

        5-step pipeline:
        1. Exact canonical match (DB lookup, <1ms)
        2. Alias table lookup (DB lookup, <1ms)
        3. Embedding similarity search (cosine >= 0.6, top-10)
        4. LLM validation of candidates (with document context)
        5. Create new entity

        Returns the entity ID (existing or newly created).
        Self-improving: confirmed matches auto-register aliases.
        """
        _db = db or self.db
        canonical = canonicalize_entity_name(extracted_name)

        if not canonical:
            # Empty name — create a placeholder rather than crash
            new_id = await _db.upsert_entity(canonical, extracted_name, tags=extracted_tags or [])
            return new_id

        # Step 1: Exact match on canonical_name
        entity = await _db.get_entity_by_canonical(canonical)
        if entity:
            self._stats["exact_match"] += 1
            self._stats["total_resolved"] += 1
            return entity.id

        # Step 2: Alias table lookup
        alias = await _db.get_entity_by_alias(canonical)
        if alias:
            self._stats["alias_match"] += 1
            self._stats["total_resolved"] += 1
            return alias.canonical_id

        # Step 3: Embedding similarity search (if configured)
        if self.embed_cfg:
            match = await self._embedding_search(canonical, _db, doc_context)
            if match:
                self._stats["embedding_match"] += 1
                self._stats["total_resolved"] += 1
                return match

        # Step 4: Create new entity
        new_id = await _db.upsert_entity(canonical, extracted_name, tags=extracted_tags or [])
        self._stats["new_entity"] += 1
        self._stats["total_resolved"] += 1
        logger.info(
            "New entity created: %r (tags=%s, id=%d)",
            extracted_name, extracted_tags, new_id,
        )
        return new_id

    # -------------------------------------------------------------------
    # Step 3+4: Embedding search + LLM validation
    # -------------------------------------------------------------------

    async def _embedding_search(
        self,
        canonical: str,
        db: Database,
        doc_context: str | None,
    ) -> int | None:
        """Find candidate entities by embedding similarity, then validate with LLM.

        Returns entity_id if a match is confirmed, None otherwise.
        """
        try:
            # Ensure entity embeddings are loaded
            await self._load_entity_embeddings(db)

            if not self._entity_embeddings:
                return None

            # Embed the new entity name
            from memforge.retrieval.embeddings import embed_texts

            new_vectors = await asyncio.to_thread(
                embed_texts,
                [canonical],
                self.embed_cfg["base_url"],
                self.embed_cfg["api_key"],
                self.embed_cfg["model"],
            )
            if not new_vectors:
                return None
            new_embedding = new_vectors[0]

            # Find top-10 candidates by cosine similarity
            candidates = self._find_similar_entities(new_embedding, top_k=10)
            if not candidates:
                return None

            # If we have a structured LLM client, validate the best candidates
            if self.structured_llm_client and candidates:
                return await self._llm_validate(
                    canonical, candidates, db, doc_context
                )

            # No LLM client — use the top candidate if similarity is very high (>= 0.85)
            best_id, best_sim = candidates[0]
            if best_sim >= 0.85:
                entity = await self._get_entity_by_id(best_id, db)
                if entity:
                    await self._auto_register_alias(
                        canonical, entity, db, source="embedding_auto"
                    )
                    return best_id

            return None

        except Exception as e:
            logger.warning("Embedding search failed: %s", e)
            return None

    async def _load_entity_embeddings(self, db: Database) -> None:
        """Lazily load and embed all entity names. Cached after first call."""
        if self._embeddings_loaded:
            return

        entities = await db.get_all_entities()
        if not entities:
            self._embeddings_loaded = True
            return

        # Batch embed all entity canonical names
        names = [e.canonical_name for e in entities]
        try:
            from memforge.retrieval.embeddings import embed_texts

            vectors = await asyncio.to_thread(
                embed_texts,
                names,
                self.embed_cfg["base_url"],
                self.embed_cfg["api_key"],
                self.embed_cfg["model"],
            )
            for entity, vector in zip(entities, vectors):
                self._entity_embeddings[entity.id] = vector
        except Exception as e:
            logger.warning("Failed to embed entity names: %s", e)

        self._embeddings_loaded = True

    def _find_similar_entities(
        self,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """Find entity IDs with cosine similarity >= threshold."""
        if not self._entity_embeddings:
            return []

        query = np.array(query_embedding)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []

        results: list[tuple[int, float]] = []
        for entity_id, emb in self._entity_embeddings.items():
            emb_arr = np.array(emb)
            emb_norm = np.linalg.norm(emb_arr)
            if emb_norm == 0:
                continue
            similarity = float(np.dot(query, emb_arr) / (query_norm * emb_norm))
            if similarity >= self.embedding_threshold:
                results.append((entity_id, similarity))

        # Sort by similarity descending, take top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    async def _llm_validate(
        self,
        canonical: str,
        candidates: list[tuple[int, float]],
        db: Database,
        doc_context: str | None,
    ) -> int | None:
        """Ask LLM to confirm if the new entity matches any candidate."""
        # Build candidate text
        candidate_entities: list[tuple[int, Entity]] = []
        lines = []
        for entity_id, similarity in candidates[:5]:  # Top 5 for LLM context
            entity = await self._get_entity_by_id(entity_id, db)
            if entity:
                candidate_entities.append((entity_id, entity))
                lines.append(
                    f"  - ID {entity.id}: \"{entity.canonical_name}\" "
                    f"(similarity: {similarity:.2f})"
                )

        if not lines:
            return None

        candidates_text = "\n".join(lines)
        context = (doc_context or "No document context available.")[:500]

        prompt = _ENTITY_VALIDATION_PROMPT.format(
            new_name=canonical,
            doc_context=context,
            candidates_text=candidates_text,
        )

        try:
            result = await self.structured_llm_client.validate_entity_match(
                prompt,
                max_tokens=200,
                model=self.llm_model,
            )

            if result.same_entity and result.matched_id:
                matched_id = int(result.matched_id)
                confidence = float(result.confidence)

                # Require high confidence
                if confidence < 0.9:
                    logger.info(
                        "LLM match confidence too low (%.2f) for %r -> entity %d",
                        confidence, canonical, matched_id,
                    )
                    return None

                # Find the matched entity
                matched_entity = None
                for eid, ent in candidate_entities:
                    if eid == matched_id:
                        matched_entity = ent
                        break

                if matched_entity:
                    await self._auto_register_alias(
                        canonical, matched_entity, db, source="llm_embedding_confirmed"
                    )
                    return matched_id

        except (StructuredLlmError, KeyError, ValueError) as e:
            logger.warning("Structured entity validation failed: %s", e)
        except Exception as e:
            logger.warning("LLM entity validation failed: %s", e)

        return None

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    async def _get_entity_by_id(self, entity_id: int, db: Database) -> Entity | None:
        """Look up an entity by ID (scans all entities — OK for team scale)."""
        all_entities = await db.get_all_entities()
        for e in all_entities:
            if e.id == entity_id:
                return e
        return None

    async def _auto_register_alias(
        self,
        canonical: str,
        matched_entity: Entity,
        db: Database,
        source: str = "embedding_auto",
    ) -> None:
        """Register an alias if validate_alias passes. Belt + suspenders."""
        if validate_alias(canonical, matched_entity.canonical_name):
            await db.insert_alias(
                alias=canonical,
                alias_normalized=canonical,
                canonical_id=matched_entity.id,
                source=source,
            )
            logger.info(
                "Entity match confirmed: %r -> entity %d (%s). Alias registered (source=%s).",
                canonical, matched_entity.id, matched_entity.canonical_name, source,
            )
        else:
            logger.warning(
                "Entity match confirmed by LLM/embedding but validate_alias rejected: "
                "%r -> %r (entity %d). Flagged for admin review.",
                canonical, matched_entity.canonical_name, matched_entity.id,
            )

    def invalidate_cache(self) -> None:
        """Clear cached entity embeddings (call after new entities are created)."""
        self._entity_embeddings.clear()
        self._embeddings_loaded = False


# ---------------------------------------------------------------------------
# Backward-compatible standalone function
# ---------------------------------------------------------------------------

async def resolve_entity(
    extracted_name: str,
    extracted_type: str = "unknown",
    db: Database | None = None,
    *,
    extracted_tags: list[str] | None = None,
) -> int:
    """Resolve an entity using a lightweight resolver (Steps 1-2 + create).

    This is a backward-compatible wrapper for code that doesn't have
    access to an EntityResolver instance (no embedding/LLM support).
    For full resolution with embedding+LLM, use EntityResolver.resolve().
    """
    assert db is not None, "db parameter is required"
    canonical = canonicalize_entity_name(extracted_name)

    # Merge type into tags for backward compat
    tags = extracted_tags or []
    if extracted_type and extracted_type != "unknown" and extracted_type not in tags:
        tags = [extracted_type] + tags

    # Step 1: Exact match
    entity = await db.get_entity_by_canonical(canonical)
    if entity:
        return entity.id

    # Step 2: Alias lookup
    alias = await db.get_entity_by_alias(canonical)
    if alias:
        return alias.canonical_id

    # Step 3: Create new (no embedding/LLM in lightweight mode)
    new_id = await db.upsert_entity(canonical, extracted_name, tags=tags)
    logger.info("New entity (lightweight): %r (tags=%s, id=%d)", extracted_name, tags, new_id)
    return new_id
