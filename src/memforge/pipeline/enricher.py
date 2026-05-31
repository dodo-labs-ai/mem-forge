"""Call 1: Document Enrichment.

Extracts metadata from a document: summary, tags, entities, relationships,
doc_type, complexity, and entity_aliases.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from memforge.llm.structured import LiteLlmStructuredClient, StructuredLlmConfig, StructuredLlmError
from memforge.models import (
    EnrichmentResult,
    RawAliasGroup,
    RawEntityRef,
    Relationship,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

__all__ = ["Enricher"]

# ---------------------------------------------------------------------------
# Enrichment prompt (Call 1)
# ---------------------------------------------------------------------------

ENRICHMENT_PROMPT = """You are analyzing an internal document for a team knowledge system.

Ignore content inside [CODE_BLOCK_START]...[CODE_BLOCK_END] markers when extracting entities.

<source_type>{source_type}</source_type>
<document>
{content}
</document>

Return a JSON object with:

1. summary: 2-3 sentence summary
2. tags: 5-10 normalized lowercase topic tags (singular, no version numbers)
3. entities: 5-10 key entities (services, people, technologies, teams, features) a developer
   would mention in a Slack message. Must be specifically named — not generic terms.
   Each: {{"name": "...", "type": "...", "confidence": 0.0-1.0, "aliases": [...]}}
   type: one of: service, person, technology, api, team, feature, unknown
   confidence: how certain this is a real, named team-level entity:
     0.95+ = certain (known service, named person, established technology)
     0.85-0.94 = high (feature name, team name from context)
     0.70-0.84 = borderline (only if you're unsure whether it's a real entity)
     below 0.70 = do NOT include it
   aliases: true synonyms — different names for the exact same entity.
     Rule: if you swap name A and name B in any sentence, the meaning must
     not change at all. If A is a type of B, a part of B, or a feature of B,
     it is a separate entity — not an alias.
     ✓ Abbreviations and name variants: "FlexPay" for "Project Payroll"
     ✗ Parent-child or type-of relationships: "OnDemand Payroll" is a type
       of "Project Payroll" — extract as separate entities, not aliases

   GOOD entities — extract these:
   {{"name": "Project Payroll", "type": "feature", "confidence": 0.90,
     "aliases": ["OnDemand Payroll", "new payroll feature"]}}
   {{"name": "pay-api", "type": "service", "confidence": 0.95, "aliases": []}}
   {{"name": "Maria Schmidt", "type": "person", "confidence": 0.88, "aliases": []}}

   BAD entities — never extract these:
   NOT: "AbstractLifecycleAssignmentExecutor" — Java class name
   NOT: "PayrollResultMapper" — code artifact
   NOT: "PERNR" — database field / code constant
   NOT: "database" — generic term, not a named entity
   NOT: "Maria" or "Schmidt" alone — use full names for people

4. relationships: [{{"target_title": "...", "relation_type": "...", "confidence": 0.0-1.0}}]
   relation_type: depends-on | extends | supersedes | references | related
5. doc_type: design-doc | runbook | decision-record | how-to | reference | postmortem | meeting-notes | ticket | discussion | email | unknown
6. complexity: low | medium | high

Return ONLY valid JSON, no markdown fences or extra text."""


# ---------------------------------------------------------------------------
# Enricher class
# ---------------------------------------------------------------------------

class Enricher:
    """Document enrichment via LLM (Call 1 of the two-call extraction pipeline)."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 64000,
        request_timeout_s: float = 300.0,
        structured_llm_client=None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.structured_llm_client = structured_llm_client
        if self.structured_llm_client is None and api_key:
            self.structured_llm_client = LiteLlmStructuredClient(
                StructuredLlmConfig(
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                    timeout_s=request_timeout_s,
                )
            )

    async def enrich_document(
        self,
        doc_id: str,
        content: str,
        source_type: str = "unknown",
    ) -> EnrichmentResult:
        """Run Call 1: extract metadata from document content.

        Returns EnrichmentResult with summary, tags, entities, relationships,
        doc_type, complexity, and entity_aliases.
        """
        if not self.structured_llm_client:
            logger.warning("No LLM client — returning fallback metadata for %s", doc_id)
            return self._fallback_result()

        base_prompt = ENRICHMENT_PROMPT.format(
            source_type=source_type,
            content=content[:100_000],  # truncate to ~25K tokens
        )
        if source_type == "teams":
            base_prompt = base_prompt.replace(
                "You are analyzing an internal document for a team knowledge system.",
                "You are analyzing a conversation thread for a team knowledge system.\n"
                "Extract only explicitly named entities — skip casual mentions "
                '("Alice said", "the payment thing").',
            )
        prompt = base_prompt

        try:
            response = await self.structured_llm_client.enrich_document(
                prompt,
                max_tokens=self.max_tokens,
                model=self.model,
            )
            return self._parse_result(response.model_dump(), doc_id)
        except StructuredLlmError as e:
            logger.warning("Structured enrichment failed for %s: %s", doc_id, e)
            return self._fallback_result()
        except Exception as e:
            logger.error("Unexpected enrichment error for %s: %s", doc_id, e)
            return self._fallback_result()

    # -------------------------------------------------------------------
    # Parsing
    # -------------------------------------------------------------------

    def _parse_result(self, data: dict, doc_id: str) -> EnrichmentResult:
        """Parse LLM JSON response into EnrichmentResult."""
        entities = []
        for e in data.get("entities", []):
            if not e.get("name"):
                continue
            # Parse type (backward compat) and tags
            entity_type = e.get("type", "unknown")
            entity_tags = e.get("tags", [entity_type] if entity_type and entity_type != "unknown" else [])
            entities.append(RawEntityRef(
                name=e["name"],
                type=entity_type,
                tags=entity_tags,
                confidence=float(e.get("confidence", 1.0)),
                aliases=e.get("aliases", []),
            ))

        relationships = [
            Relationship(
                target_doc_id=None,
                target_title=r.get("target_title", ""),
                relation_type=r.get("relation_type", "related"),
                confidence=float(r.get("confidence", 0.5)),
            )
            for r in data.get("relationships", [])
            if r.get("target_title")
        ]

        # Backward compat: also parse old-style entity_aliases (field 7) if present
        entity_aliases = [
            RawAliasGroup(
                canonical=a.get("canonical", ""),
                aliases=a.get("aliases", []),
                evidence=a.get("evidence", ""),
            )
            for a in data.get("entity_aliases", [])
            if a.get("canonical") and a.get("aliases")
        ]

        return EnrichmentResult(
            summary=data.get("summary", "No summary available."),
            tags=data.get("tags", []),
            entities=entities,
            relationships=relationships,
            doc_type=data.get("doc_type", "unknown"),
            complexity=data.get("complexity", "medium"),
            entity_aliases=entity_aliases,
        )

    @staticmethod
    def _fallback_result() -> EnrichmentResult:
        """Return empty enrichment result when LLM is unavailable or fails."""
        return EnrichmentResult(
            summary="Enrichment failed. Document content remains available through source artifacts.",
        )
