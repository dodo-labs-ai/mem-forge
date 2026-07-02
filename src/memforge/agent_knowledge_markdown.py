"""Rendering helpers for private agent knowledge concept documents."""

from __future__ import annotations

import json
from typing import Any, Protocol

from memforge.models import Visibility


class AgentConceptMarkdownDatabase(Protocol):
    async def list_agent_claims(self, concept_id: str) -> list[dict[str, Any]]: ...

    async def list_agent_claim_citations(self, claim_id: str) -> list[dict[str, Any]]: ...


async def render_agent_concept_markdown_with_patch(
    db: AgentConceptMarkdownDatabase,
    concept: dict[str, Any],
    *,
    claim_id: str,
    claim_text: str,
    citations: list[str],
) -> str:
    """Render a concept markdown body with one claim added or replaced."""

    claims = await db.list_agent_claims(concept["id"])
    citations_by_claim = {claim["id"]: await db.list_agent_claim_citations(claim["id"]) for claim in claims}
    patched_claims = []
    claim_seen = False
    for claim in claims:
        if claim["id"] == claim_id:
            patched_claims.append({**claim, "claim_text": claim_text})
            claim_seen = True
        else:
            patched_claims.append(claim)
    if not claim_seen:
        patched_claims.append({"id": claim_id, "claim_text": claim_text})

    merged_citations: dict[str, list[str]] = {}
    for claim in patched_claims:
        existing = [
            citation["citation_url"]
            for citation in citations_by_claim.get(claim["id"], [])
            if citation["citation_url"].strip()
        ]
        if claim["id"] == claim_id:
            existing.extend(citation.strip() for citation in citations if citation.strip())
        merged_citations[claim["id"]] = list(dict.fromkeys(existing))

    return render_agent_concept_markdown(
        title=concept["title"],
        concept_type=concept["concept_type"],
        repo_identifier=concept.get("repo_identifier"),
        claim_id=patched_claims[0]["id"] if patched_claims else claim_id,
        claim_text="\n\n".join(claim["claim_text"] for claim in patched_claims),
        citations=[citation for claim in patched_claims for citation in merged_citations[claim["id"]]],
    )


def render_agent_concept_markdown(
    *,
    title: str,
    concept_type: str,
    repo_identifier: str | None,
    claim_id: str,
    claim_text: str,
    citations: list[str],
) -> str:
    frontmatter = {
        "type": concept_type,
        "title": title,
        "visibility": Visibility.PRIVATE.value,
        "repo_identifier": repo_identifier,
    }
    citation_lines = "\n".join(f"- {citation}" for citation in citations if citation.strip())
    return (
        "---\n"
        f"{json.dumps(frontmatter, indent=2, sort_keys=True)}\n"
        "---\n\n"
        f"# {title}\n\n"
        "<!--\n"
        "mf:claim\n"
        f'id="{claim_id}"\n'
        "-->\n"
        f"{claim_text.strip()}\n\n"
        "# Citations\n\n"
        f"{citation_lines or '- none'}\n"
    )
