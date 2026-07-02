from __future__ import annotations

from typing import Any

import pytest

from memforge.agent_knowledge_markdown import render_agent_concept_markdown_with_patch


class FakeMarkdownDb:
    async def list_agent_claims(self, concept_id: str) -> list[dict[str, Any]]:
        assert concept_id == "concept-cli"
        return [
            {"id": "claim-old", "claim_text": "Use claude-code."},
            {"id": "claim-other", "claim_text": "Keep PRs focused."},
        ]

    async def list_agent_claim_citations(self, claim_id: str) -> list[dict[str, Any]]:
        return {
            "claim-old": [{"citation_url": "memory://old"}, {"citation_url": "memory://old"}],
            "claim-other": [{"citation_url": "memory://other"}],
        }.get(claim_id, [])


@pytest.mark.asyncio
async def test_render_agent_concept_markdown_with_patch_preserves_claims_and_dedupes_citations():
    markdown = await render_agent_concept_markdown_with_patch(
        FakeMarkdownDb(),
        {
            "id": "concept-cli",
            "title": "Claude CLI invocation convention",
            "concept_type": "procedure",
            "repo_identifier": "github.com/shno-labs/mem-forge",
        },
        claim_id="claim-old",
        claim_text="Use claude.",
        citations=["memory://old", "memory://new"],
    )

    assert "Use claude." in markdown
    assert "Use claude-code." not in markdown
    assert "Keep PRs focused." in markdown
    assert markdown.count("- memory://old") == 1
    assert "- memory://new" in markdown
    assert "- memory://other" in markdown
