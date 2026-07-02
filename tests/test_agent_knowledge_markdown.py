from __future__ import annotations

from typing import Any

import pytest

from memforge.agent_knowledge_markdown import (
    render_agent_concept_markdown,
    render_agent_concept_markdown_with_patch,
)


class FakeMarkdownDb:
    async def list_agent_claims(self, concept_id: str) -> list[dict[str, Any]]:
        if concept_id != "concept-cli":
            raise AssertionError(f"unexpected concept_id: {concept_id}")
        return [
            {"id": "claim-old", "claim_text": "Use claude-code."},
            {"id": "claim-other", "claim_text": "Keep PRs focused."},
        ]

    async def list_agent_claim_citations(self, claim_id: str) -> list[dict[str, Any]]:
        return {
            "claim-old": [
                {"citation_url": "memory://old"},
                {"citation_url": "memory://old"},
                {"citation_url": "memory://shared"},
            ],
            "claim-other": [{"citation_url": "memory://other"}, {"citation_url": "memory://shared"}],
        }.get(claim_id, [])


class EmptyMarkdownDb:
    async def list_agent_claims(self, concept_id: str) -> list[dict[str, Any]]:
        if concept_id != "concept-cli":
            raise AssertionError(f"unexpected concept_id: {concept_id}")
        return []

    async def list_agent_claim_citations(self, claim_id: str) -> list[dict[str, Any]]:
        return []


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
    assert "## Claim" not in markdown
    assert markdown.count("- memory://old") == 1
    assert markdown.count("- memory://shared") == 1
    assert "- memory://new" in markdown
    assert "- memory://other" in markdown


@pytest.mark.asyncio
async def test_render_agent_concept_markdown_with_patch_emits_marker_for_each_claim():
    markdown = await render_agent_concept_markdown_with_patch(
        FakeMarkdownDb(),
        {
            "id": "concept-cli",
            "title": "Claude CLI invocation convention",
            "concept_type": "procedure",
            "repo_identifier": "github.com/shno-labs/mem-forge",
        },
        claim_id="claim-other",
        claim_text="Keep pull requests focused.",
        citations=[],
    )

    assert markdown.count("mf:claim") == 2
    assert 'id="claim-old"' in markdown
    assert 'id="claim-other"' in markdown
    assert markdown.index('id="claim-old"') < markdown.index("Use claude-code.")
    assert markdown.index('id="claim-other"') < markdown.index("Keep pull requests focused.")


@pytest.mark.asyncio
async def test_render_agent_concept_markdown_with_patch_emits_marker_for_new_claim():
    markdown = await render_agent_concept_markdown_with_patch(
        EmptyMarkdownDb(),
        {
            "id": "concept-cli",
            "title": "Claude CLI invocation convention",
            "concept_type": "procedure",
            "repo_identifier": "github.com/shno-labs/mem-forge",
        },
        claim_id="claim-new",
        claim_text="Use claude.",
        citations=[],
    )

    assert markdown.count("mf:claim") == 1
    assert 'id="claim-new"' in markdown
    assert markdown.index('id="claim-new"') < markdown.index("Use claude.")


def test_render_agent_concept_markdown_preserves_single_claim_marker_shape():
    markdown = render_agent_concept_markdown(
        title="Claude CLI invocation convention",
        concept_type="procedure",
        repo_identifier="github.com/shno-labs/mem-forge",
        claim_id="claim-old",
        claim_text="Use claude.",
        citations=["memory://old", "memory://old"],
    )

    assert markdown.count("mf:claim") == 1
    assert 'id="claim-old"' in markdown
    assert "## Claim" not in markdown
    assert markdown.index('id="claim-old"') < markdown.index("Use claude.")
    assert markdown.count("- memory://old") == 1
