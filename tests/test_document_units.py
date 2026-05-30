from __future__ import annotations

from memforge.pipeline.document_units import (
    ExtractionContextPacker,
    UnitizationPolicy,
    unitize_markdown,
)


def test_unitizer_keeps_small_headed_document_as_one_coherent_unit():
    markdown = "\n\n".join(
        [
            "# Process Tracking",
            "Document intro.",
            "## Tracking",
            "Tracking preamble.",
            "### Explicit API Call",
            "Explicit calls use UnifiedContextApi.",
            "### OData Gateway",
            "OData gateway calls use ProcessEvent.",
            "## Retry Options",
            "Retry options preserve idempotency.",
        ]
    )

    units = unitize_markdown(
        markdown,
        policy=UnitizationPolicy(max_unit_input_tokens=200),
    )

    assert len(units) == 1
    assert units[0].heading_path == ("Process Tracking",)
    assert units[0].unit_kind == "content"
    assert units[0].split_reason == "whole_document_fits_budget"
    assert "### Explicit API Call" in units[0].unit_markdown
    assert "## Retry Options" in units[0].unit_markdown


def test_unitizer_keeps_small_jira_like_issue_as_one_unit():
    markdown = "\n\n".join(
        [
            "# [Story] PAY-123: Cutoff flow",
            "**Status**: In Progress | **Priority**: Medium | **Assignee**: Alice",
            "## Description",
            "Cutoff V2 records the picked period lifecycle before expansion.",
            "## Comments",
            "**Alice** (2026-05-20): Keep trigger expansion synchronous.",
        ]
    )

    units = unitize_markdown(markdown, policy=UnitizationPolicy(max_unit_input_tokens=200))

    assert len(units) == 1
    assert units[0].heading_path == ("[Story] PAY-123: Cutoff flow",)
    assert "## Description" in units[0].unit_markdown
    assert "## Comments" in units[0].unit_markdown


def test_unitizer_keeps_small_agent_session_summary_as_one_unit():
    markdown = "\n\n".join(
        [
            "# Generated Agent Session Summary",
            "- Client: codex",
            "- Trigger: Stop",
            "## Submitted Summary",
            "### Durable Findings",
            "- Agent session documents sync through the normal source pipeline.",
        ]
    )

    units = unitize_markdown(markdown, policy=UnitizationPolicy(max_unit_input_tokens=200))

    assert len(units) == 1
    assert units[0].heading_path == ("Generated Agent Session Summary",)
    assert "## Submitted Summary" in units[0].unit_markdown
    assert "### Durable Findings" in units[0].unit_markdown


def test_unitizer_recurses_only_into_oversized_section_subtrees():
    markdown = "\n\n".join(
        [
            "# Architecture",
            "## Small Section",
            "Small section stays intact.",
            "## Large Section",
            "### Part A",
            " ".join(["alpha"] * 24),
            "### Part B",
            " ".join(["beta"] * 24),
        ]
    )

    units = unitize_markdown(
        markdown,
        policy=UnitizationPolicy(max_unit_input_tokens=35),
    )

    assert [unit.heading_path for unit in units] == [
        ("Architecture", "Small Section"),
        ("Architecture", "Large Section", "Part A"),
        ("Architecture", "Large Section", "Part B"),
    ]
    assert [unit.split_reason for unit in units] == [
        "fits_section_subtree",
        "fits_section_subtree",
        "fits_section_subtree",
    ]


def test_unitizer_overflow_splits_headingless_leaf_by_safe_blocks():
    markdown = "\n\n".join(
        [
            "# Long Note",
            " ".join(["alpha"] * 24),
            " ".join(["beta"] * 24),
            " ".join(["gamma"] * 24),
        ]
    )

    units = unitize_markdown(
        markdown,
        policy=UnitizationPolicy(max_unit_input_tokens=30),
    )

    assert len(units) == 3
    assert [unit.split_reason for unit in units] == ["overflow", "overflow", "overflow"]
    assert all(unit.heading_path == ("Long Note",) for unit in units)


def test_unitizer_overflow_splits_no_whitespace_block_under_token_budget():
    markdown = "# Long Identifier\n\n" + ("x" * 5000)

    units = unitize_markdown(
        markdown,
        policy=UnitizationPolicy(max_unit_input_tokens=100),
    )

    assert len(units) > 1
    assert [unit.split_reason for unit in units] == ["overflow"] * len(units)
    assert all(unit.heading_path == ("Long Identifier",) for unit in units)
    assert all(len(unit.unit_markdown) <= 400 for unit in units)


def test_unitizer_ignores_markdown_headings_inside_code_fences():
    markdown = "\n".join(
        [
            "# Guide",
            "",
            "## API",
            "",
            "```markdown",
            "# Not a heading",
            "## Also not a heading",
            "```",
            "",
            "Real API content.",
        ]
    )

    units = unitize_markdown(markdown, policy=UnitizationPolicy(max_unit_input_tokens=200))

    assert len(units) == 1
    assert units[0].heading_path == ("Guide",)
    assert "# Not a heading" in units[0].unit_markdown


def test_context_packer_uses_outline_and_glossary_but_not_neighbor_headings():
    markdown = "\n\n".join(
        [
            "# Process Tracking",
            "## Terminology",
            "On-Demand (OD) means payroll triggered outside the regular schedule.",
            "## Tracking",
            "OD tracking uses UnifiedContextApi.",
        ]
    )
    units = unitize_markdown(markdown, policy=UnitizationPolicy(max_unit_input_tokens=18))
    tracking = next(unit for unit in units if unit.heading_path == ("Process Tracking", "Tracking"))

    context = ExtractionContextPacker().pack(
        document_title="Process Tracking",
        document_url="https://example.test/process-tracking",
        source_type="github_pages",
        unit=tracking,
        all_units=units,
        entities=["On-Demand", "UnifiedContextApi"],
    )

    assert "Terminology" in context.document_outline
    assert "On-Demand (OD)" in context.glossary_appendix
    assert context.previous_heading is None
    assert context.next_heading is None
    assert context.unit.unit_markdown == tracking.unit_markdown
