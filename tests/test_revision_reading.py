from __future__ import annotations

import hashlib
import json
from dataclasses import replace

from memforge.pipeline.evidence_fragments import (
    EvidenceFragment,
    build_revision_fragment_index,
    canonical_record_field_ranges,
)
from memforge.pipeline.revision_reading import build_revision_reading_index
from memforge.source_projection import (
    EvidenceCoordinateSpace,
    EvidenceRepresentationProfile,
    SourceObservationRevision,
)
from memforge.source_representation import (
    BINARY_ARTIFACT_PROFILE,
    MARKDOWN_STRUCTURAL_PROFILE,
    PLAIN_TEXT_PROFILE,
)


def _revision(
    content: str,
    profile: EvidenceRepresentationProfile,
) -> SourceObservationRevision:
    return SourceObservationRevision(
        id="obsrev-1",
        observation_id="obs-1",
        semantic_hash=hashlib.sha256(content.encode()).hexdigest(),
        content=content,
        evidence_profile=profile,
    )


def _canonical_profile(schema_name: str) -> EvidenceRepresentationProfile:
    return EvidenceRepresentationProfile(
        name="canonical-record",
        version=1,
        coordinate_space=EvidenceCoordinateSpace.UNICODE_SCALAR,
        schema_name=schema_name,
        schema_version=1,
    )


def _index(revision: SourceObservationRevision):
    fragments = build_revision_fragment_index(revision).fragments
    return fragments, build_revision_reading_index(revision, fragments)


def _fragment(fragments: tuple[EvidenceFragment, ...], text: str) -> EvidenceFragment:
    return next(fragment for fragment in fragments if fragment.presentation_text == text)


def test_section_context_keeps_definition_and_excludes_archive_sibling() -> None:
    content = """# Payroll handbook

Cedar means the active US regular-payroll population.

## Current rule

Cedar requires two approvers.

## Archive

The retired pilot allowed one approver.
"""
    revision = _revision(content, MARKDOWN_STRUCTURAL_PROFILE)
    fragments, index = _index(revision)

    selected = _fragment(fragments, "Cedar requires two approvers.")
    expansion = index.expand((selected,))
    texts = [fragment.presentation_text.strip() for fragment in expansion.fragments]

    assert texts == [
        "# Payroll handbook",
        "Cedar means the active US regular-payroll population.",
        "## Current rule",
        "Cedar requires two approvers.",
    ]
    assert "## Archive" not in texts
    assert "The retired pilot allowed one approver." not in texts


def test_appended_meeting_reads_ancestors_without_historical_sibling_bodies() -> None:
    content = """# Meeting minutes

Shared release terminology applies.

## 2026-09-19

The old meeting discussed Birch.

## 2026-09-20

The release owner approved Cedar.
"""
    revision = _revision(content, MARKDOWN_STRUCTURAL_PROFILE)
    fragments, index = _index(revision)

    expansion = index.expand((_fragment(fragments, "The release owner approved Cedar."),))
    texts = {fragment.presentation_text.strip() for fragment in expansion.fragments}

    assert texts == {
        "# Meeting minutes",
        "Shared release terminology applies.",
        "## 2026-09-20",
        "The release owner approved Cedar.",
    }
    assert "The old meeting discussed Birch." not in texts


def test_single_root_heading_adds_only_one_local_intro_not_the_whole_body() -> None:
    paragraphs = [f"Standalone meeting fact {index}." for index in range(75)]
    content = "# Minutes\n\n" + "\n\n".join(paragraphs)
    revision = _revision(content, MARKDOWN_STRUCTURAL_PROFILE)
    fragments, index = _index(revision)

    expansion = index.expand((_fragment(fragments, paragraphs[-1]),))
    texts = [fragment.presentation_text.strip() for fragment in expansion.fragments]

    assert texts == ["# Minutes", paragraphs[0], paragraphs[-1]]


def test_markdown_unordered_list_reads_lead_in_and_complete_nested_group() -> None:
    content = """# Eligible populations

Apply the following definitions together:

- Cedar
  - US regular payroll
- Birch

Outside the list.
"""
    revision = _revision(content, MARKDOWN_STRUCTURAL_PROFILE)
    fragments, index = _index(revision)
    cedar = next(fragment for fragment in fragments if fragment.presentation_text.startswith("- Cedar"))

    expansion = index.expand((replace(cedar, primary_eligible=False),))
    texts = [fragment.presentation_text.strip() for fragment in expansion.fragments]

    assert texts == [
        "# Eligible populations",
        "Apply the following definitions together:",
        "- Cedar\n  - US regular payroll",
        "- Birch",
    ]
    assert expansion.authority_anchors == (cedar.anchor,)
    assert _fragment(expansion.fragments, "- Cedar\n  - US regular payroll").primary_eligible is False
    assert cedar.reference == _fragment(expansion.fragments, "- Cedar\n  - US regular payroll").reference
    assert _fragment(fragments, "- Birch").anchor in expansion.context_anchors
    assert "Outside the list." not in texts


def test_second_paragraph_definition_is_kept_as_immediate_list_lead_in() -> None:
    content = """# Current policy

This section covers active payroll populations.

Cedar means the US regular-payroll population.

- Cedar requires two approvers.
- Birch requires one approver.

This later note is outside the list.
"""
    revision = _revision(content, MARKDOWN_STRUCTURAL_PROFILE)
    fragments, index = _index(revision)
    selected = _fragment(fragments, "- Cedar requires two approvers.")

    expansion = index.expand((selected,))
    texts = [fragment.presentation_text.strip() for fragment in expansion.fragments]

    assert texts == [
        "# Current policy",
        "This section covers active payroll populations.",
        "Cedar means the US regular-payroll population.",
        "- Cedar requires two approvers.",
        "- Birch requires one approver.",
    ]
    assert "This later note is outside the list." not in texts


def test_ordered_list_keeps_existing_whole_list_atom_and_lead_in() -> None:
    content = """# Release procedure

Run these steps in order:

1. Stop payroll.
2. Update configuration.
3. Start payroll.
"""
    revision = _revision(content, MARKDOWN_STRUCTURAL_PROFILE)
    fragments, index = _index(revision)
    ordered = next(fragment for fragment in fragments if fragment.fragment_type == "markdown-ordered-list")

    expansion = index.expand((ordered,))

    assert [fragment.presentation_text.strip() for fragment in expansion.fragments] == [
        "# Release procedure",
        "Run these steps in order:",
        "1. Stop payroll.\n2. Update configuration.\n3. Start payroll.",
    ]
    assert len([group for group in index.groups if group.kind == "markdown-list"]) == 1
    assert ordered.anchor in expansion.authority_anchors


def test_html_ul_reads_whole_list_and_html_heading_ancestry() -> None:
    content = (
        "<h1>Payroll handbook</h1><p>Definitions are normative.</p>"
        "<h2>Current</h2><p>Read all eligible plans:</p>"
        "<ul><li>Cedar<ul><li>US regular</li></ul></li><li>Birch</li></ul>"
        "<h2>Archive</h2><p>Legacy plan</p>"
    )
    revision = _revision(content, MARKDOWN_STRUCTURAL_PROFILE)
    fragments, index = _index(revision)
    cedar = next(fragment for fragment in fragments if fragment.presentation_text == "Cedar US regular")

    expansion = index.expand((cedar,))
    texts = [fragment.presentation_text for fragment in expansion.fragments]

    assert texts == [
        "Payroll handbook",
        "Definitions are normative.",
        "Current",
        "Read all eligible plans:",
        "Cedar US regular",
        "Birch",
    ]
    assert "Archive" not in texts
    assert "Legacy plan" not in texts
    assert len([group for group in index.groups if group.kind == "html-list"]) == 1


def test_canonical_nested_markdown_uses_exact_field_and_raw_escape_boundaries() -> None:
    content = json.dumps(
        {
            "created": "2026-09-20T10:00:00Z",
            "items": [
                {
                    "field": "description-old",
                    "fromString": "## Same title\n\nOld 中 body.",
                    "toString": "unchanged",
                },
                {
                    "field": "description-new",
                    "fromString": "before",
                    "toString": "## Same title\n\nNew 中 body.",
                },
            ],
        }
    )
    revision = _revision(content, _canonical_profile("jira-changelog"))
    fragments, index = _index(revision)
    selected = _fragment(fragments, "New 中 body.")

    expansion = index.expand((selected,))
    texts = [fragment.presentation_text for fragment in expansion.fragments]
    fields = {
        field.descriptor.json_pointer: field
        for field in canonical_record_field_ranges(revision)
    }
    target = fields["/items/1/toString"]

    assert texts == [
        "2026-09-20T10:00:00Z",
        "description-new",
        "## Same title",
        "New 中 body.",
    ]
    assert "description-old" not in texts
    assert "Old 中 body." not in texts
    assert target.start <= selected.anchor.range_start < selected.anchor.range_end <= target.end
    assert r"\u4e2d" in content[selected.anchor.range_start : selected.anchor.range_end]
    nested_groups = [
        group
        for group in index.groups
        if group.kind == "canonical-markdown-section"
        and group.owner == "/items/1/toString"
    ]
    assert len(nested_groups) == 1


def test_teams_canonical_html_keeps_field_bounded_heading_and_whole_ul() -> None:
    content = json.dumps(
        {
            "content": (
                "<h2>Approval policy</h2><p>Read the eligible plans together.</p>"
                "<ul><li>Cedar</li><li>Birch</li></ul>"
            ),
            "unregistered_neighbor": "<h2>Archive</h2><p>Legacy</p>",
        }
    )
    revision = _revision(content, _canonical_profile("teams-message"))
    fragments, index = _index(revision)

    expansion = index.expand((_fragment(fragments, "Cedar"),))

    assert [fragment.presentation_text for fragment in expansion.fragments] == [
        "Approval policy",
        "Read the eligible plans together.",
        "Cedar",
        "Birch",
    ]
    assert all(group.owner == "/content" for group in index.groups)
    assert {group.kind for group in index.groups} == {
        "canonical-html-section",
        "canonical-html-list",
    }
    assert "Archive" not in {fragment.presentation_text for fragment in fragments}


def test_plain_text_does_not_fabricate_semantic_sections() -> None:
    revision = _revision("First paragraph.\n\nSecond paragraph.", PLAIN_TEXT_PROFILE)
    fragments, index = _index(revision)

    expansion = index.expand((fragments[1],))

    assert index.groups == ()
    assert expansion.fragments == (fragments[1],)
    assert expansion.context_anchors == ()


def test_tables_and_binary_artifacts_are_not_changed_by_reading_groups() -> None:
    markdown = _revision(
        "| Plan | Status |\n| --- | --- |\n| Cedar | Current |",
        MARKDOWN_STRUCTURAL_PROFILE,
    )
    table_fragments, table_index = _index(markdown)

    assert table_index.expand(table_fragments).fragments == table_fragments
    assert table_index.groups == ()

    binary = _revision("", BINARY_ARTIFACT_PROFILE)
    assert build_revision_reading_index(binary, ()).groups == ()
