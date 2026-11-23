"""reStructuredText section titles in text files that are not declared Markdown."""

from __future__ import annotations

from dataclasses import replace

import pytest

from memforge.memory.evidence import EvidenceRole
from memforge.pipeline.evidence_fragments import (
    _markdown_parser,
    build_revision_fragment_index,
    compile_fragments,
    revision_changed_structural_ranges,
)
from memforge.pipeline.revision_reading import build_revision_reading_index
from memforge.source_adapters.contracts import (
    MARKDOWN_RST_TITLES_PROFILE,
    MARKDOWN_STRUCTURAL_PROFILE,
    file_evidence_profile,
)
from memforge.source_representation import (
    declared_representation_profiles,
    representation_profile_for_observation_contract,
)
from tests.test_evidence_fragments import _authority, _revision
from tests.test_source_revision_time import _item, _project

GUIDE = """\
==============
Release guide
==============

Prerequisites
=============

A clean virtual environment.

Testing other databases
~~~~~~~~~~~~~~~~~~~~~~~

Run the tests against PostgreSQL.

Selenium
--------

Pass ``--selenium`` to run browser tests.
"""


def _blocks(text: str, *, rst_titles: bool = True) -> list[tuple[str, str, str]]:
    tokens = _markdown_parser(rst_titles=rst_titles).parse(text)
    return [
        (token.type, token.tag, tokens[index + 1].content if token.type == "heading_open" else "")
        for index, token in enumerate(tokens)
        if token.type in {"heading_open", "paragraph_open", "fence", "code_block", "hr"}
    ]


def test_a_tilde_underlined_title_is_a_heading_and_the_text_under_it_stays_prose() -> None:
    assert _blocks(GUIDE) == [
        ("heading_open", "h1", "Release guide"),
        ("heading_open", "h2", "Prerequisites"),
        ("paragraph_open", "p", ""),
        ("heading_open", "h3", "Testing other databases"),
        ("paragraph_open", "p", ""),
        ("heading_open", "h4", "Selenium"),
        ("paragraph_open", "p", ""),
    ]


def test_markdown_parsing_still_reads_a_tilde_line_as_a_code_fence() -> None:
    blocks = _blocks(GUIDE, rst_titles=False)

    assert ("fence", "code", "") in blocks
    assert ("heading_open", "h4", "Selenium") not in blocks


def test_title_levels_follow_the_order_in_which_adornment_styles_first_appear() -> None:
    text = "Usage\n-----\n\nText.\n\nOptions\n~~~~~~~\n\nText.\n\nInstall\n-------\n\nText.\n"

    assert [block for block in _blocks(text) if block[0] == "heading_open"] == [
        ("heading_open", "h1", "Usage"),
        ("heading_open", "h2", "Options"),
        ("heading_open", "h1", "Install"),
    ]


@pytest.mark.parametrize(
    ("text", "title"),
    [
        ("Intro text.\nNext title\n^^^^^^^^^^\n\nBody.\n", "Next title"),
        ("A long section title\n~~~~\n\nBody.\n", "A long section title"),
        ("Title\n''''''\n\nBody.\n", "Title"),
        ("1. Introduction\n~~~~~~~~~~~~~~~\n\nRun the first step.\n", "1. Introduction"),
        ("2) Setup\n========\n\nBody.\n", "2) Setup"),
    ],
    ids=["ends-a-paragraph", "short-underline-of-four", "quote-underline", "numbered-tilde", "numbered-equals"],
)
def test_an_underline_marks_a_title(text: str, title: str) -> None:
    assert ("heading_open", "h1", title) in _blocks(text)


@pytest.mark.parametrize(
    "text",
    [
        "Text.\n\n----------\n\nMore text.\n",
        "    Indented code\n    ~~~~~~~~~~~~~\n",
        "~~~~~~\nTitle\n======\n",
        "A long section title\n~~~\nbody\n~~~\n",
        "- Item title\n~~~~~~~~~~~~\n",
        "> Quote title\n~~~~~~~~~~~~~\n",
        "- item\n\n  Para line\n~~~~~~~~~\nrest\n",
        "```\nls\n```\n",
        "~~~~~~~~\nnpm install\n~~~~~~~~\n",
    ],
    ids=[
        "transition", "indented", "overline-differs-from-underline", "underline-too-short",
        "list-item", "block-quote", "underline-outside-the-list-item", "one-line-backtick-fence",
        "one-line-tilde-fence",
    ],
)
def test_a_line_that_is_not_a_title_adornment_is_left_to_markdown(text: str) -> None:
    assert not any(block[0] == "heading_open" for block in _blocks(text))


def test_a_list_keeps_its_items_when_a_dash_line_follows_it() -> None:
    text = "Steps\n=====\n\n* Build the image\n* Push the image\n----------------\n"

    assert _blocks(text) == [("heading_open", "h1", "Steps"), ("paragraph_open", "p", ""),
                             ("paragraph_open", "p", ""), ("hr", "hr", "")]


def test_a_markdown_code_fence_in_a_text_file_stays_one_closed_code_block() -> None:
    text = "Install it.\n\nOr:\n```\npip install foo\n```\n\nThen configure the service.\n\nRestart the worker.\n"

    assert _blocks(text) == [
        ("paragraph_open", "p", ""), ("paragraph_open", "p", ""), ("fence", "code", ""),
        ("paragraph_open", "p", ""), ("paragraph_open", "p", ""),
    ]


def test_rst_title_fragments_cover_the_title_and_keep_the_prose_selectable() -> None:
    revision = _revision(GUIDE, MARKDOWN_RST_TITLES_PROFILE)

    catalog = compile_fragments(revision, (_authority(revision, EvidenceRole.PRIMARY, EvidenceRole.REQUIRED),))

    assert catalog.usable is True
    assert [fragment.fragment_type for fragment in catalog.fragments] == [
        "markdown-heading", "markdown-heading", "markdown-paragraph",
        "markdown-heading", "markdown-paragraph", "markdown-heading", "markdown-paragraph",
    ]
    selenium = catalog.fragments[5]
    assert GUIDE[selenium.anchor.range_start : selenium.anchor.range_end] == "Selenium\n--------"


def test_renaming_an_rst_title_changes_the_heading_path_of_the_text_under_it() -> None:
    base = _revision(GUIDE, MARKDOWN_RST_TITLES_PROFILE)
    renamed = GUIDE.replace("Testing other databases\n~~~~~~~~~~~~~~~~~~~~~~~", "Testing other backends\n~~~~~~~~~~~~~~~~~~~~~~")
    target = replace(_revision(renamed, MARKDOWN_RST_TITLES_PROFILE), id="obsrev-2")

    changed = [renamed[start:end] for start, end in revision_changed_structural_ranges(base, target)]

    # Selenium is the next adornment style, so it sits one level under the renamed title.
    assert changed == [
        "Testing other backends\n~~~~~~~~~~~~~~~~~~~~~~",
        "Run the tests against PostgreSQL.",
        "Selenium\n--------",
        "Pass ``--selenium`` to run browser tests.",
    ]


def test_each_rst_title_opens_a_reading_section() -> None:
    revision = _revision(GUIDE, MARKDOWN_RST_TITLES_PROFILE)
    fragments = build_revision_fragment_index(revision).fragments

    sections = [
        GUIDE[group.range_start : group.range_end].split("\n", 1)[0]
        for group in build_revision_reading_index(revision, fragments).groups
        if group.kind == "markdown-section"
    ]

    assert sections == ["==============", "Prerequisites", "Testing other databases", "Selenium"]


@pytest.mark.parametrize(
    ("path", "name"),
    [("docs/howto/release.rst", "markdown-rst-titles"), ("docs/howto/release.txt", "markdown-rst-titles"),
     ("docs/howto/release.md", "markdown-structural")],
)
def test_github_repository_files_are_projected_under_the_profile_of_their_extension(path: str, name: str) -> None:
    item = _item(path, relative_path=path, repo_owner="org", repo_name="repo")

    projection = _project("github_repo", item, GUIDE.encode(), {"relative_path": path})

    assert projection.observation_revisions[0].evidence_profile.name == name


@pytest.mark.parametrize(
    ("path", "profile"),
    [
        ("docs/howto/release.txt", MARKDOWN_RST_TITLES_PROFILE),
        ("docs/index.RST", MARKDOWN_RST_TITLES_PROFILE),
        ("README.md", None),
        ("docs/guide.adoc", None),
        ("LICENSE", None),
    ],
)
def test_file_adapters_choose_rst_titles_for_text_files_not_declared_markdown(path, profile) -> None:
    assert file_evidence_profile(path) == profile


@pytest.mark.parametrize("source_type", ["github_repo", "local_markdown"])
def test_file_sources_declare_rst_titles_beside_markdown(source_type: str) -> None:
    declared = declared_representation_profiles(source_type=source_type, observation_type="file_content")

    assert declared == {MARKDOWN_STRUCTURAL_PROFILE, MARKDOWN_RST_TITLES_PROFILE}
    assert representation_profile_for_observation_contract(
        source_type=source_type, observation_type="file_content", chosen=MARKDOWN_RST_TITLES_PROFILE,
    ) == MARKDOWN_RST_TITLES_PROFILE


def test_a_source_cannot_choose_a_profile_its_contract_does_not_declare() -> None:
    with pytest.raises(ValueError, match="declares no alternative profile"):
        representation_profile_for_observation_contract(
            source_type="github_pages", observation_type="page_content", chosen=MARKDOWN_RST_TITLES_PROFILE,
        )
