"""Notion's enhanced Markdown becomes CommonMark, one rule per tag family."""

from __future__ import annotations

from memforge.genes.notion_markdown import notion_markdown


def test_callouts_become_commonmark() -> None:
    text = '<callout icon="💡" color="gray_bg">\n\tShip on Friday\n\t- check the build\n\t\t- twice\n</callout>'

    assert notion_markdown(text) == "> 💡 Ship on Friday\n> - check the build\n> \t- twice"


def test_a_callout_with_an_image_icon_keeps_only_its_text() -> None:
    assert notion_markdown('<callout icon="https://example.com/i.png">\n\tNote\n</callout>') == "> Note"


def test_toggles_become_commonmark() -> None:
    text = (
        '<details color="red">\n<summary>Release steps</summary>\n\tTag the build\n'
        '\t<callout icon="⚠️">\n\t\tNever on Friday\n\t</callout>\n</details>\nAfter'
    )

    assert notion_markdown(text) == "**Release steps**\n\nTag the build\n> ⚠️ Never on Friday\n\nAfter"


def test_a_toggle_inside_a_list_item_stays_in_the_item() -> None:
    text = "- Item\n\t<details>\n\t<summary>More</summary>\n\t\tHidden text\n\t</details>"

    assert notion_markdown(text) == "- Item\n\n\t**More**\n\n\tHidden text"


def test_toggleable_headings_become_commonmark() -> None:
    assert notion_markdown('## Setup {toggle="true" color="blue"}\n\tInstall it') == "## Setup\nInstall it"


def test_columns_become_commonmark() -> None:
    text = "<columns>\n\t<column>\n\t\tLeft\n\t</column>\n\t<column>\n\t\tRight\n\t</column>\n</columns>"

    assert notion_markdown(text) == "Left\n\nRight"


def test_synced_blocks_become_commonmark() -> None:
    text = (
        '<synced_block url="https://app.notion.com/p/a#b">\n\tOriginal text\n</synced_block>\n'
        '<synced_block_reference url="https://app.notion.com/p/c#b">\n\tShown elsewhere\n</synced_block_reference>'
    )

    assert notion_markdown(text) == "Original text\n\nShown elsewhere"


def test_tables_become_commonmark() -> None:
    text = (
        '<table header-row="true">\n\t<colgroup>\n\t\t<col color="red">\n\t</colgroup>\n'
        '\t<tr>\n\t\t<td>Name</td>\n\t\t<td>Owner</td>\n\t</tr>\n'
        '\t<tr color="blue">\n\t\t<td>Build | deploy</td>\n\t\t<td><span color="red">Ada</span></td>\n\t</tr>\n</table>'
    )

    assert notion_markdown(text) == "| Name | Owner |\n| --- | --- |\n| Build \\| deploy | Ada |"


def test_a_table_without_a_header_row_gets_an_empty_one() -> None:
    text = "<table>\n\t<tr>\n\t\t<td>a</td>\n\t\t<td>b</td>\n\t</tr>\n</table>"

    assert notion_markdown(text) == "|  |  |\n| --- | --- |\n| a | b |"


def test_page_and_database_references_become_commonmark() -> None:
    text = (
        '<page url="https://app.notion.com/p/abc" color="red">Roadmap</page>\n'
        '<database url="https://app.notion.com/p/def" inline="true"></database>'
    )

    assert notion_markdown(text) == (
        "[Roadmap](https://app.notion.com/p/abc)\n[https://app.notion.com/p/def](https://app.notion.com/p/def)"
    )


def test_mentions_become_commonmark() -> None:
    text = (
        'Ask <mention-user url="user://1">Ada</mention-user> about '
        '<mention-page url="https://app.notion.com/p/abc">Roadmap</mention-page> and '
        '<mention-database url="https://app.notion.com/p/db">Tasks</mention-database> by '
        '<mention-date start="2026-03-01" startTime="09:30" timeZone="Europe/Berlin"/> or '
        '<mention-date start="2026-03-01" end="2026-03-05"/>'
    )

    assert notion_markdown(text) == (
        "Ask @Ada about [Roadmap](https://app.notion.com/p/abc) and [Tasks](https://app.notion.com/p/db) "
        "by 2026-03-01 09:30 (Europe/Berlin) or 2026-03-01 – 2026-03-05"
    )


def test_media_become_commonmark() -> None:
    text = (
        "![Diagram](notion-file:a/b/diagram.png)\n"
        '<file src="notion-file:a/c/plan.pdf">Q3 plan</file>\n'
        '<pdf src="notion-file:a/d/spec.pdf"></pdf>\n'
        '<video src="https://youtube.com/x" color="red">Demo</video>\n'
        '<embed src="https://figma.com/f"></embed>'
    )

    assert notion_markdown(text) == (
        "![Diagram](notion-file:a/b/diagram.png)\n[Q3 plan](notion-file:a/c/plan.pdf)\n"
        "[notion-file:a/d/spec.pdf](notion-file:a/d/spec.pdf)\n[Demo](https://youtube.com/x)\n"
        "[https://figma.com/f](https://figma.com/f)"
    )


def test_color_and_style_attributes_become_commonmark() -> None:
    text = (
        'Plain {color="red"}\n'
        '<span color="blue_bg">colored</span> and <span underline="true">underlined</span>\n'
        '> Line 1<br>Line 2 {color="gray"}'
    )

    assert notion_markdown(text) == "Plain\ncolored and underlined\n> Line 1\\\n> Line 2"


def test_empty_blocks_and_tables_of_contents_become_commonmark() -> None:
    text = 'Before\n<empty-block/>\n<table_of_contents color="gray"/>\n<custom-block definition-id="1" />\nAfter'

    assert notion_markdown(text) == "Before\n\nAfter"


def test_children_of_a_paragraph_or_heading_never_become_code() -> None:
    text = "# Title\n\tChild\n\t\tGrandchild\nParagraph\n\t```\n\tcode\n\t\tindented\n\t```\n- a\n\t- b"

    assert notion_markdown(text) == "# Title\nChild\nGrandchild\nParagraph\n```\ncode\n\tindented\n```\n- a\n\t- b"


def test_code_is_never_converted() -> None:
    text = '```html\n<callout>not a callout</callout>\n```\nInline `<page url="x">y</page>` code'

    assert notion_markdown(text) == text


def test_autolinks_are_kept() -> None:
    assert notion_markdown("See <https://example.com>") == "See <https://example.com>"


def test_an_unknown_tag_keeps_its_text() -> None:
    assert notion_markdown('<mention-agent-v2 url="x">Helper</mention-agent-v2> and <future-tag a="b"/>') == "Helper and"
