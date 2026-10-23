"""Convert the enhanced Markdown Notion returns for a page into CommonMark.

Notion's Markdown endpoint writes ordinary Markdown for most blocks and
XML-like tags for the rest (developers.notion.com, "Enhanced Markdown").
:func:`notion_markdown` keeps every word of text and turns each tag family
into its CommonMark form::

    notion_markdown('<callout icon="💡">\\n\\tShip on Friday\\n</callout>')
    # "> 💡 Ship on Friday"

One rule per tag family:

- container blocks, whose children are indented one tab deeper than the tag:
  a callout becomes a block quote led by its icon; a toggle
  (``<details>``/``<summary>``) becomes its bold title followed by its
  children; columns and synced blocks become their children in order;
- a table becomes a pipe table (an empty header row unless Notion marks the
  first row a header);
- page, database and mention tags become links named by their text, a user
  mention becomes ``@name``, and a date mention its date;
- audio, video, file, PDF and embed tags become links named by their
  caption;
- ``{color="…"}`` attributes, ``<span>`` styling, empty blocks, tables of
  contents and custom blocks keep only their text, if any;
- any other tag keeps its inner text.

Notion writes one block per line, with no blank line between blocks, while
CommonMark joins consecutive lines into one paragraph and lets a paragraph
after a quote or list item continue it. So every block is followed by a
blank line, except that the items of one list stay together::

    notion_markdown("First\\nSecond\\n- one\\n- two")
    # "First\\n\\nSecond\\n\\n- one\\n- two"

A block equation, ``$$`` lines around its formula, is one block.

Code, fenced or inline, is never changed. ``<unknown>`` tags are resolved
before this runs (the Gene reads them again), so no rule turns one into a
link.
"""

from __future__ import annotations

import html
import re

__all__ = ["notion_markdown"]

_CONTAINERS = ("callout", "details", "columns", "column", "synced_block", "synced_block_reference")
_OPEN_CONTAINER = re.compile(r"^<(" + "|".join(_CONTAINERS) + r")\b([^>]*)>(.*)$")
_FENCE = re.compile(r"^\s*(```+|~~~+)")
_EQUATION = "$$"
_LIST_ITEM = re.compile(r"^(?:[-*+]|\d+[.)])(?:\s|$)")
_ATTRIBUTE = re.compile(r'([\w-]+)="([^"]*)"')
_BLOCK_ATTRIBUTES = re.compile(r'\s*\{(?:[\w-]+="[^"]*"\s*)+\}\s*$')
_INLINE_CODE = re.compile(r"(`+)(.+?)\1")
_ROW = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.DOTALL)
_CELL = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", re.DOTALL)
_LINK_TAG = re.compile(
    r"<(page|database|mention-page|mention-database|mention-data-source|mention-agent)\b([^>]*?)(?:/>|>(.*?)</\1>)",
    re.DOTALL,
)
_USER_MENTION = re.compile(r"<mention-user\b([^>]*?)(?:/>|>(.*?)</mention-user>)", re.DOTALL)
_DATE_MENTION = re.compile(r"<mention-date\b([^>]*?)(?:/>|>(.*?)</mention-date>)", re.DOTALL)
_MEDIA_TAG = re.compile(r"<(audio|video|file|pdf|embed)\b([^>]*?)(?:/>|>(.*?)</\1>)", re.DOTALL)
_DROPPED_TAG = re.compile(r"<(empty-block|table_of_contents|custom-block)\b[^>]*?/?>(?:</\1>)?")
_BREAK = re.compile(r"<br\s*/?>")
_ANY_TAG = re.compile(r"(?<!\\)</?[a-z][\w-]*(?:\s[^<>]*?)?/?>")


def notion_markdown(text: str) -> str:
    """CommonMark for one page of Notion's enhanced Markdown; see the module docstring."""
    lines = _convert_blocks(text.replace("\r\n", "\n").split("\n"))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------


def _convert_blocks(lines: list[str]) -> list[str]:
    """Convert container blocks and tables line by line and every other line inline, one blank line between blocks.

    Items of one list are not separated, so the list stays tight.
    """
    output: list[str] = []
    previous_was_list_item: bool | None = None  # None before the first block

    def start_block(is_list_item: bool) -> None:
        nonlocal previous_was_list_item
        if previous_was_list_item is not None and not (previous_was_list_item and is_list_item):
            output.append("")
        previous_was_list_item = is_list_item

    index = 0
    while index < len(lines):
        line = lines[index]
        indent = len(line) - len(line.lstrip("\t"))
        body = line[indent:]
        fence = _FENCE.match(body)
        if fence or body.strip() == _EQUATION:
            end = _fence_end(lines, index, fence.group(1) if fence else _EQUATION)
            start_block(False)
            output.extend(lines[index : end + 1])
            index = end + 1
            continue
        container = _OPEN_CONTAINER.match(body)
        if container:
            end, inner = _container(lines, index, indent, container)
            rendered = _render_container(container.group(1), container.group(2), inner)
            start_block(False)
            output.extend("\t" * indent + rendered_line if rendered_line else "" for rendered_line in rendered)
            index = end + 1
            continue
        if re.match(r"<table\b", body):
            end = next((at for at in range(index, len(lines)) if "</table>" in lines[at]), len(lines) - 1)
            start_block(False)
            output.extend("\t" * indent + row for row in _table("\n".join(lines[index : end + 1])))
            index = end + 1
            continue
        converted = _inline(line)
        if converted.strip():
            start_block(_LIST_ITEM.match(body) is not None)
            output.append(converted)
        else:
            output.append("")  # an empty block, or a tag that keeps no text
        index += 1
    return _children_indented_as_commonmark(output)


def _children_indented_as_commonmark(lines: list[str]) -> list[str]:
    """Re-indent child blocks so that only list items nest.

    Notion indents a block's children one tab deeper than the block. Under
    a list item that is a nested list or continuation, as in CommonMark;
    under a heading, paragraph or quote it would start an indented code
    block, so such children move to their parent's level::

        "# Heading\\n\\tChild"   ->  "# Heading\\nChild"
        "- Item\\n\\t- Nested"   ->  unchanged

    Fenced code moves with its fence and keeps its own indentation.
    """
    output: list[str] = []
    stack: list[tuple[int, int, bool]] = []  # (Notion indent, CommonMark indent, is a list item)
    fence: tuple[str, int] | None = None  # (marker, tabs removed from the fence's lines)
    for line in lines:
        if fence is not None:
            output.append(_dedent(line, fence[1]))
            if line.strip().startswith(fence[0]):
                fence = None
            continue
        if not line.strip():
            output.append("")
            continue
        indent = len(line) - len(line.lstrip("\t"))
        body = line[indent:]
        while stack and stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1] if stack else None
        if parent is None:
            level = 0
        elif parent[2]:
            level = parent[1] + (indent - parent[0])
        else:
            level = parent[1]
        stack.append((indent, level, _LIST_ITEM.match(body) is not None))
        output.append("\t" * level + body)
        opened = _FENCE.match(body)
        if opened:
            fence = (opened.group(1)[0] * len(opened.group(1)), indent - level)
    return output


def _fence_end(lines: list[str], start: int, marker: str) -> int:
    for at in range(start + 1, len(lines)):
        if lines[at].strip().startswith(marker[0] * len(marker)):
            return at
    return len(lines) - 1


def _container(lines: list[str], start: int, indent: int, opened: re.Match[str]) -> tuple[int, list[str]]:
    """The index of a container's closing line and its inner lines, one indent level shallower.

    A container written on one line, ``<callout>text</callout>``, has that
    text as its only inner line.
    """
    name, rest = opened.group(1), opened.group(3)
    close = f"</{name}>"
    if close in rest:
        return start, [rest[: rest.rindex(close)]]
    depth = 1
    inner: list[str] = [rest] if rest.strip() else []
    at = start + 1
    while at < len(lines):
        body = lines[at].lstrip("\t")
        if re.match(rf"<{name}\b[^>]*>", body) and close not in body:
            depth += 1
        elif body.startswith(close):
            depth -= 1
            if depth == 0:
                break
        inner.append(_dedent(lines[at], indent + 1))
        at += 1
    return at, inner


def _dedent(line: str, levels: int) -> str:
    removed = 0
    while removed < levels and line.startswith("\t"):
        line = line[1:]
        removed += 1
    return line


def _render_container(name: str, attributes: str, inner: list[str]) -> list[str]:
    """One converted container: its children, led by what the container itself says."""
    attrs = dict(_ATTRIBUTE.findall(attributes))
    summary = re.match(r"\s*<summary\b[^>]*>(.*?)</summary>\s*$", inner[0]) if name == "details" and inner else None
    if summary:
        title = _inline(summary.group(1)).strip()
        return ["", f"**{title}**" if title else "", "", *_trimmed(_convert_blocks(inner[1:])), ""]
    inner = _convert_blocks(inner)
    if name == "callout":
        icon = attrs.get("icon", "")
        if icon and "/" not in icon and inner:
            inner = [f"{icon} {inner[0]}".rstrip(), *inner[1:]]
        return [f"> {line}".rstrip() for line in _trimmed(inner)]
    return ["", *_trimmed(inner), ""]


def _trimmed(lines: list[str]) -> list[str]:
    while lines and not lines[0].strip():
        lines = lines[1:]
    while lines and not lines[-1].strip():
        lines = lines[:-1]
    return lines


def _table(block: str) -> list[str]:
    """A ``<table>`` as a pipe table; cells keep their inline text and line breaks."""
    header_row = re.search(r'header-row="true"', block[: block.find(">") + 1]) is not None
    rows = [[_cell(cell) for cell in _CELL.findall(row)] for row in _ROW.findall(block)]
    if not rows:
        return []
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    header, body = (rows[0], rows[1:]) if header_row else ([""] * width, rows)
    lines = ["", _pipe_row(header), _pipe_row(["---"] * width)]
    lines.extend(_pipe_row(row) for row in body)
    return [*lines, ""]


def _cell(content: str) -> str:
    text = _inline(" ".join(part.strip() for part in content.strip().split("\n")))
    return text.replace("|", "\\|")


def _pipe_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


# ---------------------------------------------------------------------------
# Inline
# ---------------------------------------------------------------------------


def _inline(line: str) -> str:
    """One line with its tags converted; inline code spans are left as they are."""
    parts = _INLINE_CODE.split(line)
    # split() with two groups yields: text, ticks, code, text, ticks, code, …, text
    converted = []
    for position in range(0, len(parts), 3):
        converted.append(_inline_text(parts[position]))
        if position + 2 < len(parts):
            ticks, code = parts[position + 1], parts[position + 2]
            converted.append(f"{ticks}{code}{ticks}")
    return "".join(converted)


def _inline_text(text: str) -> str:
    text = _BLOCK_ATTRIBUTES.sub("", text)
    text = _LINK_TAG.sub(_link, text)
    text = _USER_MENTION.sub(_user, text)
    text = _DATE_MENTION.sub(_date, text)
    text = _MEDIA_TAG.sub(_media, text)
    text = _DROPPED_TAG.sub("", text)
    text = _BREAK.sub(lambda _match: _line_break(text), text)
    return _ANY_TAG.sub("", text)


def _link(match: re.Match[str]) -> str:
    url = _attr(match.group(2), "url")
    label = (match.group(3) or "").strip() or url
    return f"[{label}]({url})" if url else label


def _user(match: re.Match[str]) -> str:
    name = (match.group(2) or "").strip()
    return f"@{name}" if name else _attr(match.group(1), "url")


def _date(match: re.Match[str]) -> str:
    attributes = match.group(1)
    text = " ".join(part for part in (_attr(attributes, "start"), _attr(attributes, "startTime")) if part)
    end = " ".join(part for part in (_attr(attributes, "end"), _attr(attributes, "endTime")) if part)
    if end:
        text += f" – {end}"
    zone = _attr(attributes, "timeZone")
    return f"{text} ({zone})" if zone and text else text or (match.group(2) or "")


def _media(match: re.Match[str]) -> str:
    url = _attr(match.group(2), "src")
    caption = (match.group(3) or "").strip() or url
    return f"[{caption}]({url})" if url else caption


def _line_break(line: str) -> str:
    """A hard line break that continues the line's block quote, if it is in one."""
    prefix = re.match(r"^[\t ]*(?:>\s?)*", line)
    quote = prefix.group(0).lstrip("\t ") if prefix else ""
    return "\\\n" + quote


def _attr(attributes: str, name: str) -> str:
    for key, value in _ATTRIBUTE.findall(attributes):
        if key == name:
            return html.unescape(value)
    return ""
