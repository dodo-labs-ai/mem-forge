"""reStructuredText section titles as a Markdown block rule.

A reStructuredText title is one line of text underlined, and optionally also
overlined, by a line that repeats one punctuation character. CommonMark knows
only ``=`` and ``-`` underlines, and reads a ``~~~`` or a backtick underline as
the opening of a code fence that runs to the end of the document. This rule
reads every such title as a heading, so the text under it stays prose.

As in reStructuredText, a title's level is not fixed by its character: each
new adornment style (character, and whether it is overlined) opens the next
level, in the order the styles first appear in the document.

Markdown keeps what is unambiguously Markdown. A line that opens a bullet list
item or a block quote is not a title; a numbered line can be, since
reStructuredText does not read ``1. Introduction`` over an underline as a list
item either. A backtick or tilde line is a title adornment only as an underline
followed by a blank line: a code fence is followed by its code, and a one-line
fenced block would otherwise read as an overlined title.
"""

from __future__ import annotations

import re
import string

from markdown_it import MarkdownIt
from markdown_it.rules_block import StateBlock

# Any printable 7-bit ASCII character that is neither a letter nor a digit may adorn a title.
_ADORNMENT_CHARACTERS = frozenset(string.punctuation)
_FENCE_CHARACTERS = frozenset("`~")
# An underline shorter than its title still marks it once it is this long, as Sphinx renders it.
_MIN_SHORT_UNDERLINE = 4
_MAX_HEADING_LEVEL = 6
_STYLES_KEY = "rst_title_styles"
# A CommonMark bullet list item or block quote marker at the start of a line.
_CONTAINER_MARKER = re.compile(r"[-+*](?:[ \t]|$)|>")


def enable_rst_titles(parser: MarkdownIt) -> MarkdownIt:
    """Read reStructuredText titles as headings, ahead of code fences and thematic breaks."""

    parser.block.ruler.before("fence", "rst_title", _rst_title, {"alt": ["paragraph"]})
    return parser


def _line(state: StateBlock, line: int, end_line: int) -> str | None:
    """A line's text inside the current block, or None when it is blank, outside it or indented as code."""

    if (
        line >= end_line
        or state.isEmpty(line)
        or state.sCount[line] < state.blkIndent
        or state.sCount[line] - state.blkIndent >= 4
    ):
        return None
    return state.src[state.bMarks[line] + state.tShift[line] : state.eMarks[line]].rstrip()


def _adornment(text: str | None) -> str | None:
    """The character a line repeats when the whole line is one adornment run, else None."""

    if not text or len(text) < 2 or text[0] not in _ADORNMENT_CHARACTERS or text.strip(text[0]):
        return None
    return text[0]


def _rst_title(state: StateBlock, start_line: int, end_line: int, silent: bool) -> bool:
    first = _line(state, start_line, end_line)
    if first is None:
        return False
    overline = _adornment(first)
    if overline in _FENCE_CHARACTERS:
        return False
    title_line = start_line + 1 if overline else start_line
    title = _line(state, title_line, end_line) if overline else first
    if title is None or _adornment(title) is not None or _CONTAINER_MARKER.match(title.lstrip()):
        return False
    title = title.strip()
    underline = _line(state, title_line + 1, end_line)
    character = _adornment(underline)
    if character is None or underline is None:
        return False
    if len(underline) < len(title) and len(underline) < _MIN_SHORT_UNDERLINE:
        return False
    if overline and (character != overline or len(first) != len(underline)):
        return False
    if character in _FENCE_CHARACTERS and title_line + 2 < end_line and not state.isEmpty(title_line + 2):
        return False
    if silent:
        return True

    styles: list[tuple[str, bool]] = state.env.setdefault(_STYLES_KEY, [])
    style = (character, overline is not None)
    if style not in styles:
        styles.append(style)
    tag = f"h{min(styles.index(style) + 1, _MAX_HEADING_LEVEL)}"
    state.line = title_line + 2

    token = state.push("heading_open", tag, 1)
    token.markup = character
    token.map = [start_line, state.line]
    token = state.push("inline", "", 0)
    token.content = title
    token.map = [title_line, title_line + 1]
    token.children = []
    token = state.push("heading_close", tag, -1)
    token.markup = character
    return True
