"""Shared normaliser utilities used by gene ``normalize()`` implementations.

These helpers handle the common transformations every gene needs:

- **HTML to markdown** conversion (Confluence, Jira, Teams all emit HTML)
- **Boilerplate stripping** (empty links, TOC headers, repeated whitespace)
- **Token counting** via tiktoken (for enforcing context-window budgets)

All functions are synchronous and stateless so they can be called from both
async gene code and plain unit tests without ceremony.
"""

from __future__ import annotations

import logging
import re

import markdownify

__all__ = [
    "html_to_markdown",
    "strip_boilerplate",
    "annotate_code_blocks",
    "count_tokens",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HTML -> Markdown
# ---------------------------------------------------------------------------

# Pre-compiled patterns for post-conversion cleanup.
_RE_CONSECUTIVE_BLANK_LINES = re.compile(r"\n{3,}")
_RE_EMPTY_HEADERS = re.compile(r"^#{1,6}\s*$", re.MULTILINE)
_RE_TRAILING_WHITESPACE = re.compile(r"[ \t]+$", re.MULTILINE)


class _StructuralMarkdownConverter(markdownify.MarkdownConverter):
    """Keep HTML structures whose semantics Markdown cannot faithfully encode."""

    def convert_table(self, el, text, *args, **kwargs):
        if (el.find(attrs={"rowspan": True}) or el.find(attrs={"colspan": True})
                or el.find(["caption", "table", "pre", "ol", "ul", "dl", "figure", "blockquote", "p", "div", "br"])):
            return "\n\n" + str(el) + "\n\n"
        return super().convert_table(el, text, *args, **kwargs)

    def _complete_html(self, el, text, *args, **kwargs):
        return "\n\n" + str(el) + "\n\n"

    convert_dl = _complete_html
    convert_figure = _complete_html
    convert_pre = _complete_html
    convert_ol = _complete_html


def _clean_prose(markdown: str, transform) -> str:
    """Whitespace/boilerplate cleanup must not rewrite code or retained HTML."""
    from markdown_it import MarkdownIt

    lines = markdown.splitlines(keepends=True)
    protected = set()
    for token in MarkdownIt("commonmark").parse(markdown):
        if token.type in {"html_block", "fence", "code_block"} and token.map:
            protected.update(range(*token.map))
    result, pending = [], []
    for index, line in enumerate(lines):
        if index in protected:
            result.extend((transform("".join(pending)), line))
            pending = []
        else:
            pending.append(line)
    result.append(transform("".join(pending)))
    return "".join(result).strip("\r\n")


def html_to_markdown(html: str) -> str:
    """Convert an HTML string to clean Markdown.

    Uses *markdownify* for the heavy lifting, then applies several
    post-processing passes to tighten up the output:

    1. Strip trailing whitespace from every line.
    2. Remove empty heading lines (``#`` with no text).
    3. Collapse runs of three or more newlines down to two.
    4. Strip leading / trailing whitespace from the whole document.

    Parameters
    ----------
    html:
        Raw HTML content.  May contain full ``<html>`` / ``<body>`` wrappers
        or just a fragment -- markdownify handles both.

    Returns
    -------
    str
        Cleaned Markdown text.
    """
    if not html or not html.strip():
        return ""

    # Pre-process: remove truly useless tags via BeautifulSoup.
    # Images are kept — markdownify converts <img alt="..."> to ![alt](src),
    # preserving useful descriptions for memory extraction.
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(["script", "style", "nav", "footer"]):
        tag.decompose()

    md = _StructuralMarkdownConverter(heading_style="ATX", bullets="-").convert(str(soup))

    def clean(text):
        text = _RE_TRAILING_WHITESPACE.sub("", text)
        text = _RE_EMPTY_HEADERS.sub("", text)
        return _RE_CONSECUTIVE_BLANK_LINES.sub("\n\n", text)

    return _clean_prose(md, clean)


# ---------------------------------------------------------------------------
# Boilerplate stripping
# ---------------------------------------------------------------------------

# Patterns matched against the entire markdown body.
_BOILERPLATE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Empty markdown links:  [](...)  or  [ ](...)
    (re.compile(r"\[[\s]*\]\([^)]*\)"), ""),
    # "Table of Contents" headers (common Confluence artefact)
    (re.compile(r"^#{1,6}\s*Table of Contents\s*$", re.MULTILINE | re.IGNORECASE), ""),
    # Confluence-style TOC macro placeholders
    (re.compile(r"^#{1,6}\s*On this page\s*$", re.MULTILINE | re.IGNORECASE), ""),
    # Horizontal rules that are just noise (3+ dashes on their own line, repeated)
    (re.compile(r"(\n---+\n){2,}"), "\n---\n"),
    # Non-breaking spaces and other Unicode whitespace oddities
    (re.compile(r"\u00a0"), " "),
    # Zero-width characters
    (re.compile(r"[\u200b\u200c\u200d\ufeff]"), ""),
    # Confluence page-properties macro remnants
    (re.compile(r"^#{1,6}\s*Page Properties\s*$", re.MULTILINE | re.IGNORECASE), ""),
    # Jira-style "Generated by" footers
    (re.compile(r"^Generated by .*$", re.MULTILINE | re.IGNORECASE), ""),
]

# Final cleanup after boilerplate removal.
_RE_POST_STRIP_BLANK_LINES = re.compile(r"\n{3,}")


def strip_boilerplate(md: str) -> str:
    """Remove common boilerplate noise from Markdown text.

    This targets artefacts that frequently survive HTML-to-Markdown
    conversion from Confluence, Jira, and similar enterprise tools:

    - Empty links ``[](...)``
    - "Table of Contents" / "On this page" headings
    - Repeated horizontal rules
    - Unicode non-breaking spaces and zero-width characters
    - "Generated by ..." footers

    Parameters
    ----------
    md:
        Markdown text, typically the output of ``html_to_markdown()``.

    Returns
    -------
    str
        Cleaned Markdown with boilerplate removed.
    """
    if not md:
        return ""

    def clean(text):
        for pattern, replacement in _BOILERPLATE_PATTERNS:
            text = pattern.sub(replacement, text)
        return _RE_POST_STRIP_BLANK_LINES.sub("\n\n", text)

    return _clean_prose(md, clean)


# ---------------------------------------------------------------------------
# Code block annotation
# ---------------------------------------------------------------------------

# Match fenced code blocks: ```optional-lang\n...\n```
_RE_CODE_BLOCK = re.compile(r"(```[^\n]*\n[\s\S]*?```)", re.MULTILINE)


def annotate_code_blocks(md: str) -> str:
    """Wrap fenced code blocks with markers so the LLM can distinguish code from prose.

    The enricher prompt instructs the LLM to ignore content inside these markers
    when extracting entities, preventing Java class names and code identifiers
    from leaking into the entity list.

    Parameters
    ----------
    md:
        Markdown text, typically the output of ``html_to_markdown()`` +
        ``strip_boilerplate()``.

    Returns
    -------
    str
        Markdown with code blocks wrapped in ``[CODE_BLOCK_START]`` /
        ``[CODE_BLOCK_END]`` markers.
    """
    if not md or "```" not in md:
        return md

    return _RE_CODE_BLOCK.sub(
        lambda m: f"[CODE_BLOCK_START]\n{m.group(0)}\n[CODE_BLOCK_END]",
        md,
    )


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


def count_tokens(text: str, encoding: str = "cl100k_base") -> int:
    """Estimate token count for a text string.

    Uses a simple heuristic (words × 1.33) which is accurate enough for
    budgeting purposes. tiktoken was removed because it requires downloading
    encoding files from the internet, which fails on corporate networks.

    Parameters
    ----------
    text:
        The string to estimate tokens for.

    Returns
    -------
    int
        Estimated token count. Returns 0 for empty input.
    """
    if not text or not text.strip():
        return 0
    # ~1.33 tokens per whitespace-delimited word is a reasonable estimate
    # for English text with GPT-4/Claude tokenizers.
    return int(len(text.split()) * 1.33)
