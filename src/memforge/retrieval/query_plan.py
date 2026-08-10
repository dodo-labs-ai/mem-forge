"""Provider-neutral lexical query planning.

The plan owns candidate-eligibility policy. Storage adapters own the native
index/query mechanics and ranking implementation.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

__all__ = [
    "LexicalQueryPlan",
    "LexicalAnchor",
    "MetadataLexicalQueryPlan",
    "build_lexical_query_plan",
    "metadata_ordinary_terms",
]

_METADATA_QUERY_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "after",
        "all",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "before",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "find",
        "for",
        "from",
        "get",
        "give",
        "has",
        "have",
        "he",
        "help",
        "how",
        "i",
        "in",
        "is",
        "it",
        "just",
        "know",
        "list",
        "me",
        "my",
        "of",
        "on",
        "or",
        "please",
        "search",
        "she",
        "show",
        "tell",
        "than",
        "that",
        "the",
        "their",
        "them",
        "this",
        "to",
        "us",
        "want",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
    }
)
_METADATA_LOOKUP_MODIFIERS = frozenset(
    {
        "bug",
        "bugs",
        "issue",
        "issues",
        "jira",
        "pr",
        "prs",
        "pull",
        "request",
        "requests",
        "stories",
        "story",
        "task",
        "tasks",
        "ticket",
        "tickets",
    }
)

_QUOTED_PHRASE_RE = re.compile(r'["“]([^"”\r\n]+)["”]')
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-"
    r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b"
)
_EXTERNAL_ID_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,15}-\d+\b")
_CODE_SYMBOL_RE = re.compile(
    r"\b(?:[A-Z][a-z0-9]+){2,}\b|\b[A-Z][A-Z0-9]+(?:_[A-Z0-9]+)+\b"
)


@dataclass(frozen=True, slots=True)
class LexicalAnchor:
    """Syntax-recognizable exact lexical evidence."""

    kind: str
    value: str


@dataclass(frozen=True, slots=True)
class MetadataLexicalQueryPlan:
    """Normalized metadata terms and their shared coverage requirement."""

    ordinary_terms: tuple[str, ...]
    minimum_should_match: int
    exact_anchors: tuple[LexicalAnchor, ...] = ()


@dataclass(frozen=True, slots=True)
class LexicalQueryPlan:
    """One normalized lexical plan constructed for a user query."""

    raw_query: str
    metadata: MetadataLexicalQueryPlan


def build_lexical_query_plan(query: str) -> LexicalQueryPlan:
    """Build the shared lexical candidate policy for ``query``.

    Terms are normalized, de-duplicated, and kept in first-seen order.  Short
    metadata queries retain all-term precision; longer queries use the agreed
    recall-first 60 percent coverage gate.
    """

    anchors, anchor_spans = _exact_anchors(query)
    ordinary_query = _without_spans(query, anchor_spans)
    terms = metadata_ordinary_terms(ordinary_query)
    term_count = len(terms)
    minimum_should_match = (
        term_count if term_count <= 2 else math.ceil(0.60 * term_count)
    )
    return LexicalQueryPlan(
        raw_query=query,
        metadata=MetadataLexicalQueryPlan(
            ordinary_terms=terms,
            minimum_should_match=minimum_should_match,
            exact_anchors=anchors,
        ),
    )


def metadata_ordinary_terms(query: str) -> tuple[str, ...]:
    """Return shared ordinary metadata terms for every storage adapter."""

    terms = [
        term
        for term in _ordinary_terms(query)
        if len(term) > 1 and term not in _METADATA_QUERY_STOPWORDS
    ]
    core_terms = [term for term in terms if term not in _METADATA_LOOKUP_MODIFIERS]
    if len(core_terms) >= 2:
        terms = core_terms
    return tuple(terms[:64])


def _exact_anchors(query: str) -> tuple[tuple[LexicalAnchor, ...], tuple[tuple[int, int], ...]]:
    matches: list[tuple[int, int, LexicalAnchor]] = []
    claimed: list[tuple[int, int]] = []
    patterns = (
        ("quoted_phrase", _QUOTED_PHRASE_RE, 1),
        ("uuid", _UUID_RE, 0),
        ("external_id", _EXTERNAL_ID_RE, 0),
        ("code_symbol", _CODE_SYMBOL_RE, 0),
    )
    for kind, pattern, value_group in patterns:
        for match in pattern.finditer(query):
            span = match.span()
            if any(span[0] < end and start < span[1] for start, end in claimed):
                continue
            value = match.group(value_group).strip()
            if not value:
                continue
            claimed.append(span)
            matches.append((span[0], span[1], LexicalAnchor(kind=kind, value=value)))
    matches.sort(key=lambda item: item[0])
    anchors = tuple(dict.fromkeys(item[2] for item in matches))
    spans = tuple((item[0], item[1]) for item in matches)
    return anchors, spans


def _without_spans(query: str, spans: tuple[tuple[int, int], ...]) -> str:
    if not spans:
        return query
    chars = list(query)
    for start, end in spans:
        chars[start:end] = " " * (end - start)
    return "".join(chars)


def _ordinary_terms(query: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    current: list[str] = []

    def flush() -> None:
        if not current:
            return
        term = "".join(current).casefold()
        current.clear()
        if not term or term in seen:
            return
        seen.add(term)
        terms.append(term)

    for char in query:
        if char.isalnum() or char in {"-", "_"}:
            current.append(char)
        else:
            flush()
    flush()
    return terms
