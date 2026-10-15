"""Provider-neutral lexical query planning.

The plan owns candidate-eligibility policy and the identifier syntax a query
contains. Storage adapters own the native index/query mechanics and ranking
implementation.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, cast

__all__ = [
    "IdentifierKind",
    "LexicalQueryPlan",
    "LexicalAnchor",
    "MetadataLexicalQueryPlan",
    "QueryIdentifier",
    "QueryIdentity",
    "build_lexical_query_plan",
    "metadata_ordinary_terms",
]

IdentifierKind = Literal[
    "url",
    "quoted_phrase",
    "uuid",
    "path",
    "external_id",
    "dotted_name",
    "snake_case",
    "slash_term",
    "dotted_term",
    "camel_case",
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

# Identifier syntax shared by exact anchors and query identity.
_QUOTED_PHRASE_RE = re.compile(r'["“]([^"”\r\n]+)["”]')
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-"
    r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b"
)

# Exact anchors: identifiers searched verbatim at a high score and removed from
# the ordinary terms.  An issue key needs an upper-case project prefix; a code
# symbol is PascalCase with at least two capitalised humps or UPPER_SNAKE_CASE.
_ISSUE_KEY_ANCHOR_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,15}-\d+\b")
_CODE_SYMBOL_ANCHOR_RE = re.compile(
    r"\b(?:[A-Z][a-z0-9]+){2,}\b|\b[A-Z][A-Z0-9]+(?:_[A-Z0-9]+)+\b"
)

# Query identity: every identifier shape that can name one item.
_URL_RE = re.compile(r"(?<![A-Za-z0-9+.-])[A-Za-z][A-Za-z0-9+.-]*://[^\s\"'<>“”]+")
_EXTERNAL_REFERENCE_RE = re.compile(
    r"\b[A-Z][A-Z0-9]+-\d+\b|#\d+\b|\bINC\d+\b",
    re.IGNORECASE,
)
# A slash-joined token; ``_is_path`` decides whether it is a path.  Any other
# slash token with a letter is a slash term: prose such as "and/or" and
# "TCP/IP", or a short name such as "src/memforge".
_PATH_SEGMENT = r"[A-Za-z0-9_.](?:[A-Za-z0-9_.-]*[A-Za-z0-9_-])?"
_SLASH_TOKEN_RE = re.compile(
    rf"(?<![A-Za-z0-9_./~-])"
    rf"(?:(?:~|\.{{1,2}})?(?:/{_PATH_SEGMENT})+|{_PATH_SEGMENT}(?:/{_PATH_SEGMENT})+)/?"
    rf"(?![A-Za-z0-9_/-])"
)
# A dot-joined token of identifier parts, optionally led by a dot as in
# ".pre-commit-config.yaml"; every part but the last may join words with
# hyphens, as in "docker-compose.yml" and "2024-report.pdf".
# ``_is_dotted_name`` decides whether it is a module, attribute or file name.
# Any other dotted token is a dotted term: an abbreviation such as "e.g.", or a
# name such as "Node.js" or "Cargo.toml".
_DOTTED_PART = r"[A-Za-z_][A-Za-z0-9_]*"
_DOTTED_STEM_PART = rf"(?:[0-9]+-)*{_DOTTED_PART}(?:-[A-Za-z0-9_]+)*"
_DOTTED_TOKEN_RE = re.compile(
    rf"(?<![A-Za-z0-9_./-])\.?(?:{_DOTTED_STEM_PART}\.)+{_DOTTED_PART}"
    r"(?![A-Za-z0-9_/-]|\.[A-Za-z0-9_])"
)
_SNAKE_CASE_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+\b")
_CAMEL_CASE_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9]*[a-z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*\b"
    r"|\b[a-z][a-z0-9]*[A-Z][A-Za-z0-9]*\b"
)
_FILE_NAME_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\.[A-Za-z0-9]+")
_CAPITALISED_WORD_RE = re.compile(r"[A-Z][a-z0-9]*")


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
class QueryIdentifier:
    """One identifier recognised by its syntax alone."""

    kind: IdentifierKind
    value: str


@dataclass(frozen=True, slots=True)
class QueryIdentity:
    """The identifiers in a query and whether any other text surrounds them.

    ``has_free_text`` is true when a letter or digit of any script remains
    outside every identifier, so "MemForge" has none and "How do I configure
    workspaces in MemForge?" has some.
    """

    identifiers: tuple[QueryIdentifier, ...]
    has_free_text: bool

    @property
    def kinds(self) -> frozenset[IdentifierKind]:
        return frozenset(identifier.kind for identifier in self.identifiers)


@dataclass(frozen=True, slots=True)
class LexicalQueryPlan:
    """One normalized lexical plan constructed for a user query."""

    raw_query: str
    metadata: MetadataLexicalQueryPlan
    identity: QueryIdentity


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
        identity=_query_identity(query),
    )


def _query_identity(query: str) -> QueryIdentity:
    """Return the identifiers in ``query`` and whether free text remains.

    Each character belongs to at most one identifier; earlier kinds claim
    their span first, so a path keeps its file name and a URL keeps its host.
    """

    matches = _claim_spans(
        query,
        (
            ("url", _URL_RE, 0, None),
            ("quoted_phrase", _QUOTED_PHRASE_RE, 1, None),
            ("uuid", _UUID_RE, 0, None),
            ("path", _SLASH_TOKEN_RE, 0, _is_path),
            ("external_id", _EXTERNAL_REFERENCE_RE, 0, None),
            ("dotted_name", _DOTTED_TOKEN_RE, 0, _is_dotted_name),
            ("snake_case", _SNAKE_CASE_RE, 0, None),
            ("slash_term", _SLASH_TOKEN_RE, 0, _has_letter),
            ("dotted_term", _DOTTED_TOKEN_RE, 0, None),
            ("camel_case", _CAMEL_CASE_RE, 0, None),
        ),
    )
    remainder = _without_spans(query, tuple((start, end) for start, end, _, _ in matches))
    return QueryIdentity(
        identifiers=tuple(
            dict.fromkeys(
                QueryIdentifier(kind=cast(IdentifierKind, kind), value=value)
                for _, _, kind, value in matches
            )
        ),
        has_free_text=any(char.isalnum() for char in remainder),
    )


def _is_path(token: str) -> bool:
    """A path rather than a slash pair such as "and/or", "TCP/IP" or a date.

    A path contains a letter and names something relative to the home or
    current directory, or has two segments under the root, three or more
    segments, or a directory and a file name ("src/App.tsx").  Two capitalised
    names such as "React/Node.js" are a pair of products, not a directory and
    a file.
    """

    if not _has_letter(token):
        return False
    if token.startswith(("~/", "./", "../")):
        return True
    segments = [segment for segment in token.split("/") if segment]
    if token.startswith("/"):
        return len(segments) >= 2
    if len(segments) >= 3:
        return True
    name = segments[-1]
    if not _FILE_NAME_RE.fullmatch(name):
        return False
    return not (all(segment[0].isupper() for segment in segments) and not _is_dotted_name(name))


def _has_letter(token: str) -> bool:
    return any(char.isalpha() for char in token)


def _is_dotted_name(token: str) -> bool:
    """A module, attribute or file name rather than an abbreviation or product name.

    Abbreviations have single-letter parts ("e.g.", "i.e.", "U.S.").  A name
    of two parts that starts with one capitalised word or ends in capitals
    reads as a product ("Node.js", "ASP.NET") as readily as a file
    ("Cargo.toml"), so it is left to the surrounding text.
    """

    parts = token.removeprefix(".").split(".")
    if all(len(part) == 1 for part in parts):
        return False
    if len(parts) == 2:
        first, second = parts
        if _CAPITALISED_WORD_RE.fullmatch(first) or second.isupper():
            return False
    return True


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
    matches = _claim_spans(
        query,
        (
            ("quoted_phrase", _QUOTED_PHRASE_RE, 1, None),
            ("uuid", _UUID_RE, 0, None),
            ("external_id", _ISSUE_KEY_ANCHOR_RE, 0, None),
            ("code_symbol", _CODE_SYMBOL_ANCHOR_RE, 0, None),
        ),
    )
    anchors = tuple(dict.fromkeys(LexicalAnchor(kind=kind, value=value) for _, _, kind, value in matches))
    spans = tuple((start, end) for start, end, _, _ in matches)
    return anchors, spans


def _claim_spans(
    query: str,
    patterns: tuple[tuple[str, re.Pattern[str], int, Callable[[str], bool] | None], ...],
) -> list[tuple[int, int, str, str]]:
    """Match ``patterns`` in priority order without overlapping spans.

    Each pattern is ``(kind, regex, value_group, accept)``; ``accept`` rejects
    a syntactic match whose text is not that kind.  Matches come back in query
    order as ``(start, end, kind, value)``.
    """

    matches: list[tuple[int, int, str, str]] = []
    for kind, pattern, value_group, accept in patterns:
        for match in pattern.finditer(query):
            start, end = match.span()
            if any(start < claimed_end and claimed_start < end for claimed_start, claimed_end, _, _ in matches):
                continue
            value = match.group(value_group).strip()
            if not value or (accept is not None and not accept(value)):
                continue
            matches.append((start, end, kind, value))
    matches.sort(key=lambda item: item[0])
    return matches


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
