"""Helpers for source-metadata lexical projections."""

from __future__ import annotations

import re
from typing import Iterable


_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_DELIMITER_RE = re.compile(r"[-_/.:]+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_QUOTED_TERM_RE = re.compile(r'"([^"]+)"')


def metadata_alias_text(values: Iterable[object]) -> str:
    """Return delimiter/case-normalized metadata variants."""

    variants: list[str] = []
    seen: set[str] = set()
    for value in values:
        raw = str(value or "").strip()
        if not raw:
            continue
        for variant in (raw, _split_alias(raw)):
            normalized = _normalize_space(variant).lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                variants.append(normalized)
    return " ".join(variants)


def metadata_compact_text(values: Iterable[object]) -> str:
    """Return short metadata with punctuation and spaces removed."""

    parts: list[str] = []
    seen: set[str] = set()
    for value in values:
        raw = str(value or "").strip()
        if not raw:
            continue
        for variant in (raw, _split_alias(raw)):
            compact = _compact(variant)
            if compact and compact not in seen:
                seen.add(compact)
                parts.append(compact)
    return " ".join(parts)


def quoted_query_terms(fts_query: str) -> list[str]:
    """Extract literal terms produced by the search query sanitizer."""

    terms = [term.strip().lower() for term in _QUOTED_TERM_RE.findall(fts_query)]
    if terms:
        return [term for term in terms if term]
    return [term.strip().lower() for term in fts_query.split() if term.strip()]


def compact_query_variants(term: str) -> tuple[str, ...]:
    """Return compact substring variants for low-boost metadata recall."""

    compact = _compact(term)
    if not compact:
        return ()
    variants = [compact]
    if len(compact) > 4 and compact.endswith("s"):
        variants.append(compact[:-1])
    return tuple(dict.fromkeys(variants))


def _split_alias(value: str) -> str:
    with_boundaries = _CAMEL_BOUNDARY_RE.sub(" ", value)
    return _DELIMITER_RE.sub(" ", with_boundaries)


def _compact(value: str) -> str:
    return _NON_ALNUM_RE.sub("", _split_alias(value).lower())


def _normalize_space(value: str) -> str:
    return " ".join(value.split())
