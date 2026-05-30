"""Deterministic document units for memory extraction.

Source genes normalize content into Markdown. This module turns that Markdown
into source-agnostic extraction units with stable ownership metadata.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field

__all__ = [
    "ExtractionContext",
    "ExtractionContextPacker",
    "ExtractionUnit",
    "UnitizationPolicy",
    "unitize_markdown",
]

SEGMENTATION_VERSION = "v2"
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_FENCE_RE = re.compile(r"^(```|~~~)")
_ACRONYM_RE = re.compile(r"\b[A-Z][A-Za-z0-9][A-Za-z0-9 /&.-]{1,80}\s+\([A-Z][A-Z0-9]{1,12}\)")
_DEFINITION_HEADINGS = {"glossary", "definitions", "definition", "terminology", "terms", "abbreviations"}


@dataclass(frozen=True)
class UnitizationPolicy:
    """Source-agnostic deterministic unitization policy."""

    max_unit_input_tokens: int = 20_000


@dataclass(frozen=True)
class ExtractionUnit:
    """One owned extraction unit within a normalized document."""

    doc_id: str
    unit_id: str
    path_id: str
    content_fingerprint: str
    segmentation_version: str
    unit_kind: str
    heading_path: tuple[str, ...]
    start_line: int
    end_line: int
    split_depth: int
    split_reason: str
    unit_markdown: str


@dataclass(frozen=True)
class ExtractionContext:
    """Prompt context for one extraction unit."""

    document_title: str
    document_url: str
    source_type: str
    unit: ExtractionUnit
    document_outline: str
    glossary_appendix: str
    entities: list[str] = field(default_factory=list)
    previous_heading: str | None = None
    next_heading: str | None = None


@dataclass
class _SectionNode:
    heading_path: tuple[str, ...]
    level: int
    start_line: int
    end_line: int
    lines: list[str]
    children: list["_SectionNode"] = field(default_factory=list)

    @property
    def markdown(self) -> str:
        return "\n".join(self.lines).strip()


class ExtractionContextPacker:
    """Build deterministic read-only context around an extraction unit."""

    def pack(
        self,
        *,
        document_title: str,
        document_url: str,
        source_type: str,
        unit: ExtractionUnit,
        all_units: list[ExtractionUnit],
        entities: list[str] | None = None,
    ) -> ExtractionContext:
        return ExtractionContext(
            document_title=document_title,
            document_url=document_url,
            source_type=source_type,
            unit=unit,
            document_outline=_outline_from_units(all_units),
            glossary_appendix=_glossary_from_units(all_units),
            entities=entities or [],
        )


def unitize_markdown(
    markdown: str,
    *,
    doc_id: str = "document",
    policy: UnitizationPolicy | None = None,
) -> list[ExtractionUnit]:
    """Return deterministic extraction units for normalized Markdown."""
    policy = policy or UnitizationPolicy()
    text = markdown.strip()
    if not text:
        return []

    root = _parse_section_tree(text)
    occurrence_counts: dict[tuple[str, ...], int] = {}
    units = _partition_node(
        root,
        doc_id=doc_id,
        policy=policy,
        occurrence_counts=occurrence_counts,
        is_root=True,
    )
    return sorted(units, key=lambda unit: (unit.start_line, unit.end_line, unit.unit_id))


def _parse_section_tree(markdown: str) -> _SectionNode:
    lines = markdown.splitlines()
    headings: list[tuple[int, str, int]] = []
    in_fence = False
    for index, line in enumerate(lines, start=1):
        if _FENCE_RE.match(line.strip()):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = _HEADING_RE.match(line)
        if match:
            headings.append((len(match.group(1)), match.group(2).strip(), index))

    root = _SectionNode(
        heading_path=("Document",),
        level=0,
        start_line=1,
        end_line=len(lines),
        lines=lines,
    )
    if not headings:
        return root

    node_stack: list[_SectionNode] = [root]
    for pos, (level, title, start_line) in enumerate(headings):
        while node_stack and node_stack[-1].level >= level:
            node_stack.pop()
        parent = node_stack[-1] if node_stack else root
        heading_path = (title,) if parent is root else (*parent.heading_path, title)
        next_same_or_higher = len(lines)
        for next_level, _next_title, next_line in headings[pos + 1 :]:
            if next_level <= level:
                next_same_or_higher = next_line - 1
                break
        node = _SectionNode(
            heading_path=heading_path,
            level=level,
            start_line=start_line,
            end_line=next_same_or_higher,
            lines=lines[start_line - 1 : next_same_or_higher],
        )
        parent.children.append(node)
        node_stack.append(node)
    return root


def _partition_node(
    node: _SectionNode,
    *,
    doc_id: str,
    policy: UnitizationPolicy,
    occurrence_counts: dict[tuple[str, ...], int],
    is_root: bool = False,
) -> list[ExtractionUnit]:
    markdown = node.markdown
    if _fits_token_budget(markdown, policy):
        return _make_units(
            markdown,
            doc_id=doc_id,
            heading_path=_root_unit_heading_path(node) if is_root else node.heading_path,
            start_line=node.start_line,
            policy=policy,
            split_depth=node.level,
            split_reason="whole_document_fits_budget" if is_root else "fits_section_subtree",
            unit_kind="content",
            occurrence_counts=occurrence_counts,
        )

    if not node.children:
        return _fallback_units(
            markdown,
            doc_id=doc_id,
            heading_path=_root_unit_heading_path(node) if is_root else node.heading_path,
            start_line=node.start_line,
            policy=policy,
            split_depth=node.level,
            split_reason="overflow",
            unit_kind="overflow",
            occurrence_counts=occurrence_counts,
        )

    units: list[ExtractionUnit] = []
    preamble = _preamble_markdown(node)
    if preamble:
        units.extend(
            _fallback_units(
                preamble,
                doc_id=doc_id,
                heading_path=_root_unit_heading_path(node) if is_root else node.heading_path,
                start_line=node.start_line,
                policy=policy,
                split_depth=node.level,
                split_reason="preamble" if _fits_token_budget(preamble, policy) else "overflow",
                unit_kind="preamble",
                occurrence_counts=occurrence_counts,
            )
        )

    for child in node.children:
        units.extend(
            _partition_node(
                child,
                doc_id=doc_id,
                policy=policy,
                occurrence_counts=occurrence_counts,
            )
        )
    return units


def _root_unit_heading_path(node: _SectionNode) -> tuple[str, ...]:
    if node.children and len(node.children) == 1:
        return node.children[0].heading_path[:1]
    return node.heading_path


def _fits_token_budget(markdown: str, policy: UnitizationPolicy) -> bool:
    return _estimate_tokens(markdown) <= policy.max_unit_input_tokens


def _estimate_tokens(text: str) -> int:
    if not text or not text.strip():
        return 0
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return max(1, int(len(text.split()) * 1.33))


def _preamble_markdown(section: _SectionNode) -> str:
    lines = section.lines
    if not lines or not section.children:
        return ""
    first_child_offset = section.children[0].start_line - section.start_line
    if first_child_offset <= 0:
        return ""
    body = lines[:first_child_offset]
    in_fence = False
    content_lines: list[str] = []
    for line in body[1:] if section.level > 0 else body:
        if _FENCE_RE.match(line.strip()):
            in_fence = not in_fence
            continue
        if in_fence:
            content_lines.append(line)
            continue
        content_lines.append(line)
    content_without_heading = "\n".join(content_lines).strip()
    return "\n".join(body).strip() if content_without_heading else ""


def _make_units(
    markdown: str,
    *,
    doc_id: str,
    heading_path: tuple[str, ...],
    start_line: int,
    policy: UnitizationPolicy,
    split_depth: int,
    split_reason: str,
    unit_kind: str,
    occurrence_counts: dict[tuple[str, ...], int],
) -> list[ExtractionUnit]:
    content = markdown.strip()
    if _fits_token_budget(content, policy):
        return [
            _build_unit(
                content,
                doc_id=doc_id,
                heading_path=heading_path,
                start_line=start_line,
                split_depth=split_depth,
                split_reason=split_reason,
                unit_kind=unit_kind,
                occurrence_counts=occurrence_counts,
            )
        ]

    return _fallback_units(
        content,
        doc_id=doc_id,
        heading_path=heading_path,
        start_line=start_line,
        policy=policy,
        split_depth=split_depth,
        split_reason="overflow",
        unit_kind="overflow",
        occurrence_counts=occurrence_counts,
    )


def _fallback_units(
    markdown: str,
    *,
    doc_id: str,
    heading_path: tuple[str, ...],
    start_line: int,
    policy: UnitizationPolicy,
    split_depth: int,
    split_reason: str,
    unit_kind: str = "content",
    occurrence_counts: dict[tuple[str, ...], int] | None = None,
) -> list[ExtractionUnit]:
    occurrence_counts = occurrence_counts if occurrence_counts is not None else {}
    parts = _split_safe_blocks(markdown, policy.max_unit_input_tokens)
    units: list[ExtractionUnit] = []
    line_cursor = start_line
    for part in parts:
        units.append(
            _build_unit(
                part,
                doc_id=doc_id,
                heading_path=heading_path,
                start_line=line_cursor,
                split_depth=split_depth,
                split_reason=split_reason,
                unit_kind=unit_kind,
                occurrence_counts=occurrence_counts,
            )
        )
        line_cursor += max(1, part.count("\n") + 1)
    return units


def _split_safe_blocks(markdown: str, max_tokens: int) -> list[str]:
    blocks = re.split(r"\n\s*\n", markdown.strip())
    parts: list[str] = []
    current: list[str] = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if _estimate_tokens(block) > max_tokens:
            if current:
                parts.append("\n\n".join(current))
                current = []
            parts.extend(_split_oversized_block(block, max_tokens))
            continue
        candidate = "\n\n".join([*current, block]) if current else block
        if current and _estimate_tokens(candidate) > max_tokens:
            parts.append("\n\n".join(current))
            current = [block]
        else:
            current.append(block)
    if current:
        parts.append("\n\n".join(current))
    return [part for part in parts if part.strip()]


def _split_oversized_block(block: str, max_tokens: int) -> list[str]:
    words = block.split()
    if not words:
        return [block]
    parts: list[str] = []
    current: list[str] = []
    for word in words:
        if _estimate_tokens(word) > max_tokens:
            if current:
                parts.append(" ".join(current))
                current = []
            parts.extend(_split_oversized_word(word, max_tokens))
            continue
        candidate_words = [*current, word]
        candidate = " ".join(candidate_words)
        if current and _estimate_tokens(candidate) > max_tokens:
            parts.append(" ".join(current))
            current = [word]
        else:
            current = candidate_words
    if current:
        parts.append(" ".join(current))
    return parts


def _split_oversized_word(word: str, max_tokens: int) -> list[str]:
    chunk_chars = max(1, max_tokens * 4)
    return [word[index : index + chunk_chars] for index in range(0, len(word), chunk_chars)]


def _build_unit(
    markdown: str,
    *,
    doc_id: str,
    heading_path: tuple[str, ...],
    start_line: int,
    split_depth: int,
    split_reason: str,
    unit_kind: str,
    occurrence_counts: dict[tuple[str, ...], int],
) -> ExtractionUnit:
    occurrence = occurrence_counts.get(heading_path, 0) + 1
    occurrence_counts[heading_path] = occurrence
    base_path_id = "__".join(_slug(part) for part in heading_path if _slug(part)) or "document"
    path_id = base_path_id if occurrence == 1 else f"{base_path_id}__{occurrence}"
    content = markdown.strip()
    return ExtractionUnit(
        doc_id=doc_id,
        unit_id=f"{doc_id}::{path_id}",
        path_id=path_id,
        content_fingerprint=_fingerprint(content),
        segmentation_version=SEGMENTATION_VERSION,
        unit_kind=unit_kind,
        heading_path=heading_path,
        start_line=start_line,
        end_line=start_line + max(0, content.count("\n")),
        split_depth=split_depth,
        split_reason=split_reason,
        unit_markdown=content,
    )


def _outline_from_units(units: list[ExtractionUnit]) -> str:
    seen: set[tuple[str, ...]] = set()
    lines: list[str] = []
    for unit in sorted(units, key=lambda item: (item.start_line, item.heading_path)):
        for depth in range(1, len(unit.heading_path) + 1):
            path = unit.heading_path[:depth]
            if path in seen:
                continue
            seen.add(path)
            lines.append(f"{'  ' * (depth - 1)}{'#' * min(depth, 6)} {path[-1]}")
    return "\n".join(lines)


def _glossary_from_units(units: list[ExtractionUnit], *, max_chars: int = 2_000) -> str:
    snippets: list[str] = []
    for unit in units:
        heading_names = {_slug(part) for part in unit.heading_path}
        heading_is_definition = bool(heading_names & _DEFINITION_HEADINGS)
        for line in unit.unit_markdown.splitlines():
            stripped = line.strip()
            if not stripped or _HEADING_RE.match(stripped):
                continue
            if heading_is_definition or _ACRONYM_RE.search(stripped):
                if stripped not in snippets:
                    snippets.append(stripped)
            if len("\n".join(snippets)) >= max_chars:
                return "\n".join(snippets)[:max_chars]
    return "\n".join(snippets)[:max_chars]


def _fingerprint(text: str) -> str:
    normalized = re.sub(r"[ \t]+", " ", text.strip())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _slug(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", normalized.lower()).strip("-")
    return slug or "section"
