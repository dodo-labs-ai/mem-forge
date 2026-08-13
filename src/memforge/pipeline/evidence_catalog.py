"""Batch-local, provider-neutral addresses for textual extraction Evidence."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace

from memforge.models import RawMemory

__all__ = [
    "EVIDENCE_BLOCK_MAX_BYTES",
    "EvidenceAuthoritySpan",
    "EvidenceBlock",
    "EvidenceCatalog",
    "EvidenceResolution",
]


EVIDENCE_BLOCK_MAX_BYTES = 4_096
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]*)\]\((\S+?)(?:\s+[\"'][^\"']*[\"'])?\)")
_BLANK_LINE_RE = re.compile(r"\r?\n[ \t]*\r?\n+")
_MARKDOWN_ESCAPABLE = frozenset(r"\\`*_{}[]<>()#+-.!|")
_MARKDOWN_ESCAPE_PREFIXES = frozenset({"\\", "＼"})


@dataclass(frozen=True, slots=True)
class EvidenceAuthoritySpan:
    """One selectable source span in the current extraction work."""

    text: str
    observation_id: str | None = None
    source_start: int = 0


@dataclass(frozen=True, slots=True)
class EvidenceBlock:
    """One bounded source-derived block addressable inside a single prompt."""

    id: str
    text: str
    observation_id: str | None
    source_start: int
    source_end: int


@dataclass(frozen=True, slots=True)
class EvidenceResolution:
    """Canonical source text selected by one valid block address."""

    memory: RawMemory
    block: EvidenceBlock
    refinement: str


class EvidenceCatalog:
    """Resolve model-selected block addresses to exact current-source excerpts."""

    def __init__(self, blocks: tuple[EvidenceBlock, ...]) -> None:
        self.blocks = blocks
        self._by_id = {block.id: block for block in blocks}
        if len(self._by_id) != len(blocks):
            raise ValueError("Evidence Block IDs must be unique inside one catalog")

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        observation_id: str | None = None,
        source_start: int = 0,
        max_block_bytes: int = EVIDENCE_BLOCK_MAX_BYTES,
    ) -> EvidenceCatalog:
        return cls.from_spans(
            (
                EvidenceAuthoritySpan(
                    text=text,
                    observation_id=observation_id,
                    source_start=source_start,
                ),
            ),
            max_block_bytes=max_block_bytes,
        )

    @classmethod
    def from_spans(
        cls,
        spans: tuple[EvidenceAuthoritySpan, ...],
        *,
        max_block_bytes: int = EVIDENCE_BLOCK_MAX_BYTES,
    ) -> EvidenceCatalog:
        if max_block_bytes < 4:
            raise ValueError("Evidence Block byte limit must fit one UTF-8 character")
        pending: list[tuple[str, str | None, int, int]] = []
        for span in spans:
            for start, end in _paragraph_ranges(span.text):
                for chunk_start, chunk_end in _bounded_ranges(
                    span.text,
                    start=start,
                    end=end,
                    max_bytes=max_block_bytes,
                ):
                    pending.append(
                        (
                            span.text[chunk_start:chunk_end],
                            span.observation_id,
                            span.source_start + chunk_start,
                            span.source_start + chunk_end,
                        )
                    )
        blocks = tuple(
            EvidenceBlock(
                id=f"EB-{index:03d}",
                text=text,
                observation_id=observation_id,
                source_start=start,
                source_end=end,
            )
            for index, (text, observation_id, start, end) in enumerate(
                pending,
                start=1,
            )
        )
        return cls(blocks)

    def render(self) -> str:
        """Render only selectable source text plus transient prompt addresses."""

        return "\n\n".join(
            (
                f'<evidence_block id="{block.id}"'
                + (
                    f' source_observation_id="{block.observation_id}"'
                    if block.observation_id
                    else ""
                )
                + f">\n{block.text}\n</evidence_block>"
            )
            for block in self.blocks
        )

    def resolve(self, memory: RawMemory) -> EvidenceResolution | None:
        """Resolve one candidate; quote mismatch lowers precision, never admission."""

        block_id = (memory.evidence_block_id or "").strip()
        block = self._by_id.get(block_id) if block_id else self._resolve_legacy_quote(memory)
        if block is None:
            return None
        refined = _localize_quote_range(block.text, memory.evidence_quote)
        if refined is None:
            quote = block.text
            refinement = "block_fallback"
            relative_start, relative_end = 0, len(block.text)
        else:
            quote, refinement, relative_start, relative_end = refined
        return EvidenceResolution(
            memory=replace(
                memory,
                extraction_context=quote,
                evidence_quote=quote,
                evidence_block_id=None,
                evidence_resolved_from_block=True,
                evidence_range_start=block.source_start + relative_start,
                evidence_range_end=block.source_start + relative_end,
                source_observation_id=block.observation_id or memory.source_observation_id,
            ),
            block=block,
            refinement=refinement,
        )

    def _resolve_legacy_quote(self, memory: RawMemory) -> EvidenceBlock | None:
        """Temporarily admit old quote-only responses only when ownership is unique."""

        quote = memory.evidence_quote
        if not quote or not quote.strip():
            return None
        known_observation_ids = {
            block.observation_id
            for block in self.blocks
            if block.observation_id is not None
        }
        # A valid current-batch hint may disambiguate repeated legacy quotes.
        # A stale or out-of-scope hint remains non-authoritative and is ignored.
        observation_filter = (
            memory.source_observation_id
            if memory.source_observation_id in known_observation_ids
            else None
        )
        candidates = tuple(
            block
            for block in self.blocks
            if (
                observation_filter is None
                or block.observation_id == observation_filter
            )
            and localize_quote(block.text, quote) is not None
        )
        return candidates[0] if len(candidates) == 1 else None


def _paragraph_ranges(text: str) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for separator in _BLANK_LINE_RE.finditer(text):
        _append_trimmed_range(ranges, text, cursor, separator.start())
        cursor = separator.end()
    _append_trimmed_range(ranges, text, cursor, len(text))
    return tuple(ranges)


def _append_trimmed_range(
    ranges: list[tuple[int, int]],
    text: str,
    start: int,
    end: int,
) -> None:
    # The blank-line separator itself is excluded, but horizontal whitespace
    # inside the source paragraph remains part of its exact revision range.
    if start < end and text[start:end].strip():
        ranges.append((start, end))


def _bounded_ranges(
    text: str,
    *,
    start: int,
    end: int,
    max_bytes: int,
) -> tuple[tuple[int, int], ...]:
    if len(text[start:end].encode("utf-8")) <= max_bytes:
        return ((start, end),)
    ranges: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        limit = _utf8_end(text, cursor, end, max_bytes)
        if limit < end:
            newline = text.rfind("\n", cursor + 1, limit + 1)
            if newline > cursor:
                limit = newline
        chunk_start, chunk_end = cursor, limit
        while chunk_start < chunk_end and text[chunk_start].isspace():
            chunk_start += 1
        while chunk_end > chunk_start and text[chunk_end - 1].isspace():
            chunk_end -= 1
        if chunk_start < chunk_end:
            ranges.append((chunk_start, chunk_end))
        cursor = max(limit, cursor + 1)
        while cursor < end and text[cursor].isspace():
            cursor += 1
    return tuple(ranges)


def _utf8_end(text: str, start: int, end: int, max_bytes: int) -> int:
    used = 0
    cursor = start
    while cursor < end:
        size = len(text[cursor].encode("utf-8"))
        if used + size > max_bytes:
            break
        used += size
        cursor += 1
    return max(cursor, start + 1)


def localize_quote(authority_text: str, proposed: str | None) -> tuple[str, str] | None:
    """Map a semantically unchanged quote representation to exact source text."""

    localized = _localize_quote_range(authority_text, proposed)
    return localized[:2] if localized is not None else None


def _localize_quote_range(
    authority_text: str,
    proposed: str | None,
) -> tuple[str, str, int, int] | None:
    """Return exact source text plus its coordinates inside one authority span."""

    raw_quote = proposed or ""
    if not raw_quote.strip():
        return None
    exact_start = authority_text.find(raw_quote)
    if exact_start >= 0:
        return (
            authority_text[exact_start : exact_start + len(raw_quote)],
            "exact_quote",
            exact_start,
            exact_start + len(raw_quote),
        )
    quote = raw_quote.strip()
    for link_mode in ("label", "url", "expanded"):
        block_view, block_map = _canonical_view(authority_text, link_mode=link_mode)
        quote_view, _ = _canonical_view(quote, link_mode=link_mode)
        if not quote_view:
            continue
        offsets = _unique_match(block_view, quote_view)
        if offsets is None:
            continue
        start, end = offsets
        original_start = block_map[start]
        original_end = block_map[end - 1] + 1
        return (
            authority_text[original_start:original_end],
            "canonical_quote",
            original_start,
            original_end,
        )
    return None


def _unique_match(haystack: str, needle: str) -> tuple[int, int] | None:
    start = haystack.find(needle)
    if start < 0 or haystack.find(needle, start + 1) >= 0:
        return None
    return start, start + len(needle)


def _canonical_view(text: str, *, link_mode: str) -> tuple[str, tuple[int, ...]]:
    output: list[str] = []
    mapping: list[int] = []
    cursor = 0
    for match in _MARKDOWN_LINK_RE.finditer(text):
        _append_canonical_text(output, mapping, text, cursor, match.start())
        if link_mode == "expanded":
            label_start, label_end = match.span(1)
            url_start, url_end = match.span(2)
            output_start = len(output)
            _append_canonical_text(output, mapping, text, label_start, label_end)
            if len(output) > output_start:
                mapping[output_start] = match.start()
            output.extend((" ", "("))
            mapping.extend((match.start(), match.start()))
            _append_canonical_text(output, mapping, text, url_start, url_end)
            output.append(")")
            mapping.append(match.end() - 1)
        else:
            group = 1 if link_mode == "label" else 2
            group_start, group_end = match.span(group)
            _append_canonical_text(output, mapping, text, group_start, group_end)
        cursor = match.end()
    _append_canonical_text(output, mapping, text, cursor, len(text))
    canonical = "".join(output).strip()
    if not canonical:
        return "", ()
    leading = next(index for index, char in enumerate(output) if not char.isspace())
    trailing = len(output) - next(index for index, char in enumerate(reversed(output)) if not char.isspace())
    return canonical, tuple(mapping[leading:trailing])


def _append_canonical_text(
    output: list[str],
    mapping: list[int],
    text: str,
    start: int,
    end: int,
) -> None:
    cursor = start
    while cursor < end:
        char = text[cursor]
        if (
            char in _MARKDOWN_ESCAPE_PREFIXES
            and cursor + 1 < end
            and text[cursor + 1] in _MARKDOWN_ESCAPABLE
        ):
            cursor += 1
            char = text[cursor]
        elif char == "`":
            cursor += 1
            continue
        elif text[cursor : cursor + 2] in {"**", "__"}:
            cursor += 2
            continue
        normalized = unicodedata.normalize("NFKC", char)
        normalized = normalized.translate(
            str.maketrans(
                {
                    "‘": "'",
                    "’": "'",
                    "“": '"',
                    "”": '"',
                    "‐": "-",
                    "‑": "-",
                    "–": "-",
                    "—": "-",
                    "−": "-",
                }
            )
        )
        for value in normalized:
            if value.isspace():
                if output and output[-1] != " ":
                    output.append(" ")
                    mapping.append(cursor)
            else:
                output.append(value)
                mapping.append(cursor)
        cursor += 1
