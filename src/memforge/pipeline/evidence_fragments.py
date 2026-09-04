"""Deterministic, revision-pinned Evidence Fragment compilation.

The compiler owns representation parsing and transient catalog references.  It
does not create durable Evidence, call a model, or make lifecycle decisions.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, field, replace
from enum import Enum
from html.parser import HTMLParser
from typing import Callable, Mapping, Sequence

from markdown_it import MarkdownIt
from markdown_it.rules_inline.html_inline import html_inline as _markdown_it_html_inline

from memforge.memory.evidence import EvidenceRole
from memforge.source_artifacts import source_artifact_inference_eligibility
from memforge.source_projection import (
    AnchorKind,
    EvidenceCoordinateSpace,
    EvidenceRepresentationProfile,
    SourceAnchor,
    SourceObservationRevision,
)
from memforge.source_representation import (
    MARKDOWN_STRUCTURAL_PROFILE,
    PLAIN_TEXT_PROFILE,
    CanonicalRecordField,
    CanonicalRecordSchema,
    EvidenceRepresentationContract,
    canonical_field_comparison_value,
    representation_contract_for_profile,
)


COMPILER_CONTRACT_VERSION = 3
DEFAULT_MAX_FRAGMENTS = 2_048
DEFAULT_MAX_PRESENTATION_CHARS = 120_000
_SUPPORTING_ROLES = frozenset({EvidenceRole.PRIMARY, EvidenceRole.REQUIRED})


def _record_html_inline_source_range(state, silent: bool) -> bool:
    start = state.pos
    previous_count = len(state.tokens)
    matched = _markdown_it_html_inline(state, silent)
    if matched and not silent and len(state.tokens) > previous_count:
        state.tokens[-1].meta["source_range"] = (start, state.pos)
    return matched


def _markdown_parser() -> MarkdownIt:
    parser = MarkdownIt("commonmark").enable("table")
    parser.inline.ruler.at("html_inline", _record_html_inline_source_range)
    return parser


class EvidenceFragmentKind(str, Enum):
    TEXT = "text"
    ARTIFACT = "artifact"


class FragmentCompilationErrorCode(str, Enum):
    UNSUPPORTED_PROFILE = "unsupported_profile"
    INVALID_AUTHORITY_RANGE = "invalid_authority_range"
    MALFORMED_MARKDOWN = "malformed_markdown"
    MALFORMED_HTML = "malformed_html"
    UNSUPPORTED_HTML = "unsupported_html"
    MALFORMED_CANONICAL_RECORD = "malformed_canonical_record"
    SCHEMA_MISMATCH = "schema_mismatch"
    ARTIFACT_INELIGIBLE = "artifact_ineligible"
    NO_SELECTABLE_CONTENT = "no_selectable_content"
    CATALOG_TOO_LARGE = "catalog_too_large"


@dataclass(frozen=True, slots=True)
class StructuralUnit:
    """One complete representation-owned range safe for Planner authority."""

    start: int
    end: int


@dataclass(frozen=True, slots=True)
class CanonicalFieldRange:
    """One registered canonical field in immutable raw-Revision coordinates."""

    descriptor: CanonicalRecordField
    start: int
    end: int
    value: object
    comparison_value: object
    string_boundaries: tuple[int, ...] | None = None


class StructuralUnitTooLargeError(ValueError):
    """One complete structure cannot fit the configured presentation budget."""

    code = "structural_unit_too_large"

    def __init__(self, *, revision_id: str, start: int, end: int, budget: int) -> None:
        super().__init__(
            f"structural unit exceeds presentation budget: {revision_id}:{start}:{end}"
        )
        self.revision_id = revision_id
        self.start = start
        self.end = end
        self.budget = budget


def plan_revision_structural_units(
    revision: SourceObservationRevision,
    *,
    max_content_chars: int,
) -> tuple[StructuralUnit, ...]:
    """Pack complete representation structures without granting authority."""

    if max_content_chars < 1:
        raise ValueError("structural planning budget must be positive")
    protected = tuple(
        (unit.start, unit.end) for unit in revision_structural_ranges(revision)
    )
    packed: list[StructuralUnit] = []
    current_start: int | None = None
    current_end: int | None = None
    for start, end in protected:
        if end - start > max_content_chars:
            raise StructuralUnitTooLargeError(
                revision_id=revision.id,
                start=start,
                end=end,
                budget=max_content_chars,
            )
        if current_start is None:
            current_start, current_end = start, end
            continue
        assert current_end is not None
        if end - current_start <= max_content_chars:
            current_end = end
            continue
        packed.append(StructuralUnit(current_start, current_end))
        current_start, current_end = start, end
    if current_start is not None and current_end is not None:
        packed.append(StructuralUnit(current_start, current_end))
    return tuple(packed)


def revision_structural_ranges(
    revision: SourceObservationRevision,
) -> tuple[StructuralUnit, ...]:
    """Return the smallest representation-owned ranges safe for authority.

    Presentation packing may combine adjacent units later, but it must never
    use that packing decision to widen Primary authority.
    """

    profile = revision.evidence_profile
    if (
        profile is None
        or profile.requires_whole_observation_authority
        or profile.name not in {"markdown-structural", "plain-text"}
    ):
        return (
            (StructuralUnit(0, len(revision.content)),)
            if revision.content
            else ()
        )
    protected = (
        _markdown_protected_ranges(revision.content)
        if profile.name == "markdown-structural"
        else _paragraph_ranges(revision.content)
    )
    if not protected and revision.content:
        protected = ((0, len(revision.content)),)
    return tuple(StructuralUnit(start, end) for start, end in protected)


def revision_changed_structural_ranges(
    base: SourceObservationRevision,
    target: SourceObservationRevision,
) -> tuple[tuple[int, int], ...]:
    """Return target structures not provably unchanged from the committed base."""

    if (
        base.observation_id != target.observation_id
        or base.evidence_profile != target.evidence_profile
        or target.evidence_profile is None
        or target.evidence_profile.name
        not in {"markdown-structural", "plain-text"}
    ):
        raise ValueError("range-addressable Revision pair has incompatible identity")
    base_units = revision_structural_ranges(base)
    target_units = revision_structural_ranges(target)
    remaining_unchanged = Counter(
        _revision_structural_identities(base, base_units)
    )
    changed = []
    target_identities = _revision_structural_identities(target, target_units)
    for unit, identity in zip(target_units, target_identities, strict=True):
        if remaining_unchanged[identity] > 0:
            remaining_unchanged[identity] -= 1
            continue
        changed.append((unit.start, unit.end))
    return tuple(changed)


def _revision_structural_identities(
    revision: SourceObservationRevision,
    units: tuple[StructuralUnit, ...],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Bind Markdown structures to their deterministic heading ancestry."""

    profile = revision.evidence_profile
    if profile is None or profile.name != "markdown-structural":
        return tuple(
            (revision.content[unit.start : unit.end], ()) for unit in units
        )
    tokens = _markdown_parser().disable("inline").parse(revision.content)
    line_starts = _line_starts(revision.content)
    headings: dict[int, tuple[int, str]] = {}
    for token in tokens:
        if token.type != "heading_open" or token.map is None:
            continue
        start, end = _token_range(
            revision.content,
            line_starts,
            token.map[0],
            token.map[1],
        )
        level = int(token.tag.removeprefix("h"))
        headings[start] = (level, revision.content[start:end])
    path: list[str] = []
    identities = []
    for unit in units:
        identities.append(
            (
                revision.content[unit.start : unit.end],
                tuple(path),
            )
        )
        heading = headings.get(unit.start)
        if heading is None:
            continue
        level, heading_text = heading
        del path[level - 1 :]
        while len(path) < level - 1:
            path.append("")
        path.append(heading_text)
    return tuple(identities)


@dataclass(frozen=True, slots=True)
class FragmentCompilationError:
    code: FragmentCompilationErrorCode
    observation_revision_id: str
    message: str
    range_start: int | None = None
    range_end: int | None = None
    fatal: bool = False


@dataclass(frozen=True, slots=True)
class EvidenceCandidateRange:
    """One exact candidate Anchor and whether current work permits Primary."""

    anchor: SourceAnchor
    primary_eligible: bool

    @property
    def eligible_roles(self) -> frozenset[EvidenceRole]:
        return (
            _SUPPORTING_ROLES
            if self.primary_eligible
            else frozenset({EvidenceRole.REQUIRED})
        )


@dataclass(frozen=True, slots=True)
class EvidenceFragment:
    reference: str
    kind: EvidenceFragmentKind
    fragment_type: str
    anchor: SourceAnchor
    primary_eligible: bool
    raw_content_sha256: str
    presentation_text: str
    presentation_sha256: str


@dataclass(frozen=True, slots=True)
class EvidenceFragmentCatalog:
    observation_revision_id: str
    profile: EvidenceRepresentationProfile | None
    fragments: tuple[EvidenceFragment, ...]
    errors: tuple[FragmentCompilationError, ...]
    digest: str
    max_fragments: int
    max_presentation_chars: int

    @property
    def usable(self) -> bool:
        return bool(self.fragments) and not any(error.fatal for error in self.errors)

    def resolve(self, reference: str, role: EvidenceRole) -> EvidenceFragment | None:
        """Resolve one catalog-local selector without widening its authority."""

        if role not in {EvidenceRole.PRIMARY, EvidenceRole.REQUIRED}:
            return None
        for fragment in self.fragments:
            if fragment.reference == reference:
                if role is EvidenceRole.PRIMARY:
                    return fragment if fragment.primary_eligible else None
                return fragment
        return None

    def model_payload(self) -> tuple[Mapping[str, object], ...]:
        """Return the bounded selector view; authoritative raw text stays in the Revision."""

        return tuple(
            {
                "ref": fragment.reference,
                "kind": fragment.kind.value,
                "type": fragment.fragment_type,
                "text": fragment.presentation_text,
                "primary_eligible": fragment.primary_eligible,
            }
            for fragment in self.fragments
        )


@dataclass(frozen=True, slots=True)
class _FragmentCandidate:
    kind: EvidenceFragmentKind
    fragment_type: str
    start: int | None
    end: int | None
    eligible_roles: frozenset[EvidenceRole]
    presentation_text: str
    raw_content_sha256: str | None = None

    @property
    def primary_eligible(self) -> bool:
        return EvidenceRole.PRIMARY in self.eligible_roles


@dataclass(frozen=True, slots=True)
class _InlineHTMLRegion:
    start: int
    end: int
    markdown_source: str
    html_tokens: tuple[tuple[int, int, str], ...]


def compile_fragments(
    revision: SourceObservationRevision,
    authority_ranges: tuple[EvidenceCandidateRange, ...],
    *,
    max_fragments: int = DEFAULT_MAX_FRAGMENTS,
    max_presentation_chars: int = DEFAULT_MAX_PRESENTATION_CHARS,
) -> EvidenceFragmentCatalog:
    """Compile one immutable Observation Revision under its declared profile."""

    profile = revision.evidence_profile
    contract = representation_contract_for_profile(profile)
    if contract is None:
        return _fatal_catalog(
            revision,
            profile,
            authority_ranges,
            FragmentCompilationErrorCode.UNSUPPORTED_PROFILE,
            "Observation Revision lacks a supported Evidence Representation Profile",
            max_fragments=max_fragments,
            max_presentation_chars=max_presentation_chars,
        )
    if max_fragments <= 0 or max_presentation_chars <= 0:
        raise ValueError("Fragment catalog limits must be positive")

    range_error = _validate_authority_ranges(revision, profile, authority_ranges)
    if range_error is not None:
        return _catalog_from_candidates(
            revision,
            profile,
            authority_ranges,
            (),
            (range_error,),
            max_fragments=max_fragments,
            max_presentation_chars=max_presentation_chars,
        )

    compiler = _PROFILE_COMPILERS.get((profile.name, profile.version))
    if compiler is None:
        return _fatal_catalog(
            revision,
            profile,
            authority_ranges,
            FragmentCompilationErrorCode.UNSUPPORTED_PROFILE,
            f"unsupported Evidence Representation Profile: {profile.name}/{profile.version}",
            max_fragments=max_fragments,
            max_presentation_chars=max_presentation_chars,
        )

    candidates, errors = compiler(revision, contract, authority_ranges)
    return _catalog_from_candidates(
        revision,
        profile,
        authority_ranges,
        candidates,
        errors,
        max_fragments=max_fragments,
        max_presentation_chars=max_presentation_chars,
    )


def _validate_authority_ranges(
    revision: SourceObservationRevision,
    profile: EvidenceRepresentationProfile,
    authority_ranges: tuple[EvidenceCandidateRange, ...],
) -> FragmentCompilationError | None:
    expected_coordinate_space = (
        EvidenceCoordinateSpace.WHOLE_ARTIFACT
        if profile.name == "binary-artifact"
        else EvidenceCoordinateSpace.UNICODE_SCALAR
    )
    if profile.coordinate_space is not expected_coordinate_space:
        return _error(
            revision,
            FragmentCompilationErrorCode.UNSUPPORTED_PROFILE,
            "Evidence Representation Profile declares an incompatible coordinate space",
            fatal=True,
        )
    if not authority_ranges:
        return _error(
            revision,
            FragmentCompilationErrorCode.INVALID_AUTHORITY_RANGE,
            "Fragment compilation requires at least one authority range",
            fatal=True,
        )
    if profile.name == "binary-artifact" and len(authority_ranges) != 1:
        return _error(
            revision,
            FragmentCompilationErrorCode.INVALID_AUTHORITY_RANGE,
            "binary Artifact compilation requires exactly one authority range",
            fatal=True,
        )
    spans: list[tuple[int, int]] = []
    for item in authority_ranges:
        anchor = item.anchor
        if anchor.observation_id != revision.observation_id or anchor.observation_revision_id != revision.id:
            return _error(
                revision,
                FragmentCompilationErrorCode.INVALID_AUTHORITY_RANGE,
                "authority range belongs to another Observation Revision",
                fatal=True,
            )
        if profile.coordinate_space is EvidenceCoordinateSpace.WHOLE_ARTIFACT:
            if anchor.kind is not AnchorKind.WHOLE_OBSERVATION:
                return _error(
                    revision,
                    FragmentCompilationErrorCode.INVALID_AUTHORITY_RANGE,
                    "binary Artifact authority must cover the whole Observation",
                    fatal=True,
                )
            continue
        if anchor.kind is AnchorKind.WHOLE_OBSERVATION:
            spans.append((0, len(revision.content)))
            continue
        if anchor.kind is not AnchorKind.REVISION_RANGE:
            return _error(
                revision,
                FragmentCompilationErrorCode.INVALID_AUTHORITY_RANGE,
                "derived Fragment compilation accepts only whole or revision-range Anchors",
                fatal=True,
            )
        assert anchor.range_start is not None and anchor.range_end is not None
        if anchor.range_end > len(revision.content):
            return _error(
                revision,
                FragmentCompilationErrorCode.INVALID_AUTHORITY_RANGE,
                "authority range exceeds immutable Revision content",
                start=anchor.range_start,
                end=anchor.range_end,
                fatal=True,
            )
        spans.append((anchor.range_start, anchor.range_end))
    spans.sort()
    if any(current_start < previous_end for (_, previous_end), (current_start, _) in zip(spans, spans[1:])):
        return _error(
            revision,
            FragmentCompilationErrorCode.INVALID_AUTHORITY_RANGE,
            "authority ranges must not overlap",
            fatal=True,
        )
    return None


def _compile_markdown_profile(
    revision: SourceObservationRevision,
    contract: EvidenceRepresentationContract,
    authority_ranges: tuple[EvidenceCandidateRange, ...],
) -> tuple[tuple[_FragmentCandidate, ...], tuple[FragmentCompilationError, ...]]:
    del contract
    candidates, errors = _markdown_candidates(
        revision,
        revision.content,
        base=0,
        roles=_SUPPORTING_ROLES,
    )
    bound, authority_errors = _bind_candidates_to_authority(
        revision,
        candidates,
        authority_ranges,
    )
    return bound, (*_errors_inside_authority(errors, authority_ranges, revision.content), *authority_errors)


def _compile_plain_text_profile(
    revision: SourceObservationRevision,
    contract: EvidenceRepresentationContract,
    authority_ranges: tuple[EvidenceCandidateRange, ...],
) -> tuple[tuple[_FragmentCandidate, ...], tuple[FragmentCompilationError, ...]]:
    del contract
    candidates: list[_FragmentCandidate] = []
    for start, end in _paragraph_ranges(revision.content):
        candidates.append(
            _text_candidate(
                revision.content,
                "plain-paragraph",
                start,
                end,
                _SUPPORTING_ROLES,
                revision.content[start:end],
            )
        )
    return _bind_candidates_to_authority(revision, tuple(candidates), authority_ranges)


def _compile_binary_artifact_profile(
    revision: SourceObservationRevision,
    contract: EvidenceRepresentationContract,
    authority_ranges: tuple[EvidenceCandidateRange, ...],
) -> tuple[tuple[_FragmentCandidate, ...], tuple[FragmentCompilationError, ...]]:
    del contract
    raw_artifact = revision.metadata.get("source_artifact")
    if not isinstance(raw_artifact, Mapping):
        return (), (
            _error(
                revision,
                FragmentCompilationErrorCode.ARTIFACT_INELIGIBLE,
                "binary Artifact Revision lacks authoritative metadata",
                fatal=True,
            ),
        )
    if source_artifact_inference_eligibility(revision.metadata) is not True:
        return (), (
            _error(
                revision,
                FragmentCompilationErrorCode.ARTIFACT_INELIGIBLE,
                "binary Artifact is not inference eligible",
                fatal=True,
            ),
        )
    digest = str(raw_artifact.get("sha256") or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        return (), (
            _error(
                revision,
                FragmentCompilationErrorCode.ARTIFACT_INELIGIBLE,
                "binary Artifact lacks a valid byte digest",
                fatal=True,
            ),
        )
    return (
        tuple(
            _FragmentCandidate(
                kind=EvidenceFragmentKind.ARTIFACT,
                fragment_type="binary-artifact",
                start=None,
                end=None,
                eligible_roles=item.eligible_roles,
                presentation_text="",
                raw_content_sha256=digest,
            )
            for item in authority_ranges
        ),
        (),
    )


def _compile_canonical_record_profile(
    revision: SourceObservationRevision,
    contract: EvidenceRepresentationContract,
    authority_ranges: tuple[EvidenceCandidateRange, ...],
) -> tuple[tuple[_FragmentCandidate, ...], tuple[FragmentCompilationError, ...]]:
    schema = contract.canonical_schema
    assert isinstance(schema, CanonicalRecordSchema)
    try:
        document = _JsonDocument.parse(revision.content)
    except ValueError as exc:
        return (), (
            _error(
                revision,
                FragmentCompilationErrorCode.MALFORMED_CANONICAL_RECORD,
                str(exc),
                fatal=True,
            ),
        )

    candidates: list[_FragmentCandidate] = []
    errors: list[FragmentCompilationError] = []
    for descriptor in schema.fields:
        node = document.nodes.get(descriptor.json_pointer)
        if node is None or node.value is None:
            continue
        if descriptor.nested_profile is None:
            candidates.append(
                _text_candidate(
                    revision.content,
                    "canonical-field",
                    node.start,
                    node.end,
                    _SUPPORTING_ROLES,
                    _canonical_value_presentation(node.value),
                )
            )
            continue
        if not isinstance(node.value, str) or node.string_boundaries is None:
            errors.append(
                _error(
                    revision,
                    FragmentCompilationErrorCode.SCHEMA_MISMATCH,
                    f"canonical field {descriptor.json_pointer or '/'} is not declared text",
                    start=node.start,
                    end=node.end,
                )
            )
            continue
        if descriptor.nested_profile == "plain-text":
            nested = tuple(
                _FragmentCandidate(
                    kind=EvidenceFragmentKind.TEXT,
                    fragment_type="plain-paragraph",
                    start=start,
                    end=end,
                    eligible_roles=_SUPPORTING_ROLES,
                    presentation_text=node.value[start:end],
                )
                for start, end in _paragraph_ranges(node.value)
            )
            nested_errors: tuple[FragmentCompilationError, ...] = ()
        else:
            nested, nested_errors = _markdown_candidates(
                revision,
                node.value,
                base=0,
                roles=_SUPPORTING_ROLES,
            )
        for candidate in nested:
            assert candidate.start is not None and candidate.end is not None
            raw_start = node.string_boundaries[candidate.start]
            raw_end = node.string_boundaries[candidate.end]
            candidates.append(
                _text_candidate(
                    revision.content,
                    f"canonical-{candidate.fragment_type}",
                    raw_start,
                    raw_end,
                    candidate.eligible_roles,
                    candidate.presentation_text,
                )
            )
        for nested_error in nested_errors:
            raw_start = (
                node.string_boundaries[nested_error.range_start]
                if nested_error.range_start is not None
                else node.start
            )
            raw_end = (
                node.string_boundaries[nested_error.range_end]
                if nested_error.range_end is not None
                else node.end
            )
            errors.append(
                _error(
                    revision,
                    nested_error.code,
                    nested_error.message,
                    start=raw_start,
                    end=raw_end,
                    fatal=nested_error.fatal,
                )
            )
    bound, authority_errors = _bind_candidates_to_authority(
        revision,
        tuple(candidates),
        authority_ranges,
    )
    return bound, (
        *_errors_inside_authority(tuple(errors), authority_ranges, revision.content),
        *authority_errors,
    )


_ProfileCompiler = Callable[
    [SourceObservationRevision, EvidenceRepresentationContract, tuple[EvidenceCandidateRange, ...]],
    tuple[tuple[_FragmentCandidate, ...], tuple[FragmentCompilationError, ...]],
]

_PROFILE_COMPILERS: Mapping[tuple[str, int], _ProfileCompiler] = {
    ("markdown-structural", 1): _compile_markdown_profile,
    ("canonical-record", 1): _compile_canonical_record_profile,
    ("plain-text", 1): _compile_plain_text_profile,
    ("binary-artifact", 1): _compile_binary_artifact_profile,
}


def _markdown_protected_ranges(text: str) -> tuple[tuple[int, int], ...]:
    """Return outermost CommonMark structures in one ordered parser pass."""

    try:
        # Planning consumes block coordinates only. Inline validation belongs to
        # Fragment compilation, where its tokens and source ranges are needed.
        tokens = _markdown_parser().disable("inline").parse(text)
    except Exception as exc:
        raise ValueError(
            f"Markdown parser rejected structural planning: {type(exc).__name__}"
        ) from exc
    line_starts = _line_starts(text)
    protected_types = {
        "html_block",
        "table_open",
        "list_item_open",
        "blockquote_open",
        "heading_open",
        "fence",
        "code_block",
        "paragraph_open",
    }
    candidates = []
    for token in tokens:
        if token.type not in protected_types or token.map is None:
            continue
        start, end = _token_range(
            text,
            line_starts,
            token.map[0],
            token.map[1],
        )
        if start < end:
            candidates.append((start, end))
    selected: list[tuple[int, int]] = []
    selected_end = -1
    for start, end in sorted(candidates, key=lambda item: (item[0], -item[1])):
        if start < selected_end:
            continue
        selected.append((start, end))
        selected_end = end
    return tuple(selected)


def _markdown_candidates(
    revision: SourceObservationRevision,
    text: str,
    *,
    base: int,
    roles: frozenset[EvidenceRole],
) -> tuple[tuple[_FragmentCandidate, ...], tuple[FragmentCompilationError, ...]]:
    try:
        tokens = _markdown_parser().parse(text)
    except Exception as exc:
        return (), (
            _error(
                revision,
                FragmentCompilationErrorCode.MALFORMED_MARKDOWN,
                f"Markdown parser rejected the authority range: {type(exc).__name__}",
                start=base,
                end=base + len(text),
            ),
        )
    line_starts = _line_starts(text)
    inline_html_regions: list[_InlineHTMLRegion] = []
    for token in tokens:
        if token.type != "inline" or token.map is None or not token.children:
            continue
        html_tokens = tuple(
            (
                int(child.meta["source_range"][0]),
                int(child.meta["source_range"][1]),
                child.content,
            )
            for child in token.children
            if child.type == "html_inline" and "source_range" in child.meta
        )
        if not html_tokens:
            continue
        inline_start, inline_end = _token_range(
            text,
            line_starts,
            token.map[0],
            token.map[1],
        )
        inline_html_regions.append(
            _InlineHTMLRegion(
                start=inline_start,
                end=inline_end,
                markdown_source=token.content,
                html_tokens=html_tokens,
            )
        )
    structural: list[tuple[int, int, int, str, bool]] = []
    token_specs = {
        "html_block": (0, "html-block", True),
        "list_item_open": (1, "markdown-list-item", False),
        "tr_open": (1, "markdown-table-row", False),
        "blockquote_open": (1, "markdown-blockquote", False),
        "heading_open": (2, "markdown-heading", False),
        "fence": (2, "markdown-code-block", False),
        "code_block": (2, "markdown-code-block", False),
        "paragraph_open": (3, "markdown-paragraph", False),
    }
    for token in tokens:
        spec = token_specs.get(token.type)
        if spec is None or token.map is None:
            continue
        start, end = _token_range(text, line_starts, token.map[0], token.map[1])
        if start >= end:
            continue
        priority, fragment_type, is_html = spec
        structural.append((priority, start, end, fragment_type, is_html))

    selected: list[tuple[int, int, str, bool]] = []
    for _, start, end, fragment_type, is_html in sorted(structural):
        if any(start < chosen_end and chosen_start < end for chosen_start, chosen_end, _, _ in selected):
            continue
        selected.append((start, end, fragment_type, is_html))

    candidates: list[_FragmentCandidate] = []
    errors: list[FragmentCompilationError] = []
    for start, end, fragment_type, is_html in sorted(selected):
        raw = text[start:end]
        if is_html:
            html_candidates, html_errors = _html_candidates(
                revision,
                raw,
                base=base + start,
                roles=roles,
            )
            candidates.extend(html_candidates)
            errors.extend(html_errors)
            continue
        presentation = raw
        owned_inline_regions = tuple(
            region for region in inline_html_regions if start <= region.start and region.end <= end
        )
        if owned_inline_regions:
            inline_candidate, inline_error = _inline_html_structural_candidate(
                revision,
                raw,
                base=base + start,
                roles=roles,
                fragment_type=("markdown-inline-html" if fragment_type == "markdown-paragraph" else fragment_type),
                inline_regions=owned_inline_regions,
            )
            if inline_error is not None:
                errors.append(inline_error)
                continue
            assert inline_candidate is not None
            candidates.append(inline_candidate)
            continue
        candidates.append(
            _text_candidate(
                revision.content,
                fragment_type,
                base + start,
                base + end,
                roles,
                presentation,
            )
        )
    return tuple(candidates), tuple(errors)


@dataclass(slots=True)
class _HtmlNode:
    tag: str
    start: int
    start_tag_end: int
    end: int | None = None
    children: list[_HtmlNode] = field(default_factory=list)
    unsafe: bool = False


class _OffsetHTMLParser(HTMLParser):
    _VOID_TAGS = frozenset(
        {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
    )

    def __init__(self, source: str) -> None:
        super().__init__(convert_charrefs=False)
        self.source = source
        self.line_starts = _line_starts(source)
        self.roots: list[_HtmlNode] = []
        self.stack: list[_HtmlNode] = []
        self.failure: str | None = None
        self.internal_failure = False

    def _offset(self) -> int:
        line, column = self.getpos()
        return self.line_starts[line - 1] + column

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        start = self._offset()
        raw = self.get_starttag_text() or ""
        node = _HtmlNode(
            tag=tag.lower(),
            start=start,
            start_tag_end=start + len(raw),
            unsafe=tag.lower() in {"script", "style"},
        )
        (self.stack[-1].children if self.stack else self.roots).append(node)
        if tag.lower() in self._VOID_TAGS:
            node.end = node.start_tag_end
        else:
            self.stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if self.stack and self.stack[-1].tag == tag.lower():
            self.stack[-1].end = self.stack[-1].start_tag_end
            self.stack.pop()

    def handle_endtag(self, tag: str) -> None:
        if not self.stack or self.stack[-1].tag != tag.lower():
            self.failure = f"mismatched HTML end tag: {tag}"
            return
        start = self._offset()
        end = self.source.find(">", start)
        if end < 0:
            self.failure = f"unterminated HTML end tag: {tag}"
            return
        node = self.stack.pop()
        node.end = end + 1

    def close_checked(self) -> None:
        try:
            self.feed(self.source)
            self.close()
        except Exception as exc:
            self.failure = f"HTML parser failure: {type(exc).__name__}"
            self.internal_failure = True
        if self.stack and self.failure is None:
            self.failure = f"unclosed HTML element: {self.stack[-1].tag}"


class _HTMLTextCollector(HTMLParser):
    _SEPARATOR_TAGS = frozenset(
        {"br", "p", "li", "tr", "td", "th", "blockquote", "pre", "div", "section", "article", "dt", "dd"}
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.parts: list[str] = []
        self.unsafe_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag.lower() in {"script", "style"}:
            self.unsafe_depth += 1
        elif not self.unsafe_depth and tag.lower() in self._SEPARATOR_TAGS:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style"} and self.unsafe_depth:
            self.unsafe_depth -= 1
        elif not self.unsafe_depth and tag.lower() in self._SEPARATOR_TAGS:
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        if not self.unsafe_depth:
            self.parts.append(data)

    def handle_entityref(self, name: str) -> None:
        if not self.unsafe_depth:
            self.parts.append(html.unescape(f"&{name};"))

    def handle_charref(self, name: str) -> None:
        if not self.unsafe_depth:
            self.parts.append(html.unescape(f"&#{name};"))


_HTML_SEMANTIC_TAGS = frozenset(
    {"p", "li", "tr", "blockquote", "pre", "figcaption", "dt", "dd", "h1", "h2", "h3", "h4", "h5", "h6"}
)
_HTML_TOKEN_TAG_NAME_RE = re.compile(r"^<\s*/?\s*([A-Za-z][A-Za-z0-9-]*)\b")


def _inline_html_structural_candidate(
    revision: SourceObservationRevision,
    source: str,
    *,
    base: int,
    roles: frozenset[EvidenceRole],
    fragment_type: str,
    inline_regions: tuple[_InlineHTMLRegion, ...],
) -> tuple[_FragmentCandidate | None, FragmentCompilationError | None]:
    translated_tokens: list[tuple[str, tuple[int, ...]]] = []
    inline_cursor = 0
    for region in sorted(inline_regions, key=lambda item: (item.start, item.end)):
        boundary_map = _inline_source_boundary_map(
            source,
            region.markdown_source,
            start_cursor=inline_cursor,
        )
        if boundary_map is None:
            return None, _error(
                revision,
                FragmentCompilationErrorCode.MALFORMED_HTML,
                "CommonMark inline source cannot be mapped to the exact structural range",
                start=base,
                end=base + len(source),
            )
        translated_tokens.extend(
            (
                raw_tag,
                tuple(boundary_map[index] for index in range(start, end)),
            )
            for start, end, raw_tag in region.html_tokens
        )
        inline_cursor = boundary_map[-1]

    removed_positions: set[int] = set()
    previous_position = -1
    for raw_tag, positions in translated_tokens:
        if (
            len(positions) != len(raw_tag)
            or not positions
            or any(position <= previous for previous, position in zip(positions, positions[1:]))
            or any(source[position] != char for position, char in zip(positions, raw_tag))
            or positions[0] <= previous_position
        ):
            return None, _error(
                revision,
                FragmentCompilationErrorCode.MALFORMED_HTML,
                "CommonMark inline HTML token cannot be mapped to exact source",
                start=base,
                end=base + len(source),
            )
        tag_match = _HTML_TOKEN_TAG_NAME_RE.match(raw_tag)
        tag_name = tag_match.group(1).lower() if tag_match is not None else None
        if tag_name in {"script", "style"}:
            return None, _error(
                revision,
                FragmentCompilationErrorCode.UNSUPPORTED_HTML,
                "unsafe inline HTML is unselectable",
                start=base,
                end=base + len(source),
            )
        removed_positions.update(position for position in positions if source[position] not in {"\r", "\n"})
        previous_position = positions[-1]
    return (
        _text_candidate(
            revision.content,
            fragment_type,
            base,
            base + len(source),
            roles,
            "".join(char for index, char in enumerate(source) if index not in removed_positions),
        ),
        None,
    )


def _inline_source_boundary_map(
    source: str,
    inline_source: str,
    *,
    start_cursor: int,
) -> tuple[int, ...] | None:
    """Map CommonMark container-stripped inline coordinates back to raw source."""

    source_line_starts = _line_starts(source)
    source_line_index = max(0, bisect_right(source_line_starts, start_cursor) - 1)
    target_cursor = 0
    boundaries = [-1] * (len(inline_source) + 1)
    for target_line in inline_source.splitlines(keepends=True) or [inline_source]:
        target_body = target_line.rstrip("\r\n")
        target_ending = target_line[len(target_body) :]
        matched = False
        while source_line_index < len(source_line_starts):
            raw_start = source_line_starts[source_line_index]
            raw_end = (
                source_line_starts[source_line_index + 1]
                if source_line_index + 1 < len(source_line_starts)
                else len(source)
            )
            raw_line = source[raw_start:raw_end]
            raw_body = raw_line.rstrip("\r\n")
            minimum = max(0, start_cursor - raw_start) if target_cursor == 0 else 0
            body_offset = raw_body.find(target_body, minimum)
            if body_offset >= 0:
                body_start = raw_start + body_offset
                for index in range(len(target_body) + 1):
                    boundaries[target_cursor + index] = body_start + index
                target_cursor += len(target_body)
                if target_ending:
                    raw_ending = raw_line[len(raw_body) :]
                    if not raw_ending:
                        return None
                    if target_ending == "\n" and raw_ending == "\r\n":
                        boundaries[target_cursor] = raw_end - 1
                    for index in range(len(target_ending)):
                        boundaries[target_cursor + index + 1] = raw_end
                    target_cursor += len(target_ending)
                    source_line_index += 1
                matched = True
                break
            source_line_index += 1
        if not matched:
            return None
    if target_cursor != len(inline_source) or any(value < 0 for value in boundaries):
        return None
    return tuple(boundaries)


def _html_candidates(
    revision: SourceObservationRevision,
    source: str,
    *,
    base: int,
    roles: frozenset[EvidenceRole],
) -> tuple[tuple[_FragmentCandidate, ...], tuple[FragmentCompilationError, ...]]:
    parser = _OffsetHTMLParser(source)
    parser.close_checked()
    if parser.failure is not None:
        if parser.internal_failure:
            return (), (
                _error(
                    revision,
                    FragmentCompilationErrorCode.MALFORMED_HTML,
                    parser.failure,
                    start=base,
                    end=base + len(source),
                ),
            )
        if any(node.unsafe for node in _walk_html_nodes(parser.roots)):
            return (), (
                _error(
                    revision,
                    FragmentCompilationErrorCode.UNSUPPORTED_HTML,
                    "unsafe HTML block is unselectable",
                    start=base,
                    end=base + len(source),
                ),
            )
        presentation = _html_presentation(source)
        if presentation:
            return (
                (
                    _text_candidate(
                        revision.content,
                        "html-block-atomic",
                        base,
                        base + len(source),
                        roles,
                        presentation,
                    ),
                ),
                (),
            )
        return (), (
            _error(
                revision,
                FragmentCompilationErrorCode.NO_SELECTABLE_CONTENT,
                "CommonMark HTML block contains no selectable claim text",
                start=base,
                end=base + len(source),
            ),
        )

    all_nodes = tuple(_walk_html_nodes(parser.roots))
    unsafe_nodes = tuple(node for node in all_nodes if _html_node_has_unsafe_content(node))
    safe_nodes = tuple(
        node
        for node in all_nodes
        if node.end is not None
        and not _html_node_has_unsafe_content(node)
        and _html_presentation(source[node.start : node.end])
    )
    semantic = tuple(node for node in safe_nodes if node.tag in _HTML_SEMANTIC_TAGS)
    selected = tuple(
        node
        for node in semantic
        if not any(
            parent is not node
            and parent.end is not None
            and node.end is not None
            and parent.start <= node.start
            and node.end <= parent.end
            for parent in semantic
        )
    )
    if not selected:
        selected = tuple(
            node
            for node in safe_nodes
            if not any(
                parent is not node
                and parent.end is not None
                and node.end is not None
                and parent.start <= node.start
                and node.end <= parent.end
                for parent in safe_nodes
            )
        )

    candidates = tuple(
        _text_candidate(
            revision.content,
            f"html-{node.tag}",
            base + node.start,
            base + (node.end or node.start),
            roles,
            _html_presentation(source[node.start : node.end]),
        )
        for node in sorted(selected, key=lambda value: (value.start, value.end or value.start, value.tag))
    )
    errors = tuple(
        _error(
            revision,
            FragmentCompilationErrorCode.UNSUPPORTED_HTML,
            f"unsafe HTML element is unselectable: {node.tag}",
            start=base + node.start,
            end=base + (node.end or node.start_tag_end),
        )
        for node in unsafe_nodes
    )
    if not candidates and not errors:
        errors = (
            _error(
                revision,
                FragmentCompilationErrorCode.UNSUPPORTED_HTML,
                "HTML region contains no selectable claim text",
                start=base,
                end=base + len(source),
            ),
        )
    return candidates, errors


def _walk_html_nodes(nodes: Sequence[_HtmlNode]):
    for node in nodes:
        yield node
        yield from _walk_html_nodes(node.children)


def _html_node_has_unsafe_content(node: _HtmlNode) -> bool:
    return node.unsafe or any(_html_node_has_unsafe_content(child) for child in node.children)


def _html_presentation(source: str) -> str:
    collector = _HTMLTextCollector()
    try:
        collector.feed(source)
        collector.close()
    except Exception:
        return ""
    return " ".join("".join(collector.parts).split())


@dataclass(frozen=True, slots=True)
class _JsonNode:
    start: int
    end: int
    value: object
    string_boundaries: tuple[int, ...] | None = None


@dataclass(frozen=True, slots=True)
class _JsonDocument:
    value: object
    nodes: Mapping[str, _JsonNode]

    @classmethod
    def parse(cls, source: str) -> _JsonDocument:
        scanner = _JsonScanner(source)
        value, end = scanner.parse_value(0, "")
        end = scanner.skip_space(end)
        if end != len(source):
            raise ValueError("canonical record contains trailing data")
        return cls(value=value, nodes=dict(scanner.nodes))


def canonical_record_field_ranges(
    revision: SourceObservationRevision,
) -> tuple[CanonicalFieldRange, ...]:
    """Index registered canonical fields using the compiler's exact parser."""

    contract = representation_contract_for_profile(revision.evidence_profile)
    if contract is None or contract.canonical_schema is None:
        raise ValueError("Revision does not declare a registered canonical-record schema")
    document = _JsonDocument.parse(revision.content)
    indexed = []
    for descriptor in contract.canonical_schema.fields:
        node = document.nodes.get(descriptor.json_pointer)
        if node is None or node.value is None:
            continue
        indexed.append(
            CanonicalFieldRange(
                descriptor=descriptor,
                start=node.start,
                end=node.end,
                value=node.value,
                comparison_value=canonical_field_comparison_value(
                    descriptor,
                    node.value,
                ),
                string_boundaries=node.string_boundaries,
            )
        )
    return tuple(indexed)


def canonical_record_is_tombstoned(revision: SourceObservationRevision) -> bool:
    """Return the registered non-selectable target tombstone state."""

    contract = representation_contract_for_profile(revision.evidence_profile)
    if contract is None or contract.canonical_schema is None:
        raise ValueError("Revision does not declare a registered canonical-record schema")
    pointer = contract.canonical_schema.tombstone_pointer
    if pointer is None:
        return False
    document = _JsonDocument.parse(revision.content)
    node = document.nodes.get(pointer)
    return (
        node is not None
        and node.value is not None
        and node.value != ""
        and node.value is not False
    )


def canonical_nested_changed_raw_ranges(
    base: CanonicalFieldRange,
    target: CanonicalFieldRange,
) -> tuple[tuple[int, int], ...]:
    """Map changed nested text structures back to exact canonical JSON ranges."""

    nested_profile = target.descriptor.nested_profile
    if (
        nested_profile is None
        or nested_profile != base.descriptor.nested_profile
        or not isinstance(base.value, str)
        or not isinstance(target.value, str)
        or target.string_boundaries is None
    ):
        raise ValueError("canonical nested text field cannot be mapped exactly")
    nested_evidence_profile = (
        MARKDOWN_STRUCTURAL_PROFILE
        if nested_profile == "markdown-structural"
        else PLAIN_TEXT_PROFILE
    )
    base_revision = SourceObservationRevision(
        id="canonical-nested-base",
        observation_id="canonical-nested",
        semantic_hash="canonical-nested-base",
        content=base.value,
        evidence_profile=nested_evidence_profile,
    )
    target_revision = SourceObservationRevision(
        id="canonical-nested-target",
        observation_id="canonical-nested",
        semantic_hash="canonical-nested-target",
        content=target.value,
        evidence_profile=nested_evidence_profile,
    )
    return tuple(
        (
            target.string_boundaries[start],
            target.string_boundaries[end],
        )
        for start, end in revision_changed_structural_ranges(
            base_revision,
            target_revision,
        )
    )


class _JsonScanner:
    def __init__(self, source: str) -> None:
        self.source = source
        self.nodes: dict[str, _JsonNode] = {}

    def skip_space(self, cursor: int) -> int:
        while cursor < len(self.source) and self.source[cursor].isspace():
            cursor += 1
        return cursor

    def parse_value(self, cursor: int, pointer: str) -> tuple[object, int]:
        cursor = self.skip_space(cursor)
        if cursor >= len(self.source):
            raise ValueError("canonical record ended before a value")
        start = cursor
        char = self.source[cursor]
        if char == '"':
            value, end, boundaries = _parse_json_string(self.source, cursor)
            self.nodes[pointer] = _JsonNode(start=start, end=end, value=value, string_boundaries=boundaries)
            return value, end
        if char == "{":
            value, end = self._parse_object(cursor, pointer)
            self.nodes[pointer] = _JsonNode(start=start, end=end, value=value)
            return value, end
        if char == "[":
            value, end = self._parse_array(cursor, pointer)
            self.nodes[pointer] = _JsonNode(start=start, end=end, value=value)
            return value, end
        try:
            value, end = json.JSONDecoder().raw_decode(self.source, cursor)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid canonical JSON at character {exc.pos}") from exc
        self.nodes[pointer] = _JsonNode(start=start, end=end, value=value)
        return value, end

    def _parse_object(self, cursor: int, pointer: str) -> tuple[dict[str, object], int]:
        result: dict[str, object] = {}
        cursor = self.skip_space(cursor + 1)
        if cursor < len(self.source) and self.source[cursor] == "}":
            return result, cursor + 1
        while True:
            if cursor >= len(self.source) or self.source[cursor] != '"':
                raise ValueError("canonical record object key must be a JSON string")
            key, cursor, _ = _parse_json_string(self.source, cursor)
            if key in result:
                raise ValueError(f"canonical record contains duplicate key: {key}")
            cursor = self.skip_space(cursor)
            if cursor >= len(self.source) or self.source[cursor] != ":":
                raise ValueError("canonical record object key lacks a value")
            child_pointer = pointer + "/" + key.replace("~", "~0").replace("/", "~1")
            value, cursor = self.parse_value(cursor + 1, child_pointer)
            result[key] = value
            cursor = self.skip_space(cursor)
            if cursor < len(self.source) and self.source[cursor] == "}":
                return result, cursor + 1
            if cursor >= len(self.source) or self.source[cursor] != ",":
                raise ValueError("canonical record object is not comma-delimited")
            cursor = self.skip_space(cursor + 1)

    def _parse_array(self, cursor: int, pointer: str) -> tuple[list[object], int]:
        result: list[object] = []
        cursor = self.skip_space(cursor + 1)
        if cursor < len(self.source) and self.source[cursor] == "]":
            return result, cursor + 1
        index = 0
        while True:
            value, cursor = self.parse_value(cursor, f"{pointer}/{index}")
            result.append(value)
            index += 1
            cursor = self.skip_space(cursor)
            if cursor < len(self.source) and self.source[cursor] == "]":
                return result, cursor + 1
            if cursor >= len(self.source) or self.source[cursor] != ",":
                raise ValueError("canonical record array is not comma-delimited")
            cursor = self.skip_space(cursor + 1)


_SIMPLE_JSON_ESCAPES = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}


def _parse_json_string(source: str, start: int) -> tuple[str, int, tuple[int, ...]]:
    cursor = start + 1
    output: list[str] = []
    boundaries = [cursor]
    while cursor < len(source):
        char = source[cursor]
        if char == '"':
            return "".join(output), cursor + 1, tuple(boundaries)
        if char == "\\":
            if cursor + 1 >= len(source):
                raise ValueError("unterminated JSON escape")
            escape = source[cursor + 1]
            if escape in _SIMPLE_JSON_ESCAPES:
                output.append(_SIMPLE_JSON_ESCAPES[escape])
                cursor += 2
                boundaries.append(cursor)
                continue
            if escape != "u" or cursor + 6 > len(source):
                raise ValueError("unsupported JSON escape")
            try:
                codepoint = int(source[cursor + 2 : cursor + 6], 16)
            except ValueError as exc:
                raise ValueError("invalid Unicode JSON escape") from exc
            cursor += 6
            if 0xD800 <= codepoint <= 0xDBFF:
                if cursor + 6 > len(source) or source[cursor : cursor + 2] != "\\u":
                    raise ValueError("unpaired high-surrogate JSON escape")
                low = int(source[cursor + 2 : cursor + 6], 16)
                if not 0xDC00 <= low <= 0xDFFF:
                    raise ValueError("invalid low-surrogate JSON escape")
                cursor += 6
                codepoint = 0x10000 + ((codepoint - 0xD800) << 10) + (low - 0xDC00)
            elif 0xDC00 <= codepoint <= 0xDFFF:
                raise ValueError("unpaired low-surrogate JSON escape")
            output.append(chr(codepoint))
            boundaries.append(cursor)
            continue
        if ord(char) < 0x20:
            raise ValueError("unescaped control character in JSON string")
        output.append(char)
        cursor += 1
        boundaries.append(cursor)
    raise ValueError("unterminated JSON string")


def _catalog_from_candidates(
    revision: SourceObservationRevision,
    profile: EvidenceRepresentationProfile | None,
    authority_ranges: tuple[EvidenceCandidateRange, ...],
    candidates: tuple[_FragmentCandidate, ...],
    errors: tuple[FragmentCompilationError, ...],
    *,
    max_fragments: int,
    max_presentation_chars: int,
) -> EvidenceFragmentCatalog:
    ordered = tuple(sorted(candidates, key=_candidate_sort_key))
    if not ordered and not errors:
        errors = (
            _error(
                revision,
                FragmentCompilationErrorCode.NO_SELECTABLE_CONTENT,
                "authority ranges contain no selectable Evidence Fragment",
            ),
        )
    overlap_error = _candidate_overlap_error(revision, ordered)
    if overlap_error is not None:
        errors = (*errors, overlap_error)
        ordered = ()
    presentation_chars = sum(len(candidate.presentation_text) for candidate in ordered)
    if len(ordered) > max_fragments or presentation_chars > max_presentation_chars:
        errors = (
            *errors,
            _error(
                revision,
                FragmentCompilationErrorCode.CATALOG_TOO_LARGE,
                "compiled Fragment catalog exceeds its explicit limits",
                fatal=True,
            ),
        )
        exposed: tuple[_FragmentCandidate, ...] = ()
    elif any(error.fatal for error in errors):
        exposed = ()
    else:
        exposed = ordered

    fragments = tuple(
        _materialize_fragment(revision, candidate, index) for index, candidate in enumerate(exposed, start=1)
    )
    digest = _catalog_digest(
        revision,
        profile,
        authority_ranges,
        ordered,
        errors,
        max_fragments=max_fragments,
        max_presentation_chars=max_presentation_chars,
    )
    return EvidenceFragmentCatalog(
        observation_revision_id=revision.id,
        profile=profile,
        fragments=fragments,
        errors=tuple(sorted(errors, key=_error_sort_key)),
        digest=digest,
        max_fragments=max_fragments,
        max_presentation_chars=max_presentation_chars,
    )


def _fatal_catalog(
    revision: SourceObservationRevision,
    profile: EvidenceRepresentationProfile | None,
    authority_ranges: tuple[EvidenceCandidateRange, ...],
    code: FragmentCompilationErrorCode,
    message: str,
    *,
    max_fragments: int,
    max_presentation_chars: int,
) -> EvidenceFragmentCatalog:
    return _catalog_from_candidates(
        revision,
        profile,
        authority_ranges,
        (),
        (_error(revision, code, message, fatal=True),),
        max_fragments=max_fragments,
        max_presentation_chars=max_presentation_chars,
    )


def _materialize_fragment(
    revision: SourceObservationRevision,
    candidate: _FragmentCandidate,
    index: int,
) -> EvidenceFragment:
    if candidate.kind is EvidenceFragmentKind.ARTIFACT:
        anchor = SourceAnchor(
            kind=AnchorKind.WHOLE_OBSERVATION,
            observation_id=revision.observation_id,
            observation_revision_id=revision.id,
        )
        raw_digest = candidate.raw_content_sha256 or ""
    else:
        assert candidate.start is not None and candidate.end is not None
        anchor = SourceAnchor(
            kind=AnchorKind.REVISION_RANGE,
            observation_id=revision.observation_id,
            observation_revision_id=revision.id,
            range_start=candidate.start,
            range_end=candidate.end,
        )
        raw_digest = hashlib.sha256(revision.content[candidate.start : candidate.end].encode("utf-8")).hexdigest()
    return EvidenceFragment(
        reference=f"f{index:06d}",
        kind=candidate.kind,
        fragment_type=candidate.fragment_type,
        anchor=anchor,
        primary_eligible=candidate.primary_eligible,
        raw_content_sha256=raw_digest,
        presentation_text=candidate.presentation_text,
        presentation_sha256=hashlib.sha256(candidate.presentation_text.encode("utf-8")).hexdigest(),
    )


def _catalog_digest(
    revision: SourceObservationRevision,
    profile: EvidenceRepresentationProfile | None,
    authority_ranges: tuple[EvidenceCandidateRange, ...],
    candidates: tuple[_FragmentCandidate, ...],
    errors: tuple[FragmentCompilationError, ...],
    *,
    max_fragments: int,
    max_presentation_chars: int,
) -> str:
    payload = {
        "compiler_contract_version": COMPILER_CONTRACT_VERSION,
        "observation_id": revision.observation_id,
        "observation_revision_id": revision.id,
        "profile": (
            {
                "name": profile.name,
                "version": profile.version,
                "coordinate_space": profile.coordinate_space.value,
                "schema_name": profile.schema_name,
                "schema_version": profile.schema_version,
            }
            if profile is not None
            else "absent"
        ),
        "candidate_ranges": [
            {
                "kind": item.anchor.kind.value,
                "range_start": item.anchor.range_start,
                "range_end": item.anchor.range_end,
                "primary_eligible": item.primary_eligible,
            }
            for item in sorted(authority_ranges, key=_authority_sort_key)
        ],
        "limits": {
            "max_fragments": max_fragments,
            "max_presentation_chars": max_presentation_chars,
        },
        "fragments": [
            {
                "kind": item.kind.value,
                "fragment_type": item.fragment_type,
                "start": item.start,
                "end": item.end,
                "primary_eligible": item.primary_eligible,
                "raw_content_sha256": (
                    item.raw_content_sha256
                    if item.raw_content_sha256 is not None
                    else hashlib.sha256(revision.content[item.start : item.end].encode("utf-8")).hexdigest()
                ),
                "presentation_sha256": hashlib.sha256(item.presentation_text.encode("utf-8")).hexdigest(),
            }
            for item in candidates
        ],
        "errors": [
            {
                "code": item.code.value,
                "range_start": item.range_start,
                "range_end": item.range_end,
                "fatal": item.fatal,
            }
            for item in sorted(errors, key=_error_sort_key)
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _candidate_overlap_error(
    revision: SourceObservationRevision,
    candidates: tuple[_FragmentCandidate, ...],
) -> FragmentCompilationError | None:
    text_candidates = tuple(item for item in candidates if item.start is not None and item.end is not None)
    for previous, current in zip(text_candidates, text_candidates[1:]):
        assert previous.end is not None and current.start is not None
        if current.start < previous.end:
            return _error(
                revision,
                FragmentCompilationErrorCode.INVALID_AUTHORITY_RANGE,
                "compiled Fragments overlap",
                start=current.start,
                end=previous.end,
                fatal=True,
            )
    return None


def _candidate_sort_key(item: _FragmentCandidate) -> tuple[int, int, str, str, bool]:
    start = item.start if item.start is not None else -1
    end = item.end if item.end is not None else -1
    raw_digest = item.raw_content_sha256 or ""
    presentation_digest = hashlib.sha256(item.presentation_text.encode("utf-8")).hexdigest()
    return (
        start,
        end,
        item.fragment_type,
        raw_digest + presentation_digest,
        item.primary_eligible,
    )


def _authority_sort_key(item: EvidenceCandidateRange) -> tuple[int, int, bool]:
    return (
        item.anchor.range_start if item.anchor.range_start is not None else -1,
        item.anchor.range_end if item.anchor.range_end is not None else -1,
        item.primary_eligible,
    )


def _error_sort_key(item: FragmentCompilationError) -> tuple[int, int, str, str]:
    return (
        item.range_start if item.range_start is not None else -1,
        item.range_end if item.range_end is not None else -1,
        item.code.value,
        item.message,
    )


def _text_candidate(
    authority_text: str,
    fragment_type: str,
    start: int,
    end: int,
    roles: frozenset[EvidenceRole],
    presentation: str,
) -> _FragmentCandidate:
    return _FragmentCandidate(
        kind=EvidenceFragmentKind.TEXT,
        fragment_type=fragment_type,
        start=start,
        end=end,
        eligible_roles=roles,
        presentation_text=presentation,
        raw_content_sha256=hashlib.sha256(authority_text[start:end].encode("utf-8")).hexdigest(),
    )


def _error(
    revision: SourceObservationRevision,
    code: FragmentCompilationErrorCode,
    message: str,
    *,
    start: int | None = None,
    end: int | None = None,
    fatal: bool = False,
) -> FragmentCompilationError:
    return FragmentCompilationError(
        code=code,
        observation_revision_id=revision.id,
        message=message,
        range_start=start,
        range_end=end,
        fatal=fatal,
    )


def _anchor_span(anchor: SourceAnchor, content: str) -> tuple[int, int]:
    if anchor.kind is AnchorKind.WHOLE_OBSERVATION:
        return 0, len(content)
    assert anchor.range_start is not None and anchor.range_end is not None
    return anchor.range_start, anchor.range_end


def _bind_candidates_to_authority(
    revision: SourceObservationRevision,
    candidates: tuple[_FragmentCandidate, ...],
    authority_ranges: tuple[EvidenceCandidateRange, ...],
) -> tuple[tuple[_FragmentCandidate, ...], tuple[FragmentCompilationError, ...]]:
    bound: list[_FragmentCandidate] = []
    errors: list[FragmentCompilationError] = []
    spans = tuple((*_anchor_span(item.anchor, revision.content), item) for item in authority_ranges)
    for candidate in candidates:
        assert candidate.start is not None and candidate.end is not None
        containing = tuple(item for start, end, item in spans if start <= candidate.start and candidate.end <= end)
        if len(containing) == 1:
            bound.append(replace(candidate, eligible_roles=containing[0].eligible_roles))
            continue
        overlapping = any(candidate.start < end and start < candidate.end for start, end, _ in spans)
        if overlapping:
            errors.append(
                _error(
                    revision,
                    FragmentCompilationErrorCode.INVALID_AUTHORITY_RANGE,
                    "structural Fragment crosses an authority boundary and was not clipped",
                    start=candidate.start,
                    end=candidate.end,
                )
            )
    return tuple(bound), tuple(errors)


def _errors_inside_authority(
    errors: tuple[FragmentCompilationError, ...],
    authority_ranges: tuple[EvidenceCandidateRange, ...],
    content: str,
) -> tuple[FragmentCompilationError, ...]:
    spans = tuple(_anchor_span(item.anchor, content) for item in authority_ranges)
    return tuple(
        error
        for error in errors
        if error.fatal
        or error.range_start is None
        or error.range_end is None
        or any(error.range_start < end and start < error.range_end for start, end in spans)
    )


def _line_starts(text: str) -> tuple[int, ...]:
    starts = [0]
    starts.extend(match.end() for match in re.finditer(r"\n", text))
    return tuple(starts)


def _token_range(text: str, line_starts: tuple[int, ...], start_line: int, end_line: int) -> tuple[int, int]:
    start = line_starts[start_line]
    end = line_starts[end_line] if end_line < len(line_starts) else len(text)
    return _trim_range(text, start, end)


def _trim_range(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _paragraph_ranges(text: str) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for separator in re.finditer(r"\n[ \t]*\n+", text):
        start, end = _trim_range(text, cursor, separator.start())
        if start < end:
            ranges.append((start, end))
        cursor = separator.end()
    start, end = _trim_range(text, cursor, len(text))
    if start < end:
        ranges.append((start, end))
    return tuple(ranges)


def _canonical_value_presentation(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
