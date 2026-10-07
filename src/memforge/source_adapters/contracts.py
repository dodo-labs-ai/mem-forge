"""Source-neutral declarations consumed by the representation compiler."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Callable, Mapping

from memforge.source_projection import EvidenceRepresentationProfile, ProjectionCoverage, SourceRelationType, UnitTitle
from memforge.source_time import reported_source_time


@dataclass(frozen=True, slots=True)
class DeclaredTextOrigin:
    """Source-owned meaning added to a selection from outside its core range."""

    start: int
    end: int
    content_value: str


@dataclass(frozen=True, slots=True)
class DeclaredTextFragment:
    """One complete native selection in decoded-field Unicode coordinates.

    ``content_value`` is the format-owned comparison value, never model text.
    The compiler independently binds the range to the immutable record.
    """

    start: int
    end: int
    kind: str
    presentation: str
    content_value: str
    interpretation_origins: tuple[DeclaredTextOrigin, ...] = ()


@dataclass(frozen=True, slots=True)
class DeclaredTextGroup:
    start: int
    end: int
    context_ranges: tuple[tuple[int, int], ...] = ()
    together: bool = False


@dataclass(frozen=True, slots=True)
class ParsedDeclaredText:
    fragments: tuple[DeclaredTextFragment, ...]
    groups: tuple[DeclaredTextGroup, ...] = ()


@dataclass(frozen=True, slots=True)
class DeclaredTextFormat:
    """An adapter-owned format; invalid or unsupported source raises ValueError."""

    name: str
    version: int
    parse: Callable[[str], ParsedDeclaredText]

    def __post_init__(self) -> None:
        if not self.name or self.version <= 0:
            raise ValueError("declared text format requires a versioned name")


@dataclass(frozen=True, slots=True)
class CanonicalRecordField:
    json_pointer: str
    nested_profile: str | None = None
    comparison_keys: tuple[str, ...] = ()
    contextual: bool = False
    text_format: DeclaredTextFormat | None = None
    label: str | None = None
    framing_fields: tuple[tuple[str, str], ...] = ()
    comparison_pointer: str | None = None
    presentation_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.nested_profile not in {None, "markdown-structural", "plain-text"}:
            raise ValueError("unsupported nested canonical-record text profile")
        if self.text_format is not None and self.nested_profile is not None:
            raise ValueError("canonical text field must declare exactly one format")
        if self.label is not None and not self.label.strip():
            raise ValueError("canonical field label must not be blank")


@dataclass(frozen=True, slots=True)
class CanonicalRecordSchema:
    name: str
    version: int
    fields: tuple[CanonicalRecordField, ...]
    tombstone_pointer: str | None = None
    model_interpretation: str | None = None
    framing_fields: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class EvidenceRepresentationContract:
    profile: EvidenceRepresentationProfile
    canonical_schema: CanonicalRecordSchema | None = None

    def __post_init__(self) -> None:
        if (self.profile.name == "canonical-record") != (self.canonical_schema is not None):
            raise ValueError("canonical schema ownership must match the representation profile")


@dataclass(frozen=True, slots=True)
class _ObservationInput:
    """One Observation as the provider payload gives it.

    ``observed_at`` is the source's own time for this content (design 0.9):
    when the provider last changed it, or ``None`` when the provider records
    no usable time. It is never a discovery, fetch, submission or sync time.
    """

    observation_type: str
    provider_key: str
    content: str
    semantic_value: object
    locator: Mapping[str, object]
    observed_at: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    semantic_hash: str | None = None

    def __post_init__(self) -> None:
        # One UTC form for every provider's time, whatever format it reported.
        object.__setattr__(self, "observed_at", reported_source_time(self.observed_at))


def _unit_title(kind: str, *fields: tuple[str, object]) -> UnitTitle:
    """Name a Unit by the values present in its provider payload; absent values are omitted, never guessed."""
    return UnitTitle(
        kind=kind,
        fields=tuple(
            (name, " ".join(str(value).split()))
            for name, value in fields
            if value is not None and str(value).strip()
        ),
    )


@dataclass(frozen=True, slots=True)
class _UnitEndpoint:
    """An adapter-declared relation target outside the projected Unit."""

    unit_type: str
    provider_key: str


@dataclass(frozen=True, slots=True)
class _NativeProjection:
    """One provider payload as a Source Unit, its Observations and its Unit Title."""

    unit_type: str
    provider_key: str
    observations: tuple[_ObservationInput, ...]
    relations: tuple[
        tuple[SourceRelationType, str | _UnitEndpoint, str | _UnitEndpoint, str | None, Mapping[str, object]], ...
    ]
    coverage: ProjectionCoverage
    locator: Mapping[str, object]
    # None only when the payload tombstones the whole Unit.
    title: UnitTitle | None


def _normalized_utc_timestamp(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc).isoformat()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
