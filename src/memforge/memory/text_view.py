"""Versioned provenance of compiler-added text interpretation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping

from memforge.source_projection import AnchorKind, SourceAnchor


@dataclass(frozen=True, slots=True)
class TextViewOrigin:
    anchor: SourceAnchor
    raw_content_sha256: str

    def __post_init__(self) -> None:
        if self.anchor.kind is not AnchorKind.REVISION_RANGE:
            raise ValueError("text interpretation requires an exact range")
        if len(self.raw_content_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.raw_content_sha256):
            raise ValueError("text origin requires a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class TextEvidenceView:
    kind: str
    version: int
    origins: tuple[TextViewOrigin, ...] = ()

    def __post_init__(self) -> None:
        if not self.kind or self.version <= 0:
            raise ValueError("text view requires a versioned kind")
        if len(set(self.origins)) != len(self.origins):
            raise ValueError("text view cannot repeat origins")

    def validate_core(self, core: SourceAnchor) -> None:
        if any((o.anchor.observation_id, o.anchor.observation_revision_id)
               != (core.observation_id, core.observation_revision_id) for o in self.origins):
            raise ValueError("text interpretation origins must share the core Observation revision")

    def verify_origins(self, core: SourceAnchor, revision_content: str) -> None:
        self.validate_core(core)
        for origin in self.origins:
            start, end = origin.anchor.range_start, origin.anchor.range_end
            if start is None or end is None or end > len(revision_content):
                raise ValueError("text interpretation origin exceeds pinned revision")
            if hashlib.sha256(revision_content[start:end].encode()).hexdigest() != origin.raw_content_sha256:
                raise ValueError("text interpretation origin integrity mismatch")

    def payload(self) -> dict[str, object]:
        return {"kind": self.kind, "version": self.version, "origins": [
            {"range_start": o.anchor.range_start, "range_end": o.anchor.range_end,
             "observation_id": o.anchor.observation_id,
             "observation_revision_id": o.anchor.observation_revision_id,
             "raw_content_sha256": o.raw_content_sha256} for o in self.origins]}

    @classmethod
    def from_payload(cls, value: object) -> TextEvidenceView | None:
        if value is None:
            return None
        if not isinstance(value, Mapping) or set(value) != {"kind", "version", "origins"}:
            raise ValueError("invalid text view descriptor")
        if not isinstance(value["kind"], str) or type(value["version"]) is not int or not isinstance(value["origins"], list):
            raise ValueError("invalid text view fields")
        origins = []
        for item in value["origins"]:
            if not isinstance(item, Mapping) or set(item) != {"range_start", "range_end", "observation_id", "observation_revision_id", "raw_content_sha256"}:
                raise ValueError("invalid text view origin")
            if type(item["range_start"]) is not int or type(item["range_end"]) is not int or not all(isinstance(item[k], str) for k in ("observation_id", "observation_revision_id", "raw_content_sha256")):
                raise ValueError("invalid text origin fields")
            origins.append(TextViewOrigin(SourceAnchor(
                kind=AnchorKind.REVISION_RANGE, observation_id=item["observation_id"],
                observation_revision_id=item["observation_revision_id"],
                range_start=item["range_start"], range_end=item["range_end"]), item["raw_content_sha256"]))
        return cls(value["kind"], value["version"], tuple(origins))
