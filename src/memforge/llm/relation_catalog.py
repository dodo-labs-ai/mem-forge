"""Request-local catalogs and complete sparse relationship coverage."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Generic, Hashable, Mapping, TypeVar

T = TypeVar("T")


class CatalogCapacityError(ValueError):
    error_code = "input_capacity_exceeded"


def request_ref(prefix: str, ordinal: int) -> str:
    if not re.fullmatch(r"[A-Z]{3}", prefix) or ordinal < 1:
        raise ValueError("request references require three uppercase letters and 0001–9999")
    if ordinal > 9999:
        raise CatalogCapacityError("request catalog exceeds four-digit reference capacity")
    return f"{prefix}-{ordinal:04d}"


@dataclass
class RequestCatalog(Generic[T]):
    """Intern each record once; reject conflicting snapshots under the same key."""

    prefix: str
    records: dict[str, T] = field(default_factory=dict, init=False)
    refs: dict[Hashable, str] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        request_ref(self.prefix, 1)

    def add(self, key: Hashable, value: T) -> str:
        if key in self.refs:
            ref = self.refs[key]
            if self.records[ref] != value:
                raise ValueError("conflicting catalog snapshots")
            return ref
        ref = request_ref(self.prefix, len(self.records) + 1)
        self.refs[key] = ref
        self.records[ref] = value
        return ref


@dataclass(frozen=True)
class RelationCoverage:
    """Coverage refers to candidate completion, never to omitted pair decisions."""

    allowed: Mapping[str, frozenset[str]]

    def validate(self, rows: Any) -> None:
        seen: set[str] = set()
        for row in rows:
            candidate_id = row.candidate_id
            if candidate_id not in self.allowed or candidate_id in seen:
                raise ValueError("unknown or duplicate candidate completion")
            seen.add(candidate_id)
            refs = [edge.existing_id for edge in row.relations]
            refs.extend(getattr(row, "uncertain_existing_ids", ()))
            if len(refs) != len(set(refs)):
                raise ValueError("duplicate incumbent reference")
            if not set(refs).issubset(self.allowed[candidate_id]):
                raise ValueError("incumbent reference is outside this candidate's allowed set")
        if seen != set(self.allowed):
            raise ValueError("missing candidate completion")
