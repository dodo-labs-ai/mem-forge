"""Durable, provider-neutral contract for progressive Memory relation discovery."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256

from memforge.memory.evidence import RelationDirection
from memforge.memory.relation_classifier import MemoryRelationType


CURRENT_RELATION_EVIDENCE_PREDICATE_SQL = """
msa.memory_id = ? AND msa.source_id = ? AND msa.active = 1
AND er.role = 'primary'
AND eu.source_id = ? AND eu.source_lineage_id = ?
AND so.source_id = eu.source_id
AND so.source_unit_id = eu.source_lineage_id
AND er.observation_revision_id = so.current_revision_id
""".strip()


class RelationDiscoveryWorkStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    OBSOLETE = "obsolete"


@dataclass(frozen=True, slots=True)
class PreclassifiedRelationDecision:
    """Reusable pair decision fenced by content, current Evidence, and access."""

    candidate_memory_id: str
    expected_candidate_content_hash: str
    expected_candidate_support_set_hash: str
    expected_candidate_access_context_hash: str
    expected_challenger_access_context_hash: str
    relation_type: MemoryRelationType
    direction: RelationDirection
    reason: str
    classifier_version: str

    def to_payload(self) -> dict[str, str]:
        return {
            "candidate_memory_id": self.candidate_memory_id,
            "expected_candidate_content_hash": self.expected_candidate_content_hash,
            "expected_candidate_support_set_hash": self.expected_candidate_support_set_hash,
            "expected_candidate_access_context_hash": self.expected_candidate_access_context_hash,
            "expected_challenger_access_context_hash": self.expected_challenger_access_context_hash,
            "relation_type": self.relation_type.value,
            "direction": self.direction.value,
            "reason": self.reason,
            "classifier_version": self.classifier_version,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "PreclassifiedRelationDecision":
        return cls(
            candidate_memory_id=str(payload["candidate_memory_id"]),
            expected_candidate_content_hash=str(payload["expected_candidate_content_hash"]),
            expected_candidate_support_set_hash=str(
                payload.get("expected_candidate_support_set_hash") or ""
            ),
            expected_candidate_access_context_hash=str(
                payload.get("expected_candidate_access_context_hash") or ""
            ),
            expected_challenger_access_context_hash=str(
                payload.get("expected_challenger_access_context_hash") or ""
            ),
            relation_type=MemoryRelationType(str(payload["relation_type"])),
            direction=RelationDirection(str(payload["direction"])),
            reason=str(payload["reason"]),
            classifier_version=str(payload["classifier_version"]),
        )


def resolve_relation_discovery_actor_user_id(
    *,
    visibility: str,
    owner_user_id: str | None,
    requested_actor_user_id: str | None,
) -> str | None:
    """Resolve the principal that owns one durable relation-discovery decision."""

    if visibility == "private":
        if not owner_user_id:
            raise ValueError("private relation discovery requires owner identity")
        return owner_user_id
    return requested_actor_user_id


@dataclass(frozen=True, slots=True)
class RelationDiscoveryRequest:
    """One bounded post-commit discovery request for an activated Memory."""

    id: str
    memory_id: str
    expected_content_hash: str
    source_id: str
    source_unit_id: str
    source_unit_revision_id: str | None
    doc_id: str
    actor_user_id: str | None
    entity_ids: tuple[int, ...] = ()
    preclassified_decisions: tuple[PreclassifiedRelationDecision, ...] = ()

    def __post_init__(self) -> None:
        required = {
            "id": self.id,
            "memory_id": self.memory_id,
            "expected_content_hash": self.expected_content_hash,
            "source_id": self.source_id,
            "source_unit_id": self.source_unit_id,
            "doc_id": self.doc_id,
        }
        missing = sorted(name for name, value in required.items() if not value)
        if missing:
            raise ValueError("relation discovery request requires " + ", ".join(missing))
        candidate_ids = [item.candidate_memory_id for item in self.preclassified_decisions]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("duplicate preclassified relation candidate")


@dataclass(frozen=True, slots=True)
class RelationDiscoveryWork:
    """A leased durable request; attempts are fenced by worker and lease token."""

    request: RelationDiscoveryRequest
    lifecycle_plan_id: str
    status: RelationDiscoveryWorkStatus
    attempts: int = 0
    lease_owner: str | None = None
    lease_token: str | None = None
    lease_until: str | None = None
    next_attempt_at: str | None = None
    error: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    completed_at: str | None = None


def relation_discovery_request_id(
    *,
    lifecycle_plan_id: str,
    memory_id: str,
    expected_content_hash: str,
) -> str:
    digest = sha256("\x1f".join((lifecycle_plan_id, memory_id, expected_content_hash)).encode("utf-8")).hexdigest()[:20]
    return f"relation-work-{digest}"
