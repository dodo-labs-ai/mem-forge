"""Recoverable inference stages belonging to an existing Source derivation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Literal, Mapping, Protocol

WorkKind = Literal["support_assess", "support_scan", "support_reduce", "support_finalize", "claim_assess"]


def payload_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class DerivationWork:
    id: str
    kind: WorkKind
    manifest: Mapping[str, Any]
    status: str = "pending"
    result: Mapping[str, Any] | None = None
    result_hash: str | None = None
    error_code: str | None = None
    permanent: bool = False

    @classmethod
    def create(cls, kind: WorkKind, manifest: Mapping[str, Any]):
        identity = json.loads(json.dumps({**manifest, "kind": kind}, ensure_ascii=False))
        return cls("work-" + payload_hash(identity), kind, identity)

    def to_payload(self):
        return {
            "id": self.id,
            "kind": self.kind,
            "manifest": self.manifest,
            "status": self.status,
            "result": self.result,
            "result_hash": self.result_hash,
            "error_code": self.error_code,
            "permanent": self.permanent,
        }

    @classmethod
    def from_payload(cls, payload):
        work = cls(**payload)
        if work.kind != work.manifest.get("kind"):
            raise ValueError("derivation work kind mismatch")
        if work.id != "work-" + payload_hash(work.manifest):
            raise ValueError("derivation work identity mismatch")
        if work.status == "completed" and (work.result is None or payload_hash(work.result) != work.result_hash):
            raise ValueError("derivation work result mismatch")
        return work


class DerivationWorkStore(Protocol):
    async def stage_derivation_work(self, *, derivation_id: str, work: DerivationWork) -> DerivationWork: ...

    async def record_derivation_work(self, *, derivation_id: str, work: DerivationWork) -> DerivationWork: ...
