"""Recoverable inference stages belonging to an existing Source derivation."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import TYPE_CHECKING, Any, Literal, Mapping, Protocol, Sequence

from pydantic import BaseModel

if TYPE_CHECKING:
    from memforge.llm.batch_runner import ItemFailure, LlmRequest

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


class DerivationWorkJournal:
    """Persist each batch-runner request as one resumable DerivationWork.

    A request is identified by its rendered prompt and schema under one task
    scope, capacity policy, model and output limit. The prompt already names
    the items, their carried state and their position, so a retried sync reuses
    every request that completed and resends only the rest. Stored results are
    the raw validated responses; the runner decodes them again on reuse.
    """

    def __init__(
        self, *, store: DerivationWorkStore, derivation_id: str, kind: WorkKind,
        scope: Mapping[str, Any], budget_identity: str, model: str,
    ) -> None:
        self._store = store
        self._derivation_id = derivation_id
        self._kind = kind
        self._scope = scope
        self._budget_identity = budget_identity
        self._model = model
        self._staged: dict[str, DerivationWork] = {}
        self._items: dict[str, tuple[str, ...]] = {}
        self.works: list[DerivationWork] = []

    def works_for(self, item_id: str) -> tuple[DerivationWork, ...]:
        """Completed works whose request carried this item."""

        return tuple(work for work in self.works if item_id in self._items[work.id])

    async def stage(self, request: LlmRequest, item_ids: Sequence[str]) -> tuple[str, BaseModel | None]:
        work = DerivationWork.create(self._kind, {
            "scope": self._scope,
            "prompt_hash": payload_hash(request.prompt),
            "schema": payload_hash(request.response_format.model_json_schema()),
            "budget": self._budget_identity,
            "model": self._model,
            "output": request.max_tokens,
        })
        work = await self._store.stage_derivation_work(derivation_id=self._derivation_id, work=work)
        self._items[work.id] = tuple(item_ids)
        if work.status == "completed":
            self.works.append(work)
            return work.id, request.response_format.model_validate(work.result)
        self._staged[work.id] = work
        return work.id, None

    async def record(self, work_id: str, outcome: BaseModel | ItemFailure) -> None:
        work = self._staged[work_id]
        if isinstance(outcome, BaseModel):
            result = outcome.model_dump(mode="json")
            work = replace(work, status="completed", result=result, result_hash=payload_hash(result), error_code=None)
        else:
            work = replace(work, status="retryable_failure", error_code=outcome.error_code)
        work = await self._store.record_derivation_work(derivation_id=self._derivation_id, work=work)
        if work.status == "completed":
            self.works.append(work)
