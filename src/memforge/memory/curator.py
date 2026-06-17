"""Non-destructive memory curation primitives.

The Curator creates consolidated memories and explicit lineage. It does not
retire, delete, or otherwise mutate the atomic memories it summarizes.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Awaitable, Callable, Iterable, Protocol, Sequence

from memforge.models import (
    Memory,
    MemoryCurationRun,
    MemoryLevel,
    UNSORTED_PROJECT_KEY,
    content_hash,
    generate_memory_id,
)


@dataclass(frozen=True)
class CuratorCandidate:
    """A memory plus the source metadata needed for policy selection."""

    memory: Memory
    source_type: str
    client: str | None


@dataclass(frozen=True)
class CuratedMemoryDraft:
    """Policy-produced memory content before persistence metadata is stamped."""

    memory_type: str
    content: str
    tags: list[str]


@dataclass(frozen=True)
class CuratorCluster:
    """A deterministic group of candidates eligible for one consolidation."""

    key: str
    candidates: tuple[CuratorCandidate, ...]
    repo_identifier: str | None
    project_key: str
    topic_tags: tuple[str, ...]


@dataclass(frozen=True)
class CuratorResult:
    candidate_count: int
    created_memory_count: int
    created_memory_ids: list[str]


class CuratorStore(Protocol):
    async def insert_memory(self, memory: Memory) -> str: ...

    async def add_memory_derivation(
        self,
        parent_memory_id: str,
        child_memory_id: str,
        *,
        relation: str = "summarizes",
    ) -> None: ...

    async def record_memory_curation_run(self, run: MemoryCurationRun) -> None: ...


SummarizeCluster = Callable[[CuratorCluster], Awaitable[CuratedMemoryDraft]]


class MemoryCuratorPolicy(Protocol):
    policy_id: str
    source_type: str

    def applies_to(self, candidate: CuratorCandidate) -> bool: ...

    def cluster_key(self, candidate: CuratorCandidate) -> str: ...

    def make_cluster(
        self,
        key: str,
        candidates: Sequence[CuratorCandidate],
    ) -> CuratorCluster: ...


class AgentSessionCuratorPolicy:
    """Curator policy for Codex and Claude Code coding-session memories."""

    policy_id = "agent_session.coding.v1"
    source_type = "agent_session"
    supported_clients = frozenset(("codex", "claude-code"))

    def applies_to(self, candidate: CuratorCandidate) -> bool:
        return (
            candidate.source_type == self.source_type
            and candidate.client in self.supported_clients
            and candidate.memory.memory_level == MemoryLevel.ATOMIC.value
            and bool(candidate.memory.repo_identifier)
        )

    def cluster_key(self, candidate: CuratorCandidate) -> str:
        memory = candidate.memory
        project_key = memory.project_key or UNSORTED_PROJECT_KEY
        owner_key = (
            f"private:{memory.owner_user_id}"
            if memory.owner_user_id
            else "workspace"
        )
        topic = self._topic_signature(memory)
        return "|".join([
            "agent_session",
            owner_key,
            memory.repo_identifier or "",
            project_key,
            topic,
        ])

    def make_cluster(
        self,
        key: str,
        candidates: Sequence[CuratorCandidate],
    ) -> CuratorCluster:
        first = candidates[0].memory
        return CuratorCluster(
            key=key,
            candidates=tuple(candidates),
            repo_identifier=first.repo_identifier,
            project_key=first.project_key or UNSORTED_PROJECT_KEY,
            topic_tags=tuple(self._topic_tags(first)),
        )

    def _topic_signature(self, memory: Memory) -> str:
        tags = self._topic_tags(memory)
        return f"{memory.memory_type}:{','.join(tags)}"

    def _topic_tags(self, memory: Memory) -> list[str]:
        return sorted(memory.tags)[:3]


class MemoryCuratorRunner:
    """Run one source-type policy over an explicit candidate set."""

    def __init__(
        self,
        *,
        store: CuratorStore,
        policy: MemoryCuratorPolicy,
        summarize: SummarizeCluster,
        min_cluster_size: int = 20,
    ) -> None:
        self._store = store
        self._policy = policy
        self._summarize = summarize
        self._min_cluster_size = min_cluster_size

    async def curate(
        self,
        candidates: Sequence[CuratorCandidate],
        *,
        run_id: str | None = None,
    ) -> CuratorResult:
        eligible = [
            candidate for candidate in candidates
            if self._policy.applies_to(candidate)
        ]
        grouped: dict[str, list[CuratorCandidate]] = defaultdict(list)
        for candidate in eligible:
            grouped[self._policy.cluster_key(candidate)].append(candidate)

        created_ids: list[str] = []
        created_clusters: list[CuratorCluster] = []
        now = datetime.now(timezone.utc)
        for key, cluster_candidates in grouped.items():
            if len(cluster_candidates) < self._min_cluster_size:
                continue
            cluster = self._policy.make_cluster(key, cluster_candidates)
            draft = await self._summarize(cluster)
            first = cluster.candidates[0].memory
            memory = Memory(
                id=generate_memory_id(),
                memory_type=draft.memory_type,
                content=draft.content,
                content_hash=content_hash(draft.content),
                visibility=first.visibility,
                owner_user_id=first.owner_user_id,
                project_key=cluster.project_key,
                repo_identifier=cluster.repo_identifier,
                tags=draft.tags,
                memory_level=MemoryLevel.CONSOLIDATED.value,
                curation_cluster_id=cluster.key,
                created_at=now,
                updated_at=now,
            )
            await self._store.insert_memory(memory)
            for candidate in cluster.candidates:
                await self._store.add_memory_derivation(
                    memory.id,
                    candidate.memory.id,
                    relation="summarizes",
                )
            created_ids.append(memory.id)
            created_clusters.append(cluster)

        run_client = _single_value(
            candidate.client
            for cluster in created_clusters
            for candidate in cluster.candidates
        )
        run_repo = _single_value(cluster.repo_identifier for cluster in created_clusters)
        run_project = _single_value(cluster.project_key for cluster in created_clusters)

        await self._store.record_memory_curation_run(
            MemoryCurationRun(
                id=run_id or f"cur-run-{now.timestamp():.0f}",
                policy_id=self._policy.policy_id,
                source_type=self._policy.source_type,
                client=run_client,
                repo_identifier=run_repo,
                project_key=run_project,
                candidate_count=len(candidates),
                created_memory_count=len(created_ids),
                skipped_reason=None if created_ids else "no_eligible_cluster",
                error=None,
                started_at=now,
                completed_at=datetime.now(timezone.utc),
            )
        )
        return CuratorResult(
            candidate_count=len(candidates),
            created_memory_count=len(created_ids),
            created_memory_ids=created_ids,
        )


def _single_value(values: Iterable[str | None]) -> str | None:
    distinct = {value for value in values if value}
    if len(distinct) == 1:
        return next(iter(distinct))
    return None
