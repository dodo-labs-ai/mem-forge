"""Minimal deterministic retrieval golden runner."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from memforge.config import RetrievalConfig
from memforge.evals.retrieval.fixtures.corpus import seed_sqlite_fixture
from memforge.evals.retrieval.schema import RetrievalCase, RetrievalCaseSet
from memforge.retrieval.filters import MemorySourceFilter, MemoryTimeRange
from memforge.retrieval.search import SearchEngine
from memforge.storage.adapters.sqlite import build_sqlite_adapters


@dataclass(frozen=True)
class HardFailure:
    case_id: str
    message: str
    parity_gate: str | None = None


@dataclass(frozen=True)
class CaseRunResult:
    case_id: str
    ranked_ids: tuple[str, ...]
    scores: dict[str, float]
    total_candidates: int
    query_analysis: dict[str, Any]
    evidence_by_memory: dict[str, dict[str, Any]]

    def rank(self, memory_id: str) -> int:
        try:
            return self.ranked_ids.index(memory_id) + 1
        except ValueError:
            return len(self.ranked_ids) + 1


@dataclass(frozen=True)
class RetrievalEvalReport:
    """Golden eval report.

    ``run`` uses deterministic rank-derived surrogate scores, not raw
    SearchEngine model scores. Metrics should treat them as ordering signals.
    """

    case_results: dict[str, CaseRunResult]
    hard_failures: tuple[HardFailure, ...]
    qrels: dict[str, dict[str, int]]
    run: dict[str, dict[str, float]]

    @property
    def case_count(self) -> int:
        return len(self.case_results)

    def to_json(self) -> dict[str, Any]:
        return {
            "summary": {
                "case_count": self.case_count,
                "hard_failures": len(self.hard_failures),
            },
            "hard_failures": [
                {
                    "case_id": failure.case_id,
                    "message": failure.message,
                    "parity_gate": failure.parity_gate,
                }
                for failure in self.hard_failures
            ],
            "qrels": self.qrels,
            "run": self.run,
        }


async def run_sqlite_case_set(
    case_set: RetrievalCaseSet,
    *,
    db_path: Path,
    keep_databases: bool = False,
) -> RetrievalEvalReport:
    """Run all cases in a case set against a deterministic SQLite fixture."""

    case_results: dict[str, CaseRunResult] = {}
    hard_failures: list[HardFailure] = []
    qrels: dict[str, dict[str, int]] = {}
    run: dict[str, dict[str, float]] = {}

    for index, case in enumerate(case_set.cases):
        fixture = case_set.manifest.fixtures[case.fixture_variant]
        case_db_path = db_path.with_name(f"{db_path.stem}-{index}-{case.id}{db_path.suffix}")
        db = await seed_sqlite_fixture(db_path=case_db_path, fixture=fixture)
        try:
            adapters = build_sqlite_adapters(db, _EmptyVectorCollection())
            engine = SearchEngine(
                relational=adapters.relational,
                keyword=adapters.keyword,
                vector=adapters.vector,
                embed_cfg={},
                config=RetrievalConfig(enable_reranking=False),
                embedding_provider=lambda _query: [0.0],
            )
            result = await engine.search(
                case.query,
                time_range=_time_range_from_case(case),
                top_k=case.top_k,
                offset=case.offset,
                entities=list(case.entities),
                source_filter=_source_filter_from_case(case),
                request_scope=case.scope.to_access_scope(),
            )
        finally:
            await db.close()
            if not keep_databases:
                _remove_sqlite_files(case_db_path)

        case_result = _case_run_result(case.id, result)
        case_results[case.id] = case_result
        qrels[case.id] = dict(case.expected.relevant)
        run[case.id] = dict(case_result.scores)
        hard_failures.extend(_assert_case(case, case_result))

    return RetrievalEvalReport(
        case_results=case_results,
        hard_failures=tuple(hard_failures),
        qrels=qrels,
        run=run,
    )


def _case_run_result(case_id: str, result: dict[str, Any]) -> CaseRunResult:
    search_results = list(result["results"])
    ranked_ids = tuple(item.memory_id for item in search_results)
    return CaseRunResult(
        case_id=case_id,
        ranked_ids=ranked_ids,
        scores=_stable_run_scores(ranked_ids),
        total_candidates=int(result["total_candidates"]),
        query_analysis=dict(result["query_analysis"]),
        evidence_by_memory={
            item.memory_id: dict(item.retrieval_evidence or {})
            for item in search_results
        },
    )


def _assert_case(case: RetrievalCase, result: CaseRunResult) -> list[HardFailure]:
    failures: list[HardFailure] = []
    if case.expected.total_candidates is not None and result.total_candidates != case.expected.total_candidates:
        failures.append(
            _failure(
                case,
                (
                    f"expected total_candidates={case.expected.total_candidates}, "
                    f"got {result.total_candidates}"
                ),
            )
        )
    if case.expected.required_profile is not None:
        profile = result.query_analysis.get("ranking_profile")
        if profile != case.expected.required_profile:
            failures.append(
                _failure(
                    case,
                    f"expected ranking_profile={case.expected.required_profile}, got {profile}",
                )
            )
    for memory_id, max_rank in case.expected.max_rank.items():
        rank = result.rank(memory_id)
        if rank > max_rank:
            failures.append(
                _failure(
                    case,
                    f"expected {memory_id} rank <= {max_rank}, got {rank}",
                )
            )
    for memory_id, required_channels in case.expected.required_channels.items():
        evidence = result.evidence_by_memory.get(memory_id) or {}
        if not _has_required_channel(evidence, required_channels):
            failures.append(
                _failure(
                    case,
                    f"expected {memory_id} evidence channels {list(required_channels)}, got {evidence}",
                )
            )
    return failures


def _has_required_channel(evidence: dict[str, Any], required_channels: tuple[str, ...]) -> bool:
    return all(_has_channel(evidence, required) for required in required_channels)


def _has_channel(evidence: dict[str, Any], required: str) -> bool:
    if required in evidence:
        return True
    metadata = evidence.get("metadata_lexical")
    if not isinstance(metadata, dict):
        return False
    channel = metadata.get("channel")
    matched_fields = set(metadata.get("matched_fields") or ())
    return required == channel or required in matched_fields


def _source_filter_from_case(case: RetrievalCase) -> MemorySourceFilter | None:
    if case.source_filter is None:
        return None
    return MemorySourceFilter(
        source_ids=tuple(case.source_filter.get("source_ids") or ()),
        clients=tuple(case.source_filter.get("clients") or ()),
        repo_identifiers=tuple(case.source_filter.get("repo_identifiers") or ()),
    )


def _time_range_from_case(case: RetrievalCase) -> MemoryTimeRange | None:
    if case.time_range is None:
        return None
    return MemoryTimeRange(
        after=_parse_datetime(case.time_range.get("after")),
        before=_parse_datetime(case.time_range.get("before")),
        date_type=case.time_range.get("date_type") or "source_updated_at",
    )


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        raise ValueError(f"time_range values must be ISO datetime strings, got {type(value).__name__}")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _stable_run_scores(ranked_ids: tuple[str, ...]) -> dict[str, float]:
    return {
        memory_id: round(1.0 / rank, 12)
        for rank, memory_id in enumerate(ranked_ids, start=1)
    }


def _failure(case: RetrievalCase, message: str) -> HardFailure:
    return HardFailure(case_id=case.id, message=message, parity_gate=case.parity_gate)


def _remove_sqlite_files(db_path: Path) -> None:
    for path in (
        db_path,
        db_path.with_name(f"{db_path.name}-wal"),
        db_path.with_name(f"{db_path.name}-shm"),
    ):
        with suppress(FileNotFoundError):
            path.unlink()


class _EmptyVectorCollection:
    def query(self, **kwargs: Any) -> dict[str, list[list[Any]]]:
        return {"ids": [[]], "distances": [[]]}

    def upsert(self, **kwargs: Any) -> None:
        return None

    def delete(self, **kwargs: Any) -> None:
        return None

    def get(self, **kwargs: Any) -> dict[str, list[Any]]:
        return {"ids": []}
