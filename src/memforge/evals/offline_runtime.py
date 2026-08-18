"""Runtime wiring for service-owned offline Agent Evaluation Runs."""

from __future__ import annotations

from dataclasses import replace

from memforge.config import AppConfig
from memforge.evals.offline_evaluation import (
    AgentEvaluationCaseKind,
    AgentEvaluationRun,
    OfflineAgentEvaluation,
    OfflineEvaluationStore,
    ProductionSourceUnitDerivationReplayExecutor,
    SourceUnitReconciliationReplayExecutor,
    StructuredOfflineSemanticJudge,
)
from memforge.runtime import RuntimeProvider, get_effective_llm_config


async def build_offline_evaluation_for_run(
    store: OfflineEvaluationStore,
    config: AppConfig,
    runtime_provider: RuntimeProvider,
    run: AgentEvaluationRun,
) -> OfflineAgentEvaluation:
    """Build candidate and judge adapters from one pinned Run manifest."""

    llm = await get_effective_llm_config(store, config)
    candidate_llm = replace(
        llm,
        enrichment_model=str(run.candidate_manifest["model"]),
    )
    structured_client = runtime_provider.build_structured_llm_client(
        candidate_llm,
        max_concurrent=config.llm.enrichment_max_concurrent,
    )
    if structured_client is None:
        raise RuntimeError("offline evaluation requires a configured structured LLM client")

    return OfflineAgentEvaluation(
        store,
        executors={
            AgentEvaluationCaseKind.SOURCE_UNIT_DERIVATION: (
                ProductionSourceUnitDerivationReplayExecutor(structured_client)
            ),
            AgentEvaluationCaseKind.SOURCE_UNIT_RECONCILIATION: (
                SourceUnitReconciliationReplayExecutor(structured_client)
            ),
        },
        semantic_judge=(
            StructuredOfflineSemanticJudge(structured_client)
            if run.semantic_judge_manifest is not None
            else None
        ),
    )
