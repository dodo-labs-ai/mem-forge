"""Runtime wiring for service-owned offline Agent Evaluation Runs."""

from __future__ import annotations

from dataclasses import replace

from memforge.config import AppConfig
from memforge.evals.offline_evaluation import (
    AgentEvaluationCaseKind,
    AgentEvaluationRun,
    CrossDocumentRelationGroupReplayExecutor,
    CrossDocumentRelationReplayExecutor,
    OfflineAgentEvaluation,
    OfflineEvaluationStore,
    ProductionSourceUnitDerivationReplayExecutor,
    SourceUnitReconciliationReplayExecutor,
    SourceUnitSupportReplayExecutor,
    StructuredOfflineSemanticJudge,
)
from memforge.runtime import RuntimeProvider, resolve_models
from memforge.storage.document_store import DocumentStore


async def build_offline_evaluation_for_run(
    store: OfflineEvaluationStore,
    config: AppConfig,
    runtime_provider: RuntimeProvider,
    run: AgentEvaluationRun,
    *,
    document_store: DocumentStore,
) -> OfflineAgentEvaluation:
    """Build candidate and judge adapters from one pinned Run manifest.

    ``document_store`` holds the image Artifacts the pinned derivations and Supports read.
    """

    models = await resolve_models(store, config)
    # The Run pins one model for every step it replays, reached through the
    # generation model's endpoint and credential.
    candidate_models = replace(
        models,
        generation=(
            None
            if models.generation is None
            else replace(models.generation, route=str(run.candidate_manifest["model"]))
        ),
        decision_route=None,
    )
    structured_client = runtime_provider.build_structured_llm_client(
        candidate_models,
        max_concurrent=config.llm.enrichment_max_concurrent,
    )
    if structured_client is None:
        raise RuntimeError("offline evaluation requires a configured structured LLM client")

    return OfflineAgentEvaluation(
        store,
        executors={
            AgentEvaluationCaseKind.SOURCE_UNIT_DERIVATION: (
                ProductionSourceUnitDerivationReplayExecutor(structured_client, document_store)
            ),
            AgentEvaluationCaseKind.SOURCE_UNIT_SUPPORT: (
                SourceUnitSupportReplayExecutor(structured_client, document_store)
            ),
            AgentEvaluationCaseKind.SOURCE_UNIT_RECONCILIATION: (
                SourceUnitReconciliationReplayExecutor(structured_client)
            ),
            AgentEvaluationCaseKind.CROSS_DOCUMENT_RELATION: (
                CrossDocumentRelationReplayExecutor(structured_client)
            ),
            AgentEvaluationCaseKind.CROSS_DOCUMENT_RELATION_GROUP: (
                CrossDocumentRelationGroupReplayExecutor(structured_client)
            ),
        },
        semantic_judge=(
            StructuredOfflineSemanticJudge(structured_client)
            if run.semantic_judge_manifest is not None
            else None
        ),
    )
