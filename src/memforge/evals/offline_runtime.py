"""Runtime wiring for service-owned offline Agent Evaluation Runs."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from memforge.config import AppConfig
from memforge.evals.expectation_judge import ExpectationJudgeSpec, StructuredExpectationJudge
from memforge.evals.offline_evaluation import (
    AgentEvaluationCaseKind,
    AgentEvaluationRun,
    MemoryRelationGroupReplayExecutor,
    MemoryRelationReplayExecutor,
    OfflineAgentEvaluation,
    OfflineEvaluationStore,
    ProductionSourceUnitDerivationReplayExecutor,
    SourceUnitReconciliationReplayExecutor,
    SourceUnitSupportReplayExecutor,
    StructuredOfflineSemanticJudge,
)
from memforge.llm.bindings import ModelBinding, build_binding, compile_binding
from memforge.llm.providers import ModelConfigError
from memforge.runtime import ResolvedModels, RuntimeProvider, generation_token_limits, resolve_models
from memforge.server.model_settings_api import listed_model
from memforge.storage.document_store import DocumentStore


def configured_expectation_judge(config: AppConfig) -> tuple[ExpectationJudgeSpec, ModelBinding] | None:
    """The expectation judge this deployment configures: the identity a run pins and its binding.

    It is built as a generation binding of a registered provider, the way
    Settings builds one, and compiled with its key, so a provider, model or key
    the registry does not accept raises ``ModelConfigError`` naming the judge.
    ``None`` when no judge model is configured.
    """

    settings = config.agent_evaluation
    if not settings.judge_model:
        return None
    try:
        binding = _judge_binding(config)
        compile_binding(binding, settings.judge_api_key or None)
    except ModelConfigError as exc:
        raise ModelConfigError(exc.code, f"The expectation judge is not usable: {exc}") from exc
    return ExpectationJudgeSpec(provider=binding.provider, model=binding.route), binding


def _judge_binding(config: AppConfig, listed: Any | None = None) -> ModelBinding:
    """The judge's binding, with the token limits and schema support its listing gives, as Settings records them."""

    settings = config.agent_evaluation
    limits = {
        name: getattr(listed, name)
        for name in ("context_window_tokens", "max_output_tokens")
        if listed is not None and getattr(listed, name, None) is not None
    }
    return build_binding(
        kind="generation",
        provider=settings.judge_provider,
        model_id=settings.judge_model,
        config_source="environment",
        endpoint=settings.judge_endpoint or None,
        auth_strategy="api_key" if settings.judge_api_key else None,
        token_limits=limits,
        native_schema=getattr(listed, "native_schema", None),
    )


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
            AgentEvaluationCaseKind.MEMORY_RELATION: (
                MemoryRelationReplayExecutor(structured_client)
            ),
            AgentEvaluationCaseKind.MEMORY_RELATION_GROUP: (
                MemoryRelationGroupReplayExecutor(structured_client)
            ),
        },
        semantic_judge=(
            StructuredOfflineSemanticJudge(structured_client)
            if run.semantic_judge_manifest is not None
            else None
        ),
        expectation_judge=await _expectation_judge(config, runtime_provider, run),
    )


async def _expectation_judge(
    config: AppConfig, runtime_provider: RuntimeProvider, run: AgentEvaluationRun,
) -> StructuredExpectationJudge | None:
    """The judge the run pins, on its own client; ``None`` when this deployment does not configure that judge.

    Its requests are sized as a Settings binding's are: by the token limits its
    provider's model listing gives, and the ``MEMFORGE_LLM_*`` limits for what
    the listing leaves open or when it gives nothing the binding accepts.
    Without a judge the run still replays and scores its code checks, and
    records each expectation as not judged.
    """

    if run.expectation_judge_manifest is None:
        return None
    configured = configured_expectation_judge(config)
    if configured is None or configured[0] != ExpectationJudgeSpec.from_manifest(run.expectation_judge_manifest):
        return None
    api_key = config.agent_evaluation.judge_api_key or None
    binding = configured[1]
    try:
        binding = _judge_binding(config, await listed_model(binding, api_key))
    except ModelConfigError:
        pass
    client = runtime_provider.build_structured_llm_client(
        ResolvedModels(
            generation=compile_binding(binding, api_key),
            embedding=None,
            request_timeout_s=config.llm.request_timeout_s,
            **generation_token_limits(config.llm, binding.token_limits),
            input_budget_fraction=config.llm.input_budget_fraction,
        ),
        max_concurrent=config.llm.enrichment_max_concurrent,
    )
    return StructuredExpectationJudge(client) if client is not None else None
