import type {
  AgentEvaluationIssueGroup,
  AgentEvaluationRepresentativeCase,
} from "@/api/types";

export function buildAgentInvestigationPrompt(
  sourceName: string,
  group: AgentEvaluationIssueGroup,
  evaluationCase: AgentEvaluationRepresentativeCase,
): string {
  const values = [
    "Investigate this MemForge online-evaluation case read-only.",
    "Determine whether it is an evaluator false positive or a real runtime pipeline defect, identify the failing/degraded stage, and recommend the next bounded verification.",
    "Do not rerun source ingestion, mutate Memories, or rewrite lifecycle history without separate authorization.",
    "",
    `Source: ${sourceName} (${evaluationCase.source_id}, ${evaluationCase.source_type})`,
    `Assessment: ${evaluationCase.assessment_id}`,
    `Runtime event: ${evaluationCase.event_id}`,
    `Signal: ${group.criterion} / ${group.reason_code} / ${group.label}`,
    `Occurred at: ${evaluationCase.occurred_at}`,
    `Document: ${evaluationCase.doc_id}`,
    `Source Unit: ${evaluationCase.source_unit_id}`,
    `Target revision: ${evaluationCase.target_unit_revision_id}`,
    `Observation: ${evaluationCase.observation_id ?? "none"}`,
    `Observation revision: ${evaluationCase.observation_revision_id ?? "none"}`,
    `Projection run: ${evaluationCase.projection_run_id}`,
    `Operation: ${evaluationCase.operation_id ?? "none"}`,
    `Execution: ${evaluationCase.execution_id ?? "none"}`,
    `Derivation: ${evaluationCase.derivation_id ?? "none"}`,
    `Batch: ${evaluationCase.batch_id ?? "none"}`,
    `Trace: ${evaluationCase.trace_id ?? "none"}`,
    `Evaluator: ${group.evaluator_name}:${group.evaluator_version}`,
    `Provider/model: ${evaluationCase.provider ?? "none"} / ${evaluationCase.model ?? "none"}`,
    `Runtime contract: ${evaluationCase.contract_version ?? evaluationCase.extraction_contract_version ?? "none"}`,
    `Deployment: ${evaluationCase.deployment_revision ?? "none"}`,
  ];
  return values.join("\n");
}
