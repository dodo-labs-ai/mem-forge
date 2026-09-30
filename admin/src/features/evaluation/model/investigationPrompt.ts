import type { EvaluationCase, IssueGroup } from "./types";

const NONE = "none";

/**
 * A read-only diagnosis prompt for a coding agent. It carries identifiers,
 * versions and lineage only, never source text, so it is safe to paste into
 * any agent session.
 */
export function buildInvestigationPrompt(sourceName: string, group: IssueGroup, evaluationCase: EvaluationCase): string {
  const or = (value: string | null) => value ?? NONE;
  return [
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
    `Observation: ${or(evaluationCase.observation_id)}`,
    `Observation revision: ${or(evaluationCase.observation_revision_id)}`,
    `Projection run: ${evaluationCase.projection_run_id}`,
    `Operation: ${or(evaluationCase.operation_id)}`,
    `Execution: ${or(evaluationCase.execution_id)}`,
    `Derivation: ${or(evaluationCase.derivation_id)}`,
    `Batch: ${or(evaluationCase.batch_id)}`,
    `Trace: ${or(evaluationCase.trace_id)}`,
    `Evaluator: ${group.evaluator_name}:${group.evaluator_version}`,
    `Provider/model: ${or(evaluationCase.provider)} / ${or(evaluationCase.model)}`,
    `Runtime contract: ${evaluationCase.contract_version ?? evaluationCase.extraction_contract_version ?? NONE}`,
    `Deployment: ${or(evaluationCase.deployment_revision)}`,
  ].join("\n");
}
