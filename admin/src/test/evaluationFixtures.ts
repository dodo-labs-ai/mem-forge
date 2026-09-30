import type { components } from "../api/schema.gen";

type Schemas = components["schemas"];
type Overview = Schemas["WorkspaceAgentEvaluationResponse"];
type Coverage = Schemas["AgentEvaluationCoverageResponse"];
type IssueGroup = Schemas["AgentEvaluationIssueGroupResponse"];
type EvaluationCase = Schemas["AgentEvaluationCaseResponse"];
type SourceHealth = Schemas["AgentEvaluationSourceHealthResponse"];
type Assessment = Schemas["AgentAssessmentResponse"];

const EVALUATOR = { evaluator_name: "memforge.deterministic.runtime_contract", evaluator_version: "3" };

export function makeCoverage(overrides: Partial<Coverage> = {}): Coverage {
  return {
    policy: "semantic_evaluator_v1",
    eligible_occurrences: 1204,
    assessed_occurrences: 1168,
    pending_occurrences: 32,
    coverage_rate: 1168 / 1204,
    oldest_pending_at: "2026-09-29T11:18:00+00:00",
    evaluator_failure_occurrences: 4,
    ...overrides,
  };
}

export function makeCase(overrides: Partial<EvaluationCase> = {}): EvaluationCase {
  return {
    assessment_id: "asm-7c2e",
    event_id: "evt_4c19d2e7a8b1c2d3e4a01f55",
    label: "fail",
    criterion: "evidence_reference_validity",
    reason_code: "unknown_evidence_block_id",
    occurred_at: "2026-09-29T10:00:00+00:00",
    occurrence_count: 1,
    source_id: "src-docs",
    source_type: "github_pages",
    doc_id: "doc-scheduling-overview",
    source_unit_id: "su_8f21a0c3d4e5f6a7b89b12e4",
    target_unit_revision_id: "rev-91d0",
    observation_id: null,
    observation_revision_id: null,
    projection_run_id: "prj-20260929",
    operation_id: null,
    execution_id: "exe-55b1",
    derivation_id: "drv-0e7a",
    batch_id: null,
    trace_id: "tr_77ab31c0aa11bb22cce9d2c8",
    provider: "sap",
    model: "anthropic--claude-sonnet",
    contract_version: null,
    extraction_contract_version: "extraction-v7",
    deployment_revision: "2026.09.28-1",
    ...overrides,
  };
}

export function makeIssueGroup(overrides: Partial<IssueGroup> = {}): IssueGroup {
  return {
    group_id: "aeg-fail",
    label: "fail",
    criterion: "evidence_reference_validity",
    reason_code: "unknown_evidence_block_id",
    ...EVALUATOR,
    occurrence_count: 14,
    distinct_event_count: 14,
    criterion_occurrence_count: 402,
    criterion_rate: 14 / 402,
    affected_source_ids: ["src-docs", "src-wiki"],
    affected_source_count: 2,
    source_types: ["confluence", "github_pages"],
    first_seen_at: "2026-09-29T01:00:00+00:00",
    last_seen_at: "2026-09-29T10:00:00+00:00",
    representative_cases: [makeCase()],
    ...overrides,
  };
}

export function makeSourceHealth(overrides: Partial<SourceHealth> = {}): SourceHealth {
  return {
    source_id: "src-docs",
    name: "Architecture docs",
    type: "github_pages",
    source_status: "active",
    evaluation_status: "attention",
    action_issue_group_count: 2,
    review_issue_group_count: 0,
    fail_occurrences: 17,
    review_occurrences: 0,
    coverage: makeCoverage({ eligible_occurrences: 205, assessed_occurrences: 201, pending_occurrences: 0, coverage_rate: 201 / 205, evaluator_failure_occurrences: 0 }),
    last_event_at: "2026-09-29T10:00:00+00:00",
    ...overrides,
  };
}

export function makeAssessment(overrides: Partial<Assessment> = {}): Assessment {
  return {
    assessment_id: "asm-1",
    target_event_id: "evt-1",
    criterion: "evidence_reference_validity",
    status: "completed",
    label: "fail",
    reason_code: "unknown_evidence_block_id",
    annotator_kind: "code",
    ...EVALUATOR,
    created_at: "2026-09-29T10:00:00+00:00",
    target_result_id: null,
    target_candidate_id: null,
    annotator_id: null,
    content_policy_id: null,
    input_fingerprint: null,
    confidence: null,
    reused_from_assessment_id: null,
    occurrence_count: 3,
    schema_version: "agent-assessment-v5",
    ...overrides,
  };
}

/** A workspace with two failing patterns, one pattern to check and a coverage gap. */
export function makeOverview(overrides: Partial<Overview> = {}): Overview {
  return {
    scope: { kind: "workspace", source_id: null, source_type: null },
    window: { from: "2026-09-28T12:00:00+00:00", to: "2026-09-29T12:00:00+00:00", days: 1 },
    summary: {
      total_assessments: 1204,
      runtime_event_count: 1204,
      eligible_assessment_count: 1204,
      missing_assessment_count: 32,
      action_issue_group_count: 2,
      review_issue_group_count: 1,
      source_count: 3,
      affected_source_count: 2,
      label_counts: { fail: 17, needs_review: 9, pass: 1174 },
      criterion_counts: {},
      status_counts: { completed: 1200, failed: 4 },
      truncated: false,
      row_limit: 1000,
    },
    coverage: makeCoverage(),
    issue_groups: [
      makeIssueGroup(),
      makeIssueGroup({
        group_id: "aeg-contract",
        criterion: "structured_output_contract",
        reason_code: "schema_validation_failed",
        occurrence_count: 3,
        criterion_occurrence_count: 380,
        criterion_rate: 3 / 380,
        affected_source_ids: ["src-jira"],
        affected_source_count: 1,
        source_types: ["jira"],
        representative_cases: [makeCase({ assessment_id: "asm-contract", event_id: "evt-contract", source_id: "src-jira", source_type: "jira" })],
      }),
      makeIssueGroup({
        group_id: "aeg-review",
        label: "needs_review",
        criterion: "evidence_localization",
        reason_code: "whole_block_fallback",
        occurrence_count: 9,
        criterion_occurrence_count: 186,
        criterion_rate: 9 / 186,
        affected_source_ids: ["src-repo"],
        affected_source_count: 1,
        source_types: ["github_repo"],
        representative_cases: [makeCase({ assessment_id: "asm-review", event_id: "evt-review", label: "needs_review", source_id: "src-repo", source_type: "github_repo" })],
      }),
    ],
    available_source_types: ["confluence", "github_pages", "github_repo", "jira"],
    sources: [
      makeSourceHealth(),
      makeSourceHealth({
        source_id: "src-repo",
        name: "mem-inception",
        type: "github_repo",
        evaluation_status: "review",
        action_issue_group_count: 0,
        review_issue_group_count: 1,
        fail_occurrences: 0,
        review_occurrences: 9,
      }),
      makeSourceHealth({
        source_id: "src-handbook",
        name: "Payroll handbook",
        type: "confluence",
        evaluation_status: "healthy",
        action_issue_group_count: 0,
        fail_occurrences: 0,
      }),
    ],
    runtime_events: [],
    assessments: [
      makeAssessment(),
      makeAssessment({
        assessment_id: "asm-2",
        criterion: "evidence_localization",
        label: "needs_review",
        reason_code: "whole_block_fallback",
        occurrence_count: 1,
      }),
      makeAssessment({
        assessment_id: "asm-3",
        criterion: "extraction_completion",
        label: "pass",
        reason_code: "candidates_extracted",
        occurrence_count: 42,
      }),
    ],
    ...overrides,
  };
}
