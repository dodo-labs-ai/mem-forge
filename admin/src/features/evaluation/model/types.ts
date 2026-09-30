import type { components } from "@/api";

type Schemas = components["schemas"];

export type EvaluationOverview = Schemas["WorkspaceAgentEvaluationResponse"];
export type EvaluationCoverage = Schemas["AgentEvaluationCoverageResponse"];
export type IssueGroup = Schemas["AgentEvaluationIssueGroupResponse"];
export type EvaluationCase = Schemas["AgentEvaluationCaseResponse"];
export type SourceHealth = Schemas["AgentEvaluationSourceHealthResponse"];
export type SourceEvaluationStatus = SourceHealth["evaluation_status"];
export type Assessment = Schemas["AgentAssessmentResponse"];
export type AssessmentLabel = NonNullable<Assessment["label"]>;
